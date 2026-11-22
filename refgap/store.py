"""SQLite-backed project store.

Everything needed to resume is persisted: input file hashes and parsed column
types, the exact search parameters, and each verified candidate result. An
interrupted run can be resumed without re-verifying stored candidates.
Decisions (confirmed / excluded) live on the dataset, so they carry forward to
the next uploaded batch.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    path TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    warnings_json TEXT NOT NULL,
    UNIQUE(dataset_id, name)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    table_snapshot_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    candidate_key TEXT NOT NULL,
    width INTEGER NOT NULL,
    child_table TEXT NOT NULL,
    child_cols_json TEXT NOT NULL,
    parent_table TEXT NOT NULL,
    parent_cols_json TEXT NOT NULL,
    child_logical_json TEXT NOT NULL,
    parent_logical_json TEXT NOT NULL,
    mapping_mode TEXT NOT NULL,
    effective_json TEXT NOT NULL,
    state TEXT NOT NULL,
    reason TEXT NOT NULL,
    confidence TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    counterexample_count INTEGER NOT NULL DEFAULT 0,
    counterexample_stored INTEGER NOT NULL DEFAULT 0,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, candidate_key)
);
CREATE INDEX IF NOT EXISTS idx_candidates_run_state ON candidates(run_id, state);
CREATE TABLE IF NOT EXISTS counterexamples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    candidate_key TEXT NOT NULL,
    seq INTEGER NOT NULL,
    child_row INTEGER,
    values_json TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1,
    null_key INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, candidate_key, seq)
);
CREATE INDEX IF NOT EXISTS idx_counterexamples_lookup
    ON counterexamples(run_id, candidate_key, seq);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    candidate_key TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('confirmed', 'excluded')),
    note TEXT NOT NULL DEFAULT '',
    spec_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(dataset_id, candidate_key)
);
CREATE TABLE IF NOT EXISTS validations (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    relations_json TEXT NOT NULL,
    files_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Store:
    def __init__(self, path: str | os.PathLike[str]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- datasets ---------------------------------------------------------
    def create_dataset(self, name: str, note: str = "") -> str:
        dataset_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO datasets (id, name, created_at, note) VALUES (?, ?, ?, ?)",
                (dataset_id, name, now(), note),
            )
        return dataset_id

    def add_table(
        self,
        dataset_id: str,
        name: str,
        original_filename: str,
        path: str,
        source_hash: str,
        columns: Sequence[dict[str, Any]],
        row_count: int,
        warnings: Sequence[str],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO tables
                   (dataset_id, name, original_filename, path, source_hash,
                    columns_json, row_count, warnings_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    name,
                    original_filename,
                    path,
                    source_hash,
                    json.dumps(list(columns), ensure_ascii=False),
                    row_count,
                    json.dumps(list(warnings), ensure_ascii=False),
                ),
            )

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            data["tables"] = self.list_tables(dataset_id)
            data["decisions"] = self.list_decisions(dataset_id)
            return data

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT d.*,
                          (SELECT COUNT(*) FROM tables t WHERE t.dataset_id = d.id) AS table_count,
                          (SELECT COUNT(*) FROM validations v WHERE v.dataset_id = d.id) AS validation_count
                   FROM datasets d ORDER BY d.created_at DESC, d.rowid DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def list_tables(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tables WHERE dataset_id = ? ORDER BY id", (dataset_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["columns"] = json.loads(d.pop("columns_json"))
            d["warnings"] = json.loads(d.pop("warnings_json"))
            out.append(d)
        return out

    # -- runs -------------------------------------------------------------
    def create_run(
        self, dataset_id: str, params: dict[str, Any], table_snapshot: list[dict[str, Any]]
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        ts = now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO runs
                   (id, dataset_id, created_at, updated_at, params_json, status, scope_json, table_snapshot_json)
                   VALUES (?, ?, ?, ?, ?, 'running', '{}', ?)""",
                (
                    run_id,
                    dataset_id,
                    ts,
                    ts,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(table_snapshot, ensure_ascii=False),
                ),
            )
        return run_id

    def update_run_scope(self, run_id: str, scope: dict[str, Any], status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET scope_json = ?, status = ?, updated_at = ? WHERE id = ?",
                (json.dumps(scope, ensure_ascii=False), status, now(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json"))
        d["scope"] = json.loads(d.pop("scope_json"))
        d["table_snapshot"] = json.loads(d.pop("table_snapshot_json"))
        return d

    def list_runs(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE dataset_id = ? ORDER BY created_at DESC, rowid DESC",
                (dataset_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d.pop("params_json"))
            d["scope"] = json.loads(d.pop("scope_json"))
            d["table_snapshot"] = json.loads(d.pop("table_snapshot_json"))
            out.append(d)
        return out

    def completed_candidate_keys(self, run_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT candidate_key FROM candidates WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {r["candidate_key"] for r in rows}

    def save_candidate(self, run_id: str, result: Any) -> None:
        """Store the candidate and every counterexample it produced.

        Counterexamples live in their own table so the download is complete
        even when there are many of them; the candidate row only carries the
        count. Re-saving the same candidate replaces its rows atomically.
        """
        spec = result.spec
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO candidates
                       (run_id, candidate_key, width, child_table, child_cols_json,
                        parent_table, parent_cols_json, child_logical_json, parent_logical_json,
                        mapping_mode, effective_json, state, reason, confidence, stats_json,
                        counterexample_count, counterexample_stored, elapsed_ms, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        result.key,
                        spec.width,
                        spec.child_table,
                        json.dumps(list(spec.child_cols), ensure_ascii=False),
                        spec.parent_table,
                        json.dumps(list(spec.parent_cols), ensure_ascii=False),
                        json.dumps(list(spec.logical_child), ensure_ascii=False),
                        json.dumps(list(spec.logical_parent), ensure_ascii=False),
                        spec.mapping_mode,
                        json.dumps(list(spec.effective), ensure_ascii=False),
                        result.state,
                        result.reason,
                        result.confidence,
                        json.dumps(result.stats, ensure_ascii=False),
                        int(result.stats.get("missing_keys", len(result.counterexamples))),
                        len(result.counterexamples),
                        result.elapsed_ms,
                        now(),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM counterexamples WHERE run_id = ? AND candidate_key = ?",
                    (run_id, result.key),
                )
                rows = []
                for seq, item in enumerate(result.counterexamples):
                    rows.append(
                        (
                            run_id,
                            result.key,
                            seq,
                            item.get("row"),
                            json.dumps(item.get("values", []), ensure_ascii=False),
                            int(item.get("occurrences", 1)),
                            int(bool(item.get("null_key"))),
                        )
                    )
                if rows:
                    self._conn.executemany(
                        """INSERT INTO counterexamples
                           (run_id, candidate_key, seq, child_row, values_json, occurrences, null_key)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                self._conn.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (now(), run_id))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_candidate(self, run_id: str, candidate_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM candidates WHERE run_id = ? AND candidate_key = ?",
                (run_id, candidate_key),
            ).fetchone()
        return self._candidate_row(row) if row else None

    def list_candidates(
        self,
        run_id: str,
        *,
        state: str | None = None,
        width: int | None = None,
        q: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses = ["run_id = ?"]
        args: list[Any] = [run_id]
        if state:
            clauses.append("state = ?")
            args.append(state)
        if width:
            clauses.append("width = ?")
            args.append(width)
        if q:
            clauses.append("candidate_key LIKE ?")
            args.append(f"%{q}%")
        where = " AND ".join(clauses)
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM candidates WHERE {where}", args
            ).fetchone()["n"]
            rows = self._conn.execute(
                f"""SELECT * FROM candidates WHERE {where}
                    ORDER BY CASE state WHEN 'contained' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END,
                             confidence = 'high' DESC, confidence = 'medium' DESC,
                             child_table, width, candidate_key
                    LIMIT ? OFFSET ?""",
                args + [limit, offset],
            ).fetchall()
            return [self._candidate_row(r) for r in rows], total

    def state_counts(self, run_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM candidates WHERE run_id = ? GROUP BY state",
                (run_id,),
            ).fetchall()
        return {r["state"]: r["n"] for r in rows}

    @staticmethod
    def _candidate_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["child_cols"] = json.loads(d.pop("child_cols_json"))
        d["parent_cols"] = json.loads(d.pop("parent_cols_json"))
        d["child_logical"] = json.loads(d.pop("child_logical_json"))
        d["parent_logical"] = json.loads(d.pop("parent_logical_json"))
        d["effective"] = json.loads(d.pop("effective_json"))
        d["stats"] = json.loads(d.pop("stats_json"))
        d["counterexamples"] = []
        d["counterexamples_truncated"] = (
            d.get("counterexample_count", 0) > d.get("counterexample_stored", 0)
        )
        return d

    def counterexamples(
        self, run_id: str, candidate_key: str, *, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = """SELECT child_row, values_json, occurrences, null_key
                 FROM counterexamples WHERE run_id = ? AND candidate_key = ?
                 ORDER BY seq"""
        args: list[Any] = [run_id, candidate_key]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            args.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            {
                "row": r["child_row"],
                "values": json.loads(r["values_json"]),
                "occurrences": r["occurrences"],
                "null_key": bool(r["null_key"]),
            }
            for r in rows
        ]

    def counterexample_count(self, run_id: str, candidate_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM counterexamples WHERE run_id = ? AND candidate_key = ?",
                (run_id, candidate_key),
            ).fetchone()
        return row["n"]

    # -- decisions --------------------------------------------------------
    def set_decision(
        self, dataset_id: str, candidate_key: str, action: str, spec: dict[str, Any], note: str = ""
    ) -> None:
        if action not in ("confirmed", "excluded"):
            raise ValueError("decision action must be confirmed or excluded")
        with self._lock:
            existing = self._conn.execute(
                "SELECT action FROM decisions WHERE dataset_id = ? AND candidate_key = ?",
                (dataset_id, candidate_key),
            ).fetchone()
            if existing and existing["action"] != action:
                raise ValueError(
                    f"该候选已有相反的决定（{existing['action']}），请先撤销后再设置"
                )
            self._conn.execute(
                """INSERT INTO decisions
                   (dataset_id, candidate_key, action, note, spec_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_id, candidate_key) DO UPDATE SET
                     note = excluded.note, updated_at = excluded.updated_at""",
                (dataset_id, candidate_key, action, note, json.dumps(spec, ensure_ascii=False), now()),
            )

    def clear_decision(self, dataset_id: str, candidate_key: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM decisions WHERE dataset_id = ? AND candidate_key = ?",
                (dataset_id, candidate_key),
            )

    def list_decisions(self, dataset_id: str, action: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if action:
                rows = self._conn.execute(
                    "SELECT * FROM decisions WHERE dataset_id = ? AND action = ? ORDER BY id",
                    (dataset_id, action),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM decisions WHERE dataset_id = ? ORDER BY id", (dataset_id,)
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["spec"] = json.loads(d.pop("spec_json"))
            out.append(d)
        return out

    def confirmed_relations(self, dataset_id: str) -> list[dict[str, Any]]:
        """Only explicitly confirmed candidates become validation constraints."""
        out = []
        for d in self.list_decisions(dataset_id, "confirmed"):
            spec = d["spec"]
            out.append(
                {
                    "candidate_key": d["candidate_key"],
                    "status": "confirmed",
                    "child_table": spec["child_table"],
                    "child_cols": spec["child_cols"],
                    "parent_table": spec["parent_table"],
                    "parent_cols": spec["parent_cols"],
                    "mapping_mode": spec.get("mapping_mode", "auto"),
                    "effective": spec.get("effective", []),
                    "child_logical": spec.get("child_logical", []),
                    "parent_logical": spec.get("parent_logical", []),
                    "null_policy": spec.get("null_policy", "skip_child_nulls"),
                    "note": d.get("note", ""),
                }
            )
        return out

    # -- validations ------------------------------------------------------
    def save_validation(
        self,
        dataset_id: str,
        relations: list[dict[str, Any]],
        files: dict[str, str],
        result: dict[str, Any],
        label: str = "",
    ) -> str:
        validation_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                """INSERT INTO validations
                   (id, dataset_id, created_at, label, relations_json, files_json, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    validation_id,
                    dataset_id,
                    now(),
                    label,
                    json.dumps(relations, ensure_ascii=False),
                    json.dumps(files, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        return validation_id

    def list_validations(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, dataset_id, created_at, label, result_json
                   FROM validations WHERE dataset_id = ? ORDER BY created_at DESC, rowid DESC""",
                (dataset_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            result = json.loads(d.pop("result_json"))
            d["ok"] = result.get("ok")
            d["relation_count"] = len(result.get("relations", []))
            d["orphan_rows"] = sum(rel.get("orphan_rows", 0) for rel in result.get("relations", []))
            d["orphan_relations"] = sum(
                1 for rel in result.get("relations", []) if rel.get("kind") == "orphans"
            )
            d["structure_relations"] = sum(
                1
                for rel in result.get("relations", [])
                if rel.get("status") in ("fail", "error") and rel.get("kind") != "orphans"
            )
            out.append(d)
        return out

    def get_validation(self, validation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM validations WHERE id = ?", (validation_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["relations"] = json.loads(d.pop("relations_json"))
        d["files"] = json.loads(d.pop("files_json"))
        d["result"] = json.loads(d.pop("result_json"))
        return d

    def previous_validation(self, dataset_id: str, exclude_id: str | None = None) -> dict[str, Any] | None:
        """Return the batch that immediately precedes ``exclude_id``.

        Ordering is by insertion (rowid), not by the second-resolution
        created_at: two batches can share a timestamp, and an older validation
        page must never be compared against a *later* batch.
        """
        with self._lock:
            if exclude_id:
                row = self._conn.execute(
                    """SELECT id FROM validations
                       WHERE dataset_id = ?
                         AND rowid < (SELECT rowid FROM validations WHERE id = ?)
                       ORDER BY rowid DESC LIMIT 1""",
                    (dataset_id, exclude_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """SELECT id FROM validations WHERE dataset_id = ?
                       ORDER BY rowid DESC LIMIT 1""",
                    (dataset_id,),
                ).fetchone()
        return self.get_validation(row["id"]) if row else None
