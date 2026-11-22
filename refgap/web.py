"""Flask application: upload, configure, inspect candidates, decide, validate.

The UI exists to make the evidence reviewable: every candidate shows the
real records it covers and the rows it cannot match. Decisions are explicit
and reversible; only confirmed relations become constraints downstream.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from validator.core import validate as run_validation

from .gaps import compare_batches
from .loader import SUPPORTED_EXTENSIONS, LoadError, load_table
from .search import SearchParams, candidate_key, run_search
from .store import Store
from .types import MAPPING_LABELS, NULL_POLICY_LABELS

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
ALLOWED_EXT = tuple(sorted(SUPPORTED_EXTENSIONS))


def _safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    cleaned = "".join(ch for ch in base if ch.isalnum() or ch in "._- ()[]")
    return cleaned or "upload"


def _table_name_from(filename: str, taken: set[str]) -> str:
    stem = Path(filename).stem.strip() or "table"
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in stem)
    cleaned = cleaned.strip("_") or "table"
    name = cleaned
    i = 2
    while name in taken:
        name = f"{cleaned}_{i}"
        i += 1
    return name


class Job:
    """Background search job with cooperative stop for resume testing."""

    def __init__(self, run_id: str, dataset_id: str, params: SearchParams):
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.params = params
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error: str | None = None
        self.thread: threading.Thread | None = None


class AppState:
    def __init__(self, store: Store, upload_root: str):
        self.store = store
        self.upload_root = upload_root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def job(self, run_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(run_id)


def create_app(db_path: str | None = None, upload_root: str | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    db_path = db_path or os.environ.get("REFGAP_DB", ".refgap/store.sqlite3")
    upload_root = upload_root or os.environ.get(
        "REFGAP_UPLOADS", tempfile.mkdtemp(prefix="refgap-uploads-")
    )
    os.makedirs(upload_root, exist_ok=True)
    state = AppState(Store(db_path), upload_root)
    app.extensions["refgap"] = state

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        return {
            "mapping_labels": MAPPING_LABELS,
            "null_labels": NULL_POLICY_LABELS,
            "states": {
                "contained": "包含",
                "partial": "部分包含",
                "pruned": "已排除（结构性）",
                "error": "错误",
            },
            "run_states": {
                "running": "运行中",
                "completed": "已完成",
                "interrupted": "已中断",
                "failed": "失败",
            },
        }

    # --- pages -----------------------------------------------------------
    @app.get("/")
    def index():
        datasets = state.store.list_datasets()
        return render_template("index.html", datasets=datasets)

    @app.post("/datasets")
    def create_dataset():
        name = (request.form.get("name") or "").strip() or "未命名批次"
        dataset_id = state.store.create_dataset(name)
        return redirect(url_for("dataset_page", dataset_id=dataset_id))

    @app.get("/datasets/<dataset_id>")
    def dataset_page(dataset_id: str):
        dataset = state.store.get_dataset(dataset_id)
        if not dataset:
            abort(404)
        runs = state.store.list_runs(dataset_id)
        validations = state.store.list_validations(dataset_id)
        return render_template(
            "dataset.html",
            dataset=dataset,
            runs=runs,
            validations=validations,
            allowed_ext=",".join(ALLOWED_EXT),
        )

    @app.post("/datasets/<dataset_id>/upload")
    def upload(dataset_id: str):
        dataset = state.store.get_dataset(dataset_id)
        if not dataset:
            abort(404)
        files = request.files.getlist("files")
        if not files or all(not f.filename for f in files):
            return redirect(url_for("dataset_page", dataset_id=dataset_id))
        dest_dir = Path(state.upload_root) / dataset_id / uuid.uuid4().hex[:8]
        dest_dir.mkdir(parents=True, exist_ok=True)
        taken = {t["name"] for t in dataset["tables"]}
        errors: list[str] = []
        added: list[str] = []
        for upload_file in files:
            if not upload_file.filename:
                continue
            ext = Path(upload_file.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append(f"{upload_file.filename}: 不支持的文件类型 {ext}")
                continue
            safe = _safe_name(upload_file.filename)
            target = dest_dir / safe
            upload_file.save(target)
            name = _table_name_from(upload_file.filename, taken)
            try:
                table = load_table(name, str(target))
            except LoadError as exc:
                errors.append(f"{upload_file.filename}: {exc}")
                target.unlink(missing_ok=True)
                continue
            taken.add(name)
            added.append(name)
            state.store.add_table(
                dataset_id,
                name,
                upload_file.filename,
                str(target),
                table.source_hash,
                [{"name": c.name, "type": c.logical_type, "index": c.index} for c in table.columns],
                len(table.rows),
                table.warnings,
            )
        if errors:
            return render_template(
                "dataset.html",
                dataset=state.store.get_dataset(dataset_id),
                runs=state.store.list_runs(dataset_id),
                validations=state.store.list_validations(dataset_id),
                allowed_ext=",".join(ALLOWED_EXT),
                upload_errors=errors,
                upload_added=added,
            ), 400
        return redirect(url_for("dataset_page", dataset_id=dataset_id))

    # --- search ----------------------------------------------------------
    @app.post("/datasets/<dataset_id>/search")
    def start_search(dataset_id: str):
        dataset = state.store.get_dataset(dataset_id)
        if not dataset:
            abort(404)
        try:
            params = _params_from_form(request.form, dataset)
        except ValueError as exc:
            return render_template(
                "dataset.html",
                dataset=dataset,
                runs=state.store.list_runs(dataset_id),
                validations=state.store.list_validations(dataset_id),
                allowed_ext=",".join(ALLOWED_EXT),
                upload_errors=[str(exc)],
            ), 400
        tables = _load_dataset_tables(dataset)
        table_snapshot = [
            {"name": t.name, "source_hash": t.source_hash, "rows": len(t.rows)} for t in tables
        ]
        run_id = state.store.create_run(dataset_id, params.to_dict(), table_snapshot)
        job = Job(run_id, dataset_id, params)
        with state.lock:
            state.jobs[run_id] = job

        def work() -> None:
            try:
                def on_result(result):
                    state.store.save_candidate(run_id, result)

                outcome = run_search(
                    tables,
                    params,
                    on_result=on_result,
                    should_stop=job.stop.is_set,
                )
                status = "interrupted" if outcome["scope"].get("interrupted") else "completed"
                state.store.update_run_scope(run_id, outcome["scope"], status)
            except Exception as exc:  # pragma: no cover - surfaced in UI
                job.error = str(exc)
                state.store.update_run_scope(
                    run_id, {"error": str(exc), "params": params.to_dict()}, "failed"
                )
            finally:
                job.done.set()

        job.thread = threading.Thread(target=work, daemon=True, name=f"search-{run_id}")
        job.thread.start()
        return redirect(url_for("run_page", dataset_id=dataset_id, run_id=run_id))

    @app.post("/datasets/<dataset_id>/runs/<run_id>/stop")
    def stop_search(dataset_id: str, run_id: str):
        job = state.job(run_id)
        if job:
            job.stop.set()
        return jsonify({"stopping": True})

    @app.post("/datasets/<dataset_id>/runs/<run_id>/resume")
    def resume_search(dataset_id: str, run_id: str):
        dataset = state.store.get_dataset(dataset_id)
        run = state.store.get_run(run_id)
        if not dataset or not run:
            abort(404)
        if run["status"] == "running":
            return jsonify({"error": "该运行仍在进行"}), 409
        tables = _load_dataset_tables(dataset)
        params = SearchParams.from_dict(run["params"])
        completed = state.store.completed_candidate_keys(run_id)
        job = Job(run_id, dataset_id, params)
        with state.lock:
            state.jobs[run_id] = job

        def work() -> None:
            try:
                outcome = run_search(
                    tables,
                    params,
                    completed_keys=completed,
                    on_result=lambda r: state.store.save_candidate(run_id, r),
                    should_stop=job.stop.is_set,
                )
                status = "interrupted" if outcome["scope"].get("interrupted") else "completed"
                previous_scope = run.get("scope") or {}
                merged = dict(outcome["scope"])
                merged["resumed_from"] = {
                    "already_verified": previous_scope.get("already_verified", 0),
                    "verified_this_pass": previous_scope.get("verified_this_pass", 0),
                }
                state.store.update_run_scope(run_id, merged, status)
            except Exception as exc:  # pragma: no cover
                job.error = str(exc)
            finally:
                job.done.set()

        job.thread = threading.Thread(target=work, daemon=True, name=f"resume-{run_id}")
        job.thread.start()
        return jsonify({"resuming": True})

    @app.get("/datasets/<dataset_id>/runs/<run_id>")
    def run_page(dataset_id: str, run_id: str):
        dataset = state.store.get_dataset(dataset_id)
        run = state.store.get_run(run_id)
        if not dataset or not run:
            abort(404)
        state_filter = request.args.get("state") or None
        if state_filter == "all":
            state_filter = None
        width = request.args.get("width", type=int)
        q = request.args.get("q") or None
        page = max(1, request.args.get("page", default=1, type=int))
        limit = 200
        candidates, total = state.store.list_candidates(
            run_id, state=state_filter, width=width, q=q, limit=limit, offset=(page - 1) * limit
        )
        decisions = {d["candidate_key"]: d for d in dataset["decisions"]}
        job = state.job(run_id)
        running = bool(job and not job.done.is_set())
        return render_template(
            "run.html",
            dataset=dataset,
            run=run,
            candidates=candidates,
            total=total,
            page=page,
            limit=limit,
            counts=state.store.state_counts(run_id),
            decisions=decisions,
            running=running,
            state_filter=state_filter or "all",
            width=width,
            q=q or "",
        )

    @app.get("/datasets/<dataset_id>/runs/<run_id>/status")
    def run_status(dataset_id: str, run_id: str):
        run = state.store.get_run(run_id)
        if not run:
            abort(404)
        job = state.job(run_id)
        return jsonify(
            {
                "status": run["status"],
                "running": bool(job and not job.done.is_set()),
                "error": job.error if job else None,
                "counts": state.store.state_counts(run_id),
                "scope": run["scope"],
            }
        )

    @app.get("/datasets/<dataset_id>/runs/<run_id>/candidates/<path:candidate_key>")
    def candidate_page(dataset_id: str, run_id: str, candidate_key: str):
        dataset = state.store.get_dataset(dataset_id)
        cand = state.store.get_candidate(run_id, candidate_key)
        if not dataset or not cand:
            abort(404)
        decision = next(
            (d for d in dataset["decisions"] if d["candidate_key"] == candidate_key), None
        )
        cex_limit = 200
        cex = state.store.counterexamples(run_id, candidate_key, limit=cex_limit)
        total_cex = state.store.counterexample_count(run_id, candidate_key)
        return render_template(
            "candidate.html",
            dataset=dataset,
            run_id=run_id,
            cand=cand,
            decision=decision,
            counterexamples=cex,
            counterexamples_total=total_cex,
            counterexamples_shown_limit=cex_limit,
        )

    @app.get("/datasets/<dataset_id>/runs/<run_id>/candidates/<path:candidate_key>/counterexamples.csv")
    def counterexamples_csv(dataset_id: str, run_id: str, candidate_key: str):
        cand = state.store.get_candidate(run_id, candidate_key)
        if not cand:
            abort(404)
        child_cols = cand["child_cols"]
        lines = ["row," + ",".join(_csv_cell(c) for c in child_cols) + ",occurrences,null_key"]
        for item in state.store.counterexamples(run_id, candidate_key):
            row = [str(item.get("row", ""))]
            row += [_csv_cell("" if v is None else str(v)) for v in item.get("values", [])]
            row += [str(item.get("occurrences", "")), "yes" if item.get("null_key") else "no"]
            lines.append(",".join(row))
        payload = "﻿" + "\n".join(lines) + "\n"
        filename = f"counterexamples-{abs(hash(candidate_key)) % 10**8}.csv"
        return send_file(
            io.BytesIO(payload.encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename,
        )

    @app.post("/datasets/<dataset_id>/runs/<run_id>/candidates/<path:candidate_key>/decision")
    def decide(dataset_id: str, run_id: str, candidate_key: str):
        dataset = state.store.get_dataset(dataset_id)
        cand = state.store.get_candidate(run_id, candidate_key)
        if not dataset or not cand:
            abort(404)
        action = request.form.get("action")
        note = request.form.get("note", "")
        wants_json = "application/json" in (request.headers.get("Accept") or "")
        if action == "clear":
            state.store.clear_decision(dataset_id, candidate_key)
            if wants_json:
                return jsonify({"ok": True, "decision": None})
            return redirect(url_for("candidate_page", dataset_id=dataset_id, run_id=run_id, candidate_key=candidate_key))
        if action not in ("confirmed", "excluded"):
            abort(400)
        if action == "confirmed" and cand["state"] == "partial":
            allow_partial = request.form.get("allow_partial") == "on"
            if not allow_partial:
                message = "该候选存在不能匹配的记录；确认前必须勾选「我知道仍存在缺口」。"
                if wants_json:
                    return jsonify({"ok": False, "error": message}), 400
                return render_template(
                    "candidate.html",
                    dataset=dataset,
                    run_id=run_id,
                    cand=cand,
                    decision=None,
                    decision_error=message,
                ), 400
        spec = {
            "child_table": cand["child_table"],
            "child_cols": cand["child_cols"],
            "parent_table": cand["parent_table"],
            "parent_cols": cand["parent_cols"],
            "mapping_mode": cand["mapping_mode"],
            "effective": cand["effective"],
            "child_logical": cand["child_logical"],
            "parent_logical": cand["parent_logical"],
            "null_policy": cand["stats"].get("null_policy", "skip_child_nulls"),
            "state_at_decision": cand["state"],
            "coverage_at_decision": cand["stats"].get("key_coverage"),
        }
        try:
            state.store.set_decision(dataset_id, candidate_key, action, spec, note)
        except ValueError as exc:
            if wants_json:
                return jsonify({"ok": False, "error": str(exc)}), 409
            return render_template(
                "candidate.html",
                dataset=dataset,
                run_id=run_id,
                cand=cand,
                decision=None,
                decision_error=str(exc),
            ), 409
        if wants_json:
            return jsonify({"ok": True, "decision": action})
        return redirect(url_for("candidate_page", dataset_id=dataset_id, run_id=run_id, candidate_key=candidate_key))

    # --- validation ------------------------------------------------------
    @app.post("/datasets/<dataset_id>/validate")
    def validate_batch(dataset_id: str):
        dataset = state.store.get_dataset(dataset_id)
        if not dataset:
            abort(404)
        relations = state.store.confirmed_relations(dataset_id)
        if not relations:
            return render_template(
                "validation.html",
                dataset=dataset,
                result=None,
                comparison=None,
                error="没有已确认的引用关系，无法执行入仓校验。请先在候选页确认关系。",
                validations=state.store.list_validations(dataset_id),
            ), 400
        label = request.form.get("label", "").strip()
        files = request.files.getlist("files")
        dest_dir = Path(state.upload_root) / dataset_id / ("validate-" + uuid.uuid4().hex[:8])
        dest_dir.mkdir(parents=True, exist_ok=True)
        present: dict[str, str] = {}
        known = {t["name"]: t for t in dataset["tables"]}
        for upload_file in files:
            if not upload_file.filename:
                continue
            ext = Path(upload_file.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            name = _table_name_from(upload_file.filename, set())
            # Prefer an exact existing table name when the stem matches.
            stem = Path(upload_file.filename).stem
            match = next((t for t in known if t == stem), None)
            if match:
                name = match
            target = dest_dir / _safe_name(upload_file.filename)
            upload_file.save(target)
            present[name] = str(target)
        if not present:
            return render_template(
                "validation.html",
                dataset=dataset,
                result=None,
                comparison=None,
                error="本批次没有可用文件。",
                validations=state.store.list_validations(dataset_id),
            ), 400
        try:
            result = run_validation(relations, present, publication_label=label)
        except (ValueError, OSError) as exc:
            return render_template(
                "validation.html",
                dataset=dataset,
                result=None,
                comparison=None,
                error=f"校验失败：{exc}",
                validations=state.store.list_validations(dataset_id),
            ), 400
        referenced = {r["parent_table"] for r in relations} | {
            r["child_table"] for r in relations
        }
        result["missing_tables"] = sorted(referenced - set(present))
        previous = state.store.previous_validation(dataset_id)
        comparison = None
        if previous:
            comparison = compare_batches(
                previous["result"], previous["files"], result, present
            )
        validation_id = state.store.save_validation(
            dataset_id, relations, present, result, label
        )
        return render_template(
            "validation.html",
            dataset=dataset,
            result=result,
            comparison=comparison,
            validation_id=validation_id,
            files=present,
            error=None,
            validations=state.store.list_validations(dataset_id),
        )

    @app.get("/datasets/<dataset_id>/validations/<validation_id>")
    def validation_page(dataset_id: str, validation_id: str):
        dataset = state.store.get_dataset(dataset_id)
        rec = state.store.get_validation(validation_id)
        if not dataset or not rec:
            abort(404)
        previous = state.store.previous_validation(dataset_id, exclude_id=validation_id)
        comparison = None
        if previous:
            comparison = compare_batches(previous["result"], previous["files"], rec["result"], rec["files"])
        return render_template(
            "validation.html",
            dataset=dataset,
            result=rec["result"],
            comparison=comparison,
            validation_id=validation_id,
            files=rec["files"],
            error=None,
            validations=state.store.list_validations(dataset_id),
        )

    @app.get("/datasets/<dataset_id>/manifest.json")
    def manifest(dataset_id: str):
        dataset = state.store.get_dataset(dataset_id)
        if not dataset:
            abort(404)
        relations = state.store.confirmed_relations(dataset_id)
        payload = json.dumps({"dataset": dataset["name"], "relations": relations}, ensure_ascii=False, indent=2)
        return send_file(
            io.BytesIO(payload.encode("utf-8")),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"confirmed-relations-{dataset_id}.json",
        )

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.errorhandler(404)
    def not_found(_exc):
        return (
            render_template(
                "error.html",
                code=404,
                message="找不到这个页面。运行或候选可能已被删除。",
            ),
            404,
        )

    @app.errorhandler(413)
    def too_large(_exc):
        return (
            render_template(
                "error.html",
                code=413,
                message=f"上传内容超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 限制。请拆分文件后重试。",
            ),
            413,
        )

    return app


def _csv_cell(value: str) -> str:
    if any(ch in value for ch in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _params_from_form(form, dataset: dict[str, Any]) -> SearchParams:
    width = int(form.get("width", 2))
    null_policy = form.get("null_policy", "skip_child_nulls")
    mapping = form.get("mapping", "auto")
    max_verified = int(form.get("max_verified", 400))
    max_seconds = float(form.get("max_seconds", 20))
    include_self = form.get("include_self") == "on"
    column_limit: dict[str, tuple[str, ...]] = {}
    for table in dataset["tables"]:
        raw = form.get(f"cols_{table['name']}", "")
        if raw.strip():
            wanted = [c.strip() for c in raw.split(",") if c.strip()]
            known = {c["name"] for c in table["columns"]}
            unknown = [c for c in wanted if c not in known]
            if unknown:
                raise ValueError(f"表 {table['name']} 中不存在列：{', '.join(unknown)}")
            column_limit[table["name"]] = tuple(wanted)
    params = SearchParams(
        width_max=width,
        null_policy=null_policy,
        mapping=mapping,
        column_limit=column_limit,
        include_self=include_self,
        max_verified=max(0, max_verified),
        max_seconds=max(0.1, max_seconds),
    )
    params.validate()
    return params


def _load_dataset_tables(dataset: dict[str, Any]):
    tables = []
    for t in dataset["tables"]:
        decl = {c["name"]: c["type"] for c in t["columns"]}
        tables.append(load_table(t["name"], t["path"], decl))
    return tables


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RefGap 网页界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=".refgap/store.sqlite3")
    parser.add_argument("--uploads", default=None)
    args = parser.parse_args()
    app = create_app(args.db, args.uploads)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
