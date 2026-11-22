"""Compare validation batches, showing how reference gaps changed.

A "gap" is the set of child keys that could not be resolved in the parent.
Keys are encoded with the *validator's* own canonicalisation, so the gap list
shown in the UI is exactly the set the publishing gate rejects on.
"""
from __future__ import annotations

from typing import Any, Sequence

from validator.core import CastError, canonical, load_file


def _logical(rel: dict[str, Any], side: str, pos: int) -> str:
    types = rel.get(f"{side}_logical") or []
    return types[pos] if pos < len(types) else ""


def relation_identity(rel: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rel.get("child_table"),
        tuple(rel.get("child_cols", [])),
        rel.get("parent_table"),
        tuple(rel.get("parent_cols", [])),
        tuple(rel.get("effective") or ()),
        rel.get("null_policy", "skip_child_nulls"),
    )


def gap_keys(relation: dict[str, Any], files: dict[str, str]) -> dict[tuple, dict[str, Any]]:
    """Return {canonical child key: {row, values, occurrences}} for one relation."""
    child = load_file(files[relation["child_table"]], relation["child_table"])
    parent = load_file(files[relation["parent_table"]], relation["parent_table"])
    cols = relation["child_cols"]
    modes = relation.get("effective") or ["none"] * len(cols)
    if len(modes) != len(cols):
        modes = [relation.get("mapping_effective", "none")] * len(cols)
    policy = relation.get("null_policy", "skip_child_nulls")

    child_idx = [child.col(c) for c in cols]
    parent_idx = [parent.col(c) for c in relation["parent_cols"]]

    parent_keys: set[tuple] = set()
    for row in parent.rows:
        parts = tuple(
            canonical(row[i] if i < len(row) else "", m, _logical(relation, "parent", pos))
            for pos, (i, m) in enumerate(zip(parent_idx, modes))
        )
        if policy == "match_nulls":
            if all(p[0] == "null" for p in parts):
                continue
        elif any(p[0] == "null" for p in parts):
            continue
        parent_keys.add(parts)

    gaps: dict[tuple, dict[str, Any]] = {}
    for row_no, row in enumerate(child.rows, start=1):
        parts = tuple(
            canonical(row[i] if i < len(row) else "", m, _logical(relation, "child", pos))
            for pos, (i, m) in enumerate(zip(child_idx, modes))
        )
        has_null = any(p[0] == "null" for p in parts)
        all_null = all(p[0] == "null" for p in parts)
        if policy == "skip_child_nulls" and has_null:
            continue
        if policy == "match_nulls" and all_null:
            continue
        if parts in parent_keys:
            continue
        entry = gaps.setdefault(
            parts,
            {"row": row_no, "values": [row[i] if i < len(row) else "" for i in child_idx], "occurrences": 0},
        )
        entry["occurrences"] += 1
    return gaps


def compare_batches(
    previous: dict[str, Any] | None,
    previous_files: dict[str, str] | None,
    current: dict[str, Any],
    current_files: dict[str, str],
) -> dict[str, Any]:
    """Diff relation outcomes, and exact gap keys when both batches have files.

    File failures are reported per relation; they never change the pass/fail
    verdict already computed by the validator.
    """
    prev_relations = {relation_identity(r): r for r in (previous or {}).get("relations", [])}
    cur_relations = {relation_identity(r): r for r in current.get("relations", [])}
    rows: list[dict[str, Any]] = []
    for ident in sorted(cur_relations.keys() | prev_relations.keys(), key=str):
        cur = cur_relations.get(ident)
        prev = prev_relations.get(ident)
        cur_orphans = (cur or {}).get("orphan_rows", 0)
        prev_orphans = (prev or {}).get("orphan_rows", 0)
        entry: dict[str, Any] = {
            "candidate_key": (cur or prev or {}).get("candidate_key", ""),
            "status_previous": (prev or {}).get("status", "absent"),
            "status_current": (cur or {}).get("status", "absent"),
            "orphans_previous": prev_orphans,
            "orphans_current": cur_orphans,
            "delta": cur_orphans - prev_orphans,
            "new_gap_keys": None,
            "resolved_gap_keys": None,
            "persisting_gap_keys": None,
        }
        if cur and prev and previous_files and current_files:
            try:
                prev_gaps = gap_keys(prev, previous_files)
                cur_gaps = gap_keys(cur, current_files)
                new_keys = set(cur_gaps) - set(prev_gaps)
                entry["new_gap_keys"] = len(new_keys)
                entry["resolved_gap_keys"] = len(set(prev_gaps) - set(cur_gaps))
                entry["persisting_gap_keys"] = len(set(cur_gaps) & set(prev_gaps))
                entry["new_gap_samples"] = [
                    cur_gaps[k] for k in sorted(new_keys, key=lambda k: cur_gaps[k]["row"])[:10]
                ]
            except (KeyError, CastError, OSError, ValueError) as exc:
                entry["key_diff_error"] = str(exc)
        rows.append(entry)
    return {
        "relations": rows,
        "new_failures": sum(1 for r in rows if r["orphans_current"] > r["orphans_previous"]),
        "improved_relations": sum(1 for r in rows if r["orphans_current"] < r["orphans_previous"]),
        "total_orphans_previous": sum(r["orphans_previous"] for r in rows),
        "total_orphans_current": sum(r["orphans_current"] for r in rows),
    }
