"""Persistence, resumption, decisions, and gap comparison."""
from __future__ import annotations

import json

import pytest

from refgap.loader import load_table
from refgap.search import SearchParams, run_search
from refgap.store import Store


def tables(make_csv):
    # Two columns per table. child[k] has a hole inside the parent range, so
    # it is a genuine "partial" candidate (not structurally pruned), and
    # child[v] is contained. That gives two candidates to verify.
    a = load_table(
        "parent",
        str(make_csv("parent", ["k", "v"], [["A", "x"], ["C", "y"], ["D", "z"]])),
    )
    b = load_table(
        "child",
        str(make_csv("child", ["k", "v"], [["A", "x"], ["B", "y"]])),
    )
    return [b, a]


def test_run_and_candidates_persist(make_csv, tmp_path):
    store = Store(tmp_path / "s.sqlite3")
    ds = store.create_dataset("d")
    ts = tables(make_csv)
    params = SearchParams(width_max=1, mapping="auto", max_verified=100)
    run_id = store.create_run(ds, params.to_dict(), [{"name": t.name} for t in ts])
    out = run_search(ts, params, on_result=lambda r: store.save_candidate(run_id, r))
    store.update_run_scope(run_id, out["scope"], "completed")

    run = store.get_run(run_id)
    assert run["params"]["width_max"] == 1
    assert run["params"]["mapping"] == "auto"
    assert run["table_snapshot"][0]["name"] == "child"
    cands, total = store.list_candidates(run_id)
    assert total == len(out["results"])
    key = "child[k] -> parent[k] @auto"
    cand = store.get_candidate(run_id, key)
    assert cand is not None
    assert cand["state"] == "partial"
    cex = store.counterexamples(run_id, key)
    assert [c["values"] for c in cex] == [["B"]]
    assert store.counterexample_count(run_id, key) == 1
    assert cand["counterexamples"] == []
    assert cand["counterexample_count"] == 1
    assert store.completed_candidate_keys(run_id) == {r.key for r in out["results"]}


def test_interrupted_run_resumes_without_recomputing(make_csv, tmp_path):
    store = Store(tmp_path / "s.sqlite3")
    ds = store.create_dataset("d")
    ts = tables(make_csv)
    params = SearchParams(width_max=2, mapping="auto", max_verified=100, max_seconds=100)
    run_id = store.create_run(ds, params.to_dict(), [])

    calls = {"n": 0}

    def stop_after_one():
        calls["n"] += 1
        return calls["n"] > 1

    first = run_search(
        ts,
        params,
        on_result=lambda r: store.save_candidate(run_id, r),
        should_stop=stop_after_one,
    )
    store.update_run_scope(run_id, first["scope"], "interrupted")
    assert first["scope"]["interrupted"] is True
    stored = store.completed_candidate_keys(run_id)
    assert first["scope"]["verified_this_pass"] == len(stored)
    assert first["scope"]["unverified"] > 0

    second = run_search(
        ts,
        params,
        completed_keys=stored,
        on_result=lambda r: store.save_candidate(run_id, r),
    )
    assert second["scope"]["already_verified"] == len(stored)
    assert second["scope"]["unverified"] == 0
    # Nothing already stored was emitted again.
    assert not ({r.key for r in second["results"]} & stored)
    # Structurally pruned candidates are deterministic across the resume.
    assert second["scope"]["pruned"] == first["scope"]["pruned"]
    assert second["scope"]["eligible"] == first["scope"]["eligible"]


def test_budget_exhaustion_reports_actual_scope(make_csv):
    ts = tables(make_csv)
    params = SearchParams(width_max=2, mapping="auto", max_verified=1, max_seconds=100)
    out = run_search(ts, params)
    scope = out["scope"]
    assert scope["budget_exhausted"] is True
    assert scope["verified"] == 1
    assert scope["unverified"] == scope["eligible"] - 1
    assert "未进入核验" in scope["note"]


def test_decisions_are_per_dataset_and_only_confirmed_export(make_csv, tmp_path):
    store = Store(tmp_path / "s.sqlite3")
    ds = store.create_dataset("d")
    spec = {
        "child_table": "child",
        "child_cols": ["k"],
        "parent_table": "parent",
        "parent_cols": ["k"],
        "mapping_mode": "auto",
        "effective": ["none"],
        "child_logical": ["string"],
        "parent_logical": ["string"],
        "null_policy": "skip_child_nulls",
    }
    store.set_decision(ds, "child[k] -> parent[k] @auto", "confirmed", spec, "ok")
    store.set_decision(ds, "x[y] -> z[y] @auto", "excluded", spec, "guess")
    confirmed = store.confirmed_relations(ds)
    assert [c["candidate_key"] for c in confirmed] == ["child[k] -> parent[k] @auto"]
    assert confirmed[0]["status"] == "confirmed"


def test_opposite_decision_requires_explicit_clear(make_csv, tmp_path):
    store = Store(tmp_path / "s.sqlite3")
    ds = store.create_dataset("d")
    spec = {"child_table": "c", "child_cols": ["k"], "parent_table": "p", "parent_cols": ["k"]}
    store.set_decision(ds, "c[k] -> p[k] @auto", "confirmed", spec)
    with pytest.raises(ValueError):
        store.set_decision(ds, "c[k] -> p[k] @auto", "excluded", spec)
    store.clear_decision(ds, "c[k] -> p[k] @auto")
    store.set_decision(ds, "c[k] -> p[k] @auto", "excluded", spec)
    assert store.confirmed_relations(ds) == []


def test_validation_history_and_previous_lookup(tmp_path):
    store = Store(tmp_path / "s.sqlite3")
    ds = store.create_dataset("d")
    first = store.save_validation(ds, [], {}, {"ok": True, "relations": []}, "batch-1")
    second = store.save_validation(ds, [], {}, {"ok": False, "relations": [{"orphan_rows": 2}]}, "batch-2")
    prev = store.previous_validation(ds, exclude_id=second)
    assert prev["id"] == first
    latest = store.previous_validation(ds)
    assert latest["id"] == second
    listing = store.list_validations(ds)
    assert len(listing) == 2
    assert listing[0]["label"] == "batch-2"


def test_previous_validation_is_insertion_order_not_timestamp(tmp_path):
    """Batches saved within the same second must still diff against the
    batch that preceded them, not a later one."""
    store = Store(tmp_path / "s.sqlite3")
    ds = store.create_dataset("d")
    first = store.save_validation(ds, [], {}, {"ok": True, "relations": []}, "batch-1")
    second = store.save_validation(ds, [], {}, {"ok": True, "relations": []}, "batch-2")
    third = store.save_validation(ds, [], {}, {"ok": True, "relations": []}, "batch-3")
    conn = store._conn
    conn.execute("UPDATE validations SET created_at = '2026-11-23T00:00:00Z'")
    assert store.previous_validation(ds, exclude_id=first) is None
    assert store.previous_validation(ds, exclude_id=second)["id"] == first
    assert store.previous_validation(ds, exclude_id=third)["id"] == second
