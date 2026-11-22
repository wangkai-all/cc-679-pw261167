"""The behavioural contract of the search.

Each test states one rule from the brief, so a regression names the rule it
breaks rather than just a function.
"""
from __future__ import annotations

import pytest

from refgap.loader import load_table
from refgap.search import (
    CONTAINED,
    PARTIAL,
    PRUNED,
    SearchParams,
    classify_candidate,
    enumerate_specs,
    run_search,
)


def tables(make_csv, spec):
    out = []
    for name, header, rows in spec:
        path = make_csv(name, header, rows)
        out.append(load_table(name, str(path)))
    return out


def results_by_key(outcome):
    return {r.key: r for r in outcome["results"]}


def test_same_column_name_is_not_evidence(make_csv):
    """A shared name with disjoint values is never a reference."""
    ts = tables(
        make_csv,
        [
            ("a", ["product_id"], [["P-1"], ["P-2"]]),
            ("b", ["product_id"], [["X-9"], ["X-8"]]),
        ],
    )
    outcome = run_search(ts, SearchParams(width_max=2, mapping="auto", max_verified=10_000))
    for r in outcome["results"]:
        assert r.state != CONTAINED
    assert outcome["scope"]["pruned_by_reason"].get("range", 0) >= 1


def test_two_singles_contained_but_pair_not_contained(make_csv):
    """Singles are a necessary condition only; the pair must be measured."""
    ts = tables(
        make_csv,
        [
            (
                "parent",
                ["store", "sku"],
                [["S1", "SKU1"], ["S2", "SKU2"]],
            ),
            (
                "child",
                ["store", "sku"],
                [["S1", "SKU2"], ["S2", "SKU1"]],
            ),
        ],
    )
    params = SearchParams(width_max=2, mapping="auto", max_verified=10_000)
    out = run_search(ts, params)
    res = results_by_key(out)
    pair = res["child[store,sku] -> parent[store,sku] @auto"]
    assert pair.state == PARTIAL
    assert pair.stats["single_column"]["both_hold"] is True
    assert pair.stats["key_coverage"] == 0.0
    assert pair.stats["missing_keys"] == 2


def test_passing_single_projection_never_removes_the_pair(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["store", "sku"], [["S1", "A"], ["S2", "B"]]),
            ("child", ["store", "sku"], [["S1", "A"]]),
        ],
    )
    params = SearchParams(width_max=2, mapping="auto", max_verified=10_000)
    pair_key = "child[store,sku] -> parent[store,sku] @auto"
    out = run_search(ts, params)
    res = results_by_key(out)
    assert pair_key in res
    assert res[pair_key].state == CONTAINED
    assert res[pair_key].stats["single_column"]["both_hold"] is True


def test_failing_single_projection_prunes_only_from_compared_keys(make_csv):
    """A value that is excluded by the null rule must not cause a false prune."""
    ts = tables(
        make_csv,
        [
            ("parent", ["store", "sku"], [["S1", "A"]]),
            # The only "S9" store value sits in a row whose sku is null, so it
            # never participates in a compared composite key.
            ("child", ["store", "sku"], [["S1", "A"], ["S9", ""]]),
        ],
    )
    params = SearchParams(width_max=2, mapping="auto", max_verified=10_000)
    out = run_search(ts, params)
    res = results_by_key(out)
    pair = res["child[store,sku] -> parent[store,sku] @auto"]
    assert pair.state == CONTAINED
    assert pair.stats["key_coverage"] == 1.0


def test_column_order_participates(make_csv):
    """Order is part of the candidate: the reversed mapping is not the same one."""
    ts = tables(
        make_csv,
        [
            ("parent", ["store", "sku"], [["S1", "P1"], ["S2", "P2"]]),
            ("child", ["store", "sku"], [["S1", "P1"], ["S2", "P2"]]),
        ],
    )
    params = SearchParams(width_max=2, mapping="auto", max_verified=10_000)
    out = run_search(ts, params)
    res = results_by_key(out)
    assert res["child[store,sku] -> parent[store,sku] @auto"].state == CONTAINED
    # Reversed column correspondence pairs store values with sku values, so it
    # is a different candidate and cannot inherit the verdict.
    assert "child[store,sku] -> parent[sku,store] @auto" not in res
    assert out["scope"]["pruned_by_reason"].get("projection", 0) >= 1


def test_parent_duplicates_cannot_inflate_coverage(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k"], [["A"], ["A"], ["A"], ["B"]]),
            ("child", ["k"], [["A"], ["A"]]),
        ],
    )
    out = run_search(ts, SearchParams(width_max=1, mapping="auto", max_verified=1000))
    r = results_by_key(out)["child[k] -> parent[k] @auto"]
    assert r.stats["key_coverage"] == 1.0
    assert r.stats["parent_rows"] == 4
    assert r.stats["parent_distinct_keys"] == 2
    assert r.stats["parent_rows_collapsed"] == 2


def test_coverage_is_by_distinct_key_and_by_row(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k"], [["A"], ["B"]]),
            ("child", ["k"], [["A"], ["A"], ["B"]]),
        ],
    )
    out = run_search(ts, SearchParams(width_max=1, mapping="auto", max_verified=1000))
    r = results_by_key(out)["child[k] -> parent[k] @auto"]
    assert r.state == CONTAINED
    assert r.stats["key_coverage"] == pytest.approx(1.0)
    assert r.stats["row_coverage"] == pytest.approx(1.0)
    assert r.stats["covered_sample"][0]["occurrences"] >= 1


def test_empty_table_is_never_a_high_confidence_reference(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k"], [["A"]]),
            ("empty", ["k"], []),
        ],
    )
    out = run_search(ts, SearchParams(width_max=2, mapping="auto", max_verified=1000))
    reasons = out["scope"]["pruned_by_reason"]
    # Every candidate involving the empty table is structurally excluded, on
    # whichever side it appears.
    assert reasons.get("child_empty", 0) >= 1
    assert reasons.get("parent_empty", 0) >= 1
    for r in out["results"]:
        assert r.confidence != "high" or "empty" not in r.key


def test_all_null_column_is_not_high_confidence(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k", "v"], [["A", "x"]]),
            ("nulls", ["k", "v"], [["", "1"], ["nan", "2"], ["N/A", "3"]]),
        ],
    )
    out = run_search(ts, SearchParams(width_max=1, mapping="auto", max_verified=1000))
    assert out["scope"]["pruned_by_reason"].get("child_all_null", 0) >= 1
    assert out["scope"]["pruned_by_reason"].get("parent_all_null", 0) >= 1
    for r in out["results"]:
        assert not (r.state == CONTAINED and r.confidence == "high" and r.key.startswith("nulls["))


def test_null_policies_change_the_verdict_explicitly(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k", "v"], [["A", "x"]]),
            ("child", ["k", "v"], [["A", "x"], ["", "y"]]),
        ],
    )
    skip = run_search(ts, SearchParams(width_max=1, null_policy="skip_child_nulls", mapping="auto", max_verified=100))
    fail = run_search(ts, SearchParams(width_max=1, null_policy="fail_child_nulls", mapping="auto", max_verified=100))
    r_skip = results_by_key(skip)["child[k] -> parent[k] @auto"]
    r_fail = results_by_key(fail)["child[k] -> parent[k] @auto"]
    assert r_skip.state == CONTAINED
    assert r_skip.stats["child_null_key_rows"] == 1
    assert r_fail.state == PARTIAL
    assert r_fail.stats["null_violation_rows"] == 1


def test_match_nulls_policy_matches_nulls(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k", "v"], [["A", "x"], ["", "y"]]),
            ("child", ["k", "v"], [["A", "x"], ["", "y"]]),
        ],
    )
    out = run_search(ts, SearchParams(width_max=1, null_policy="match_nulls", mapping="auto", max_verified=100))
    r = results_by_key(out)["child[k] -> parent[k] @auto"]
    assert r.state == CONTAINED


def test_incompatible_types_are_reported_not_compared(make_csv):
    ts = tables(
        make_csv,
        [
            ("parent", ["k"], [["A"]]),
            ("child", ["n"], [["1"]]),
        ],
    )
    out = run_search(ts, SearchParams(width_max=1, mapping="auto", max_verified=100))
    assert out["scope"]["pruned_by_reason"].get("type_incompatible", 0) >= 1


def test_duplicate_candidates_are_unique(make_csv):
    """Each (child cols, parent cols) pair appears exactly once."""
    ts = tables(
        make_csv,
        [
            ("a", ["x", "y"], [["1", "2"]]),
            ("b", ["x", "y"], [["1", "2"]]),
        ],
    )
    params = SearchParams(width_max=2, mapping="auto", max_verified=10_000)
    specs = enumerate_specs(ts, params)
    keys = [s.key for s in specs]
    assert len(keys) == len(set(keys))
