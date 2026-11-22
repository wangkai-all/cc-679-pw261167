"""Batch-to-batch gap comparison."""
from __future__ import annotations

from refgap.gaps import compare_batches


def rel():
    return {
        "candidate_key": "orders[store] -> stores[store] @auto",
        "status": "confirmed",
        "child_table": "orders",
        "child_cols": ["store"],
        "parent_table": "stores",
        "parent_cols": ["store"],
        "effective": ["none"],
        "child_logical": ["string"],
        "parent_logical": ["string"],
        "null_policy": "skip_child_nulls",
    }


def write(tmp_path, name, rows):
    path = tmp_path / f"{name}.csv"
    path.write_text("store\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def outcome(gaps, status):
    return {
        "relations": [
            {
                **rel(),
                "status": status,
                "orphan_rows": len(gaps),
                "sample": [{"row": 1, "values": [g], "occurrences": 1} for g in gaps],
            }
        ]
    }


def test_new_and_resolved_gaps_are_named(tmp_path):
    prev_files = {
        "orders": write(tmp_path, "o1", ["S1", "S9"]),
        "stores": write(tmp_path, "s1", ["S1"]),
    }
    cur_files = {
        "orders": write(tmp_path, "o2", ["S1", "S8"]),
        "stores": write(tmp_path, "s2", ["S1"]),
    }
    cmp = compare_batches(outcome(["S9"], "fail"), prev_files, outcome(["S8"], "fail"), cur_files)
    row = cmp["relations"][0]
    assert row["new_gap_keys"] == 1
    assert row["resolved_gap_keys"] == 1
    assert row["persisting_gap_keys"] == 0
    assert row["new_gap_samples"][0]["values"] == ["S8"]
    assert cmp["total_orphans_current"] == 1


def test_persisting_gap_is_counted(tmp_path):
    prev_files = {
        "orders": write(tmp_path, "o1", ["S9"]),
        "stores": write(tmp_path, "s1", ["S1"]),
    }
    cur_files = {
        "orders": write(tmp_path, "o2", ["S9", "S9"]),
        "stores": write(tmp_path, "s2", ["S1"]),
    }
    cmp = compare_batches(outcome(["S9"], "fail"), prev_files, outcome(["S9"], "fail"), cur_files)
    row = cmp["relations"][0]
    assert row["persisting_gap_keys"] == 1
    assert row["new_gap_keys"] == 0


def test_without_previous_batch_there_is_no_diff(tmp_path):
    cmp = compare_batches(None, None, outcome([], "pass"), {})
    assert cmp["relations"][0]["status_previous"] == "absent"
    assert cmp["relations"][0]["new_gap_keys"] is None


def test_validator_outcomes_carry_identity_so_diff_works_end_to_end(tmp_path):
    """The stored result must contain the relation identity; otherwise the
    batch comparison silently degrades to "—" for every gap-key column."""
    from validator.core import validate

    files = {
        "orders": write(tmp_path, "o1", ["S9"]),
        "stores": write(tmp_path, "s1", ["S1"]),
    }
    result = validate([rel()], files)
    stored = result["relations"][0]
    assert stored["child_table"] == "orders"
    assert stored["child_cols"] == ["store"]
    cmp = compare_batches(result, files, result, files)
    row = cmp["relations"][0]
    assert "key_diff_error" not in row
    assert row["persisting_gap_keys"] == 1
    assert row["new_gap_keys"] == 0
