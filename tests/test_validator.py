"""The ingress gate: only confirmed relations, orphans block publication."""
from __future__ import annotations

import json

import pytest

from validator.cli import main as cli_main
from validator.core import ManifestError, load_manifest, validate


def rel(child, child_cols, parent, parent_cols, mapping="none", null_policy="skip_child_nulls"):
    logical = ["string"] * len(child_cols)
    return {
        "status": "confirmed",
        "child_table": child,
        "child_cols": child_cols,
        "parent_table": parent,
        "parent_cols": parent_cols,
        "mapping_mode": "auto",
        "effective": [mapping] * len(child_cols),
        "child_logical": logical,
        "parent_logical": logical,
        "null_policy": null_policy,
        "candidate_key": f"{child}[{','.join(child_cols)}] -> {parent}[{','.join(parent_cols)}]",
    }


def write(tmp_path, name, header, rows):
    path = tmp_path / f"{name}.csv"
    path.write_text(
        ",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_orphans_block_publication_and_list_missing_keys(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store"], [["S1"], ["S99"]]),
        "stores": write(tmp_path, "stores", ["store"], [["S1"]]),
    }
    result = validate([rel("orders", ["store"], "stores", ["store"])], files)
    assert result["ok"] is False
    assert result["publication_allowed"] is False
    rr = result["relations"][0]
    assert rr["status"] == "fail"
    assert rr["orphan_rows"] == 1
    assert rr["sample"][0]["values"] == ["S99"]
    assert rr["sample"][0]["row"] == 2


def test_clean_batch_publishes(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store"], [["S1"]]),
        "stores": write(tmp_path, "stores", ["store"], [["S1"]]),
    }
    result = validate([rel("orders", ["store"], "stores", ["store"])], files)
    assert result["ok"] is True


def test_excluded_relations_are_never_enforced(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store"], [["S99"]]),
        "stores": write(tmp_path, "stores", ["store"], [["S1"]]),
    }
    excluded = rel("orders", ["store"], "stores", ["store"])
    excluded["status"] = "excluded"
    result = validate([excluded], files)
    assert result["ok"] is True
    assert result["enforced_count"] == 0
    assert result["not_applicable_count"] == 1


def test_missing_parent_is_a_failure_not_a_silent_pass(tmp_path):
    files = {"orders": write(tmp_path, "orders", ["store"], [["S1"]])}
    result = validate([rel("orders", ["store"], "stores", ["store"])], files)
    assert result["ok"] is False
    rr = result["relations"][0]
    assert "缺失" in rr["reason"]
    # Structural failures must not be reported as orphan rows.
    assert rr["kind"] == "structure"
    assert result["orphan_relation_count"] == 0
    assert result["structure_failure_count"] == 1
    assert "结构校验" in result["refusal_reason"]
    assert "孤儿" not in result["refusal_reason"]


def test_refusal_reason_distinguishes_orphans_from_structure(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store"], [["S99"]]),
        "stores": write(tmp_path, "stores", ["store"], [["S1"]]),
        "other": write(tmp_path, "other", ["store"], [["S1"]]),
    }
    result = validate(
        [
            rel("orders", ["store"], "stores", ["store"]),
            rel("other", ["store"], "missing_parent", ["store"]),
        ],
        files,
    )
    assert result["failures"] == 2
    assert result["orphan_relation_count"] == 1
    assert result["structure_failure_count"] == 1
    assert result["orphan_rows"] == 1
    assert "缺失" in result["refusal_reason"]
    assert "结构校验" in result["refusal_reason"]


def test_missing_child_table_is_not_applicable(tmp_path):
    files = {"stores": write(tmp_path, "stores", ["store"], [["S1"]])}
    result = validate([rel("orders", ["store"], "stores", ["store"])], files)
    assert result["ok"] is True
    assert result["relations"][0]["status"] == "not_applicable"


def test_missing_child_column_is_not_applicable(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store_v2"], [["S1"]]),
        "stores": write(tmp_path, "stores", ["store"], [["S1"]]),
    }
    result = validate([rel("orders", ["store"], "stores", ["store"])], files)
    assert result["relations"][0]["status"] == "not_applicable"
    assert "缺少列" in result["relations"][0]["reason"]


def test_null_policy_skip_counts_instead_of_failing(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store", "v"], [["S1", "x"], ["", "y"]]),
        "stores": write(tmp_path, "stores", ["store", "v"], [["S1", "x"]]),
    }
    result = validate([rel("orders", ["store"], "stores", ["store"])], files)
    assert result["ok"] is True
    assert result["relations"][0]["null_rows"] == 1


def test_null_policy_fail_turns_nulls_into_orphans(tmp_path):
    files = {
        "orders": write(tmp_path, "orders", ["store"], [["S1"], [""]]),
        "stores": write(tmp_path, "stores", ["store"], [["S1"]]),
    }
    result = validate(
        [rel("orders", ["store"], "stores", ["store"], null_policy="fail_child_nulls")], files
    )
    assert result["ok"] is False
    assert result["relations"][0]["orphan_rows"] == 1


def test_composite_key_order_is_enforced(tmp_path):
    files = {
        "child": write(tmp_path, "child", ["a", "b"], [["P1", "S1"]]),
        "parent": write(tmp_path, "parent", ["a", "b"], [["S1", "P1"]]),
    }
    forward = validate([rel("child", ["a", "b"], "parent", ["a", "b"])], files)
    swapped = validate([rel("child", ["a", "b"], "parent", ["b", "a"])], files)
    assert forward["ok"] is False
    assert swapped["ok"] is True


def test_numeric_mapping_equates_one_and_one_point_zero(tmp_path):
    files = {
        "child": write(tmp_path, "child", ["n"], [["1.0"]]),
        "parent": write(tmp_path, "parent", ["n"], [["1"]]),
    }
    strict = validate([rel("child", ["n"], "parent", ["n"], mapping="none")], files)
    numeric = validate([rel("child", ["n"], "parent", ["n"], mapping="number")], files)
    assert strict["ok"] is False
    assert numeric["ok"] is True


def test_delimiter_in_values_cannot_collide(tmp_path):
    files = {
        "child": write(tmp_path, "child", ["a", "b"], [["a|b", "c"]]),
        "parent": write(tmp_path, "parent", ["a", "b"], [["a", "b|c"]]),
    }
    result = validate([rel("child", ["a", "b"], "parent", ["a", "b"])], files)
    assert result["ok"] is False


def test_duplicate_parent_rows_do_not_change_the_verdict(tmp_path):
    files = {
        "child": write(tmp_path, "child", ["k"], [["A"], ["B"]]),
        "parent": write(tmp_path, "parent", ["k"], [["A"], ["A"], ["A"]]),
    }
    result = validate([rel("child", ["k"], "parent", ["k"])], files)
    assert result["ok"] is False
    assert result["relations"][0]["orphan_rows"] == 1


def test_manifest_loader_rejects_incomplete_confirmed_relation(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"relations": [{"status": "confirmed", "child_table": "a"}]}))
    with pytest.raises(ManifestError):
        load_manifest(str(path))


def test_cli_exit_codes(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"relations": [rel("orders", ["store"], "stores", ["store"])]}),
        encoding="utf-8",
    )
    orders = write(tmp_path, "orders", ["store"], [["S99"]])
    stores = write(tmp_path, "stores", ["store"], [["S1"]])
    code = cli_main(
        ["--manifest", str(manifest), "--file", f"orders={orders}", "--file", f"stores={stores}"]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "S99" in out
    assert "拒绝" in out


def test_cli_refuses_empty_manifest(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"relations": []}))
    code = cli_main(["--manifest", str(manifest)])
    assert code == 2


def test_cli_unreadable_file_is_exit_code_2_not_a_publication_refusal(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"relations": [rel("orders", ["store"], "stores", ["store"])]}),
        encoding="utf-8",
    )
    code = cli_main(
        ["--manifest", str(manifest), "--file", f"orders={tmp_path / 'missing.csv'}"]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "无法读取本批文件" in err
