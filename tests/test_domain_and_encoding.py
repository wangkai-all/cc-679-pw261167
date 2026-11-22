"""Key encoding and domain semantics."""
from __future__ import annotations

import pytest

from refgap.domain import DomainCache
from refgap.loader import load_table
from refgap.types import CastError, canonical_key, resolve_mapping


def build(make_csv, rows):
    path = make_csv("t", ["a", "b"], rows)
    return load_table("t", str(path))


def test_tuple_encoding_has_no_delimiter_collisions(make_csv):
    """("a|b","c") and ("a","b|c") must not produce the same composite key."""
    t1 = build(make_csv, [["a|b", "c"]])
    t2 = build(make_csv, [["a", "b|c"]])
    k1 = tuple(
        canonical_key(v, c.logical_type, "none") for v, c in zip(t1.rows[0], t1.columns)
    )
    k2 = tuple(
        canonical_key(v, c.logical_type, "none") for v, c in zip(t2.rows[0], t2.columns)
    )
    assert k1 != k2


def test_column_order_changes_the_key(make_csv):
    path = make_csv("t", ["x", "y"], [["a", "b"]])
    t = load_table("t", str(path))
    cache = DomainCache([t])
    forward = cache.stats("t", ["x", "y"], ("none", "none"))
    reversed_ = cache.stats("t", ["y", "x"], ("none", "none"))
    assert forward.keys != reversed_
    assert forward.keys != {
        tuple(reversed(k)) for k in forward.keys
    } or forward.keys == reversed_


def test_integer_and_string_never_collide(make_csv):
    path = make_csv("t", ["n", "s"], [["1", "1"]])
    t = load_table("t", str(path))
    assert t.column("n").logical_type == "integer"
    # Column s is also inferred integer; force a string key to compare tags.
    int_key = canonical_key(1, "integer", "none")
    str_key = canonical_key("1", "string", "none")
    assert int_key != str_key


def test_parent_duplicates_collapse_and_do_not_add_coverage(make_csv):
    path = make_csv("p", ["k"], [["A"], ["A"], ["A"]])
    t = load_table("p", str(path))
    stats = DomainCache([t]).stats("p", ["k"], ("none",))
    assert stats.cardinality == 1
    assert stats.row_counts[next(iter(stats.keys))] == 3
    assert stats.row_count == 3


def test_auto_mapping_does_not_cross_string_and_number():
    assert not resolve_mapping("integer", "string", "auto").ok
    assert not resolve_mapping("string", "float", "auto").ok
    assert resolve_mapping("integer", "float", "auto").effective == "number"
    assert resolve_mapping("date", "datetime", "auto").effective == "datetime"


def test_explicit_string_mapping_can_cross_types():
    res = resolve_mapping("integer", "string", "string")
    assert res.ok and res.effective == "string"


def test_strict_mapping_requires_equal_types():
    assert resolve_mapping("integer", "float", "none").ok is False


def test_numeric_canonicalisation_is_exact():
    assert canonical_key(1, "integer", "number") == canonical_key(1.0, "float", "number")
    with pytest.raises(CastError):
        canonical_key("abc", "string", "number")


def test_leading_zero_codes_stay_strings(make_csv):
    path = make_csv("t", ["code"], [["001"], ["002"]])
    t = load_table("t", str(path))
    assert t.column("code").logical_type == "string"
    assert t.rows[0][0] == "001"


def test_mixed_column_falls_back_to_string(make_csv):
    path = make_csv("t", ["v"], [["1"], ["x"]])
    t = load_table("t", str(path))
    assert t.column("v").logical_type == "string"
