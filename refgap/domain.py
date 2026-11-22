"""Cached, canonical value domains and column statistics.

Domains are de-duplicated by construction. Every key is tagged so that
integer ``1`` and string ``"1"`` never collide. Domains keep every distinct
key, including keys with null components, and expose selection helpers so the
declared null policy is applied once, explicitly, by the verifier.

Duplicate rows on the parent side collapse into one key: multiplicity is
recorded (``row_counts``) for display but never used as coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .loader import Table
from .types import CastError, canonical_key

Key = tuple[tuple[str, Any], ...]


def _has_boolean(types: Sequence[str]) -> bool:
    return any(t == "boolean" for t in types)


@dataclass
class DomainStats:
    table: str
    columns: tuple[str, ...]
    logical_types: tuple[str, ...]
    effective: tuple[str, ...]
    keys: set[Key] = field(default_factory=set)
    row_counts: dict[Key, int] = field(default_factory=dict)
    sample: dict[Key, tuple[Any, ...]] = field(default_factory=dict)
    first_row: dict[Key, int] = field(default_factory=dict)
    row_count: int = 0
    errors: list[str] = field(default_factory=list)
    clean_min: tuple[Any, ...] | None = None
    clean_max: tuple[Any, ...] | None = None
    ordered: bool = False

    @property
    def cardinality(self) -> int:
        return len(self.keys)

    @property
    def empty(self) -> bool:
        return self.row_count == 0

    @property
    def null_component_rows(self) -> int:
        return sum(n for key, n in self.row_counts.items() if _key_has_null(key))

    @property
    def all_null_rows(self) -> int:
        return sum(n for key, n in self.row_counts.items() if _key_all_null(key))

    @property
    def clean_keys(self) -> set[Key]:
        return {k for k in self.keys if not _key_has_null(k)}

    @property
    def matchable_keys(self) -> set[Key]:
        """Keys eligible under MATCH-NULLS: partial nulls stay, all-null rows go."""
        return {k for k in self.keys if not _key_all_null(k)}

    @property
    def all_null(self) -> bool:
        """No key can ever participate in a reference on this side."""
        return not self.keys or all(_key_all_null(k) for k in self.keys)

    @property
    def every_key_null(self) -> bool:
        """Every distinct key is entirely null (empty domains included)."""
        return not self.keys or all(_key_all_null(k) for k in self.keys)

    def selected_keys(self, null_policy: str) -> set[Key]:
        if null_policy == "match_nulls":
            return self.matchable_keys
        return self.clean_keys

    def clean_row_count(self) -> int:
        return sum(n for k, n in self.row_counts.items() if not _key_has_null(k))


def _key_has_null(key: Key) -> bool:
    return any(part[0] == "null" for part in key)


def _key_all_null(key: Key) -> bool:
    return all(part[0] == "null" for part in key)


def _component(value: Any) -> Any:
    return None if value[0] == "null" else value[1]


class DomainCache:
    """Builds (and memoises) domains for a table column tuple under a mapping."""

    def __init__(self, tables: Sequence[Table]):
        self.tables = {t.name: t for t in tables}
        self._cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], DomainStats] = {}

    def stats(self, table_name: str, columns: Sequence[str], effective: Sequence[str]) -> DomainStats:
        columns = tuple(columns)
        effective = tuple(effective)
        if len(columns) != len(effective):
            raise ValueError("effective mapping must have one entry per column")
        cache_key = (table_name, columns, effective)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        table = self.tables[table_name]
        idxs = [table.column(c).index for c in columns]
        logical = tuple(table.columns[i].logical_type for i in idxs)
        stats = DomainStats(
            table=table_name,
            columns=columns,
            logical_types=logical,
            effective=effective,
            row_count=len(table.rows),
        )
        if stats.row_count:
            for row_no, row in enumerate(table.rows, start=1):
                raw = tuple(row[i] for i in idxs)
                try:
                    key = tuple(
                        canonical_key(value, logical_type, mode)
                        for logical_type, mode, value in zip(logical, effective, raw)
                    )
                except CastError as exc:
                    stats.errors.append(f"第 {row_no} 行: {exc}")
                    continue
                stats.keys.add(key)
                stats.row_counts[key] = stats.row_counts.get(key, 0) + 1
                if key not in stats.sample:
                    stats.sample[key] = raw
                    stats.first_row[key] = row_no
        self._compute_range(stats)
        self._cache[cache_key] = stats
        return stats

    def _compute_range(self, stats: DomainStats) -> None:
        clean = stats.clean_keys
        if stats.errors or not clean or _has_boolean(stats.logical_types):
            return
        if any(part[0] == "null" for key in clean for part in key):
            return
        try:
            mins: list[Any] = []
            maxs: list[Any] = []
            for pos in range(len(stats.columns)):
                comps = [_component(k[pos]) for k in clean]
                mins.append(min(comps))
                maxs.append(max(comps))
        except TypeError:
            return
        stats.clean_min = tuple(mins)
        stats.clean_max = tuple(maxs)
        stats.ordered = True
