"""Inclusion-dependency search: enumerate, classify/prune, then verify exactly.

Rules encoded here (each has a test):

* A candidate is containment of values, never similarity of names. This module
  never reads column names except as opaque identifiers.
* Column order is part of the candidate: ``(a, b) -> (x, y)`` and
  ``(b, a) -> (x, y)`` are distinct.
* Composite keys are canonical, tagged tuples. ``("a|b", "c")`` can never
  collide with ``("a", "b|c")``.
* Single-column containment is a **necessary** condition for a width-2
  candidate. A failing projection may prune the pair. A passing projection is
  never evidence for the pair and never removes it from verification.
* Domains are de-duplicated, so parent duplicate rows cannot raise coverage.
* Empty tables / all-null columns are classified as degenerate and reported by
  reason; they are never verified as high-confidence references.
* Classification (cheap, deterministic) is separate from verification (budgeted).
  Budget exhaustion is reported with the exact searched range. Because
  classification is deterministic, a resumed run reproduces the same pruned
  counts and only verifies the candidates that were not yet verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from itertools import permutations, zip_longest
import time
from typing import Any, Callable, Iterable, Sequence

from .domain import DomainCache, DomainStats, Key
from .loader import Table
from .types import NULL_POLICIES, MappingResolution, describe_mapping, resolve_mapping

WIDTHS = (1, 2)

# Verified candidate states
CONTAINED = "contained"
PARTIAL = "partial"
ERROR = "error"

# Classification states (never stored as verified candidates)
PRUNED = "pruned"

REASON_LABELS = {
    "ok": "完全包含",
    "counterexamples": "存在不能匹配的记录",
    "child_empty": "子表为空",
    "child_all_null": "子表选中列全为空",
    "child_no_eligible_rows": "子表没有可参与比较的行",
    "parent_empty": "父表为空",
    "parent_all_null": "父表选中列全为空",
    "parent_no_eligible_rows": "父表没有可参与比较的键",
    "type_incompatible": "类型映射不兼容",
    "cast_errors": "存在无法转换的值",
    "cardinality": "子表去重键多于父表，必然不能包含",
    "range": "子表取值范围超出父表",
    "projection": "单列投影不包含，组合必然不包含",
    "budget": "预算用尽，未核验",
    "interrupted": "运行中断，未核验",
    "mapping": "类型映射问题",
}

@dataclass(frozen=True)
class CandidateSpec:
    seq: int
    child_table: str
    child_cols: tuple[str, ...]
    parent_table: str
    parent_cols: tuple[str, ...]
    mapping_mode: str
    effective: tuple[str, ...]
    logical_child: tuple[str, ...]
    logical_parent: tuple[str, ...]

    @property
    def key(self) -> str:
        return candidate_key(
            self.child_table,
            self.child_cols,
            self.parent_table,
            self.parent_cols,
            self.mapping_mode,
        )

    @property
    def width(self) -> int:
        return len(self.child_cols)


def candidate_key(
    child_table: str,
    child_cols: Sequence[str],
    parent_table: str,
    parent_cols: Sequence[str],
    mapping_mode: str,
) -> str:
    return (
        f"{child_table}[{','.join(child_cols)}] -> "
        f"{parent_table}[{','.join(parent_cols)}] @{mapping_mode}"
    )


@dataclass
class SearchParams:
    width_max: int = 2
    null_policy: str = "skip_child_nulls"
    mapping: str = "auto"
    column_limit: dict[str, tuple[str, ...]] = field(default_factory=dict)
    include_self: bool = False
    max_verified: int = 400
    max_seconds: float = 20.0
    sample_size: int = 5
    max_counterexamples_store: int = 2_000_000
    max_pruned_examples: int = 60

    def validate(self) -> None:
        if self.width_max not in WIDTHS:
            raise ValueError("组合宽度只支持 1 或 2")
        if self.null_policy not in NULL_POLICIES:
            raise ValueError(f"未知空值规则 {self.null_policy!r}")
        if self.max_verified < 0 or self.max_seconds <= 0:
            raise ValueError("预算参数必须为正")
        if self.sample_size < 0:
            raise ValueError("样例数量不能为负")

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_max": self.width_max,
            "null_policy": self.null_policy,
            "mapping": self.mapping,
            "column_limit": {k: list(v) for k, v in self.column_limit.items()},
            "include_self": self.include_self,
            "max_verified": self.max_verified,
            "max_seconds": self.max_seconds,
            "sample_size": self.sample_size,
            "max_counterexamples_store": self.max_counterexamples_store,
            "max_pruned_examples": self.max_pruned_examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchParams":
        return cls(
            width_max=int(data.get("width_max", 2)),
            null_policy=data.get("null_policy", "skip_child_nulls"),
            mapping=data.get("mapping", "auto"),
            column_limit={k: tuple(v) for k, v in (data.get("column_limit") or {}).items()},
            include_self=bool(data.get("include_self", False)),
            max_verified=int(data.get("max_verified", 400)),
            max_seconds=float(data.get("max_seconds", 20.0)),
            sample_size=int(data.get("sample_size", 5)),
            max_counterexamples_store=int(data.get("max_counterexamples_store", 2_000_000)),
            max_pruned_examples=int(data.get("max_pruned_examples", 60)),
        )


@dataclass
class CandidateResult:
    spec: CandidateSpec
    state: str
    reason: str
    stats: dict[str, Any]
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def confidence(self) -> str:
        if self.state == CONTAINED:
            if self.stats.get("degenerate"):
                return "insufficient"
            if self.stats.get("trivial_domain"):
                return "medium"
            return "high"
        if self.state == PARTIAL:
            ratio = self.stats.get("key_coverage", 0.0)
            return "medium" if ratio >= 0.98 else "low"
        return "none"


def _columns_for(table: Table, params: SearchParams) -> list[str]:
    names = table.column_names()
    limited = params.column_limit.get(table.name)
    if limited is None:
        return names
    unknown = [c for c in limited if c not in names]
    if unknown:
        raise ValueError(f"表 {table.name} 中不存在列 {', '.join(unknown)}")
    if not limited:
        raise ValueError(f"表 {table.name} 的候选列不能为空")
    return list(limited)


def enumerate_specs(tables: Sequence[Table], params: SearchParams) -> list[CandidateSpec]:
    params.validate()
    by_name = {t.name: t for t in tables}
    cols = {t.name: _columns_for(t, params) for t in tables}
    specs: list[CandidateSpec] = []
    seq = 0
    table_order = {t.name: i for i, t in enumerate(tables)}
    ordered_tables = sorted(tables, key=lambda t: table_order[t.name])
    for child in ordered_tables:
        for parent in ordered_tables:
            if child.name == parent.name and not params.include_self:
                continue
            for width in range(1, params.width_max + 1):
                if width > len(cols[child.name]) or width > len(cols[parent.name]):
                    continue
                for child_cols in permutations(cols[child.name], width):
                    for parent_cols in permutations(cols[parent.name], width):
                        if child.name == parent.name and child_cols == parent_cols:
                            # Same table, same columns in the same order is a
                            # tautology, not a reference.
                            continue
                        effective: list[str] = []
                        logical_child: list[str] = []
                        logical_parent: list[str] = []
                        for cc, pc in zip(child_cols, parent_cols):
                            ct = by_name[child.name].column(cc).logical_type
                            pt = by_name[parent.name].column(pc).logical_type
                            logical_child.append(ct)
                            logical_parent.append(pt)
                            res = resolve_mapping(ct, pt, params.mapping)
                            effective.append(res.effective if res.ok else "!")
                        seq += 1
                        specs.append(
                            CandidateSpec(
                                seq=seq,
                                child_table=child.name,
                                child_cols=tuple(child_cols),
                                parent_table=parent.name,
                                parent_cols=tuple(parent_cols),
                                mapping_mode=params.mapping,
                                effective=tuple(effective),
                                logical_child=tuple(logical_child),
                                logical_parent=tuple(logical_parent),
                            )
                        )
    return specs


def _mapping_error(spec: CandidateSpec) -> str | None:
    for pos, (ct, pt) in enumerate(zip(spec.logical_child, spec.logical_parent)):
        res: MappingResolution = resolve_mapping(ct, pt, spec.mapping_mode)
        if not res.ok:
            return f"第 {pos + 1} 列：{res.reason}"
        if spec.effective[pos] == "!":
            return f"第 {pos + 1} 列：映射无法解析"
    return None


def _has_null(key: Key) -> bool:
    return any(part[0] == "null" for part in key)


def _select_keys(
    child: DomainStats, parent: DomainStats, policy: str
) -> tuple[set[Key], set[Key], str]:
    """Apply the declared null rule once, explicitly.

    skip_child_nulls: child rows whose key contains NULL are skipped; parent
        keys containing NULL are not referenceable.
    fail_child_nulls: those child rows are violations (they can never match a
        clean parent key).
    match_nulls: NULL = NULL for partially-null keys; rows where every key
        column is NULL are not keys and are skipped on both sides.
    """
    if policy == "match_nulls":
        return child.matchable_keys, parent.matchable_keys, "空值等于空值；整键为空的行不计为键"
    if policy == "fail_child_nulls":
        return set(child.keys), parent.clean_keys, "子表含空键的行计为违规"
    return child.clean_keys, parent.clean_keys, "子表含空键的行跳过，父表空键不参与匹配"


def _single_column_containment(
    spec: CandidateSpec, child_keys: set[Key], parent_keys: set[Key]
) -> dict[str, Any]:
    """Necessary-condition check over the exact sets being compared.

    The projection is taken from the composite key sets *after* the null rule
    has been applied, so a failure can only come from a value that is really
    part of a compared composite key. This keeps the prune sound: if a
    projection is not contained, no tuple containing it can be contained.

    A passing projection is NOT evidence for the pair and never removes the
    pair from verification.
    """
    pairs: list[dict[str, Any]] = []
    both = True
    for pos in range(len(spec.child_cols)):
        child_proj = {key[pos] for key in child_keys}
        parent_proj = {key[pos] for key in parent_keys}
        missing = child_proj - parent_proj
        holds = not missing and bool(child_proj)
        pairs.append(
            {
                "child_col": spec.child_cols[pos],
                "parent_col": spec.parent_cols[pos],
                "holds": holds,
                "child_keys": len(child_proj),
                "parent_keys": len(parent_proj),
                "missing_keys": len(missing),
                "missing_sample": sorted(
                    (_jsonable(part[1]) for part in missing), key=str
                )[:3],
            }
        )
        both = both and holds
    return {"pairs": pairs, "both_hold": both}


def _range_miss(child: DomainStats, parent: DomainStats, child_keys: set[Key]) -> str | None:
    if not (child.ordered and parent.ordered):
        return None
    if not child.clean_min or not parent.clean_min:
        return None
    if len(child.clean_min) != len(parent.clean_min):
        return None
    # Only clean keys are range-checked; null-bearing keys make ordering
    # ill-defined and are filtered by policy before verification.
    if len(child_keys) != len(child.clean_keys):
        return None
    for pos in range(len(child.clean_min)):
        cmin, cmax = child.clean_min[pos], child.clean_max[pos]
        pmin, pmax = parent.clean_min[pos], parent.clean_max[pos]
        try:
            if cmin < pmin or cmax > pmax:
                return (
                    f"第 {pos + 1} 列：子表范围 [{_jsonable(cmin)}, {_jsonable(cmax)}] "
                    f"超出父表 [{_jsonable(pmin)}, {_jsonable(pmax)}]"
                )
        except TypeError:
            return None
    return None


@dataclass
class Classification:
    spec: CandidateSpec
    state: str  # "eligible" | PRUNED
    reason: str
    stats: dict[str, Any]
    child: DomainStats | None = None
    parent: DomainStats | None = None
    child_keys: set[Key] | None = None
    parent_keys: set[Key] | None = None


def classify_candidate(spec: CandidateSpec, cache: DomainCache, params: SearchParams) -> Classification:
    """Cheap deterministic checks. No counterexample scan happens here."""
    mapping_error = _mapping_error(spec)
    if mapping_error:
        return Classification(spec, PRUNED, "type_incompatible", {"mapping_error": mapping_error})
    effective = spec.effective
    try:
        child = cache.stats(spec.child_table, spec.child_cols, effective)
        parent = cache.stats(spec.parent_table, spec.parent_cols, effective)
    except Exception as exc:
        return Classification(spec, PRUNED, "mapping", {"error": str(exc)})

    base: dict[str, Any] = {
        "null_policy": params.null_policy,
        "mapping": spec.mapping_mode,
        "mapping_effective": [describe_mapping(m) for m in effective],
        "child_rows": child.row_count,
        "parent_rows": parent.row_count,
        "child_distinct_keys": child.cardinality,
        "parent_distinct_keys": parent.cardinality,
        "parent_duplicate_keys": sum(1 for n in parent.row_counts.values() if n > 1),
        "parent_rows_collapsed": parent.row_count - parent.cardinality,
        "trivial_domain": any(t == "boolean" for t in spec.logical_child)
        or any(t == "boolean" for t in spec.logical_parent),
    }

    if child.errors or parent.errors:
        errors = child.errors or parent.errors
        base["cast_error_count"] = len(errors)
        base["cast_error_sample"] = errors[:5]
        return Classification(spec, PRUNED, "cast_errors", base, child, parent)

    if child.empty:
        return Classification(spec, PRUNED, "child_empty", base, child, parent)
    if parent.empty:
        return Classification(spec, PRUNED, "parent_empty", base, child, parent)

    child_keys, parent_keys, null_note = _select_keys(child, parent, params.null_policy)
    base["null_handling"] = null_note
    base["child_null_key_rows"] = sum(n for k, n in child.row_counts.items() if _has_null(k))

    if not child.keys or child.every_key_null:
        return Classification(spec, PRUNED, "child_all_null", base, child, parent)
    if not parent.keys or parent.every_key_null:
        return Classification(spec, PRUNED, "parent_all_null", base, child, parent)
    if not child_keys:
        return Classification(spec, PRUNED, "child_no_eligible_rows", base, child, parent)
    if not parent_keys:
        return Classification(spec, PRUNED, "parent_no_eligible_rows", base, child, parent)

    strict_child = {k for k in child_keys if not _has_null(k)}
    strict_parent = {k for k in parent_keys if not _has_null(k)}
    if spec.width == 2:
        single = _single_column_containment(spec, strict_child, strict_parent)
        base["single_column"] = single
        # A null-bearing row that the policy counts as a violation must not
        # prune the candidate: it is exactly what verification has to report.
        if strict_child and not single["both_hold"]:
            return Classification(spec, PRUNED, "projection", base, child, parent)
    else:
        base["single_column"] = None

    # Cardinality and range are properties of the compared key sets, so the
    # prune is exact. Null-bearing keys are excluded from these counts: under
    # fail_child_nulls they are policy violations, not evidence that the
    # reference is structurally impossible, and the candidate must still be
    # verified so those rows can be listed.
    if len(strict_child) > len(parent_keys):
        base["child_eligible_keys"] = len(child_keys)
        base["parent_eligible_keys"] = len(parent_keys)
        base["strict_child_keys"] = len(strict_child)
        return Classification(spec, PRUNED, "cardinality", base, child, parent)

    range_reason = _range_miss(child, parent, strict_child)
    if range_reason:
        base["range_detail"] = range_reason
        return Classification(spec, PRUNED, "range", base, child, parent)

    base["child_eligible_keys"] = len(child_keys)
    base["child_eligible_rows"] = sum(child.row_counts.get(k, 0) for k in child_keys)
    base["parent_eligible_keys"] = len(parent_keys)
    return Classification(spec, "eligible", "eligible", base, child, parent, child_keys, parent_keys)


def _sample_rows(
    stats: DomainStats, keys: Iterable[Key], limit: int, with_rows: bool = True
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in keys:
        if len(out) >= limit:
            break
        item: dict[str, Any] = {
            "values": [_jsonable(v) for v in stats.sample[key]],
            "key": [_jsonable(part[1]) for part in key],
        }
        if with_rows:
            item["row"] = stats.first_row[key]
            item["occurrences"] = stats.row_counts.get(key, 0)
        out.append(item)
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def verify_candidate(
    spec: CandidateSpec,
    classification: Classification,
    params: SearchParams,
) -> CandidateResult:
    """Exact check of the full canonical key set. Cheap prune has already run."""
    started = time.perf_counter()
    child = classification.child
    parent = classification.parent
    child_keys = classification.child_keys or set()
    parent_keys = classification.parent_keys or set()
    base = dict(classification.stats)

    missing = [k for k in child_keys if k not in parent_keys]
    missing.sort(key=lambda k: (child.first_row.get(k, 0), k) if child else (0, k))
    missing_set = set(missing)
    covered = len(child_keys) - len(missing)
    child_rows_eligible = base.get("child_eligible_rows", 0)
    covered_rows = sum(
        child.row_counts.get(k, 0) for k in child_keys if k not in missing_set
    ) if child else 0
    null_violation_rows = sum(
        n for k, n in child.row_counts.items() if _has_null(k) and k in missing_set
    ) if child else 0
    base.update(
        {
            "covered_keys": covered,
            "key_coverage": round(covered / len(child_keys), 6) if child_keys else 0.0,
            "covered_rows": covered_rows,
            "row_coverage": round(covered_rows / child_rows_eligible, 6) if child_rows_eligible else 0.0,
            "missing_keys": len(missing),
            "missing_rows": child_rows_eligible - covered_rows,
            "null_violation_rows": null_violation_rows,
            "covered_sample": _sample_rows(
                child, [k for k in child_keys if k not in missing_set], params.sample_size
            ) if child else [],
        }
    )
    if not missing:
        base["reason_label"] = REASON_LABELS["ok"]
        return CandidateResult(spec, CONTAINED, "ok", base, [], _elapsed(started))

    cex: list[dict[str, Any]] = []
    if child:
        for key in missing[: params.max_counterexamples_store]:
            cex.append(
                {
                    "row": child.first_row[key],
                    "values": [_jsonable(v) for v in child.sample[key]],
                    "occurrences": child.row_counts.get(key, 0),
                    "null_key": _has_null(key),
                }
            )
    base["counterexamples_stored"] = len(cex)
    base["counterexamples_truncated"] = len(missing) > len(cex)
    base["reason_label"] = REASON_LABELS["counterexamples"]
    return CandidateResult(spec, PARTIAL, "counterexamples", base, cex, _elapsed(started))


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def run_search(
    tables: Sequence[Table],
    params: SearchParams,
    *,
    completed_keys: set[str] | None = None,
    on_result: Callable[[CandidateResult], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Classify every candidate, then verify the eligible ones within budget.

    ``completed_keys`` enables resumption: verified candidates already stored
    are skipped and counted as already verified, never silently recomputed nor
    silently dropped from the reported scope.
    """
    params.validate()
    started = clock()
    specs = enumerate_specs(tables, params)
    cache = DomainCache(tables)
    completed = set(completed_keys or ())

    pruned_by_reason: dict[str, int] = {}
    pruned_examples: dict[str, list[dict[str, Any]]] = {}
    eligible: list[Classification] = []
    pruned_example_count = 0
    for spec in specs:
        cls = classify_candidate(spec, cache, params)
        if cls.state == PRUNED:
            pruned_by_reason[cls.reason] = pruned_by_reason.get(cls.reason, 0) + 1
            if pruned_example_count < params.max_pruned_examples:
                pruned_examples.setdefault(cls.reason, []).append(_pruned_example(cls))
                pruned_example_count += 1
            continue
        eligible.append(cls)

    eligible = _interleave_classifications(eligible)
    eligible_keys = {c.spec.key for c in eligible}
    already_done = len(eligible_keys & completed)

    results: list[CandidateResult] = []
    verified_new = 0
    skipped = 0
    interrupted = False
    deadline = started + params.max_seconds
    for cls in eligible:
        if cls.spec.key in completed:
            continue
        if should_stop and should_stop():
            interrupted = True
            skipped += 1
            continue
        if verified_new >= params.max_verified or clock() >= deadline:
            skipped += 1
            continue
        res = verify_candidate(cls.spec, cls, params)
        verified_new += 1
        results.append(res)
        if on_result:
            on_result(res)

    verified_total = already_done + verified_new
    unverified = max(0, len(eligible) - verified_total)
    scope: dict[str, Any] = {
        "enumerated": len(specs),
        "enumerated_by_width": _width_counts(specs),
        "eligible": len(eligible),
        "eligible_by_width": _width_counts_class(eligible),
        "pruned": len(specs) - len(eligible),
        "pruned_by_reason": pruned_by_reason,
        "pruned_examples": pruned_examples,
        "already_verified": already_done,
        "verified_this_pass": verified_new,
        "verified": verified_total,
        "unverified": unverified,
        "budget_skipped_this_pass": skipped,
        "budget_exhausted": skipped > 0 and not interrupted,
        "interrupted": interrupted,
        "elapsed_seconds": round(clock() - started, 3),
        "results_stored_this_pass": len(results),
        "params": params.to_dict(),
        "note": "",
    }
    if interrupted:
        scope["note"] = (
            f"运行被中断：本次新核验 {verified_new} 个候选，"
            f"仍有 {unverified} 个可核验候选未处理；已完成结果已保存，可继续。"
        )
    elif skipped:
        scope["note"] = (
            f"预算用尽：可核验候选共 {len(eligible)} 个，已核验 {verified_total} 个，"
            f"剩余 {unverified} 个未进入核验。本次结论仅代表该范围。"
        )
    return {
        "results": results,
        "scope": scope,
        "eligible_keys": [c.spec.key for c in eligible],
    }


def _pruned_example(cls: Classification) -> dict[str, Any]:
    return {
        "candidate_key": cls.spec.key,
        "reason": cls.reason,
        "reason_label": REASON_LABELS.get(cls.reason, cls.reason),
        "stats": cls.stats,
    }


def _interleave_classifications(items: Sequence[Classification]) -> list[Classification]:
    groups: dict[int, list[Classification]] = {}
    for item in items:
        groups.setdefault(item.spec.width, []).append(item)
    ordered = [groups[w] for w in sorted(groups)]
    out: list[Classification] = []
    for bundle in zip_longest(*ordered, fillvalue=None):
        for item in bundle:
            if item is not None:
                out.append(item)
    return out


def _width_counts(specs: Sequence[CandidateSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        counts[str(spec.width)] = counts.get(str(spec.width), 0) + 1
    return counts


def _width_counts_class(items: Sequence[Classification]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.spec.width)
        counts[key] = counts.get(key, 0) + 1
    return counts
