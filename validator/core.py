"""Independent reference-integrity validator.

This module is intentionally standalone: it does not import the discovery
package or its database. It accepts a relations manifest (JSON) that lists only
*confirmed* relations, plus a set of files, and returns a pass/fail verdict.

Contract:

* Only relations explicitly marked ``confirmed`` in the manifest are enforced.
  An excluded or absent candidate is never turned into a constraint.
* A relation whose child columns are missing from the new file is reported as
  ``not_applicable`` (the file does not carry that reference) rather than
  silently passing.
* A relation whose *parent* table is missing or empty while the child has
  eligible rows is a hard failure: there is nowhere for the keys to resolve.
* Any child row whose key is not in the parent key set is an orphan. Orphans
  are listed with row numbers and key values, capped for display but counted in
  full.
* Nulls follow the relation's declared null policy, recorded at confirmation
  time; the validator never invents its own rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence
import csv
import io
import json
import os
import re

NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a", "#n/a", "nil", "-"}


class ManifestError(ValueError):
    pass


@dataclass
class FileTable:
    name: str
    headers: list[str]
    rows: list[list[str]]  # raw strings, exact
    types: dict[str, str] = field(default_factory=dict)

    def col(self, name: str) -> int:
        try:
            return self.headers.index(name)
        except ValueError:
            raise KeyError(name) from None

    def logical_type(self, name: str) -> str:
        if name in self.types:
            return self.types[name]
        idx = self.col(name)
        inferred = infer_type([row[idx] if idx < len(row) else "" for row in self.rows])
        self.types[name] = inferred
        return inferred


def load_file(path: str, name: str) -> FileTable:
    ext = Path(path).suffix.lower()
    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            try:
                headers = [h.strip() for h in next(reader)]
            except StopIteration:
                headers = []
            rows = [r for r in reader if not (len(r) == 1 and not r[0].strip())]
        return FileTable(name=name, headers=headers, rows=rows)
    if ext in (".jsonl", ".ndjson"):
        records = []
        with open(path, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        headers: list[str] = []
        for rec in records:
            for k in rec:
                if k not in headers:
                    headers.append(k)
        rows = [[_to_text(rec.get(h)) for h in headers] for rec in records]
        return FileTable(name=name, headers=headers, rows=rows)
    if ext in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError:
            raise ValueError("openpyxl is required for Excel files") from None
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.active
            it = ws.iter_rows(values_only=True)
            try:
                headers = [str(h).strip() if h is not None else f"column_{i+1}" for i, h in enumerate(next(it))]
            except StopIteration:
                headers = []
            rows = [[_to_text(v) for v in row] for row in it]
        finally:
            wb.close()
        return FileTable(name=name, headers=headers, rows=rows)
    raise ValueError(f"unsupported file type {ext!r}")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def is_null(text: str) -> bool:
    return text.strip().lower() in NULL_TOKENS


class CastError(ValueError):
    pass


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
_BOOL_TRUE = {"true", "t", "yes", "y"}
_BOOL_FALSE = {"false", "f", "no", "n"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)


def infer_type(texts: list[str]) -> str:
    """Infer one logical type from raw cells, mirroring the discovery loader.

    Returned values are "integer" | "float" | "boolean" | "date" |
    "datetime" | "string". Mixed columns become "string" rather than being
    partially coerced.
    """
    present = [t.strip() for t in texts if not is_null(t)]
    if not present:
        return "string"
    if all(_INT_RE.match(t) and not _has_leading_zero(t) for t in present):
        return "integer"
    if all(_FLOAT_RE.match(t) for t in present):
        return "float"
    if all(t.lower() in _BOOL_TRUE | _BOOL_FALSE for t in present):
        return "boolean"
    if all(_DATE_RE.match(t) for t in present) and all(_valid_date(t) for t in present):
        return "date"
    if all(_DATETIME_RE.match(t) for t in present) and all(_valid_datetime(t) for t in present):
        return "datetime"
    return "string"


def _has_leading_zero(text: str) -> bool:
    signless = text.lstrip("+-")
    return len(signless) > 1 and signless[0] == "0" and signless[1].isdigit()


def _valid_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
        return True
    except ValueError:
        return False


def _valid_datetime(text: str) -> bool:
    try:
        _parse_datetime(text)
        return True
    except ValueError:
        return False


def _parse_datetime(text: str) -> datetime:
    cleaned = text.strip().replace("Z", "+00:00")
    if len(cleaned) == 10:
        return datetime.combine(date.fromisoformat(cleaned), datetime.min.time())
    return datetime.fromisoformat(cleaned.replace(" ", "T"))


def canonical(text: str, effective: str, logical: str) -> tuple[str, Any]:
    """Canonical key component for one raw cell.

    Mirrors the encoding the relation was confirmed under: the manifest
    records both the effective comparison domain and the column's logical
    type, so the gate compares exactly what discovery compared. If a value
    cannot be represented, CastError is raised and the relation fails loudly
    instead of silently comparing unequal representations.
    """
    if is_null(text):
        return ("null", None)
    raw = text.strip()
    if effective == "string":
        return ("str", raw)
    if effective == "number":
        try:
            number = Decimal(raw)
        except Exception:
            raise CastError(f"无法将 {text!r} 转为数值") from None
        if not number.is_finite():
            raise CastError(f"非有限数值 {text!r}")
        return ("num", number)
    if effective == "datetime":
        try:
            return ("dt", _parse_datetime(raw))
        except ValueError:
            raise CastError(f"无法将 {text!r} 转为日期时间") from None
    if effective == "boolean":
        low = raw.lower()
        if low in _BOOL_TRUE or low == "1":
            return ("bool", True)
        if low in _BOOL_FALSE or low == "0":
            return ("bool", False)
        raise CastError(f"无法将 {text!r} 转为布尔值")
    # identity ("none"): compare the value in its own declared logical type
    if logical == "boolean":
        low = raw.lower()
        if low in _BOOL_TRUE:
            return ("bool", True)
        if low in _BOOL_FALSE:
            return ("bool", False)
        raise CastError(f"该列声明为布尔值，但 {text!r} 不是")
    if logical == "integer":
        if not _INT_RE.match(raw):
            raise CastError(f"该列声明为整数，但 {text!r} 不是")
        return ("num", Decimal(int(raw)))
    if logical == "float":
        try:
            number = Decimal(raw)
        except Exception:
            raise CastError(f"该列声明为小数，但 {text!r} 不是") from None
        if not number.is_finite():
            raise CastError(f"非有限数值 {text!r}")
        return ("num", number)
    if logical == "date":
        try:
            return ("date", date.fromisoformat(raw))
        except ValueError:
            raise CastError(f"该列声明为日期，但 {text!r} 不是") from None
    if logical == "datetime":
        try:
            return ("dt", _parse_datetime(raw))
        except ValueError:
            raise CastError(f"该列声明为日期时间，但 {text!r} 不是") from None
    return ("str", raw)


@dataclass
class RelationOutcome:
    candidate_key: str
    status: str  # "pass" | "fail" | "not_applicable" | "error"
    reason: str
    kind: str = ""  # on failure: "orphans" | "structure" | "cast" | "policy"
    child_table: str = ""
    child_cols: list[str] = field(default_factory=list)
    parent_table: str = ""
    parent_cols: list[str] = field(default_factory=list)
    effective: list[str] = field(default_factory=list)
    null_policy: str = ""
    child_logical: list[str] = field(default_factory=list)
    parent_logical: list[str] = field(default_factory=list)
    child_rows: int = 0
    eligible_rows: int = 0
    null_rows: int = 0
    orphan_rows: int = 0
    orphan_keys: int = 0
    sample: list[dict[str, Any]] = field(default_factory=list)
    sample_truncated: bool = False
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def _validate_relation(
    rel: dict[str, Any], tables: dict[str, FileTable], sample_limit: int
) -> RelationOutcome:
    key = rel.get("candidate_key") or _manual_key(rel)
    outcome = RelationOutcome(candidate_key=key, status="error", reason="")
    # Carry the relation identity into the outcome: the batch-to-batch diff
    # needs the same comparison domain that this outcome was computed under.
    outcome.child_table = rel.get("child_table", "")
    outcome.child_cols = list(rel.get("child_cols") or [])
    outcome.parent_table = rel.get("parent_table", "")
    outcome.parent_cols = list(rel.get("parent_cols") or [])
    outcome.effective = list(rel.get("effective") or [])
    if len(outcome.effective) != len(outcome.child_cols):
        outcome.effective = [rel.get("mapping_effective", "none")] * len(outcome.child_cols)
    outcome.null_policy = rel.get("null_policy", "skip_child_nulls")
    outcome.child_logical = list(rel.get("child_logical") or [])
    outcome.parent_logical = list(rel.get("parent_logical") or [])
    for required in ("child_table", "child_cols", "parent_table", "parent_cols"):
        if required not in rel:
            outcome.reason = f"关系缺少字段 {required}"
            outcome.kind = "structure"
            return outcome
    if rel.get("status", "confirmed") != "confirmed":
        outcome.status = "not_applicable"
        outcome.reason = "关系未确认，不作为约束"
        return outcome
    child = tables.get(rel["child_table"])
    parent = tables.get(rel["parent_table"])
    if child is None:
        outcome.status = "not_applicable"
        outcome.reason = f"本批次没有子表 {rel['child_table']}"
        return outcome
    missing_cols = [c for c in rel["child_cols"] if c not in child.headers]
    if missing_cols:
        outcome.status = "not_applicable"
        outcome.reason = f"子表 {child.name} 缺少列 {', '.join(missing_cols)}，本批次不适用"
        return outcome
    if parent is None:
        outcome.status = "fail"
        outcome.kind = "structure"
        outcome.reason = f"父表 {rel['parent_table']} 缺失，子表键无处解析"
        outcome.child_rows = len(child.rows)
        return outcome
    missing_parent_cols = [c for c in rel["parent_cols"] if c not in parent.headers]
    if missing_parent_cols:
        outcome.status = "fail"
        outcome.kind = "structure"
        outcome.reason = f"父表 {parent.name} 缺少列 {', '.join(missing_parent_cols)}"
        return outcome

    mapping_modes = rel.get("effective") or []
    if len(mapping_modes) != len(rel["child_cols"]):
        mapping_modes = [rel.get("mapping_effective", "none")] * len(rel["child_cols"])
    child_logical = rel.get("child_logical") or []
    parent_logical = rel.get("parent_logical") or []
    if len(child_logical) != len(rel["child_cols"]):
        child_logical = [""] * len(rel["child_cols"])
    if len(parent_logical) != len(rel["parent_cols"]):
        parent_logical = [""] * len(rel["parent_cols"])
    warnings: list[str] = []
    for cols, declared_types, table in (
        (rel["child_cols"], child_logical, child),
        (rel["parent_cols"], parent_logical, parent),
    ):
        for col, declared in zip(cols, declared_types):
            actual = table.logical_type(col)
            if declared and declared != actual:
                warnings.append(
                    f"{table.name}.{col} 本批解析为 {actual}，确认时为 {declared}"
                )
    null_policy = rel.get("null_policy", "skip_child_nulls")
    if null_policy not in ("skip_child_nulls", "fail_child_nulls", "match_nulls"):
        outcome.reason = f"未知空值规则 {null_policy!r}"
        outcome.kind = "policy"
        return outcome

    child_idx = [child.col(c) for c in rel["child_cols"]]
    parent_idx = [parent.col(c) for c in rel["parent_cols"]]

    parent_keys: set[tuple[tuple[str, Any], ...]] = set()
    for row in parent.rows:
        parts = []
        try:
            for i, mode, logical in zip(parent_idx, mapping_modes, parent_logical):
                parts.append(canonical(row[i] if i < len(row) else "", mode, logical))
        except CastError as exc:
            outcome.status = "error"
            outcome.kind = "cast"
            outcome.reason = f"父表 {parent.name} 存在无法转换的值：{exc}"
            return outcome
        kt = tuple(parts)
        if null_policy == "match_nulls":
            if all(p[0] == "null" for p in kt):
                continue
        elif any(p[0] == "null" for p in kt):
            continue
        parent_keys.add(kt)
    parent_has_keys = bool(parent_keys)

    seen_missing: dict[tuple, int] = {}
    first_row: dict[tuple, int] = {}
    occurrences: dict[tuple, int] = {}
    for row_no, row in enumerate(child.rows, start=1):
        outcome.child_rows += 1
        try:
            parts = tuple(
                canonical(row[i] if i < len(row) else "", mode, logical)
                for i, mode, logical in zip(child_idx, mapping_modes, child_logical)
            )
        except CastError as exc:
            outcome.status = "error"
            outcome.kind = "cast"
            outcome.reason = f"子表 {child.name} 存在无法转换的值：{exc}"
            return outcome
        has_null = any(p[0] == "null" for p in parts)
        all_null = all(p[0] == "null" for p in parts)
        if null_policy == "skip_child_nulls" and has_null:
            outcome.null_rows += 1
            continue
        if null_policy == "match_nulls" and all_null:
            outcome.null_rows += 1
            continue
        outcome.eligible_rows += 1
        if parts in parent_keys:
            continue
        occurrences[parts] = occurrences.get(parts, 0) + 1
        first_row.setdefault(parts, row_no)
        if parts not in seen_missing and len(seen_missing) < sample_limit:
            seen_missing[parts] = row_no

    outcome.warnings = warnings
    if not parent_has_keys:
        if outcome.eligible_rows == 0:
            outcome.status = "pass"
            outcome.reason = (
                f"父表 {parent.name} 没有可用键，但子表也没有需要解析的键"
            )
            return outcome
        outcome.status = "fail"
        outcome.kind = "structure"
        outcome.reason = f"父表 {parent.name} 没有可用键，无法满足任何引用"
        return outcome
    outcome.orphan_rows = sum(occurrences.values())
    outcome.orphan_keys = len(occurrences)
    if outcome.orphan_rows:
        outcome.status = "fail"
        outcome.kind = "orphans"
        outcome.reason = f"{outcome.orphan_rows} 行 / {outcome.orphan_keys} 个键在父表中不存在"
        for parts, row_no in seen_missing.items():
            raw = [
                child.rows[row_no - 1][i] if i < len(child.rows[row_no - 1]) else ""
                for i in child_idx
            ]
            outcome.sample.append(
                {
                    "row": row_no,
                    "values": raw,
                    "occurrences": occurrences[parts],
                }
            )
        outcome.sample_truncated = outcome.orphan_keys > len(outcome.sample)
    else:
        outcome.status = "pass"
        outcome.reason = "全部子表键都能在父表找到"
    return outcome


def _manual_key(rel: dict[str, Any]) -> str:
    return (
        f"{rel.get('child_table')}[{','.join(rel.get('child_cols', []))}] -> "
        f"{rel.get('parent_table')}[{','.join(rel.get('parent_cols', []))}]"
    )


def validate(
    relations: Sequence[dict[str, Any]],
    files: dict[str, str],
    *,
    sample_limit: int = 20,
    publication_label: str = "",
) -> dict[str, Any]:
    """Validate confirmed relations against a new batch of files.

    Returns ``ok=False`` if any confirmed relation fails, which the calling
    pipeline must treat as a refusal to publish.
    """
    tables = {name: load_file(path, name) for name, path in files.items()}
    outcomes: list[RelationOutcome] = []
    for rel in relations:
        outcomes.append(_validate_relation(rel, tables, sample_limit))
    failures = [o for o in outcomes if o.status in ("fail", "error")]
    result = {
        "ok": not failures,
        "publication_allowed": not failures,
        "publication_label": publication_label,
        "tables_checked": sorted(tables.keys()),
        "relation_count": len(outcomes),
        "enforced_count": sum(1 for o in outcomes if o.status in ("pass", "fail")),
        "not_applicable_count": sum(1 for o in outcomes if o.status == "not_applicable"),
        "failures": len(failures),
        "orphan_relation_count": sum(1 for o in failures if o.kind == "orphans"),
        "structure_failure_count": sum(1 for o in failures if o.kind in ("structure", "policy")),
        "cast_failure_count": sum(1 for o in failures if o.kind == "cast"),
        "orphan_rows": sum(o.orphan_rows for o in outcomes),
        "relations": [_outcome_dict(o) for o in outcomes],
    }
    if failures:
        parts = []
        if result["orphan_relation_count"]:
            parts.append(
                f"{result['orphan_relation_count']} 条已确认引用的子表键在父表中缺失"
                f"（共 {result['orphan_rows']} 行）"
            )
        if result["structure_failure_count"]:
            parts.append(f"{result['structure_failure_count']} 条关系无法结构校验（缺表/缺列）")
        if result["cast_failure_count"]:
            parts.append(f"{result['cast_failure_count']} 条关系存在无法转换的值")
        result["refusal_reason"] = "；".join(parts) + "，拒绝发布本批次"
    return result


def _outcome_dict(o: RelationOutcome) -> dict[str, Any]:
    return {
        "candidate_key": o.candidate_key,
        "status": o.status,
        "kind": o.kind,
        "reason": o.reason,
        "child_table": o.child_table,
        "child_cols": o.child_cols,
        "parent_table": o.parent_table,
        "parent_cols": o.parent_cols,
        "effective": o.effective,
        "null_policy": o.null_policy,
        "child_logical": o.child_logical,
        "parent_logical": o.parent_logical,
        "child_rows": o.child_rows,
        "eligible_rows": o.eligible_rows,
        "null_rows": o.null_rows,
        "orphan_rows": o.orphan_rows,
        "orphan_keys": o.orphan_keys,
        "sample": o.sample,
        "sample_truncated": o.sample_truncated,
        "detail": o.detail,
        "warnings": o.warnings,
    }


def load_manifest(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("relations", [])
    if not isinstance(data, list):
        raise ManifestError("manifest 必须是关系数组或包含 relations 的对象")
    for i, rel in enumerate(data):
        if not isinstance(rel, dict):
            raise ManifestError(f"第 {i + 1} 条关系不是对象")
        if rel.get("status", "confirmed") != "confirmed":
            continue
        for required in ("child_table", "child_cols", "parent_table", "parent_cols"):
            if required not in rel:
                raise ManifestError(f"第 {i + 1} 条已确认关系缺少 {required}")
    return data
