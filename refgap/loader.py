"""Load tabular files into typed, deduplicated value domains.

The loader is deliberately conservative: it returns plain Python values with a
declared logical type per column, so that equality is exact and testable
(1 and "1" are different values unless the caller opts into coercion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".jsonl", ".ndjson", ".xlsx", ".xlsm"}


class LoadError(Exception):
    """Raised when a file cannot be parsed into a table."""


@dataclass(frozen=True)
class Column:
    name: str
    index: int
    logical_type: str  # "integer" | "float" | "boolean" | "date" | "datetime" | "string"


@dataclass
class Table:
    name: str
    path: str
    columns: list[Column]
    rows: list[tuple[Any, ...]]
    source_hash: str
    warnings: list[str] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def values(self, name: str) -> list[Any]:
        idx = self.column(name).index
        return [row[idx] for row in self.rows]


_NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a", "#n/a", "nil", "-"}


def is_null(value: Any) -> bool:
    """A single, explicit null rule shared by discovery and validation."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULL_TOKENS
    return False


def stable_hash_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_header(raw: Sequence[Any]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, cell in enumerate(raw):
        name = str(cell).strip() if cell is not None else ""
        if not name:
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
_BOOL_TRUE = {"true", "t", "yes", "y"}
_BOOL_FALSE = {"false", "f", "no", "n"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _try_int(text: str) -> int | None:
    if len(text) > 1 and text[0] == "0" and text[1].isdigit():
        return None  # keep leading-zero codes as strings
    if len(text) > 2 and text[0] in "+-" and text[1] == "0" and text[2].isdigit():
        return None
    if _INT_RE.match(text) and abs(int(text)) < 2**63:
        return int(text)
    return None


def _try_float(text: str) -> float | None:
    # "001" is an identifier, not a number: keep zero-padded codes textual.
    if _INT_RE.match(text) and _try_int(text) is None:
        return None
    if _FLOAT_RE.match(text):
        try:
            val = float(text)
        except ValueError:
            return None
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return val
    return None


def _try_bool(text: str) -> bool | None:
    low = text.lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    return None


def _try_date(text: str) -> date | None:
    if not _DATE_RE.match(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _try_datetime(text: str) -> datetime | None:
    if not _DATETIME_RE.match(text):
        return None
    cleaned = text.replace("Z", "+00:00").replace(" ", "T")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _infer(raw_values: Iterable[Any], declared: str | None) -> tuple[str, list[Any], list[str]]:
    """Infer one logical type for a column and convert every value to it.

    Mixed columns fall back to string rather than partially coercing, so the
    resulting domain stays exact.
    """
    warnings: list[str] = []
    present = [v for v in raw_values if not is_null(v)]
    if not present:
        return (declared or "string"), [None for _ in raw_values], ["该列全部为空"]

    if declared and declared not in ("integer", "float", "boolean", "date", "datetime", "string"):
        warnings.append(f"已声明类型 {declared!r} 无法识别，改用推断")
        declared = None

    candidates: list[tuple[str, Any]] = [
        ("integer", _try_int),
        ("float", _try_float),
        ("boolean", _try_bool),
        ("date", _try_date),
        ("datetime", _try_datetime),
    ]
    if declared:
        order = [c for c in candidates if c[0] == declared] + [c for c in candidates if c[0] != declared]
    else:
        order = candidates

    for logical, fn in order:
        converted: list[Any] = []
        ok = True
        for v in raw_values:
            if is_null(v):
                converted.append(None)
                continue
            if not isinstance(v, str):
                ok = False
                break
            out = fn(v.strip())
            if out is None:
                ok = False
                break
            converted.append(out)
        if ok:
            return logical, converted, warnings

    out = [None if is_null(v) else str(v).strip() for v in raw_values]
    return "string", out, warnings


def _read_csv(path: str, delimiter: str) -> tuple[list[str], list[list[Any]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise LoadError("file is empty") from None
        # Only truly blank lines are noise. A line like ",," is a real row of
        # nulls and must be counted (it can be a null-key violation).
        rows = [
            row for row in reader
            if not (len(row) == 1 and not row[0].strip())
        ]
    return _normalize_header(header), rows


def _read_jsonl(path: str) -> tuple[list[str], list[list[Any]]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LoadError(f"invalid JSON line: {exc}") from None
            if not isinstance(obj, dict):
                raise LoadError("each JSONL line must be an object")
            records.append(obj)
    if not records:
        raise LoadError("file is empty")
    names: list[str] = []
    for rec in records:
        for key in rec:
            if key not in names:
                names.append(str(key))
    header = _normalize_header(names)
    rows = [[_json_scalar(rec.get(key)) for key in names] for rec in records]
    return header, rows


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _read_xlsx(path: str) -> tuple[list[str], list[list[Any]]]:
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise LoadError("openpyxl is required to read Excel files") from None
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        iterator = ws.iter_rows(values_only=True)
        try:
            header = next(iterator)
        except StopIteration:
            raise LoadError("workbook sheet is empty") from None
        rows = [[_json_scalar(cell) if not isinstance(cell, str) else cell for cell in row] for row in iterator]
    finally:
        wb.close()
    return _normalize_header(header), rows


def load_table(name: str, path: str, declared_types: dict[str, str] | None = None) -> Table:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoadError(f"unsupported file type {ext!r}")
    warnings: list[str] = []
    if ext in (".csv",):
        header, raw_rows = _read_csv(path, ",")
    elif ext == ".tsv":
        header, raw_rows = _read_csv(path, "\t")
    elif ext in (".jsonl", ".ndjson"):
        header, raw_rows = _read_jsonl(path)
    else:
        header, raw_rows = _read_xlsx(path)

    if not header:
        raise LoadError("no columns found")
    # drop trailing entirely-empty rows; keep width aligned
    width = len(header)
    normalized: list[list[Any]] = []
    for row in raw_rows:
        cells = list(row) + [None] * (width - len(row))
        normalized.append(cells[:width])

    columns: list[Column] = []
    converted_rows: list[list[Any]] = [[] for _ in normalized]
    for idx, col_name in enumerate(header):
        raw_col = [row[idx] for row in normalized]
        declared = (declared_types or {}).get(col_name)
        logical, converted, col_warnings = _infer(raw_col, declared)
        warnings.extend(f"{col_name}: {w}" for w in col_warnings)
        columns.append(Column(name=col_name, index=idx, logical_type=logical))
        for row_idx, value in enumerate(converted):
            converted_rows[row_idx].append(value)

    table = Table(
        name=name,
        path=str(Path(path).resolve()),
        columns=columns,
        rows=[tuple(r) for r in converted_rows],
        source_hash=stable_hash_file(path),
        warnings=warnings,
    )
    return table


def load_tables(specs: Sequence[dict[str, Any]]) -> list[Table]:
    return [load_table(s["name"], s["path"], s.get("types")) for s in specs]
