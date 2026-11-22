"""Type compatibility, explicit type mapping, and canonical key encoding.

Equality is exact. Canonicalisation never silently coerces an incompatible
value: if a cast cannot be applied the caller gets a CastError and the
candidate is reported as ``mapping_error`` instead of being verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

LOGICAL_TYPES = ("integer", "float", "boolean", "date", "datetime", "string")
NULL_POLICIES = ("skip_child_nulls", "fail_child_nulls", "match_nulls")
MAPPING_MODES = ("auto", "none", "string", "number", "datetime", "boolean")

NULL_POLICY_LABELS = {
    "skip_child_nulls": "子表空键行跳过（不计入覆盖，也不计为违规）",
    "fail_child_nulls": "子表空键行视为违规",
    "match_nulls": "空值等于空值（两侧 NULL 可互相匹配）",
}

MAPPING_LABELS = {
    "auto": "自动：同类型精确；整数/小数按数值；日期/时间按日期时间（不跨 string/数值）",
    "none": "严格：仅同类型精确比较",
    "string": "显式：两侧都转为字符串",
    "number": "显式：两侧都转为数值",
    "datetime": "显式：两侧都转为日期时间",
    "boolean": "显式：两侧都转为布尔值",
}


class CastError(ValueError):
    """A value cannot be represented in the requested comparison domain."""


@dataclass(frozen=True)
class MappingResolution:
    ok: bool
    effective: str  # "none" | "string" | "number" | "datetime" | "boolean"
    reason: str = ""


def _is_numeric(t: str) -> bool:
    return t in ("integer", "float")


def resolve_mapping(child_type: str, parent_type: str, mode: str) -> MappingResolution:
    """Resolve a user-requested mapping into the actual comparison domain.

    ``auto`` is intentionally conservative: it only widens within a family
    (integer -> float as numbers, date -> datetime) and never crosses
    string/number or boolean/number boundaries.
    """
    if mode not in MAPPING_MODES:
        return MappingResolution(False, "none", f"未知类型映射 {mode!r}")
    if mode == "auto":
        if child_type == parent_type:
            return MappingResolution(True, "none")
        if _is_numeric(child_type) and _is_numeric(parent_type):
            # Both sides are numeric columns, so numeric widening is exact.
            return MappingResolution(True, "number")
        if child_type in ("date", "datetime") and parent_type in ("date", "datetime"):
            return MappingResolution(True, "datetime")
        # Never cross string <-> number or boolean <-> number implicitly: a
        # textual "1" and numeric 1 are different values unless the user says
        # otherwise with an explicit mapping.
        return MappingResolution(
            False, "none", f"类型 {child_type} 与 {parent_type} 在自动映射下不兼容"
        )
    if mode == "none":
        if child_type == parent_type:
            return MappingResolution(True, "none")
        return MappingResolution(
            False, "none", f"严格映射要求类型相同，实际为 {child_type} / {parent_type}"
        )
    return MappingResolution(True, mode)


def _canonical_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            raise CastError("NaN 不能作为键")
        return repr(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _canonical_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "t", "yes", "y", "1"):
            return True
        if low in ("false", "f", "no", "n", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise CastError(f"无法将 {value!r} 转为布尔值")


def _canonical_number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise CastError("布尔值不参与数值映射")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CastError("非有限小数不能作为键")
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except (InvalidOperation, ValueError):
            raise CastError(f"无法将 {value!r} 转为数值") from None
        if not number.is_finite():
            raise CastError(f"非有限数值 {value!r}")
        return number
    raise CastError(f"无法将 {type(value).__name__} 转为数值")


def _canonical_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            if len(text) == 10:
                return datetime.combine(date.fromisoformat(text), time.min)
            return datetime.fromisoformat(text.replace(" ", "T"))
        except ValueError:
            raise CastError(f"无法将 {value!r} 转为日期时间") from None
    raise CastError(f"无法将 {type(value).__name__} 转为日期时间")


def canonical_key(value: Any, logical_type: str, effective: str) -> tuple[str, Any]:
    """Return a tagged, hashable key component.

    Tags prevent collisions between e.g. integer 1 and string "1". Decimal
    makes 1 and 1.0 compare equal in numeric mode without binary-float drift.
    """
    if value is None:
        return ("null", None)
    if effective == "string":
        return ("str", _canonical_string(value))
    if effective == "number":
        return ("num", _canonical_number(value))
    if effective == "datetime":
        return ("dt", _canonical_datetime(value))
    if effective == "boolean":
        return ("bool", _canonical_boolean(value))
    # identity: tag by the column's logical type, but use exact canonical value
    if logical_type == "boolean":
        if not isinstance(value, bool):
            raise CastError("该列声明为布尔值，但值不是布尔值")
        return ("bool", value)
    if logical_type == "integer":
        if isinstance(value, bool):
            raise CastError("布尔值不是整数键")
        return ("num", Decimal(str(value)))
    if logical_type == "float":
        return ("num", _canonical_number(value))
    if logical_type == "date":
        if not isinstance(value, date) or isinstance(value, datetime):
            raise CastError("该列声明为日期，但值不是日期")
        return ("date", value)
    if logical_type == "datetime":
        return ("dt", _canonical_datetime(value))
    return ("str", _canonical_string(value))


def describe_mapping(effective: str) -> str:
    return {
        "none": "同类型精确",
        "string": "字符串",
        "number": "数值",
        "datetime": "日期时间",
        "boolean": "布尔值",
    }.get(effective, effective)


def is_null_token(text: str) -> bool:
    """The single null-token rule shared by discovery and the ingress validator."""
    return text.strip().lower() in {
        "",
        "null",
        "none",
        "nan",
        "na",
        "n/a",
        "#n/a",
        "nil",
    }
