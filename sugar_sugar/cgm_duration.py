"""CGM usage duration as a (value, unit) pair.

Older sessions and CSVs stored a bare integer of years. New records store
``value,unit`` (e.g. ``6,months``). Years remain the comparison unit for
age checks and any analysis that still wants a single number.
"""
from __future__ import annotations

from typing import Any, Optional

CGM_DURATION_UNITS: tuple[str, ...] = ("weeks", "months", "years")
DEFAULT_CGM_DURATION_UNIT: str = "years"

_UNIT_TO_YEARS: dict[str, float] = {
    "weeks": 1.0 / 52.0,
    "months": 1.0 / 12.0,
    "years": 1.0,
}


def normalize_cgm_duration_unit(unit: Optional[str]) -> str:
    text = str(unit or DEFAULT_CGM_DURATION_UNIT).strip().lower()
    if text in {"week", "weeks"}:
        return "weeks"
    if text in {"month", "months"}:
        return "months"
    if text in {"year", "years"}:
        return "years"
    return DEFAULT_CGM_DURATION_UNIT


def cgm_duration_to_years(
    value: Optional[int | float],
    unit: Optional[str] = None,
) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value) * _UNIT_TO_YEARS[normalize_cgm_duration_unit(unit)]


def serialize_cgm_duration(
    value: Optional[int | float],
    unit: Optional[str] = None,
) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    if number.is_integer():
        rendered = str(int(number))
    else:
        rendered = f"{number:g}"
    return f"{rendered},{normalize_cgm_duration_unit(unit)}"


def parse_cgm_duration(raw: Any) -> tuple[Optional[float], str]:
    """Parse a stored duration into ``(value, unit)``.

    Accepts a number (legacy years), a ``value,unit`` string, a
    ``(value, unit)`` / ``[value, unit]`` pair, or empty.
    """
    if raw is None or raw == "":
        return None, DEFAULT_CGM_DURATION_UNIT
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return _as_number(raw[0]), normalize_cgm_duration_unit(str(raw[1]))
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), DEFAULT_CGM_DURATION_UNIT

    text = str(raw).strip()
    if not text:
        return None, DEFAULT_CGM_DURATION_UNIT
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if "," in text:
        left, right = text.split(",", 1)
        return _as_number(left), normalize_cgm_duration_unit(right)
    return _as_number(text), DEFAULT_CGM_DURATION_UNIT


def migrate_cgm_duration_cell(raw: Any) -> str:
    """Rewrite one CSV cell to ``value,unit``, treating bare numbers as years."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    value, unit = parse_cgm_duration(text)
    if value is None:
        return text
    return serialize_cgm_duration(value, unit)


def cgm_duration_csv_value(user_info: Optional[dict[str, Any]]) -> str:
    info = user_info or {}
    pair = info.get("cgm_duration")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return serialize_cgm_duration(pair[0], pair[1])
    years = info.get("cgm_duration_years")
    if years in (None, ""):
        return ""
    value, unit = parse_cgm_duration(years)
    if value is None:
        return ""
    return serialize_cgm_duration(value, unit)


def _as_number(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    return float(raw)
