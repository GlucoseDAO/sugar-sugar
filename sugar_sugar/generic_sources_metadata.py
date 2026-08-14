from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sugar_sugar.subject_sources import (
    GenericDatasetSource,
    discover_generic_dataset_sources,
    discover_legacy_generic_sources,
)


@dataclass(frozen=True, slots=True)
class GenericSourceMetadata:
    age: str
    gender: str
    weight: str
    sensor: str = ""
    diabetic: bool | None = None


def _project_root() -> Path:
    # Repo layout: <root>/sugar_sugar/<this_file>.py
    return Path(__file__).resolve().parents[1]


def _metadata_from_source(source: GenericDatasetSource) -> GenericSourceMetadata | None:
    if source.age_years is None and not source.gender and source.diabetic is None:
        return None
    age = f"{source.age_years} years old" if source.age_years is not None else ""
    return GenericSourceMetadata(
        age=age,
        gender=source.gender,
        weight=source.weight,
        sensor=source.sensor,
        diabetic=source.diabetic,
    )


def _parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "diabetic"}:
        return True
    if raw in {"0", "false", "no", "non-diabetic", "nondiabetic"}:
        return False
    return None


def _metadata_from_json_entry(value: dict[str, Any]) -> GenericSourceMetadata | None:
    age = str(value.get("age") or "").strip()
    gender = str(value.get("gender") or "").strip()
    weight = str(value.get("weight") or "").strip()
    sensor = str(value.get("sensor") or "").strip()
    diabetic = _parse_optional_bool(value.get("diabetic"))
    if not age and not gender and diabetic is None:
        return None
    return GenericSourceMetadata(
        age=age,
        gender=gender,
        weight=weight,
        sensor=sensor,
        diabetic=diabetic,
    )


def load_generic_sources_metadata() -> dict[str, GenericSourceMetadata]:
    path = _project_root() / "data" / "generic_sources_metadata.json"
    out: dict[str, GenericSourceMetadata] = {}

    if path.exists():
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key, value in raw.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                meta = _metadata_from_json_entry(value)
                if meta is not None:
                    out[key] = meta

    for source in discover_legacy_generic_sources() + discover_generic_dataset_sources():
        meta = _metadata_from_source(source)
        if meta is not None:
            out[source.source_name] = meta

    return out


def resolve_source_metadata(source_name: str) -> GenericSourceMetadata | None:
    """Look up demographics for a stored source name, including a live scan."""
    key = Path(str(source_name or "")).name
    if not key:
        return None
    cached = load_generic_sources_metadata().get(key)
    if cached is not None:
        return cached
    for source in discover_legacy_generic_sources() + discover_generic_dataset_sources():
        if source.source_name == key:
            return _metadata_from_source(source)
    return None


def _gender_display(gender: str, *, locale: str) -> str:
    from sugar_sugar.i18n import t

    key_map = {
        "m": "male",
        "f": "female",
        "n/a": "na",
        "male": "male",
        "female": "female",
        "na": "na",
    }
    raw = str(gender or "").strip().lower()
    normalized = key_map.get(raw, raw)
    if normalized in ("male", "female", "na"):
        return t(f"ui.startup.gender_{normalized}", locale=locale)
    return str(gender or "")


def _diabetes_status_label(diabetic: bool | None, *, locale: str) -> str:
    from sugar_sugar.i18n import t

    if diabetic is True:
        return t("ui.header.diabetic", locale=locale)
    if diabetic is False:
        return t("ui.header.non_diabetic", locale=locale)
    return ""


def _format_demographics_line(
    *,
    age_display: str,
    gender_display: str,
    weight_display: str,
    sensor: str,
    locale: str,
    diabetic: bool | None = None,
) -> str:
    from sugar_sugar.i18n import t

    status = _diabetes_status_label(diabetic, locale=locale)
    if locale == "en":
        if age_display and gender_display:
            details = f"{age_display} yr old {gender_display}"
            if weight_display:
                details = f"{details}, weight {weight_display}"
        else:
            bits: list[str] = []
            if age_display:
                bits.append(f"{age_display} yr old")
            if gender_display:
                bits.append(gender_display)
            if weight_display:
                bits.append(f"weight {weight_display}")
            details = ", ".join(bits)
        if status and details:
            return f"{status} · {details}"
        return status or details

    detail_parts: list[str] = []
    if status:
        detail_parts.append(status)
    if age_display:
        detail_parts.append(f"{t('ui.startup.age_label', locale=locale)}: {age_display}")
    if gender_display:
        detail_parts.append(f"{t('ui.startup.gender_label', locale=locale)}: {gender_display}")
    if weight_display:
        detail_parts.append(f"{t('ui.header.weight_label', locale=locale)}: {weight_display}")
    if sensor:
        detail_parts.append(f"{t('ui.ending.sensor_label', locale=locale)}: {sensor}")
    return " · ".join(detail_parts)


def _append_metadata_notes(
    line: str,
    *,
    locale: str,
    show_no_carbs_note: bool,
    show_carbs_info_note: bool,
) -> str:
    from sugar_sugar.i18n import t

    note_parts: list[str] = []
    if show_carbs_info_note:
        note_parts.append(t("ui.header.carbs_info_note", locale=locale))
    if show_no_carbs_note:
        note_parts.append(t("ui.header.no_carbs_note", locale=locale))
    if not note_parts:
        return line
    notes = " · ".join(note_parts)
    if line:
        return f"{line} · {notes}"
    return notes


def format_source_notes(
    *,
    locale: str,
    show_no_carbs_note: bool = False,
    show_carbs_info_note: bool = False,
) -> str:
    from sugar_sugar.i18n import normalize_locale

    return _append_metadata_notes(
        "",
        locale=normalize_locale(locale),
        show_no_carbs_note=show_no_carbs_note,
        show_carbs_info_note=show_carbs_info_note,
    )


def format_participant_demographics(
    age: int | float | str,
    gender: str,
    *,
    locale: str,
    weight: str = "",
    diabetic: bool | None = None,
    show_no_carbs_note: bool = False,
    show_carbs_info_note: bool = False,
) -> str:
    from sugar_sugar.i18n import normalize_locale

    locale = normalize_locale(locale)
    age_display = str(int(float(age))) if age not in (None, "") else ""
    gender_display = _gender_display(gender, locale=locale)
    weight_display = str(weight).replace(" ", "")
    line = _format_demographics_line(
        age_display=age_display,
        gender_display=gender_display,
        weight_display=weight_display,
        sensor="",
        locale=locale,
        diabetic=diabetic,
    )
    return _append_metadata_notes(
        line,
        locale=locale,
        show_no_carbs_note=show_no_carbs_note,
        show_carbs_info_note=show_carbs_info_note,
    )


def format_generic_source_metadata(
    meta: GenericSourceMetadata,
    *,
    locale: str,
    show_no_carbs_note: bool = False,
    show_carbs_info_note: bool = False,
) -> str:
    from sugar_sugar.i18n import normalize_locale

    locale = normalize_locale(locale)
    gender_display = _gender_display(meta.gender, locale=locale)
    age_display = (
        str(meta.age)
        .replace("years old", "")
        .replace("year old", "")
        .strip()
    )
    weight_display = str(meta.weight).replace(" ", "")
    line = _format_demographics_line(
        age_display=age_display,
        gender_display=gender_display,
        weight_display=weight_display,
        sensor=str(meta.sensor or ""),
        locale=locale,
        diabetic=meta.diabetic,
    )
    return _append_metadata_notes(
        line,
        locale=locale,
        show_no_carbs_note=show_no_carbs_note,
        show_carbs_info_note=show_carbs_info_note,
    )
