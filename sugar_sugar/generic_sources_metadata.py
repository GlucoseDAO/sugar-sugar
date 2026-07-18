from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sugar_sugar.subject_sources import GenericDatasetSource, discover_generic_dataset_sources


@dataclass(frozen=True, slots=True)
class GenericSourceMetadata:
    age: str
    gender: str
    weight: str
    sensor: str = ""


def _project_root() -> Path:
    # Repo layout: <root>/sugar_sugar/<this_file>.py
    return Path(__file__).resolve().parents[1]


def _metadata_from_source(source: GenericDatasetSource) -> GenericSourceMetadata | None:
    if source.age_years is None:
        return None
    age = f"{source.age_years} years old"
    return GenericSourceMetadata(
        age=age,
        gender=source.gender,
        weight=source.weight,
        sensor=source.sensor,
    )


def _metadata_from_json_entry(value: dict[str, Any]) -> GenericSourceMetadata | None:
    age = str(value.get("age") or "").strip()
    gender = str(value.get("gender") or "").strip()
    weight = str(value.get("weight") or "").strip()
    sensor = str(value.get("sensor") or "").strip()
    if not age:
        return None
    return GenericSourceMetadata(
        age=age,
        gender=gender,
        weight=weight,
        sensor=sensor,
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

    for source in discover_generic_dataset_sources():
        meta = _metadata_from_source(source)
        if meta is not None:
            out[source.source_name] = meta

    return out


def format_generic_source_metadata(
    meta: GenericSourceMetadata,
    *,
    locale: str,
    show_no_carbs_note: bool = False,
) -> str:
    from sugar_sugar.i18n import normalize_locale, t

    locale = normalize_locale(locale)
    gender_raw = str(meta.gender or "").strip().lower()
    if gender_raw in ("male", "female", "na"):
        gender_display = t(f"ui.startup.gender_{gender_raw}", locale=locale)
    else:
        gender_display = meta.gender

    age_display = (
        str(meta.age)
        .replace("years old", "")
        .replace("year old", "")
        .strip()
    )
    weight_display = str(meta.weight).replace(" ", "")

    note_parts: list[str] = []
    if show_no_carbs_note:
        note_parts.append(t("ui.header.no_carbs_note", locale=locale))

    if locale == "en" and age_display and gender_display and weight_display:
        line = f"{age_display} yr old {gender_display}, weight {weight_display}"
    else:
        detail_parts: list[str] = []
        if age_display:
            detail_parts.append(f"{t('ui.startup.age_label', locale=locale)}: {age_display}")
        if gender_display:
            detail_parts.append(f"{t('ui.startup.gender_label', locale=locale)}: {gender_display}")
        if weight_display:
            detail_parts.append(f"{t('ui.header.weight_label', locale=locale)}: {weight_display}")
        if meta.sensor:
            detail_parts.append(f"{t('ui.ending.sensor_label', locale=locale)}: {meta.sensor}")
        line = " · ".join(detail_parts)

    if note_parts:
        if line:
            return f"{line} · {note_parts[0]}"
        return note_parts[0]
    return line
