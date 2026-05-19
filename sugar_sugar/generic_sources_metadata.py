from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GenericSourceMetadata:
    age: str
    gender: str
    weight: str
    sensor: str = ""


def _project_root() -> Path:
    # Repo layout: <root>/sugar_sugar/<this_file>.py
    return Path(__file__).resolve().parents[1]


def load_generic_sources_metadata() -> dict[str, GenericSourceMetadata]:
    path = _project_root() / "data" / "generic_sources_metadata.json"
    if not path.exists():
        return {}

    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}

    out: dict[str, GenericSourceMetadata] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        age = str(value.get("age") or "").strip()
        gender = str(value.get("gender") or "").strip()
        weight = str(value.get("weight") or "").strip()
        sensor = str(value.get("sensor") or "").strip()
        if not age or not gender or not weight:
            continue
        out[key] = GenericSourceMetadata(age=age, gender=gender, weight=weight, sensor=sensor)
    return out

