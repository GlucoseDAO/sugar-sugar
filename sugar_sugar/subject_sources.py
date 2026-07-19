from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_sugar.data import load_glucose_data, load_loop_chronological_data

_ADULT_MIN_AGE = 18
_AGE_AT_BASELINE_RE = re.compile(r"^ageAtBaseline:\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
_AGE_AT_ENROLLMENT_RE = re.compile(r"^AgeAtEnrollment:\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
_GENDER_RE = re.compile(r"^gender:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_WEIGHT_LB_RE = re.compile(r"^weight_lb:\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
_CGM_CODE_RE = re.compile(r"^what_cgm_code:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)

_CGM_LABELS: dict[str, str] = {
    "1": "Medtronic Guardian",
    "2": "Abbott FreeStyle Libre",
    "3": "Dexcom CGM",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subjects_dir() -> Path:
    return _project_root() / "data" / "subjects"


def _example_csv_path() -> Path:
    return _project_root() / "data" / "example.csv"


@dataclass(frozen=True, slots=True)
class GenericDatasetSource:
    source_name: str
    csv_path: Path
    age_years: int | None
    gender: str
    weight: str
    sensor: str


def _parse_info_txt(info_path: Path) -> tuple[int | None, str, str, str]:
    text = info_path.read_text(encoding="utf-8")
    age: int | None = None
    for pattern in (_AGE_AT_BASELINE_RE, _AGE_AT_ENROLLMENT_RE):
        match = pattern.search(text)
        if match:
            age = int(float(match.group(1)))
            break

    gender_raw = ""
    gender_match = _GENDER_RE.search(text)
    if gender_match:
        gender_raw = gender_match.group(1).strip()
    gender = _normalize_gender(gender_raw)

    weight = ""
    weight_match = _WEIGHT_LB_RE.search(text)
    if weight_match:
        weight_lb = float(weight_match.group(1))
        weight_kg = round(weight_lb * 0.453592)
        weight = f"{weight_kg} kg"

    sensor = ""
    cgm_match = _CGM_CODE_RE.search(text)
    if cgm_match:
        sensor = _CGM_LABELS.get(cgm_match.group(1), "CGM")

    return age, gender, weight, sensor


def _normalize_gender(raw: str) -> str:
    lowered = raw.strip().lower()
    if not lowered or lowered == "unknown":
        return ""
    if "female" in lowered or lowered == "woman":
        return "female"
    if "male" in lowered or lowered == "man":
        return "male"
    if "nonbinary" in lowered or "non-binary" in lowered:
        return "na"
    return raw.strip()


def _is_adult(age: int | None) -> bool:
    return age is not None and age >= _ADULT_MIN_AGE


def discover_generic_dataset_sources() -> list[GenericDatasetSource]:
    sources: list[GenericDatasetSource] = []

    example_path = _example_csv_path()
    if example_path.exists():
        sources.append(
            GenericDatasetSource(
                source_name=example_path.name,
                csv_path=example_path,
                age_years=28,
                gender="female",
                weight="67 kg",
                sensor="Dexcom G6",
            )
        )

    subjects_root = _subjects_dir()
    if not subjects_root.is_dir():
        return sources

    for subject_dir in sorted(subjects_root.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("loop_"):
            continue

        info_path = subject_dir / "info.txt"
        csv_candidates = sorted(subject_dir.glob("*_chronological.csv"))
        if not info_path.exists() or not csv_candidates:
            continue

        age, gender, weight, sensor = _parse_info_txt(info_path)
        if not _is_adult(age):
            continue

        csv_path = csv_candidates[0]
        sources.append(
            GenericDatasetSource(
                source_name=csv_path.name,
                csv_path=csv_path,
                age_years=age,
                gender=gender,
                weight=weight,
                sensor=sensor,
            )
        )

    return sources


def pick_random_generic_source(*, exclude: set[str] | None = None) -> GenericDatasetSource:
    sources = discover_generic_dataset_sources()
    if not sources:
        raise FileNotFoundError("No generic dataset sources are configured")

    blocked = {name.lower() for name in (exclude or set())}
    pool = [source for source in sources if source.source_name.lower() not in blocked]
    if not pool:
        pool = sources
    return random.choice(pool)


def load_generic_dataset_source(source: GenericDatasetSource) -> tuple[pl.DataFrame, pl.DataFrame]:
    with start_action(
        action_type=u"load_generic_dataset_source",
        source_name=source.source_name,
        csv_path=str(source.csv_path),
    ):
        if source.csv_path.name == "example.csv":
            return load_glucose_data(source.csv_path)
        return load_loop_chronological_data(source.csv_path)


def load_random_generic_dataset(
    *, exclude: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, GenericDatasetSource]:
    source = pick_random_generic_source(exclude=exclude)
    glucose_df, events_df = load_generic_dataset_source(source)
    return glucose_df, events_df, source
