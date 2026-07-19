from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.data import load_glucose_data, load_loop_chronological_data

_ADULT_MIN_AGE = 18
_SAME_SOURCE_BUFFER = timedelta(hours=2)
_PICK_ATTEMPTS_PER_SOURCE = 64
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


def resolve_generic_source_path(source_name: str) -> Path | None:
    """Map a stored ``data_source_name`` (file basename) to its on-disk path."""
    name = Path(str(source_name or "")).name
    if not name:
        return None
    for source in discover_generic_dataset_sources():
        if source.source_name == name:
            return source.csv_path
    return None


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


def _normalize_time_token(value: str) -> str:
    return value.strip().replace(" ", "T")


def generic_window_slice_key_from_values(times: list[str], glucose: list[float]) -> str:
    """Stable fingerprint for a generic window (content-based, file-agnostic)."""
    payload = "|".join(
        f"{_normalize_time_token(t)}:{g:.1f}" for t, g in zip(times, glucose)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def generic_window_slice_key(window_df: pl.DataFrame) -> str:
    times = window_df.get_column("time").dt.strftime("%Y-%m-%dT%H:%M:%S").to_list()
    glucose = [round(float(g), 1) for g in window_df.get_column("gl").to_list()]
    return generic_window_slice_key_from_values(times, glucose)


def generic_window_slice_key_from_round(round_info: dict[str, Any]) -> str | None:
    if not bool(round_info.get("is_example_data", True)):
        return None
    stored = round_info.get("generic_slice_key")
    if stored:
        return str(stored)

    times_raw = round_info.get("window_times") or []
    if len(times_raw) < 2:
        return None
    times = [_normalize_time_token(str(t)) for t in times_raw]

    table = round_info.get("prediction_table_data") or []
    actual_row = next(
        (row for row in table if row.get("metric") == "Actual Glucose"),
        None,
    )
    if not actual_row:
        return None

    glucose: list[float] = []
    for idx in range(len(times)):
        raw = actual_row.get(f"t{idx}")
        if raw is None or raw == "-":
            return None
        glucose.append(round(float(raw), 1))
    return generic_window_slice_key_from_values(times, glucose)


@dataclass(frozen=True, slots=True)
class GenericRoundWindow:
    source_name: str
    window_start: datetime
    window_end: datetime
    anchor_time: datetime
    slice_key: str


def _parse_window_datetime(value: str) -> datetime:
    return datetime.fromisoformat(_normalize_time_token(value))


def generic_round_window_from_df(
    window_df: pl.DataFrame,
    *,
    source_name: str,
) -> GenericRoundWindow:
    times = window_df.get_column("time")
    anchor_idx = max(0, len(window_df) - PREDICTION_HOUR_OFFSET)
    return GenericRoundWindow(
        source_name=source_name,
        window_start=times[0],
        window_end=times[-1],
        anchor_time=times[anchor_idx],
        slice_key=generic_window_slice_key(window_df),
    )


def generic_round_window_from_round(round_info: dict[str, Any]) -> GenericRoundWindow | None:
    if not bool(round_info.get("is_example_data", True)):
        return None

    times_raw = round_info.get("window_times") or []
    if len(times_raw) < 2:
        return None

    times = [_parse_window_datetime(str(value)) for value in times_raw]
    anchor_idx = max(0, len(times) - PREDICTION_HOUR_OFFSET)
    slice_key = generic_window_slice_key_from_round(round_info)
    if not slice_key:
        return None

    return GenericRoundWindow(
        source_name=str(round_info.get("data_source_name") or ""),
        window_start=times[0],
        window_end=times[-1],
        anchor_time=times[anchor_idx],
        slice_key=slice_key,
    )


def collect_generic_round_history(
    rounds: list[dict[str, Any]] | None,
    user_info: dict[str, Any] | None = None,
) -> list[GenericRoundWindow]:
    """Completed generic rounds for this game (``rounds`` store)."""
    del user_info
    history: list[GenericRoundWindow] = []
    for round_info in rounds or []:
        entry = generic_round_window_from_round(round_info)
        if entry:
            history.append(entry)
    return history


def windows_conflict(existing: GenericRoundWindow, candidate: GenericRoundWindow) -> bool:
    """True for exact duplicates, or same-source windows within ±2h of each other."""
    if (
        existing.slice_key
        and candidate.slice_key
        and candidate.slice_key == existing.slice_key
    ):
        return True
    if existing.source_name != candidate.source_name:
        return False
    buffered_start = existing.window_start - _SAME_SOURCE_BUFFER
    buffered_end = existing.window_end + _SAME_SOURCE_BUFFER
    return candidate.window_start <= buffered_end and candidate.window_end >= buffered_start


def _candidate_start_indices(row_count: int, points: int) -> list[int]:
    """Random sample of valid start indices (avoids scanning entire LOOP files)."""
    max_start = row_count - points
    if max_start <= 0:
        return [0]
    pool_size = max_start + 1
    if pool_size <= _PICK_ATTEMPTS_PER_SOURCE:
        starts = list(range(pool_size))
        random.shuffle(starts)
        return starts
    return random.sample(range(pool_size), _PICK_ATTEMPTS_PER_SOURCE)


@dataclass(frozen=True, slots=True)
class GenericWindowSelection:
    window_df: pl.DataFrame
    events_df: pl.DataFrame
    source: GenericDatasetSource
    start_index: int
    slice_key: str


def pick_unique_generic_window(
    points: int,
    history: list[GenericRoundWindow] | None = None,
) -> GenericWindowSelection:
    """Pick a random generic window that does not duplicate prior rounds this game.

    Rules (per game / session history):
    - never reuse the same window content (``slice_key``)
    - never reuse the same source within ±2h of a prior window's timestamps
    """
    sources = discover_generic_dataset_sources()
    if not sources:
        raise FileNotFoundError("No generic dataset sources are configured")

    prior = list(history or [])
    source_pool = list(sources)
    random.shuffle(source_pool)
    fallback: GenericWindowSelection | None = None

    with start_action(
        action_type=u"pick_unique_generic_window",
        points=points,
        history_count=len(prior),
    ) as action:
        for source in source_pool:
            glucose_df, events_df = load_generic_dataset_source(source)
            for start_index in _candidate_start_indices(len(glucose_df), points):
                window_df = glucose_df.slice(start_index, points)
                slice_key = generic_window_slice_key(window_df)
                round_window = GenericRoundWindow(
                    source_name=source.source_name,
                    window_start=window_df.get_column("time")[0],
                    window_end=window_df.get_column("time")[-1],
                    anchor_time=window_df.get_column("time")[
                        max(0, len(window_df) - PREDICTION_HOUR_OFFSET)
                    ],
                    slice_key=slice_key,
                )
                selection = GenericWindowSelection(
                    window_df=window_df,
                    events_df=events_df,
                    source=source,
                    start_index=start_index,
                    slice_key=slice_key,
                )
                if fallback is None:
                    fallback = selection
                if any(windows_conflict(old, round_window) for old in prior):
                    continue
                action.log(
                    message_type="unique_slice_selected",
                    source_name=source.source_name,
                    start_index=start_index,
                    slice_key=slice_key,
                    window_start=round_window.window_start.isoformat(),
                    window_end=round_window.window_end.isoformat(),
                )
                return selection

        if fallback is None:
            raise ValueError("Could not pick any generic window")

        action.log(
            message_type="slice_pool_exhausted_reusing",
            source_name=fallback.source.source_name,
            start_index=fallback.start_index,
            slice_key=fallback.slice_key,
        )
        return fallback
