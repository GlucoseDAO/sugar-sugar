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

from sugar_sugar.bigideas import discover_bigideas_sources
from sugar_sugar.cgmacros import window_has_visible_food_photo
from sugar_sugar.config import PREDICTION_HOUR_OFFSET, SEQUENCE_GAP_MINUTES
from sugar_sugar.d1namo import discover_d1namo_sources
from sugar_sugar.challenge_unknown import (
    challenge_unknown_active,
    challenge_unknown_weights,
    encode_mix_policy,
    parse_mix_policy,
)
from sugar_sugar.data import load_glucose_data

GENERIC_INTERVENTION_BIGIDEAS: str = "bigideas"
GENERIC_INTERVENTION_D1NAMO: str = "d1namo"
GENERIC_INTERVENTION_MIX_T2: str = "mix_t2"
GENERIC_INTERVENTION_MIX_PREDIABETES: str = "mix_prediabetes"
GENERIC_INTERVENTION_MIX_LADA: str = "mix_lada"

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
    intervention: str = ""
    diabetic: bool | None = None


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


def discover_legacy_generic_sources() -> list[GenericDatasetSource]:
    """example.csv + LOOP subjects. Temporarily unused as the generic play pool."""
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
                # An insulin-treated trace: 73 insulin events, 42% of readings
                # above 180 mg/dL, range 39-401. Stating it matters because this
                # file is the Format A fallback when the corpora are absent, and
                # an undeclared source lands on neither scoreboard.
                diabetic=True,
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
                diabetic=True,
            )
        )

    return sources


def generic_intervention_for_user(user_info: dict[str, Any] | None) -> str:
    """Format A source policy from diabetes status / type.

    no diabetes / gestational → BIG IDEAs
    type 1 → D1NAMO
    type 2 → 50/50 mix
    prediabetes → 75% BIG IDEAs / 25% D1NAMO
    LADA → 75% D1NAMO / 25% BIG IDEAs
    Challenge the unknown (non-diabetic or type 1, formats A/C) → 50/50 opposite mix
    """
    if challenge_unknown_active(user_info):
        return encode_mix_policy(challenge_unknown_weights(user_info))
    if not user_info or user_info.get("diabetic") is not True:
        return GENERIC_INTERVENTION_BIGIDEAS
    kind = str(user_info.get("diabetic_type") or "").strip().lower()
    if kind in {"type 1", "type1", "t1"}:
        return GENERIC_INTERVENTION_D1NAMO
    if kind in {"type 2", "type2", "t2"}:
        return GENERIC_INTERVENTION_MIX_T2
    if kind.startswith("prediab"):
        return GENERIC_INTERVENTION_MIX_PREDIABETES
    if kind == "lada":
        return GENERIC_INTERVENTION_MIX_LADA
    if kind.startswith("gestational"):
        return GENERIC_INTERVENTION_BIGIDEAS
    return GENERIC_INTERVENTION_D1NAMO


def intervention_pool_weights(policy: str | None) -> dict[str, float]:
    """Per-round dataset weights for a stored ``generic_intervention`` policy."""
    mixed = parse_mix_policy(policy)
    if mixed:
        return mixed
    key = str(policy or "").strip().lower()
    if key == GENERIC_INTERVENTION_D1NAMO:
        return {GENERIC_INTERVENTION_D1NAMO: 1.0}
    if key == GENERIC_INTERVENTION_MIX_T2:
        return {GENERIC_INTERVENTION_BIGIDEAS: 0.5, GENERIC_INTERVENTION_D1NAMO: 0.5}
    if key == GENERIC_INTERVENTION_MIX_PREDIABETES:
        return {GENERIC_INTERVENTION_BIGIDEAS: 0.75, GENERIC_INTERVENTION_D1NAMO: 0.25}
    if key == GENERIC_INTERVENTION_MIX_LADA:
        return {GENERIC_INTERVENTION_D1NAMO: 0.75, GENERIC_INTERVENTION_BIGIDEAS: 0.25}
    if key == GENERIC_INTERVENTION_BIGIDEAS:
        return {GENERIC_INTERVENTION_BIGIDEAS: 1.0}
    return {GENERIC_INTERVENTION_BIGIDEAS: 1.0, GENERIC_INTERVENTION_D1NAMO: 1.0}


def _bigideas_generic_sources() -> list[GenericDatasetSource]:
    sources: list[GenericDatasetSource] = []
    for bigideas in discover_bigideas_sources():
        if bigideas.age_years is not None and not _is_adult(bigideas.age_years):
            continue
        sources.append(
            GenericDatasetSource(
                source_name=bigideas.source_name,
                csv_path=bigideas.csv_path,
                age_years=bigideas.age_years,
                gender=bigideas.gender,
                weight=bigideas.weight,
                sensor=bigideas.sensor,
                intervention=GENERIC_INTERVENTION_BIGIDEAS,
                diabetic=False,
            )
        )
    return sources


def _d1namo_generic_sources() -> list[GenericDatasetSource]:
    sources: list[GenericDatasetSource] = []
    for d1namo in discover_d1namo_sources():
        if d1namo.age_years is not None and not _is_adult(d1namo.age_years):
            continue
        sources.append(
            GenericDatasetSource(
                source_name=d1namo.source_name,
                csv_path=d1namo.csv_path,
                age_years=d1namo.age_years,
                gender=d1namo.gender,
                weight=d1namo.weight,
                sensor=d1namo.sensor,
                intervention=GENERIC_INTERVENTION_D1NAMO,
                diabetic=True,
            )
        )
    return sources


def _sources_for_pool(pool: str) -> list[GenericDatasetSource]:
    if pool == GENERIC_INTERVENTION_D1NAMO:
        return _d1namo_generic_sources()
    if pool == GENERIC_INTERVENTION_BIGIDEAS:
        return _bigideas_generic_sources()
    return []


def discover_generic_dataset_sources(
    *,
    intervention: str | None = None,
) -> list[GenericDatasetSource]:
    sources: list[GenericDatasetSource] = []

    # Temporarily disabled: generic play uses BIG IDEAs / D1NAMO by diabetes type.
    # sources.extend(discover_legacy_generic_sources())

    weights = intervention_pool_weights(intervention)
    for pool in (GENERIC_INTERVENTION_BIGIDEAS, GENERIC_INTERVENTION_D1NAMO):
        if weights.get(pool, 0) > 0:
            sources.extend(_sources_for_pool(pool))

    if not sources:
        for pool in (GENERIC_INTERVENTION_BIGIDEAS, GENERIC_INTERVENTION_D1NAMO):
            sources.extend(_sources_for_pool(pool))

    if not sources:
        sources.extend(discover_legacy_generic_sources())

    return sources


def resolve_generic_source_path(source_name: str) -> Path | None:
    """Map a stored ``data_source_name`` (file basename) to its on-disk path."""
    name = Path(str(source_name or "")).name
    if not name:
        return None
    for source in discover_generic_dataset_sources() + discover_legacy_generic_sources():
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
        return load_glucose_data(source.csv_path)


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


def window_is_continuous(
    window_df: pl.DataFrame,
    *,
    gap_minutes: int = SEQUENCE_GAP_MINUTES,
) -> bool:
    """True when a window holds one unbroken run of readings.

    A CGM trace is not one continuous run: sensors are replaced, transmitters
    drop out, people shower. ``data/example.csv`` alone breaks into eleven
    stretches, two of them a single reading long, and windows are sliced by row
    index -- so without this check a player can be shown a 36-point "hour" whose
    middle silently jumps three days. That is not a harder round, it is an
    unanswerable one that reads as a broken chart.

    Continuity is read straight off the timestamps rather than from a stamped
    ``sequence_id`` column, so it holds for a frame reconstructed from a session
    store just as well as for one fresh off disk, and the app-wide glucose
    schema stays the five columns every caller expects.

    ``gap_minutes`` matches ``cgm-format``'s own ``small_gap_max_minutes``
    default, so the app and the library agree on what counts as continuous.
    """
    if window_df.height < 2 or "time" not in window_df.columns:
        return True
    largest_gap = window_df.get_column("time").sort().diff().dt.total_minutes().max()
    return largest_gap is None or largest_gap <= gap_minutes


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


def _weighted_pool_order(
    weights: dict[str, float],
    available: dict[str, list[GenericDatasetSource]],
) -> list[str]:
    """Draw one pool from ``weights``, then list the rest only as last-resort fallbacks."""
    names = [name for name, sources in available.items() if sources and weights.get(name, 0) > 0]
    if not names:
        return list(available)
    chosen = random.choices(names, weights=[weights[name] for name in names], k=1)[0]
    return [chosen] + [name for name in names if name != chosen]


def _search_sources_for_window(
    sources: list[GenericDatasetSource],
    points: int,
    prior: list[GenericRoundWindow],
) -> tuple[GenericWindowSelection | None, bool, int]:
    """Best window from ``sources`` only.

    Ranking: unique+continuous+food, else unique+continuous, else continuous, else any.
    The bool is True when the returned window does not conflict with ``prior``.
    """
    shuffled = list(sources)
    random.shuffle(shuffled)
    fallback: GenericWindowSelection | None = None
    continuous_fallback: GenericWindowSelection | None = None
    unique_fallback: GenericWindowSelection | None = None
    discontinuous_rejected = 0
    for source in shuffled:
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
            # A window spanning a sensor gap is not a harder round, it is an
            # unanswerable one: the hour the player is asked to continue may
            # start days after the hour they were shown. Rank this above the
            # food-photo preference -- a gap-free window with no meal marker
            # is merely duller, a discontinuous one is broken.
            if not window_is_continuous(window_df):
                discontinuous_rejected += 1
                continue
            if continuous_fallback is None:
                continuous_fallback = selection
            if any(windows_conflict(old, round_window) for old in prior):
                continue
            if unique_fallback is None:
                unique_fallback = selection
            if not window_has_visible_food_photo(window_df, events_df):
                continue
            return selection, True, discontinuous_rejected
    chosen = unique_fallback or continuous_fallback or fallback
    return chosen, unique_fallback is not None, discontinuous_rejected


def pick_unique_generic_window(
    points: int,
    history: list[GenericRoundWindow] | None = None,
    *,
    intervention: str | None = None,
) -> GenericWindowSelection:
    """Pick a random generic window that does not duplicate prior rounds this game.

    Rules (per game / session history):
    - never reuse the same window content (``slice_key``)
    - never reuse the same source within ±2h of a prior window's timestamps
    - honour the intervention mix: search the weighted pool first and only
      leave it when that pool has no unused continuous window left
    """
    weights = intervention_pool_weights(intervention)
    by_pool: dict[str, list[GenericDatasetSource]] = {}
    for pool, weight in weights.items():
        if weight <= 0:
            continue
        pool_sources = _sources_for_pool(pool)
        if pool_sources:
            by_pool[pool] = pool_sources
    if not by_pool:
        sources = discover_generic_dataset_sources(intervention=intervention)
        if not sources:
            raise FileNotFoundError("No generic dataset sources are configured")
        by_pool = {"all": sources}

    prior = list(history or [])
    pool_order = _weighted_pool_order(weights, by_pool)
    last_selection: GenericWindowSelection | None = None
    last_unique = False
    discontinuous_rejected = 0

    with start_action(
        action_type=u"pick_unique_generic_window",
        points=points,
        history_count=len(prior),
        intervention=intervention,
        pool_order=pool_order,
    ) as action:
        for index, pool_name in enumerate(pool_order):
            selection, unique, rejected = _search_sources_for_window(
                by_pool[pool_name], points, prior
            )
            discontinuous_rejected += rejected
            if selection is None:
                continue
            last_selection = selection
            last_unique = unique
            if unique or index == len(pool_order) - 1:
                action.log(
                    message_type="unique_slice_selected" if unique else "slice_pool_exhausted_reusing",
                    source_name=selection.source.source_name,
                    start_index=selection.start_index,
                    slice_key=selection.slice_key,
                    pool=pool_name,
                    has_food_photo=window_has_visible_food_photo(
                        selection.window_df, selection.events_df
                    ),
                    is_continuous=window_is_continuous(selection.window_df),
                    discontinuous_rejected=discontinuous_rejected,
                    window_start=selection.window_df.get_column("time")[0].isoformat(),
                    window_end=selection.window_df.get_column("time")[-1].isoformat(),
                )
                return selection

        if last_selection is None:
            raise ValueError("Could not pick any generic window")
        action.log(
            message_type="slice_pool_exhausted_reusing",
            source_name=last_selection.source.source_name,
            start_index=last_selection.start_index,
            slice_key=last_selection.slice_key,
            has_food_photo=window_has_visible_food_photo(
                last_selection.window_df, last_selection.events_df
            ),
            is_continuous=window_is_continuous(last_selection.window_df),
            discontinuous_rejected=discontinuous_rejected,
            reused_non_unique=not last_unique,
        )
        return last_selection
