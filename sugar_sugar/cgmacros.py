"""Import PhysioNet CGMacros subjects into the app store schema.

Parsing is ``cgm-format``'s job: :func:`cgm_format.FormatParser.parse_tracks`
reads a subject table into the extended unified schema, carrying macronutrients
and the meal photograph reference that the six core data columns cannot hold.

Two things the library deliberately does not do, and this module still must:

* **Downsample.** CGMacros is a 1-minute interpolated table and the game grid is
  5-minute. ``synchronize_timestamps`` snaps readings onto a grid, it does not
  reduce cadence -- asking it for 5 minutes yields five rows on the same
  timestamp -- so :func:`_downsample_glucose_5min` still owns that.
* **Cluster meals.** The library emits one event per source meal row, faithfully.
  Merging a meal and the snack logged minutes after it into one chart marker is
  this app's editorial choice, so :func:`_cluster_meal_rows` still owns that.

Tracks are alternatives, not shards: a subject yields a complete ``libre`` view
and a complete ``dexcom`` view of the same days, meals replicated into both, so
concatenating them would double-count every meal. Exactly one is
played -- Dexcom, because the corpus's Libre series is badly calibrated and the
library's synthetic ``mean`` track would blend that in. See :func:`_playable_track`.

Download a local copy with ``uv run download-cgmacros``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from cgm_format import FormatParser
from eliot import start_action

from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.corpus import adapt_unified, empty_events_frame
from sugar_sugar.download_cgmacros import default_dest, dataset_is_present

logger = logging.getLogger(__name__)

_PHOTO_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic")
_SUBJECT_DIR_RE = re.compile(r"^CGMacros-(\d+)$", re.IGNORECASE)
_SUBJECT_CSV_RE = re.compile(r"^CGMacros-(\d+)\.csv$", re.IGNORECASE)
_LB_TO_KG = 0.453592

_SUBJECT_ID_ALIASES: tuple[str, ...] = (
    "subject",
    "subject_id",
    "subjectid",
    "id",
    "participant",
    "participant_id",
    "pid",
)
_AGE_ALIASES: tuple[str, ...] = ("age",)
_GENDER_ALIASES: tuple[str, ...] = ("gender", "sex")
_WEIGHT_ALIASES: tuple[str, ...] = ("body weight", "bodyweight", "weight", "weight_lb")


@dataclass(frozen=True, slots=True)
class CGMacrosBio:
    subject_id: int
    age_years: int | None
    gender: str
    weight: str


@dataclass(frozen=True, slots=True)
class CGMacrosSource:
    subject_id: int
    source_name: str
    csv_path: Path
    subject_dir: Path
    age_years: int | None
    gender: str
    weight: str
    sensor: str


def dataset_root(dest: Optional[Path] = None) -> Path:
    return dest if dest is not None else default_dest()


def cgmacros_table_dir(dest: Optional[Path] = None) -> Path:
    return dataset_root(dest) / "CGMacros"


def is_cgmacros_csv(file_path: Path) -> bool:
    """True when *file_path* is a CGMacros participant table."""
    name = file_path.name
    if _SUBJECT_CSV_RE.match(name):
        return True
    if not file_path.is_file():
        return False
    header = file_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if not header:
        return False
    columns = {part.strip().lower() for part in header[0].split(",")}
    has_glucose = "dexcom gl" in columns or "libre gl" in columns
    has_meal = "meal type" in columns or "image path" in columns
    return has_glucose and has_meal


def subject_id_from_path(path: Path) -> int | None:
    for token in (path.name, path.stem, path.parent.name):
        match = _SUBJECT_DIR_RE.match(token) or _SUBJECT_CSV_RE.match(token)
        if match:
            return int(match.group(1))
    return None


def _norm_header(name: str) -> str:
    return " ".join(name.replace("\ufeff", "").strip().lower().split())


def _column(df: pl.DataFrame, aliases: Iterable[str]) -> str | None:
    by_norm = {_norm_header(name): name for name in df.columns}
    for alias in aliases:
        found = by_norm.get(_norm_header(alias))
        if found is not None:
            return found
    return None


def _normalize_gender(raw: str) -> str:
    lowered = raw.strip().lower()
    if lowered in {"f", "female", "woman"}:
        return "female"
    if lowered in {"m", "male", "man"}:
        return "male"
    if not lowered or lowered in {"unknown", "n/a", "na"}:
        return ""
    return raw.strip()


def _weight_display(weight_lb: float | None) -> str:
    if weight_lb is None:
        return ""
    kilograms = round(float(weight_lb) * _LB_TO_KG)
    return f"{kilograms} kg"


def load_cgmacros_bio(dest: Optional[Path] = None) -> dict[int, CGMacrosBio]:
    """Load ``bio.csv`` keyed by participant id (1 for CGMacros-001)."""
    bio_path = cgmacros_table_dir(dest) / "bio.csv"
    if not bio_path.is_file():
        return {}

    raw = pl.read_csv(bio_path, infer_schema_length=2000)
    if raw.height == 0:
        return {}

    id_col = _column(raw, _SUBJECT_ID_ALIASES)
    age_col = _column(raw, _AGE_ALIASES)
    gender_col = _column(raw, _GENDER_ALIASES)
    weight_col = _column(raw, _WEIGHT_ALIASES)

    out: dict[int, CGMacrosBio] = {}
    for index, row in enumerate(raw.iter_rows(named=True), start=1):
        subject_id: int | None = None
        if id_col is not None:
            token = str(row.get(id_col) or "").strip()
            digits = re.search(r"(\d+)", token)
            if digits:
                subject_id = int(digits.group(1))
        if subject_id is None:
            subject_id = index

        age: int | None = None
        if age_col is not None and row.get(age_col) not in (None, ""):
            age = int(float(row[age_col]))

        gender = ""
        if gender_col is not None:
            gender = _normalize_gender(str(row.get(gender_col) or ""))

        weight_lb: float | None = None
        if weight_col is not None and row.get(weight_col) not in (None, ""):
            weight_lb = float(row[weight_col])

        out[subject_id] = CGMacrosBio(
            subject_id=subject_id,
            age_years=age,
            gender=gender,
            weight=_weight_display(weight_lb),
        )
    return out


def discover_cgmacros_sources(dest: Optional[Path] = None) -> list[CGMacrosSource]:
    """Discover participant CSVs under ``data/cgmmacros/`` (or *dest*)."""
    root = dataset_root(dest)
    table_dir = cgmacros_table_dir(dest)
    if not dataset_is_present(root) and not table_dir.is_dir():
        return []

    bio = load_cgmacros_bio(dest)
    sources: list[CGMacrosSource] = []
    for subject_dir in sorted(table_dir.glob("CGMacros-*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_id_from_path(subject_dir)
        csv_path = subject_dir / f"{subject_dir.name}.csv"
        if subject_id is None or not csv_path.is_file():
            continue
        record = bio.get(subject_id)
        sources.append(
            CGMacrosSource(
                subject_id=subject_id,
                source_name=csv_path.name,
                csv_path=csv_path,
                subject_dir=subject_dir,
                age_years=record.age_years if record else None,
                gender=record.gender if record else "",
                weight=record.weight if record else "",
                sensor="Dexcom G6 Pro",
            )
        )
    return sources


def resolve_photo_path(image_path: str, subject_dir: Path) -> str:
    """Return a subject-relative posix path, or the original token if unresolved."""
    raw = image_path.strip().replace("\\", "/")
    if not raw:
        return ""
    parts = [part for part in raw.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        return ""
    candidate = subject_dir.joinpath(*parts)
    if candidate.is_file():
        return "/".join(parts)
    filename = Path(raw).name
    photos_fallback = subject_dir / "photos" / filename
    if photos_fallback.is_file():
        return f"photos/{filename}"
    return raw


def _downsample_glucose_5min(glucose_df: pl.DataFrame) -> pl.DataFrame:
    """Keep one reading per 5-minute clock bucket (game grid)."""
    if glucose_df.height == 0:
        return glucose_df
    return (
        glucose_df.sort("time")
        .group_by(pl.col("time").dt.truncate("5m").alias("bucket"), maintain_order=True)
        .agg(
            [
                pl.col("gl").first(),
                pl.col("prediction").first(),
                pl.col("age").first(),
                pl.col("user_id").first(),
            ]
        )
        .select(
            [
                pl.col("bucket").alias("time"),
                pl.col("gl"),
                pl.col("prediction"),
                pl.col("age"),
                pl.col("user_id"),
            ]
        )
        .filter(pl.col("time").is_not_null() & pl.col("gl").is_not_null())
    )


def _cluster_meal_rows(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not rows:
        return []
    clusters: list[list[dict[str, object]]] = [[rows[0]]]
    for row in rows[1:]:
        previous = clusters[-1][-1]
        same_meal = (
            str(row.get("meal_type") or "") == str(previous.get("meal_type") or "")
            and bool(row.get("meal_type"))
        )
        previous_time = previous["time"]
        row_time = row["time"]
        close = (
            previous_time is not None
            and row_time is not None
            and (row_time - previous_time) <= _MEAL_CLUSTER_GAP
        )
        if close and (same_meal or not row.get("meal_type") or not previous.get("meal_type")):
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


#: Sensor tracks in preference order. Dexcom leads; see :func:`_playable_track`.
_TRACK_PREFERENCE: tuple[str, ...] = ("dexcom", "libre")


def _playable_track(csv_path: Path) -> tuple[str, pl.DataFrame]:
    """Pick the one sensor series to play from. Dexcom unless it is empty.

    A CGMacros subject wears two sensors, and the library offers three views:
    ``libre``, ``dexcom``, and a synthetic ``mean`` whose ``mean_horizontal``
    ignores nulls -- so it averages where both read and passes a lone reading
    through untouched, exactly reproducing the coalesce this module used to do
    by hand, and recovering the ~8% of rows Dexcom alone does not cover.

    **The mean is not used, because the two sensors do not agree.** Fitting
    ``libre = slope*dexcom + intercept`` over all 45 published subjects gives a
    median slope of **0.70** (range 0.10-1.16) with a small intercept: Libre is
    not offset from Dexcom, it *compresses* the excursion to roughly two thirds,
    by a factor that differs per subject. Median correlation is 0.82 and 15 of
    45 subjects fall below 0.7, so for a third of the corpus the two series
    disagree in shape and not merely in level. Averaging assumes two comparable
    estimates of one quantity with unbiased independent error; all three of
    those fail here, and the mean would carry ~85% of true excursion amplitude
    with a per-subject distortion -- straight into a study that measures human
    prediction error in mg/dL.

    Libre is also the implausible one: it reads below 70 mg/dL for 82% and 86%
    of subjects 007 and 015 respectively, against 0.4% of all Dexcom readings.
    So the span Dexcom-only gives up is not good data being discarded -- of the
    57,755 rows only Libre covers, 10.1% are sub-70 against Dexcom's 0.4%. And
    span is not the binding constraint: ten Dexcom days is ~2850 five-minute
    readings, some 79 non-overlapping 36-point windows per subject.

    Libre remains the fallback for a subject whose Dexcom series is empty --
    a degraded trace beats no round at all.
    """
    tracks = FormatParser.parse_tracks(csv_path)
    for name in _TRACK_PREFERENCE:
        frame = tracks.get(name)
        if frame is not None and frame.get_column("glucose").drop_nulls().len() > 0:
            if name != _TRACK_PREFERENCE[0]:
                logger.warning(
                    "CGMacros %s: no Dexcom readings, falling back to %r, whose "
                    "calibration in this corpus is unreliable.",
                    csv_path.name,
                    name,
                )
            return name, frame
    name = next(iter(tracks))
    return name, tracks[name]


def _cluster_meal_events(events_df: pl.DataFrame) -> pl.DataFrame:
    """Collapse meal rows logged within ``_MEAL_CLUSTER_GAP`` into one marker.

    The library reports every meal row the source recorded; a subject who logged
    a main dish and its side six minutes apart gets two events. On the chart that
    is two markers on top of each other, so they are merged here -- keeping the
    first row's label and the first photograph and carbohydrate figure the
    cluster offers.
    """
    if events_df.height == 0:
        return events_df
    meals = events_df.filter(pl.col("event_type") == "Carbohydrates").sort("time")
    others = events_df.filter(pl.col("event_type") != "Carbohydrates")
    if meals.height == 0:
        return events_df

    merged: list[dict[str, object]] = []
    for cluster in _cluster_meal_rows(meals.to_dicts()):
        head = next((row for row in cluster if row.get("meal_type")), cluster[0])
        merged.append(
            {
                **head,
                "photo_path": next(
                    (str(row.get("photo_path") or "") for row in cluster if row.get("photo_path")),
                    "",
                ),
                "carbs_g": next(
                    (row.get("carbs_g") for row in cluster if row.get("carbs_g") is not None),
                    None,
                ),
            }
        )
    clustered = pl.DataFrame(merged, schema=events_df.schema)
    return pl.concat([others, clustered], how="vertical").sort("time")


def cgmacros_photo_url(source_name: str, photo_path: str) -> str:
    """Public URL for a meal photo belonging to ``CGMacros-NNN.csv``."""
    subject = Path(str(source_name or "")).stem
    rel = str(photo_path or "").replace("\\", "/").lstrip("/")
    return f"/cgmacros/{subject}/photo/{rel}"


def resolve_served_photo(
    subject: str,
    rel_path: str,
    dest: Optional[Path] = None,
) -> Path | None:
    """Resolve a meal photo under the CGMacros extract. Rejects path escape."""
    match = _SUBJECT_DIR_RE.match(str(subject or "").strip())
    if match is None:
        return None
    subject_name = f"CGMacros-{int(match.group(1)):03d}"
    raw = str(rel_path or "").replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None
    if Path(parts[-1]).suffix.lower() not in _PHOTO_SUFFIXES:
        return None

    roots = [cgmacros_table_dir(dest) / subject_name]
    for source in discover_cgmacros_sources(dest):
        if source.subject_dir.name == subject_name:
            roots.append(source.subject_dir)

    for root in roots:
        candidate = root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def window_has_visible_food_photo(
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
) -> bool:
    """True when a meal photo sits in the visible (non-hidden) part of the window."""
    return bool(visible_food_photo_events(window_df, events_df, hide_last_hour=True))


def visible_food_photo_events(
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
    *,
    hide_last_hour: bool,
) -> list[dict[str, object]]:
    """Meal rows that have a photo and are allowed to render on this slice."""
    if window_df.height == 0 or events_df.height == 0:
        return []
    if "photo_path" not in events_df.columns or "time" not in events_df.columns:
        return []
    start_time = window_df.get_column("time")[0]
    if hide_last_hour and window_df.height > PREDICTION_HOUR_OFFSET:
        visible_end = window_df.get_column("time")[
            window_df.height - PREDICTION_HOUR_OFFSET - 1
        ]
    else:
        visible_end = window_df.get_column("time")[-1]
    rows: list[dict[str, object]] = []
    for row in events_df.iter_rows(named=True):
        photo = str(row.get("photo_path") or "").strip()
        note = str(row.get("food_note") or "").strip()
        event_time = row.get("time")
        if (not photo and not note) or event_time is None:
            continue
        if event_time < start_time or event_time > visible_end:
            continue
        rows.append(row)
    return rows


def load_cgmacros_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one CGMacros participant CSV into the app store schema."""
    path = Path(file_path)
    with start_action(
        action_type=u"load_cgmacros_data",
        file_path=str(path),
    ) as action:
        subject_id = subject_id_from_path(path) or 1
        subject_dir = path.parent
        track_name, unified_df = _playable_track(path)
        glucose_df, events_df = adapt_unified(
            unified_df,
            subject_id=subject_id,
            photo_resolver=lambda raw: resolve_photo_path(raw, subject_dir),
        )
        glucose_df = _downsample_glucose_5min(glucose_df)
        events_df = _cluster_meal_events(events_df) if events_df.height else empty_events_frame()
        action.add_success_fields(
            subject_id=subject_id,
            track=track_name,
            glucose_rows=glucose_df.height,
            event_rows=events_df.height,
            photo_events=events_df.filter(pl.col("photo_path") != "").height,
        )
        return glucose_df, events_df
