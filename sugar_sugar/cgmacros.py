"""CGMacros discovery, photo serving, and library-backed load.

Parsing is ``cgm-format`` 0.10+ (``parse_tracks``). This module keeps the
app-specific pieces: participant listing, ``bio.csv`` demographics, meal-photo
URLs, and the visible-food helper used to pick Format A windows.
Download a local copy with ``uv run download-cgmacros``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.download_cgmacros import dataset_is_present, default_dest

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
    """Load one CGMacros participant through ``cgm-format`` into the app schema."""
    from sugar_sugar.data import load_glucose_data

    return load_glucose_data(Path(file_path))
