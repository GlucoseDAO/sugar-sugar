"""Import D1NAMO (Dubosson) T1D tables into the app store schema.

This is an in-app formatter, not a library and not ``cgm-format``. D1NAMO is
the type-1 / insulin-using Format A arm (glucose / insulin / food + photos).
Non-insulin arms use BIG IDEAs instead.

Download a local copy with ``uv run download-d1namo``.
Paper: Dubosson et al., Informatics in Medicine Unlocked, 2018.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from eliot import start_action

from sugar_sugar.download_d1namo import dataset_is_present, default_dest

_PHOTO_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic")
_SUBJECT_DIR_RE = re.compile(r"^(\d{3})$")
_SOURCE_NAME_RE = re.compile(r"^D1NAMO-(\d{3})\.csv$", re.IGNORECASE)
_MMOL_TO_MGDL: float = 18.0
_GLUCOSE_MMOL_MAX: float = 40.0

_DATE_ALIASES: tuple[str, ...] = ("date",)
_TIME_ALIASES: tuple[str, ...] = ("time",)
_DATETIME_ALIASES: tuple[str, ...] = ("datetime", "timestamp", "begin")
_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y:%m:%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
)
_GLUCOSE_ALIASES: tuple[str, ...] = ("glucose", "gl", "cgm")
_TYPE_ALIASES: tuple[str, ...] = ("type",)
_FAST_ALIASES: tuple[str, ...] = ("fast_insulin", "fast insulin", "insulin_fast")
_SLOW_ALIASES: tuple[str, ...] = ("slow_insulin", "slow insulin", "insulin_slow")
_PICTURE_ALIASES: tuple[str, ...] = ("picture", "image", "photo", "filename", "image path")
_CARBS_ALIASES: tuple[str, ...] = ("carbs", "carbohydrates", "cho")
_DESCRIPTION_ALIASES: tuple[str, ...] = ("description", "comments", "comment", "food")


@dataclass(frozen=True, slots=True)
class D1NamoSource:
    subject_id: str
    source_name: str
    csv_path: Path
    subject_dir: Path
    age_years: int | None
    gender: str
    weight: str
    sensor: str
    intervention: str = "d1namo"


def dataset_root(dest: Optional[Path] = None) -> Path:
    return dest if dest is not None else default_dest()


def is_d1namo_source_name(source_name: str) -> bool:
    return bool(_SOURCE_NAME_RE.match(Path(str(source_name or "")).name))


def is_d1namo_path(file_path: Path) -> bool:
    """True when *file_path* is a D1NAMO glucose table or virtual source name."""
    path = Path(file_path)
    if is_d1namo_source_name(path.name):
        return True
    if path.name.lower() != "glucose.csv":
        return False
    if _SUBJECT_DIR_RE.match(path.parent.name):
        return True
    return "d1namo" in str(path).replace("\\", "/").lower()


def subject_id_from_path(path: Path) -> str | None:
    for token in (path.name, path.stem, path.parent.name):
        source_match = _SOURCE_NAME_RE.match(token)
        if source_match:
            return source_match.group(1)
        dir_match = _SUBJECT_DIR_RE.match(token)
        if dir_match:
            return dir_match.group(1)
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


def _numeric_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .replace({"": None, "na": None, "n/a": None, "-": None})
        .cast(pl.Float64, strict=False)
    )


def _text_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()


def _parse_datetime_text(text: pl.Series) -> pl.Series:
    """Parse D1NAMO timestamps. Never infer: colon dates like 2014:10:01 crash Polars."""
    parsed = text.str.to_datetime(format=_DATETIME_FORMATS[0], strict=False)
    for fmt in _DATETIME_FORMATS[1:]:
        if parsed.null_count() == 0:
            break
        parsed = parsed.fill_null(text.str.to_datetime(format=fmt, strict=False))
    return parsed


def _combine_date_time(df: pl.DataFrame) -> pl.Series:
    datetime_col = _column(df, _DATETIME_ALIASES)
    if datetime_col is not None:
        text = df.get_column(datetime_col).cast(pl.Utf8, strict=False).str.strip_chars()
        return _parse_datetime_text(text)

    date_col = _column(df, _DATE_ALIASES)
    time_col = _column(df, _TIME_ALIASES)
    if date_col is None or time_col is None:
        return pl.Series("time", [None] * df.height, dtype=pl.Datetime)

    combined = (
        df.get_column(date_col).cast(pl.Utf8, strict=False).str.strip_chars()
        + " "
        + df.get_column(time_col).cast(pl.Utf8, strict=False).str.strip_chars()
    )
    return _parse_datetime_text(combined)


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [],
            "event_type": [],
            "event_subtype": [],
            "insulin_value": [],
            "photo_path": [],
            "meal_type": [],
            "carbs_g": [],
        },
        schema={
            "time": pl.Datetime,
            "event_type": pl.String,
            "event_subtype": pl.String,
            "insulin_value": pl.Float64,
            "photo_path": pl.String,
            "meal_type": pl.String,
            "carbs_g": pl.Float64,
        },
    )


def _glucose_to_mgdl(values: pl.Series) -> pl.Series:
    numeric = values.cast(pl.Float64, strict=False)
    peak = numeric.max()
    if peak is not None and float(peak) < _GLUCOSE_MMOL_MAX:
        return numeric * _MMOL_TO_MGDL
    return numeric


def resolve_photo_path(image_path: str, subject_dir: Path) -> str:
    raw = str(image_path or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts = [part for part in raw.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        return ""
    candidate = subject_dir.joinpath(*parts)
    if candidate.is_file():
        return "/".join(parts)
    filename = Path(raw).name
    for folder in ("pictures", "photos", "food", "food_pictures"):
        fallback = subject_dir / folder / filename
        if fallback.is_file():
            return f"{folder}/{filename}"
    return raw


def _read_optional_csv(path: Path) -> pl.DataFrame | None:
    if not path.is_file():
        return None
    return pl.read_csv(path, infer_schema_length=2000)


def _insulin_events(insulin_df: pl.DataFrame) -> list[dict[str, object]]:
    times = _combine_date_time(insulin_df)
    working = insulin_df.with_columns(times.alias("time"))
    fast_col = _column(working, _FAST_ALIASES)
    slow_col = _column(working, _SLOW_ALIASES)
    events: list[dict[str, object]] = []
    for row in working.iter_rows(named=True):
        event_time = row.get("time")
        if event_time is None:
            continue
        if fast_col is not None:
            fast = row.get(fast_col)
            if fast not in (None, "") and float(fast) != 0:
                events.append(
                    {
                        "time": event_time,
                        "event_type": "Insulin",
                        "event_subtype": "Fast Acting",
                        "insulin_value": float(fast),
                        "photo_path": "",
                        "meal_type": "",
                        "carbs_g": None,
                    }
                )
        if slow_col is not None:
            slow = row.get(slow_col)
            if slow not in (None, "") and float(slow) != 0:
                events.append(
                    {
                        "time": event_time,
                        "event_type": "Insulin",
                        "event_subtype": "Long-Acting",
                        "insulin_value": float(slow),
                        "photo_path": "",
                        "meal_type": "",
                        "carbs_g": None,
                    }
                )
    return events


def _food_events(food_df: pl.DataFrame, subject_dir: Path) -> list[dict[str, object]]:
    times = _combine_date_time(food_df)
    working = food_df.with_columns(times.alias("time"))
    picture_col = _column(working, _PICTURE_ALIASES)
    carbs_col = _column(working, _CARBS_ALIASES)
    description_col = _column(working, _DESCRIPTION_ALIASES)
    events: list[dict[str, object]] = []
    for row in working.iter_rows(named=True):
        event_time = row.get("time")
        if event_time is None:
            continue
        photo_raw = str(row.get(picture_col) or "") if picture_col else ""
        photo_path = resolve_photo_path(photo_raw, subject_dir) if photo_raw else ""
        events.append(
            {
                "time": event_time,
                "event_type": "Carbohydrates",
                "event_subtype": "Carbs",
                "insulin_value": None,
                "photo_path": photo_path,
                "meal_type": str(row.get(description_col) or "") if description_col else "",
                "carbs_g": (
                    float(row[carbs_col])
                    if carbs_col is not None and row.get(carbs_col) not in (None, "")
                    else None
                ),
            }
        )
    return events


def format_d1namo_frames(
    glucose_df: pl.DataFrame,
    *,
    insulin_df: pl.DataFrame | None = None,
    food_df: pl.DataFrame | None = None,
    subject_dir: Optional[Path] = None,
    subject_id: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Format D1NAMO participant tables into app glucose + events frames."""
    glucose_col = _column(glucose_df, _GLUCOSE_ALIASES)
    if glucose_col is None:
        raise ValueError("D1NAMO glucose.csv is missing a glucose column")

    working = glucose_df.with_columns(_combine_date_time(glucose_df).alias("time"))
    type_col = _column(working, _TYPE_ALIASES)
    if type_col is not None:
        type_text = _text_expr(type_col).str.to_lowercase()
        working = working.filter(type_text.is_in(["cgm", "cgm ", ""]))
    working = working.with_columns(_numeric_expr(glucose_col).alias("_gl_raw"))
    filtered = working.filter(pl.col("time").is_not_null() & pl.col("_gl_raw").is_not_null())
    glucose_out = pl.DataFrame(
        {
            "time": filtered.get_column("time"),
            "gl": _glucose_to_mgdl(filtered.get_column("_gl_raw")),
            "prediction": [0.0] * filtered.height,
            "age": [0] * filtered.height,
            "user_id": [int(subject_id)] * filtered.height,
        }
    ).sort("time")

    events: list[dict[str, object]] = []
    if insulin_df is not None and insulin_df.height > 0:
        events.extend(_insulin_events(insulin_df))
    if food_df is not None and food_df.height > 0 and subject_dir is not None:
        events.extend(_food_events(food_df, subject_dir))
    if not events:
        return glucose_out, _empty_events()
    events_df = pl.DataFrame(
        events,
        schema={
            "time": pl.Datetime,
            "event_type": pl.String,
            "event_subtype": pl.String,
            "insulin_value": pl.Float64,
            "photo_path": pl.String,
            "meal_type": pl.String,
            "carbs_g": pl.Float64,
        },
    ).sort("time")
    return glucose_out, events_df


def discover_d1namo_sources(dest: Optional[Path] = None) -> list[D1NamoSource]:
    """Discover D1NAMO subject folders under ``data/d1namo/`` (or *dest*)."""
    root = dataset_root(dest)
    if not dataset_is_present(root) and not root.is_dir():
        return []

    sources: list[D1NamoSource] = []
    seen: set[str] = set()
    for glucose_path in sorted(root.rglob("glucose.csv")):
        subject_id = subject_id_from_path(glucose_path)
        if subject_id is None or subject_id in seen:
            continue
        seen.add(subject_id)
        sources.append(
            D1NamoSource(
                subject_id=subject_id,
                source_name=f"D1NAMO-{subject_id}.csv",
                csv_path=glucose_path,
                subject_dir=glucose_path.parent,
                age_years=None,
                gender="",
                weight="",
                sensor="Medtronic iPro2",
            )
        )
    return sources


def d1namo_photo_url(source_name: str, photo_path: str) -> str:
    """Public URL for a meal photo belonging to ``D1NAMO-NNN.csv``."""
    match = _SOURCE_NAME_RE.match(Path(str(source_name or "")).name)
    subject = match.group(1) if match else Path(str(source_name or "")).stem
    rel = str(photo_path or "").replace("\\", "/").lstrip("/")
    return f"/d1namo/{subject}/photo/{rel}"


def resolve_served_photo(
    subject: str,
    rel_path: str,
    dest: Optional[Path] = None,
) -> Path | None:
    """Resolve a meal photo under the D1NAMO extract. Rejects path escape."""
    match = _SUBJECT_DIR_RE.match(str(subject or "").strip()) or _SOURCE_NAME_RE.match(
        str(subject or "").strip()
    )
    if match is None:
        digits = re.search(r"(\d{3})", str(subject or ""))
        if digits is None:
            return None
        subject_id = digits.group(1)
    else:
        subject_id = match.group(1)

    raw = str(rel_path or "").replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None
    if Path(parts[-1]).suffix.lower() not in _PHOTO_SUFFIXES:
        return None

    roots: list[Path] = []
    for source in discover_d1namo_sources(dest):
        if source.subject_id == subject_id:
            roots.append(source.subject_dir)
    if not roots:
        roots.append(dataset_root(dest) / subject_id)

    for root in roots:
        candidate = root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def load_d1namo_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one D1NAMO participant into the app store schema."""
    path = Path(file_path)
    with start_action(action_type=u"load_d1namo_data", file_path=str(path)) as action:
        glucose_path = path if path.name.lower() == "glucose.csv" else path.parent / "glucose.csv"
        if not glucose_path.is_file() and is_d1namo_source_name(path.name):
            subject_id = subject_id_from_path(path)
            for source in discover_d1namo_sources():
                if source.subject_id == subject_id:
                    glucose_path = source.csv_path
                    break
        if not glucose_path.is_file():
            raise FileNotFoundError(f"D1NAMO glucose.csv not found for {path}")

        subject_dir = glucose_path.parent
        subject_token = subject_id_from_path(glucose_path) or "001"
        raw_glucose = pl.read_csv(glucose_path, infer_schema_length=10000)
        insulin_df = _read_optional_csv(subject_dir / "insulin.csv")
        food_df = _read_optional_csv(subject_dir / "food.csv")
        glucose_df, events_df = format_d1namo_frames(
            raw_glucose,
            insulin_df=insulin_df,
            food_df=food_df,
            subject_dir=subject_dir,
            subject_id=int(subject_token),
        )
        action.add_success_fields(
            glucose_rows=glucose_df.height,
            event_rows=events_df.height,
            photo_events=events_df.filter(pl.col("photo_path") != "").height,
            insulin_events=events_df.filter(pl.col("event_type") == "Insulin").height,
        )
        return glucose_df, events_df
