"""Import BIG IDEAs Dexcom + food-log tables into the app store schema.

This is an in-app formatter, not a library and not ``cgm-format``. BIG IDEAs
has no meal photographs: food events carry a ``food_note`` built from the
participant food log.

Download a local copy with ``uv run download-bigideas``.
Paper: Bent et al., npj Digital Medicine, 2021.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from eliot import start_action

from sugar_sugar.download_bigideas import dataset_is_present, default_dest

_SUBJECT_DIR_RE = re.compile(r"^(\d{3})$")
_SOURCE_NAME_RE = re.compile(r"^BIGIDEAS-(\d{3})\.csv$", re.IGNORECASE)
_DEXCOM_NAME_RE = re.compile(r"^dexcom_(\d{3})\.csv$", re.IGNORECASE)
_FOOD_CLUSTER_GAP = timedelta(minutes=30)

_TIMESTAMP_ALIASES: tuple[str, ...] = (
    "timestamp (yyyy-mm-ddthh:mm:ss)",
    "timestamp",
    "time",
    "datetime",
)
_GLUCOSE_ALIASES: tuple[str, ...] = (
    "glucose value (mg/dl)",
    "glucose value",
    "glucose",
    "egv",
    "value",
)
_EVENT_TYPE_ALIASES: tuple[str, ...] = ("event type", "event_type")
_FOOD_TIME_ALIASES: tuple[str, ...] = ("time_begin", "datetime", "timestamp", "begin")
_LOGGED_FOOD_ALIASES: tuple[str, ...] = ("logged_food", "food", "item")
_SEARCHED_FOOD_ALIASES: tuple[str, ...] = ("searched_food",)
_AMOUNT_ALIASES: tuple[str, ...] = ("amount",)
_UNIT_ALIASES: tuple[str, ...] = ("unit",)
_CARB_ALIASES: tuple[str, ...] = ("total_carb", "total_carb ", "carbs", "carbohydrates")


@dataclass(frozen=True, slots=True)
class BigIdeasSource:
    subject_id: str
    source_name: str
    csv_path: Path
    subject_dir: Path
    age_years: int | None
    gender: str
    weight: str
    sensor: str
    hba1c: str
    intervention: str = "bigideas"


def dataset_root(dest: Optional[Path] = None) -> Path:
    return dest if dest is not None else default_dest()


def is_bigideas_source_name(source_name: str) -> bool:
    return bool(_SOURCE_NAME_RE.match(Path(str(source_name or "")).name))


def is_bigideas_path(file_path: Path) -> bool:
    path = Path(file_path)
    if is_bigideas_source_name(path.name):
        return True
    if _DEXCOM_NAME_RE.match(path.name):
        return True
    return "bigideas" in str(path).replace("\\", "/").lower() and path.suffix.lower() == ".csv"


def subject_id_from_path(path: Path) -> str | None:
    for token in (path.name, path.stem, path.parent.name):
        source_match = _SOURCE_NAME_RE.match(token) or _DEXCOM_NAME_RE.match(token)
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


def _header_row_index(path: Path) -> int:
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        lowered = line.lower()
        if "timestamp" in lowered and ("glucose" in lowered or "egv" in lowered or "value" in lowered):
            return index
    return 0


def _parse_datetime_series(series: pl.Series) -> pl.Series:
    text = series.cast(pl.Utf8, strict=False).str.strip_chars()
    formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S%.f",
    )
    parsed = text.str.to_datetime(format=formats[0], strict=False)
    for fmt in formats[1:]:
        if parsed.null_count() == 0:
            break
        parsed = parsed.fill_null(text.str.to_datetime(format=fmt, strict=False))
    return parsed


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
            "food_note": [],
        },
        schema={
            "time": pl.Datetime,
            "event_type": pl.String,
            "event_subtype": pl.String,
            "insulin_value": pl.Float64,
            "photo_path": pl.String,
            "meal_type": pl.String,
            "carbs_g": pl.Float64,
            "food_note": pl.String,
        },
    )


def _normalize_gender(raw: str) -> str:
    lowered = str(raw or "").strip().lower()
    if lowered == "female":
        return "female"
    if lowered == "male":
        return "male"
    return ""


def load_bigideas_demographics(dest: Optional[Path] = None) -> dict[str, tuple[str, str]]:
    """Map subject id ``001`` → (gender, hba1c)."""
    path = dataset_root(dest) / "Demographics.csv"
    if not path.is_file():
        return {}
    raw = pl.read_csv(path, infer_schema_length=200)
    id_col = _column(raw, ("id", "subject", "participant"))
    gender_col = _column(raw, ("gender", "sex"))
    hba1c_col = _column(raw, ("hba1c", "hba1c "))
    if id_col is None:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for row in raw.iter_rows(named=True):
        raw_id = str(row.get(id_col) or "").strip()
        if not raw_id:
            continue
        digits = re.search(r"(\d+)", raw_id)
        if digits is None:
            continue
        subject_id = f"{int(digits.group(1)):03d}"
        gender = _normalize_gender(str(row.get(gender_col) or "") if gender_col else "")
        hba1c = str(row.get(hba1c_col) or "").strip() if hba1c_col else ""
        out[subject_id] = (gender, hba1c)
    return out


def _format_food_line(row: dict[str, object], logged_col: str | None, searched_col: str | None, amount_col: str | None, unit_col: str | None) -> str:
    name = ""
    if logged_col:
        name = str(row.get(logged_col) or "").strip()
    if not name and searched_col:
        name = str(row.get(searched_col) or "").strip()
    if not name:
        return ""
    amount = str(row.get(amount_col) or "").strip() if amount_col else ""
    unit = str(row.get(unit_col) or "").strip() if unit_col else ""
    extras = " ".join(part for part in (amount, unit) if part and part.lower() not in {"none", "nan"})
    if extras:
        return f"{name} ({extras})"
    return name


def _food_events(food_df: pl.DataFrame) -> list[dict[str, object]]:
    time_col = _column(food_df, _FOOD_TIME_ALIASES)
    if time_col is None:
        date_col = _column(food_df, ("date",))
        clock_col = _column(food_df, ("time", "time_of_day"))
        if date_col is None or clock_col is None:
            return []
        combined = (
            food_df.get_column(date_col).cast(pl.Utf8, strict=False).str.strip_chars()
            + " "
            + food_df.get_column(clock_col).cast(pl.Utf8, strict=False).str.strip_chars()
        )
        times = _parse_datetime_series(combined)
    else:
        times = _parse_datetime_series(food_df.get_column(time_col))

    working = food_df.with_columns(times.alias("time")).filter(pl.col("time").is_not_null()).sort("time")
    logged_col = _column(working, _LOGGED_FOOD_ALIASES)
    searched_col = _column(working, _SEARCHED_FOOD_ALIASES)
    amount_col = _column(working, _AMOUNT_ALIASES)
    unit_col = _column(working, _UNIT_ALIASES)
    carbs_col = _column(working, _CARB_ALIASES)

    clusters: list[list[dict[str, object]]] = []
    for row in working.iter_rows(named=True):
        if not clusters:
            clusters.append([row])
            continue
        previous = clusters[-1][-1]["time"]
        current = row["time"]
        if current - previous <= _FOOD_CLUSTER_GAP:
            clusters[-1].append(row)
        else:
            clusters.append([row])

    events: list[dict[str, object]] = []
    for cluster in clusters:
        lines = [
            line
            for line in (
                _format_food_line(row, logged_col, searched_col, amount_col, unit_col)
                for row in cluster
            )
            if line
        ]
        if not lines:
            continue
        carbs_total = 0.0
        carbs_seen = False
        if carbs_col is not None:
            for row in cluster:
                raw = row.get(carbs_col)
                if raw in (None, ""):
                    continue
                carbs_total += float(raw)
                carbs_seen = True
        events.append(
            {
                "time": cluster[0]["time"],
                "event_type": "Carbohydrates",
                "event_subtype": "Carbs",
                "insulin_value": None,
                "photo_path": "",
                "meal_type": lines[0],
                "carbs_g": carbs_total if carbs_seen else None,
                "food_note": "\n".join(lines),
            }
        )
    return events


def format_bigideas_frames(
    glucose_df: pl.DataFrame,
    *,
    food_df: pl.DataFrame | None = None,
    subject_id: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Format BIG IDEAs Dexcom + food-log tables into app frames."""
    time_col = _column(glucose_df, _TIMESTAMP_ALIASES)
    glucose_col = _column(glucose_df, _GLUCOSE_ALIASES)
    if time_col is None or glucose_col is None:
        raise ValueError("BIG IDEAs Dexcom CSV is missing Timestamp / glucose columns")

    working = glucose_df.with_columns(_parse_datetime_series(glucose_df.get_column(time_col)).alias("time"))
    type_col = _column(working, _EVENT_TYPE_ALIASES)
    if type_col is not None:
        type_text = _text_expr(type_col).str.to_lowercase()
        working = working.filter(type_text.is_in(["egv", "glucose", ""]))
    working = working.with_columns(_numeric_expr(glucose_col).alias("gl"))
    filtered = working.filter(pl.col("time").is_not_null() & pl.col("gl").is_not_null()).sort("time")
    glucose_out = pl.DataFrame(
        {
            "time": filtered.get_column("time"),
            "gl": filtered.get_column("gl"),
            "prediction": [0.0] * filtered.height,
            "age": [0] * filtered.height,
            "user_id": [int(subject_id)] * filtered.height,
        }
    )

    events = _food_events(food_df) if food_df is not None and food_df.height > 0 else []
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
            "food_note": pl.String,
        },
    ).sort("time")
    return glucose_out, events_df


def discover_bigideas_sources(dest: Optional[Path] = None) -> list[BigIdeasSource]:
    """Discover BIG IDEAs Dexcom tables under ``data/bigideas/`` (or *dest*)."""
    root = dataset_root(dest)
    if not dataset_is_present(root) and not root.is_dir():
        return []

    demographics = load_bigideas_demographics(root)
    sources: list[BigIdeasSource] = []
    seen: set[str] = set()
    for dexcom_path in sorted(root.rglob("Dexcom_*.csv")):
        subject_id = subject_id_from_path(dexcom_path)
        if subject_id is None or subject_id in seen:
            continue
        seen.add(subject_id)
        gender, hba1c = demographics.get(subject_id, ("", ""))
        sources.append(
            BigIdeasSource(
                subject_id=subject_id,
                source_name=f"BIGIDEAS-{subject_id}.csv",
                csv_path=dexcom_path,
                subject_dir=dexcom_path.parent,
                age_years=None,
                gender=gender,
                weight="",
                sensor="Dexcom G6",
                hba1c=hba1c,
            )
        )
    return sources


def load_bigideas_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one BIG IDEAs participant into the app store schema."""
    path = Path(file_path)
    with start_action(action_type=u"load_bigideas_data", file_path=str(path)) as action:
        dexcom_path = path
        if not dexcom_path.is_file() and is_bigideas_source_name(path.name):
            subject_id = subject_id_from_path(path)
            for source in discover_bigideas_sources():
                if source.subject_id == subject_id:
                    dexcom_path = source.csv_path
                    break
        if not dexcom_path.is_file():
            raise FileNotFoundError(f"BIG IDEAs Dexcom CSV not found for {path}")

        subject_dir = dexcom_path.parent
        subject_token = subject_id_from_path(dexcom_path) or "001"
        raw_glucose = pl.read_csv(
            dexcom_path,
            skip_rows=_header_row_index(dexcom_path),
            infer_schema_length=5000,
        )
        food_path = subject_dir / f"Food_Log_{subject_token}.csv"
        food_df = pl.read_csv(food_path, infer_schema_length=2000) if food_path.is_file() else None
        glucose_df, events_df = format_bigideas_frames(
            raw_glucose,
            food_df=food_df,
            subject_id=int(subject_token),
        )
        action.add_success_fields(
            glucose_rows=glucose_df.height,
            event_rows=events_df.height,
            food_notes=events_df.filter(pl.col("food_note") != "").height,
        )
        return glucose_df, events_df
