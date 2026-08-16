import base64
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import polars as pl
from cgm_format import (
    ExtendedFormatProcessor,
    FormatParser,
    FormatProcessor,
    UnifiedEventType,
)
from eliot import start_action

from sugar_sugar.bigideas import is_bigideas_path, load_bigideas_data
from sugar_sugar.cgmacros import (
    is_cgmacros_csv,
    resolve_photo_path as resolve_cgmacros_photo,
    subject_id_from_path as cgmacros_subject_id,
)
from sugar_sugar.d1namo import (
    discover_d1namo_sources,
    is_d1namo_path,
    is_d1namo_source_name,
    resolve_photo_path as resolve_d1namo_photo,
    subject_id_from_path as d1namo_subject_id,
)


def decode_upload_bytes(payload: Optional[str]) -> Optional[bytes]:
    """Decode an upload payload to raw file bytes.

    Uploads are gzip-compressed client-side (``"gzip:<base64>"``) so the ~3.3 MB
    base64 of a multi-MB CGM export never has to cross the wire from the phone --
    mobile browsers reliably fail to POST a payload that large, which is why a big
    Dexcom export uploaded fine on desktop but silently failed on mobile (server
    parsing was never the problem). Falls back to a plain data URL
    (``"<mime>,<base64>"``) for desktop/older browsers without CompressionStream.
    Returns None if the payload is unrecognisable.
    """
    if not payload:
        return None
    try:
        if payload.startswith("gzip:"):
            return gzip.decompress(base64.b64decode(payload[5:]))
        if "," in payload:
            return base64.b64decode(payload.split(",", 1)[1])
    except Exception:
        return None
    return None

_RENDERED_EVENT_TYPES: tuple[str, ...] = (
    UnifiedEventType.CARBOHYDRATES.value,
    UnifiedEventType.INSULIN_FAST.value,
    UnifiedEventType.INSULIN_SLOW.value,
    UnifiedEventType.EXERCISE_LIGHT.value,
    UnifiedEventType.EXERCISE_MEDIUM.value,
    UnifiedEventType.EXERCISE_HEAVY.value,
)

_PHOTO_ANNOTATION_KEYS: tuple[str, ...] = ("picture", "image_path", "image")
_MEAL_ANNOTATION_KEYS: tuple[str, ...] = ("description", "meal_type_raw", "meal_type")
_NOTE_ANNOTATION_KEYS: tuple[str, ...] = ("description", "food_note", "note")


def load_glucose_data_from_nightscout(
    base_url: str,
    *,
    token: Optional[str] = None,
    api_secret: Optional[str] = None,
    days: Optional[int] = None,
    save_dir: Path = Path("data/input/users"),
) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    """Fetch CGM data from a Nightscout server and adapt to the app store schema.

    Downloads entries and treatments via the Nightscout REST API, serialises the
    unified DataFrame to a timestamped CSV under *save_dir* (so subsequent rounds
    can reload it via ``load_glucose_data``), and returns the adapted frames.

    Returns:
        (glucose_df, events_df, save_path)
    """
    with start_action(action_type=u"load_glucose_data_from_nightscout", base_url=base_url):
        unified_df = FormatParser.from_nightscout_url(
            base_url, token=token, api_secret=api_secret, days=days
        )
        glucose_df, events_df = FormatProcessor.split_glucose_events(unified_df)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{timestamp}_nightscout.csv"
        FormatParser.to_csv_file(unified_df, str(save_path))
        return _adapt_glucose_df(glucose_df), _adapt_events_df(events_df), save_path


def load_glucose_data(file_path: Path = Path("data/example.csv")) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data through cgm-format and adapt it to the app store schema."""
    with start_action(action_type=u"load_glucose_data", file_path=str(file_path)):
        if _is_loop_chronological_csv(file_path):
            return load_loop_chronological_data(file_path)
        if is_d1namo_path(file_path):
            return _load_d1namo_via_library(file_path)
        if is_cgmacros_csv(file_path):
            return _load_cgmacros_via_library(file_path)
        if is_bigideas_path(file_path):
            return load_bigideas_data(file_path)
        unified_df = FormatParser.parse_file(file_path)
        glucose_df, events_df = FormatProcessor.split_glucose_events(unified_df)
        return _adapt_glucose_df(glucose_df), _adapt_events_df(events_df)


def _is_loop_chronological_csv(file_path: Path) -> bool:
    name = file_path.name.lower()
    if name.endswith("_chronological.csv"):
        return True
    if not file_path.exists():
        return False
    header = file_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if not header:
        return False
    return "Glucose (mg/dL)" in header[0] and "Recommended Split" in header[0]


def _non_empty_str(column: str) -> pl.Expr:
    as_text = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.col(column).is_not_null() & (as_text != "")


def _parse_loop_numeric(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars().cast(pl.Float64, strict=False)


def load_loop_chronological_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load LOOP study chronological CSV exports into the app store schema."""
    with start_action(action_type=u"load_loop_chronological_data", file_path=str(file_path)):
        raw_df = pl.read_csv(file_path, infer_schema_length=10000)
        time_expr = pl.col("Timestamp").str.to_datetime(strict=False)

        glucose_df = (
            raw_df.filter(pl.col("Event Type").is_in(["EGV", "Interpolated"]))
            .filter(_non_empty_str("Glucose (mg/dL)"))
            .select(
                [
                    time_expr.alias("time"),
                    _parse_loop_numeric("Glucose (mg/dL)").alias("gl"),
                    pl.lit(0.0).alias("prediction"),
                    pl.lit(0).alias("age"),
                    pl.lit(1).alias("user_id"),
                ]
            )
            .filter(pl.col("time").is_not_null() & pl.col("gl").is_not_null())
            .sort("time")
        )

        carb_events = (
            raw_df.filter(_non_empty_str("Carbohydrates (g)"))
            .select(
                [
                    time_expr.alias("time"),
                    pl.lit("Carbohydrates").alias("event_type"),
                    pl.lit("Carbs").alias("event_subtype"),
                    pl.lit(None, dtype=pl.Float64).alias("insulin_value"),
                ]
            )
            .filter(pl.col("time").is_not_null())
        )
        bolus_events = (
            raw_df.filter(_non_empty_str("Bolus Insulin (U)"))
            .select(
                [
                    time_expr.alias("time"),
                    pl.lit("Insulin").alias("event_type"),
                    pl.lit("Fast Acting").alias("event_subtype"),
                    _parse_loop_numeric("Bolus Insulin (U)").alias("insulin_value"),
                ]
            )
            .filter(
                pl.col("time").is_not_null()
                & pl.col("insulin_value").is_not_null()
                & (pl.col("insulin_value") != 0)
            )
        )
        basal_events = (
            raw_df.filter(_non_empty_str("Basal Rate (U/h)"))
            .select(
                [
                    time_expr.alias("time"),
                    pl.lit("Insulin").alias("event_type"),
                    pl.lit("Long Acting").alias("event_subtype"),
                    _parse_loop_numeric("Basal Rate (U/h)").alias("insulin_value"),
                ]
            )
            .filter(
                pl.col("time").is_not_null()
                & pl.col("insulin_value").is_not_null()
                & (pl.col("insulin_value") != 0)
            )
        )

        events_df = pl.concat([carb_events, bolus_events, basal_events], how="vertical").sort("time")
        return glucose_df, events_df


def _resolve_d1namo_subject_dir(file_path: Path) -> Path:
    path = Path(file_path)
    glucose_path = path if path.name.lower() == "glucose.csv" else path.parent / "glucose.csv"
    if not glucose_path.is_file() and is_d1namo_source_name(path.name):
        subject_id = d1namo_subject_id(path)
        for source in discover_d1namo_sources():
            if source.subject_id == subject_id:
                return source.subject_dir
    if glucose_path.is_file():
        return glucose_path.parent
    raise FileNotFoundError(f"D1NAMO glucose.csv not found for {path}")


def _load_d1namo_via_library(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    subject_dir = _resolve_d1namo_subject_dir(file_path)
    subject_token = d1namo_subject_id(subject_dir) or "001"
    with start_action(action_type=u"load_d1namo_via_library", file_path=str(subject_dir)) as action:
        unified_df = FormatParser.parse_subject_directory(subject_dir)
        glucose_df, events_df = ExtendedFormatProcessor.split_glucose_events(unified_df)
        adapted_glucose = _adapt_glucose_df(glucose_df, user_id=int(subject_token))
        adapted_events = _adapt_events_df(
            events_df,
            subject_dir=subject_dir,
            photo_resolver=resolve_d1namo_photo,
        )
        action.add_success_fields(
            glucose_rows=adapted_glucose.height,
            event_rows=adapted_events.height,
        )
        return adapted_glucose, adapted_events


def _pick_cgmacros_track(tracks: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Prefer Dexcom when it has readings; otherwise Libre; otherwise first track."""
    ordered: list[pl.DataFrame] = []
    for name in ("dexcom", "libre"):
        frame = tracks.get(name)
        if frame is not None:
            ordered.append(frame)
    for name, frame in tracks.items():
        if name not in {"dexcom", "libre"}:
            ordered.append(frame)
    if not ordered:
        raise ValueError("CGMacros parse_tracks returned no frames")

    def _glucose_rows(frame: pl.DataFrame) -> int:
        if "glucose" not in frame.columns:
            return 0
        return frame.filter(pl.col("glucose").is_not_null()).height

    return next((frame for frame in ordered if _glucose_rows(frame) > 0), ordered[0])


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


def _load_cgmacros_via_library(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    path = Path(file_path)
    subject_id = cgmacros_subject_id(path) or 1
    with start_action(action_type=u"load_cgmacros_via_library", file_path=str(path)) as action:
        tracks = FormatParser.parse_tracks(path)
        unified_df = _pick_cgmacros_track(tracks)
        glucose_df, events_df = ExtendedFormatProcessor.split_glucose_events(unified_df)
        adapted_glucose = _downsample_glucose_5min(
            _adapt_glucose_df(glucose_df, user_id=int(subject_id))
        )
        adapted_events = _adapt_events_df(
            events_df,
            subject_dir=path.parent,
            photo_resolver=resolve_cgmacros_photo,
        )
        action.add_success_fields(
            glucose_rows=adapted_glucose.height,
            event_rows=adapted_events.height,
        )
        return adapted_glucose, adapted_events


def _adapt_glucose_df(glucose_df: pl.DataFrame, *, user_id: int = 1) -> pl.DataFrame:
    return (
        glucose_df.filter(pl.col("datetime").is_not_null() & pl.col("glucose").is_not_null())
        .select(
            [
                pl.col("datetime").alias("time"),
                pl.col("glucose").alias("gl"),
                pl.lit(0.0).alias("prediction"),
                pl.lit(0).alias("age"),
                pl.lit(int(user_id)).alias("user_id"),
            ]
        )
        .sort("time")
    )


def _annotation_mapping(raw: object) -> dict[str, object]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _annotation_text(mapping: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", "null"):
            continue
        text = str(value).strip()
        if text and text.lower() != "null":
            return text
    return ""


def _adapt_events_df(
    events_df: pl.DataFrame,
    *,
    subject_dir: Path | None = None,
    photo_resolver: Callable[[str, Path], str] | None = None,
) -> pl.DataFrame:
    annotations = (
        events_df.get_column("annotations").to_list()
        if "annotations" in events_df.columns
        else [None] * events_df.height
    )
    mappings = [_annotation_mapping(raw) for raw in annotations]
    photos: list[str] = []
    meal_types: list[str] = []
    food_notes: list[str] = []
    for mapping in mappings:
        photo = _annotation_text(mapping, _PHOTO_ANNOTATION_KEYS)
        if photo and subject_dir is not None and photo_resolver is not None:
            photo = photo_resolver(photo, subject_dir)
        photos.append(photo)
        meal_types.append(_annotation_text(mapping, _MEAL_ANNOTATION_KEYS))
        food_notes.append(_annotation_text(mapping, _NOTE_ANNOTATION_KEYS))

    carbs = (
        events_df.get_column("carbs").to_list()
        if "carbs" in events_df.columns
        else [None] * events_df.height
    )
    working = events_df.with_columns(
        [
            pl.Series("photo_path", photos, dtype=pl.String),
            pl.Series("meal_type", meal_types, dtype=pl.String),
            pl.Series("food_note", food_notes, dtype=pl.String),
            pl.Series("carbs_g", carbs, dtype=pl.Float64),
        ]
    )
    rendered = set(_RENDERED_EVENT_TYPES)
    keep = [
        (
            str(row.get("event_type") or "") in rendered
            or bool(str(row.get("photo_path") or "").strip())
            or bool(str(row.get("food_note") or "").strip())
        )
        for row in working.iter_rows(named=True)
    ]
    working = working.with_columns(pl.Series("_keep", keep, dtype=pl.Boolean)).filter(pl.col("_keep"))

    insulin_fast = (
        pl.col("insulin_fast")
        if "insulin_fast" in working.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    insulin_slow = (
        pl.col("insulin_slow")
        if "insulin_slow" in working.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    adapted = (
        working.filter(pl.col("datetime").is_not_null())
        .select(
            [
                pl.col("datetime").alias("time"),
                _legacy_event_type_expr().alias("event_type"),
                _legacy_event_subtype_expr().alias("event_subtype"),
                pl.coalesce([insulin_fast, insulin_slow])
                .cast(pl.Float64, strict=False)
                .alias("insulin_value"),
                pl.col("photo_path"),
                pl.col("meal_type"),
                pl.col("carbs_g"),
                pl.col("food_note"),
            ]
        )
        .filter(
            (pl.col("event_type") != "Insulin")
            | (pl.col("insulin_value").is_not_null() & (pl.col("insulin_value") != 0))
        )
        .sort("time")
    )
    has_extras = adapted.height > 0 and (
        adapted.filter(
            (pl.col("photo_path") != "")
            | (pl.col("food_note") != "")
            | (pl.col("meal_type") != "")
        ).height
        > 0
    )
    if has_extras:
        return adapted.with_columns(
            pl.when(pl.col("event_type") == "")
            .then(pl.lit("Carbohydrates"))
            .otherwise(pl.col("event_type"))
            .alias("event_type")
        )
    return adapted.select(["time", "event_type", "event_subtype", "insulin_value"])


def _legacy_event_type_expr() -> pl.Expr:
    event_type = pl.col("event_type")
    insulin_events = [UnifiedEventType.INSULIN_FAST.value, UnifiedEventType.INSULIN_SLOW.value]
    exercise_events = [
        UnifiedEventType.EXERCISE_LIGHT.value,
        UnifiedEventType.EXERCISE_MEDIUM.value,
        UnifiedEventType.EXERCISE_HEAVY.value,
    ]
    return (
        pl.when(event_type == UnifiedEventType.CARBOHYDRATES.value)
        .then(pl.lit("Carbohydrates"))
        .when(event_type.is_in(insulin_events))
        .then(pl.lit("Insulin"))
        .when(event_type.is_in(exercise_events))
        .then(pl.lit("Exercise"))
        .otherwise(pl.lit(""))
    )


def _legacy_event_subtype_expr() -> pl.Expr:
    event_type = pl.col("event_type")
    return (
        pl.when(event_type == UnifiedEventType.CARBOHYDRATES.value)
        .then(pl.lit("Carbs"))
        .when(event_type == UnifiedEventType.INSULIN_FAST.value)
        .then(pl.lit("Fast Acting"))
        .when(event_type == UnifiedEventType.INSULIN_SLOW.value)
        .then(pl.lit("Long Acting"))
        .when(event_type == UnifiedEventType.EXERCISE_LIGHT.value)
        .then(pl.lit("Light"))
        .when(event_type == UnifiedEventType.EXERCISE_MEDIUM.value)
        .then(pl.lit("Medium"))
        .when(event_type == UnifiedEventType.EXERCISE_HEAVY.value)
        .then(pl.lit("Heavy"))
        .otherwise(pl.lit(""))
    )
