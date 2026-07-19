import base64
import gzip
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from cgm_format import FormatParser, FormatProcessor, UnifiedEventType
from eliot import start_action


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
            .filter(pl.col("time").is_not_null() & pl.col("insulin_value").is_not_null())
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
            .filter(pl.col("time").is_not_null() & pl.col("insulin_value").is_not_null())
        )

        events_df = pl.concat([carb_events, bolus_events, basal_events], how="vertical").sort("time")
        return glucose_df, events_df


def _adapt_glucose_df(glucose_df: pl.DataFrame) -> pl.DataFrame:
    return (
        glucose_df.filter(pl.col("datetime").is_not_null() & pl.col("glucose").is_not_null())
        .select(
            [
                pl.col("datetime").alias("time"),
                pl.col("glucose").alias("gl"),
                pl.lit(0.0).alias("prediction"),
                pl.lit(0).alias("age"),
                pl.lit(1).alias("user_id"),
            ]
        )
        .sort("time")
    )


def _adapt_events_df(events_df: pl.DataFrame) -> pl.DataFrame:
    return (
        events_df.filter(
            pl.col("datetime").is_not_null() & pl.col("event_type").is_in(_RENDERED_EVENT_TYPES)
        )
        .select(
            [
                pl.col("datetime").alias("time"),
                _legacy_event_type_expr().alias("event_type"),
                _legacy_event_subtype_expr().alias("event_subtype"),
                pl.coalesce([pl.col("insulin_fast"), pl.col("insulin_slow")])
                .cast(pl.Float64, strict=False)
                .alias("insulin_value"),
            ]
        )
        .sort("time")
    )


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
