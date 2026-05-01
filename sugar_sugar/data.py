from pathlib import Path

import polars as pl
from cgm_format import FormatParser, FormatProcessor, UnifiedEventType
from eliot import start_action

_RENDERED_EVENT_TYPES: tuple[str, ...] = (
    UnifiedEventType.CARBOHYDRATES.value,
    UnifiedEventType.INSULIN_FAST.value,
    UnifiedEventType.INSULIN_SLOW.value,
    UnifiedEventType.EXERCISE_LIGHT.value,
    UnifiedEventType.EXERCISE_MEDIUM.value,
    UnifiedEventType.EXERCISE_HEAVY.value,
)


def load_glucose_data(file_path: Path = Path("data/example.csv")) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data through cgm-format and adapt it to the app store schema."""
    with start_action(action_type=u"load_glucose_data", file_path=str(file_path)):
        unified_df = FormatParser.parse_file(file_path)
        glucose_df, events_df = FormatProcessor.split_glucose_events(unified_df)
        return _adapt_glucose_df(glucose_df), _adapt_events_df(events_df)


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
