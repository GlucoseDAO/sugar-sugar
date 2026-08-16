"""The single boundary between ``cgm-format``'s unified frames and the app's stores.

Every loader in the app converges on one pair of frames:

* glucose -- ``(time, gl, prediction, age, user_id)``
* events  -- ``(time, event_type, event_subtype, insulin_value, photo_path, meal_type, carbs_g[, food_note])``

``cgm-format`` speaks a different language: a single unified frame carrying
``datetime``/``glucose``/``carbs``/``insulin_fast``/``insulin_slow``, eight-character
``UnifiedEventType`` codes, and -- for the research corpora -- a JSON ``annotations``
column holding whatever the source said that the schema has no column for.
Translating between the two used to live in three near-identical copies
(``cgmacros.py``, ``d1namo.py``, ``data.py``); it lives here now.

This module must not import from ``sugar_sugar.data`` -- ``data`` imports the corpus
loaders, which import this. The legacy event-string mapping therefore lives here and
``data`` re-uses it, rather than the other way round.
"""

from __future__ import annotations

from typing import Callable, Optional, Type

import polars as pl
from cgm_format import (
    CGM_SCHEMA_EXTENDED,
    ExtendedFormatProcessor,
    FormatProcessor,
    UnifiedEventType,
)

#: Event types the chart actually draws. Everything else in the unified frame
#: (``CALIBRAT`` fingersticks, alerts, health markers) is dropped here rather
#: than filtered again downstream.
RENDERED_EVENT_TYPES: tuple[str, ...] = (
    UnifiedEventType.CARBOHYDRATES.value,
    UnifiedEventType.INSULIN_FAST.value,
    UnifiedEventType.INSULIN_SLOW.value,
    UnifiedEventType.EXERCISE_LIGHT.value,
    UnifiedEventType.EXERCISE_MEDIUM.value,
    UnifiedEventType.EXERCISE_HEAVY.value,
)

#: The app's events frame. ``food_note`` is deliberately absent: only BIG IDEAs
#: carries one, and the store writer in ``app.py`` adds the column only when
#: present.
#:
#: The millisecond time unit is not cosmetic -- it is what ``cgm-format`` emits
#: for ``datetime``, so pinning it here keeps a loader's glucose and events
#: frames in the same unit. ``events_within_window`` compares the two directly,
#: and a us/ms mismatch inside one loader is a bug waiting for a timezone-shaped
#: afternoon. (LOOP is parsed in-repo and produces microseconds throughout; it is
#: internally consistent too, which is what actually matters.)
CORE_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Datetime("ms"),
    "event_type": pl.String,
    "event_subtype": pl.String,
    "insulin_value": pl.Float64,
}

#: The corpus events frame: the core four plus the meal detail a research
#: corpus carries and a vendor export does not.
EVENTS_SCHEMA: dict[str, pl.DataType] = {
    **CORE_EVENTS_SCHEMA,
    "photo_path": pl.String,
    "meal_type": pl.String,
    "carbs_g": pl.Float64,
}

#: BIG IDEAs' events frame: the corpus seven plus the text note that stands in
#: for the meal photograph it does not have. ``bigideas.py`` is the only writer;
#: the apple icon opens this note in a notepad instead of a lightbox.
FOOD_NOTE_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    **EVENTS_SCHEMA,
    "food_note": pl.String,
}

#: Resolves a source-reported photo reference to a subject-relative posix path.
#: ``d1namo.resolve_photo_path`` / ``cgmacros.resolve_photo_path`` bound to a
#: subject directory. The library only carries the reference; resolving and
#: serving it stays in the app.
PhotoResolver = Callable[[str], str]


def unified_processor(unified_df: pl.DataFrame) -> Type[FormatProcessor]:
    """Pick the processor whose schema matches the frame's shape.

    ``FormatProcessor.schema`` is the ten-column ``CGM_SCHEMA``; the research
    corpora (and any uploaded unified-extended CSV) target the twenty-two-column
    ``CGM_SCHEMA_EXTENDED`` and die against it with "Schema has 10 columns,
    dataframe has 22 columns". Dispatch on the frame in front of us, mirroring
    ``cgm_format.cgm_cli._processor_for``.
    """
    if tuple(unified_df.columns) == tuple(CGM_SCHEMA_EXTENDED.get_column_names(data_only=False)):
        return ExtendedFormatProcessor
    return FormatProcessor


def legacy_event_type_expr() -> pl.Expr:
    """``UnifiedEventType`` -> the Dexcom-ish strings the chart filters on."""
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


def legacy_event_subtype_expr() -> pl.Expr:
    """``UnifiedEventType`` -> the legacy event subtype strings."""
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


def empty_events_frame() -> pl.DataFrame:
    """A zero-row events frame with the app's schema."""
    return pl.DataFrame(schema=EVENTS_SCHEMA)


def annotation_field(*paths: str) -> pl.Expr:
    """First non-null value among several JSON paths of the ``annotations`` column.

    The corpora name the same thing differently -- a meal's label is
    ``meal_type_raw`` in CGMacros and ``description`` in D1NAMO, its photo is
    ``image_path`` there and ``picture`` here -- so the adapter coalesces across
    every vocabulary instead of branching on which corpus produced the frame.
    Empty string, never null: a missing annotation and an annotation the source
    left blank say the same thing to a reader, and downstream text expressions
    would propagate the null through every concatenation.
    """
    annotations = pl.col("annotations")
    return pl.coalesce(
        [annotations.str.json_path_match(f"$.{path}") for path in paths]
    ).fill_null("")


def adapt_glucose_df(glucose_df: pl.DataFrame, *, subject_id: int = 1) -> pl.DataFrame:
    """Unified glucose rows -> the app's glucose store schema."""
    return (
        glucose_df.filter(pl.col("datetime").is_not_null() & pl.col("glucose").is_not_null())
        .select(
            [
                pl.col("datetime").alias("time"),
                pl.col("glucose").alias("gl"),
                pl.lit(0.0).alias("prediction"),
                pl.lit(0).alias("age"),
                pl.lit(int(subject_id)).alias("user_id"),
            ]
        )
        .sort("time")
    )


def adapt_events_df(
    events_df: pl.DataFrame,
    *,
    photo_resolver: Optional[PhotoResolver] = None,
) -> pl.DataFrame:
    """Unified event rows -> the app's events store schema.

    Meal detail that the six core data columns cannot hold -- the photo
    reference, the human label -- arrives in the extended schema's JSON
    ``annotations`` column, and is unpacked here into ``photo_path`` /
    ``meal_type`` / ``carbs_g``.

    Those three columns appear **only** for a source that has annotations, i.e.
    a research corpus. A Dexcom or Libre export has no meal photographs and no
    meal labels, so giving it the wider frame would mean shipping an array of
    empty strings to the browser for every upload -- `events-df` is a
    ``storage_type='local'`` store that the client re-uploads with every
    callback, and that is exactly the padding the 2026-07-28 freeze was about.
    Four columns in, four columns out.
    """
    has_annotations = "annotations" in events_df.columns
    if has_annotations and events_df.height > 0:
        # Photo-only CGMacros rows (end-of-meal JPEGs, no macros) arrive as
        # OTHEREVT. Promote them so the chart still gets the apple.
        has_photo = (
            annotation_field("image_path", "picture").fill_null("").str.strip_chars() != ""
        )
        events_df = events_df.with_columns(
            pl.when(has_photo & ~pl.col("event_type").is_in(list(RENDERED_EVENT_TYPES)))
            .then(pl.lit(UnifiedEventType.CARBOHYDRATES.value))
            .otherwise(pl.col("event_type"))
            .alias("event_type")
        )
    base_columns = [
        pl.col("datetime").alias("time"),
        legacy_event_type_expr().alias("event_type"),
        legacy_event_subtype_expr().alias("event_subtype"),
        pl.coalesce([pl.col("insulin_fast"), pl.col("insulin_slow")])
        .cast(pl.Float64, strict=False)
        .alias("insulin_value"),
    ]
    meal_columns = (
        [
            annotation_field("image_path", "picture").alias("photo_path"),
            annotation_field("meal_type_raw", "description", "meal_type").alias("meal_type"),
            pl.col("carbs").cast(pl.Float64, strict=False).alias("carbs_g"),
        ]
        if has_annotations
        else []
    )

    adapted = (
        events_df.filter(
            pl.col("datetime").is_not_null() & pl.col("event_type").is_in(RENDERED_EVENT_TYPES)
        )
        .select(base_columns + meal_columns)
        # An insulin event with no dose is bookkeeping, not something to draw.
        .filter(
            (pl.col("event_type") != "Insulin")
            | (pl.col("insulin_value").is_not_null() & (pl.col("insulin_value") != 0))
        )
        .sort("time")
    )

    if has_annotations and photo_resolver is not None and adapted.height > 0:
        adapted = adapted.with_columns(
            pl.col("photo_path")
            .map_elements(
                lambda raw: photo_resolver(str(raw or "")),
                return_dtype=pl.String,
            )
            .alias("photo_path")
        )
    schema = EVENTS_SCHEMA if has_annotations else CORE_EVENTS_SCHEMA
    return adapted.cast(schema)  # type: ignore[arg-type]


def adapt_unified(
    unified_df: pl.DataFrame,
    *,
    subject_id: int = 1,
    photo_resolver: Optional[PhotoResolver] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a unified frame and adapt both halves to the app's store schemas."""
    processor = unified_processor(unified_df)
    glucose_df, events_df = processor.split_glucose_events(unified_df)
    return (
        adapt_glucose_df(glucose_df, subject_id=subject_id),
        adapt_events_df(events_df, photo_resolver=photo_resolver),
    )
