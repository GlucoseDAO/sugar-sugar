from __future__ import annotations

from datetime import datetime, time

import polars as pl

# Clock-time bands where participants would normally expect meal carb logging.
MEAL_WINDOWS: tuple[tuple[time, time], ...] = (
    (time(hour=6), time(hour=10)),   # breakfast
    (time(hour=11), time(hour=14)),  # lunch
    (time(hour=17), time(hour=21)),  # dinner
)


def _clock_in_meal_window(clock: time) -> bool:
    for start, end in MEAL_WINDOWS:
        if start <= clock < end:
            return True
    return False


def window_covers_meal_time(window_df: pl.DataFrame) -> bool:
    if window_df.height == 0 or "time" not in window_df.columns:
        return False
    for dt in window_df.get_column("time"):
        if isinstance(dt, datetime) and _clock_in_meal_window(dt.time()):
            return True
    return False


def window_has_carb_events(window_df: pl.DataFrame, events_df: pl.DataFrame) -> bool:
    if window_df.height == 0 or events_df.height == 0:
        return False
    if "time" not in window_df.columns or "time" not in events_df.columns:
        return False

    start_time = window_df.get_column("time")[0]
    end_time = window_df.get_column("time")[-1]
    carbs = events_df.filter(
        (pl.col("time") >= start_time)
        & (pl.col("time") <= end_time)
        & (pl.col("event_type") == "Carbohydrates")
    )
    return carbs.height > 0


def should_show_no_carbs_note(window_df: pl.DataFrame, events_df: pl.DataFrame) -> bool:
    """True when the visible window spans a meal period but has no carb events."""
    return window_covers_meal_time(window_df) and not window_has_carb_events(window_df, events_df)
