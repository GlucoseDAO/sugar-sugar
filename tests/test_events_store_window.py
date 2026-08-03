"""The `events-df` store must never carry more than the played window.

It is a localStorage store that the browser re-uploads with every callback
request reading it, so shipping a whole subject's event log (62k rows / 3.4 MB
for loop_467) made each click take seconds in production (2026-07-28).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from sugar_sugar.app import (
    events_dataframe_to_store_dict,
    events_store_for_window,
    events_within_window,
)
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.data import load_glucose_data

BIG_SUBJECT = Path("data/subjects/loop_467/loop_467_chronological.csv")


def make_events(count: int, *, step_minutes: int = 5) -> pl.DataFrame:
    start = datetime(2026, 1, 1, 0, 0)
    return pl.DataFrame({
        "time": [start + timedelta(minutes=step_minutes * i) for i in range(count)],
        "event_type": ["Carbohydrates"] * count,
        "event_subtype": [""] * count,
        "insulin_value": [None] * count,
    }, schema_overrides={"insulin_value": pl.Float64})


def make_window(offset_steps: int, points: int = 36) -> pl.DataFrame:
    start = datetime(2026, 1, 1, 0, 0) + timedelta(minutes=5 * offset_steps)
    times = [start + timedelta(minutes=5 * i) for i in range(points)]
    return pl.DataFrame({
        "time": times,
        "gl": [120.0] * points,
        "prediction": [0.0] * points,
        "age": [30] * points,
        "user_id": [1] * points,
    })


def test_events_are_trimmed_to_the_window_span() -> None:
    events = make_events(500)
    window = make_window(offset_steps=100, points=36)

    trimmed = events_within_window(events, window)

    assert trimmed.height == 36  # one event per 5-min slot, inclusive of both ends
    assert trimmed.get_column("time")[0] == window.get_column("time")[0]
    assert trimmed.get_column("time")[-1] == window.get_column("time")[-1]


def test_empty_inputs_do_not_blow_up() -> None:
    events = make_events(10)
    assert events_within_window(events, make_window(0).clear()).height == 0
    assert events_within_window(events.clear(), make_window(0)).height == 0
    assert events_store_for_window(events.clear(), make_window(0))["time"] == []


@pytest.mark.skipif(not BIG_SUBJECT.exists(), reason="generic subject data not present")
def test_biggest_generic_source_store_stays_small() -> None:
    glucose_df, events_df = load_glucose_data(BIG_SUBJECT)
    window = glucose_df.with_columns(pl.lit(0.0).alias("prediction")).slice(0, 36)

    whole_subject = json.dumps(events_dataframe_to_store_dict(events_df))
    windowed = json.dumps(events_store_for_window(events_df, window))

    assert len(whole_subject) > 1_000_000, "fixture no longer exercises the big case"
    assert len(windowed) < 50_000


@pytest.mark.skipif(not BIG_SUBJECT.exists(), reason="generic subject data not present")
def test_trimming_does_not_change_what_the_chart_draws() -> None:
    glucose_df, events_df = load_glucose_data(BIG_SUBJECT)
    glucose_df = glucose_df.with_columns(pl.lit(0.0).alias("prediction"))

    window = next(
        w for w in (glucose_df.slice(s, 36) for s in range(0, 4000, 36))
        if events_within_window(events_df, w).height > 3
    )

    def rendered(store: dict[str, list]) -> tuple[int, int]:
        chart = GlucoseChart.__new__(GlucoseChart)
        chart.hide_last_hour = True
        chart._display_unit = "mg/dL"
        chart._display_factor = 1.0
        events = chart._reconstruct_events_dataframe_from_dict(store)
        figure = chart._build_figure(window, events, "loop_467", locale="en")
        return (
            len(figure.layout.images),
            sum(len(trace.x) for trace in figure.data if trace.mode == "markers"),
        )

    assert rendered(events_store_for_window(events_df, window)) == rendered(
        events_dataframe_to_store_dict(events_df)
    )
