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

from dash import no_update

from sugar_sugar.app import (
    compacted_events_store,
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


def test_compacting_shrinks_a_whole_subject_store_to_the_window() -> None:
    """The localStorage-side twin of events_within_window, run on navigation."""
    events = events_dataframe_to_store_dict(make_events(500))
    window = {"time": make_window(offset_steps=100, points=36)
              .get_column("time").dt.strftime("%Y-%m-%dT%H:%M:%S").to_list()}

    compacted = compacted_events_store(events, window)

    assert compacted is not no_update
    assert len(compacted["time"]) == 36
    assert compacted["time"][0] == window["time"][0]
    assert compacted["time"][-1] == window["time"][-1]
    # Every column is trimmed in step, not just `time`.
    assert {len(values) for values in compacted.values()} == {36}


def test_compacting_no_ops_when_there_is_nothing_to_trim() -> None:
    """`no_update` keeps an already-small store from being rewritten every navigation."""
    window_df = make_window(offset_steps=0, points=36)
    window = {"time": window_df.get_column("time").dt.strftime("%Y-%m-%dT%H:%M:%S").to_list()}
    already_trimmed = events_store_for_window(make_events(500), window_df)

    assert compacted_events_store(already_trimmed, window) is no_update
    assert compacted_events_store(None, window) is no_update
    assert compacted_events_store({}, window) is no_update
    assert compacted_events_store(already_trimmed, None) is no_update
    assert compacted_events_store(already_trimmed, {"time": []}) is no_update


def test_compacting_refuses_a_ragged_store() -> None:
    """Mismatched column lengths would index out of range -- leave the store alone."""
    ragged = {"time": ["2026-01-01T00:00:00", "2026-01-01T00:05:00"], "event_type": ["Carbohydrates"]}
    assert compacted_events_store(ragged, {"time": ["2026-01-01T00:00:00"]}) is no_update


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
