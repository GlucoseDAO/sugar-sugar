from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.config import PREDICTION_HOUR_OFFSET


def _window(points: int = 36) -> tuple[pl.DataFrame, pl.DataFrame]:
    start = datetime(2024, 6, 1, 19, 56)
    times = [start + timedelta(minutes=5 * i) for i in range(points)]
    glucose = [140.0 - i * 1.2 for i in range(points)]
    window = pl.DataFrame({
        "time": times,
        "gl": glucose,
        "prediction": [None] * (points - PREDICTION_HOUR_OFFSET) + glucose[-PREDICTION_HOUR_OFFSET:],
        "age": [40] * points,
        "user_id": [1] * points,
    })
    events = pl.DataFrame({
        "time": [times[8]],
        "event_type": ["Insulin"],
        "event_subtype": [""],
        "insulin_value": [2.0],
        "photo_path": [""],
        "food_note": [""],
    })
    return window, events


def test_compact_layout_shrinks_margins_and_hour_ticks() -> None:
    window, events = _window()
    desktop = GlucoseChart.build_static_figure(
        window, events, "example.csv", compact=False,
        prediction_boundary=len(window) - PREDICTION_HOUR_OFFSET,
    )
    mobile = GlucoseChart.build_static_figure(
        window, events, "example.csv", compact=True,
        prediction_boundary=len(window) - PREDICTION_HOUR_OFFSET,
    )

    assert mobile.layout.margin.t < desktop.layout.margin.t
    assert mobile.layout.margin.t <= 2
    assert desktop.layout.margin.t <= 8
    assert mobile.layout.margin.b == 40
    assert desktop.layout.margin.b == 36
    assert desktop.layout.xaxis.tickangle in (None, 0)
    assert mobile.layout.xaxis.automargin is False
    assert desktop.layout.xaxis.automargin is False
    assert mobile.layout.yaxis.automargin is False
    assert desktop.layout.legend.y <= 1.0
    assert mobile.layout.legend.y <= 1.0
    assert mobile.layout.xaxis.tickangle == -90
    assert len(mobile.layout.xaxis.tickvals) < len(desktop.layout.xaxis.tickvals)
    assert mobile.layout.xaxis.tickvals[0] == 0
    assert mobile.layout.xaxis.tickvals[-1] == len(window) - 1
    assert (mobile.layout.xaxis.title.text or "") == ""
    assert (desktop.layout.xaxis.title.text or "") == ""


def test_prediction_and_results_share_the_same_tight_margins() -> None:
    window, events = _window()
    results = GlucoseChart.build_static_figure(
        window, events, "example.csv", compact=True,
        prediction_boundary=len(window) - PREDICTION_HOUR_OFFSET,
    )
    # Same builder as /prediction: hide_last_hour only changes the left gutter
    # for the unit chip, not the vertical paper strips.
    live = GlucoseChart.__new__(GlucoseChart)
    live.hide_last_hour = True
    live._display_unit = "mg/dL"
    live._display_factor = 1.0
    pred_fig = live._build_figure(
        window, events, "example.csv", locale="en", compact=True,
    )

    assert results.layout.margin.t == pred_fig.layout.margin.t == 2
    assert results.layout.margin.b == pred_fig.layout.margin.b == 40
    assert (results.layout.xaxis.title.text or "") == ""
    assert results.layout.xaxis.automargin is False
    assert pred_fig.layout.xaxis.automargin is False


def test_live_builder_defaults_to_desktop_without_request() -> None:
    """/prediction update_chart_figure omits compact= so a desktop request
    keeps horizontal HH:MM ticks. No Flask request → desktop, not compact."""
    window, events = _window()
    live = GlucoseChart.__new__(GlucoseChart)
    live.hide_last_hour = True
    live._display_unit = "mg/dL"
    live._display_factor = 1.0
    fig = live._build_figure(window, events, "example.csv", locale="en")

    assert fig.layout.xaxis.tickangle in (None, 0)
    assert fig.layout.margin.b == 36
