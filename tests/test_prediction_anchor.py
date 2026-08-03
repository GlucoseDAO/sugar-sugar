from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from sugar_sugar.app import anchor_predictions_at_boundary
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.config import PREDICTION_HOUR_OFFSET

WINDOW_POINTS = 2 * PREDICTION_HOUR_OFFSET
BOUNDARY = WINDOW_POINTS - PREDICTION_HOUR_OFFSET


def make_window(first_drawn_offset: int) -> pl.DataFrame:
    """Window whose first drawn prediction sits ``first_drawn_offset`` slots
    after the known/hidden boundary (0 == drawn right on the boundary)."""
    start = datetime(2026, 1, 1, 8, 0)
    times = [start + timedelta(minutes=5 * i) for i in range(WINDOW_POINTS)]
    glucose = [100.0 + i for i in range(WINDOW_POINTS)]
    predictions = [0.0] * WINDOW_POINTS
    for i in range(BOUNDARY + first_drawn_offset, WINDOW_POINTS):
        predictions[i] = 150.0
    return pl.DataFrame({
        "time": times,
        "gl": glucose,
        "prediction": predictions,
        "age": [30] * WINDOW_POINTS,
        "user_id": [1] * WINDOW_POINTS,
    })


def test_anchor_fills_gap_between_known_line_and_first_drawn_point() -> None:
    df = make_window(first_drawn_offset=3)
    anchored = anchor_predictions_at_boundary(df).get_column("prediction").to_list()

    # Boundary slot takes the ground truth there, gap slots are interpolated.
    assert anchored[BOUNDARY] == df.get_column("gl")[BOUNDARY]
    assert all(v != 0.0 for v in anchored[BOUNDARY:])
    assert anchored[BOUNDARY + 1] < anchored[BOUNDARY + 2] < 150.0
    # Known region and the drawn values themselves are untouched.
    assert anchored[:BOUNDARY] == [0.0] * BOUNDARY
    assert anchored[BOUNDARY + 3] == 150.0


def test_anchor_is_a_noop_without_predictions_or_when_already_joined() -> None:
    empty = make_window(first_drawn_offset=0).with_columns(pl.lit(0.0).alias("prediction"))
    assert anchor_predictions_at_boundary(empty).get_column("prediction").to_list() == [0.0] * WINDOW_POINTS

    joined = make_window(first_drawn_offset=0)
    assert (
        anchor_predictions_at_boundary(joined).get_column("prediction").to_list()
        == joined.get_column("prediction").to_list()
    )


def test_anchor_is_idempotent() -> None:
    once = anchor_predictions_at_boundary(make_window(first_drawn_offset=4))
    twice = anchor_predictions_at_boundary(once)
    assert twice.get_column("prediction").to_list() == once.get_column("prediction").to_list()


def test_chart_joins_prediction_line_to_last_known_point() -> None:
    """Even for a window that was never anchored (restore / prefill / staging),
    the red line must start on the blue line's last point."""
    df = make_window(first_drawn_offset=3)

    chart = GlucoseChart.__new__(GlucoseChart)
    chart.hide_last_hour = True
    chart._display_unit = "mg/dL"
    chart._display_factor = 1.0
    figure = chart._build_figure(df, pl.DataFrame(), "test.csv", locale="en")

    blue = next(tr for tr in figure.data if tr.line.color == "blue")
    red_segments = [tr for tr in figure.data if tr.mode == "lines" and tr.line.color == "red"]

    assert red_segments, "prediction line segments missing"
    assert (red_segments[0].x[0], red_segments[0].y[0]) == (blue.x[-1], blue.y[-1])
