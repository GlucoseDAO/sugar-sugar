"""Meals inside the predicted hour must be visible while the player draws.

Reported from prod: a BIG IDEAs window put a meal a few minutes past the
known/predicted divider. The marker was clipped at the boundary, so the player
saw a flat hour, drew a flat-then-falling line, and only met the post-meal rise
on the results screen. Nobody predicts their own glucose without knowing they
are about to eat, so the marker belongs on the chart during the round -- while
the glucose values themselves stay hidden.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from sugar_sugar.components.glucose import (
    GlucoseChart,
    cluster_visible_food_events,
    meal_food_bubble_children,
)
from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.corpus import FOOD_NOTE_EVENTS_SCHEMA

WINDOW_POINTS = 2 * PREDICTION_HOUR_OFFSET
BOUNDARY = WINDOW_POINTS - PREDICTION_HOUR_OFFSET
START = datetime(2026, 1, 1, 8, 0)

#: The hidden hour rises steeply -- exactly the post-meal excursion a clipped
#: marker used to hide. Any marker y that matches these is a leak of the answer.
HIDDEN_GLUCOSE = 260.0


def _window() -> pl.DataFrame:
    times = [START + timedelta(minutes=5 * i) for i in range(WINDOW_POINTS)]
    # The boundary point itself is known (it anchors the prediction line), so the
    # rise starts one step after it.
    glucose = [90.0] * (BOUNDARY + 1) + [HIDDEN_GLUCOSE] * (WINDOW_POINTS - BOUNDARY - 1)
    return pl.DataFrame(
        {
            "time": times,
            "gl": glucose,
            "prediction": [0.0] * WINDOW_POINTS,
            "age": [30] * WINDOW_POINTS,
            "user_id": [1] * WINDOW_POINTS,
        }
    )


def _events(*, note: str = "", photo: str = "") -> pl.DataFrame:
    """One carb event and one insulin event, both inside the hidden hour."""
    meal_time = START + timedelta(minutes=5 * (BOUNDARY + 2))
    return pl.DataFrame(
        {
            "time": [meal_time, meal_time],
            "event_type": ["Carbohydrates", "Insulin"],
            "event_subtype": ["", ""],
            "insulin_value": [None, 4.0],
            "photo_path": [photo, ""],
            "meal_type": ["", ""],
            "carbs_g": [45.0, None],
            "food_note": [note, ""],
        },
        schema=FOOD_NOTE_EVENTS_SCHEMA,
    )


def _chart(df: pl.DataFrame, events: pl.DataFrame):
    chart = GlucoseChart.__new__(GlucoseChart)
    chart.hide_last_hour = True
    chart._display_unit = "mg/dL"
    chart._display_factor = 1.0
    return chart._build_figure(df, events, "test.csv", locale="en")


def test_meal_in_hidden_hour_is_drawn_during_prediction() -> None:
    figure = _chart(_window(), _events())
    apples = [
        image for image in figure.layout.images
        if "apple" in str(image.source).lower() or image.x > BOUNDARY
    ]
    assert apples, "carb marker inside the predicted hour was not drawn"
    assert any(image.x > BOUNDARY for image in figure.layout.images)


def test_hidden_hour_marker_does_not_leak_the_glucose_value() -> None:
    """The marker shows *when*, never *what* -- its y must not be the answer."""
    df = _window()
    figure = _chart(df, _events())
    for image in figure.layout.images:
        if image.x > BOUNDARY:
            assert abs(float(image.y) - HIDDEN_GLUCOSE) > 1.0

    # The hidden glucose trace itself is still withheld.
    blue = next(tr for tr in figure.data if tr.line.color == "blue")
    assert max(blue.x) <= BOUNDARY
    assert HIDDEN_GLUCOSE not in list(blue.y)


def test_insulin_in_hidden_hour_stays_hidden() -> None:
    """Only meals were reported; insulin keeps its existing gating."""
    figure = _chart(_window(), _events())
    syringes = [
        image for image in figure.layout.images
        if "syringe" in str(image.source).lower() or "insulin" in str(image.source).lower()
    ]
    assert all(image.x <= BOUNDARY for image in syringes)


def test_hidden_hour_meal_gets_a_guide_line() -> None:
    """Without the trace behind it an icon floats; the dotted line anchors it."""
    figure = _chart(_window(), _events())
    guides = [
        shape for shape in figure.layout.shapes
        if shape.type == "line" and shape.x0 == shape.x1 and float(shape.x0) > BOUNDARY
    ]
    assert guides, "no dotted guide line for the meal in the predicted hour"


def test_photo_meal_in_hidden_hour_still_gets_its_bubble() -> None:
    """The FOOD speech bubble is the other meal representation -- same rule."""
    events = _events(note="Oatmeal and banana")
    clusters = cluster_visible_food_events(_window(), events, source_name="test.csv")
    assert clusters, "meal cluster inside the predicted hour was dropped"
    assert clusters[0].x_pos > BOUNDARY

    bubbles = meal_food_bubble_children(_window(), events, source_name="test.csv")
    assert len(bubbles) == 1
