"""Event markers inside the predicted hour must be visible while drawing.

Reported from prod: a BIG IDEAs window put a meal a few minutes past the
known/predicted divider. The marker was clipped at the boundary, so the player
saw a flat hour, drew a flat-then-falling line, and only met the post-meal rise
on the results screen. Meals and exercise stay on the chart during the round;
glucose values stay hidden. Insulin is the exception: its circle sits on the
true glucose value, so doses past the divider wait for the results chart.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from sugar_sugar.components.glucose import (
    GlucoseChart,
    _INSULIN_CIRCLE_MAX_PX,
    _INSULIN_CIRCLE_MIN_PX,
    _insulin_circle_size,
    _insulin_compact_label,
    _insulin_label_positions,
    _svg_data_uri,
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


def _events(*, note: str = "", photo: str = "", with_exercise: bool = False) -> pl.DataFrame:
    """Carb + insulin (and optionally exercise) events inside the hidden hour."""
    meal_time = START + timedelta(minutes=5 * (BOUNDARY + 2))
    types = ["Carbohydrates", "Insulin"]
    insulin = [None, 4.0]
    photos = [photo, ""]
    carbs = [45.0, None]
    notes = [note, ""]
    if with_exercise:
        types.append("Exercise")
        insulin.append(None)
        photos.append("")
        carbs.append(None)
        notes.append("")
    return pl.DataFrame(
        {
            "time": [meal_time] * len(types),
            "event_type": types,
            "event_subtype": [""] * len(types),
            "insulin_value": insulin,
            "photo_path": photos,
            "meal_type": [""] * len(types),
            "carbs_g": carbs,
            "food_note": notes,
        },
        schema=FOOD_NOTE_EVENTS_SCHEMA,
    )


def _visible_insulin_events(
    *,
    dose: float = 4.0,
    extra_doses: list[float] | None = None,
) -> pl.DataFrame:
    """Insulin dose(s) in the known (visible) half of the window.

    Extra doses share the same timestamp so the chart has to stack them.
    """
    event_time = START + timedelta(minutes=5 * 4)
    doses = [dose, *(extra_doses or [])]
    n = len(doses)
    return pl.DataFrame(
        {
            "time": [event_time] * n,
            "event_type": ["Insulin"] * n,
            "event_subtype": [""] * n,
            "insulin_value": doses,
            "photo_path": [""] * n,
            "meal_type": [""] * n,
            "carbs_g": [None] * n,
            "food_note": [""] * n,
        },
        schema=FOOD_NOTE_EVENTS_SCHEMA,
    )


def _chart(
    df: pl.DataFrame,
    events: pl.DataFrame,
    *,
    hide_last_hour: bool = True,
    compact: bool | None = None,
):
    chart = GlucoseChart.__new__(GlucoseChart)
    chart.hide_last_hour = hide_last_hour
    chart._display_unit = "mg/dL"
    chart._display_factor = 1.0
    return chart._build_figure(
        df, events, "test.csv", locale="en", compact=compact,
    )


def _insulin_circle_traces(figure):
    color = GlucoseChart.EVENT_STYLES["Insulin"]["color"]
    traces = []
    for tr in figure.data:
        if getattr(tr.marker, "symbol", None) != "circle":
            continue
        if str(tr.marker.color) != color:
            continue
        if getattr(tr.marker, "opacity", None) == 0.0:
            continue
        traces.append(tr)
    return traces


def _syringe_images(figure):
    """On-plot syringes only — the paper-coord legend icon is excluded."""
    uri = _svg_data_uri("syringe.svg")
    return [
        image for image in figure.layout.images
        if image.source == uri and getattr(image, "yref", "y") == "y"
    ]


def _insulin_connector_shapes(figure):
    color = GlucoseChart.EVENT_STYLES["Insulin"]["color"]
    return [
        shape for shape in figure.layout.shapes
        if shape.type == "line"
        and str(shape.line.color) == color
        and getattr(shape, "yref", "paper") == "y"
    ]


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


def test_insulin_in_hidden_hour_is_withheld_during_prediction() -> None:
    """The dose circle sits on true glucose, so it would leak the answer."""
    figure = _chart(_window(), _events())
    assert _insulin_circle_traces(figure) == []
    assert _syringe_images(figure) == []
    assert _insulin_connector_shapes(figure) == []
    past = [image for image in figure.layout.images if image.x > BOUNDARY]
    for image in past:
        assert abs(float(image.y) - HIDDEN_GLUCOSE) > 1.0


def test_exercise_in_hidden_hour_is_pinned_not_plotted_at_the_answer() -> None:
    """Exercise was never boundary-gated, so its star read off the y-axis."""
    events = _events(with_exercise=True)
    figure = _chart(_window(), events)
    stars = [tr for tr in figure.data if getattr(tr.marker, "symbol", None) == "star"]
    assert stars, "exercise marker missing"
    past = [
        (x, y) for tr in stars for x, y in zip(tr.x, tr.y) if x > BOUNDARY
    ]
    assert past, "exercise marker inside the predicted hour was dropped"
    for _, y in past:
        assert abs(float(y) - HIDDEN_GLUCOSE) > 1.0


def test_hidden_hour_markers_get_guide_lines_in_their_own_colour() -> None:
    """Without the trace behind it an icon floats; the dotted line anchors it.

    Insulin is withheld in the hidden hour, so the guide is meal-green only.
    """
    figure = _chart(_window(), _events())
    guides = [
        shape for shape in figure.layout.shapes
        if shape.type == "line" and shape.x0 == shape.x1 and float(shape.x0) > BOUNDARY
    ]
    assert guides, "no dotted guide line for markers in the predicted hour"
    colours = {str(shape.line.color) for shape in guides}
    assert GlucoseChart.EVENT_STYLES["Carbohydrates"]["color"] in colours
    assert GlucoseChart.EVENT_STYLES["Insulin"]["color"] not in colours


def test_photo_meal_in_hidden_hour_still_gets_its_bubble() -> None:
    """The FOOD speech bubble is the other meal representation -- same rule."""
    events = _events(note="Oatmeal and banana")
    clusters = cluster_visible_food_events(_window(), events, source_name="test.csv")
    assert clusters, "meal cluster inside the predicted hour was dropped"
    assert clusters[0].x_pos > BOUNDARY

    bubbles = meal_food_bubble_children(_window(), events, source_name="test.csv")
    assert len(bubbles) == 1


def test_insulin_circle_sits_on_the_glucose_curve() -> None:
    df = _window()
    figure = _chart(df, _visible_insulin_events(dose=4.0))
    traces = _insulin_circle_traces(figure)
    assert len(traces) == 1
    assert list(traces[0].x) == [4.0]
    assert list(traces[0].y) == [90.0]


def test_insulin_syringe_sits_near_the_plot_base() -> None:
    df = _window()
    figure = _chart(df, _visible_insulin_events(dose=4.0))
    syringes = _syringe_images(figure)
    assert len(syringes) == 1
    y_min = float(figure.layout.yaxis.range[0])
    y_max = float(figure.layout.yaxis.range[1])
    y_span = y_max - y_min
    assert syringes[0].y < y_min + 0.2 * y_span
    assert syringes[0].y > y_min
    assert abs(float(syringes[0].x) - 4.0) < 0.01


def test_insulin_connector_joins_circle_to_base() -> None:
    figure = _chart(_window(), _visible_insulin_events(dose=4.0))
    connectors = _insulin_connector_shapes(figure)
    assert len(connectors) == 1
    line = connectors[0]
    assert float(line.x0) == 4.0
    assert float(line.y0) == 90.0
    y_min = float(figure.layout.yaxis.range[0])
    y_max = float(figure.layout.yaxis.range[1])
    assert float(line.y1) < y_min + 0.2 * (y_max - y_min)


def test_insulin_circle_size_clamps_to_min_and_max() -> None:
    assert _insulin_circle_size(1.0) == _INSULIN_CIRCLE_MIN_PX
    assert _insulin_circle_size(0.5) == _INSULIN_CIRCLE_MIN_PX
    assert _insulin_circle_size(10.0) == _INSULIN_CIRCLE_MAX_PX
    assert _insulin_circle_size(15.0) == _INSULIN_CIRCLE_MAX_PX
    mid = _insulin_circle_size(5.5)
    assert _INSULIN_CIRCLE_MIN_PX < mid < _INSULIN_CIRCLE_MAX_PX

    small = _chart(_window(), _visible_insulin_events(dose=1.0))
    large = _chart(_window(), _visible_insulin_events(dose=10.0))
    small_sizes = list(_insulin_circle_traces(small)[0].marker.size)
    large_sizes = list(_insulin_circle_traces(large)[0].marker.size)
    assert small_sizes == [_INSULIN_CIRCLE_MIN_PX]
    assert large_sizes == [_INSULIN_CIRCLE_MAX_PX]


def test_results_chart_draws_hidden_hour_insulin() -> None:
    """After submit the hour is revealed, so the dose circle is safe to show."""
    figure = _chart(_window(), _events(), hide_last_hour=False)
    traces = _insulin_circle_traces(figure)
    assert traces, "results chart dropped the hidden-hour dose"
    past = [
        (x, y) for tr in traces for x, y in zip(tr.x, tr.y) if x > BOUNDARY
    ]
    assert past, "hidden-hour insulin missing on the results figure"
    for x, y in past:
        assert abs(float(y) - HIDDEN_GLUCOSE) < 1.0
    syringes = _syringe_images(figure)
    assert any(float(image.x) > BOUNDARY for image in syringes)


def test_compact_insulin_labels_use_timestamp_font_size() -> None:
    compact = _chart(
        _window(), _visible_insulin_events(dose=4.0), compact=True,
    )
    desktop = _chart(
        _window(), _visible_insulin_events(dose=4.0), compact=False,
    )
    compact_tr = _insulin_circle_traces(compact)[0]
    desktop_tr = _insulin_circle_traces(desktop)[0]
    assert "text" in compact_tr.mode
    assert list(compact_tr.text) == [_insulin_compact_label(4.0)]
    assert compact_tr.textfont.size == 8
    assert compact.layout.xaxis.tickfont.size == 8
    assert compact_tr.textfont.size == compact.layout.xaxis.tickfont.size
    assert desktop_tr.mode == "markers"
    assert desktop_tr.text is None


def test_insulin_compact_label_drops_trailing_zeros() -> None:
    assert _insulin_compact_label(4.0) == "4u"
    assert _insulin_compact_label(2.5) == "2.5u"


def test_insulin_labels_alternate_above_and_below_when_crowded() -> None:
    """5-min neighbours must not all sit on the same side of the circle."""
    isolated = _insulin_label_positions(
        [{"x": 4.0, "glucose_y": 90.0}],
        y_min=50.0,
        y_max=250.0,
    )
    assert isolated == ["top center"]

    neighbours = _insulin_label_positions(
        [
            {"x": 4.0, "glucose_y": 90.0},
            {"x": 5.0, "glucose_y": 90.0},
            {"x": 6.0, "glucose_y": 92.0},
        ],
        y_min=50.0,
        y_max=250.0,
    )
    assert neighbours == ["top center", "bottom center", "top center"]

    near_ceiling = _insulin_label_positions(
        [
            {"x": 4.0, "glucose_y": 230.0},
            {"x": 5.0, "glucose_y": 232.0},
        ],
        y_min=50.0,
        y_max=250.0,
    )
    assert near_ceiling == ["bottom center", "top center"]


def test_compact_insulin_labels_alternate_on_adjacent_doses() -> None:
    t0 = START + timedelta(minutes=5 * 4)
    t1 = START + timedelta(minutes=5 * 5)
    events = pl.DataFrame(
        {
            "time": [t0, t1],
            "event_type": ["Insulin", "Insulin"],
            "event_subtype": ["", ""],
            "insulin_value": [1.1, 0.2],
            "photo_path": ["", ""],
            "meal_type": ["", ""],
            "carbs_g": [None, None],
            "food_note": ["", ""],
        },
        schema=FOOD_NOTE_EVENTS_SCHEMA,
    )
    figure = _chart(_window(), events, compact=True)
    tr = _insulin_circle_traces(figure)[0]
    assert list(tr.textposition) == ["top center", "bottom center"]


def test_overlapping_insulin_circles_stack_vertically() -> None:
    """Two doses at the same minute must not sit on one glucose point."""
    figure = _chart(
        _window(),
        _visible_insulin_events(dose=2.0, extra_doses=[6.0]),
    )
    traces = _insulin_circle_traces(figure)
    assert traces
    xs = list(traces[0].x)
    ys = list(traces[0].y)
    assert xs == [4.0, 4.0]
    assert len(set(round(float(y), 3) for y in ys)) == 2
    assert 90.0 in [float(y) for y in ys]
    syringes = _syringe_images(figure)
    assert len(syringes) == 2
    syringe_ys = [float(image.y) for image in syringes]
    assert len(set(round(y, 3) for y in syringe_ys)) == 2
    syringe_xs = [float(image.x) for image in syringes]
    assert all(abs(x - 4.0) < 0.01 for x in syringe_xs)


def test_insulin_five_minutes_apart_stays_on_the_curve() -> None:
    """Adjacent slots are different injections — each circle sits on its reading."""
    t0 = START + timedelta(minutes=5 * 4)
    t1 = START + timedelta(minutes=5 * 5)
    events = pl.DataFrame(
        {
            "time": [t0, t1],
            "event_type": ["Insulin", "Insulin"],
            "event_subtype": ["", ""],
            "insulin_value": [2.0, 6.0],
            "photo_path": ["", ""],
            "meal_type": ["", ""],
            "carbs_g": [None, None],
            "food_note": ["", ""],
        },
        schema=FOOD_NOTE_EVENTS_SCHEMA,
    )
    df = _window()
    figure = _chart(df, events)
    traces = _insulin_circle_traces(figure)
    assert traces
    xs = [float(x) for x in traces[0].x]
    ys = [float(y) for y in traces[0].y]
    assert xs == [4.0, 5.0]
    assert ys == [90.0, 90.0]
    syringes = _syringe_images(figure)
    assert len(syringes) == 2
    syringe_ys = [round(float(image.y), 3) for image in syringes]
    assert len(set(syringe_ys)) == 1
