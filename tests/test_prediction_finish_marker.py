"""The playing chart says where the line has to end, and Submit looks disabled.

Reported August 2026: a player stopped drawing partway through the hidden hour,
pressed Submit, got nothing, and took it for a bug. Three things lined up:

* nothing on the chart marked the last point the line had to reach;
* ``prediction-progress-label``, which says so in words, is ``display: none`` on
  mobile ``/prediction`` -- where most rounds are played;
* Submit carries Fomantic's ``ui green button``, whose background is
  ``!important``, so the callback's inline grey never applied. A disabled Submit
  rendered as green at 45% opacity: it looked pressable.
"""
from __future__ import annotations

import json
from typing import Any

import polars as pl

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    _register_all_callbacks,
    app,
    dataframe_to_store_dict,
    load_dataset,
)
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.components.submit import (
    SUBMIT_DISABLED_CLASS,
    SUBMIT_ENABLED_CLASS,
)
from sugar_sugar.config import PREDICTION_HOUR_OFFSET


def _window(drawn: int) -> pl.DataFrame:
    """A 24-point window whose hidden hour carries ``drawn`` predictions."""
    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    window = full_df.head(24)
    first_hidden = len(window) - PREDICTION_HOUR_OFFSET
    index = pl.int_range(pl.len())
    return window.with_columns(
        pl.when((index >= first_hidden) & (index < first_hidden + drawn))
        .then(pl.col("gl") + 4.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )


def _playing_figure(window: pl.DataFrame, *, locale: str = "en") -> Any:
    chart = GlucoseChart(hide_last_hour=True)
    chart.hide_last_hour = True
    return chart._build_figure(
        window,
        pl.DataFrame({"time": [], "event_type": [], "event_subtype": [], "insulin_value": []}),
        "example.csv",
        locale=locale,
        compact=False,
    )


def _annotation_texts(figure: Any) -> list[str]:
    return [str(a.text) for a in (figure.layout.annotations or [])]


def _finish_shapes(figure: Any, x: float) -> list[Any]:
    return [
        shape
        for shape in (figure.layout.shapes or [])
        if shape.type == "line" and shape.x0 == x and shape.x1 == x
    ]


def test_unfinished_line_gets_a_flag_at_the_last_point() -> None:
    window = _window(drawn=6)
    figure = _playing_figure(window)
    last_x = float(len(window) - 1)

    assert "Draw to here" in _annotation_texts(figure)
    marker = _finish_shapes(figure, last_x)
    assert len(marker) == 1
    assert marker[0].line.dash == "dash"
    # No events in this window, so the flag is the only layout image, pinned to
    # the finish line and hanging leftwards clear of the plot border.
    images = list(figure.layout.images or [])
    assert len(images) == 1, "the flag icon is the marker players recognise"
    assert images[0].x == last_x
    assert images[0].xanchor == "right"


def test_completed_line_turns_the_marker_green() -> None:
    pending = _playing_figure(_window(drawn=6))
    done = _playing_figure(_window(drawn=PREDICTION_HOUR_OFFSET))

    assert "Line complete" in _annotation_texts(done)
    assert "Draw to here" not in _annotation_texts(done)
    last_x = float(23)
    assert _finish_shapes(pending, last_x)[0].line.color != _finish_shapes(done, last_x)[0].line.color


def test_marker_is_translated() -> None:
    figure = _playing_figure(_window(drawn=2), locale="de")
    assert "Bis hierhin zeichnen" in _annotation_texts(figure)


def test_results_chart_has_no_finish_marker() -> None:
    """On /ending the hour is revealed; a "draw to here" flag would be noise."""
    window = _window(drawn=PREDICTION_HOUR_OFFSET)
    figure = GlucoseChart.build_static_figure(
        window,
        pl.DataFrame({"time": [], "event_type": [], "event_subtype": [], "insulin_value": []}),
        "example.csv",
        unit="mg/dL",
        locale="en",
        prediction_boundary=len(window) - PREDICTION_HOUR_OFFSET,
        compact=False,
    )
    texts = _annotation_texts(figure)
    assert "Draw to here" not in texts
    assert "Line complete" not in texts


def _submit_state(window: pl.DataFrame, locale: str = "en") -> dict[str, Any]:
    """Run the real Submit-gate callback and return its ``submit-button`` props."""
    _register_all_callbacks()
    key = next(k for k in app.callback_map if "submit-button.disabled" in k)
    outputs = [
        {"id": part.strip(".").split(".")[0], "property": part.strip(".").split(".")[1]}
        for part in key.split("...")
    ]
    raw = app.callback_map[key]["callback"](
        dataframe_to_store_dict(window), locale, None, outputs_list=outputs,
    )
    return json.loads(raw)["response"]["submit-button"]


def test_disabled_submit_drops_the_green_class_and_shows_the_count() -> None:
    state = _submit_state(_window(drawn=6))
    assert state["disabled"] is True
    # Fomantic's green background is !important: the class has to go, not the style.
    assert state["className"] == SUBMIT_DISABLED_CLASS
    assert "green" not in state["className"]
    # The count rides on the button because the italic label is hidden on mobile.
    assert state["children"] == "Submit (6/12)"


def test_completed_line_enables_a_green_submit() -> None:
    state = _submit_state(_window(drawn=PREDICTION_HOUR_OFFSET))
    assert state["disabled"] is False
    assert state["className"] == SUBMIT_ENABLED_CLASS


def test_submit_count_is_translated() -> None:
    assert _submit_state(_window(drawn=3), locale="de")["children"] == "Senden (3/12)"
