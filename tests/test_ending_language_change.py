"""Switching language on /ending must retranslate the chart, not just the copy.

Reported August 2026: "if you play in one language and submit your prediction
then on temp result page you want to switch the language doesn't change".
Everything on /ending *did* switch except the results graph -- its legend, its
y-axis title and the "<- Known | Predicted ->" divider are baked into the figure
at build time, and the per-element language callback never rebuilt it. The chart
is the largest thing on the page, so the page read as untranslated.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    app,
    create_ending_layout,
    dataframe_to_store_dict,
    load_dataset,
    update_ending_text_on_language_change,
)
from sugar_sugar.config import DEFAULT_POINTS, PREDICTION_HOUR_OFFSET


def _by_id(node: Any, target: str) -> Any:
    if getattr(node, "id", None) == target:
        return node
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found = _by_id(kid, target)
            if found is not None:
                return found
    elif kids is not None and not isinstance(kids, str):
        return _by_id(kids, target)
    return None


def _output_index(target: str) -> int:
    """Position of ``target`` ("id.prop") in the ending callback's output tuple.

    Read off the registered callback rather than hardcoded, so inserting an
    output moves the assertions with it instead of silently checking the
    neighbour.
    """
    key = next(k for k in app.callback_map if "ending-title.children" in k)
    outputs = [part.strip(".") for part in key.split("...")]
    return outputs.index(target)


def _window_with_predictions() -> pl.DataFrame:
    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    window = full_df.head(DEFAULT_POINTS)
    size = len(window)
    return window.with_columns(
        pl.when(pl.int_range(pl.len()) >= size - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 7.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )


def _table_data(window: pl.DataFrame) -> list[dict[str, str]]:
    actual: dict[str, str] = {"metric": "Actual Glucose"}
    pred: dict[str, str] = {"metric": "Predicted"}
    gl = window.get_column("gl").to_list()
    pr = window.get_column("prediction").to_list()
    for i in range(len(window)):
        actual[f"t{i}"] = f"{float(gl[i]):.1f}"
        pred[f"t{i}"] = "-" if pr[i] == 0.0 else f"{float(pr[i]):.1f}"
    return [actual, pred]


def _user_info(window: pl.DataFrame) -> dict[str, Any]:
    return {
        "prediction_table_data": _table_data(window),
        "prediction_window_start": 0,
        "prediction_window_size": len(window),
        "is_example_data": True,
        "data_source_name": "example.csv",
        "format": "A",
        "uses_cgm": False,
        "max_rounds": 12,
        "current_round_number": 3,
        "rounds": [{"round_number": i} for i in range(1, 4)],
        "consent_completed": True,
    }


def _figure_text(figure: Any) -> set[str]:
    """Every human-readable string baked into the results figure."""
    payload = figure.to_plotly_json() if hasattr(figure, "to_plotly_json") else figure
    texts: set[str] = set()
    for trace in payload.get("data", []):
        name = trace.get("name")
        if name:
            texts.add(str(name))
    layout = payload.get("layout", {})
    for annotation in layout.get("annotations", []) or []:
        text = annotation.get("text")
        if text:
            texts.add(str(text))
    y_title = ((layout.get("yaxis") or {}).get("title") or {}).get("text")
    if y_title:
        texts.add(str(y_title))
    return texts


def _switch(locale: str, *, window_store: Any, user_info: dict[str, Any]) -> tuple:
    return update_ending_text_on_language_change(
        locale, "/ending", user_info, "mg/dL", window_store, None,
    )


def test_language_switch_rebuilds_the_results_chart() -> None:
    window = _window_with_predictions()
    store = dataframe_to_store_dict(window)
    info = _user_info(window)
    figure_at = _output_index("ending-static-graph.figure")

    english = _figure_text(_switch("en", window_store=store, user_info=info)[figure_at])
    german = _figure_text(_switch("de", window_store=store, user_info=info)[figure_at])

    assert "Glucose Level" in english
    assert "← Known | Predicted →" in english
    # Nothing English survives into the German figure.
    assert not (english & german), f"untranslated figure text: {english & german}"
    assert any("Bekannt" in text for text in german)


def test_rebuilt_chart_matches_a_fresh_german_render() -> None:
    """The re-render must not drift from what the layout itself would build."""
    window = _window_with_predictions()
    store = dataframe_to_store_dict(window)
    info = _user_info(window)

    rebuilt = _switch("de", window_store=store, user_info=info)[
        _output_index("ending-static-graph.figure")
    ]
    fresh = _by_id(
        create_ending_layout(store, None, info, "mg/dL", locale="de"),
        "ending-static-graph",
    ).figure

    assert _figure_text(rebuilt) == _figure_text(fresh)
    assert len(rebuilt.data) == len(fresh.data)
    # The drawn prediction line is part of the figure: rebuilding must not drop it.
    red = sum(1 for trace in rebuilt.data if str(getattr(trace.line, "color", "")) == "red")
    assert red > 0
    assert red == sum(
        1 for trace in fresh.data if str(getattr(trace.line, "color", "")) == "red"
    )


def test_chart_survives_a_missing_window_store() -> None:
    """A resumed session can reach /ending without ``current-window-df``.

    The layout falls back to the recorded window offsets; the callback used to
    fall back to an *empty* frame, which would have blanked the graph on a
    language switch instead of translating it.
    """
    window = _window_with_predictions()
    info = _user_info(window)
    figure = _switch("de", window_store=None, user_info=info)[
        _output_index("ending-static-graph.figure")
    ]

    assert len(figure.data) > 0
    assert any("Bekannt" in text for text in _figure_text(figure))
