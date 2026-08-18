"""Last-round /ending turns the X into Results; format CTAs stay hidden."""
from __future__ import annotations

from typing import Any

import polars as pl

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    create_ending_layout,
    dataframe_to_store_dict,
    load_dataset,
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


def _window_with_predictions() -> pl.DataFrame:
    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    window = full_df.head(DEFAULT_POINTS)
    size = len(window)
    return window.with_columns(
        pl.when(pl.int_range(pl.len()) >= size - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 2.0)
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


def _layout(*, last_round: bool, uses_cgm: bool) -> Any:
    window = _window_with_predictions()
    max_rounds = 12
    round_n = max_rounds if last_round else 3
    user_info: dict[str, Any] = {
        "prediction_table_data": _table_data(window),
        "prediction_window_start": 0,
        "prediction_window_size": len(window),
        "is_example_data": True,
        "data_source_name": "example.csv",
        "format": "A",
        "uses_cgm": uses_cgm,
        "max_rounds": max_rounds,
        "current_round_number": round_n,
        "rounds": [{"round_number": i} for i in range(1, round_n + 1)],
        "consent_completed": True,
    }
    return create_ending_layout(
        dataframe_to_store_dict(window),
        None,
        user_info,
        "mg/dL",
        locale="en",
    )


def test_last_round_finish_button_is_green_results() -> None:
    layout = _layout(last_round=True, uses_cgm=True)
    assert _by_id(layout, "finish-confirm-overlay-ending") is not None
    assert _by_id(layout, "finish-confirm-button-ending") is not None
    assert _by_id(layout, "finish-confirm-overlay-prediction") is None
    row = _by_id(layout, "ending-submit-row")
    assert row is not None
    assert row.className == "ending-submit-row-last"

    child_ids = [getattr(child, "id", None) for child in row.children]
    assert child_ids == [
        "finish-study-button-ending",
        "next-round-button",
        "switch-format-a",
        "switch-format-b",
        "switch-format-c",
    ]

    finish = _by_id(row, "finish-study-button-ending")
    assert finish.children == "Results"
    assert "finish-study-results" in finish.className
    assert "huge" in finish.className
    assert finish.style["backgroundColor"] == "#4CBB17"
    assert finish.style["minWidth"] == "320px"
    assert finish.style["height"] == "80px"
    assert finish.style["fontSize"] == "32px"

    gamification = _by_id(layout, "ending-gamification")
    assert gamification is not None
    assert gamification.className == "ending-gamification-complete"
    reaction = _by_id(layout, "ending-reaction-line")
    assert reaction is not None
    assert reaction.className == "ending-celebrate"
    assert reaction.style["fontSize"] == "24px"
    milestone = _by_id(layout, "ending-milestone")
    assert milestone is not None
    assert milestone.className == "ending-celebrate"
    assert milestone.style["fontSize"] == "22px"

    next_btn = _by_id(row, "next-round-button")
    assert next_btn.style["display"] == "none"

    # Remaining formats belong on /final, not last-round /ending.
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"

    # The title id stays for language updates, but the card stays hidden.
    assert _by_id(layout, "ending-switch-format-title") is not None
    assert _by_id(layout, "ending-switch-format-title").style["display"] == "none"


def test_mid_round_keeps_next_and_hides_switch_buttons() -> None:
    layout = _layout(last_round=False, uses_cgm=True)
    row = _by_id(layout, "ending-submit-row")
    assert row is not None
    assert row.className == ""
    assert _by_id(row, "next-round-button").style["display"] == "inline-flex"
    finish = _by_id(row, "finish-study-button-ending")
    assert "finish-study-results" not in (finish.className or "")
    assert "huge" not in (finish.className or "")
    assert finish.className == "ui button finish-study-exit"
    assert finish.style["backgroundColor"] == "#E81123"
    assert finish.style["width"] == "48px"
    assert _by_id(layout, "ending-gamification").className == ""
    assert _by_id(layout, "ending-reaction-line").className == ""
    assert _by_id(layout, "ending-reaction-line").style["fontSize"] == "14px"
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"


def test_ending_keeps_source_on_the_chart_card() -> None:
    """Source stays on the results card, same plaque as /prediction."""
    layout = _layout(last_round=False, uses_cgm=False)
    source = _by_id(layout, "ending-source-info")
    assert source is not None
    assert source.className == "prediction-source-plaque"
    assert _by_id(source, "ending-source-name").children == "example.csv"
    assert _by_id(source, "ending-source-label").className == "prediction-source-label"
    assert _by_id(source, "ending-source-metadata").className == "prediction-source-metadata"
    assert _by_id(source, "ending-source-time").children
    graph = _by_id(layout, "ending-graph-card")
    assert _by_id(graph, "ending-source-info") is source
    assert _by_id(layout, "ending-food-bubbles") is not None


def test_last_round_without_cgm_only_shows_results() -> None:
    layout = _layout(last_round=True, uses_cgm=False)
    row = _by_id(layout, "ending-submit-row")
    assert row is not None
    finish = _by_id(row, "finish-study-button-ending")
    assert finish.children == "Results"
    assert "finish-study-results" in finish.className
    assert _by_id(row, "next-round-button").style["display"] == "none"
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"
