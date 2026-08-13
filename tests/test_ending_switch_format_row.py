"""Last-round /ending puts Try-another-option buttons on the X/submit row."""
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


def test_last_round_switch_buttons_sit_on_submit_row_next_to_x() -> None:
    layout = _layout(last_round=True, uses_cgm=True)
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

    next_btn = _by_id(row, "next-round-button")
    assert next_btn.style["display"] == "none"

    # Played format A; CGM users still have B and C left.
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "inline-flex"
    assert _by_id(row, "switch-format-c").style["display"] == "inline-flex"
    assert _by_id(row, "switch-format-b").children == "Try My Data"
    assert _by_id(row, "switch-format-c").children == "Try Generic + My Data"

    # The old below-the-fold card is gone; the title id stays for language updates.
    assert _by_id(layout, "ending-switch-format-title") is not None
    assert _by_id(layout, "ending-switch-format-title").style["display"] == "none"


def test_mid_round_keeps_next_and_hides_switch_buttons() -> None:
    layout = _layout(last_round=False, uses_cgm=True)
    row = _by_id(layout, "ending-submit-row")
    assert row is not None
    assert row.className == ""
    assert _by_id(row, "next-round-button").style["display"] == "inline-flex"
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"


def test_last_round_without_cgm_only_shows_x() -> None:
    layout = _layout(last_round=True, uses_cgm=False)
    row = _by_id(layout, "ending-submit-row")
    assert row is not None
    assert _by_id(row, "next-round-button").style["display"] == "none"
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"
