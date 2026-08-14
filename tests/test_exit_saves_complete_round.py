"""Finish/Exit from /prediction stores a fully drawn round and opens results."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import polars as pl

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    append_round_from_window,
    capture_complete_round_on_exit,
    create_prediction_layout,
    dataframe_to_store_dict,
    handle_finish_study_from_prediction,
    load_dataset,
)
from sugar_sugar.subject_sources import generic_window_slice_key
from sugar_sugar.components.submit import SubmitComponent, hidden_area_is_complete
from sugar_sugar.config import DEFAULT_POINTS, PREDICTION_HOUR_OFFSET


def _ids(node: Any) -> set[str]:
    found: set[str] = set()
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        found.add(node_id)
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found |= _ids(kid)
    elif kids is not None and not isinstance(kids, str):
        found |= _ids(kids)
    return found


def _complete_window() -> pl.DataFrame:
    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    window = full_df.head(DEFAULT_POINTS)
    size = len(window)
    return window.with_columns(
        pl.when(pl.int_range(pl.len()) >= size - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 2.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )


def _incomplete_window() -> pl.DataFrame:
    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    return full_df.head(DEFAULT_POINTS)


def test_hidden_area_is_complete_requires_last_hidden_point() -> None:
    assert hidden_area_is_complete(_complete_window()) is True
    assert hidden_area_is_complete(_incomplete_window()) is False


def test_capture_complete_round_on_exit_appends_only_when_drawn() -> None:
    complete = capture_complete_round_on_exit(
        {"rounds": [], "format": "A", "is_example_data": True},
        dataframe_to_store_dict(_complete_window()),
        0,
    )
    assert len(complete.get("rounds") or []) == 1
    assert complete["rounds"][0]["round_number"] == 1

    incomplete = capture_complete_round_on_exit(
        {"rounds": [], "format": "A"},
        dataframe_to_store_dict(_incomplete_window()),
        0,
    )
    assert incomplete.get("rounds") == []


def test_append_round_stores_source_and_slice_key_for_own_data() -> None:
    window = _complete_window()
    expected_key = generic_window_slice_key(window)
    info = append_round_from_window(
        {
            "rounds": [],
            "format": "B",
            "is_example_data": False,
            "data_source_name": "Clarity_Export.csv",
            "current_generic_slice_key": "stale-from-format-a",
        },
        window,
        0,
    )
    round_info = info["rounds"][0]
    assert round_info["data_source_name"] == "Clarity_Export.csv"
    assert round_info["is_example_data"] is False
    assert round_info["generic_slice_key"] == expected_key
    assert round_info["generic_slice_key"] != "stale-from-format-a"


def test_append_round_keeps_generic_slice_key_for_format_c_odd_round() -> None:
    window = _complete_window()
    info = append_round_from_window(
        {
            "rounds": [],
            "format": "C",
            "is_example_data": True,
            "data_source_name": "D1NAMO-002.csv",
            "current_generic_slice_key": "picked-generic-key",
        },
        window,
        0,
    )
    round_info = info["rounds"][0]
    assert round_info["data_source_name"] == "D1NAMO-002.csv"
    assert round_info["is_example_data"] is True
    assert round_info["generic_slice_key"] == "picked-generic-key"


def test_finish_from_prediction_goes_to_ending_when_round_is_complete() -> None:
    pathname, info, mode, last_page = handle_finish_study_from_prediction(
        1,
        {"consent_completed": False, "rounds": [], "format": "A", "is_example_data": True},
        dataframe_to_store_dict(_complete_window()),
        0,
    )
    assert pathname == "/ending"
    assert last_page == "/ending"
    assert mode == {"hide_last_hour": False}
    assert len(info.get("rounds") or []) == 1


def test_finish_from_prediction_goes_to_final_when_prior_rounds_exist() -> None:
    prior = {
        "round_number": 1,
        "prediction_table_data": [{"metric": "Actual Glucose"}],
        "format": "A",
    }
    pathname, info, mode, last_page = handle_finish_study_from_prediction(
        1,
        {
            "consent_completed": False,
            "rounds": [prior],
            "format": "A",
            "prediction_table_data": prior["prediction_table_data"],
        },
        dataframe_to_store_dict(_incomplete_window()),
        0,
    )
    assert pathname == "/final"
    assert last_page == "/final"
    assert mode == {"hide_last_hour": False}
    assert len(info.get("rounds") or []) == 1


def test_finish_from_prediction_goes_to_landing_when_nothing_to_show() -> None:
    pathname, info, _mode, last_page = handle_finish_study_from_prediction(
        1,
        {"consent_completed": False, "rounds": [], "format": "A"},
        dataframe_to_store_dict(_incomplete_window()),
        0,
    )
    assert pathname == "/"
    assert last_page is None
    assert info.get("rounds") == []


def test_save_statistics_can_skip_ranking(tmp_path: Path) -> None:
    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    window = _complete_window()
    captured = capture_complete_round_on_exit(
        {
            "study_id": "exit-only",
            "run_id": "r1",
            "number": 1,
            "consent_completed": True,
            "format": "A",
            "run_format": "A",
            "age": 30,
            "email": "exit@example.com",
            "rounds": [],
        },
        dataframe_to_store_dict(window),
        0,
    )
    submit.save_statistics(captured, write_ranking=False)
    with submit._stats_csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["rounds_played"] == "1"
    assert not submit._ranking_csv_path.exists()
    assert not submit._ranking_by_format_paths["A"].exists()


def test_prediction_layout_keeps_finish_and_meta_ids() -> None:
    layout = create_prediction_layout(
        locale="en",
        format_value="A",
        user_info={
            "format": "A",
            "max_rounds": 12,
            "consent_completed": True,
        },
    )
    ids = _ids(layout)
    assert "prediction-meta-row" in ids
    assert "prediction-round-summary" in ids
    assert "prediction-source-plaque" in ids
    assert "finish-study-button" in ids
    assert "submit-button" in ids
    assert "prediction-submit-row" in ids
    assert "prediction-upload-slot" in ids
    assert layout.id == "prediction-page"
