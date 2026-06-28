"""Tests for the server-side dataset refactor (no full-df client store).

Covers: the load_dataset cache, dataset-identity resolution, format-C per-round
time correctness in save_statistics, that create_ending_layout renders from the
window alone (no full-df), and that handle_next_round_button returns a tuple whose
arity matches its (full-df-free) Output list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from dash import html

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    create_ending_layout,
    dataframe_to_store_dict,
    get_random_data_window,
    handle_example_data_button,
    handle_next_round_button,
    handle_nightscout_load,
    initialize_data_on_url_change,
    load_dataset,
    resolve_dataset_identity,
)
from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.config import DEFAULT_POINTS, PREDICTION_HOUR_OFFSET


def test_load_dataset_is_cached_by_path() -> None:
    g1, e1 = load_dataset(EXAMPLE_DATASET_PATH)
    g2, e2 = load_dataset(EXAMPLE_DATASET_PATH)
    # Same path -> served from the lru_cache (identical objects, no re-read).
    assert g1 is g2
    assert e1 is e2
    # Schema matches the window store schema (incl. the reset prediction column).
    assert set(g1.columns) == {"time", "gl", "prediction", "age", "user_id"}
    assert set(g1.get_column("prediction").to_list()) == {0.0}


def test_resolve_dataset_identity_by_format() -> None:
    example = EXAMPLE_DATASET_PATH
    uploaded = "/data/input/users/20260101_000000_x.csv"

    # Current-window identity trusts is_example_data.
    assert resolve_dataset_identity({"is_example_data": True}) == example
    assert resolve_dataset_identity(
        {"is_example_data": False, "uploaded_data_path": uploaded}
    ) == Path(uploaded)

    # Per-round identity mirrors handle_next_round_button.
    info_b = {"format": "B", "uploaded_data_path": uploaded}
    assert resolve_dataset_identity(info_b, round_number=1) == Path(uploaded)

    info_a = {"format": "A", "uploaded_data_path": uploaded}
    assert resolve_dataset_identity(info_a, round_number=1) == example

    info_c = {"format": "C", "uploaded_data_path": uploaded}
    assert resolve_dataset_identity(info_c, round_number=2) == example  # even -> example
    assert resolve_dataset_identity(info_c, round_number=1) == Path(uploaded)  # odd -> uploaded


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


def _times(window: pl.DataFrame) -> list[str]:
    return window.get_column("time").dt.strftime("%Y-%m-%d %H:%M:%S").to_list()


def test_save_statistics_uses_per_round_window_times(tmp_path: Path) -> None:
    """Format-C correctness: each round's prediction_times come from that round's
    own stored window_times, not from a single shared dataframe."""
    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    w1 = full_df.slice(0, DEFAULT_POINTS).with_columns(
        pl.when(pl.int_range(pl.len()) >= DEFAULT_POINTS - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 1.0).otherwise(pl.col("prediction")).alias("prediction")
    )
    w2 = full_df.slice(DEFAULT_POINTS, DEFAULT_POINTS).with_columns(
        pl.when(pl.int_range(pl.len()) >= DEFAULT_POINTS - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 1.0).otherwise(pl.col("prediction")).alias("prediction")
    )
    times1, times2 = _times(w1), _times(w2)
    assert times1 != times2  # different windows -> different absolute times

    user_info: dict[str, Any] = {
        "study_id": "fmt-c", "run_id": "r", "number": 1, "consent_completed": True,
        "format": "C", "run_format": "C", "age": 40,
        "rounds": [
            {"round_number": 1, "prediction_window_size": len(w1),
             "prediction_table_data": _table_data(w1), "window_times": times1,
             "format": "C", "is_example_data": True, "data_source_name": "example.csv"},
            {"round_number": 2, "prediction_window_size": len(w2),
             "prediction_table_data": _table_data(w2), "window_times": times2,
             "format": "C", "is_example_data": True, "data_source_name": "example.csv"},
        ],
    }
    submit.save_statistics(user_info)

    import csv
    with submit._stats_csv_path.open(newline="") as fh:
        row = list(csv.DictReader(fh))[0]
    pt = row["prediction_times"]
    # Times from BOTH rounds' own windows must appear in the saved record.
    assert times1[-1] in pt
    assert times2[-1] in pt


def test_create_ending_layout_renders_from_window_without_full_df() -> None:
    window = _window_with_predictions()
    user_info = {
        "prediction_table_data": _table_data(window),
        "prediction_window_start": 0,
        "prediction_window_size": len(window),
        "is_example_data": True,
        "data_source_name": "example.csv",
        "rounds": [{"round_number": 1}],
    }
    layout = create_ending_layout(
        dataframe_to_store_dict(window),
        None,            # no events store -> loads from dataset server-side
        user_info,
        "mg/dL",
        locale="en",
    )
    assert isinstance(layout, html.Div)
    # It is NOT one of the early "no data / no predictions" fallbacks.
    assert not (isinstance(layout.children, str) and "No " in layout.children)


def test_handle_next_round_button_arity_has_no_full_df() -> None:
    user_info = {
        "format": "A", "prediction_window_size": DEFAULT_POINTS,
        "rounds": [], "max_rounds": 12, "current_round_number": 1,
        "is_example_data": True, "data_source_name": "example.csv",
    }
    result = handle_next_round_button(1, user_info)
    # url, user-info, chart-mode, current-window, events, is-example, source,
    # randomization-initialized, initial-slider  == 9 (full-df dropped).
    assert len(result) == 9
    assert result[0] == "/prediction"


def test_producer_callback_arities_have_no_full_df() -> None:
    """Other producers whose Output lists lost full-df: confirm their return
    arity matches (these fire without a Dash callback context)."""
    # handle_example_data_button: 8 outputs (full-df dropped).
    assert len(handle_example_data_button(0)) == 8  # no-click early return
    assert len(handle_example_data_button(1)) == 8  # active path

    # initialize_data_on_url_change: 6 outputs (full-df dropped).
    assert len(initialize_data_on_url_change("/about", None, None)) == 6  # non-prediction

    # handle_nightscout_load error path: 8 no-update + status == 9 outputs.
    res = handle_nightscout_load(1, "", None, None, None)
    assert len(res) == 9
