"""A saved round's ``data_source_name`` names the file its ``real_values`` come from.

The values a round saves are the window's glucose (``current-window-df``); its
name used to be copied from ``user_info['data_source_name']``, a different store.
``handle_time_slider`` re-slices the window from the file ``user_info`` names, and
``initialize_data_on_url_change`` / ``handle_example_data_button`` update that
name only through the chained ``sync_data_source_into_user_info``. A stale read
in between put one file's values under another file's name in
``prediction_statistics.csv``. Separately, rounds with no name of their own were
given the run-level name, which is only the last round's source.

These tests replay that sequence through the real callbacks and then check the
saved CSV against the files on disk.
"""
from __future__ import annotations

import ast
import csv
import random
from pathlib import Path
from typing import Any

import polars as pl

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    append_round_from_window,
    convert_df_to_dict,
    handle_next_round_button,
    handle_time_slider,
    load_dataset,
    reconstruct_dataframe_from_dict,
    sync_data_source_into_user_info,
)
from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.config import DEFAULT_POINTS, PREDICTION_HOUR_OFFSET
from sugar_sugar.data import load_glucose_data
from sugar_sugar.subject_sources import resolve_generic_source_path, window_in_dataset

LOOP_NAME = "loop_1082_chronological.csv"
LOOP_PATH = Path("data/subjects/loop_1082/loop_1082_chronological.csv")
SLIDER_START = 100


def _drawn(window: pl.DataFrame) -> pl.DataFrame:
    """The window with the hidden hour drawn through to its last point."""
    return window.with_columns(
        pl.when(pl.int_range(pl.len()) >= pl.len() - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 3.0)
        .otherwise(0.0)
        .alias("prediction")
    )


def _session(**extra: Any) -> dict[str, Any]:
    return {
        "study_id": "source-check",
        "run_id": "run-1",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "age": 30,
        "rounds": [],
        "max_rounds": 12,
        "prediction_window_size": DEFAULT_POINTS,
        **extra,
    }


def _submit_component(tmp_path: Path) -> SubmitComponent:
    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}
    return submit


def _saved_rounds(submit: SubmitComponent) -> list[dict[str, Any]]:
    """Each saved round's name, flag, and ``(time, actual)`` pairs, read back from the CSV."""
    with submit._stats_csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    per_round = ast.literal_eval(row["per_round_metrics"])
    contexts = ast.literal_eval(row["round_context"])
    reals = ast.literal_eval(row["real_values"])
    times = ast.literal_eval(row["prediction_times"])
    saved: list[dict[str, Any]] = []
    for position, (metrics, context) in enumerate(zip(per_round, contexts), start=1):
        pairs = [
            (t["value"], float(v["value"]))
            for v, t in zip(reals, times)
            if v["round"] == position and t["round"] == position
        ]
        assert len(pairs) >= PREDICTION_HOUR_OFFSET - 1
        assert context["data_source_name"] == metrics["data_source_name"]
        saved.append({
            "data_source_name": metrics["data_source_name"],
            "is_example_data": metrics["is_example_data"],
            "pairs": pairs,
        })
    return saved


def _file_holds(path: Path, pairs: list[tuple[str, float]]) -> bool:
    """Read the file afresh (not through the app's cache) and look up every value."""
    glucose, _ = load_glucose_data(path)
    by_time = dict(zip(
        glucose.get_column("time").dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
        glucose.get_column("gl").to_list(),
    ))
    return all(
        stamp in by_time and abs(float(by_time[stamp]) - value) <= 0.06
        for stamp, value in pairs
    )


def _assert_every_round_names_its_file(saved: list[dict[str, Any]]) -> None:
    for entry in saved:
        name = entry["data_source_name"]
        path = resolve_generic_source_path(name)
        assert path is not None, f"{name!r} is not a file the game can serve"
        assert _file_holds(path, entry["pairs"]), (
            f"round saved as {name!r} holds values that are not in that file"
        )


def test_window_in_dataset_matches_only_the_file_it_was_cut_from() -> None:
    example, _ = load_dataset(EXAMPLE_DATASET_PATH)
    loop, _ = load_dataset(LOOP_PATH)
    window = example.slice(SLIDER_START, DEFAULT_POINTS)
    assert window_in_dataset(window, example)
    assert not window_in_dataset(window, loop)
    assert window_in_dataset(loop.slice(SLIDER_START, DEFAULT_POINTS), loop)

    nudged = window.with_columns(
        pl.when(pl.int_range(pl.len()) == DEFAULT_POINTS - 1)
        .then(pl.col("gl") + 1.0)
        .otherwise(pl.col("gl"))
        .alias("gl")
    )
    assert not window_in_dataset(nudged, example)


def test_stale_slider_reslice_is_saved_under_the_file_it_came_from(tmp_path: Path) -> None:
    """The window is example.csv's, but the session names a LOOP file by Submit."""
    # The session still names example.csv while a new pick (a LOOP subject)
    # has already reached current-window-df and the data-source-name store.
    user_info = _session(data_source_name="example.csv", is_example_data=True)
    loop, _ = load_dataset(LOOP_PATH)
    picked = convert_df_to_dict(loop.slice(SLIDER_START, DEFAULT_POINTS))

    # The slider fires with the stale user-info-store and re-slices example.csv.
    _clicked, window_data, _events = handle_time_slider(SLIDER_START, user_info, picked)
    window = reconstruct_dataframe_from_dict(window_data)
    example, _ = load_dataset(EXAMPLE_DATASET_PATH)
    assert window_in_dataset(window, example)
    assert not window_in_dataset(window, loop)

    # Then the chained sync lands the new name on user-info-store.
    user_info = sync_data_source_into_user_info(LOOP_NAME, True, user_info)
    assert user_info["data_source_name"] == LOOP_NAME

    info = append_round_from_window(user_info, _drawn(window), SLIDER_START)
    assert info["rounds"][0]["data_source_name"] == "example.csv"
    assert info["rounds"][0]["is_example_data"] is True
    # /ending's source plaque and the run row read the session's name.
    assert info["data_source_name"] == "example.csv"

    submit = _submit_component(tmp_path)
    submit.save_statistics(info, write_ranking=False)
    saved = _saved_rounds(submit)
    assert [entry["data_source_name"] for entry in saved] == ["example.csv"]
    _assert_every_round_names_its_file(saved)


def test_window_no_file_holds_is_saved_without_a_name() -> None:
    example, _ = load_dataset(EXAMPLE_DATASET_PATH)
    forged = example.slice(SLIDER_START, DEFAULT_POINTS).with_columns(pl.col("gl") + 0.5)
    info = append_round_from_window(
        _session(data_source_name="example.csv", is_example_data=True),
        _drawn(forged),
        SLIDER_START,
    )
    assert info["rounds"][0]["data_source_name"] == ""


def test_every_round_of_a_game_names_the_file_holding_its_values(tmp_path: Path) -> None:
    """Several Format A rounds, one of them drifted, all saved against the right file."""
    random.seed(20261028)
    user_info = _session(data_source_name="example.csv", is_example_data=True)
    for round_number in range(1, 5):
        (_path, user_info, _mode, window_data, _events, _is_example,
         source_name, _rand, start) = handle_next_round_button(round_number, user_info)
        assert user_info["data_source_name"] == source_name
        window = reconstruct_dataframe_from_dict(window_data)
        if round_number == 3:
            # Round 3 drifts: the window is another file's while the name is not.
            example, _ = load_dataset(EXAMPLE_DATASET_PATH)
            loop, _ = load_dataset(LOOP_PATH)
            other = loop if window_in_dataset(window, example) else example
            window = other.slice(SLIDER_START, DEFAULT_POINTS)
        user_info = append_round_from_window(user_info, _drawn(window), start)

    submit = _submit_component(tmp_path)
    submit.save_statistics(user_info, write_ranking=False)
    saved = _saved_rounds(submit)
    assert len(saved) == 4
    assert all(entry["data_source_name"] for entry in saved)
    _assert_every_round_names_its_file(saved)


def _legacy_round(window: pl.DataFrame, round_number: int, source_name: str) -> dict[str, Any]:
    """A round captured before per-round tracking: no name, no is_example flag."""
    info = append_round_from_window(_session(data_source_name=source_name), _drawn(window), 0)
    legacy = dict(info["rounds"][0])
    legacy["round_number"] = round_number
    legacy.pop("data_source_name")
    legacy.pop("is_example_data")
    return legacy


def test_round_without_its_own_name_does_not_take_the_runs_last_source(tmp_path: Path) -> None:
    example, _ = load_dataset(EXAMPLE_DATASET_PATH)
    loop, _ = load_dataset(LOOP_PATH)
    rounds = [
        _legacy_round(example.slice(SLIDER_START, DEFAULT_POINTS), 1, "example.csv"),
        _legacy_round(loop.slice(SLIDER_START, DEFAULT_POINTS), 2, LOOP_NAME),
    ]
    # The run-level name is the last round's source; round 1 is not from it.
    user_info = _session(rounds=rounds, data_source_name=LOOP_NAME, is_example_data=True)
    submit = _submit_component(tmp_path)
    submit.save_statistics(user_info, write_ranking=False)
    saved = _saved_rounds(submit)
    assert [entry["data_source_name"] for entry in saved] == ["", ""]


def test_format_b_round_without_its_own_name_keeps_the_upload(tmp_path: Path) -> None:
    example, _ = load_dataset(EXAMPLE_DATASET_PATH)
    rounds = [_legacy_round(example.slice(SLIDER_START, DEFAULT_POINTS), 1, "example.csv")]
    user_info = _session(
        rounds=rounds,
        format="B",
        run_format="B",
        data_source_name="Clarity_Export.csv",
        is_example_data=False,
    )
    submit = _submit_component(tmp_path)
    submit.save_statistics(user_info, write_ranking=False)
    saved = _saved_rounds(submit)
    assert [entry["data_source_name"] for entry in saved] == ["Clarity_Export.csv"]
