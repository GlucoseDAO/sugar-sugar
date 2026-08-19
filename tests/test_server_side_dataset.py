"""Tests for the server-side dataset refactor (no full-df client store).

Covers: the load_dataset cache, dataset-identity resolution, format-C per-round
time correctness in save_statistics, that create_ending_layout renders from the
window alone (no full-df), and that handle_next_round_button returns a tuple whose
arity matches its (full-df-free) Output list.
"""
from __future__ import annotations

import ast
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

    # Format C: round 1 is the generic warm-up, then own/generic alternate
    # (ODD round -> generic example, EVEN round -> uploaded own data).
    info_c = {"format": "C", "uploaded_data_path": uploaded}
    assert resolve_dataset_identity(info_c, round_number=1) == example  # odd -> example (warm-up)
    assert resolve_dataset_identity(info_c, round_number=2) == Path(uploaded)  # even -> uploaded
    assert resolve_dataset_identity(info_c, round_number=3) == example  # odd -> example


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


def test_save_statistics_records_round_context(tmp_path: Path) -> None:
    """Each result remains traceable to its source and exact time window."""
    import csv

    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    window = _window_with_predictions()
    times = _times(window)
    prediction_start_index = len(window) - PREDICTION_HOUR_OFFSET
    user_info: dict[str, Any] = {
        "study_id": "round-context",
        "run_id": "run-a",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "age": 40,
        "rounds": [
            {
                "round_number": 1,
                "prediction_window_start": 123,
                "prediction_window_size": len(window),
                "prediction_table_data": _table_data(window),
                "window_times": times,
                "format": "A",
                "is_example_data": True,
                "data_source_name": "BIGIDEAS-001.csv",
                "generic_slice_key": "slice-abc",
            }
        ],
    }
    submit.save_statistics(user_info, write_ranking=False)

    with submit._stats_csv_path.open(newline="") as file_handle:
        row = list(csv.DictReader(file_handle))[0]
    context = ast.literal_eval(row["round_context"])
    assert context == [
        {
            "round_number": 1,
            "format": "A",
            "data_source_name": "BIGIDEAS-001.csv",
            "is_example_data": True,
            "generic_slice_key": "slice-abc",
            "prediction_window_start_index": 123,
            "prediction_window_size": len(window),
            "window_start_time": times[0],
            "prediction_start_time": times[prediction_start_index],
            "window_end_time": times[-1],
        }
    ]


def test_round_context_upgrade_preserves_existing_statistics_rows(tmp_path: Path) -> None:
    """Adding round_context must not rewrite or discard previously saved games."""
    import csv

    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}
    submit._stats_csv_path.write_text(
        "study_id,run_id,format,predicted_values\n"
        "existing-game,old-run,A,old-predictions\n",
        encoding="utf-8",
    )

    window = _window_with_predictions()
    submit.save_statistics(
        {
            "study_id": "new-game",
            "run_id": "new-run",
            "number": 33,
            "consent_completed": True,
            "format": "A",
            "run_format": "A",
            "age": 40,
            "rounds": [
                {
                    "round_number": 1,
                    "prediction_window_size": len(window),
                    "prediction_table_data": _table_data(window),
                    "window_times": _times(window),
                    "format": "A",
                    "is_example_data": True,
                    "data_source_name": "example.csv",
                }
            ],
        },
        write_ranking=False,
    )

    with submit._stats_csv_path.open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    by_study = {row["study_id"]: row for row in rows}
    assert set(by_study) == {"existing-game", "new-game"}
    assert by_study["existing-game"]["predicted_values"] == "old-predictions"
    assert by_study["existing-game"]["round_context"] == ""
    assert ast.literal_eval(by_study["new-game"]["round_context"])[0]["data_source_name"] == "example.csv"


def test_save_statistics_records_per_round_generic_sources(tmp_path: Path) -> None:
    """Format A picks a new generic subject each round; the stats CSV must
    keep every source, not only the last ``user_info['data_source_name']``."""
    import csv

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
    user_info: dict[str, Any] = {
        "study_id": "generic-trace",
        "run_id": "r-a",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "age": 40,
        "data_source_name": "D1NAMO-002.csv",
        "rounds": [
            {
                "round_number": 1,
                "prediction_window_size": len(w1),
                "prediction_table_data": _table_data(w1),
                "window_times": _times(w1),
                "format": "A",
                "is_example_data": True,
                "data_source_name": "BIGIDEAS-001.csv",
                "generic_slice_key": "slice-aaa",
            },
            {
                "round_number": 2,
                "prediction_window_size": len(w2),
                "prediction_table_data": _table_data(w2),
                "window_times": _times(w2),
                "format": "A",
                "is_example_data": True,
                "data_source_name": "D1NAMO-002.csv",
                "generic_slice_key": "slice-bbb",
            },
        ],
    }
    submit.save_statistics(user_info)

    with submit._stats_csv_path.open(newline="") as fh:
        row = list(csv.DictReader(fh))[0]
    per_round = ast.literal_eval(row["per_round_metrics"])
    assert [entry["data_source_name"] for entry in per_round] == [
        "BIGIDEAS-001.csv",
        "D1NAMO-002.csv",
    ]
    assert [entry["generic_slice_key"] for entry in per_round] == [
        "slice-aaa",
        "slice-bbb",
    ]
    assert [entry["is_example_data"] for entry in per_round] == [True, True]


def test_save_statistics_records_per_round_sources_for_b_and_c(tmp_path: Path) -> None:
    """Format B is one uploaded file; Format C alternates generic + own data.
    Both must keep per-round ``data_source_name`` and ``generic_slice_key``."""
    import csv

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

    def _save(study_id: str, fmt: str, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        user_info: dict[str, Any] = {
            "study_id": study_id,
            "run_id": f"run-{fmt}",
            "number": 1,
            "consent_completed": True,
            "format": fmt,
            "run_format": fmt,
            "age": 40,
            "data_source_name": str(rounds[-1]["data_source_name"]),
            "is_example_data": bool(rounds[-1]["is_example_data"]),
            "rounds": rounds,
        }
        submit.save_statistics(user_info)
        with submit._stats_csv_path.open(newline="") as fh:
            rows = [row for row in csv.DictReader(fh) if row["study_id"] == study_id]
        return ast.literal_eval(rows[-1]["per_round_metrics"])

    own = "Clarity_Export.csv"
    b_rounds = [
        {
            "round_number": 1,
            "prediction_window_size": len(w1),
            "prediction_table_data": _table_data(w1),
            "window_times": _times(w1),
            "format": "B",
            "is_example_data": False,
            "data_source_name": own,
            "generic_slice_key": "own-window-1",
        },
        {
            "round_number": 2,
            "prediction_window_size": len(w2),
            "prediction_table_data": _table_data(w2),
            "window_times": _times(w2),
            "format": "B",
            "is_example_data": False,
            "data_source_name": own,
            "generic_slice_key": "own-window-2",
        },
    ]
    b_metrics = _save("fmt-b", "B", b_rounds)
    assert [entry["data_source_name"] for entry in b_metrics] == [own, own]
    assert [entry["generic_slice_key"] for entry in b_metrics] == [
        "own-window-1",
        "own-window-2",
    ]
    assert [entry["is_example_data"] for entry in b_metrics] == [False, False]

    c_rounds = [
        {
            "round_number": 1,
            "prediction_window_size": len(w1),
            "prediction_table_data": _table_data(w1),
            "window_times": _times(w1),
            "format": "C",
            "is_example_data": True,
            "data_source_name": "BIGIDEAS-001.csv",
            "generic_slice_key": "generic-odd",
        },
        {
            "round_number": 2,
            "prediction_window_size": len(w2),
            "prediction_table_data": _table_data(w2),
            "window_times": _times(w2),
            "format": "C",
            "is_example_data": False,
            "data_source_name": own,
            "generic_slice_key": "own-even",
        },
    ]
    c_metrics = _save("fmt-c-mix", "C", c_rounds)
    assert [entry["data_source_name"] for entry in c_metrics] == [
        "BIGIDEAS-001.csv",
        own,
    ]
    assert [entry["generic_slice_key"] for entry in c_metrics] == [
        "generic-odd",
        "own-even",
    ]
    assert [entry["is_example_data"] for entry in c_metrics] == [True, False]


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
    result = handle_next_round_button(1, 0, user_info)
    # url, user-info, chart-mode, current-window, events, is-example, source,
    # randomization-initialized, initial-slider  == 9 (full-df dropped).
    assert len(result) == 9
    assert result[0] == "/prediction"


def test_producer_callback_arities_have_no_full_df() -> None:
    """Other producers whose Output lists lost full-df: confirm their return
    arity matches (these fire without a Dash callback context)."""
    # handle_example_data_button: 8 outputs (full-df dropped).
    assert len(handle_example_data_button(0, None)) == 8  # no-click early return
    assert len(handle_example_data_button(1, None)) == 8  # active path

    # initialize_data_on_url_change: 6 outputs (full-df dropped).
    assert len(initialize_data_on_url_change("/about", None, None)) == 6  # non-prediction

    # handle_nightscout_load error path: 8 no-update + status == 9 outputs.
    res = handle_nightscout_load(1, "", None, None, None)
    assert len(res) == 9


def test_save_statistics_upserts_incomplete_then_complete(tmp_path: Path) -> None:
    """Start (0 rounds) then first submit must update the same study_id row."""
    import csv

    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    window = _window_with_predictions()
    times = _times(window)
    user_info: dict[str, Any] = {
        "study_id": "forgot-exit",
        "run_id": "r1",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "age": 30,
        "email": "forgot@example.com",
        "rounds": [],
    }
    submit.save_statistics(user_info)
    with submit._stats_csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["rounds_played"] == "0"
    assert not submit._ranking_csv_path.exists()

    user_info["rounds"] = [
        {
            "round_number": 1,
            "prediction_window_size": len(window),
            "prediction_table_data": _table_data(window),
            "window_times": times,
            "format": "A",
            "is_example_data": True,
            "data_source_name": "example.csv",
        }
    ]
    submit.save_statistics(user_info)
    with submit._stats_csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["study_id"] == "forgot-exit"
    assert rows[0]["rounds_played"] == "1"
    # Short runs are stored in ranking CSVs too; the public board hides them.
    assert submit._ranking_csv_path.exists()
    with submit._ranking_csv_path.open(newline="") as fh:
        ranking_rows = list(csv.DictReader(fh))
    assert len(ranking_rows) == 1
    assert ranking_rows[0]["rounds_played"] == "1"


def test_save_statistics_keeps_each_format_run(tmp_path: Path) -> None:
    """Two rounds in A then two in B must leave both format rows on disk."""
    import csv

    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    window = _window_with_predictions()
    times = _times(window)
    one_round = {
        "round_number": 1,
        "prediction_window_size": len(window),
        "prediction_table_data": _table_data(window),
        "window_times": times,
        "is_example_data": True,
        "data_source_name": "example.csv",
    }
    user_info: dict[str, Any] = {
        "study_id": "multi-format",
        "run_id": "run-a",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "age": 30,
        "email": "fields@example.com",
        "rounds": [{**one_round, "round_number": i, "format": "A"} for i in (1, 2)],
    }
    submit.save_statistics(user_info)

    user_info["run_id"] = "run-b"
    user_info["format"] = "B"
    user_info["run_format"] = "B"
    user_info["rounds"] = [{**one_round, "round_number": i, "format": "B"} for i in (1, 2)]
    user_info["runs_by_format"] = {
        "A": [
            {
                "run_id": "archive-a",
                "active_run_id": "run-a",
                "format": "A",
                "rounds": [{**one_round, "round_number": i, "format": "A"} for i in (1, 2)],
                "is_example_data": True,
                "data_source_name": "example.csv",
            }
        ]
    }
    submit.save_statistics(user_info)

    with submit._stats_csv_path.open(newline="") as fh:
        stats = list(csv.DictReader(fh))
    by_format = {row["format"]: row for row in stats}
    assert set(by_format) == {"A", "B"}
    assert by_format["A"]["rounds_played"] == "2"
    assert by_format["B"]["rounds_played"] == "2"
    assert by_format["A"]["run_id"] == "run-a"
    assert by_format["B"]["run_id"] == "run-b"

    with submit._ranking_by_format_paths["A"].open(newline="") as fh:
        fmt_a = list(csv.DictReader(fh))
    with submit._ranking_by_format_paths["B"].open(newline="") as fh:
        fmt_b = list(csv.DictReader(fh))
    assert len(fmt_a) == 1 and fmt_a[0]["rounds_played"] == "2"
    assert len(fmt_b) == 1 and fmt_b[0]["rounds_played"] == "2"

    with submit._ranking_csv_path.open(newline="") as fh:
        overall = list(csv.DictReader(fh))
    assert len(overall) == 1
    assert overall[0]["rounds_played"] == "4"


def test_save_statistics_writes_ranking_at_min_useful_rounds(tmp_path: Path) -> None:
    """A full useful run still writes both the format and overall ranking rows."""
    import csv

    from sugar_sugar.config import MIN_USEFUL_ROUNDS

    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    window = _window_with_predictions()
    times = _times(window)
    one_round = {
        "round_number": 1,
        "prediction_window_size": len(window),
        "prediction_table_data": _table_data(window),
        "window_times": times,
        "format": "A",
        "is_example_data": True,
        "data_source_name": "example.csv",
    }
    user_info: dict[str, Any] = {
        "study_id": "six-rounds",
        "run_id": "r1",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "age": 30,
        "email": "six@example.com",
        "rounds": [{**one_round, "round_number": i} for i in range(1, MIN_USEFUL_ROUNDS + 1)],
    }
    submit.save_statistics(user_info)
    with submit._ranking_csv_path.open(newline="") as fh:
        overall = list(csv.DictReader(fh))
    with submit._ranking_by_format_paths["A"].open(newline="") as fh:
        fmt_a = list(csv.DictReader(fh))
    assert len(overall) == 1
    assert overall[0]["rounds_played"] == str(MIN_USEFUL_ROUNDS)
    assert len(fmt_a) == 1
    assert fmt_a[0]["rounds_played"] == str(MIN_USEFUL_ROUNDS)
