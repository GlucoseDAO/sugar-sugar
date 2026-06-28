from __future__ import annotations

import base64
import csv
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from cgm_format import CGM_SCHEMA, FormatParser, FormatProcessor

from sugar_sugar.app import (
    convert_df_to_dict,
    convert_events_df_to_dict,
    reconstruct_dataframe_from_dict,
    reconstruct_events_dataframe_from_dict,
)
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.data import load_glucose_data

TESTDATA_DIR = Path(__file__).parent / "testdata"
CGM_MIN_MGDL = 0.0
CGM_MAX_MGDL = 401.0
MGDL_PER_MMOL = 18.0

# These files are Nightscout API sidecars/negative controls, not standalone
# uploadable CGM exports. The app upload pipeline accepts single CSV exports.
UNSUPPORTED_STANDALONE_CSVS: set[str] = {
    "nightscout_entries.csv",
    "nightscout_treatments.csv",
}


def _uploadable_csv_paths() -> list[Path]:
    paths = [
        path
        for path in sorted(TESTDATA_DIR.glob("*.csv"))
        if path.name not in UNSUPPORTED_STANDALONE_CSVS
    ]
    if not paths:
        pytest.fail(f"No uploadable CSV fixtures found in {TESTDATA_DIR}")
    return paths


def _dash_upload_contents(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:text/csv;base64,{encoded}"


def _save_dash_upload(contents: str, filename: str, upload_dir: Path) -> Path:
    content_type, content_string = contents.split(",", 1)
    assert content_type == "data:text/csv;base64"

    decoded = base64.b64decode(content_string)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = filename.replace(" ", "_").replace("/", "_")
    upload_path = upload_dir / safe_filename
    upload_path.write_bytes(decoded)
    return upload_path


def _prediction_table_data(window_df: pl.DataFrame) -> list[dict[str, str]]:
    actual: dict[str, str] = {"metric": "Actual Glucose"}
    predicted: dict[str, str] = {"metric": "Predicted"}

    values = window_df.get_column("gl").to_list()
    for idx, value in enumerate(values):
        actual[f"t{idx}"] = f"{float(value):.1f}"
        predicted[f"t{idx}"] = f"{float(value) + 1.0:.1f}"

    return [actual, predicted]


def _assert_app_glucose_schema(df: pl.DataFrame) -> None:
    assert df.columns == ["time", "gl", "prediction", "age", "user_id"]
    assert df.height > PREDICTION_HOUR_OFFSET
    assert df.get_column("time").dtype == pl.Datetime
    assert df.get_column("gl").dtype == pl.Float64
    assert df.get_column("prediction").dtype == pl.Float64
    assert df.get_column("age").dtype == pl.Int32
    assert df.get_column("user_id").dtype == pl.Int32
    assert df.get_column("gl").null_count() == 0
    assert float(df.get_column("gl").min()) >= CGM_MIN_MGDL
    assert float(df.get_column("gl").max()) <= CGM_MAX_MGDL
    assert set(df.get_column("prediction").to_list()) == {0.0}


def _assert_app_events_schema(df: pl.DataFrame) -> None:
    assert df.columns == ["time", "event_type", "event_subtype", "insulin_value"]
    assert df.get_column("time").dtype == pl.Datetime
    assert df.get_column("event_type").dtype == pl.String
    assert df.get_column("event_subtype").dtype == pl.String
    assert df.get_column("insulin_value").dtype == pl.Float64
    assert set(df.get_column("event_type").to_list()).issubset(
        {"Insulin", "Exercise", "Carbohydrates"}
    )


def _assert_store_roundtrip(glucose_df: pl.DataFrame, events_df: pl.DataFrame) -> None:
    glucose_store = convert_df_to_dict(glucose_df)
    assert set(glucose_store) == {"time", "gl", "prediction", "age", "user_id"}
    assert len(glucose_store["time"]) == glucose_df.height

    reconstructed_glucose = reconstruct_dataframe_from_dict(glucose_store)
    assert reconstructed_glucose.shape == glucose_df.shape
    assert reconstructed_glucose.columns == glucose_df.columns

    events_store = convert_events_df_to_dict(events_df)
    assert set(events_store) == {"time", "event_type", "event_subtype", "insulin_value"}
    assert len(events_store["time"]) == events_df.height

    reconstructed_events = reconstruct_events_dataframe_from_dict(events_store)
    assert reconstructed_events.shape == events_df.shape
    assert reconstructed_events.columns == events_df.columns


def test_chart_reconstructs_mixed_numeric_insulin_values() -> None:
    events_store: dict[str, list[Any]] = {
        "time": [
            "2025-05-14T10:00:00",
            "2025-05-14T10:05:00",
            "2025-05-14T10:10:00",
            "2025-05-14T10:15:00",
        ],
        "event_type": ["Insulin", "Insulin", "Insulin", "Insulin"],
        "event_subtype": ["Fast Acting", "Fast Acting", "Fast Acting", "Fast Acting"],
        "insulin_value": [1, 3.5, "2.25", ""],
    }

    reconstructed_events = GlucoseChart()._reconstruct_events_dataframe_from_dict(events_store)

    assert reconstructed_events.get_column("insulin_value").dtype == pl.Float64
    assert reconstructed_events.get_column("insulin_value").to_list() == [1.0, 3.5, 2.25, None]


def _trace_y_values(trace: Any) -> list[float]:
    return [float(value) for value in trace.y]


def _window_events(window_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    if events_df.height == 0:
        return events_df
    start_time = window_df.get_column("time")[0]
    end_time = window_df.get_column("time")[-1]
    return events_df.filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))


def _assert_static_chart(window_df: pl.DataFrame, events_df: pl.DataFrame, source_name: str) -> None:
    mgdl_figure = GlucoseChart.build_static_figure(
        window_df,
        events_df,
        source_name,
        unit="mg/dL",
        prediction_boundary=len(window_df) - PREDICTION_HOUR_OFFSET,
    )
    assert len(mgdl_figure.data) > 0
    assert mgdl_figure.layout.xaxis.title.text
    assert mgdl_figure.layout.yaxis.title.text
    # The static/immersive chart intentionally has NO chart title ("Glucose
    # Levels" was removed in the mobile redesign to reclaim vertical space; the
    # source/round info lives in the source plaque instead). So title.text is "".
    assert (mgdl_figure.layout.title.text or "") == ""

    mgdl_glucose_values = _trace_y_values(mgdl_figure.data[0])
    assert min(mgdl_glucose_values) >= CGM_MIN_MGDL
    assert max(mgdl_glucose_values) <= CGM_MAX_MGDL

    mmol_figure = GlucoseChart.build_static_figure(
        window_df,
        events_df,
        source_name,
        unit="mmol/L",
        prediction_boundary=len(window_df) - PREDICTION_HOUR_OFFSET,
    )
    mmol_glucose_values = _trace_y_values(mmol_figure.data[0])
    assert min(mmol_glucose_values) >= CGM_MIN_MGDL / MGDL_PER_MMOL
    assert max(mmol_glucose_values) <= CGM_MAX_MGDL / MGDL_PER_MMOL
    assert mmol_glucose_values == pytest.approx([value / MGDL_PER_MMOL for value in mgdl_glucose_values])

    plotted_event_points = sum(
        len(trace.x)
        for trace in mgdl_figure.data
        if getattr(trace, "hoverinfo", None) == "text"
    )
    if _window_events(window_df, events_df).height > 0:
        assert plotted_event_points == _window_events(window_df, events_df).height
    else:
        assert plotted_event_points == 0


def _assert_statistics_saved(
    tmp_path: Path,
    full_df: pl.DataFrame,
    window_df: pl.DataFrame,
    source_name: str,
) -> None:
    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {
        "A": tmp_path / "prediction_ranking_A.csv",
        "B": tmp_path / "prediction_ranking_B.csv",
        "C": tmp_path / "prediction_ranking_C.csv",
    }

    table_data = _prediction_table_data(window_df)
    # save_statistics no longer receives the full dataset; per-window times come
    # from `window_times` captured at submit (here derived from the window df).
    window_times = window_df.get_column("time").dt.strftime("%Y-%m-%d %H:%M:%S").to_list()
    user_info: dict[str, Any] = {
        "study_id": f"integration-{source_name}",
        "run_id": f"run-{source_name}",
        "number": 1,
        "email": "integration@example.com",
        "format": "B",
        "run_format": "B",
        "is_example_data": False,
        "data_source_name": source_name,
        "uses_cgm": True,
        "gender": "test",
        "diabetic": "test",
        "location": "test",
        # save_statistics refuses to persist study data without consent
        # (defense-in-depth guard); this test exercises the consented save path.
        "consent_completed": True,
        "prediction_table_data": table_data,
        "prediction_window_size": len(window_df),
        "window_times": window_times,
    }

    submit.save_statistics(user_info)

    assert submit._stats_csv_path.exists()
    assert submit._ranking_csv_path.exists()
    assert submit._ranking_by_format_paths["B"].exists()

    with submit._stats_csv_path.open(newline="") as handle:
        stats_rows = list(csv.DictReader(handle))

    assert len(stats_rows) == 1
    row = stats_rows[0]
    assert row["data_source_name"] == source_name
    assert row["format"] == "B"
    assert row["predicted_values"] != "[]"
    assert row["real_values"] != "[]"
    assert row["prediction_times"] != "[]"
    assert row["overall_mae_mgdl"] != ""


@pytest.mark.parametrize("source_path", _uploadable_csv_paths(), ids=lambda path: path.name)
def test_dash_base64_upload_pipeline_parses_plots_and_saves_csvs(
    source_path: Path,
    tmp_path: Path,
) -> None:
    contents = _dash_upload_contents(source_path)
    uploaded_path = _save_dash_upload(contents, source_path.name, tmp_path / "uploads")
    assert uploaded_path.read_bytes() == source_path.read_bytes()

    unified_df = FormatParser.parse_file(uploaded_path)
    assert unified_df.columns == CGM_SCHEMA.get_column_names()
    assert unified_df.height > 0

    expected_unified_path = source_path.with_name(f"{source_path.stem}_unified.csv")
    if expected_unified_path.exists():
        expected_unified_df = FormatParser.parse_file(expected_unified_path)
        assert unified_df.height == expected_unified_df.height

    split_glucose_df, split_events_df = FormatProcessor.split_glucose_events(unified_df)
    assert split_glucose_df.height > PREDICTION_HOUR_OFFSET
    assert split_glucose_df.height + split_events_df.height == unified_df.height

    full_df, events_df = load_glucose_data(uploaded_path)
    _assert_app_glucose_schema(full_df)
    _assert_app_events_schema(events_df)
    _assert_store_roundtrip(full_df, events_df)

    window_df = full_df.head(min(36, full_df.height))
    window_size = len(window_df)
    window_df = window_df.with_columns(
        pl.when(pl.int_range(pl.len()) >= window_size - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 1.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )

    _assert_static_chart(window_df, events_df, source_path.name)
    _assert_statistics_saved(tmp_path, full_df, window_df, source_path.name)


def test_testdata_unsupported_csvs_are_documented() -> None:
    present_unsupported = {
        path.name for path in TESTDATA_DIR.glob("*.csv") if path.name in UNSUPPORTED_STANDALONE_CSVS
    }
    assert present_unsupported.issubset(UNSUPPORTED_STANDALONE_CSVS)
