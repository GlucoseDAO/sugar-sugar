from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from sugar_sugar.config import DEFAULT_POINTS
from sugar_sugar.data import load_loop_chronological_data
from sugar_sugar.generic_sources_metadata import (
    format_generic_source_metadata,
    format_participant_demographics,
    load_generic_sources_metadata,
)
from sugar_sugar.prediction_window_context import (
    should_show_no_carbs_note,
    window_covers_meal_time,
    window_has_carb_events,
)
from sugar_sugar.subject_sources import (
    GenericRoundWindow,
    collect_generic_round_history,
    discover_generic_dataset_sources,
    generic_round_window_from_df,
    generic_window_slice_key,
    generic_window_slice_key_from_round,
    load_generic_dataset_source,
    pick_unique_generic_window,
    resolve_generic_source_path,
    windows_conflict,
)


def _window_df(times: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [datetime.fromisoformat(value) for value in times],
            "gl": [100.0] * len(times),
            "prediction": [0.0] * len(times),
            "age": [0] * len(times),
            "user_id": [1] * len(times),
        }
    )


def test_under_18_subjects_are_excluded() -> None:
    sources = discover_generic_dataset_sources()
    names = {source.source_name for source in sources}
    assert "loop_154_chronological.csv" not in names
    assert "loop_1017_chronological.csv" not in names
    assert "loop_1029_chronological.csv" not in names
    assert "example.csv" in names
    assert any(name.startswith("loop_") for name in names)


def test_adult_subject_metadata_is_available() -> None:
    metadata = load_generic_sources_metadata()
    loop_meta = metadata["loop_556_chronological.csv"]
    assert loop_meta.age == "23 years old"
    assert loop_meta.gender == "female"
    assert loop_meta.weight == "84 kg"


def test_loop_chronological_csv_loads_glucose_and_events() -> None:
    csv_path = Path("data/subjects/loop_556/loop_556_chronological.csv")
    glucose_df, events_df = load_loop_chronological_data(csv_path)
    assert len(glucose_df) > 30000
    assert {"time", "gl", "prediction", "age", "user_id"}.issubset(set(glucose_df.columns))
    assert len(events_df) > 0
    assert "Carbohydrates" in events_df.get_column("event_type").to_list()


def test_meal_window_without_carbs_triggers_note() -> None:
    window_df = _window_df(
        [
            "2019-04-15T11:30:00",
            "2019-04-15T11:35:00",
            "2019-04-15T11:40:00",
        ]
    )
    events_df = pl.DataFrame(
        {
            "time": [datetime(2019, 4, 15, 8, 0, 0)],
            "event_type": ["Insulin"],
            "event_subtype": ["Fast Acting"],
            "insulin_value": [2.0],
        }
    )
    assert window_covers_meal_time(window_df)
    assert not window_has_carb_events(window_df, events_df)
    assert should_show_no_carbs_note(window_df, events_df)


def test_night_window_without_carbs_does_not_trigger_note() -> None:
    window_df = _window_df(
        [
            "2019-04-15T02:00:00",
            "2019-04-15T02:05:00",
            "2019-04-15T02:10:00",
        ]
    )
    events_df = pl.DataFrame(
        {
            "time": [],
            "event_type": [],
            "event_subtype": [],
            "insulin_value": [],
        }
    )
    assert not window_covers_meal_time(window_df)
    assert not should_show_no_carbs_note(window_df, events_df)


def test_meal_window_with_carbs_does_not_trigger_note() -> None:
    window_df = _window_df(
        [
            "2019-04-15T12:00:00",
            "2019-04-15T12:05:00",
            "2019-04-15T12:10:00",
        ]
    )
    events_df = pl.DataFrame(
        {
            "time": [datetime(2019, 4, 15, 12, 0, 0)],
            "event_type": ["Carbohydrates"],
            "event_subtype": ["Carbs"],
            "insulin_value": [None],
        }
    )
    assert window_covers_meal_time(window_df)
    assert window_has_carb_events(window_df, events_df)
    assert not should_show_no_carbs_note(window_df, events_df)


def test_participant_demographics_match_example_style() -> None:
    line = format_participant_demographics(28, "F", locale="en")
    assert line == "28 yr old Female"


def test_participant_demographics_include_carbs_notes() -> None:
    line = format_participant_demographics(
        28,
        "F",
        locale="en",
        show_carbs_info_note=True,
        show_no_carbs_note=True,
    )
    assert line.startswith("28 yr old Female · ")
    assert "Carb markers appear only when your dataset includes them." in line
    assert "This meal-time window has no logged carbohydrate data." in line


def test_generic_metadata_english_without_weight_uses_age_gender_only() -> None:
    metadata = load_generic_sources_metadata()
    loop_meta = metadata["loop_556_chronological.csv"]
    line = format_generic_source_metadata(loop_meta, locale="en")
    assert "23 yr old female" in line.lower()
    assert "84" in line


def test_zero_insulin_values_are_excluded_from_loop_events() -> None:
    csv_path = Path("data/subjects/loop_556/loop_556_chronological.csv")
    _, events_df = load_loop_chronological_data(csv_path)
    insulin = events_df.filter(pl.col("event_type") == "Insulin")
    assert insulin.height == 0 or insulin.filter(pl.col("insulin_value") == 0).height == 0


def test_generic_slice_key_is_stable_for_same_window() -> None:
    sources = discover_generic_dataset_sources()
    source = next(s for s in sources if s.source_name == "example.csv")
    full_df, _ = load_generic_dataset_source(source)
    window_a = full_df.slice(0, DEFAULT_POINTS)
    window_b = full_df.slice(0, DEFAULT_POINTS)
    assert generic_window_slice_key(window_a) == generic_window_slice_key(window_b)


def test_windows_conflict_blocks_same_source_within_two_hours() -> None:
    prior = GenericRoundWindow(
        source_name="loop_556_chronological.csv",
        window_start=datetime(2020, 1, 1, 18, 0, 0),
        window_end=datetime(2020, 1, 1, 21, 0, 0),
        anchor_time=datetime(2020, 1, 1, 20, 0, 0),
        slice_key="a",
    )
    near_same_source = GenericRoundWindow(
        source_name="loop_556_chronological.csv",
        window_start=datetime(2020, 1, 1, 22, 0, 0),
        window_end=datetime(2020, 1, 2, 1, 0, 0),
        anchor_time=datetime(2020, 1, 1, 0, 0, 0),
        slice_key="b",
    )
    far_same_source = GenericRoundWindow(
        source_name="loop_556_chronological.csv",
        window_start=datetime(2020, 1, 2, 0, 0, 0),
        window_end=datetime(2020, 1, 2, 3, 0, 0),
        anchor_time=datetime(2020, 1, 2, 2, 0, 0),
        slice_key="c",
    )
    other_source_same_clock = GenericRoundWindow(
        source_name="loop_730_chronological.csv",
        window_start=datetime(2020, 6, 1, 18, 0, 0),
        window_end=datetime(2020, 6, 1, 21, 0, 0),
        anchor_time=datetime(2020, 6, 1, 20, 0, 0),
        slice_key="d",
    )
    assert windows_conflict(prior, near_same_source)
    assert not windows_conflict(prior, far_same_source)
    assert not windows_conflict(prior, other_source_same_clock)


def test_windows_conflict_blocks_exact_slice_key() -> None:
    prior = GenericRoundWindow(
        source_name="loop_556_chronological.csv",
        window_start=datetime(2020, 1, 1, 18, 0, 0),
        window_end=datetime(2020, 1, 1, 21, 0, 0),
        anchor_time=datetime(2020, 1, 1, 20, 0, 0),
        slice_key="same",
    )
    duplicate = GenericRoundWindow(
        source_name="example.csv",
        window_start=datetime(2019, 4, 15, 8, 0, 0),
        window_end=datetime(2019, 4, 15, 11, 0, 0),
        anchor_time=datetime(2019, 4, 15, 10, 0, 0),
        slice_key="same",
    )
    assert windows_conflict(prior, duplicate)


def test_pick_unique_generic_window_avoids_used_slices() -> None:
    sources = discover_generic_dataset_sources()
    source = next(s for s in sources if s.source_name == "example.csv")
    full_df, _ = load_generic_dataset_source(source)
    first = generic_round_window_from_df(full_df.slice(0, DEFAULT_POINTS), source_name=source.source_name)

    selection = pick_unique_generic_window(DEFAULT_POINTS, [first])
    assert selection.slice_key != first.slice_key


def test_pick_unique_generic_window_avoids_same_source_two_hour_overlap() -> None:
    sources = discover_generic_dataset_sources()
    source = next(s for s in sources if s.source_name == "example.csv")
    full_df, _ = load_generic_dataset_source(source)
    first_window = generic_round_window_from_df(
        full_df.slice(0, DEFAULT_POINTS),
        source_name=source.source_name,
    )

    history = [first_window]
    for _ in range(5):
        selection = pick_unique_generic_window(DEFAULT_POINTS, history)
        chosen = generic_round_window_from_df(
            selection.window_df,
            source_name=selection.source.source_name,
        )
        assert not any(windows_conflict(prior, chosen) for prior in history)
        history.append(chosen)


def test_resolve_generic_source_path_maps_known_sources() -> None:
    example_path = resolve_generic_source_path("example.csv")
    assert example_path is not None
    assert example_path.name == "example.csv"
    loop_path = resolve_generic_source_path("loop_556_chronological.csv")
    assert loop_path is not None
    assert loop_path.name == "loop_556_chronological.csv"
    assert resolve_generic_source_path("missing.csv") is None


def test_collect_generic_round_history_reads_completed_rounds() -> None:
    sources = discover_generic_dataset_sources()
    source = next(s for s in sources if s.source_name == "example.csv")
    full_df, _ = load_generic_dataset_source(source)
    window = full_df.slice(0, DEFAULT_POINTS)
    times = window.get_column("time").dt.strftime("%Y-%m-%d %H:%M:%S").to_list()
    gl = window.get_column("gl").to_list()
    table = [
        {
            "metric": "Actual Glucose",
            **{f"t{i}": f"{float(v):.1f}" for i, v in enumerate(gl)},
        }
    ]
    slice_key = generic_window_slice_key(window)
    rounds = [
        {
            "round_number": 1,
            "is_example_data": True,
            "window_times": times,
            "prediction_table_data": table,
            "generic_slice_key": slice_key,
            "data_source_name": source.source_name,
        }
    ]
    history = collect_generic_round_history(rounds, None)
    assert len(history) == 1
    assert history[0].slice_key == slice_key
    assert generic_window_slice_key_from_round(rounds[0]) == slice_key
