from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from sugar_sugar.data import load_loop_chronological_data
from sugar_sugar.generic_sources_metadata import load_generic_sources_metadata
from sugar_sugar.prediction_window_context import (
    should_show_no_carbs_note,
    window_covers_meal_time,
    window_has_carb_events,
)
from sugar_sugar.subject_sources import discover_generic_dataset_sources


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
