from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import polars as pl
import pytest

from sugar_sugar.bigideas import (
    discover_bigideas_sources,
    format_bigideas_frames,
    is_bigideas_path,
    is_bigideas_source_name,
    load_bigideas_data,
    load_bigideas_demographics,
    subject_id_from_path,
)
from sugar_sugar.components.glucose import GlucoseChart, meal_food_bubble_children
from sugar_sugar.data import load_glucose_data
from sugar_sugar.download_bigideas import dataset_is_present, default_dest
from sugar_sugar.subject_sources import (
    GENERIC_INTERVENTION_BIGIDEAS,
    GENERIC_INTERVENTION_D1NAMO,
    GENERIC_INTERVENTION_MIX_LADA,
    GENERIC_INTERVENTION_MIX_PREDIABETES,
    GENERIC_INTERVENTION_MIX_T2,
    discover_generic_dataset_sources,
    generic_intervention_for_user,
    intervention_pool_weights,
    load_generic_dataset_source,
    resolve_generic_source_path,
)

FIXTURE_ROOT = Path(__file__).parent / "testdata" / "bigideas"
SUBJECT_001 = FIXTURE_ROOT / "001" / "Dexcom_001.csv"
SUBJECT_002 = FIXTURE_ROOT / "002" / "Dexcom_002.csv"


def test_is_bigideas_path_detects_dexcom_and_virtual_name() -> None:
    assert is_bigideas_path(SUBJECT_001)
    assert is_bigideas_path(Path("BIGIDEAS-001.csv"))
    assert is_bigideas_source_name("BIGIDEAS-001.csv")
    assert not is_bigideas_path(Path("data/example.csv"))
    assert not is_bigideas_path(Path("D1NAMO-001.csv"))


def test_subject_id_from_path() -> None:
    assert subject_id_from_path(SUBJECT_001) == "001"
    assert subject_id_from_path(Path("BIGIDEAS-007.csv")) == "007"
    assert subject_id_from_path(Path("other.csv")) is None


def test_demographics_and_discover(monkeypatch: pytest.MonkeyPatch) -> None:
    bio = load_bigideas_demographics(FIXTURE_ROOT)
    assert bio["001"] == ("female", "5.5")
    assert bio["002"][0] == "male"
    sources = discover_bigideas_sources(FIXTURE_ROOT)
    names = {source.source_name for source in sources}
    assert names == {"BIGIDEAS-001.csv", "BIGIDEAS-002.csv"}
    first = next(source for source in sources if source.subject_id == "001")
    assert first.sensor == "Dexcom G6"
    assert first.gender == "female"


def test_format_keeps_egv_drops_calibration_and_clusters_food() -> None:
    raw = pl.read_csv(SUBJECT_001)
    food = pl.read_csv(SUBJECT_001.parent / "Food_Log_001.csv")
    glucose_df, events_df = format_bigideas_frames(raw, food_df=food, subject_id=1)
    assert glucose_df.height == 9
    assert datetime(2020, 2, 13, 18, 0) in glucose_df.get_column("time").to_list()
    assert 99.0 not in glucose_df.get_column("gl").to_list()
    notes = events_df.get_column("food_note").to_list()
    assert len(notes) == 2
    assert "Berry Smoothie (20.0 fluid ounce)" in notes[0]
    assert "Chicken Leg (1.0)" in notes[0]
    assert notes[1].startswith("Asparagus")
    assert events_df.get_column("photo_path").to_list() == ["", ""]
    assert events_df.get_column("carbs_g").to_list()[0] == 85.0


def test_load_glucose_data_routes_bigideas() -> None:
    glucose_df, events_df = load_glucose_data(SUBJECT_001)
    assert "gl" in glucose_df.columns
    assert "food_note" in events_df.columns
    assert events_df.filter(pl.col("food_note") != "").height == 2


def test_generic_pipeline_resolves_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_bigideas_sources",
        lambda dest=None: discover_bigideas_sources(FIXTURE_ROOT),
    )
    path = resolve_generic_source_path("BIGIDEAS-001.csv")
    assert path == SUBJECT_001
    sources = discover_generic_dataset_sources(intervention=GENERIC_INTERVENTION_BIGIDEAS)
    source = next(item for item in sources if item.source_name == "BIGIDEAS-001.csv")
    glucose_df, events_df = load_generic_dataset_source(source)
    assert glucose_df.height > 0
    assert "Berry Smoothie" in events_df.get_column("food_note").to_list()[0]


def test_generic_intervention_follows_diabetes_type() -> None:
    assert generic_intervention_for_user({"diabetic": False}) == GENERIC_INTERVENTION_BIGIDEAS
    assert generic_intervention_for_user(None) == GENERIC_INTERVENTION_BIGIDEAS
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 1"}) == (
        GENERIC_INTERVENTION_D1NAMO
    )
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 2"}) == (
        GENERIC_INTERVENTION_MIX_T2
    )
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Prediabetes"}) == (
        GENERIC_INTERVENTION_MIX_PREDIABETES
    )
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "LADA"}) == (
        GENERIC_INTERVENTION_MIX_LADA
    )
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Gestational"}) == (
        GENERIC_INTERVENTION_BIGIDEAS
    )
    assert intervention_pool_weights(GENERIC_INTERVENTION_MIX_PREDIABETES) == {
        GENERIC_INTERVENTION_BIGIDEAS: 0.75,
        GENERIC_INTERVENTION_D1NAMO: 0.25,
    }
    assert intervention_pool_weights(GENERIC_INTERVENTION_MIX_LADA) == {
        GENERIC_INTERVENTION_D1NAMO: 0.75,
        GENERIC_INTERVENTION_BIGIDEAS: 0.25,
    }


def test_meal_bubbles_open_notepad_not_photo() -> None:
    glucose_df, events_df = load_bigideas_data(SUBJECT_001)
    bubbles = meal_food_bubble_children(
        glucose_df,
        events_df,
        source_name="BIGIDEAS-001.csv",
        hide_last_hour=False,
    )
    assert len(bubbles) == 1
    note = events_df.get_column("food_note").to_list()[0]
    assert bubbles[0].id["index"] == f"note:{quote(note, safe='')}"
    figure = GlucoseChart.build_static_figure(
        glucose_df,
        events_df,
        "BIGIDEAS-001.csv",
        locale="en",
    )
    dotted = [
        shape
        for shape in figure.layout.shapes or []
        if getattr(shape.line, "dash", None) == "dot"
    ]
    assert dotted


@pytest.mark.skipif(
    not dataset_is_present(default_dest()),
    reason="BIG IDEAs extract is not present; run uv run download-bigideas",
)
def test_real_bigideas_extract_formats_to_app_schema() -> None:
    sources = discover_bigideas_sources()
    assert sources
    glucose_df, events_df = load_bigideas_data(sources[0].csv_path)
    assert glucose_df.height > 12
    assert {"time", "gl", "prediction", "age", "user_id"}.issubset(set(glucose_df.columns))
    assert "food_note" in events_df.columns
