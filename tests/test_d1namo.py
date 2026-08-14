from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from sugar_sugar.cgmacros import discover_cgmacros_sources
from sugar_sugar.components.glucose import GlucoseChart, meal_food_bubble_children
from sugar_sugar.d1namo import (
    d1namo_photo_url,
    discover_d1namo_sources,
    format_d1namo_frames,
    is_d1namo_path,
    is_d1namo_source_name,
    load_d1namo_data,
    resolve_photo_path,
    resolve_served_photo,
    subject_id_from_path,
)
from sugar_sugar.data import load_glucose_data
from sugar_sugar.download_d1namo import dataset_is_present, default_dest
from sugar_sugar.subject_sources import (
    GENERIC_INTERVENTION_CGMACROS,
    GENERIC_INTERVENTION_D1NAMO,
    discover_generic_dataset_sources,
    generic_intervention_for_user,
    load_generic_dataset_source,
    resolve_generic_source_path,
)

FIXTURE_ROOT = Path(__file__).parent / "testdata" / "d1namo"
SUBJECT_001 = FIXTURE_ROOT / "001" / "glucose.csv"
SUBJECT_002 = FIXTURE_ROOT / "002" / "glucose.csv"
CGMACROS_FIXTURE = Path(__file__).parent / "testdata" / "cgmacros"


def test_is_d1namo_path_detects_glucose_and_virtual_name() -> None:
    assert is_d1namo_path(SUBJECT_001)
    assert is_d1namo_path(Path("D1NAMO-001.csv"))
    assert is_d1namo_source_name("D1NAMO-001.csv")
    assert not is_d1namo_path(Path("data/example.csv"))
    assert not is_d1namo_path(Path("CGMacros-001.csv"))


def test_subject_id_from_path() -> None:
    assert subject_id_from_path(SUBJECT_001) == "001"
    assert subject_id_from_path(SUBJECT_001.parent) == "001"
    assert subject_id_from_path(Path("D1NAMO-007.csv")) == "007"
    assert subject_id_from_path(Path("other.csv")) is None


def test_discover_d1namo_sources_reads_fixture_tree() -> None:
    sources = discover_d1namo_sources(FIXTURE_ROOT)
    names = {source.source_name for source in sources}
    assert names == {"D1NAMO-001.csv", "D1NAMO-002.csv"}
    first = next(source for source in sources if source.subject_id == "001")
    assert first.intervention == "d1namo"
    assert first.sensor == "Medtronic iPro2"
    assert first.csv_path == SUBJECT_001


def test_food_colon_datetime_parses_without_infer() -> None:
    food_df = pl.DataFrame(
        {
            "picture": ["001.jpg"],
            "description": ["Pasta"],
            "calories": [637],
            "datetime": ["2014:10:01 12:21:59"],
        }
    )
    glucose_df = pl.DataFrame(
        {
            "date": ["2014-10-01"],
            "time": ["12:20:00"],
            "glucose": [6.0],
            "type": ["cgm"],
        }
    )
    _, events_df = format_d1namo_frames(
        glucose_df,
        food_df=food_df,
        subject_dir=SUBJECT_001.parent,
        subject_id=1,
    )
    meals = events_df.filter(pl.col("event_type") == "Carbohydrates")
    assert meals.get_column("time").to_list() == [datetime(2014, 10, 1, 12, 21, 59)]
    assert meals.get_column("meal_type").to_list() == ["Pasta"]


def test_format_keeps_cgm_converts_mmol_and_drops_fingerstick() -> None:
    raw = pl.read_csv(SUBJECT_001)
    glucose_df, events_df = format_d1namo_frames(
        raw,
        insulin_df=pl.read_csv(SUBJECT_001.parent / "insulin.csv"),
        food_df=pl.read_csv(SUBJECT_001.parent / "food.csv"),
        subject_dir=SUBJECT_001.parent,
        subject_id=1,
    )
    assert glucose_df.columns == ["time", "gl", "prediction", "age", "user_id"]
    times = glucose_df.get_column("time").to_list()
    assert times[0] == datetime(2014, 10, 1, 12, 0)
    assert datetime(2014, 10, 1, 12, 2) not in times
    assert glucose_df.height == 13
    assert glucose_df.get_column("gl").to_list()[0] == 108.0
    insulin = events_df.filter(pl.col("event_type") == "Insulin")
    assert insulin.height == 2
    subtypes = set(insulin.get_column("event_subtype").to_list())
    assert subtypes == {"Fast Acting", "Long-Acting"}
    meals = events_df.filter(pl.col("event_type") == "Carbohydrates")
    assert meals.height == 2
    assert meals.get_column("photo_path").to_list() == ["pictures/meal.jpg", ""]
    assert meals.get_column("meal_type").to_list() == ["Lunch", "Dinner without photo"]


def test_format_leaves_mgdl_glucose_unchanged() -> None:
    raw = pl.read_csv(SUBJECT_002)
    glucose_df, events_df = format_d1namo_frames(raw, subject_id=2)
    assert glucose_df.get_column("gl").to_list() == [140.0, 142.0, 144.0]
    assert events_df.height == 0


def test_resolve_photo_path_prefers_existing_file() -> None:
    subject_dir = SUBJECT_001.parent
    assert resolve_photo_path("pictures/meal.jpg", subject_dir) == "pictures/meal.jpg"
    assert resolve_photo_path("meal.jpg", subject_dir) == "pictures/meal.jpg"
    assert resolve_photo_path("../outside.jpg", subject_dir) == ""


def test_load_glucose_data_routes_d1namo() -> None:
    glucose_df, events_df = load_glucose_data(SUBJECT_001)
    assert "gl" in glucose_df.columns
    assert "Insulin" in events_df.get_column("event_type").to_list()
    assert "Carbohydrates" in events_df.get_column("event_type").to_list()
    assert "photo_path" in events_df.columns


def test_generic_pipeline_can_resolve_and_load_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_d1namo_sources",
        lambda dest=None: discover_d1namo_sources(FIXTURE_ROOT),
    )
    path = resolve_generic_source_path("D1NAMO-001.csv")
    assert path == SUBJECT_001
    sources = discover_generic_dataset_sources(intervention=GENERIC_INTERVENTION_D1NAMO)
    source = next(item for item in sources if item.source_name == "D1NAMO-001.csv")
    assert source.intervention == GENERIC_INTERVENTION_D1NAMO
    glucose_df, events_df = load_generic_dataset_source(source)
    assert glucose_df.height > 0
    photos = events_df.filter(pl.col("photo_path") != "").get_column("photo_path").to_list()
    assert photos == ["pictures/meal.jpg"]


def test_generic_intervention_follows_diabetic_status() -> None:
    assert generic_intervention_for_user({"diabetic": True}) == GENERIC_INTERVENTION_D1NAMO
    assert generic_intervention_for_user({"diabetic": False}) == GENERIC_INTERVENTION_CGMACROS
    assert generic_intervention_for_user({}) == GENERIC_INTERVENTION_CGMACROS
    assert generic_intervention_for_user(None) == GENERIC_INTERVENTION_CGMACROS


def test_discover_routes_diabetic_to_d1namo_and_others_to_cgmacros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_d1namo_sources",
        lambda dest=None: discover_d1namo_sources(FIXTURE_ROOT),
    )
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_cgmacros_sources",
        lambda dest=None: discover_cgmacros_sources(CGMACROS_FIXTURE),
    )
    d1namo_names = {
        source.source_name
        for source in discover_generic_dataset_sources(intervention=GENERIC_INTERVENTION_D1NAMO)
    }
    cgmacros_names = {
        source.source_name
        for source in discover_generic_dataset_sources(intervention=GENERIC_INTERVENTION_CGMACROS)
    }
    assert d1namo_names == {"D1NAMO-001.csv", "D1NAMO-002.csv"}
    assert cgmacros_names == {"CGMacros-001.csv", "CGMacros-002.csv"}


def test_d1namo_pool_falls_back_to_cgmacros_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_d1namo_sources",
        lambda dest=None: [],
    )
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_cgmacros_sources",
        lambda dest=None: discover_cgmacros_sources(CGMACROS_FIXTURE),
    )
    names = {
        source.source_name
        for source in discover_generic_dataset_sources(intervention=GENERIC_INTERVENTION_D1NAMO)
    }
    assert names == {"CGMacros-001.csv", "CGMacros-002.csv"}


def test_meal_bubbles_use_d1namo_photo_url() -> None:
    glucose_df, events_df = load_d1namo_data(SUBJECT_001)
    bubbles = meal_food_bubble_children(
        glucose_df,
        events_df,
        source_name="D1NAMO-001.csv",
        hide_last_hour=False,
    )
    assert len(bubbles) == 1
    assert bubbles[0].id["index"] == "/d1namo/001/photo/pictures/meal.jpg"
    figure = GlucoseChart.build_static_figure(
        glucose_df,
        events_df,
        "D1NAMO-001.csv",
        locale="en",
    )
    dotted = [
        shape
        for shape in figure.layout.shapes or []
        if getattr(shape.line, "dash", None) == "dot"
    ]
    assert dotted


def test_photo_url_and_served_path() -> None:
    assert d1namo_photo_url("D1NAMO-001.csv", "pictures/meal.jpg") == (
        "/d1namo/001/photo/pictures/meal.jpg"
    )
    found = resolve_served_photo("001", "pictures/meal.jpg", dest=FIXTURE_ROOT)
    assert found is not None
    assert found.name == "meal.jpg"
    assert resolve_served_photo("001", "../glucose.csv", dest=FIXTURE_ROOT) is None
    assert resolve_served_photo("001", "pictures/meal.csv", dest=FIXTURE_ROOT) is None


@pytest.mark.skipif(
    not dataset_is_present(default_dest()),
    reason="D1NAMO extract is not present; run uv run download-d1namo",
)
def test_real_d1namo_extract_formats_to_app_schema() -> None:
    sources = discover_d1namo_sources()
    assert sources
    glucose_df, events_df = load_d1namo_data(sources[0].csv_path)
    assert glucose_df.height > 12
    assert {"time", "gl", "prediction", "age", "user_id"}.issubset(set(glucose_df.columns))
    assert {
        "time",
        "event_type",
        "event_subtype",
        "insulin_value",
        "photo_path",
    }.issubset(set(events_df.columns))
    assert events_df.filter(pl.col("event_type") == "Insulin").height > 0
