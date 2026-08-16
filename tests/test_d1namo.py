from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from sugar_sugar.components.glucose import GlucoseChart, meal_food_bubble_children
from sugar_sugar.d1namo import (
    d1namo_photo_url,
    discover_d1namo_sources,
    is_d1namo_path,
    is_d1namo_source_name,
    load_d1namo_data,
    resolve_photo_path,
    resolve_served_photo,
    subject_format,
    subject_id_from_path,
)
from sugar_sugar.data import load_glucose_data
from sugar_sugar.download_d1namo import dataset_is_present, default_dest
from sugar_sugar.subject_sources import (
    GENERIC_INTERVENTION_D1NAMO,
    discover_generic_dataset_sources,
    generic_intervention_for_user,
    load_generic_dataset_source,
    resolve_generic_source_path,
)

FIXTURE_ROOT = Path(__file__).parent / "testdata" / "d1namo"
SUBJECT_001 = FIXTURE_ROOT / "001" / "glucose.csv"
SUBJECT_002 = FIXTURE_ROOT / "002" / "glucose.csv"


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
    """The EXIF-style ``2014:10:01 12:15:00`` in ``food.csv`` reaches the chart.

    D1NAMO mixes four timestamp conventions inside one subject directory, and
    this is the one that silently produces wrong data: Polars cannot infer a
    colon-separated date, so it must be parsed by explicit format.
    """
    _glucose_df, events_df = load_d1namo_data(SUBJECT_001)
    meals = events_df.filter(pl.col("event_type") == "Carbohydrates")
    assert meals.get_column("time").to_list()[0] == datetime(2014, 10, 1, 12, 15)
    assert meals.get_column("meal_type").to_list()[0] == "Lunch"


def test_keeps_cgm_converts_mmol_and_drops_fingerstick() -> None:
    """Sensor readings only, in mg/dL, with meals and insulin as events.

    D1NAMO glucose is mmol/L in both subsets and is converted through the unit
    the schema declares (18.0182), not a locally guessed factor. The 12:02
    fingerstick is a ``CALIBRAT`` event, so it stays out of the trace the player
    is asked to continue.
    """
    glucose_df, events_df = load_d1namo_data(SUBJECT_001)
    assert glucose_df.columns == ["time", "gl", "prediction", "age", "user_id"]
    times = glucose_df.get_column("time").to_list()
    assert times[0] == datetime(2014, 10, 1, 12, 0)
    assert datetime(2014, 10, 1, 12, 2) not in times
    assert glucose_df.height == 13
    assert glucose_df.get_column("gl").to_list()[0] == pytest.approx(6.0 * 18.0182)

    insulin = events_df.filter(pl.col("event_type") == "Insulin")
    # The 12:12 row records a skipped bolus of 0 U -- bookkeeping, not a dose.
    assert insulin.height == 2
    assert set(insulin.get_column("event_subtype").to_list()) == {
        "Fast Acting",
        "Long Acting",
    }
    meals = events_df.filter(pl.col("event_type") == "Carbohydrates")
    assert meals.height == 2
    assert meals.get_column("photo_path").to_list() == ["pictures/meal.jpg", ""]
    assert meals.get_column("meal_type").to_list() == ["Lunch", "Dinner without photo"]
    # D1NAMO has no carbohydrate column anywhere: null is "the source did not
    # say", which is not the same as zero.
    assert meals.get_column("carbs_g").to_list() == [None, None]


def test_glucose_always_converts_from_mmol() -> None:
    """Every D1NAMO subject is mmol/L -- there is no unit to guess at.

    The old formatter inferred the unit from the series maximum ("< 40 means
    mmol/L"), which would misread a subject whose readings all sat high. The
    library takes the unit from the schema instead.
    """
    glucose_df, events_df = load_d1namo_data(SUBJECT_002)
    assert glucose_df.get_column("gl").to_list() == pytest.approx(
        [7.8 * 18.0182, 7.9 * 18.0182, 8.0 * 18.0182]
    )
    assert events_df.get_column("event_subtype").to_list() == ["Long Acting"]


def test_subject_format_rejects_a_directory_that_is_not_a_subject(
    tmp_path: Path,
) -> None:
    """A folder holding a bare ``glucose.csv`` is not a parseable D1NAMO subject.

    The two subsets are told apart by ``insulin.csv`` (diabetes) vs
    ``annotations.csv`` (healthy); without either, the library cannot say which
    parser applies. Discovery skips such a folder rather than offering the round
    picker a source that raises when loaded.
    """
    orphan = tmp_path / "007"
    orphan.mkdir()
    (orphan / "glucose.csv").write_text("date,time,glucose,type,comments\n")
    assert subject_format(orphan) is None
    assert subject_format(SUBJECT_001.parent) is not None
    assert discover_d1namo_sources(tmp_path) == []


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


def test_generic_intervention_type_1_is_d1namo() -> None:
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 1"}) == (
        GENERIC_INTERVENTION_D1NAMO
    )


def test_discover_routes_type_1_to_d1namo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_d1namo_sources",
        lambda dest=None: discover_d1namo_sources(FIXTURE_ROOT),
    )
    monkeypatch.setattr(
        "sugar_sugar.subject_sources.discover_bigideas_sources",
        lambda dest=None: [],
    )
    names = {
        source.source_name
        for source in discover_generic_dataset_sources(intervention=GENERIC_INTERVENTION_D1NAMO)
    }
    assert names == {"D1NAMO-001.csv", "D1NAMO-002.csv"}


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
