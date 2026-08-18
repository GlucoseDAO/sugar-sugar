from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from sugar_sugar.cgmacros import (
    cgmacros_photo_url,
    discover_cgmacros_sources,
    is_cgmacros_csv,
    load_cgmacros_bio,
    load_cgmacros_data,
    resolve_photo_path,
    resolve_served_photo,
    subject_id_from_path,
    visible_food_photo_events,
)
from sugar_sugar.components.glucose import (
    FOOD_COMPOSITE_MAX,
    GlucoseChart,
    cluster_visible_food_events,
    meal_food_bubble_children,
)
from sugar_sugar.i18n import setup_i18n, t

setup_i18n()
from sugar_sugar.data import load_glucose_data
from sugar_sugar.download_cgmacros import dataset_is_present, default_dest

FIXTURE_ROOT = Path(__file__).parent / "testdata" / "cgmacros"
SUBJECT_001 = FIXTURE_ROOT / "CGMacros" / "CGMacros-001" / "CGMacros-001.csv"
SUBJECT_002 = FIXTURE_ROOT / "CGMacros" / "CGMacros-002" / "CGMacros-002.csv"


def test_is_cgmacros_csv_detects_filename_and_columns() -> None:
    assert is_cgmacros_csv(SUBJECT_001)
    assert is_cgmacros_csv(SUBJECT_002)
    assert not is_cgmacros_csv(Path("data/example.csv"))


def test_subject_id_from_path() -> None:
    assert subject_id_from_path(SUBJECT_001) == 1
    assert subject_id_from_path(SUBJECT_001.parent) == 1
    assert subject_id_from_path(Path("other.csv")) is None


def test_load_bio_maps_subject_demographics() -> None:
    bio = load_cgmacros_bio(FIXTURE_ROOT)
    assert bio[1].age_years == 42
    assert bio[1].gender == "female"
    assert bio[1].weight == "70 kg"
    assert bio[2].gender == "male"


def test_discover_cgmacros_sources_reads_fixture_tree() -> None:
    sources = discover_cgmacros_sources(FIXTURE_ROOT)
    names = {source.source_name for source in sources}
    assert names == {"CGMacros-001.csv", "CGMacros-002.csv"}
    first = next(source for source in sources if source.subject_id == 1)
    assert first.age_years == 42
    assert first.sensor == "Dexcom G6 Pro"


def test_plays_dexcom_only_and_downsamples_to_five_minutes() -> None:
    """Dexcom alone is played, and the 1-minute table is thinned to the game grid.

    Never the average of the two sensors: across the 45 published subjects Libre
    compresses the excursion to a median 0.70 of Dexcom's, by a subject-specific
    factor, so a mean trace would carry a distorted amplitude into a study that
    scores human error in mg/dL.

    cgm-format parses tracks but resamples neither, and CGMacros is a 1-minute
    interpolated table -- without ``_downsample_glucose_5min`` a 36-point window
    would cover 36 minutes rather than three hours.
    """
    glucose_df, _events = load_cgmacros_data(SUBJECT_001)
    assert glucose_df.columns == ["time", "gl", "prediction", "age", "user_id"]
    times = glucose_df.get_column("time").to_list()
    assert times[0] == datetime(2020, 5, 10, 12, 0)
    deltas = [
        (later - earlier).total_seconds()
        for earlier, later in zip(times, times[1:])
    ]
    assert deltas
    assert all(delta == 300 for delta in deltas)
    # 12:00 reads Dexcom 108.0 and Libre 110.0 -- neither their mean (109.0)
    # nor Libre.
    assert glucose_df.get_column("gl").to_list()[0] == 108.0


def test_falls_back_to_libre_only_when_dexcom_has_no_readings() -> None:
    """A degraded trace beats no round, but only when there is no alternative.

    CGMacros-002's Dexcom column is populated for the last two rows only, so
    Dexcom still wins; a subject with none at all would fall back to Libre and
    log that its calibration is unreliable.
    """
    glucose_df, events_df = load_cgmacros_data(SUBJECT_002)
    assert glucose_df.get_column("gl").to_list() == [143.5, 146.0]
    assert events_df.height == 1
    assert events_df.get_column("meal_type").to_list() == ["Dinner"]
    assert events_df.get_column("carbs_g").to_list() == [60.0]

    from sugar_sugar.cgmacros import _playable_track

    name, _frame = _playable_track(SUBJECT_001)
    assert name == "dexcom"


def test_library_keeps_meal_photo_macros_and_end_photo() -> None:
    glucose_df, events_df = load_cgmacros_data(SUBJECT_001)
    assert glucose_df.height > 0
    meals = events_df.filter(pl.col("event_type") == "Carbohydrates").sort("time")
    assert meals.height == 2
    assert meals.get_column("meal_type").to_list() == ["Breakfast", ""]
    assert meals.get_column("carbs_g").to_list() == [48.0, None]
    assert meals.get_column("photo_path").to_list() == [
        "photos/meal-before.jpg",
        "photos/meal-after.jpg",
    ]
    assert meals.get_column("time").to_list() == [
        datetime(2020, 5, 10, 12, 7),
        datetime(2020, 5, 10, 12, 45),
    ]


def test_resolve_photo_path_prefers_existing_file() -> None:
    subject_dir = SUBJECT_001.parent
    assert resolve_photo_path("photos/meal-before.jpg", subject_dir) == "photos/meal-before.jpg"
    assert resolve_photo_path("photos/missing.jpg", subject_dir) == "photos/missing.jpg"
    assert resolve_photo_path("../outside.jpg", subject_dir) == ""


def test_load_glucose_data_routes_cgmacros() -> None:
    glucose_df, events_df = load_glucose_data(SUBJECT_001)
    assert "gl" in glucose_df.columns
    assert "Carbohydrates" in events_df.get_column("event_type").to_list()
    assert "photo_path" in events_df.columns


def _pad_window(base: pl.DataFrame, *, hours_before: int = 0, hours_after: int = 0) -> pl.DataFrame:
    def _block(start: datetime, count: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [start + timedelta(minutes=5 * i) for i in range(count)],
                "gl": [100.0] * count,
                "prediction": [0.0] * count,
                "age": [0] * count,
                "user_id": [1] * count,
            }
        )

    parts: list[pl.DataFrame] = []
    if hours_before:
        first = base.get_column("time")[0]
        parts.append(_block(first - timedelta(hours=hours_before), hours_before * 12))
    parts.append(base)
    if hours_after:
        last = base.get_column("time")[-1]
        parts.append(_block(last + timedelta(minutes=5), hours_after * 12))
    schema = dict(base.schema)
    return pl.concat([part.cast(schema) for part in parts])


def test_food_photos_in_last_hour_are_hidden() -> None:
    glucose_df, events_df = load_cgmacros_data(SUBJECT_001)
    visible = _pad_window(glucose_df, hours_after=2)
    shown = visible_food_photo_events(visible, events_df, hide_last_hour=True)
    assert len(shown) == 2
    hidden_slice = _pad_window(glucose_df, hours_before=2)
    last_hour_hidden = visible_food_photo_events(
        hidden_slice, events_df, hide_last_hour=True
    )
    assert last_hour_hidden == []
    revealed = visible_food_photo_events(
        hidden_slice, events_df, hide_last_hour=False
    )
    assert len(revealed) == 1


def test_close_meal_photos_cluster_into_one_composite_bubble() -> None:
    start = datetime(2020, 5, 10, 12, 0)
    window_df = pl.DataFrame(
        {
            "time": [start + timedelta(minutes=5 * i) for i in range(12)],
            "gl": [100.0] * 12,
            "prediction": [0.0] * 12,
            "age": [0] * 12,
            "user_id": [1] * 12,
        }
    )
    events_df = pl.DataFrame(
        {
            "time": [start + timedelta(minutes=5), start + timedelta(minutes=10)],
            "event_type": ["Carbohydrates", "Carbohydrates"],
            "event_subtype": ["Carbs", "Carbs"],
            "insulin_value": [None, None],
            "photo_path": ["photos/one.jpg", "photos/two.jpg"],
            "food_note": ["", ""],
        }
    )
    clusters = cluster_visible_food_events(
        window_df,
        events_df,
        source_name="CGMacros-001.csv",
        hide_last_hour=False,
    )
    assert len(clusters) == 1
    assert clusters[0].photo_urls == [
        "/cgmacros/CGMacros-001/photo/photos/one.jpg",
        "/cgmacros/CGMacros-001/photo/photos/two.jpg",
    ]
    bubbles = meal_food_bubble_children(
        window_df,
        events_df,
        source_name="CGMacros-001.csv",
        hide_last_hour=False,
    )
    assert len(bubbles) == 1
    assert bubbles[0].id["index"].startswith("composite:")
    assert "photos/one.jpg" in bubbles[0].id["index"]
    assert "photos/two.jpg" in bubbles[0].id["index"]
    bubble_props = bubbles[0].to_plotly_json()["props"]
    assert bubble_props.get("data-count") == "2"
    assert bubble_props["aria-label"] == "Carbohydrates"
    assert bubble_props["children"][1].children == "FOOD"


def test_lightbox_uses_fixed_composite_tiles() -> None:
    from sugar_sugar.app import app

    lightbox = next(
        child
        for child in app.layout.children
        if getattr(child, "id", None) == "meal-food-lightbox"
    )
    gallery = next(
        child
        for child in lightbox.children
        if getattr(child, "id", None) == "meal-food-lightbox-gallery"
    )
    assert [child.id for child in gallery.children] == [
        {"type": "meal-food-lightbox-tile", "index": i}
        for i in range(FOOD_COMPOSITE_MAX)
    ]


def test_meal_bubbles_and_food_line_use_photo() -> None:
    glucose_df, events_df = load_cgmacros_data(SUBJECT_001)
    bubbles = meal_food_bubble_children(
        glucose_df,
        events_df,
        source_name="CGMacros-001.csv",
        hide_last_hour=False,
    )
    indexes = {bubble.id["index"] for bubble in bubbles}
    assert "/cgmacros/CGMacros-001/photo/photos/meal-before.jpg" in indexes
    figure = GlucoseChart.build_static_figure(
        glucose_df,
        events_df,
        "CGMacros-001.csv",
        locale="en",
    )
    dotted = [
        shape
        for shape in figure.layout.shapes or []
        if getattr(shape.line, "dash", None) == "dot"
    ]
    assert dotted
    vertical_food_labels = [
        ann.text
        for ann in (figure.layout.annotations or [])
        if ann.text == "FOOD"
    ]
    assert not vertical_food_labels
    ro_bubbles = meal_food_bubble_children(
        glucose_df,
        events_df,
        source_name="CGMacros-001.csv",
        hide_last_hour=False,
        locale="ro",
    )
    assert ro_bubbles
    assert ro_bubbles[0].children[1].children == "MÂNCARE"
    assert t("ui.chart.food_label", locale="en") == "FOOD"
    assert t("ui.chart.food_label", locale="de") == "ESSEN"
    assert t("ui.chart.food_label", locale="fr") == "REPAS"
    assert t("ui.chart.food_label", locale="es") == "COMIDA"
    assert t("ui.chart.food_label", locale="ro") == "MÂNCARE"
    assert t("ui.chart.food_label", locale="ru") == "ЕДА"
    assert t("ui.chart.food_label", locale="uk") == "ЇЖА"
    assert t("ui.chart.food_label", locale="zh") == "食物"


def test_photo_url_and_served_path() -> None:
    assert cgmacros_photo_url("CGMacros-001.csv", "photos/meal-before.jpg") == (
        "/cgmacros/CGMacros-001/photo/photos/meal-before.jpg"
    )
    found = resolve_served_photo("CGMacros-001", "photos/meal-before.jpg", dest=FIXTURE_ROOT)
    assert found is not None
    assert found.name == "meal-before.jpg"
    assert resolve_served_photo("CGMacros-001", "../bio.csv", dest=FIXTURE_ROOT) is None


@pytest.mark.skipif(
    not dataset_is_present(default_dest()),
    reason="CGMacros extract is not present; run uv run download-cgmacros",
)
def test_real_cgmacros_extract_formats_to_app_schema() -> None:
    sources = discover_cgmacros_sources()
    assert sources
    glucose_df, events_df = load_cgmacros_data(sources[0].csv_path)
    assert glucose_df.height > 12
    assert {"time", "gl", "prediction", "age", "user_id"}.issubset(set(glucose_df.columns))
    assert {"time", "event_type", "event_subtype", "insulin_value", "photo_path"}.issubset(
        set(events_df.columns)
    )
    times = glucose_df.get_column("time").to_list()
    if len(times) > 1:
        assert (times[1] - times[0]).total_seconds() == 300
