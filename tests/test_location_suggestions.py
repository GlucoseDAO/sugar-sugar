from __future__ import annotations

import json
from pathlib import Path

from sugar_sugar.location_suggestions import (
    MAX_SUGGESTIONS,
    export_suggestions_asset,
    filter_location_suggestions,
    write_suggestions_asset,
)


def test_filter_requires_two_characters() -> None:
    assert filter_location_suggestions("b") == []
    assert filter_location_suggestions("  p") == []


def test_filter_prefers_prefix_matches_in_english() -> None:
    matches = filter_location_suggestions("ber", locale="en")
    assert matches[0] == "Berlin, Germany"
    assert len(matches) <= MAX_SUGGESTIONS


def test_filter_matches_localized_spelling() -> None:
    matches_de = filter_location_suggestions("münc", locale="de")
    assert "München, Deutschland" in matches_de

    matches_de_country = filter_location_suggestions("deutsch", locale="de")
    assert "Deutschland" in matches_de_country

    matches_ru = filter_location_suggestions("моск", locale="ru")
    assert "Москва, Россия" in matches_ru

    matches_zh = filter_location_suggestions("北京", locale="zh")
    assert "北京, 中国" in matches_zh


def test_filter_ascii_fallback_for_umlauts() -> None:
    matches = filter_location_suggestions("munc", locale="de")
    assert "München, Deutschland" in matches


def test_filter_allows_free_text_not_in_list() -> None:
    """Autocomplete is optional; validation still accepts arbitrary locations."""
    assert filter_location_suggestions("Small Village, Nowhere") == []


def test_every_country_has_up_to_ten_cities() -> None:
    from sugar_sugar.location_catalog import COUNTRY_NAMES, _top_cities_by_country

    top_cities = _top_cities_by_country()
    for country in COUNTRY_NAMES:
        cities = top_cities.get(country, ())
        assert 1 <= len(cities) <= 10


def test_asset_matches_python_source() -> None:
    asset_path = Path(__file__).resolve().parents[1] / "assets" / "location-suggestions.json"
    assert asset_path.exists()
    loaded = json.loads(asset_path.read_text(encoding="utf-8"))
    assert loaded == export_suggestions_asset()


def test_write_suggestions_asset(tmp_path: Path) -> None:
    target = tmp_path / "location-suggestions.json"
    write_suggestions_asset(target)
    assert json.loads(target.read_text(encoding="utf-8")) == export_suggestions_asset()
