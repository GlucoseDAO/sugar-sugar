from __future__ import annotations

from pathlib import Path

from sugar_sugar.cgm_duration import (
    cgm_duration_csv_value,
    cgm_duration_to_years,
    migrate_cgm_duration_cell,
    parse_cgm_duration,
    serialize_cgm_duration,
)
from sugar_sugar.migrate_cgm_duration import migrate_cgm_duration_csv


def test_serialize_and_parse_round_trip() -> None:
    assert serialize_cgm_duration(6, "months") == "6,months"
    assert parse_cgm_duration("6,months") == (6.0, "months")
    assert parse_cgm_duration([3, "years"]) == (3.0, "years")


def test_legacy_integer_is_years() -> None:
    assert parse_cgm_duration(4) == (4.0, "years")
    assert parse_cgm_duration("4") == (4.0, "years")
    assert migrate_cgm_duration_cell("4") == "4,years"
    assert migrate_cgm_duration_cell("6,months") == "6,months"


def test_duration_converts_to_years() -> None:
    assert cgm_duration_to_years(26, "weeks") == 26 / 52
    assert cgm_duration_to_years(18, "months") == 1.5
    assert cgm_duration_to_years(3, "years") == 3.0


def test_csv_value_prefers_tuple() -> None:
    assert cgm_duration_csv_value({"cgm_duration": [8, "months"]}) == "8,months"
    assert cgm_duration_csv_value({"cgm_duration_years": 2}) == "2,years"
    assert cgm_duration_csv_value({}) == ""


def test_migrate_csv_rewrites_legacy_integers(tmp_path: Path) -> None:
    path = tmp_path / "prediction_statistics.csv"
    path.write_text(
        "study_id,cgm_duration_years\n"
        "a,3\n"
        'b,"6,months"\n'
        "c,\n",
        encoding="utf-8",
    )
    changed = migrate_cgm_duration_csv(path)
    assert changed == 1
    text = path.read_text(encoding="utf-8")
    assert "3,years" in text
    assert "6,months" in text
