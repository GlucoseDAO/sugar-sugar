"""End-of-game identity form must complete every CSV column for the study."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.consent import (
    identity_is_complete,
    results_destination,
    stamp_identity_fields,
)


def test_identity_is_complete_from_flag() -> None:
    assert identity_is_complete({"identity_completed": True}) is True
    assert identity_is_complete({}) is False


def test_identity_is_complete_from_legacy_startup_fields() -> None:
    assert identity_is_complete({"age": 30, "gender": "F", "location": "Berlin"}) is True
    assert identity_is_complete({"age": 16, "gender": "F", "location": "Berlin"}) is False
    assert identity_is_complete({"age": 30, "gender": "F"}) is False


def test_results_destination_gates_on_identity() -> None:
    assert results_destination({"age": 30, "gender": "F", "location": "Berlin"}) == "/final"
    assert results_destination({"format": "A"}) == "/profile"


def test_stamp_identity_fields_marks_complete() -> None:
    info = stamp_identity_fields(
        {"consent_upload_own_data": True},
        nickname="SugarNinja",
        email="player@example.com",
        age=34,
        gender="M",
        location="Oslo",
        receive_results=True,
        keep_updated=False,
    )
    assert info["identity_completed"] is True
    assert info["email"] == "player@example.com"
    assert info["age"] == 34
    assert info["consent_receive_results_later"] is True
    assert info["nickname"] == "SugarNinja"


def test_backfill_identity_completes_stub_and_played_rows(tmp_path: Path) -> None:
    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {k: tmp_path / f"r_{k}.csv" for k in ("A", "B", "C")}

    stub: dict[str, Any] = {
        "study_id": "split-form",
        "run_id": "r-start",
        "number": 1,
        "consent_completed": True,
        "format": "A",
        "run_format": "A",
        "email": "",
        "age": 0,
        "gender": "",
        "location": "",
        "diabetic": True,
        "diabetic_type": "Type 1",
        "diabetes_duration": 6,
        "uses_cgm": True,
        "cgm_duration_years": 2,
        "rounds": [],
    }
    submit.save_statistics(stub)

    with submit._stats_csv_path.open(newline="", encoding="utf-8") as fh:
        stub_rows = list(csv.DictReader(fh))
    assert stub_rows
    stub_row = stub_rows[0]
    for column in (
        "email",
        "age",
        "gender",
        "location",
        "diabetic",
        "diabetic_type",
        "diabetes_duration",
        "uses_cgm",
        "cgm_duration_years",
        "format",
    ):
        assert column in stub_row
    assert stub_row["diabetic_type"] == "Type 1"
    assert stub_row["email"] == ""
    assert stub_row["gender"] == ""
    assert stub_row["location"] == ""

    later = dict(stub)
    later.update(
        {
            "email": "later@example.com",
            "age": 41,
            "gender": "F",
            "location": "Kyiv",
            "nickname": "LaterPlayer",
            "identity_completed": True,
        }
    )
    changed = submit.backfill_identity_columns(later)
    assert changed >= 1

    with submit._stats_csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["email"] == "later@example.com"
    assert rows[0]["age"] == "41"
    assert rows[0]["gender"] == "F"
    assert rows[0]["location"] == "Kyiv"
    assert rows[0]["diabetic_type"] == "Type 1"
    for column in (
        "study_id",
        "run_id",
        "email",
        "age",
        "gender",
        "location",
        "diabetic",
        "diabetic_type",
        "diabetes_duration",
        "uses_cgm",
        "cgm_duration_years",
        "format",
    ):
        assert column in rows[0]
        assert str(rows[0][column]).strip() != ""
