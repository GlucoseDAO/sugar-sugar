"""`SubmitComponent.set_study_nickname` stamps ONE study's ranking rows.

The `/final` rename box is used after `save_statistics` has already written the run's
rows, so those rows have to be updated in place. The hard guarantee is the negative
one: a returning player who picks a new name must not rewrite the rows of their
earlier study entries, even though those rows share the same `email_key`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

import pytest

from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.i18n import setup_i18n

_HEADER = (
    "study_id,run_id,number,timestamp,email_key,nickname,format,rounds_played,"
    "is_example_data,data_source_name,overall_mae_mgdl,overall_mse_mgdl,"
    "overall_rmse_mgdl,overall_mape_pct\n"
)

_LEGACY_HEADER = (
    "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
    "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
    "overall_mape_pct\n"
)


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


@pytest.fixture()
def component(tmp_path: Path) -> Iterator[SubmitComponent]:
    """A SubmitComponent whose four ranking CSVs live in a throwaway tree."""
    submit = SubmitComponent()
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {
        fmt: tmp_path / f"prediction_ranking_{fmt}.csv" for fmt in ("A", "B", "C")
    }
    yield submit


def _row(study_id: str, *, key: str = "", nickname: str = "", fmt: str = "ALL") -> str:
    return (
        f"{study_id},run1,1,2026-08-01 10:00:00,{key},{nickname},{fmt},"
        "12,True,example,18.0,0,0,0\n"
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_stamps_the_named_study_across_all_ranking_files(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(_HEADER + _row("s1", key="k"), encoding="utf-8")
    for fmt, path in component._ranking_by_format_paths.items():
        path.write_text(_HEADER + _row("s1", key="k", fmt=fmt), encoding="utf-8")

    changed = component.set_study_nickname(study_id="s1", key="k", nickname="Ninja")

    assert changed == 4  # overall + A + B + C
    assert _rows(component._ranking_csv_path)[0]["nickname"] == "Ninja"
    for path in component._ranking_by_format_paths.values():
        assert _rows(path)[0]["nickname"] == "Ninja"


def test_other_studies_keep_their_nickname_even_with_the_same_email(
    component: SubmitComponent,
) -> None:
    """The "don't overwrite previous nicknames" guarantee."""
    component._ranking_csv_path.write_text(
        _HEADER
        + _row("s1", key="k", nickname="Ninja")
        + _row("s7", key="k", nickname="")
        + _row("s9", key="other", nickname="Bob"),
        encoding="utf-8",
    )

    changed = component.set_study_nickname(study_id="s7", key="k", nickname="Ninja2")

    assert changed == 1
    by_study = {row["study_id"]: row["nickname"] for row in _rows(component._ranking_csv_path)}
    assert by_study == {"s1": "Ninja", "s7": "Ninja2", "s9": "Bob"}


def test_all_rows_of_the_same_study_are_stamped(component: SubmitComponent) -> None:
    """One nickname per study, even though the overall CSV holds many rows per run."""
    component._ranking_csv_path.write_text(
        _HEADER + _row("s1", key="k") + _row("s1", key="k") + _row("s1", key="k"),
        encoding="utf-8",
    )
    assert component.set_study_nickname(study_id="s1", key="k", nickname="Ninja") == 3
    assert {row["nickname"] for row in _rows(component._ranking_csv_path)} == {"Ninja"}


def test_legacy_csv_gains_the_columns(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(
        _LEGACY_HEADER
        + "s1,run1,1,2026-08-01 10:00:00,ALL,12,True,example,18.0,0,0,0\n",
        encoding="utf-8",
    )
    component.set_study_nickname(study_id="s1", key="k", nickname="Ninja")

    rows = _rows(component._ranking_csv_path)
    assert rows[0]["nickname"] == "Ninja"
    assert rows[0]["email_key"] == "k"  # backfilled while we were rewriting anyway


def test_an_existing_email_key_is_never_rewritten(component: SubmitComponent) -> None:
    """Backfill only fills a hole; it must not relabel a row's identity."""
    component._ranking_csv_path.write_text(
        _HEADER + _row("s1", key="original"), encoding="utf-8"
    )
    component.set_study_nickname(study_id="s1", key="different", nickname="Ninja")
    assert _rows(component._ranking_csv_path)[0]["email_key"] == "original"


def test_nickname_is_normalized_before_writing(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(_HEADER + _row("s1", key="k"), encoding="utf-8")
    component.set_study_nickname(study_id="s1", key="k", nickname="  Sugar   Ninja  ")
    assert _rows(component._ranking_csv_path)[0]["nickname"] == "Sugar Ninja"


def test_clearing_the_nickname_blanks_this_study_only(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(
        _HEADER + _row("s1", key="k", nickname="Ninja") + _row("s7", key="k", nickname="Keep"),
        encoding="utf-8",
    )
    component.set_study_nickname(study_id="s1", key="k", nickname="")
    by_study = {row["study_id"]: row["nickname"] for row in _rows(component._ranking_csv_path)}
    assert by_study == {"s1": "", "s7": "Keep"}


def test_missing_files_and_blank_study_id_are_no_ops(component: SubmitComponent) -> None:
    assert component.set_study_nickname(study_id="s1", key="k", nickname="Ninja") == 0
    component._ranking_csv_path.write_text(_HEADER + _row("s1", key="k"), encoding="utf-8")
    assert component.set_study_nickname(study_id="", key="k", nickname="Ninja") == 0
    assert _rows(component._ranking_csv_path)[0]["nickname"] == ""


def test_no_temp_file_is_left_behind(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(_HEADER + _row("s1", key="k"), encoding="utf-8")
    component.set_study_nickname(study_id="s1", key="k", nickname="Ninja")
    assert not component._ranking_csv_path.with_suffix(".tmp").exists()
