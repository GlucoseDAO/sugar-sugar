"""Grouping of ranking rows into one board entry per *player*.

The ranking CSVs are keyed by `study_id`, but a new device or a wiped localStorage
mints a fresh one -- so one person could occupy several rows. `_leaderboard_snapshot`
merges rows that share an `email_key`, taking the best of each study's representative
score, and labels the merged entry with the newest nickname.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from sugar_sugar.app import _leaderboard_snapshot, _rank_from_ranking_csv

_HEADER = (
    "study_id,run_id,number,timestamp,email_key,nickname,format,rounds_played,"
    "is_example_data,data_source_name,overall_mae_mgdl,overall_mse_mgdl,"
    "overall_rmse_mgdl,overall_mape_pct\n"
)


def _row(
    study_id: str,
    mae: float,
    *,
    fmt: str = "ALL",
    key: str = "",
    nickname: str = "",
    ts: str = "2026-08-01 10:00:00",
) -> str:
    return (
        f"{study_id},run1,1,{ts},{key},{nickname},{fmt},12,True,example,{mae},0,0,0\n"
    )


@pytest.fixture()
def ranking_csv(tmp_path: Path) -> Path:
    return tmp_path / "prediction_ranking.csv"


def _snapshot(path: Path, **kwargs: Any) -> Optional[dict[str, Any]]:
    kwargs.setdefault("study_id", "")
    kwargs.setdefault("format_filter", "ALL")
    kwargs.setdefault("mode", "latest")
    return _leaderboard_snapshot(path, **kwargs)


def test_two_study_ids_with_one_email_collapse_to_a_single_entry(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 22.0, key="hash-ann", nickname="Ninja", ts="2026-08-01 10:00:00")
        + _row("s7", 14.0, key="hash-ann", nickname="Ninja2", ts="2026-08-02 10:00:00")
        + _row("s3", 19.0, nickname="Bob"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s7", key="hash-ann")
    assert snapshot is not None
    # Two players, not three rows.
    assert snapshot["total"] == 2
    # Best of the merged studies wins, labelled with the newest nickname.
    assert snapshot["rank"] == 1
    assert snapshot["mae"] == 14.0
    assert [e["nickname"] for e in snapshot["top"]] == ["Ninja2", "Bob"]


def test_newest_nickname_wins_even_when_it_is_on_the_worse_row(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 12.0, key="hash-ann", nickname="Old", ts="2026-08-01 10:00:00")
        + _row("s7", 30.0, key="hash-ann", nickname="New", ts="2026-08-05 10:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", key="hash-ann")
    assert snapshot is not None
    assert snapshot["mae"] == 12.0
    assert snapshot["top"][0]["nickname"] == "New"


def test_a_later_blank_nickname_does_not_erase_an_earlier_one(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 20.0, key="hash-ann", nickname="Ninja", ts="2026-08-01 10:00:00")
        + _row("s1", 18.0, key="hash-ann", nickname="", ts="2026-08-09 10:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", key="hash-ann")
    assert snapshot is not None
    assert snapshot["top"][0]["nickname"] == "Ninja"


def test_latest_mode_still_applies_within_one_study(ranking_csv: Path) -> None:
    """The overall CSV appends a cumulative row per round; a flattering early row
    (few rounds, low MAE) must not beat the full run."""
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 5.0, key="hash-ann", ts="2026-08-01 10:00:00")   # after round 1
        + _row("s1", 15.0, key="hash-ann", ts="2026-08-01 11:00:00")  # after round 12
        + _row("s2", 10.0, key="hash-bob"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", key="hash-ann")
    assert snapshot is not None
    assert snapshot["mae"] == 15.0
    assert snapshot["rank"] == 2  # loses to s2's 10.0


def test_best_mode_takes_the_minimum_within_one_study(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 15.0, fmt="A", key="hash-ann", ts="2026-08-01 10:00:00")
        + _row("s1", 9.0, fmt="A", key="hash-ann", ts="2026-08-01 11:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(
        ranking_csv, study_id="s1", key="hash-ann", format_filter="A", mode="best"
    )
    assert snapshot is not None
    assert snapshot["mae"] == 9.0


def test_rows_without_an_email_stay_separate_players(ranking_csv: Path) -> None:
    """Anonymous players must not all merge into one entry."""
    ranking_csv.write_text(
        _HEADER + _row("s1", 20.0) + _row("s2", 10.0) + _row("s3", 30.0),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1")
    assert snapshot is not None
    assert snapshot["total"] == 3
    assert snapshot["rank"] == 2
    assert all(entry["nickname"] == "" for entry in snapshot["top"])


def test_study_id_still_matches_rows_written_before_an_email_was_given(
    ranking_csv: Path,
) -> None:
    """A player who supplies an email later must still own their older rows."""
    ranking_csv.write_text(_HEADER + _row("s1", 20.0) + _row("s2", 10.0), encoding="utf-8")
    snapshot = _snapshot(ranking_csv, study_id="s1", key="hash-ann")
    assert snapshot is not None
    assert snapshot["rank"] == 2
    assert any(entry["is_you"] for entry in snapshot["top"])


def test_only_the_players_own_row_is_flagged(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 20.0, key="hash-ann")
        + _row("s2", 10.0, key="hash-bob"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", key="hash-ann")
    assert snapshot is not None
    assert [e["is_you"] for e in snapshot["top"]] == [False, True]


def test_visitor_without_a_session_gets_a_board_and_no_you_row(ranking_csv: Path) -> None:
    ranking_csv.write_text(_HEADER + _row("s1", 20.0, nickname="Bob"), encoding="utf-8")
    snapshot = _snapshot(ranking_csv)
    assert snapshot is not None
    assert snapshot["rank"] is None and snapshot["mae"] is None
    assert snapshot["top"] == [{"rank": 1, "mae": 20.0, "nickname": "Bob", "is_you": False}]


def test_pre_nickname_csv_schema_still_ranks(tmp_path: Path) -> None:
    """Files written before the nickname columns existed must keep working."""
    legacy = tmp_path / "prediction_ranking.csv"
    legacy.write_text(
        "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
        "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
        "overall_mape_pct\n"
        "s1,run1,1,2026-08-01 10:00:00,ALL,12,True,example,18.0,0,0,0\n",
        encoding="utf-8",
    )
    snapshot = _snapshot(legacy, study_id="s1")
    assert snapshot is not None
    assert snapshot["rank"] == 1
    assert snapshot["top"][0]["nickname"] == ""


def test_your_row_is_appended_when_you_are_outside_the_top(ranking_csv: Path) -> None:
    rows = "".join(_row(f"s{i}", float(i), key=f"hash-{i}") for i in range(1, 8))
    ranking_csv.write_text(_HEADER + rows, encoding="utf-8")
    snapshot = _snapshot(ranking_csv, study_id="s7", key="hash-7", top_n=3)
    assert snapshot is not None
    assert len(snapshot["top"]) == 4
    assert snapshot["top"][-1] == {"rank": 7, "mae": 7.0, "nickname": "", "is_you": True}


def test_missing_csv_yields_no_snapshot(tmp_path: Path) -> None:
    assert _snapshot(tmp_path / "nope.csv", study_id="s1") is None


def test_rank_helper_agrees_with_the_snapshot(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 22.0, key="hash-ann")
        + _row("s7", 14.0, key="hash-ann")
        + _row("s3", 19.0),
        encoding="utf-8",
    )
    assert _rank_from_ranking_csv(
        ranking_csv, study_id="s1", key="hash-ann", format_filter="ALL", mode="latest"
    ) == (1, 2)
