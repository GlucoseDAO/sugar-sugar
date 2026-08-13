"""Arcade placement: every finished game keeps its own slot.

Nothing is merged or collapsed away. If you beat your own score, the old one stays on
the board below the new one -- hiding it was the confusing behaviour. The hashed email
no longer decides placement; it only answers "is this slot mine?" (so slots set on
another device are still highlighted) and feeds the /final nickname suggestion.
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
    rounds: int = 12,
    ts: str = "2026-08-01 10:00:00",
) -> str:
    return (
        f"{study_id},run1,1,{ts},{key},{nickname},{fmt},{rounds},True,example,"
        f"{mae},0,0,0\n"
    )


@pytest.fixture()
def ranking_csv(tmp_path: Path) -> Path:
    return tmp_path / "prediction_ranking.csv"


def _snapshot(path: Path, **kwargs: Any) -> Optional[dict[str, Any]]:
    kwargs.setdefault("study_id", "")
    kwargs.setdefault("format_filter", "ALL")
    return _leaderboard_snapshot(path, **kwargs)


# --- nothing is hidden -------------------------------------------------------


def test_beating_your_own_score_keeps_the_old_one_on_the_board(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 25.0, key="k", nickname="Ninja", ts="2026-07-01 10:00:00")
        + _row("s1", 15.0, key="k", nickname="Ninja", ts="2026-08-01 10:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", key="k")
    assert snapshot is not None
    assert snapshot["total"] == 2  # both slots stand
    assert [entry["mae"] for entry in snapshot["top"]] == [15.0, 25.0]
    assert all(entry["is_you"] for entry in snapshot["top"])
    # Your placement is your best slot.
    assert snapshot["rank"] == 1 and snapshot["mae"] == 15.0


def test_one_person_two_devices_holds_two_slots(ranking_csv: Path) -> None:
    """The email no longer merges rows -- it only marks them as yours."""
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 22.0, key="k", nickname="Ninja")
        + _row("s7", 14.0, key="k", nickname="Ninja")
        + _row("s3", 19.0, nickname="Bob"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s7", key="k")
    assert snapshot is not None
    assert snapshot["total"] == 3
    assert [entry["mae"] for entry in snapshot["top"]] == [14.0, 19.0, 22.0]
    # Both of the player's slots are highlighted, including the one from the
    # other device (matched by hashed email, not study_id).
    assert [entry["is_you"] for entry in snapshot["top"]] == [True, False, True]


def test_players_counts_people_while_total_counts_slots(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 22.0, key="k")
        + _row("s7", 14.0, key="k")
        + _row("s3", 19.0, key="other"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="")
    assert snapshot is not None
    assert snapshot["total"] == 3
    assert snapshot["players"] == 2


# --- ordering ----------------------------------------------------------------


def test_slots_are_ordered_by_score(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER + _row("s1", 30.0) + _row("s2", 10.0) + _row("s3", 20.0), encoding="utf-8"
    )
    snapshot = _snapshot(ranking_csv)
    assert snapshot is not None
    assert [entry["mae"] for entry in snapshot["top"]] == [10.0, 20.0, 30.0]
    assert [entry["rank"] for entry in snapshot["top"]] == [1, 2, 3]


def test_ties_go_to_whoever_got_there_first(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("late", 12.0, nickname="Late", ts="2026-08-05 10:00:00")
        + _row("early", 12.0, nickname="Early", ts="2026-07-01 10:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv)
    assert snapshot is not None
    assert [entry["nickname"] for entry in snapshot["top"]] == ["Early", "Late"]


# --- the rounds column ------------------------------------------------------


def test_rounds_are_reported_per_slot(ranking_csv: Path) -> None:
    """Two cumulative slots of one player are told apart by their round count."""
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 20.0, key="k", rounds=12, ts="2026-07-01 10:00:00")
        + _row("s1", 14.0, key="k", rounds=24, ts="2026-08-01 10:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", key="k")
    assert snapshot is not None
    assert [entry["rounds"] for entry in snapshot["top"]] == [24, 12]


def test_missing_rounds_column_is_tolerated(tmp_path: Path) -> None:
    legacy = tmp_path / "prediction_ranking.csv"
    legacy.write_text(
        "study_id,timestamp,format,overall_mae_mgdl\n"
        "s1,2026-08-01 10:00:00,ALL,18.0\n",
        encoding="utf-8",
    )
    snapshot = _snapshot(legacy, study_id="s1")
    assert snapshot is not None
    assert snapshot["top"][0]["rounds"] is None


# --- nicknames per slot ------------------------------------------------------


def test_each_slot_keeps_the_name_it_was_set_under(ranking_csv: Path) -> None:
    """Arcade initials belong to the score, not to the person's current name."""
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 25.0, key="k", nickname="OldName", ts="2026-07-01 10:00:00")
        + _row("s7", 15.0, key="k", nickname="NewName", ts="2026-08-01 10:00:00"),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s7", key="k")
    assert snapshot is not None
    assert [entry["nickname"] for entry in snapshot["top"]] == ["NewName", "OldName"]


def test_anonymous_slots_report_no_nickname(ranking_csv: Path) -> None:
    ranking_csv.write_text(_HEADER + _row("s1", 20.0), encoding="utf-8")
    snapshot = _snapshot(ranking_csv, study_id="s1")
    assert snapshot is not None
    assert snapshot["top"][0]["nickname"] == ""


# --- "you" resolution --------------------------------------------------------


def test_study_id_matches_slots_set_before_an_email_was_given(ranking_csv: Path) -> None:
    ranking_csv.write_text(_HEADER + _row("s1", 20.0) + _row("s2", 10.0), encoding="utf-8")
    snapshot = _snapshot(ranking_csv, study_id="s1", key="k")
    assert snapshot is not None
    assert snapshot["rank"] == 2
    assert [entry["is_you"] for entry in snapshot["top"]] == [False, True]


def test_visitor_without_a_session_sees_the_board_and_no_you_row(ranking_csv: Path) -> None:
    ranking_csv.write_text(_HEADER + _row("s1", 20.0, nickname="Bob"), encoding="utf-8")
    snapshot = _snapshot(ranking_csv)
    assert snapshot is not None
    assert snapshot["rank"] is None and snapshot["mae"] is None
    assert snapshot["top"] == [
        {"rank": 1, "mae": 20.0, "rounds": 12, "nickname": "Bob", "is_you": False}
    ]


def test_your_best_slot_is_appended_when_outside_the_top(ranking_csv: Path) -> None:
    rows = "".join(_row(f"s{i}", float(i)) for i in range(1, 8))
    ranking_csv.write_text(_HEADER + rows, encoding="utf-8")
    snapshot = _snapshot(ranking_csv, study_id="s7", top_n=3)
    assert snapshot is not None
    assert len(snapshot["top"]) == 4
    assert snapshot["top"][-1]["rank"] == 7 and snapshot["top"][-1]["is_you"] is True


# --- format filter and edge cases -------------------------------------------


def test_format_filter_restricts_the_board(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER + _row("s1", 20.0, fmt="A") + _row("s2", 10.0, fmt="B"), encoding="utf-8"
    )
    snapshot = _snapshot(ranking_csv, study_id="s1", format_filter="A")
    assert snapshot is not None
    assert snapshot["total"] == 1 and snapshot["top"][0]["mae"] == 20.0


def test_short_runs_are_excluded_from_the_board(ranking_csv: Path) -> None:
    """Fewer than MIN_USEFUL_ROUNDS (exclusive) stays off the leaderboard."""
    ranking_csv.write_text(
        _HEADER
        + _row("short", 1.0, rounds=5)
        + _row("just", 20.0, rounds=6)
        + _row("full", 10.0, rounds=12),
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv)
    assert snapshot is not None
    assert snapshot["total"] == 2
    assert [entry["mae"] for entry in snapshot["top"]] == [10.0, 20.0]
    assert _snapshot(ranking_csv, study_id="short")["rank"] is None
    assert _rank_from_ranking_csv(
        ranking_csv, study_id="short", format_filter="ALL"
    ) is None


def test_unrankable_rows_are_dropped(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER + _row("s1", 20.0) + "s2,run1,1,2026-08-01 10:00:00,,,ALL,12,True,example,,0,0,0\n",
        encoding="utf-8",
    )
    snapshot = _snapshot(ranking_csv, study_id="s1")
    assert snapshot is not None
    assert snapshot["total"] == 1


def test_missing_csv_yields_no_snapshot(tmp_path: Path) -> None:
    assert _snapshot(tmp_path / "nope.csv", study_id="s1") is None


def test_rank_helper_agrees_with_the_snapshot(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER + _row("s1", 22.0, key="k") + _row("s7", 14.0, key="k") + _row("s3", 19.0),
        encoding="utf-8",
    )
    # Best of the player's two slots, out of three slots total.
    assert _rank_from_ranking_csv(
        ranking_csv, study_id="s1", key="k", format_filter="ALL"
    ) == (1, 3)
