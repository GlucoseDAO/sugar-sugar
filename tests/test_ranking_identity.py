"""Per-player rollup of ranking rows (`_ranking_identities`).

NOT what `/highscore` or `/final` render -- those boards are arcade-style, one slot
per finished game (see `tests/test_ranking_arcade_board.py`). This is the "your best,
across every device you played on" aggregation kept for the planned individual stats
page. Tested now so it does not rot before that page exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from sugar_sugar.app import _match_identity, _ranking_identities

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


def _identities(path: Path, *, mode: str = "latest", fmt: Optional[str] = "ALL") -> Any:
    return _ranking_identities(path, format_filter=fmt, mode=mode)


def _me(path: Path, *, study_id: str, key: str = "", mode: str = "latest") -> Optional[dict[str, Any]]:
    return _match_identity(_identities(path, mode=mode), study_id=study_id, key=key)


# --- grouping ----------------------------------------------------------------


def test_two_study_ids_with_one_email_collapse_to_a_single_player(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 22.0, key="hash-ann", nickname="Ninja", ts="2026-08-01 10:00:00")
        + _row("s7", 14.0, key="hash-ann", nickname="Ninja2", ts="2026-08-02 10:00:00")
        + _row("s3", 19.0, nickname="Bob"),
        encoding="utf-8",
    )
    identities = _identities(ranking_csv)
    assert identities is not None
    assert identities.height == 2  # two people, three rows

    me = _me(ranking_csv, study_id="s7", key="hash-ann")
    assert me is not None
    assert me['mae'] == 14.0  # best across the merged studies
    assert me['nickname'] == "Ninja2"  # newest name
    assert sorted(me['study_ids']) == ["s1", "s7"]
    assert me['games'] == 2


def test_rows_without_an_email_stay_separate_players(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER + _row("s1", 20.0) + _row("s2", 10.0) + _row("s3", 30.0), encoding="utf-8"
    )
    identities = _identities(ranking_csv)
    assert identities is not None and identities.height == 3


def test_newest_nickname_wins_even_on_the_worse_row(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 12.0, key="hash-ann", nickname="Old", ts="2026-08-01 10:00:00")
        + _row("s7", 30.0, key="hash-ann", nickname="New", ts="2026-08-05 10:00:00"),
        encoding="utf-8",
    )
    me = _me(ranking_csv, study_id="s1", key="hash-ann")
    assert me is not None
    assert me['mae'] == 12.0 and me['nickname'] == "New"


def test_a_later_blank_nickname_does_not_erase_an_earlier_one(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 20.0, key="hash-ann", nickname="Ninja", ts="2026-08-01 10:00:00")
        + _row("s1", 18.0, key="hash-ann", nickname="", ts="2026-08-09 10:00:00"),
        encoding="utf-8",
    )
    me = _me(ranking_csv, study_id="s1", key="hash-ann")
    assert me is not None and me['nickname'] == "Ninja"


# --- mode semantics ----------------------------------------------------------


def test_latest_mode_takes_the_newest_row_within_one_study(ranking_csv: Path) -> None:
    """The overall CSV's rows are cumulative, so an early partial row must not win."""
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 5.0, key="k", rounds=1, ts="2026-08-01 10:00:00")
        + _row("s1", 15.0, key="k", rounds=12, ts="2026-08-01 11:00:00"),
        encoding="utf-8",
    )
    me = _me(ranking_csv, study_id="s1", key="k", mode="latest")
    assert me is not None
    assert me['mae'] == 15.0 and me['rounds'] == 12


def test_best_mode_takes_the_minimum_within_one_study(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER
        + _row("s1", 15.0, fmt="A", key="k", ts="2026-08-01 10:00:00")
        + _row("s1", 9.0, fmt="A", key="k", ts="2026-08-01 11:00:00"),
        encoding="utf-8",
    )
    me = _match_identity(
        _ranking_identities(ranking_csv, format_filter="A", mode="best"),
        study_id="s1",
        key="k",
    )
    assert me is not None and me['mae'] == 9.0


# --- matching ----------------------------------------------------------------


def test_study_id_fallback_finds_pre_email_rows(ranking_csv: Path) -> None:
    ranking_csv.write_text(_HEADER + _row("s1", 20.0) + _row("s2", 10.0), encoding="utf-8")
    me = _me(ranking_csv, study_id="s1", key="hash-ann")
    assert me is not None and me['mae'] == 20.0


def test_unknown_player_matches_nothing(ranking_csv: Path) -> None:
    ranking_csv.write_text(_HEADER + _row("s1", 20.0), encoding="utf-8")
    assert _me(ranking_csv, study_id="nobody", key="nope") is None


def test_missing_csv_yields_nothing(tmp_path: Path) -> None:
    assert _identities(tmp_path / "nope.csv") is None
    assert _match_identity(None, study_id="s1", key="") is None


def test_pre_nickname_csv_schema_still_aggregates(tmp_path: Path) -> None:
    legacy = tmp_path / "prediction_ranking.csv"
    legacy.write_text(
        "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
        "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
        "overall_mape_pct\n"
        "s1,run1,1,2026-08-01 10:00:00,ALL,12,True,example,18.0,0,0,0\n",
        encoding="utf-8",
    )
    me = _me(legacy, study_id="s1")
    assert me is not None
    assert me['mae'] == 18.0 and me['nickname'] == ""


def test_ranking_is_best_first(ranking_csv: Path) -> None:
    ranking_csv.write_text(
        _HEADER + _row("s1", 30.0, key="a") + _row("s2", 10.0, key="b") + _row("s3", 20.0, key="c"),
        encoding="utf-8",
    )
    identities = _identities(ranking_csv)
    assert identities is not None
    assert identities.get_column('mae').to_list() == [10.0, 20.0, 30.0]
    assert identities.get_column('rank_idx').to_list() == [0, 1, 2]
