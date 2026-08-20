"""Access vs completed counters for the landing / highscore social-proof chips."""

from __future__ import annotations

from pathlib import Path

import pytest

from sugar_sugar.components.landing import (
    count_people_who_accessed,
    count_people_who_completed,
    count_ranking_slots,
    games_played_counter,
)
from sugar_sugar.i18n import setup_i18n
from sugar_sugar.config import MIN_USEFUL_ROUNDS

_STATS_HEADER = "study_id,rounds_played,email\n"
_RANK_HEADER = (
    "study_id,run_id,number,timestamp,email_key,nickname,format,rounds_played,"
    "is_example_data,data_source_name,overall_mae_mgdl,overall_mse_mgdl,"
    "overall_rmse_mgdl,overall_mape_pct\n"
)


def _stats_row(study_id: str, rounds: int) -> str:
    return f"{study_id},{rounds},a@x.com\n"


def _rank_row(study_id: str, rounds: int) -> str:
    return (
        f"{study_id},run1,1,2026-08-01 10:00:00,,,ALL,{rounds},True,example,"
        "10,0,0,0\n"
    )


def test_accessed_counts_every_starter(tmp_path: Path) -> None:
    stats = tmp_path / "prediction_statistics.csv"
    stats.write_text(
        _STATS_HEADER + _stats_row("zero", 0) + _stats_row("short", 3) + _stats_row("done", 12),
        encoding="utf-8",
    )
    assert count_people_who_accessed(stats) == 3


def test_completed_requires_the_round_floor(tmp_path: Path) -> None:
    stats = tmp_path / "prediction_statistics.csv"
    ranking = tmp_path / "prediction_ranking.csv"
    stats.write_text(
        _STATS_HEADER + _stats_row("zero", 0) + _stats_row("short", 5) + _stats_row("done", 6),
        encoding="utf-8",
    )
    ranking.write_text(_RANK_HEADER + _rank_row("done", 6), encoding="utf-8")
    assert count_people_who_completed(
        min_rounds=MIN_USEFUL_ROUNDS,
        stats_path=stats,
        ranking_paths=(ranking,),
    ) == 1


def test_completed_unions_stats_and_ranking(tmp_path: Path) -> None:
    """A 6-round chart-exit still counts even if ranking was skipped."""
    stats = tmp_path / "prediction_statistics.csv"
    ranking = tmp_path / "prediction_ranking.csv"
    stats.write_text(_STATS_HEADER + _stats_row("exit", 8), encoding="utf-8")
    ranking.write_text(_RANK_HEADER + _rank_row("board", 12), encoding="utf-8")
    assert count_people_who_completed(
        stats_path=stats,
        ranking_paths=(ranking,),
    ) == 2


def test_landing_counter_mentions_completers_and_the_punchline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sugar_sugar.components.landing as landing

    setup_i18n()
    stats = tmp_path / "prediction_statistics.csv"
    ranking = tmp_path / "prediction_ranking.csv"
    stats.write_text(
        _STATS_HEADER + _stats_row("a", 0) + _stats_row("b", 12),
        encoding="utf-8",
    )
    ranking.write_text(
        _RANK_HEADER + _rank_row("short", 3) + _rank_row("b", 12),
        encoding="utf-8",
    )
    monkeypatch.setattr(landing, "_PREDICTION_STATS_CSV", stats)
    monkeypatch.setattr(landing, "_RANKING_CSV_PATHS", (ranking,))
    monkeypatch.setattr(landing, "_PREDICTION_RANKING_CSV", ranking)

    counter = games_played_counter("en")
    children = list(counter.children)
    assert children[0].children == "2 players so far"
    assert children[0].className == "games-played-completed"
    assert children[1].className == "games-played-count"
    props = children[1].to_plotly_json().get("props") or {}
    assert props.get("data-target") == "1"
    assert children[2].children == "of them completed the task"
    assert "Be part of the victors" in children[3].children
    assert children[4].children == (
        "2 games started, 1 finished — one person can play several times"
    )
    assert children[4].className == "games-played-slots"


def test_ranking_slots_split_short_and_complete_runs(tmp_path: Path) -> None:
    ranking = tmp_path / "prediction_ranking.csv"
    ranking.write_text(
        _RANK_HEADER + _rank_row("a", 2) + _rank_row("b", 5) + _rank_row("c", 6) + _rank_row("d", 12),
        encoding="utf-8",
    )
    assert count_ranking_slots(ranking) == (4, 2)
