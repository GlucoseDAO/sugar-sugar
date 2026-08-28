"""The class scoreboard: diabetic vs non-diabetic data, best-N-round scoring.

Why these invariants exist:

* Non-diabetic traces are flatter and objectively easier, so a mixed board
  systematically ranks BIG IDEAs players above D1NAMO players -- rounds are
  classified by the data actually predicted and ranked per class.
* A cumulative 12-round MAE cannot get lucky the way a 6-round one can, so
  every entry is scored on its best ``CLASS_SCORE_ROUNDS`` rounds of the class.
* Hard-mode and veteran badges are derived, never stored; the public player id
  must expose neither ``study_id`` nor ``email_key``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Optional

import pytest

from sugar_sugar.config import MIN_USEFUL_ROUNDS
from sugar_sugar.nickname import email_key, identity_key
from sugar_sugar.scoreboard import (
    declared_source_classes,
    earned_medals,
    CLASS_SCORE_ROUNDS,
    DATA_CLASS_DIABETIC,
    DATA_CLASS_NONDIABETIC,
    build_scoreboard,
    classify_round_source,
    entry_is_own,
    is_hard_mode,
    public_player_id,
)

STATS_HEADER = (
    "study_id,run_id,number,timestamp,email,format,is_example_data,data_source_name,"
    "age,user_id,gender,uses_cgm,cgm_duration_years,diabetic,diabetic_type,"
    "diabetes_duration,location,rounds_played,predicted_values,real_values,"
    "prediction_times,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
    "overall_mape_pct,per_round_metrics\n"
)
RANK_HEADER = (
    "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
    "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
    "overall_mape_pct,email_key,nickname\n"
)


def _per_round_cell(maes_sources: list[tuple[float, str]]) -> str:
    rows = [
        {
            "round_number": i + 1,
            "mae": mae,
            "mse": mae * mae,
            "rmse": mae,
            "mape": 5.0,
            "data_source_name": src,
            "is_example_data": src == "example.csv",
            "generic_slice_key": "",
        }
        for i, (mae, src) in enumerate(maes_sources)
    ]
    return '"' + str(rows).replace('"', '""') + '"'


def _stats_row(
    study: str,
    run: str,
    *,
    ts: str = "2026-08-01 10:00:00",
    email: str = "",
    fmt: str = "A",
    diabetic: str = "False",
    rounds: int = 12,
    mae: float = 20.0,
    per_round: str = '"[]"',
    source: str = "src.csv",
    is_example: str = "False",
) -> str:
    return (
        f"{study},{run},1,{ts},{email},{fmt},{is_example},{source},30,1,female,"
        f"True,1,{diabetic},,,,{rounds},x,x,x,{mae},0,0,0,{per_round}\n"
    )


@pytest.fixture()
def input_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "input"
    d.mkdir(parents=True)
    return d


def _write_stats(
    input_dir: Path,
    rows: str,
    nicknames: Optional[dict[str, str]] = None,
) -> None:
    """Write the statistics CSV *and* the per-format ranking rows it implies.

    `save_statistics` writes both together, and the boards are built from the
    ranking CSVs (a run absent from them was never on the leaderboard), so a
    fixture that wrote only statistics would be testing a state production
    cannot produce.
    """
    (input_dir / "prediction_statistics.csv").write_text(STATS_HEADER + rows, encoding="utf-8")

    names = nicknames or {}
    by_format: dict[str, list[list[str]]] = {}
    header = STATS_HEADER.strip().split(",")
    for row in csv.reader(io.StringIO(rows)):
        if not row:
            continue
        cell = dict(zip(header, row))
        fmt = cell.get("format") or "A"
        by_format.setdefault(fmt, []).append([
            cell.get("study_id", ""), cell.get("run_id", ""), "1",
            cell.get("timestamp", ""), fmt, cell.get("rounds_played", ""),
            cell.get("is_example_data", ""), cell.get("data_source_name", ""),
            cell.get("overall_mae_mgdl", ""), "0", "0", "0",
            email_key(cell.get("email", "")), names.get(cell.get("study_id", ""), ""),
        ])
    for fmt, entries in by_format.items():
        path = input_dir / f"prediction_ranking_{fmt}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(RANK_HEADER.strip().split(","))
            writer.writerows(entries)


def test_classify_round_source_by_corpus_and_player() -> None:
    """`declared={}` pins the filename/player logic alone, with no shipped metadata."""
    assert classify_round_source("D1NAMO-001.csv", is_example=False, player_diabetic=False, declared={}) == DATA_CLASS_DIABETIC
    assert classify_round_source("BIGIDEAS-003.csv", is_example=False, player_diabetic=True, declared={}) == DATA_CLASS_NONDIABETIC
    # Legacy LOOP subjects are T1 loop users.
    assert classify_round_source("loop467_chronological.csv", is_example=False, player_diabetic=None, declared={}) == DATA_CLASS_DIABETIC
    # Own uploaded data inherits the player's own status.
    assert classify_round_source("my_dexcom.csv", is_example=False, player_diabetic=True, declared={}) == DATA_CLASS_DIABETIC
    assert classify_round_source("my_dexcom.csv", is_example=False, player_diabetic=False, declared={}) == DATA_CLASS_NONDIABETIC
    # Own data with no stated status classifies nowhere.
    assert classify_round_source("my_dexcom.csv", is_example=False, player_diabetic=None, declared={}) is None


def test_undeclared_generic_source_never_borrows_the_players_status() -> None:
    """A public trace is somebody else's body, so the player's own status says
    nothing about it: an undeclared generic source stays off both boards."""
    for player in (True, False, None):
        assert classify_round_source(
            "mystery_generic.csv", is_example=True, player_diabetic=player, declared={}
        ) is None


def test_declared_metadata_classifies_a_source_the_filename_cannot() -> None:
    declared = {"house.csv": DATA_CLASS_DIABETIC, "flat.csv": DATA_CLASS_NONDIABETIC}
    assert classify_round_source("house.csv", is_example=True, player_diabetic=False, declared=declared) == DATA_CLASS_DIABETIC
    assert classify_round_source("flat.csv", is_example=True, player_diabetic=True, declared=declared) == DATA_CLASS_NONDIABETIC


def test_shipped_metadata_declares_example_csv_diabetic() -> None:
    """example.csv is the Format A fallback when the corpora are absent, and it is
    an insulin-treated trace (73 insulin events, 42% of readings above 180 mg/dL).
    While it was undeclared, every round of a corpus-less deployment landed on no
    board at all -- which is what emptied the staging leaderboard."""
    assert declared_source_classes().get("example.csv") == DATA_CLASS_DIABETIC
    assert classify_round_source(
        "example.csv", is_example=True, player_diabetic=False
    ) == DATA_CLASS_DIABETIC


def test_hard_mode_is_playing_foreign_data() -> None:
    assert is_hard_mode(DATA_CLASS_DIABETIC, False) is True
    assert is_hard_mode(DATA_CLASS_NONDIABETIC, True) is True
    assert is_hard_mode(DATA_CLASS_DIABETIC, True) is False
    assert is_hard_mode(DATA_CLASS_NONDIABETIC, False) is False
    assert is_hard_mode(DATA_CLASS_DIABETIC, None) is False


def test_public_player_id_reveals_nothing() -> None:
    identity = identity_key(key="", study_id="secret-study")
    pid = public_player_id(identity)
    assert pid and "secret" not in pid and len(pid) == 16
    # Stable, and distinct across identities.
    assert pid == public_player_id(identity)
    assert pid != public_player_id(identity_key(key="", study_id="other"))


def test_score_is_best_n_rounds_so_long_runs_are_not_handicapped(input_dir: Path) -> None:
    """A 12-round run is judged on the same number of rounds as a 6-round one:
    a mean over 12 cannot get lucky the way a mean over 6 can."""
    twelve = [(10.0 + i, "BIGIDEAS-001.csv") for i in range(12)]  # best 6: 10..15
    six = [(14.0 + i, "BIGIDEAS-002.csv") for i in range(6)]      # all 6: 14..19
    _write_stats(
        input_dir,
        _stats_row("long", "r1", rounds=12, mae=15.5, per_round=_per_round_cell(twelve))
        + _stats_row("short", "r1", rounds=6, mae=16.5, per_round=_per_round_cell(six)),
    )
    board = build_scoreboard(input_dir).boards[DATA_CLASS_NONDIABETIC]
    assert [e.study_id for e in board] == ["long", "short"]
    assert board[0].score_mae == pytest.approx(12.5)  # mean of 10..15
    assert board[1].score_mae == pytest.approx(16.5)  # mean of 14..19


def test_runs_below_the_class_floor_stay_off_that_board(input_dir: Path) -> None:
    """Fewer than CLASS_SCORE_ROUNDS rounds of a class cannot enter its board."""
    few = [(10.0, "D1NAMO-001.csv")] * (CLASS_SCORE_ROUNDS - 1)
    _write_stats(
        input_dir,
        _stats_row("p", "r1", diabetic="True", rounds=CLASS_SCORE_ROUNDS - 1,
                   per_round=_per_round_cell(few)),
    )
    boards = build_scoreboard(input_dir).boards
    assert boards[DATA_CLASS_DIABETIC] == []
    assert boards[DATA_CLASS_NONDIABETIC] == []


def test_mixed_challenge_run_lands_on_both_boards_with_hard_mode_badge(input_dir: Path) -> None:
    """Challenge-the-unknown: half home corpus, half opposite -- one slot each."""
    mixed = (
        [(15.0, "BIGIDEAS-001.csv")] * 6 + [(30.0, "D1NAMO-001.csv")] * 6
    )
    _write_stats(
        input_dir,
        _stats_row("nd", "r1", diabetic="False", rounds=12, per_round=_per_round_cell(mixed)),
    )
    boards = build_scoreboard(input_dir).boards
    assert len(boards[DATA_CLASS_NONDIABETIC]) == 1
    assert len(boards[DATA_CLASS_DIABETIC]) == 1
    assert boards[DATA_CLASS_NONDIABETIC][0].hard_mode is False  # home data
    assert boards[DATA_CLASS_DIABETIC][0].hard_mode is True      # foreign data


def test_example_rounds_rank_on_the_diabetic_board(input_dir: Path) -> None:
    """A deployment without the corpora serves example.csv for every public round.
    Those runs must still rank -- an empty board was the symptom that found this."""
    rounds = [(15.0, "example.csv")] * 12
    _write_stats(
        input_dir,
        _stats_row("p", "r1", rounds=12, per_round=_per_round_cell(rounds),
                   source="example.csv", is_example="True"),
    )
    boards = build_scoreboard(input_dir).boards
    assert len(boards[DATA_CLASS_DIABETIC]) == 1
    assert boards[DATA_CLASS_NONDIABETIC] == []
    # A non-diabetic player predicting that insulin-treated trace is in hard mode.
    assert boards[DATA_CLASS_DIABETIC][0].hard_mode is True


def test_unclassifiable_rounds_still_rank_nowhere(input_dir: Path) -> None:
    """The floor that remains: own data from a player who never stated a status."""
    rounds = [(15.0, "my_upload.csv")] * 12
    _write_stats(
        input_dir,
        _stats_row("p", "r1", rounds=12, per_round=_per_round_cell(rounds),
                   source="my_upload.csv", diabetic="", is_example="False"),
    )
    boards = build_scoreboard(input_dir).boards
    assert boards[DATA_CLASS_DIABETIC] == [] and boards[DATA_CLASS_NONDIABETIC] == []


def test_legacy_run_without_per_round_metrics_uses_overall_mae(input_dir: Path) -> None:
    """Pre-per-round rows still enter the board their run-level source classifies to."""
    _write_stats(
        input_dir,
        _stats_row("old", "r1", rounds=12, mae=21.0, per_round='"[]"',
                   source="BIGIDEAS-004.csv"),
    )
    board = build_scoreboard(input_dir).boards[DATA_CLASS_NONDIABETIC]
    assert len(board) == 1
    assert board[0].score_mae == pytest.approx(21.0)


def test_veteran_needs_more_than_one_finished_game(input_dir: Path) -> None:
    rounds = _per_round_cell([(15.0, "BIGIDEAS-001.csv")] * 12)
    _write_stats(
        input_dir,
        _stats_row("solo", "r1", rounds=12, per_round=rounds)
        + _stats_row("vet", "r1", email="vet@x.com", rounds=12, per_round=rounds)
        + _stats_row("vet", "r2", email="vet@x.com", rounds=12, per_round=rounds,
                     ts="2026-08-02 10:00:00"),
    )
    sb = build_scoreboard(input_dir)
    by_study = {e.study_id: e for e in sb.boards[DATA_CLASS_NONDIABETIC]}
    assert by_study["solo"].games == 1
    assert by_study["vet"].games == 2
    # Player pages exist for every finisher; the veteran's lists both games.
    vet_pid = by_study["vet"].public_id
    assert len(sb.players[vet_pid].games) == 2
    # A short run does not count as a finished game.
    assert all(g.total_rounds >= MIN_USEFUL_ROUNDS for g in sb.players[vet_pid].games)


def test_email_merges_devices_into_one_identity(input_dir: Path) -> None:
    rounds = _per_round_cell([(15.0, "BIGIDEAS-001.csv")] * 12)
    _write_stats(
        input_dir,
        _stats_row("dev1", "r1", email="ann@x.com", rounds=12, per_round=rounds)
        + _stats_row("dev2", "r1", email="Ann@X.com ", rounds=12, per_round=rounds,
                     ts="2026-08-02 10:00:00"),
    )
    sb = build_scoreboard(input_dir)
    board = sb.boards[DATA_CLASS_NONDIABETIC]
    assert len(board) == 2  # arcade: both slots stand
    assert board[0].identity == board[1].identity
    assert board[0].games == 2  # both runs count for the one person
    assert sb.player_count() == 1


def test_nickname_comes_from_the_ranking_csv(input_dir: Path) -> None:
    rounds = _per_round_cell([(15.0, "BIGIDEAS-001.csv")] * 12)
    _write_stats(
        input_dir,
        _stats_row("s1", "r1", email="ann@x.com", rounds=12, per_round=rounds),
        nicknames={"s1": "SugarNinja"},
    )
    board = build_scoreboard(input_dir).boards[DATA_CLASS_NONDIABETIC]
    assert board[0].nickname == "SugarNinja"


def test_entry_is_own_matches_study_id_or_email_key(input_dir: Path) -> None:
    rounds = _per_round_cell([(15.0, "BIGIDEAS-001.csv")] * 12)
    _write_stats(input_dir, _stats_row("s1", "r1", email="ann@x.com", rounds=12, per_round=rounds))
    entry = build_scoreboard(input_dir).boards[DATA_CLASS_NONDIABETIC][0]
    assert entry_is_own(entry, study_id="s1", key="")
    assert entry_is_own(entry, study_id="other-device", key=email_key("ann@x.com"))
    assert not entry_is_own(entry, study_id="other-device", key=email_key("bob@x.com"))


def test_boards_sort_by_score_then_timestamp(input_dir: Path) -> None:
    def flat(mae: float) -> str:
        return _per_round_cell([(mae, "BIGIDEAS-001.csv")] * 6)

    _write_stats(
        input_dir,
        _stats_row("late", "r1", rounds=6, per_round=flat(10.0), ts="2026-08-02 10:00:00")
        + _stats_row("early", "r1", rounds=6, per_round=flat(10.0), ts="2026-08-01 10:00:00")
        + _stats_row("worse", "r1", rounds=6, per_round=flat(12.0), ts="2026-07-01 10:00:00"),
    )
    board = build_scoreboard(input_dir).boards[DATA_CLASS_NONDIABETIC]
    assert [e.study_id for e in board] == ["early", "late", "worse"]


def test_medals_are_cumulative_and_tiered() -> None:
    """Bronze and silver come from how far a single game got; gold is a property
    of the player across games, so one long game never earns it."""
    assert earned_medals(total_rounds=5, games=1) == []
    assert earned_medals(total_rounds=6, games=1) == ["bronze"]
    assert earned_medals(total_rounds=12, games=1) == ["bronze", "silver"]
    assert earned_medals(total_rounds=12, games=4) == ["bronze", "silver", "gold"]
    # A returning player whose games were short still earns gold.
    assert earned_medals(total_rounds=2, games=2) == ["gold"]


def test_class_rounds_argument_widens_and_narrows_the_board(input_dir: Path) -> None:
    """The /highscore selector passes this; the floor and the averaging window
    are one number, so every entry in a view is scored over the same N."""
    _write_stats(
        input_dir,
        _stats_row("long", "r1", rounds=12,
                   per_round=_per_round_cell([(10.0 + i, "BIGIDEAS-001.csv") for i in range(12)]))
        + _stats_row("short", "r1", rounds=4,
                     per_round=_per_round_cell([(9.0, "BIGIDEAS-002.csv")] * 4)),
    )
    at_six = build_scoreboard(input_dir, class_rounds=6).boards[DATA_CLASS_NONDIABETIC]
    assert [e.study_id for e in at_six] == ["long"]

    at_three = build_scoreboard(input_dir, class_rounds=3).boards[DATA_CLASS_NONDIABETIC]
    # The short run now qualifies, and its flat 9.0 still beats the long run's best 3.
    assert [e.study_id for e in at_three] == ["short", "long"]
    assert at_three[0].score_mae == pytest.approx(9.0)
    assert at_three[1].score_mae == pytest.approx(11.0)  # mean of 10, 11, 12

    at_twelve = build_scoreboard(input_dir, class_rounds=12).boards[DATA_CLASS_NONDIABETIC]
    assert [e.study_id for e in at_twelve] == ["long"]


def test_unranked_runs_are_counted_for_the_page_note(input_dir: Path) -> None:
    _write_stats(
        input_dir,
        _stats_row("ranked", "r1", rounds=12,
                   per_round=_per_round_cell([(10.0, "BIGIDEAS-001.csv")] * 12))
        + _stats_row("tooshort", "r1", rounds=2,
                     per_round=_per_round_cell([(9.0, "BIGIDEAS-002.csv")] * 2)),
    )
    board = build_scoreboard(input_dir, class_rounds=6)
    assert board.total_runs == 2
    assert board.unranked_runs == 1


def test_a_ranking_row_with_no_statistics_row_still_ranks(input_dir: Path) -> None:
    """Production had leaderboard rows whose study_id is absent from the
    statistics CSV -- a player who finished 12 rounds on mobile was on prod's
    board and vanished from these. The ranking CSVs are where the leaderboard is
    written, so they are the row source; statistics only enriches a row it has.
    """
    (input_dir / "prediction_ranking_A.csv").write_text(
        RANK_HEADER
        + "0abe913b,e0ab6ea8,21,2026-08-20 13:54:12,A,12,True,BIGIDEAS-016.csv,"
          "15.319444444444445,479.37,21.89,13.76,ad7467f9997413cb,Newton Winter\n",
        encoding="utf-8",
    )
    (input_dir / "prediction_statistics.csv").write_text(STATS_HEADER, encoding="utf-8")

    board = build_scoreboard(input_dir).boards[DATA_CLASS_NONDIABETIC]
    assert [e.nickname for e in board] == ["Newton Winter"]
    assert board[0].class_rounds == 12
    assert board[0].score_mae == pytest.approx(15.319444, abs=1e-4)
    # Identity comes off the row's own email_key, so no raw address is needed.
    assert board[0].identity == "e:ad7467f9997413cb"
    # The player's own diabetes status is unknown without a statistics row, so
    # hard mode cannot be asserted either way -- it must not be guessed.
    assert board[0].hard_mode is False
