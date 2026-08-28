#!/usr/bin/env python3
"""Explain what the /highscore class boards will show, and why.

    uv run python scripts/diagnose-scoreboard.py [data/input]

The class boards are stricter than the old ranking board: an entry needs at
least CLASS_SCORE_ROUNDS rounds *of one data class*, and a round only has a
class if its source can be identified (a BIG IDEAs / D1NAMO / LOOP file, or
own-data plus a known diabetes status).  When a board comes out empty this
prints which of those conditions failed, per source and per run.

Prints no email addresses; study ids are truncated.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Import from the app so this can never drift from what the page actually does.
from sugar_sugar.config import MIN_USEFUL_ROUNDS
from sugar_sugar.scoreboard import (
    CLASS_SCORE_ROUNDS,
    DATA_CLASSES,
    _parse_per_round_metrics,
    _read_csv,
    build_scoreboard,
    classify_round_source,
    parse_diabetic_flag,
)


def _cell(row: dict[str, object], column: str) -> str:
    return str(row.get(column) or "").strip()


def main(argv: list[str]) -> int:
    input_dir = Path(argv[1] if len(argv) > 1 else "data/input").resolve()
    print(f"Reading {input_dir}\n")
    if not input_dir.is_dir():
        print("ERROR: not a directory")
        return 2

    stats_path = input_dir / "prediction_statistics.csv"
    df = _read_csv(stats_path)
    if df is None:
        print(f"ERROR: cannot read {stats_path.name} -- the class boards read this file,")
        print("       not the ranking CSVs. Without it both boards are empty.")
        return 2
    print(f"{stats_path.name}: {df.height} rows, {df.width} columns")
    for name in ("prediction_ranking.csv", "prediction_ranking_A.csv",
                 "prediction_ranking_B.csv", "prediction_ranking_C.csv"):
        other = _read_csv(input_dir / name)
        print(f"{name}: {other.height if other is not None else 0} rows"
              f"{'' if other is not None else ' (missing)'}")
    missing = [c for c in ("per_round_metrics", "diabetic", "rounds_played") if c not in df.columns]
    if missing:
        print(f"\nWARNING: columns absent from the statistics CSV: {', '.join(missing)}")

    # ---- ranking CSVs vs statistics -----------------------------------------
    # The OLD board ranked from the ranking CSVs; the class boards rank from the
    # statistics CSV. If a player is on one and not the other, that is the answer
    # to "where did my finished games go" -- and it is not the classifier.
    print("\n--- ranking CSVs vs statistics (the old board read the ranking ones) ---")
    stats_rounds: dict[str, int] = {}
    stats_newest = ""
    for row in df.iter_rows(named=True):
        study_id = _cell(row, "study_id")
        try:
            stats_rounds[study_id] = max(
                stats_rounds.get(study_id, 0), int(float(_cell(row, "rounds_played") or "0"))
            )
        except ValueError:
            pass
        stats_newest = max(stats_newest, _cell(row, "timestamp"))

    for name in ("prediction_ranking.csv", "prediction_ranking_A.csv",
                 "prediction_ranking_B.csv", "prediction_ranking_C.csv"):
        rank_df = _read_csv(input_dir / name)
        if rank_df is None or rank_df.height == 0:
            continue
        long_runs: list[str] = []
        absent: list[str] = []
        newest = ""
        for row in rank_df.iter_rows(named=True):
            study_id = _cell(row, "study_id")
            newest = max(newest, _cell(row, "timestamp"))
            try:
                rounds = int(float(_cell(row, "rounds_played") or "0"))
            except ValueError:
                rounds = 0
            label = _cell(row, "nickname") or f"({study_id[:8]})"
            if rounds >= 12:
                long_runs.append(f"{label}:{rounds}")
            if study_id not in stats_rounds:
                absent.append(label)
            elif rounds > stats_rounds[study_id]:
                absent.append(f"{label}(ranking says {rounds}, statistics says {stats_rounds[study_id]})")
        print(f"  {name}: {rank_df.height} rows, newest {newest or '?'}")
        if long_runs:
            print(f"    12+ round rows: {', '.join(long_runs)}")
        if absent:
            print(f"    NOT MATCHED in statistics: {', '.join(absent)}")
    print(f"  statistics newest row: {stats_newest or '?'}")

    # Every name on the board, so "player X is missing" is answerable at a glance
    # -- and so a stale source directory (no recent names, old timestamps) is
    # obvious rather than being mistaken for a conversion bug.
    names: set[str] = set()
    for name in ("prediction_ranking.csv", "prediction_ranking_A.csv",
                 "prediction_ranking_B.csv", "prediction_ranking_C.csv"):
        rank_df = _read_csv(input_dir / name)
        if rank_df is None or "nickname" not in rank_df.columns:
            continue
        names.update(
            _cell(row, "nickname") for row in rank_df.iter_rows(named=True) if _cell(row, "nickname")
        )
    print(f"  nicknames in the ranking CSVs: {', '.join(sorted(names)) if names else '(none)'}")

    # ---- every distinct source, and how it classifies -----------------------
    sources: Counter[tuple[str, bool, str]] = Counter()
    rows_without_per_round = 0
    for row in df.iter_rows(named=True):
        diabetic = parse_diabetic_flag(row.get("diabetic"))
        run_source = _cell(row, "data_source_name")
        run_example = _cell(row, "is_example_data").lower() == "true"
        per_round = _parse_per_round_metrics(row.get("per_round_metrics"))
        if not per_round:
            rows_without_per_round += 1
            sources[(run_source or "(blank)", run_example, str(diabetic))] += 1
            continue
        for entry in per_round:
            name = str(entry.get("data_source_name") or run_source) or "(blank)"
            example = bool(entry.get("is_example_data", run_example))
            sources[(name, example, str(diabetic))] += 1

    print("\n--- round sources and their class ---")
    print(f"{'source':<34} {'rounds':>7}  {'is_example':<10} {'player diabetic':<15} class")
    unclassified = 0
    for (name, example, diabetic_repr), count in sources.most_common():
        player_diabetic = {"True": True, "False": False}.get(diabetic_repr)
        data_class = classify_round_source(
            name, is_example=example, player_diabetic=player_diabetic
        )
        if data_class is None:
            unclassified += count
        print(f"{name[:34]:<34} {count:>7}  {str(example):<10} {diabetic_repr:<15} "
              f"{data_class or 'UNCLASSIFIED -> counts for no board'}")

    # ---- per-run verdict ----------------------------------------------------
    print("\n--- per run ---")
    print(f"{'study_id':<10} {'run':<8} {'rounds':>6} {'diabetic':<9} "
          f"{'nondiab':>8} {'diab':>5}  verdict")
    ranked_runs = 0
    for row in df.iter_rows(named=True):
        diabetic = parse_diabetic_flag(row.get("diabetic"))
        run_source = _cell(row, "data_source_name")
        run_example = _cell(row, "is_example_data").lower() == "true"
        per_round = _parse_per_round_metrics(row.get("per_round_metrics"))
        counts = {cls: 0 for cls in DATA_CLASSES}
        for entry in per_round:
            if entry.get("mae") is None:
                continue
            data_class = classify_round_source(
                str(entry.get("data_source_name") or run_source),
                is_example=bool(entry.get("is_example_data", run_example)),
                player_diabetic=diabetic,
            )
            if data_class:
                counts[data_class] += 1

        try:
            rounds_played = int(float(_cell(row, "rounds_played") or "0"))
        except ValueError:
            rounds_played = 0

        if not per_round:
            # Mirrors the legacy branch of _runs_from_statistics: the whole run
            # classifies by its run-level source, each round at the overall MAE.
            legacy_class = classify_round_source(
                run_source, is_example=run_example, player_diabetic=diabetic
            )
            if legacy_class and rounds_played >= CLASS_SCORE_ROUNDS:
                counts[legacy_class] = rounds_played

        boards = [cls for cls, n in counts.items() if n >= CLASS_SCORE_ROUNDS]
        if boards:
            verdict = "ranked on " + ", ".join(boards)
            if not per_round:
                verdict += " (legacy row: no per_round_metrics, scored at the overall MAE)"
            ranked_runs += 1
        elif not per_round:
            verdict = ("no per_round_metrics; run-level source "
                       f"{run_source or '(blank)'} classifies to "
                       f"{classify_round_source(run_source, is_example=run_example, player_diabetic=diabetic) or 'nothing'}")
        elif not any(counts.values()):
            verdict = "no round has a class (see the source table above)"
        else:
            have = ", ".join(f"{cls}={n}" for cls, n in counts.items() if n)
            verdict = f"below the {CLASS_SCORE_ROUNDS}-round floor ({have})"

        print(f"{_cell(row, 'study_id')[:8]:<10} {_cell(row, 'run_id')[:6]:<8} "
              f"{rounds_played:>6} {str(diabetic):<9} "
              f"{counts['nondiabetic']:>8} {counts['diabetic']:>5}  {verdict}")

    # ---- what the page will actually render --------------------------------
    board = build_scoreboard(input_dir)
    print("\n--- what /highscore will show ---")
    for cls in DATA_CLASSES:
        entries = board.boards.get(cls) or []
        print(f"  {cls:<12} {len(entries)} entries")
    print(f"  players      {board.player_count()}")
    print(f"  player pages {len(board.players)}")

    if ranked_runs == 0:
        print("\nBoth boards are empty. The usual causes, in order:")
        if unclassified:
            print(f"  * {unclassified} rounds have an unclassifiable source. If those are")
            print("    example.csv, this deployment never downloaded the corpora, so Format A")
            print("    fell back to the bundled example trace -- which has no diabetes status")
            print("    and so belongs to neither board. `uv run download` on the box that")
            print("    generates the data, or decide example.csv's class deliberately.")
            print("    If they are own-data uploads, the player's `diabetic` column was not")
            print("    True/False, so the trace could not inherit a status.")
        if rows_without_per_round:
            print(f"  * {rows_without_per_round} rows have no per_round_metrics (older runs);")
            print("    they classify only if the run-level data_source_name identifies a corpus.")
        print(f"  * a run needs {CLASS_SCORE_ROUNDS} rounds of ONE class "
              f"(MIN_USEFUL_ROUNDS={MIN_USEFUL_ROUNDS}); rounds split across")
        print("    both classes do not add up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
