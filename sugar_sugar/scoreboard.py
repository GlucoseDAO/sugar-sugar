"""Class scoreboards: diabetic vs non-diabetic data, scored per round class.

Why the split: non-diabetic traces (BIG IDEAs) are objectively easier to
predict -- flatter lines yield a lower MAE for the same skill -- so one mixed
board systematically ranks BIG IDEAs players above D1NAMO players.  The public
board therefore splits into two tables keyed by the **data being predicted**,
not by who predicted it:

* ``nondiabetic`` -- BIG IDEAs rounds, plus a non-diabetic player's own data.
* ``diabetic``    -- D1NAMO and legacy LOOP rounds, plus a diabetic player's
  own data.

Rounds that classify to neither side (``example.csv``, own data with no stated
diabetes status) stay off both boards.

Why per-run "best N rounds" scoring: a cumulative MAE over 12 rounds carries
twice the regression-to-the-mean of a 6-round run -- the short run can get
lucky, the long one cannot, so 12-round players were handicapped.  Every
entry is instead scored on its **best ``MIN_USEFUL_ROUNDS`` rounds of that
class**, so everyone is judged on the same number of rounds and playing more
rounds can only help.  A run needs at least that many rounds *of the class*
to enter the class board at all (same floor rationale as the old board).

Badges:

* **hard mode** -- the entry's player predicted data foreign to their own
  condition (a diabetic on the non-diabetic board or vice versa; this is what
  "Challenge the unknown" runs produce, and what mixed-policy players do half
  the time).
* **veteran** -- the identity behind the entry finished more than one game.
  Veteran rows link to ``/player/<public_id>``, a public per-player page.

``public_id`` is an HMAC of the leaderboard identity under the deployment
salt (domain-prefixed, like share ids), so the page URL exposes neither
``study_id`` nor ``email_key``.

Data sources: ``prediction_statistics.csv`` is the base -- one row per
finished run (``study_id`` + ``run_id``) with ``per_round_metrics`` carrying
each round's MAE and data source -- joined with the ranking CSVs for the
public nickname.  The raw email is reduced to :func:`~sugar_sugar.nickname.email_key`
immediately and never leaves this module.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional

import polars as pl

from sugar_sugar.bigideas import is_bigideas_source_name
from sugar_sugar.config import MIN_USEFUL_ROUNDS
from sugar_sugar.d1namo import is_d1namo_source_name
from sugar_sugar.nickname import deployment_salt, email_key, identity_key, normalize_nickname

DATA_CLASS_DIABETIC: Final[str] = "diabetic"
DATA_CLASS_NONDIABETIC: Final[str] = "nondiabetic"
DATA_CLASSES: Final[tuple[str, str]] = (DATA_CLASS_NONDIABETIC, DATA_CLASS_DIABETIC)

# Every board entry is scored on its best this-many rounds of the class, and
# needs at least this many such rounds to be ranked at all.
CLASS_SCORE_ROUNDS: Final[int] = MIN_USEFUL_ROUNDS

_PLAYER_ID_DOMAIN: Final[bytes] = b"player-id:"
_PLAYER_ID_HEX_LENGTH: Final[int] = 16


def public_player_id(identity: str) -> str:
    """Stable public identifier for a leaderboard identity.

    Domain-prefixed HMAC under the deployment salt, so it can appear in a URL
    without exposing (or being correlatable with) ``study_id`` / ``email_key``.
    """
    if not identity:
        return ""
    digest = hmac.new(
        deployment_salt(), _PLAYER_ID_DOMAIN + identity.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return digest[:_PLAYER_ID_HEX_LENGTH]


def parse_diabetic_flag(raw: Any) -> Optional[bool]:
    """The player's own diabetes status as stored in the statistics CSV."""
    if isinstance(raw, bool):
        return raw
    text = str(raw or "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def classify_round_source(
    source_name: str,
    *,
    is_example: bool,
    player_diabetic: Optional[bool],
) -> Optional[str]:
    """Which class board a round belongs to, or ``None`` for neither.

    Corpus rounds classify by the corpus itself; a player's own uploaded data
    classifies by the player's stated status.  ``example.csv`` rounds carry no
    provenance and stay unclassified.
    """
    name = Path(str(source_name or "")).name
    if is_d1namo_source_name(name) or name.endswith("_chronological.csv"):
        return DATA_CLASS_DIABETIC
    if is_bigideas_source_name(name):
        return DATA_CLASS_NONDIABETIC
    if is_example or name == "example.csv" or not name:
        return None
    # Own uploaded data: the trace is the player's, so it inherits their status.
    if player_diabetic is True:
        return DATA_CLASS_DIABETIC
    if player_diabetic is False:
        return DATA_CLASS_NONDIABETIC
    return None


def is_hard_mode(data_class: str, player_diabetic: Optional[bool]) -> bool:
    """True when the player predicted data foreign to their own condition."""
    if player_diabetic is None:
        return False
    if data_class == DATA_CLASS_DIABETIC:
        return not player_diabetic
    if data_class == DATA_CLASS_NONDIABETIC:
        return player_diabetic
    return False


def _parse_per_round_metrics(raw: Any) -> list[dict[str, Any]]:
    """The ``per_round_metrics`` cell -- a Python-literal list -- or ``[]``."""
    text = str(raw or "").strip()
    if not text.startswith("["):
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


@dataclass(frozen=True, slots=True)
class BoardEntry:
    """One finished run's slot on one class board.  ``study_id`` is matching
    context for the "You" highlight and must never be rendered."""

    identity: str
    public_id: str
    study_id: str
    nickname: str
    timestamp: str
    score_mae: float
    class_rounds: int
    total_rounds: int
    hard_mode: bool
    games: int


@dataclass(frozen=True, slots=True)
class PlayerGame:
    """One finished run on a player's public statistics page."""

    timestamp: str
    format: str
    total_rounds: int
    overall_mae: Optional[float]
    class_rounds: dict[str, int]
    class_maes: dict[str, float]
    hard_mode: bool


@dataclass(frozen=True, slots=True)
class PlayerStats:
    public_id: str
    nickname: str
    diabetic: Optional[bool]
    games: list[PlayerGame]
    best_scores: dict[str, float]
    hard_mode: bool


@dataclass(frozen=True, slots=True)
class Scoreboard:
    boards: dict[str, list[BoardEntry]] = field(default_factory=dict)
    players: dict[str, PlayerStats] = field(default_factory=dict)

    def player_count(self) -> int:
        return len(
            {entry.identity for entries in self.boards.values() for entry in entries}
        )


@dataclass(frozen=True, slots=True)
class _Run:
    study_id: str
    identity: str
    timestamp: str
    format: str
    total_rounds: int
    overall_mae: Optional[float]
    diabetic: Optional[bool]
    class_maes: dict[str, list[float]]


def _read_csv(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    try:
        return pl.read_csv(path, infer_schema_length=0)
    except Exception:
        return None


def _cell(row: dict[str, Any], column: str) -> str:
    return str(row.get(column) or "").strip()


def _nickname_by_study(ranking_paths: list[Path]) -> dict[str, str]:
    """Newest non-empty nickname each ``study_id`` set on any ranking CSV.

    Nicknames live only in the ranking CSVs (they are a public label, not study
    data) and are stamped per ``study_id`` by ``set_study_nickname``.
    """
    latest: dict[str, tuple[str, str]] = {}
    for path in ranking_paths:
        df = _read_csv(path)
        if df is None or "study_id" not in df.columns or "nickname" not in df.columns:
            continue
        for row in df.iter_rows(named=True):
            name = normalize_nickname(_cell(row, "nickname"))
            study_id = _cell(row, "study_id")
            if not name or not study_id:
                continue
            timestamp = _cell(row, "timestamp")
            if study_id not in latest or timestamp >= latest[study_id][0]:
                latest[study_id] = (timestamp, name)
    return {study_id: name for study_id, (_, name) in latest.items()}


def _runs_from_statistics(stats_path: Path) -> list[_Run]:
    df = _read_csv(stats_path)
    if df is None or "study_id" not in df.columns:
        return []
    runs: list[_Run] = []
    for row in df.iter_rows(named=True):
        study_id = _cell(row, "study_id")
        if not study_id:
            continue
        try:
            total_rounds = int(float(_cell(row, "rounds_played") or "0"))
        except ValueError:
            total_rounds = 0
        try:
            overall_mae: Optional[float] = float(_cell(row, "overall_mae_mgdl"))
        except ValueError:
            overall_mae = None
        diabetic = parse_diabetic_flag(row.get("diabetic"))
        run_is_example = _cell(row, "is_example_data").lower() == "true"
        run_source = _cell(row, "data_source_name")

        class_maes: dict[str, list[float]] = {cls: [] for cls in DATA_CLASSES}
        per_round = _parse_per_round_metrics(row.get("per_round_metrics"))
        for entry in per_round:
            mae = entry.get("mae")
            if mae is None:
                continue
            cls = classify_round_source(
                str(entry.get("data_source_name") or run_source),
                is_example=bool(entry.get("is_example_data", run_is_example)),
                player_diabetic=diabetic,
            )
            if cls is not None:
                class_maes[cls].append(float(mae))
        if not per_round and overall_mae is not None and total_rounds > 0:
            # Legacy rows without per-round records: the whole run classifies (or
            # not) by its run-level source, each round scored at the overall MAE.
            cls = classify_round_source(
                run_source, is_example=run_is_example, player_diabetic=diabetic
            )
            if cls is not None:
                class_maes[cls] = [overall_mae] * total_rounds

        runs.append(
            _Run(
                study_id=study_id,
                identity=identity_key(key=email_key(_cell(row, "email")), study_id=study_id),
                timestamp=_cell(row, "timestamp"),
                format=_cell(row, "format"),
                total_rounds=total_rounds,
                overall_mae=overall_mae,
                diabetic=diabetic,
                class_maes=class_maes,
            )
        )
    return runs


def _class_score(maes: list[float]) -> Optional[float]:
    """Mean MAE over the best ``CLASS_SCORE_ROUNDS`` rounds, or ``None`` below
    the floor.  Judging everyone on the same round count is what removes the
    6-vs-12-round handicap."""
    if len(maes) < CLASS_SCORE_ROUNDS:
        return None
    best = sorted(maes)[:CLASS_SCORE_ROUNDS]
    return sum(best) / len(best)


def build_scoreboard(input_dir: Path) -> Scoreboard:
    """Build both class boards and the per-player rollup from the CSVs."""
    stats_path = input_dir / "prediction_statistics.csv"
    ranking_paths = [input_dir / "prediction_ranking.csv"] + [
        input_dir / f"prediction_ranking_{fmt}.csv" for fmt in ("A", "B", "C")
    ]
    runs = _runs_from_statistics(stats_path)
    nicknames = _nickname_by_study(ranking_paths)

    finished_games: dict[str, int] = {}
    for run in runs:
        if run.total_rounds >= MIN_USEFUL_ROUNDS:
            finished_games[run.identity] = finished_games.get(run.identity, 0) + 1

    boards: dict[str, list[BoardEntry]] = {cls: [] for cls in DATA_CLASSES}
    games_by_identity: dict[str, list[PlayerGame]] = {}
    best_by_identity: dict[str, dict[str, float]] = {}
    profile_by_identity: dict[str, dict[str, Any]] = {}

    for run in runs:
        nickname = nicknames.get(run.study_id, "")
        pid = public_player_id(run.identity)
        profile = profile_by_identity.setdefault(
            run.identity, {"nickname": "", "diabetic": None, "study_ids": set()}
        )
        profile["study_ids"].add(run.study_id)
        if nickname:
            profile["nickname"] = nickname
        if run.diabetic is not None:
            profile["diabetic"] = run.diabetic

        run_hard = any(
            is_hard_mode(cls, run.diabetic) and run.class_maes[cls] for cls in DATA_CLASSES
        )
        if run.total_rounds >= MIN_USEFUL_ROUNDS:
            games_by_identity.setdefault(run.identity, []).append(
                PlayerGame(
                    timestamp=run.timestamp,
                    format=run.format,
                    total_rounds=run.total_rounds,
                    overall_mae=run.overall_mae,
                    class_rounds={
                        cls: len(maes) for cls, maes in run.class_maes.items() if maes
                    },
                    class_maes={
                        cls: sum(maes) / len(maes)
                        for cls, maes in run.class_maes.items()
                        if maes
                    },
                    hard_mode=run_hard,
                )
            )

        for cls in DATA_CLASSES:
            score = _class_score(run.class_maes[cls])
            if score is None:
                continue
            boards[cls].append(
                BoardEntry(
                    identity=run.identity,
                    public_id=pid,
                    study_id=run.study_id,
                    nickname=nickname,
                    timestamp=run.timestamp,
                    score_mae=score,
                    class_rounds=len(run.class_maes[cls]),
                    total_rounds=run.total_rounds,
                    hard_mode=is_hard_mode(cls, run.diabetic),
                    games=finished_games.get(run.identity, 0),
                )
            )
            best = best_by_identity.setdefault(run.identity, {})
            if cls not in best or score < best[cls]:
                best[cls] = score

    for cls in DATA_CLASSES:
        boards[cls].sort(key=lambda e: (e.score_mae, e.timestamp, e.study_id))

    players: dict[str, PlayerStats] = {}
    for identity, games in games_by_identity.items():
        profile = profile_by_identity[identity]
        players[public_player_id(identity)] = PlayerStats(
            public_id=public_player_id(identity),
            nickname=str(profile["nickname"]),
            diabetic=profile["diabetic"],
            games=sorted(games, key=lambda g: g.timestamp),
            best_scores=best_by_identity.get(identity, {}),
            hard_mode=any(game.hard_mode for game in games),
        )

    return Scoreboard(boards=boards, players=players)


def entry_is_own(entry: BoardEntry, *, study_id: str, key: str) -> bool:
    """Whether a board entry belongs to the current visitor.

    Same matching as the old board's ``_own_entries``: the hashed email merges
    devices, ``study_id`` covers rows written before an email was given.
    """
    if study_id and entry.study_id == study_id:
        return True
    return bool(key) and entry.identity == identity_key(key=key, study_id="")
