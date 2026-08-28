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
entry is instead scored on its **best ``CLASS_SCORE_ROUNDS`` rounds of that
class**, so everyone is judged on the same number of rounds and playing more
rounds can only help.  A run needs at least that many rounds *of the class*
to enter the class board at all (same floor rationale as the old board), and
the `/highscore` selector lets a visitor raise or lower that number.

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
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Optional

import polars as pl

from sugar_sugar.bigideas import is_bigideas_source_name
from sugar_sugar.config import MIN_USEFUL_ROUNDS, SCOREBOARD_CLASS_ROUNDS
from sugar_sugar.d1namo import is_d1namo_source_name
from sugar_sugar.nickname import deployment_salt, email_key, identity_key, normalize_nickname

DATA_CLASS_DIABETIC: Final[str] = "diabetic"
DATA_CLASS_NONDIABETIC: Final[str] = "nondiabetic"
DATA_CLASSES: Final[tuple[str, str]] = (DATA_CLASS_NONDIABETIC, DATA_CLASS_DIABETIC)

# Every board entry is scored on its best this-many rounds of the class, and
# needs at least this many such rounds to be ranked at all. Tunable via
# SCOREBOARD_CLASS_ROUNDS: it defaults to MIN_USEFUL_ROUNDS, but a deployment
# where most players stop after a few rounds can lower the board's bar without
# redefining what counts as useful study data everywhere else.
CLASS_SCORE_ROUNDS: Final[int] = SCOREBOARD_CLASS_ROUNDS

# Medal tiers, earned cumulatively and shown together so progress is visible:
# bronze for reaching BRONZE_ROUNDS in a game, silver for SILVER_ROUNDS, gold for
# having finished more than one game (what the veteran badge used to mark).
MEDAL_BRONZE_ROUNDS: Final[int] = 6
MEDAL_SILVER_ROUNDS: Final[int] = 12
MEDAL_GOLD_GAMES: Final[int] = 2

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


@lru_cache(maxsize=1)
def declared_source_classes() -> dict[str, str]:
    """Filename -> data class for every source that declares a diabetes status.

    Read from ``data/generic_sources_metadata.json`` and the discovered subject
    dirs, so a source's class is a property of the data rather than a filename
    pattern hardcoded here.  This is what places ``example.csv`` -- a real
    insulin-treated trace, and the bundled fallback when the corpora are not
    downloaded -- on the diabetic board instead of on neither.

    Cached: the underlying lookup scans the corpus directories, and this is
    consulted once per round.  Editing the metadata needs a restart, like the
    rest of the app's configuration.
    """
    from sugar_sugar.generic_sources_metadata import load_generic_sources_metadata

    classes: dict[str, str] = {}
    for name, meta in load_generic_sources_metadata().items():
        if meta.diabetic is True:
            classes[name] = DATA_CLASS_DIABETIC
        elif meta.diabetic is False:
            classes[name] = DATA_CLASS_NONDIABETIC
    return classes


def classify_round_source(
    source_name: str,
    *,
    is_example: bool,
    player_diabetic: Optional[bool],
    declared: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Which class board a round belongs to, or ``None`` for neither.

    In order: the two corpora and legacy LOOP subjects identify themselves by
    filename; anything else that *declares* a diabetes status in the generic
    source metadata uses it; a player's own uploaded data inherits the player's
    own status.  Only a source that is none of those -- an upload from someone
    who left the diabetes question blank -- stays unclassified.

    ``declared`` overrides the cached metadata lookup (tests, and callers that
    want to resolve it once for a whole board).
    """
    name = Path(str(source_name or "")).name
    if not name:
        return None
    if is_d1namo_source_name(name) or name.endswith("_chronological.csv"):
        return DATA_CLASS_DIABETIC
    if is_bigideas_source_name(name):
        return DATA_CLASS_NONDIABETIC

    lookup = declared_source_classes() if declared is None else declared
    if name in lookup:
        return lookup[name]

    # A generic/example source with nothing declared says nothing about its
    # subject, and must not borrow the *player's* status -- they are not the
    # person in the trace.
    if is_example or name == "example.csv":
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


def earned_medals(*, total_rounds: int, games: int) -> list[str]:
    """Medals an entry has earned, weakest first, awarded cumulatively.

    Bronze and silver come from how far the player got in *this* game; gold is a
    property of the player across games, which is why a single long game earns
    bronze and silver but never gold.
    """
    medals: list[str] = []
    if total_rounds >= MEDAL_BRONZE_ROUNDS:
        medals.append("bronze")
    if total_rounds >= MEDAL_SILVER_ROUNDS:
        medals.append("silver")
    if games >= MEDAL_GOLD_GAMES:
        medals.append("gold")
    return medals


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
    # Runs in the statistics CSV that produced no board slot at all. Surfaced on
    # the page so the gap between "N games played" and a short board reads as a
    # known rule rather than as data having gone missing.
    unranked_runs: int = 0
    total_runs: int = 0

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
    nickname: str = ""


def _read_csv(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    try:
        return pl.read_csv(path, infer_schema_length=0)
    except Exception:
        return None


def _cell(row: dict[str, Any], column: str) -> str:
    return str(row.get(column) or "").strip()


def _statistics_index(stats_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """``(study_id, run_id)`` -> the per-round detail only this file carries.

    The statistics CSV is the *research* record; it is read here purely to enrich
    a leaderboard row with per-round sources (which decide the class of each
    round) and the player's own diabetes status (which decides hard mode). It is
    deliberately NOT the row source: a run present here but absent from the
    ranking CSVs was never on the board, and -- as production showed -- the
    reverse happens too.
    """
    df = _read_csv(stats_path)
    if df is None or "study_id" not in df.columns:
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        study_id = _cell(row, "study_id")
        if not study_id:
            continue
        index[(study_id, _cell(row, "run_id"))] = {
            "diabetic": parse_diabetic_flag(row.get("diabetic")),
            "per_round": _parse_per_round_metrics(row.get("per_round_metrics")),
            "is_example": _cell(row, "is_example_data").lower() == "true",
            "source": _cell(row, "data_source_name"),
        }
    return index


def _runs_from_ranking(
    ranking_paths: list[Path],
    stats_index: dict[tuple[str, str], dict[str, Any]],
) -> list[_Run]:
    """Every board row, taken from the ranking CSVs the leaderboard is written to.

    One row per format run (the per-format files), never the cumulative "ALL"
    file, whose rounds span formats and whose source column is only the last one
    used.  Identity comes from the ``email_key`` stored on the row itself, so
    this module never needs a raw address, and a copied deployment works as soon
    as those keys are re-derived under its own salt.

    Rows with a matching statistics entry are scored per round, so a mixed run
    can land on both boards.  Rows without one -- players the research file does
    not have -- are scored at their overall MAE against the class their run-level
    source names, which is how the previous board treated every row.
    """
    runs: list[_Run] = []
    for path in ranking_paths:
        df = _read_csv(path)
        if df is None or "study_id" not in df.columns:
            continue
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
            if overall_mae is None:
                continue

            detail = stats_index.get((study_id, _cell(row, "run_id"))) or {}
            diabetic = detail.get("diabetic")
            run_source = _cell(row, "data_source_name") or str(detail.get("source") or "")
            run_is_example = _cell(row, "is_example_data").lower() == "true"

            class_maes: dict[str, list[float]] = {cls: [] for cls in DATA_CLASSES}
            per_round: list[dict[str, Any]] = list(detail.get("per_round") or [])
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
            if not any(class_maes.values()) and total_rounds > 0:
                cls = classify_round_source(
                    run_source, is_example=run_is_example, player_diabetic=diabetic
                )
                if cls is not None:
                    class_maes[cls] = [overall_mae] * total_rounds

            runs.append(
                _Run(
                    study_id=study_id,
                    identity=identity_key(key=_cell(row, "email_key"), study_id=study_id),
                    timestamp=_cell(row, "timestamp"),
                    format=_cell(row, "format"),
                    total_rounds=total_rounds,
                    overall_mae=overall_mae,
                    diabetic=diabetic,
                    class_maes=class_maes,
                    nickname=normalize_nickname(_cell(row, "nickname")),
                )
            )
    return runs


def _class_score(maes: list[float], class_rounds: int) -> Optional[float]:
    """Mean MAE over the best ``class_rounds`` rounds, or ``None`` below the floor.

    Scoring everyone on the same number of rounds is what removes the length
    handicap: a mean over 12 rounds cannot get lucky the way a mean over 6 can,
    so without this a longer game is punished for its own sample size. Taking
    each player's best N equalises the count, and playing more rounds can then
    only help.

    A row the statistics file has no per-round detail for arrives here as its
    overall mean repeated once per round (see :func:`_runs_from_ranking`), so
    its best N is that mean -- the only figure it has. In a healthy deployment
    every board row has per-round detail and is scored identically; a row
    reduced to its mean is scored slightly less favourably than one whose best
    rounds could be picked, which is the safe direction for a missing record.
    """
    if len(maes) < class_rounds:
        return None
    return sum(sorted(maes)[:class_rounds]) / class_rounds


def build_scoreboard(input_dir: Path, *, class_rounds: Optional[int] = None) -> Scoreboard:
    """Build both class boards and the per-player rollup from the CSVs.

    ``class_rounds`` is the per-class round floor and the averaging window;
    defaults to :data:`CLASS_SCORE_ROUNDS`.  The `/highscore` selector passes it
    so a visitor can widen the board without the deployment changing anything.
    """
    floor = max(1, class_rounds or CLASS_SCORE_ROUNDS)
    stats_path = input_dir / "prediction_statistics.csv"
    # Per-format files only: one row per format run. The cumulative "ALL" file
    # would double-count a player who switched formats.
    runs = _runs_from_ranking(
        [input_dir / f"prediction_ranking_{fmt}.csv" for fmt in ("A", "B", "C")],
        _statistics_index(stats_path),
    )
    if not runs:
        # A deployment that only ever wrote the cumulative file still gets a board.
        runs = _runs_from_ranking([input_dir / "prediction_ranking.csv"], _statistics_index(stats_path))

    finished_games: dict[str, int] = {}
    for run in runs:
        if run.total_rounds >= MIN_USEFUL_ROUNDS:
            finished_games[run.identity] = finished_games.get(run.identity, 0) + 1

    boards: dict[str, list[BoardEntry]] = {cls: [] for cls in DATA_CLASSES}
    games_by_identity: dict[str, list[PlayerGame]] = {}
    best_by_identity: dict[str, dict[str, float]] = {}
    profile_by_identity: dict[str, dict[str, Any]] = {}

    for run in runs:
        nickname = run.nickname
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
            score = _class_score(run.class_maes[cls], floor)
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

    ranked_studies = {
        (entry.study_id, entry.timestamp)
        for entries in boards.values()
        for entry in entries
    }
    unranked = sum(
        1 for run in runs if (run.study_id, run.timestamp) not in ranked_studies
    )

    return Scoreboard(
        boards=boards,
        players=players,
        unranked_runs=unranked,
        total_runs=len(runs),
    )


def entry_is_own(entry: BoardEntry, *, study_id: str, key: str) -> bool:
    """Whether a board entry belongs to the current visitor.

    Same matching as the old board's ``_own_entries``: the hashed email merges
    devices, ``study_id`` covers rows written before an email was given.
    """
    if study_id and entry.study_id == study_id:
        return True
    return bool(key) and entry.identity == identity_key(key=key, study_id="")
