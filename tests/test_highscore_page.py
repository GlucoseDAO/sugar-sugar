"""The public `/highscore` page renders its two class boards from the CSVs alone.

The page splits into a non-diabetic-data and a diabetic-data table (flatter
non-diabetic traces are objectively easier, so the two are never ranked against
each other), scores every entry on its best ``CLASS_SCORE_ROUNDS`` rounds of the
class, badges hard-mode (foreign data) and veteran (>1 finished game) entries,
and links veterans to their public `/player/<id>` page.  It is reachable from
the desktop navbar and the mobile burger menu, so it must render for a visitor
with no session at all, highlight the current player's rows when there *is* a
session, and degrade to an explanatory empty state when nobody has finished a
game yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from sugar_sugar import app as app_module
from sugar_sugar.app import HARD_MODE_BADGE, VETERAN_BADGE, create_highscore_page, create_player_page
from sugar_sugar.nickname import email_key, identity_key
from sugar_sugar.scoreboard import public_player_id

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
            "is_example_data": False,
            "generic_slice_key": "",
        }
        for i, (mae, src) in enumerate(maes_sources)
    ]
    return '"' + str(rows).replace('"', '""') + '"'


def _flat_rounds(mae: float, source: str, count: int = 12) -> str:
    return _per_round_cell([(mae, source)] * count)


def _stats_row(
    study: str,
    run: str = "r1",
    *,
    ts: str = "2026-08-01 10:00:00",
    email: str = "",
    fmt: str = "A",
    diabetic: str = "False",
    rounds: int = 12,
    mae: float = 20.0,
    per_round: str = '"[]"',
) -> str:
    return (
        f"{study},{run},1,{ts},{email},{fmt},False,src.csv,30,1,female,"
        f"True,1,{diabetic},,,,{rounds},x,x,x,{mae},0,0,0,{per_round}\n"
    )


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the app's `data/input` CSV lookups at a throwaway tree."""
    (tmp_path / "data" / "input").mkdir(parents=True)
    monkeypatch.setattr(app_module, "project_root", tmp_path)
    yield tmp_path


def _write_stats(root: Path, rows: str) -> None:
    (root / "data" / "input" / "prediction_statistics.csv").write_text(
        STATS_HEADER + rows, encoding="utf-8"
    )


def _texts(component: Any) -> list[str]:
    """Every string in the rendered tree, flattened."""
    out: list[str] = []
    children = getattr(component, "children", None)
    if isinstance(component, str):
        return [component]
    if isinstance(children, str):
        out.append(children)
    elif isinstance(children, (list, tuple)):
        for child in children:
            out.extend(_texts(child))
    elif children is not None:
        out.extend(_texts(children))
    return out


def _board_titles(component: Any) -> list[str]:
    """Titles of every rendered leaderboard board, in document order."""
    out: list[str] = []
    if getattr(component, "className", None) == "final-leaderboard-board-title":
        out.append(str(component.children))
        return out
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            out.extend(_board_titles(child))
    elif children is not None and not isinstance(children, str):
        out.extend(_board_titles(children))
    return out


def _class_names(component: Any) -> list[str]:
    out: list[str] = []
    cls = getattr(component, "className", None)
    if isinstance(cls, str):
        out.append(cls)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            out.extend(_class_names(child))
    elif children is not None and not isinstance(children, str):
        out.extend(_class_names(children))
    return out


def _links(component: Any) -> list[str]:
    """hrefs of every dcc.Link in the tree."""
    out: list[str] = []
    href = getattr(component, "href", None)
    if isinstance(href, str):
        out.append(href)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            out.extend(_links(child))
    elif children is not None and not isinstance(children, str):
        out.extend(_links(children))
    return out


def test_highscore_splits_into_the_two_class_boards(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("nd", per_round=_flat_rounds(15.0, "BIGIDEAS-001.csv"))
        + _stats_row("d1", diabetic="True", per_round=_flat_rounds(30.0, "D1NAMO-001.csv")),
    )
    page = create_highscore_page(None, None, locale="en")
    assert _board_titles(page) == ["Non-diabetic data", "Diabetic data"]
    joined = " ".join(_texts(page))
    assert "15.00" in joined and "30.00" in joined


def test_highscore_hides_runs_below_the_class_round_floor(data_root: Path) -> None:
    """5 rounds of a class is study data, not a board slot."""
    _write_stats(
        data_root,
        _stats_row("short", rounds=5, mae=1.0, per_round=_flat_rounds(1.0, "BIGIDEAS-001.csv", 5))
        + _stats_row("full", mae=18.0, per_round=_flat_rounds(18.0, "BIGIDEAS-002.csv")),
    )
    texts = " ".join(_texts(create_highscore_page(None, None, locale="en")))
    assert "1.00" not in texts
    assert "18.00" in texts
    assert "Player 1" in texts
    assert "Player 2" not in texts


def test_highscore_scores_on_best_rounds_so_12_rounds_beat_no_handicap(data_root: Path) -> None:
    """The 12-round run's best 6 rounds beat the 6-round run's only 6."""
    twelve = _per_round_cell([(10.0 + i, "BIGIDEAS-001.csv") for i in range(12)])
    six = _per_round_cell([(14.0 + i, "BIGIDEAS-002.csv") for i in range(6)])
    _write_stats(
        data_root,
        _stats_row("long", rounds=12, per_round=twelve)
        + _stats_row("short", rounds=6, per_round=six),
    )
    texts = _texts(create_highscore_page(None, None, locale="en"))
    joined = " ".join(texts)
    assert "12.50" in joined  # long run: mean of rounds 10..15
    assert "16.50" in joined  # short run: mean of rounds 14..19
    assert texts.index("12.50") < texts.index("16.50")


def test_highscore_empty_state_without_any_csv(data_root: Path) -> None:
    page = create_highscore_page(None, None, locale="en")
    texts = " ".join(_texts(page))
    assert "No ranked games yet" in texts
    assert "Play now" in texts


def test_highscore_renders_anonymous_board_without_a_session(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("s1", per_round=_flat_rounds(18.0, "BIGIDEAS-001.csv"))
        + _stats_row("s2", per_round=_flat_rounds(9.0, "BIGIDEAS-002.csv"))
        + _stats_row("s3", per_round=_flat_rounds(27.0, "BIGIDEAS-003.csv")),
    )
    page = create_highscore_page(None, None, locale="en")
    texts = _texts(page)
    joined = " ".join(texts)
    assert texts.index("Player 1") < texts.index("Player 2") < texts.index("Player 3")
    assert "s1" not in texts and "s2" not in texts
    assert "9.00" in joined  # best score, rank 1
    assert "3 players" in joined
    assert "You" not in texts


def test_highscore_highlights_the_current_player_with_hero_line(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("me", per_round=_flat_rounds(20.0, "BIGIDEAS-001.csv"))
        + _stats_row("other", per_round=_flat_rounds(10.0, "BIGIDEAS-002.csv")),
    )
    page = create_highscore_page({"study_id": "me"}, "mg/dL", locale="en")
    texts = _texts(page)
    assert "You" in texts
    assert "final-leaderboard-row you" in _class_names(page)
    assert "Non-diabetic data: #2 / 2 · MAE 20.00 mg/dL" in " ".join(texts)


def test_highscore_marks_hard_mode_with_a_badge(data_root: Path) -> None:
    """A non-diabetic player's D1NAMO score carries the hard-mode badge."""
    mixed = _per_round_cell(
        [(15.0, "BIGIDEAS-001.csv")] * 6 + [(30.0, "D1NAMO-001.csv")] * 6
    )
    _write_stats(data_root, _stats_row("challenger", per_round=mixed))
    page = create_highscore_page(None, None, locale="en")
    texts = _texts(page)
    # Exactly one badge: the diabetic-data slot, not the home nondiabetic one.
    assert texts.count(HARD_MODE_BADGE) == 1


def test_highscore_veteran_gets_badge_and_player_link(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("vet", "r1", email="vet@x.com", per_round=_flat_rounds(15.0, "BIGIDEAS-001.csv"))
        + _stats_row("vet", "r2", email="vet@x.com", ts="2026-08-02 10:00:00",
                     per_round=_flat_rounds(14.0, "BIGIDEAS-002.csv"))
        + _stats_row("solo", per_round=_flat_rounds(16.0, "BIGIDEAS-003.csv")),
    )
    page = create_highscore_page(None, None, locale="en")
    texts = _texts(page)
    assert texts.count(VETERAN_BADGE) >= 2  # both of the veteran's slots
    pid = public_player_id(identity_key(key=email_key("vet@x.com"), study_id="vet"))
    player_links = [href for href in _links(page) if href.startswith("/player/")]
    assert player_links == [f"/player/{pid}", f"/player/{pid}"]


def test_highscore_shows_nicknames_instead_of_player_n(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("s1", email="ann@x.com", per_round=_flat_rounds(9.0, "BIGIDEAS-001.csv"))
        + _stats_row("s2", per_round=_flat_rounds(18.0, "BIGIDEAS-002.csv")),
    )
    (data_root / "data" / "input" / "prediction_ranking.csv").write_text(
        RANK_HEADER
        + f"s1,r1,1,2026-08-01 10:00:00,ALL,12,False,src.csv,9.0,0,0,0,{email_key('ann@x.com')},SugarNinja\n",
        encoding="utf-8",
    )
    page = create_highscore_page(None, None, locale="en")
    texts = _texts(page)
    assert "SugarNinja" in texts
    assert "Player 1" not in texts
    assert "Player 2" in texts


def test_highscore_never_leaks_email_hash_or_study_id(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("me", email="ann@x.com", per_round=_flat_rounds(9.0, "BIGIDEAS-001.csv")),
    )
    page = create_highscore_page({"study_id": "me", "email": "ann@x.com"}, None, locale="en")
    joined = " ".join(_texts(page))
    assert email_key("ann@x.com") not in joined
    assert "ann@x.com" not in joined and "@" not in joined
    assert "me" not in _texts(page)


def test_highscore_keeps_every_finished_game_on_the_board(data_root: Path) -> None:
    """Arcade rules: beating your own score does not remove the old one."""
    _write_stats(
        data_root,
        _stats_row("vet", "r1", email="ann@x.com", per_round=_flat_rounds(22.0, "BIGIDEAS-001.csv"))
        + _stats_row("vet", "r2", email="ann@x.com", ts="2026-08-02 10:00:00",
                     per_round=_flat_rounds(14.0, "BIGIDEAS-002.csv"))
        + _stats_row("s3", per_round=_flat_rounds(19.0, "BIGIDEAS-003.csv")),
    )
    joined = " ".join(_texts(create_highscore_page(None, None, locale="en")))
    assert "14.00" in joined and "22.00" in joined and "19.00" in joined
    assert "2 players" in joined


def test_highscore_shows_the_played_datetime_and_rounds(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("s1", ts="2026-08-02 18:30:00", per_round=_flat_rounds(14.0, "BIGIDEAS-001.csv")),
    )
    texts = _texts(create_highscore_page(None, None, locale="en"))
    assert "When" in texts
    assert "2026-08-02" in texts
    assert "18:30" in texts
    assert "18:30:00" not in texts
    assert "Rounds" in texts
    assert "12" in texts  # the class-round count behind the score


def test_highscore_explains_what_it_stores(data_root: Path) -> None:
    _write_stats(data_root, _stats_row("s1", per_round=_flat_rounds(18.0, "BIGIDEAS-001.csv")))
    joined = " ".join(_texts(create_highscore_page(None, None, locale="en")))
    assert "one-way hash" in joined
    assert "not part of the study data" in joined


def test_highscore_converts_mae_to_mmol(data_root: Path) -> None:
    _write_stats(data_root, _stats_row("me", per_round=_flat_rounds(18.0, "BIGIDEAS-001.csv")))
    page = create_highscore_page({"study_id": "me"}, "mmol/L", locale="en")
    joined = " ".join(_texts(page))
    assert "1.00" in joined  # 18 mg/dL == 1 mmol/L
    assert "MAE (mmol/L)" in joined


def test_player_page_lists_timestamped_games(data_root: Path) -> None:
    _write_stats(
        data_root,
        _stats_row("vet", "r1", email="vet@x.com", per_round=_flat_rounds(15.0, "BIGIDEAS-001.csv"))
        + _stats_row("vet", "r2", email="vet@x.com", fmt="C", ts="2026-08-02 11:30:00",
                     per_round=_per_round_cell(
                         [(14.0, "BIGIDEAS-002.csv")] * 6 + [(28.0, "D1NAMO-001.csv")] * 6
                     )),
    )
    pid = public_player_id(identity_key(key=email_key("vet@x.com"), study_id="vet"))
    page = create_player_page(pid, "mg/dL", locale="en")
    texts = _texts(page)
    joined = " ".join(texts)
    assert "Player statistics" in texts
    assert "2 finished games" in joined
    assert "2026-08-01 10:00" in joined and "2026-08-02 11:30" in joined
    assert "Best score on non-diabetic data: 14.00 mg/dL" in joined
    assert "Best score on diabetic data: 28.00 mg/dL" in joined
    # The mixed run is hard mode (foreign D1NAMO rounds for a non-diabetic).
    assert HARD_MODE_BADGE in texts
    # Nothing private on a public page.
    assert "vet@x.com" not in joined and email_key("vet@x.com") not in joined
    assert "vet" not in texts


def test_player_page_unknown_id_renders_not_found(data_root: Path) -> None:
    page = create_player_page("deadbeefdeadbeef", None, locale="en")
    joined = " ".join(_texts(page))
    assert "no player page" in joined
    assert "Back to the highscore board" in joined
