"""The public `/highscore` page renders from the ranking CSVs alone.

It is reachable from the desktop navbar and the mobile burger menu, so it must
render for a visitor with no session at all (no `user-info-store`), highlight the
current player's row when there *is* a session, and degrade to an explanatory
empty state when nobody has finished a game yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from sugar_sugar import app as app_module
from sugar_sugar.app import create_highscore_page

_HEADER = (
    "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
    "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,overall_mape_pct\n"
)


def _row(study_id: str, fmt: str, mae: float, *, ts: str = "2026-08-01 10:00:00") -> str:
    return f"{study_id},run1,1,{ts},{fmt},12,True,example,{mae},0,0,0\n"


@pytest.fixture()
def ranking_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the app's `data/input` ranking lookups at a throwaway tree."""
    (tmp_path / "data" / "input").mkdir(parents=True)
    monkeypatch.setattr(app_module, "project_root", tmp_path)
    yield tmp_path


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


def test_highscore_empty_state_without_any_ranking_csv(ranking_root: Path) -> None:
    page = create_highscore_page(None, None, locale="en")
    texts = " ".join(_texts(page))
    assert "No finished games yet" in texts
    assert "Play now" in texts


def test_highscore_renders_anonymous_board_without_a_session(ranking_root: Path) -> None:
    overall = ranking_root / "data" / "input" / "prediction_ranking.csv"
    overall.write_text(
        _HEADER + _row("s1", "ALL", 18.0) + _row("s2", "ALL", 9.0) + _row("s3", "ALL", 27.0),
        encoding="utf-8",
    )

    page = create_highscore_page(None, None, locale="en")
    texts = _texts(page)
    joined = " ".join(texts)
    # Ranked best-first, anonymous, and never leaking a study_id.
    assert texts.index("Player 1") < texts.index("Player 2") < texts.index("Player 3")
    assert "s1" not in texts and "s2" not in texts
    assert "9.00" in joined  # best MAE, rank 1
    assert "3 players" in joined
    assert "You" not in texts


def test_highscore_highlights_the_current_player(ranking_root: Path) -> None:
    overall = ranking_root / "data" / "input" / "prediction_ranking.csv"
    overall.write_text(
        _HEADER + _row("me", "ALL", 20.0) + _row("other", "ALL", 10.0),
        encoding="utf-8",
    )

    page = create_highscore_page({"study_id": "me"}, "mg/dL", locale="en")
    texts = _texts(page)
    assert "You" in texts
    assert "final-leaderboard-row you" in _class_names(page)
    # "Your place" hero comes from the same snapshot as `/final`.
    assert "#2 / 2" in " ".join(texts)


def test_highscore_shows_a_board_per_played_data_source(ranking_root: Path) -> None:
    input_dir = ranking_root / "data" / "input"
    (input_dir / "prediction_ranking.csv").write_text(_HEADER + _row("me", "ALL", 20.0), encoding="utf-8")
    (input_dir / "prediction_ranking_A.csv").write_text(_HEADER + _row("me", "A", 20.0), encoding="utf-8")
    (input_dir / "prediction_ranking_C.csv").write_text(_HEADER + _row("me", "C", 12.0), encoding="utf-8")

    page = create_highscore_page({"study_id": "me"}, "mg/dL", locale="en")
    assert "By data source" in " ".join(_texts(page))
    # One overall board plus one board per format that has a CSV; B has none.
    titles = _board_titles(page)
    assert titles == [
        "Overall",
        app_module._format_label("A", locale="en"),
        app_module._format_label("C", locale="en"),
    ]


def test_highscore_converts_mae_to_mmol(ranking_root: Path) -> None:
    (ranking_root / "data" / "input" / "prediction_ranking.csv").write_text(
        _HEADER + _row("me", "ALL", 18.0), encoding="utf-8"
    )
    page = create_highscore_page({"study_id": "me"}, "mmol/L", locale="en")
    joined = " ".join(_texts(page))
    assert "1.00" in joined  # 18 mg/dL == 1 mmol/L
    assert "MAE (mmol/L)" in joined
