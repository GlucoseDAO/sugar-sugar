"""The `/final` rename box and the suggestion it offers.

A player returning on a new device has an empty localStorage but the same email, so
the box looks their previous nickname up by `email_key` and pre-types it. It is only
a *suggestion*: saving stamps the current study's rows, never the older ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from sugar_sugar import app as app_module
from sugar_sugar import nickname as nickname_module
from sugar_sugar.app import (
    _nickname_editor_children,
    create_final_layout,
    create_highscore_page,
    stored_nickname,
)

_HEADER = (
    "study_id,run_id,number,timestamp,email_key,nickname,format,rounds_played,"
    "is_example_data,data_source_name,overall_mae_mgdl,overall_mse_mgdl,"
    "overall_rmse_mgdl,overall_mape_pct\n"
)


def _row(
    study_id: str,
    *,
    fmt: str = "ALL",
    key: str = "",
    nickname: str = "",
    ts: str = "2026-08-01 10:00:00",
) -> str:
    return (
        f"{study_id},run1,1,{ts},{key},{nickname},{fmt},12,True,example,18.0,0,0,0\n"
    )


@pytest.fixture()
def ranking_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the app's `data/input` ranking lookups at a throwaway tree."""
    (tmp_path / "data" / "input").mkdir(parents=True)
    monkeypatch.setattr(app_module, "project_root", tmp_path)
    monkeypatch.setattr(nickname_module, "RANKING_EMAIL_SALT", "test-salt")
    nickname_module._salt.cache_clear()
    yield tmp_path
    nickname_module._salt.cache_clear()


def _input_value(children: list[Any]) -> str:
    """The value pre-typed into `final-nickname-input`."""
    def walk(node: Any) -> Any:
        if getattr(node, "id", None) == "final-nickname-input":
            return node
        kids = getattr(node, "children", None)
        if isinstance(kids, (list, tuple)):
            for kid in kids:
                found = walk(kid)
                if found is not None:
                    return found
        elif kids is not None and not isinstance(kids, str):
            return walk(kids)
        return None

    for child in children:
        found = walk(child)
        if found is not None:
            return str(found.value)
    raise AssertionError("final-nickname-input not rendered")


def _overall(root: Path) -> Path:
    return root / "data" / "input" / "prediction_ranking.csv"


# --- stored_nickname ---------------------------------------------------------


def test_previous_nickname_is_found_by_email_across_sessions(ranking_root: Path) -> None:
    key = nickname_module.email_key("ann@x.com")
    _overall(ranking_root).write_text(_HEADER + _row("s1", key=key, nickname="Ninja"), encoding="utf-8")
    # New device -> brand-new study_id, same email.
    assert stored_nickname(study_id="s99", key=key) == "Ninja"


def test_newest_nickname_is_the_one_suggested(ranking_root: Path) -> None:
    key = nickname_module.email_key("ann@x.com")
    _overall(ranking_root).write_text(
        _HEADER
        + _row("s1", key=key, nickname="Old", ts="2026-08-01 10:00:00")
        + _row("s7", key=key, nickname="Newer", ts="2026-08-05 10:00:00"),
        encoding="utf-8",
    )
    assert stored_nickname(study_id="s99", key=key) == "Newer"


def test_per_format_csv_is_consulted_when_the_overall_one_has_nothing(
    ranking_root: Path,
) -> None:
    key = nickname_module.email_key("ann@x.com")
    (ranking_root / "data" / "input" / "prediction_ranking_A.csv").write_text(
        _HEADER + _row("s1", fmt="A", key=key, nickname="Ninja"), encoding="utf-8"
    )
    assert stored_nickname(study_id="s1", key=key) == "Ninja"


def test_no_email_only_finds_your_own_study(ranking_root: Path) -> None:
    _overall(ranking_root).write_text(
        _HEADER + _row("s1", nickname="Mine") + _row("s2", nickname="Theirs"),
        encoding="utf-8",
    )
    assert stored_nickname(study_id="s1", key="") == "Mine"
    assert stored_nickname(study_id="s404", key="") == ""


def test_nothing_stored_yields_empty(ranking_root: Path) -> None:
    assert stored_nickname(study_id="s1", key="") == ""
    assert stored_nickname(study_id="", key="") == ""


# --- editor prefill precedence ----------------------------------------------


def test_own_nickname_wins_over_the_stored_one(ranking_root: Path) -> None:
    key = nickname_module.email_key("ann@x.com")
    _overall(ranking_root).write_text(_HEADER + _row("s1", key=key, nickname="Stored"), encoding="utf-8")
    children = _nickname_editor_children(
        {"study_id": "s1", "email": "ann@x.com", "nickname": "Typed"}, locale="en"
    )
    assert _input_value(children) == "Typed"


def test_stored_nickname_prefills_a_fresh_device(ranking_root: Path) -> None:
    key = nickname_module.email_key("ann@x.com")
    _overall(ranking_root).write_text(_HEADER + _row("s1", key=key, nickname="Ninja"), encoding="utf-8")
    children = _nickname_editor_children(
        {"study_id": "s99", "email": "ann@x.com"}, locale="en"
    )
    assert _input_value(children) == "Ninja"


def test_box_is_empty_for_a_player_with_no_history(ranking_root: Path) -> None:
    children = _nickname_editor_children({"study_id": "s1"}, locale="en")
    assert _input_value(children) == ""


def test_editor_renders_without_any_session(ranking_root: Path) -> None:
    assert _input_value(_nickname_editor_children(None, locale="en")) == ""


def _ids(node: Any) -> set[str]:
    found: set[str] = set()
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        found.add(node_id)
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found |= _ids(kid)
    elif kids is not None and not isinstance(kids, str):
        found |= _ids(kids)
    return found


def _finished_user(study_id: str = "s1") -> dict[str, Any]:
    return {
        "study_id": study_id,
        "email": "ann@x.com",
        "format": "A",
        "max_rounds": 12,
        "rounds": [
            {
                "format": "A",
                "is_example_data": True,
                "data_source_name": "example.csv",
                "prediction_table_data": [
                    {"metric": "Actual Glucose", **{f"t{i}": "100" for i in range(6)}},
                    {"metric": "Predicted", **{f"t{i}": "110" for i in range(6)}},
                ],
            }
        ],
    }


def test_final_page_renders_the_editor_ids(ranking_root: Path) -> None:
    """All three `final-nickname-*` ids must be in the DOM or the save callback
    silently never fires."""
    key = nickname_module.email_key("ann@x.com")
    _overall(ranking_root).write_text(_HEADER + _row("s1", key=key), encoding="utf-8")

    ids = _ids(create_final_layout(_finished_user(), "mg/dL", locale="en"))
    assert {"final-nickname-input", "final-nickname-save", "final-nickname-status"} <= ids
    # The wrapper the save callback swaps must be there too.
    assert "final-ranking-list" in ids


def test_highscore_page_has_no_editor(ranking_root: Path) -> None:
    """`/highscore` is public and session-free -- a rename box there is meaningless."""
    _overall(ranking_root).write_text(_HEADER + _row("s1"), encoding="utf-8")
    ids = _ids(create_highscore_page({"study_id": "s1"}, "mg/dL", locale="en"))
    assert not any(component_id.startswith("final-nickname") for component_id in ids)


def test_no_editor_when_the_player_has_no_board_presence(ranking_root: Path) -> None:
    """"Play only" participants are never written to the ranking CSVs."""
    ids = _ids(create_final_layout(_finished_user(), "mg/dL", locale="en"))
    assert "final-nickname-input" not in ids


def test_editor_states_the_nickname_is_optional_and_public(ranking_root: Path) -> None:
    children = _nickname_editor_children({"study_id": "s1"}, locale="en")

    def texts(node: Any) -> list[str]:
        if isinstance(node, str):
            return [node]
        kids = getattr(node, "children", None)
        if isinstance(kids, str):
            return [kids]
        if isinstance(kids, (list, tuple)):
            return [text for kid in kids for text in texts(kid)]
        if kids is not None:
            return texts(kids)
        return []

    joined = " ".join(text for child in children for text in texts(child))
    assert "Optional and public" in joined
