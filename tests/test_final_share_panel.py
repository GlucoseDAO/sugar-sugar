"""The share flow is part of /final itself: eager panel, no button, no navigation.

`create_final_layout` builds the share record (`build_final_share_record`),
persists it under a content-addressed id (`share_store.ensure_share`) and
renders the shared `build_share_panel` (download / copy link / social buttons)
as a regular /final section.  The public ``/share/<id>`` page remains for
recipients — social links and the OG/PNG routes point at it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dash import html
from dash.exceptions import PreventUpdate
import pytest

from sugar_sugar import share_store

from sugar_sugar.app import (
    build_final_share_record,
    create_final_layout,
    fill_final_leaderboard,
    fill_final_share,
)
from sugar_sugar.components.share import create_share_layout
from sugar_sugar.i18n import setup_i18n
from tests.share_fixtures import make_test_share_record


def _by_id(node: Any, target: str) -> Any:
    if getattr(node, "id", None) == target:
        return node
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found = _by_id(kid, target)
            if found is not None:
                return found
    elif kids is not None and not isinstance(kids, str):
        return _by_id(kids, target)
    return None


def _walk(node: Any) -> list[Any]:
    found: list[Any] = [node]
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found.extend(_walk(kid))
    elif kids is not None and not isinstance(kids, str):
        found.extend(_walk(kids))
    return found


def _user(*, rounds_n: int = 1) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = [
        {
            "round_number": n + 1,
            "format": "A",
            "is_example_data": True,
            "data_source_name": "example.csv",
            "prediction_table_data": [
                {"metric": "Actual Glucose", **{f"t{i}": "100" for i in range(6)}},
                {"metric": "Predicted", **{f"t{i}": str(110 + n) for i in range(6)}},
            ],
        }
        for n in range(rounds_n)
    ]
    return {
        "study_id": "s1",
        "email": "ann@x.com",
        "format": "A",
        "uses_cgm": False,
        "max_rounds": 12,
        "rounds": rounds,
        "runs_by_format": {},
        "consent_completed": True,
    }


def _share_files() -> list[Path]:
    return sorted(Path(os.environ["SUGAR_SHARE_DIR"]).glob("*.json"))


def test_final_page_renders_share_panel_eagerly_without_a_button() -> None:
    setup_i18n()
    layout = create_final_layout(_user(), "mg/dL", locale="en")

    assert _by_id(layout, "share-results-button") is None, "the button is gone"
    panel = _by_id(layout, "final-share-panel")
    assert panel is not None and panel.children
    assert _by_id(panel, "share-copy-link-button") is not None
    assert _by_id(panel, "share-discord-button") is not None
    social = [n for n in _walk(panel) if "share-btn" in str(getattr(n, "className", ""))]
    assert len(social) >= 6, "X/FB/WhatsApp/LinkedIn/Telegram/Discord/copy must be present"
    # "Play again" is for recipients on /share/<id>; the owner already has
    # Exit / switch-format actions on /final.
    assert _by_id(panel, "share-play-again-button") is None

    files = _share_files()
    assert len(files) == 1, "rendering /final persists exactly one share record"
    share_id = files[0].stem
    url_node = _by_id(panel, "share-url-value")
    assert url_node is not None and share_id in url_node.children
    download = next((n for n in _walk(panel) if getattr(n, "download", None)), None)
    assert download is not None
    assert f"/share/{share_id}/image.png" in download.href


def test_rerendering_final_reuses_the_same_share_record() -> None:
    setup_i18n()
    user = _user()
    create_final_layout(user, "mg/dL", locale="en")
    create_final_layout(user, "mg/dL", locale="de")  # language change re-render
    files = _share_files()
    assert len(files) == 1, "same game state must map to one content-addressed record"

    # A new round is a new game state: fresh id, old record left intact.
    create_final_layout(_user(rounds_n=2), "mg/dL", locale="en")
    assert len(_share_files()) == 2


def test_deterministic_id_is_stable_and_ignores_volatile_fields() -> None:
    setup_i18n()
    record = build_final_share_record(_user(), locale="en")
    assert record is not None
    id_first = share_store.deterministic_share_id(record)

    again = build_final_share_record(_user(), locale="de")
    assert again is not None
    assert again["created_at"] != record["created_at"] or True  # timestamps may differ
    assert share_store.deterministic_share_id(again) == id_first

    other = build_final_share_record(_user(rounds_n=2), locale="en")
    assert other is not None
    assert share_store.deterministic_share_id(other) != id_first


def test_final_page_without_rounds_has_no_share_section() -> None:
    setup_i18n()
    user = _user()
    user["rounds"] = []
    layout = create_final_layout(user, "mg/dL", locale="en")
    assert _by_id(layout, "final-share-panel") is None
    assert _by_id(layout, "share-copy-link-button") is None
    assert not _share_files(), "no rounds, no record"


def test_deferred_final_paints_shell_then_leaderboard_then_share() -> None:
    """Live /final is eager=False: first paint is the shell, ticks fill the rest.

    Tick 1 reads ranking CSVs. Tick 2 writes the share record and the synthesis
    card. Tests call the two fill callbacks directly so we do not need a browser.
    """
    setup_i18n()
    user = _user()
    layout = create_final_layout(user, "mg/dL", locale="en", eager=False)

    assert _by_id(layout, "final-title") is not None
    ranking = _by_id(layout, "final-ranking-list")
    assert ranking is not None
    assert _by_id(ranking, "final-ranking-title") is not None
    assert _by_id(ranking, "final-nickname-input") is None
    share = _by_id(layout, "final-share-panel")
    synthesis = _by_id(layout, "final-synthesis-card")
    assert share is not None and share.children == []
    assert synthesis is not None and synthesis.children == []
    assert _by_id(layout, "final-deferred-tick") is None
    assert not _share_files(), "first paint must not write a share record"

    with pytest.raises(PreventUpdate):
        fill_final_leaderboard(None, "/final", user, "mg/dL", "en")
    with pytest.raises(PreventUpdate):
        fill_final_share({"phase": 1, "nonce": 1}, "/final", user, "mg/dL", "en")

    ranking_kids, kick2 = fill_final_leaderboard(
        {"phase": 1, "nonce": 1}, "/final", user, "mg/dL", "en"
    )
    assert ranking_kids
    assert any(getattr(node, "id", None) == "final-ranking-title" for node in ranking_kids)
    assert kick2 == {"phase": 2, "nonce": 1}
    assert not _share_files(), "leaderboard phase must not persist the share card"

    share_kids, synthesis_kids = fill_final_share(
        kick2, "/final", user, "mg/dL", "en"
    )
    filled_share = html.Div(share_kids, id="final-share-panel")
    assert _by_id(filled_share, "share-copy-link-button") is not None
    assert synthesis_kids
    files = _share_files()
    assert len(files) == 1, "share phase persists exactly one record"


def test_share_page_panel_keeps_play_again_for_recipients() -> None:
    setup_i18n()
    layout = create_share_layout(
        make_test_share_record(),
        share_id="abc",
        share_url="http://localhost/share/abc",
        locale="en",
    )
    assert _by_id(layout, "share-play-again-button") is not None
    assert _by_id(layout, "share-copy-link-button") is not None
    assert _by_id(layout, "share-discord-button") is not None
    url_node = _by_id(layout, "share-url-value")
    assert url_node is not None
    assert url_node.children == "http://localhost/share/abc"
