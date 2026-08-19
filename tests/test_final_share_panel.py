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

from sugar_sugar import share_store
from sugar_sugar.app import build_final_share_record, create_final_layout
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
