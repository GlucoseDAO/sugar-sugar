from __future__ import annotations

from pathlib import Path

import pytest

from sugar_sugar.app import create_faq_page
from sugar_sugar.faq_board import add_faq_question, add_faq_reply, load_faq_questions
from sugar_sugar.i18n import setup_i18n


@pytest.fixture(autouse=True)
def _faq_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SUGAR_FAQ_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


def test_add_question_and_reply(tmp_path: Path) -> None:
    posted = add_faq_question(
        text="Where do I export Libre CSV?",
        section="participant",
        tags=["download", "data"],
        name="Just",
    )
    assert posted is not None
    assert posted["section"] == "participant"
    assert posted["tags"] == ["download", "data"]
    reply = add_faq_reply(posted["id"], text="Use LibreView on the website.", section="developer")
    assert reply is not None
    items = load_faq_questions()
    assert len(items) == 1
    assert items[0]["replies"][0]["text"].startswith("Use LibreView")


def test_empty_question_is_rejected() -> None:
    assert add_faq_question(text="   ", section="participant") is None


def test_faq_page_has_ask_form() -> None:
    layout = create_faq_page(locale="en")
    ids = _collect_ids(layout)
    assert "faq-ask-text" in ids
    assert "faq-ask-submit" in ids
    assert "faq-ask-tags" in ids
    assert "faq-ask-section" in ids
    assert "faq-board" in ids


def _collect_ids(node: object) -> set[str]:
    found: set[str] = set()
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        found.add(node_id)
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found.update(_collect_ids(kid))
    elif kids is not None and not isinstance(kids, str):
        found.update(_collect_ids(kids))
    return found
