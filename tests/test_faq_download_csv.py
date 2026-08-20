from __future__ import annotations

from typing import Any

import pytest

from sugar_sugar.app import create_faq_page
from sugar_sugar.components.startup import format_help_children
from sugar_sugar.i18n import SUPPORTED_LOCALES, setup_i18n, t_raw


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


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


@pytest.mark.parametrize("locale", list(SUPPORTED_LOCALES))
def test_faq_has_download_csv_section(locale: str) -> None:
    sections = t_raw("ui.faq.sections", locale=locale)
    ids = [str(section.get("id") or "") for section in sections]
    assert "download-csv" in ids
    layout = create_faq_page(locale=locale)
    assert _by_id(layout, "download-csv") is not None


def test_faq_ask_form_is_hidden_by_default() -> None:
    layout = create_faq_page(locale="en")
    assert _by_id(layout, "faq-ask-form") is None
    assert _by_id(layout, "faq-board") is None


def test_format_help_links_to_faq_download_section() -> None:
    children = format_help_children("en")
    link = children[-1].children[1]
    assert link.href == "/faq#download-csv"
    assert link.children == "search here."
