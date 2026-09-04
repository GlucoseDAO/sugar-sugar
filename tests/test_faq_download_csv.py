from __future__ import annotations

from typing import Any

import pytest

from pathlib import Path

from sugar_sugar.app import create_faq_page
from sugar_sugar.components.startup import SENSOR_EXPORT_SITES, format_help_children
from sugar_sugar.i18n import SUPPORTED_LOCALES, setup_i18n, t, t_raw

DEXCOM_EXPORT_SCREENSHOT = "/assets/images/dexcom_instruction.jpg"
DEXCOM_EXPORT_SCREENSHOT_FILE = (
    Path(__file__).resolve().parent.parent / "assets" / "images" / "dexcom_instruction.jpg"
)


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


def _walk_nodes(node: Any) -> list[Any]:
    if isinstance(node, (list, tuple)):
        found: list[Any] = []
        for kid in node:
            found.extend(_walk_nodes(kid))
        return found
    found = [node]
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found.extend(_walk_nodes(kid))
    elif kids is not None and not isinstance(kids, str):
        found.extend(_walk_nodes(kids))
    return found


def _collect_hrefs(node: Any) -> list[str]:
    hrefs: list[str] = []
    for item in _walk_nodes(node):
        href = getattr(item, "href", None)
        if isinstance(href, str):
            hrefs.append(href)
    return hrefs


def _collect_link_labels(node: Any) -> list[str]:
    labels: list[str] = []
    for item in _walk_nodes(node):
        if getattr(item, "href", None) and isinstance(getattr(item, "children", None), str):
            labels.append(item.children)
    return labels


def test_format_help_lists_export_sites_and_faq() -> None:
    children = format_help_children("en")
    hrefs = _collect_hrefs(children)
    expected = [href for _key, href, _host in SENSOR_EXPORT_SITES]
    assert hrefs[: len(expected)] == expected
    assert hrefs[-1] == "/faq#download-csv"
    assert _collect_link_labels(children)[-1] == "FAQ"
    others = t("ui.startup.format_find_others", locale="en")
    assert "search" in others
    assert "FAQ" not in others


@pytest.mark.parametrize("locale", list(SUPPORTED_LOCALES))
def test_format_help_export_sites_are_translated(locale: str) -> None:
    children = format_help_children(locale)
    hrefs = _collect_hrefs(children)
    for label_key, href, host in SENSOR_EXPORT_SITES:
        assert href in hrefs
        assert host in _collect_link_labels(children)
        label = t(label_key, locale=locale)
        assert not label.startswith("ui.")
        assert label
    assert "/faq#download-csv" in hrefs
    faq_label = t("ui.startup.format_find_faq_link", locale=locale)
    assert not faq_label.startswith("ui.")
    others = t("ui.startup.format_find_others", locale=locale)
    assert others
    assert not others.startswith("ui.")


def _dexcom_answer(locale: str) -> str:
    sections = t_raw("ui.faq.sections", locale=locale)
    for section in sections:
        if section.get("id") != "download-csv":
            continue
        for item in section.get("items", []):
            question = str(item.get("q") or "")
            if "Dexcom" in question:
                return str(item.get("a") or "")
    raise AssertionError(f"Dexcom FAQ item missing for locale {locale}")


def test_dexcom_export_screenshot_is_shipped() -> None:
    assert DEXCOM_EXPORT_SCREENSHOT_FILE.is_file()
    assert DEXCOM_EXPORT_SCREENSHOT_FILE.stat().st_size > 0


@pytest.mark.parametrize("locale", list(SUPPORTED_LOCALES))
def test_dexcom_faq_shows_top_right_export_icon(locale: str) -> None:
    answer = _dexcom_answer(locale)
    assert DEXCOM_EXPORT_SCREENSHOT in answer
    assert "![ " not in answer
    assert answer.count(DEXCOM_EXPORT_SCREENSHOT) == 1


def test_dexcom_faq_english_explains_the_top_right_icon() -> None:
    answer = _dexcom_answer("en")
    assert "export icon in the top-right corner" in answer
    assert "There is no Export button in the top menu" in answer
    assert "click **Export** in the top bar" not in answer
