"""Share page title + brag stamp follow the player's nickname."""
from __future__ import annotations

import re
from typing import Any

from sugar_sugar.components.share import (
    _safe_display_name,
    _share_results_title,
    build_share_card_figure,
    create_share_layout,
)
from sugar_sugar.i18n import setup_i18n
from tests.share_fixtures import make_test_share_record


def _plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _walk(node: Any) -> list[Any]:
    found: list[Any] = [node]
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found.extend(_walk(kid))
    elif kids is not None and not isinstance(kids, str):
        found.extend(_walk(kids))
    return found


def test_anonymous_title_says_my() -> None:
    setup_i18n()
    assert _share_results_title("", locale="en") == "My Sugar-Sugar results"
    assert "Ninja" not in _share_results_title("", locale="en")


def test_named_title_uses_the_nickname_in_every_locale() -> None:
    setup_i18n()
    expected: dict[str, str] = {
        "en": "Ninja's Sugar-Sugar results",
        "de": "Sugar-Sugar-Ergebnisse von Ninja",
        "fr": "Résultats Sugar-Sugar de Ninja",
        "es": "Resultados Sugar-Sugar de Ninja",
        "ro": "Rezultatele Sugar-Sugar ale lui Ninja",
        "ru": "Результаты Ninja в Sugar-Sugar",
        "uk": "Результати Ninja у Sugar-Sugar",
        "zh": "Ninja 的 Sugar-Sugar 成绩",
    }
    for locale, title in expected.items():
        assert _share_results_title("Ninja", locale=locale) == title


def test_display_name_prefers_nickname_over_legacy_name() -> None:
    setup_i18n()
    assert _safe_display_name({"nickname": "Ninja", "name": "Legacy"}) == "Ninja"
    assert _safe_display_name({"name": "Dev Tester"}) == "Dev Tester"
    assert _safe_display_name({"name": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}) == ""
    assert _safe_display_name({}) == ""


def test_share_page_stamps_the_nickname() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["nickname"] = "SugarNinja"
    layout = create_share_layout(
        record, share_id="abc", share_url="http://localhost/share/abc", locale="en"
    )
    nodes = _walk(layout)
    stamp = next(
        (n for n in nodes if getattr(n, "className", None) == "share-name-stamp"),
        None,
    )
    assert stamp is not None
    assert stamp.children == "SugarNinja"
    texts = [n.children for n in nodes if isinstance(getattr(n, "children", None), str)]
    assert "SugarNinja's Sugar-Sugar results" in texts
    assert "My Sugar-Sugar results" not in texts


def test_anonymous_share_page_has_no_stamp() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["nickname"] = ""
    record["user_info"]["name"] = ""
    layout = create_share_layout(
        record, share_id="abc", share_url="http://localhost/share/abc", locale="en"
    )
    nodes = _walk(layout)
    assert not any(getattr(n, "className", None) == "share-name-stamp" for n in nodes)
    texts = [n.children for n in nodes if isinstance(getattr(n, "children", None), str)]
    assert "My Sugar-Sugar results" in texts


def test_share_card_png_figure_stamps_the_nickname() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["nickname"] = "SugarNinja"
    fig = build_share_card_figure(
        record,
        share_url="http://localhost/share/abc",
        locale="en",
        seed="abc",
    )
    annotations = list(fig.layout.annotations or [])
    stamped = [a for a in annotations if abs(float(a.textangle or 0)) > 1]
    assert stamped, "the kaleido card must rotate the nickname stamp into the PNG"
    stamp = stamped[0]
    assert any("SugarNinja" in str(a.text or "") for a in stamped)
    assert float(stamp.x) < 0.2
    assert float(stamp.y) < 0.2
    assert abs(float(stamp.textangle)) <= 12
    titles = _plain(" ".join(str(a.text or "") for a in annotations))
    assert "SugarNinja" in titles
    assert "My Sugar-Sugar results" not in titles


def test_share_page_stamps_challenge_the_unknown() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["challenge_unknown"] = True
    layout = create_share_layout(
        record, share_id="abc", share_url="http://localhost/share/abc", locale="en"
    )
    nodes = _walk(layout)
    stamp = next(
        (n for n in nodes if getattr(n, "className", None) == "share-unknown-stamp"),
        None,
    )
    assert stamp is not None
    texts = [c for c in (stamp.children or []) if isinstance(c, str)]
    assert " ".join(texts) == "went into the unknown"
    style = getattr(stamp, "style", None) or {}
    assert "#dc2626" in str(style.get("border", ""))
    assert style.get("color") == "#dc2626"


def test_share_page_hides_unknown_stamp_when_challenge_is_off() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["challenge_unknown"] = False
    layout = create_share_layout(
        record, share_id="abc", share_url="http://localhost/share/abc", locale="en"
    )
    nodes = _walk(layout)
    assert not any(getattr(n, "className", None) == "share-unknown-stamp" for n in nodes)


def test_share_card_png_figure_stamps_the_unknown() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["nickname"] = "SugarNinja"
    record["user_info"]["challenge_unknown"] = True
    fig = build_share_card_figure(
        record,
        share_url="http://localhost/share/abc",
        locale="en",
        seed="abc",
    )
    annotations = list(fig.layout.annotations or [])
    unknown = [a for a in annotations if "unknown" in str(a.text or "").lower()]
    assert unknown
    stamp = unknown[0]
    assert float(stamp.x) > 0.15
    assert abs(float(stamp.textangle or 0)) > 1
    assert "#dc2626" in str(stamp.font.color)
    assert "#dc2626" in str(stamp.bordercolor)


def test_anonymous_share_card_has_no_rotated_stamp() -> None:
    setup_i18n()
    record = make_test_share_record()
    record["user_info"]["nickname"] = ""
    record["user_info"]["name"] = ""
    fig = build_share_card_figure(
        record,
        share_url="http://localhost/share/abc",
        locale="en",
        seed="abc",
    )
    annotations = list(fig.layout.annotations or [])
    assert not any(abs(float(a.textangle or 0)) > 1 for a in annotations)
    titles = _plain(" ".join(str(a.text or "") for a in annotations))
    assert "My Sugar-Sugar results" in titles
