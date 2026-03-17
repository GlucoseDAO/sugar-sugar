from __future__ import annotations

from sugar_sugar.components.navbar import NavBar, get_navbar_back_href
from sugar_sugar.i18n import t


def test_navbar_contains_home_button_and_back(monkeypatch):
    # use a known page to verify back mapping
    current = "/prediction"
    navbar = NavBar(locale="en", current_page=current)
    children = navbar.children
    # should have four elements: back, home, title, spacer
    assert len(children) >= 4

    back_btn, home_btn = children[:2]
    # verify back button properties
    assert back_btn.href == get_navbar_back_href(current)
    assert "Back" in back_btn.children or t("ui.common.back", locale="en") in back_btn.children

    # verify home button points to root with correct label
    assert home_btn.href == "/"
    assert home_btn.children == t("ui.common.home", locale="en")
