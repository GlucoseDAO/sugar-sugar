from __future__ import annotations

from sugar_sugar.components.navbar import LANGUAGES, MobileNavBar, NavBar
from sugar_sugar.i18n import t


def _get_dropdown_items(navbar: NavBar) -> list:
    """Extract language items from the dropdown menu inside the navbar."""
    right_menu = navbar.children[-1]
    dropdown = right_menu.children[0]
    menu_div = dropdown.children[3]
    return menu_div.children


def test_navbar_fomantic_menu_structure():
    """NavBar renders a Fomantic UI massive blue inverted tabular menu with correct items."""
    navbar = NavBar(locale="en", current_page="/prediction")
    assert "ui massive blue inverted tabular menu" in navbar.className

    children = navbar.children
    # 6 left links + 1 right menu div = 7
    assert len(children) == 7
    game_item, highscore_item, study_item, faq_item, video_item, contact_item, right_menu = children

    assert game_item.href == "/"
    assert t("ui.common.game", locale="en") == game_item.children

    assert highscore_item.href == "/highscore"
    assert t("ui.common.highscore", locale="en") == highscore_item.children

    assert study_item.href == "/about"
    assert t("ui.common.the_study", locale="en") == study_item.children

    assert faq_item.href == "/faq"

    assert video_item.href == "/demo"
    assert t("ui.common.video_instructions", locale="en") == video_item.children

    assert contact_item.href == "/contact"
    assert t("ui.common.contact_us", locale="en") == contact_item.children

    # right menu contains the language dropdown
    assert "right menu" in right_menu.className
    assert len(right_menu.children) == 1
    assert "dropdown" in right_menu.children[0].className


def test_navbar_game_always_visible():
    """Game item is always shown, including on landing page."""
    navbar = NavBar(locale="en", current_page="/")
    assert len(navbar.children) == 7
    game_item = navbar.children[0]
    assert game_item.href == "/"
    assert "active" in game_item.className


def test_navbar_game_active_on_game_flow_pages():
    """Game tab is active on all game-flow pages."""
    for page in ("/", "/consent-form", "/startup", "/prediction", "/ending", "/final"):
        navbar = NavBar(locale="en", current_page=page)
        game_item = navbar.children[0]
        assert "active" in game_item.className, f"Game not active on {page}"


def test_navbar_active_page_highlighted():
    navbar = NavBar(locale="en", current_page="/about")
    study_item = navbar.children[2]
    assert "active" in study_item.className

    game_item = navbar.children[0]
    assert "active" not in game_item.className


def test_navbar_highscore_active_and_not_a_game_page():
    """The Highscore tab lights up on /highscore, and Game stays inactive there."""
    navbar = NavBar(locale="en", current_page="/highscore")
    game_item, highscore_item = navbar.children[0], navbar.children[1]
    assert "active" in highscore_item.className
    assert "active" not in game_item.className


def test_mobile_navbar_contains_highscore_link():
    """The burger drawer exposes the same /highscore destination as the desktop bar."""
    navbar = MobileNavBar(locale="en", current_page="/highscore")
    drawer = navbar.children[1]
    links = {link.href: link for link in drawer.children}
    assert "/highscore" in links
    assert t("ui.common.highscore", locale="en") == links["/highscore"].children
    assert "active" in links["/highscore"].className
    assert "active" not in links["/"].className


def test_navbar_active_language_in_dropdown():
    """Active language is marked with 'active' class inside the dropdown."""
    navbar = NavBar(locale="de", current_page="/")
    items = _get_dropdown_items(navbar)
    de_item = items[1]  # en=0, de=1
    assert "active" in de_item.className
    en_item = items[0]
    assert "active" not in en_item.className


def test_navbar_dropdown_trigger_shows_active_language():
    """Dropdown trigger displays the flag and label of the active language."""
    navbar = NavBar(locale="fr", current_page="/")
    right_menu = navbar.children[-1]
    dropdown = right_menu.children[0]
    trigger_img = dropdown.children[0]
    trigger_label = dropdown.children[1]
    assert "/assets/flags/fr.svg" in trigger_img.src
    assert "FR" in trigger_label.children


def test_navbar_dropdown_contains_all_languages():
    """Dropdown menu lists every supported language."""
    navbar = NavBar(locale="en", current_page="/")
    items = _get_dropdown_items(navbar)
    assert len(items) == len(LANGUAGES)
    ids = {item.id for item in items}
    expected_ids = {f"lang-{code}" for code, _, _ in LANGUAGES}
    assert ids == expected_ids
