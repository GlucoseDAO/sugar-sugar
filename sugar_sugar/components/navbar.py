from dash import dcc, html
from sugar_sugar.i18n import t


_GAME_PAGES = frozenset({"/", "/consent-form", "/startup", "/prediction", "/ending", "/final"})

LANGUAGES: list[tuple[str, str, str]] = [
    ("en", "/assets/flags/gb.svg", "EN"),
    ("de", "/assets/flags/de.svg", "DE"),
    ("uk", "/assets/flags/ua.svg", "UA"),
    ("ro", "/assets/flags/ro.svg", "RO"),
    ("ru", "/assets/flags/ru.svg", "RU"),
    ("zh", "/assets/flags/cn.svg", "ZH"),
    ("fr", "/assets/flags/fr.svg", "FR"),
    ("es", "/assets/flags/es.svg", "ES"),
]


class NavBar(html.Div):
    """Fomantic UI massive blue inverted tabular menu navbar.

    Left items:  Game | The Study | Video instructions | Contact us
    Right items: language dropdown (active flag + dropdown with all languages)

    Uses ``dcc.Link`` for navigation so page switches happen via client-side
    routing (``pushState``) without a full page reload.  This preserves all
    in-memory and localStorage-backed ``dcc.Store`` values and avoids the
    hydration race that previously caused users to fall back to the landing
    page when returning to the Game tab.
    """

    def __init__(self, *, locale: str = "en", current_page: str = "/") -> None:
        self._locale: str = locale
        self._current_page: str = current_page

        super().__init__(
            children=self._create_navbar(),
            className="ui massive blue inverted tabular menu",
            style={"borderRadius": "0", "marginBottom": "0", "borderBottom": "none"},
            disable_n_clicks=True,
        )

    def _active_cls(self, *pages: str) -> str:
        """Return 'active item' if current page matches, else 'item'."""
        return "active item" if self._current_page in pages else "item"

    def _create_navbar(self) -> list:
        left_items: list = [
            html.A(
                t("ui.common.game", locale=self._locale),
                id="home-button",
                className=self._active_cls(*_GAME_PAGES),
                style={"color": "inherit", "cursor": "pointer"},
            ),
            dcc.Link(
                t("ui.common.the_study", locale=self._locale),
                href="/about",
                className=self._active_cls("/about"),
            ),
            dcc.Link(
                t("ui.common.faq", locale=self._locale),
                href="/faq",
                className=self._active_cls("/faq"),
            ),
            dcc.Link(
                t("ui.common.video_instructions", locale=self._locale),
                href="/demo",
                className=self._active_cls("/demo"),
            ),
            dcc.Link(
                t("ui.common.contact_us", locale=self._locale),
                href="/contact",
                className=self._active_cls("/contact"),
            ),
        ]

        dark_mode_toggle = html.Div(
            [
                html.I(className="moon icon", id="dark-mode-icon", style={"cursor": "pointer", "marginRight": "8px"}, disable_n_clicks=True),
                # A hidden input just to give dash something to hook on, or we can use the Div itself.
                # Actually, dash doesn't trigger callbacks on html.I directly if we don't have n_clicks allowed.
                # So we let this div track n_clicks.
            ],
            className="item",
            id="dark-mode-toggle",
            style={"cursor": "pointer", "padding": "0 10px", "display": "flex", "alignItems": "center"},
            n_clicks=0,
            disable_n_clicks=False,
        )

        right_menu = html.Div(
            [dark_mode_toggle, self._language_dropdown()],
            className="right menu",
            disable_n_clicks=True,
        )

        return left_items + [right_menu]

    def _language_dropdown(self) -> html.Div:
        """Fomantic 'simple dropdown' showing the active language flag as the
        trigger and all languages in a hover menu.  Each dropdown item keeps
        its ``id="lang-{code}"`` so the existing ``set_interface_language``
        callback works unchanged."""
        current = next((lang for lang in LANGUAGES if lang[0] == self._locale), LANGUAGES[0])

        dropdown_items: list = []
        for code, flag_src, label in LANGUAGES:
            active_cls = " active" if code == self._locale else ""
            dropdown_items.append(
                html.A(
                    [
                        html.Img(src=flag_src, className="lang-flag", disable_n_clicks=True),
                        html.Span(f" {label}", style={"marginLeft": "6px"}, disable_n_clicks=True),
                    ],
                    id=f"lang-{code}",
                    className=f"item lang-dropdown-item{active_cls}",
                    style={"cursor": "pointer"},
                )
            )

        return html.Div(
            [
                html.Img(
                    src=current[1],
                    className="lang-flag",
                    style={"opacity": "1"},
                    disable_n_clicks=True,
                ),
                html.Span(
                    f" {current[2]}",
                    style={"marginLeft": "4px", "color": "inherit"},
                    disable_n_clicks=True,
                ),
                html.I(className="dropdown icon", style={"color": "inherit"}, disable_n_clicks=True),
                html.Div(dropdown_items, className="menu", disable_n_clicks=True),
            ],
            className="ui simple dropdown item lang-dropdown",
            style={"color": "inherit"},
            disable_n_clicks=True,
        )
