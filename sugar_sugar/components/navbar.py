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


def build_language_dropdown(locale: str) -> html.Div:
    """Fomantic 'simple dropdown' showing the active language flag as the trigger
    and all languages in a hover menu.  Each item keeps ``id="lang-{code}"`` so
    the existing ``set_interface_language`` callback works unchanged.  Shared by
    the desktop ``NavBar`` and the mobile ``MobileNavBar``."""
    current = next((lang for lang in LANGUAGES if lang[0] == locale), LANGUAGES[0])

    dropdown_items: list = []
    for code, flag_src, label in LANGUAGES:
        active_cls = " active" if code == locale else ""
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
            html.Img(src=current[1], className="lang-flag", style={"opacity": "1"}, disable_n_clicks=True),
            html.Span(f" {current[2]}", style={"marginLeft": "4px"}, disable_n_clicks=True),
            html.I(className="dropdown icon", disable_n_clicks=True),
            html.Div(dropdown_items, className="menu", disable_n_clicks=True),
        ],
        className="ui simple dropdown item lang-dropdown",
        disable_n_clicks=True,
    )


class MobileNavBar(html.Div):
    """Compact, portrait-first navbar: a burger button + title + language flag.

    The desktop ``NavBar`` is a single-row Fomantic ``massive tabular menu`` whose
    6 word-items are ~1280px wide.  Under a ``device-width`` viewport ANY element
    wider than the screen forces the browser to expand the layout viewport and
    zoom the page out -- so we cannot reuse it on mobile.  This bar is
    intrinsically narrow; the destinations live in a drawer (``mobile-nav-drawer``)
    that ``toggle_mobile_nav`` shows/hides.  Tapping a link navigates (dcc.Link),
    which re-renders the navbar fresh and so closes the drawer automatically.
    """

    LINKS: list[tuple[str, str, tuple[str, ...]]] = [
        ("ui.common.game", "/", tuple(_GAME_PAGES)),
        ("ui.common.the_study", "/about", ("/about",)),
        ("ui.common.faq", "/faq", ("/faq",)),
        ("ui.common.video_instructions", "/demo", ("/demo",)),
        ("ui.common.contact_us", "/contact", ("/contact",)),
    ]

    def __init__(self, *, locale: str = "en", current_page: str = "/") -> None:
        self._locale = locale
        self._current_page = current_page

        bar = html.Div(
            [
                html.Button(
                    "☰",  # burger glyph; avoids icon-font dependency
                    id="mobile-nav-toggle",
                    className="mobile-nav-burger",
                    n_clicks=0,
                    **{"aria-label": "Menu"},
                ),
                html.Div(
                    t("ui.common.app_title", locale=locale),
                    className="mobile-nav-title",
                    disable_n_clicks=True,
                ),
                html.Div(
                    build_language_dropdown(locale),
                    className="mobile-nav-lang",
                    disable_n_clicks=True,
                ),
            ],
            className="mobile-nav-bar",
            disable_n_clicks=True,
        )

        links = [
            dcc.Link(
                t(key, locale=locale),
                href=href,
                className=("mobile-nav-link active" if current_page in pages else "mobile-nav-link"),
            )
            for key, href, pages in self.LINKS
        ]
        drawer = html.Div(
            links,
            id="mobile-nav-drawer",
            className="mobile-nav-drawer",
            style={"display": "none"},
            disable_n_clicks=True,
        )

        super().__init__(
            children=[bar, drawer],
            className="mobile-navbar",
            disable_n_clicks=True,
        )


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
            dcc.Link(
                t("ui.common.game", locale=self._locale),
                href="/",
                className=self._active_cls(*_GAME_PAGES),
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

        right_menu = html.Div(
            [self._language_dropdown()],
            className="right menu",
            disable_n_clicks=True,
        )

        return left_items + [right_menu]

    def _language_dropdown(self) -> html.Div:
        return build_language_dropdown(self._locale)
