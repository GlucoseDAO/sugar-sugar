from __future__ import annotations

from dash import html

from sugar_sugar.consent_notice_text import consent_notice_children
from sugar_sugar.i18n import t


class ConsentFormPage(html.Div):
    """Standalone, read-only consent reader (``/consent-form``).

    Laid out like a mobile TOS/EULA dialog: a single full-width box that holds
    the scrollable consent text and fills the screen, with the back button
    anchored directly below it.  No nested card -- the consent iframe owns the
    only scrollbar (CLAUDE.md single-scrollbar rule), so we must NOT wrap it in
    an ``overflowY: auto`` container.  The shell is capped at 900px and centred
    on desktop; on a phone it is naturally full width.
    """

    def __init__(self, *, locale: str = "en") -> None:
        self.component_id: str = "consent-form-page"
        self._locale: str = locale

        super().__init__(
            id=self.component_id,
            children=[
                html.Div(
                    className="consent-form-shell",
                    style={
                        "boxSizing": "border-box",
                        "padding": "20px",
                        "maxWidth": "900px",
                        "margin": "0 auto",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "14px",
                        "background": "linear-gradient(135deg, #eff6ff 0%, #f8fafc 35%, #fff7ed 100%)",
                    },
                    children=[
                        html.Div(
                            # The iframe height reserves room for the navbar, the
                            # back button and paddings so the button stays on-screen
                            # without a second (page-level) scrollbar -- the iframe
                            # owns the only scrollbar (CLAUDE.md single-scrollbar).
                            consent_notice_children(
                                locale, iframe_style={"height": "calc(100vh - 190px)"}
                            ),
                            id="consent-form-scroll",
                            className="consent-form-text",
                            style={
                                "border": "1px solid rgba(15, 23, 42, 0.10)",
                                "borderRadius": "12px",
                                "overflow": "hidden",
                                "background": "white",
                            },
                            disable_n_clicks=True,
                        ),
                        html.A(
                            t("ui.common.go_to_start", locale=locale),
                            href="/",
                            className="ui basic secondary button consent-form-back",
                            style={"fontWeight": "800", "flex": "0 0 auto", "textAlign": "center"},
                        ),
                    ],
                )
            ],
        )

