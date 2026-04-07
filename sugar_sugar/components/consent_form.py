from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from sugar_sugar.consent_notice_text import consent_notice_children
from sugar_sugar.i18n import t


class ConsentFormPage(html.Div):
    def __init__(self, *, locale: str = "en", theme: str = "light") -> None:
        self.component_id: str = "consent-form-page"
        self._locale: str = locale
        self._theme: str = theme

        background = (
            "linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 35%, #3a3a3a 100%)"
            if theme == "dark"
            else "linear-gradient(135deg, #eff6ff 0%, #f8fafc 35%, #fff7ed 100%)"
        )

        border_color = "rgba(255, 255, 255, 0.2)" if theme == "dark" else "rgba(15, 23, 42, 0.10)"
        card_bg = "transparent"

        super().__init__(
            id=self.component_id,
            children=[
                html.Div(
                    id="consent-form-background",
                    style={
                        "height": "100vh",
                        "overflow": "hidden",
                        "padding": "28px 18px",
                        "background": background,
                    },
                    children=dbc.Container(
                        fluid=False,
                        style={"maxWidth": "900px"},
                        children=[
                            dbc.Card(
                                id="consent-form-card",
                                className="bg-transparent",
                                style={"borderRadius": "14px", "border": f"1px solid {border_color}", "backgroundColor": card_bg},
                                children=dbc.CardBody(
                                    className="bg-transparent",
                                    style={"backgroundColor": "transparent"},
                                    children=[
                                        html.Div(
                                            consent_notice_children(locale, theme),
                                            id="consent-form-scroll",
                                            style={"background": "transparent"},
                                            disable_n_clicks=True,
                                        ),
                                        html.Hr(style={"margin": "18px 0"}),
                                        html.A(
                                            t("ui.common.go_to_start", locale=locale),
                                            href="/",
                                            className="ui basic secondary button",
                                            style={"fontWeight": "800", "marginBottom": "14px"},
                                        ),
                                    ]
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

