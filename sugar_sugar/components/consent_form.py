from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from sugar_sugar.i18n import t, t_list
from sugar_sugar.components.consent_info import get_consent_info_components


class ConsentFormPage(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self.component_id: str = "consent-form-page"
        self._locale: str = locale

        super().__init__(
            id=self.component_id,
            children=[
                html.Div(
                    style={
                        "minHeight": "100vh",
                        "padding": "28px 18px",
                        "background": "linear-gradient(135deg, #eff6ff 0%, #f8fafc 35%, #fff7ed 100%)",
                    },
                    children=dbc.Container(
                        fluid=False,
                        style={"maxWidth": "900px"},
                        children=[
                            dbc.Card(
                                style={"borderRadius": "14px", "border": "1px solid rgba(15, 23, 42, 0.10)"},
                                children=dbc.CardBody(
                                    [
                                        html.Div(
                                            t("ui.consent_form.subtitle", locale=locale),
                                            style={"color": "#334155", "lineHeight": "1.6", "marginBottom": "12px"},
                                        ),
                                        dbc.Alert(
                                            t("ui.consent_form.adults_only", locale=locale),
                                            color="warning",
                                            style={"marginBottom": "14px"},
                                        ),
                                        html.Div(get_consent_info_components(locale)),
                                        html.Hr(style={"margin": "18px 0"}),
                                        html.A(
                                            t("ui.common.back", locale=locale),
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

