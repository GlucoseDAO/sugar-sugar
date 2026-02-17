from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from sugar_sugar.i18n import t, t_list


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
                                        html.H1(
                                            t("ui.consent_form.title", locale=locale),
                                            style={"fontSize": "34px", "fontWeight": "800", "color": "#0f172a"},
                                        ),
                                        html.Div(
                                            t("ui.consent_form.subtitle", locale=locale),
                                            style={"color": "#334155", "lineHeight": "1.6", "marginBottom": "12px"},
                                        ),
                                        dbc.Alert(
                                            t("ui.consent_form.adults_only", locale=locale),
                                            color="warning",
                                            style={"marginBottom": "14px"},
                                        ),
                                        html.H3(
                                            t("ui.consent_form.summary_title", locale=locale),
                                            style={"fontSize": "20px", "fontWeight": "800", "color": "#0f172a"},
                                        ),
                                        html.Ul(
                                            [html.Li(item) for item in t_list("ui.consent_form.summary_bullets", locale=locale)],
                                            style={"color": "#334155", "lineHeight": "1.6"},
                                        ),
                                        html.H3(
                                            t("ui.consent_form.data_title", locale=locale),
                                            style={"fontSize": "20px", "fontWeight": "800", "color": "#0f172a"},
                                        ),
                                        html.Ul(
                                            [html.Li(item) for item in t_list("ui.consent_form.data_bullets", locale=locale)],
                                            style={"color": "#334155", "lineHeight": "1.6"},
                                        ),
                                        html.H3(
                                            t("ui.consent_form.withdraw_title", locale=locale),
                                            style={"fontSize": "20px", "fontWeight": "800", "color": "#0f172a"},
                                        ),
                                        html.Div(
                                            t("ui.consent_form.withdraw_text", locale=locale),
                                            style={"color": "#334155", "lineHeight": "1.6"},
                                        ),
                                        html.Hr(style={"margin": "18px 0"}),
                                        dbc.Button(
                                            t("ui.common.back", locale=locale),
                                            href="/",
                                            color="secondary",
                                            outline=True,
                                            style={"fontWeight": "800", "marginBottom": "14px"},
                                        ),
                                        html.Div(
                                            t("ui.consent_form.footer_note", locale=locale),
                                            style={"color": "#64748b", "fontSize": "14px"},
                                        ),
                                    ]
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

