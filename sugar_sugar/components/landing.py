from __future__ import annotations

import base64
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State

from sugar_sugar.consent_notice_text import consent_notice_children
from sugar_sugar.consent import ensure_consent_agreement_row, get_next_study_number
from sugar_sugar.i18n import t, t_list
from sugar_sugar.config import STORAGE_TYPE


@lru_cache(maxsize=4)
def _image_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "png" if suffix == "png" else suffix
    return f"data:image/{mime};base64,{b64}"


class LandingPage(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self.component_id: str = "landing-page"
        self._locale: str = locale

        project_root = Path(__file__).resolve().parents[2]
        screenshot_path = project_root / "images" / "screenshot.png"
        screenshot_src = _image_data_uri(screenshot_path)

        hero = dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            t("ui.common.app_title", locale=locale),
                            style={
                                "fontSize": "56px",
                                "fontWeight": "800",
                                "letterSpacing": "-0.02em",
                                "marginBottom": "10px",
                            },
                        ),
                        html.Div(
                            t("ui.landing.tagline", locale=locale),
                            style={
                                "fontSize": "20px",
                                "color": "#334155",
                                "marginBottom": "18px",
                                "lineHeight": "1.4",
                            },
                        ),
                        html.Div(
                            [
                                html.H3(
                                    t("ui.landing.how_it_works", locale=locale),
                                    style={"fontSize": "22px", "fontWeight": "800", "color": "#1565c0", "marginBottom": "8px"},
                                ),
                                html.Ul(
                                    [html.Li(item) for item in t_list("ui.landing.how_it_works_steps", locale=locale)],
                                    style={
                                        "marginBottom": "0",
                                        "color": "#334155",
                                        "lineHeight": "1.6",
                                    },
                                ),
                            ],
                            style={
                                "background": "rgba(255,255,255,0.75)",
                                "border": "1px solid rgba(15, 23, 42, 0.10)",
                                "borderRadius": "14px",
                                "padding": "14px 16px",
                                "backdropFilter": "blur(8px)",
                            },
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=screenshot_src,
                                    style={
                                        "width": "100%",
                                        "borderRadius": "14px",
                                        "border": "1px solid rgba(15, 23, 42, 0.10)",
                                    },
                                )
                                if screenshot_src
                                else html.Div(
                                    [
                                        html.Div(
                                            className="fa-solid fa-chart-line",
                                            style={"fontSize": "56px", "color": "#1d4ed8"},
                                        ),
                                        html.Div(
                                            t("ui.landing.preview", locale=locale),
                                            style={"fontWeight": "700", "marginTop": "10px"},
                                        ),
                                    ],
                                    style={
                                        "height": "320px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "background": "rgba(255,255,255,0.75)",
                                        "border": "1px solid rgba(15, 23, 42, 0.10)",
                                        "borderRadius": "14px",
                                    },
                                )
                            ],
                        ),
                    ],
                    md=6,
                ),
            ],
            className="g-4",
            style={"alignItems": "center"},
        )

        study_info = dbc.Card(
            dbc.CardBody(
                [
                    html.H3(
                        t("ui.landing.about_study_title", locale=locale),
                        style={"fontSize": "22px", "fontWeight": "800", "color": "#1565c0"},
                    ),
                    html.Div(
                        t("ui.landing.about_study_text", locale=locale),
                        style={"color": "#334155", "lineHeight": "1.6"},
                    ),
                ]
            ),
            style={"borderRadius": "14px", "border": "1px solid rgba(15, 23, 42, 0.10)"},
        )

        consent_notice_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H3(
                        t("ui.landing.patient_consent_form_title", locale=locale),
                        style={"fontSize": "22px", "fontWeight": "800", "color": "#1565c0", "marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            *consent_notice_children(locale),
                            html.Hr(style={"margin": "14px 0"}),
                            html.H4(
                                t("ui.landing.required_consents_title", locale=locale),
                                style={"fontSize": "18px", "fontWeight": "800", "color": "#0f172a", "marginBottom": "8px"},
                            ),
                            dbc.Checklist(
                                id="consent-acknowledge",
                                options=[{"label": f" {t('ui.landing.consent_acknowledge_label', locale=locale)}", "value": "ack"}],
                                value=[],
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px", "marginBottom": "10px"},
                            ),
                            dbc.Checklist(
                                id="consent-gdpr",
                                options=[{"label": f" {t('ui.landing.consent_gdpr_label', locale=locale)}", "value": "gdpr"}],
                                value=[],
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px"},
                            ),
                            html.Hr(style={"margin": "14px 0"}),
                            html.H4(
                                t("ui.landing.optional_consents_title", locale=locale),
                                style={"fontSize": "18px", "fontWeight": "800", "color": "#0f172a", "marginBottom": "8px"},
                            ),
                            dbc.Checklist(
                                id="consent-upload-own-data",
                                options=[{"label": f" {t('ui.landing.consent_upload_own_data', locale=locale)}", "value": "upload_own_data"}],
                                value=[],
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px", "marginBottom": "10px"},
                            ),
                            dbc.Checklist(
                                id="consent-play-only",
                                options=[{"label": f" {t('ui.landing.consent_play_only', locale=locale)}", "value": "play_only"}],
                                value=[],
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px"},
                            ),
                            dbc.Checklist(
                                id="consent-receive-results",
                                options=[{"label": f" {t('ui.landing.consent_receive_results', locale=locale)}", "value": "receive_results"}],
                                value=[],
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px", "marginTop": "10px"},
                            ),
                            dbc.Checklist(
                                id="consent-keep-updated",
                                options=[{"label": f" {t('ui.landing.consent_keep_updated', locale=locale)}", "value": "keep_updated"}],
                                value=[],
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px", "marginTop": "10px"},
                            ),
                        ],
                        style={
                            "maxHeight": "calc(100vh - 360px)",
                            "overflowY": "auto",
                            "paddingRight": "10px",
                            "minHeight": "0",
                        },
                        id="consent-notice-scroll",
                    ),
                    html.Div(id="landing-error", style={"marginTop": "12px"}),
                    html.Button(
                        t("ui.common.continue", locale=locale),
                        id="landing-continue",
                        className="ui green button",
                        disabled=True,
                        style={
                            "marginTop": "14px",
                            "width": "220px",
                            "fontWeight": "700",
                            "backgroundColor": "#555555",
                            "color": "white",
                            "cursor": "not-allowed",
                        },
                    ),
                    html.Div(
                        t("ui.landing.next_hint", locale=locale),
                        style={"color": "#64748b", "marginTop": "10px", "fontSize": "14px"},
                    ),
                ],
                style={
                    "minHeight": "0",
                },
            ),
            style={
                "borderRadius": "14px",
                "border": "1px solid rgba(15, 23, 42, 0.10)",
                "flex": "1",
                "display": "flex",
                "flexDirection": "column",
                "minHeight": "0",
            },
        )

        layout = dbc.Container(
            [
                hero,
                html.Div(style={"height": "18px"}),
                study_info,
                html.Div(style={"height": "18px"}),
                consent_notice_card,
                dcc.Store(id="consent-scroll-complete", data=False, storage_type=STORAGE_TYPE),
                dcc.Interval(id="consent-scroll-poll", interval=500, n_intervals=0),
            ],
            fluid=False,
            style={"maxWidth": "1100px"},
        )

        super().__init__(
            children=[
                html.Div(
                    style={
                        "minHeight": "100vh",
                        "padding": "28px 18px",
                        "background": "linear-gradient(135deg, #eff6ff 0%, #f8fafc 35%, #fff7ed 100%)",
                    },
                    children=layout,
                )
            ],
            id=self.component_id,
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        # update continue button based on mandatory consents
        @app.callback(
            [
                Output("landing-continue", "disabled"),
                Output("landing-continue", "style"),
            ],
            [
                Input("consent-scroll-complete", "data"),
                Input("consent-acknowledge", "value"),
                Input("consent-gdpr", "value"),
            ],
        )
        def update_continue_button(
            scroll_complete: Optional[bool],
            acknowledge_value: Optional[list[str]],
            gdpr_value: Optional[list[str]],
        ) -> tuple[bool, dict[str, Any]]:
            scrolled_to_end = bool(scroll_complete)
            acknowledged = bool(acknowledge_value and "ack" in acknowledge_value)
            gdpr_consented = bool(gdpr_value and "gdpr" in gdpr_value)

            base_style = {"marginTop": "14px", "width": "220px", "fontWeight": "700", "color": "white"}
            if scrolled_to_end and acknowledged and gdpr_consented:
                # blue button when enabled
                return False, base_style
            else:
                # darker grey when disabled
                disabled_style = base_style.copy()
                disabled_style.update({
                    "color": "white",
                    "cursor": "not-allowed",
                })
                return True, disabled_style

        @app.callback(
            [
                Output("url", "pathname", allow_duplicate=True),
                Output("user-info-store", "data", allow_duplicate=True),
                Output("landing-error", "children"),
            ],
            [Input("landing-continue", "n_clicks")],
            [
                State("consent-acknowledge", "value"),
                State("consent-gdpr", "value"),
                State("consent-upload-own-data", "value"),
                State("consent-play-only", "value"),
                State("consent-receive-results", "value"),
                State("consent-keep-updated", "value"),
                State("user-info-store", "data"),
                State("interface-language", "data"),
            ],
            prevent_initial_call=True,
        )
        def handle_landing_continue(
            n_clicks: Optional[int],
            acknowledge_value: Optional[list[str]],
            gdpr_value: Optional[list[str]],
            upload_own_data_value: Optional[list[str]],
            play_only_value: Optional[list[str]],
            receive_results_value: Optional[list[str]],
            keep_updated_value: Optional[list[str]],
            user_info: Optional[dict[str, Any]],
            interface_language: Optional[str],
        ) -> tuple[str, dict[str, Any], Any]:
            if not n_clicks:
                return no_update, no_update, no_update

            acknowledged = bool(acknowledge_value and "ack" in acknowledge_value)
            gdpr_consented = bool(gdpr_value and "gdpr" in gdpr_value)
            if not acknowledged or not gdpr_consented:
                return (
                    no_update,
                    no_update,
                    dbc.Alert(t("ui.landing.consent_required_error", locale=interface_language), color="danger"),
                )

            info: dict[str, Any] = dict(user_info or {})
            if not info.get("study_id"):
                # Stable ID used across consent + (optional) later stats
                import uuid

                info["study_id"] = str(uuid.uuid4())

            any_selected = bool(play_only_value) or bool(receive_results_value) or bool(keep_updated_value)
            no_selection = not any_selected

            upload_own_data = bool(upload_own_data_value and "upload_own_data" in upload_own_data_value)
            play_only = bool(play_only_value and "play_only" in play_only_value)
            receive_results = bool(receive_results_value and "receive_results" in receive_results_value)
            keep_updated = bool(keep_updated_value and "keep_updated" in keep_updated_value)

            info["consent_gdpr"] = gdpr_consented
            info["consent_upload_own_data"] = upload_own_data
            info["consent_play_only"] = play_only
            # If user didn't select anything, record that they did not consent to participate.
            info["consent_participate_in_study"] = (not play_only) and (not no_selection)
            info["consent_receive_results_later"] = receive_results
            info["consent_keep_up_to_date"] = keep_updated
            info["consent_no_selection"] = no_selection
            info["consent_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if info.get("number") is None:
                info["number"] = get_next_study_number()

            ensure_consent_agreement_row(
                {
                    "study_id": info["study_id"],
                    "number": info.get("number", ""),
                    "timestamp": info["consent_timestamp"],
                    "gdpr_consent": gdpr_consented,
                    "upload_own_data": upload_own_data,
                    "play_only": play_only,
                    "participate_in_study": info["consent_participate_in_study"],
                    "receive_results_later": receive_results,
                    "keep_up_to_date": keep_updated,
                    "no_selection": no_selection,
                }
            )

            return "/startup", info, None

