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

from sugar_sugar.consent import ensure_consent_agreement_row, get_next_study_number


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
    def __init__(self) -> None:
        self.component_id: str = "landing-page"

        project_root = Path(__file__).resolve().parents[2]
        screenshot_path = project_root / "images" / "screenshot.png"
        screenshot_src = _image_data_uri(screenshot_path)

        hero = dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Sugar Sugar",
                            style={
                                "fontSize": "56px",
                                "fontWeight": "800",
                                "letterSpacing": "-0.02em",
                                "marginBottom": "10px",
                            },
                        ),
                        html.Div(
                            "A game to test your glucose-predicting superpowers.",
                            style={
                                "fontSize": "20px",
                                "color": "#334155",
                                "marginBottom": "18px",
                                "lineHeight": "1.4",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "How it works",
                                    style={
                                        "fontWeight": "700",
                                        "marginBottom": "8px",
                                        "color": "#0f172a",
                                    },
                                ),
                                html.Ul(
                                    [
                                        html.Li("Upload Dexcom/Libre CSV (or use the sample dataset)"),
                                        html.Li("Draw your predicted glucose trend on the chart"),
                                        html.Li("Compare predictions to the real values"),
                                        html.Li("See accuracy metrics (MAE/RMSE/MAPE)"),
                                    ],
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
                                            "Preview",
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
                        "About the study",
                        style={"fontSize": "22px", "fontWeight": "800", "color": "#0f172a"},
                    ),
                    html.Div(
                        "We're studying how accurate humans are at predicting glucose trends, and what patterns experienced CGM users notice. "
                        "If you choose to participate, your gameplay and uploaded CGM data may be used for further research.",
                        style={"color": "#334155", "lineHeight": "1.6"},
                    ),
                ]
            ),
            style={"borderRadius": "14px", "border": "1px solid rgba(15, 23, 42, 0.10)"},
        )

        consent_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H3(
                        "Your choices",
                        style={"fontSize": "22px", "fontWeight": "800", "color": "#0f172a"},
                    ),
                    html.Div(
                        "By default, we assume you agree to participate in the study. If you only want to play, check the first box.",
                        style={"color": "#334155", "lineHeight": "1.6", "marginBottom": "10px"},
                    ),
                    dbc.Checklist(
                        id="consent-play-only",
                        options=[
                            {
                                "label": " I just want to play (do not store my CGM / gameplay data)",
                                "value": "play_only",
                            }
                        ],
                        value=[],
                        style={"fontSize": "16px"},
                    ),
                    dbc.Checklist(
                        id="consent-receive-results",
                        options=[{"label": " I want to receive my results later", "value": "receive_results"}],
                        value=[],
                        style={"fontSize": "16px", "marginTop": "10px"},
                    ),
                    dbc.Checklist(
                        id="consent-keep-updated",
                        options=[{"label": " Keep me up to date about project updates", "value": "keep_updated"}],
                        value=[],
                        style={"fontSize": "16px", "marginTop": "10px"},
                    ),
                    html.Div(
                        id="landing-error",
                        style={"marginTop": "12px"},
                    ),
                    dbc.Button(
                        "Continue",
                        id="landing-continue",
                        color="primary",
                        style={"marginTop": "14px", "width": "220px", "fontWeight": "700"},
                    ),
                    html.Div(
                        "Next: you'll fill in a short form, then upload CGM data (or use the sample dataset).",
                        style={"color": "#64748b", "marginTop": "10px", "fontSize": "14px"},
                    ),
                ]
            ),
            style={"borderRadius": "14px", "border": "1px solid rgba(15, 23, 42, 0.10)"},
        )

        layout = dbc.Container(
            [
                hero,
                html.Div(style={"height": "18px"}),
                dbc.Row(
                    [
                        dbc.Col(study_info, md=6),
                        dbc.Col(consent_card, md=6),
                    ],
                    className="g-4",
                ),
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
        @app.callback(
            [Output("url", "pathname", allow_duplicate=True), Output("user-info-store", "data", allow_duplicate=True)],
            [Input("landing-continue", "n_clicks")],
            [
                State("consent-play-only", "value"),
                State("consent-receive-results", "value"),
                State("consent-keep-updated", "value"),
                State("user-info-store", "data"),
            ],
            prevent_initial_call=True,
        )
        def handle_landing_continue(
            n_clicks: Optional[int],
            play_only_value: Optional[list[str]],
            receive_results_value: Optional[list[str]],
            keep_updated_value: Optional[list[str]],
            user_info: Optional[dict[str, Any]],
        ) -> tuple[str, dict[str, Any]]:
            if not n_clicks:
                return no_update, no_update

            info: dict[str, Any] = dict(user_info or {})
            if not info.get("study_id"):
                # Stable ID used across consent + (optional) later stats
                import uuid

                info["study_id"] = str(uuid.uuid4())

            any_selected = bool(play_only_value) or bool(receive_results_value) or bool(keep_updated_value)
            no_selection = not any_selected

            play_only = bool(play_only_value and "play_only" in play_only_value)
            receive_results = bool(receive_results_value and "receive_results" in receive_results_value)
            keep_updated = bool(keep_updated_value and "keep_updated" in keep_updated_value)

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
                    "play_only": play_only,
                    "participate_in_study": info["consent_participate_in_study"],
                    "receive_results_later": receive_results,
                    "keep_up_to_date": keep_updated,
                    "no_selection": no_selection,
                }
            )

            return "/startup", info

