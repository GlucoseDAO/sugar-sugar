"""End-of-game identity form: nickname, email, age, gender, location.

Collected after the player exits or finishes, before ``/final`` results.
Play-format fields stay on ``/startup``; these ids must NOT appear there
or ``handle_start_button`` / startup validation would drag them onto the
wrong page (same Dash trap as ``final-nickname-*``).
"""

from __future__ import annotations

from typing import Any, Optional

from dash import dcc, html

from sugar_sugar.components.startup import MAX_AGE
from sugar_sugar.config import STORAGE_TYPE
from sugar_sugar.i18n import t
from sugar_sugar.nickname import MAX_NICKNAME_LENGTH

PROFILE_FIELD_IDS: frozenset[str] = frozenset(
    {
        "nickname-input",
        "email-input",
        "age-input",
        "gender-dropdown",
        "location-input",
        "profile-receive-results",
        "profile-keep-updated",
        "profile-submit-button",
        "profile-switch-format-a",
        "profile-switch-format-b",
        "profile-switch-format-c",
    }
)


def _label(text: str, required_id: Optional[str] = None) -> html.Div:
    children: list[Any] = [
        html.Label(
            text,
            style={
                "fontSize": "22px",
                "fontWeight": "800",
                "marginBottom": "10px",
                "color": "#0f172a",
                "display": "inline-block",
            },
        )
    ]
    if required_id:
        children.append(
            html.Span(
                id=required_id,
                children=" *",
                style={"color": "#d32f2f", "fontSize": "22px", "fontWeight": "bold"},
            )
        )
    return html.Div(children, style={"marginBottom": "10px"}, disable_n_clicks=True)


def _input_style() -> dict[str, str]:
    return {"width": "100%", "padding": "10px", "fontSize": "20px", "marginBottom": "20px"}


def create_profile_layout(user_info: Optional[dict[str, Any]], *, locale: str) -> html.Div:
    """Identity + email prefs + other-format CTAs, shown before results."""
    info: dict[str, Any] = dict(user_info or {})
    uses_cgm = bool(info.get("uses_cgm", False))
    current_format = str(info.get("format") or "A")
    runs_by_format: dict[str, list[dict[str, Any]]] = dict(info.get("runs_by_format") or {})
    already_played: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
    if info.get("rounds"):
        already_played.add(current_format)
    allowed: list[str] = (["C", "B", "A"] if uses_cgm else ["A"])
    switch_targets: list[str] = [fmt for fmt in allowed if fmt not in already_played]

    receive_value = ["receive_results"] if info.get("consent_receive_results_later") else []
    keep_value = ["keep_updated"] if info.get("consent_keep_up_to_date") else []

    def _format_button(code: str) -> html.Button:
        visible = code in switch_targets
        return html.Button(
            t(f"ui.switch_format.try_{code.lower()}_short", locale=locale),
            id=f"profile-switch-format-{code.lower()}",
            className="ui blue button",
            n_clicks=0,
            style={
                "backgroundColor": "#1d4ed8",
                "color": "white",
                "padding": "0 14px",
                "border": "none",
                "borderRadius": "8px",
                "fontSize": "16px",
                "fontWeight": "700",
                "cursor": "pointer",
                "height": "48px",
                "minHeight": "48px",
                "display": "inline-flex" if visible else "none",
                "alignItems": "center",
                "justifyContent": "center",
                "margin": "0 6px 8px 0",
            },
        )

    return html.Div(
        [
            html.H1(
                t("ui.profile.title", locale=locale),
                id="profile-title",
                style={
                    "textAlign": "center",
                    "marginBottom": "12px",
                    "fontSize": "clamp(24px, 4vw, 40px)",
                    "color": "#2c5282",
                },
                disable_n_clicks=True,
            ),
            html.P(
                t("ui.profile.intro", locale=locale),
                id="profile-intro",
                style={
                    "textAlign": "center",
                    "color": "#475569",
                    "fontSize": "16px",
                    "marginBottom": "24px",
                    "lineHeight": "1.5",
                },
                disable_n_clicks=True,
            ),
            html.Div(
                [
                    _label(t("ui.startup.nickname_label", locale=locale)),
                    dcc.Input(
                        id="nickname-input",
                        type="text",
                        maxLength=MAX_NICKNAME_LENGTH,
                        value=str(info.get("nickname") or ""),
                        placeholder=t("ui.startup.nickname_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style=_input_style(),
                    ),
                    html.Small(
                        t("ui.startup.nickname_hint", locale=locale),
                        style={"color": "#666", "fontSize": "15px", "display": "block", "marginBottom": "20px"},
                    ),
                    _label(t("ui.startup.email_label", locale=locale), "email-required"),
                    dcc.Input(
                        id="email-input",
                        type="email",
                        value=str(info.get("email") or ""),
                        placeholder=t("ui.startup.email_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style=_input_style(),
                    ),
                    html.Div(
                        [
                            html.H3(
                                t("ui.startup.contact_prefs_title", locale=locale),
                                style={"fontSize": "20px", "marginBottom": "10px", "color": "#2c5282"},
                                disable_n_clicks=True,
                            ),
                            html.P(
                                t("ui.startup.contact_prefs_text", locale=locale),
                                style={"fontSize": "16px", "lineHeight": "1.5", "marginBottom": "10px", "color": "#555"},
                                disable_n_clicks=True,
                            ),
                            dcc.Checklist(
                                id="profile-receive-results",
                                options=[
                                    {
                                        "label": f" {t('ui.landing.consent_receive_results', locale=locale)}",
                                        "value": "receive_results",
                                    }
                                ],
                                value=receive_value,
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px", "marginBottom": "8px"},
                            ),
                            dcc.Checklist(
                                id="profile-keep-updated",
                                options=[
                                    {
                                        "label": f" {t('ui.landing.consent_keep_updated', locale=locale)}",
                                        "value": "keep_updated",
                                    }
                                ],
                                value=keep_value,
                                persistence=True,
                                persistence_type=STORAGE_TYPE,
                                style={"fontSize": "16px"},
                            ),
                        ],
                        style={
                            "backgroundColor": "#f8f9fa",
                            "padding": "16px",
                            "borderRadius": "8px",
                            "marginBottom": "20px",
                        },
                        disable_n_clicks=True,
                    ),
                    _label(t("ui.startup.age_label", locale=locale), "age-required"),
                    dcc.Input(
                        id="age-input",
                        type="number",
                        value=info.get("age"),
                        placeholder=t("ui.startup.age_placeholder", locale=locale),
                        min=0,
                        max=MAX_AGE,
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style=_input_style(),
                    ),
                    html.Div(
                        id="age-error",
                        children="",
                        style={"color": "#d32f2f", "fontSize": "16px", "marginTop": "-12px", "marginBottom": "20px"},
                        disable_n_clicks=True,
                    ),
                    _label(t("ui.startup.gender_label", locale=locale), "gender-required"),
                    dcc.Dropdown(
                        id="gender-dropdown",
                        options=[
                            {"label": t("ui.startup.gender_male", locale=locale), "value": "M"},
                            {"label": t("ui.startup.gender_female", locale=locale), "value": "F"},
                            {"label": t("ui.startup.gender_na", locale=locale), "value": "N/A"},
                        ],
                        value=info.get("gender"),
                        placeholder=t("ui.startup.gender_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={"fontSize": "20px", "marginBottom": "20px"},
                    ),
                    _label(t("ui.startup.location_label", locale=locale), "location-required"),
                    html.Div(
                        dcc.Input(
                            id="location-input",
                            type="text",
                            value=str(info.get("location") or ""),
                            placeholder=t("ui.startup.location_placeholder", locale=locale),
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={"width": "100%", "padding": "10px", "fontSize": "20px"},
                        ),
                        className="location-autocomplete-host",
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        id="profile-duration-error",
                        children="",
                        style={"color": "#d32f2f", "fontSize": "16px", "marginBottom": "12px"},
                        disable_n_clicks=True,
                    ),
                    html.Div(
                        id="profile-missing-fields",
                        children="",
                        style={"color": "#d32f2f", "fontSize": "14px", "marginBottom": "12px", "textAlign": "center"},
                        disable_n_clicks=True,
                    ),
                    html.H3(
                        t("ui.profile.other_formats_title", locale=locale),
                        id="profile-other-formats-title",
                        style={"fontSize": "22px", "color": "#2c5282", "margin": "24px 0 8px", "textAlign": "center"},
                        disable_n_clicks=True,
                    ),
                    html.P(
                        t("ui.profile.other_formats_text", locale=locale),
                        id="profile-other-formats-text",
                        style={"textAlign": "center", "color": "#475569", "marginBottom": "12px"},
                        disable_n_clicks=True,
                    ),
                    html.Div(
                        [
                            _format_button("A"),
                            _format_button("B"),
                            _format_button("C"),
                        ],
                        id="profile-switch-format-row",
                        style={"textAlign": "center", "marginBottom": "16px"},
                        disable_n_clicks=True,
                    ),
                    html.Div(id="profile-switch-format-error", disable_n_clicks=True),
                    html.Button(
                        t("ui.profile.see_results", locale=locale),
                        id="profile-submit-button",
                        className="ui green button",
                        n_clicks=0,
                        disabled=True,
                        style={
                            "backgroundColor": "#cccccc",
                            "color": "white",
                            "padding": "18px 30px",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontSize": "22px",
                            "cursor": "not-allowed",
                            "width": "100%",
                            "marginTop": "8px",
                        },
                    ),
                ],
                style={"maxWidth": "600px", "margin": "0 auto", "padding": "20px"},
                disable_n_clicks=True,
            ),
        ],
        id="profile-page",
        className="info-page profile-page",
        disable_n_clicks=True,
        style={
            "padding": "20px",
            "backgroundColor": "#f5f5f5",
            "minHeight": "100vh",
        },
    )
