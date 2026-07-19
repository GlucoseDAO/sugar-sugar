from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import no_update
import dash
from dataclasses import dataclass
from typing import Any, Optional
# DEBUG_MODE will be imported dynamically to get the latest value
from sugar_sugar.components.landing import consent_controls_children
from sugar_sugar.components.submit import _is_mobile_ua
from sugar_sugar.i18n import t
from sugar_sugar.config import STORAGE_TYPE
from flask import has_request_context, request as flask_request

MAX_AGE: int = 130
MIN_AGE: int = 18

_MISSING_FIELDS_STYLE: dict[str, str] = {
    'color': '#d32f2f',
    'fontSize': '14px',
    'marginTop': '8px',
    'marginBottom': '8px',
    'lineHeight': '1.4',
}


def _truthy_consent_flag(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def prior_upload_data_consent(user_info: Optional[dict[str, Any]]) -> bool:
    """True if upload/CGM-data usage consent is already established for this session.

    Sources (any one is enough — do not re-ask on prediction / format switch):
    - landing/startup flags: ``consent_use_uploaded_data`` / ``consent_upload_own_data``
    - a successful CGM upload already on the session (``uploaded_data_path``)
    - any completed own-data round (current run or archived ``runs_by_format``)
    - any archived B/C run that recorded upload consent or completed rounds
    """
    info = user_info or {}
    if _truthy_consent_flag(info.get("consent_use_uploaded_data")):
        return True
    if _truthy_consent_flag(info.get("consent_upload_own_data")):
        return True
    if info.get("uploaded_data_path"):
        return True

    for round_info in info.get("rounds") or []:
        if isinstance(round_info, dict) and round_info.get("is_example_data") is False:
            return True

    for fmt, runs in (info.get("runs_by_format") or {}).items():
        for run in runs or []:
            if not isinstance(run, dict):
                continue
            if _truthy_consent_flag(run.get("consent_use_uploaded_data")):
                return True
            for round_info in run.get("rounds") or []:
                if isinstance(round_info, dict) and round_info.get("is_example_data") is False:
                    return True
            # Already finished a B/C game this session => upload consent was required.
            if str(fmt) in ("B", "C") and int(run.get("rounds_played") or len(run.get("rounds") or [])) > 0:
                return True
    return False


def stamp_upload_data_consent(user_info: dict[str, Any]) -> dict[str, Any]:
    """Persist upload-consent flags when any prior evidence exists."""
    if prior_upload_data_consent(user_info):
        user_info["consent_use_uploaded_data"] = True
        if not _truthy_consent_flag(user_info.get("consent_upload_own_data")):
            # Keep the landing optional flag aligned once usage consent is known.
            user_info["consent_upload_own_data"] = True
    return user_info


def import_controls_children(locale: str) -> list[Any]:
    """Startup-stage 'import your CGM data' block: CSV upload + Nightscout fetch.

    Rendered inside the data-usage-consent container (revealed for formats B/C), so
    the user can bring their data in on the roomy startup screen instead of the
    cramped prediction page. Uses dedicated ``startup-*`` ids so it never clashes
    with the /prediction upload (id ``upload-data``). The import callbacks write
    ``uploaded_data_path`` into ``user-info-store``; the game picks it up from there.
    """
    _input_style: dict[str, str] = {
        'width': '100%', 'marginBottom': '8px', 'padding': '10px 12px',
        'borderRadius': '6px', 'border': '1px solid #cbd5e1', 'fontSize': '16px',
        'boxSizing': 'border-box',
    }
    return [
        html.Div(
            t("ui.startup.import_title", locale=locale),
            style={'fontWeight': '800', 'fontSize': '18px', 'marginTop': '16px',
                   'marginBottom': '4px', 'color': '#0f172a'},
        ),
        html.Div(
            t("ui.startup.import_subtitle", locale=locale),
            style={'fontSize': '13px', 'color': '#64748b', 'marginBottom': '10px', 'lineHeight': '1.4'},
        ),
        dcc.Upload(
            id='startup-upload-data',
            multiple=False,
            accept='.csv,text/csv',
            children=html.Div(t("ui.startup.import_upload_prompt", locale=locale), id='startup-upload-prompt'),
            style={
                'width': '100%', 'minHeight': '56px', 'display': 'flex',
                'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center',
                'padding': '10px', 'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderColor': '#2185d0', 'borderRadius': '8px', 'color': '#2185d0',
                'cursor': 'pointer', 'backgroundColor': '#f8fbff', 'boxSizing': 'border-box',
            },
        ),
        html.Div(
            t("ui.startup.import_or", locale=locale),
            style={'textAlign': 'center', 'color': '#94a3b8', 'fontSize': '13px', 'margin': '10px 0'},
        ),
        dcc.Input(
            id='startup-ns-url', type='url',
            placeholder=t("ui.startup.import_ns_url_placeholder", locale=locale),
            debounce=True, style=_input_style,
        ),
        dcc.Input(
            id='startup-ns-token', type='text',
            placeholder=t("ui.startup.import_ns_token_placeholder", locale=locale),
            debounce=True, style=_input_style,
        ),
        html.Button(
            t("ui.startup.import_ns_button", locale=locale),
            id='startup-ns-import', type='button', className='ui blue button',
            n_clicks=0, style={'width': '100%', 'marginBottom': '8px'},
        ),
        dcc.Loading(
            html.Div(id='startup-import-status', style={'marginTop': '4px', 'fontSize': '15px', 'lineHeight': '1.4'}),
            type='dot',
        ),
    ]


def _import_status_msg(text: str, *, ok: bool) -> Any:
    """Small coloured status pill for the startup import block."""
    return html.Div(
        text,
        style={
            'padding': '8px 10px', 'borderRadius': '6px', 'marginTop': '4px',
            'backgroundColor': '#dcfce7' if ok else '#fee2e2',
            'color': '#166534' if ok else '#b91c1c',
            'fontWeight': '600', 'lineHeight': '1.4',
        },
    )


@dataclass(frozen=True)
class StartupValidationResult:
    """Outcome of startup-form field checks for one wizard step or the full form."""

    age_error: str
    diabetes_duration_error: str
    cgm_duration_error: str
    data_usage_error: str
    missing_label_keys: tuple[str, ...]
    has_range_errors: bool
    step_complete: bool
    form_complete: bool
    _locale: Optional[str] = None

    @property
    def missing_fields_message(self) -> str:
        if not self.missing_label_keys:
            return ""
        labels = [t(key, locale=self._locale) for key in self.missing_label_keys]
        return t("ui.startup.missing_fields", locale=self._locale, fields=", ".join(labels))

    @property
    def hint_message(self) -> str:
        parts: list[str] = []
        missing = self.missing_fields_message
        if missing:
            parts.append(missing)
        if self.has_range_errors:
            parts.append(t("ui.startup.values_check_warning", locale=self._locale))
        return "\n".join(parts)

    def hint_children(self) -> Any:
        """Dash-friendly children for the missing-fields / values-check hint."""
        text = self.hint_message
        if not text:
            return ""
        lines = [line for line in text.split("\n") if line]
        if len(lines) == 1:
            return lines[0]
        return [html.Div(line, style={'marginBottom': '4px'}) for line in lines]


def _wants_contact_from_user_info(user_info: Optional[dict[str, Any]]) -> bool:
    info: dict[str, Any] = dict(user_info or {})
    return bool(
        info.get('consent_receive_results_later') or
        info.get('consent_keep_up_to_date')
    )


def _age_field_errors(age: Optional[int | float], locale: Optional[str]) -> tuple[str, bool]:
    if age is None:
        return "", False
    age_f = float(age)
    if age_f < MIN_AGE:
        return t("ui.startup.age_must_be_18_error", locale=locale), True
    if age_f > MAX_AGE:
        return t("ui.startup.age_max_error", locale=locale, max_age=MAX_AGE), True
    return "", True


def _duration_exceeds_age(
    duration: Optional[int | float],
    age: Optional[int | float],
) -> bool:
    if duration is None or age is None:
        return False
    return float(duration) > float(age)


def validate_startup_form(
    *,
    email: Optional[str],
    age: Optional[int | float],
    gender: Optional[str],
    format_value: Optional[str],
    data_usage_consent: Optional[list[str]],
    is_diabetic: Optional[bool],
    diabetic_type: Optional[str],
    diabetes_duration: Optional[int | float],
    location: Optional[str],
    uses_cgm: Optional[bool],
    cgm_duration: Optional[int | float],
    wants_contact: bool,
    locale: Optional[str],
    wizard_step: Optional[int] = None,
    prior_upload_consent: bool = False,
) -> StartupValidationResult:
    """Validate startup fields for a wizard step (0-5) or the full form (``wizard_step=None``)."""
    age_error, age_in_range = _age_field_errors(age, locale)
    diabetes_duration_error = (
        t("ui.startup.diabetes_duration_exceeds_age_error", locale=locale)
        if _duration_exceeds_age(diabetes_duration, age)
        else ""
    )
    cgm_duration_error = (
        t("ui.startup.cgm_duration_exceeds_age_error", locale=locale)
        if uses_cgm is True and _duration_exceeds_age(cgm_duration, age)
        else ""
    )

    needs_data_consent = format_value in ("B", "C")
    has_data_consent = (
        bool(data_usage_consent and "agree" in data_usage_consent)
        or bool(prior_upload_consent)
    )
    data_usage_error = (
        t("ui.startup.data_usage_consent_required", locale=locale)
        if needs_data_consent and not has_data_consent
        else ""
    )

    has_range_errors = bool(
        age_error or diabetes_duration_error or cgm_duration_error
    )

    missing: list[str] = []

    def _identity_missing() -> None:
        if wants_contact and not email:
            missing.append("ui.startup.email_label")
        if not age:
            missing.append("ui.startup.age_label")
        if not gender:
            missing.append("ui.startup.gender_label")
        if not location:
            missing.append("ui.startup.location_label")

    def _cgm_missing() -> None:
        if uses_cgm is None:
            missing.append("ui.startup.cgm_label")
        elif uses_cgm is True and cgm_duration is None:
            missing.append("ui.startup.cgm_duration_label")

    def _diabetes_missing() -> None:
        if is_diabetic is None:
            missing.append("ui.startup.diabetic_label")
        elif is_diabetic:
            if not diabetic_type:
                missing.append("ui.startup.diabetes_type_label")
            if diabetes_duration is None:
                missing.append("ui.startup.diabetes_duration_label")

    def _format_missing() -> None:
        if not format_value:
            missing.append("ui.startup.format_label")
        if needs_data_consent and not has_data_consent:
            missing.append("ui.startup.data_usage_consent_label")

    if wizard_step is None:
        _identity_missing()
        _cgm_missing()
        _diabetes_missing()
        _format_missing()
    elif wizard_step == 1:
        _identity_missing()
    elif wizard_step == 2:
        _cgm_missing()
    elif wizard_step == 3:
        _diabetes_missing()
    elif wizard_step == 4:
        _format_missing()

    step_range_ok = True
    if wizard_step in (None, 1):
        step_range_ok = step_range_ok and age_in_range and not age_error
    if wizard_step in (None, 2) and uses_cgm is True:
        step_range_ok = step_range_ok and not cgm_duration_error
    if wizard_step in (None, 3) and is_diabetic:
        step_range_ok = step_range_ok and not diabetes_duration_error

    step_complete = not missing and step_range_ok
    if wizard_step == 4:
        step_complete = step_complete and not data_usage_error

    is_adult = (age is not None) and (float(age) >= MIN_AGE) and (float(age) <= MAX_AGE)
    form_complete = (
        not missing
        and is_adult
        and not has_range_errors
        and format_value
        and is_diabetic is not None
        and location
        and (email if wants_contact else True)
        and uses_cgm is not None
        and (not uses_cgm or cgm_duration is not None)
        and (not is_diabetic or (diabetic_type and diabetes_duration is not None))
        and (not needs_data_consent or has_data_consent)
    )

    return StartupValidationResult(
        age_error=age_error,
        diabetes_duration_error=diabetes_duration_error,
        cgm_duration_error=cgm_duration_error,
        data_usage_error=data_usage_error,
        missing_label_keys=tuple(missing),
        has_range_errors=has_range_errors,
        step_complete=step_complete,
        form_complete=form_complete,
        _locale=locale,
    )


def _step_hint_children(
    *,
    email: Optional[str],
    age: Optional[int | float],
    gender: Optional[str],
    format_value: Optional[str],
    data_usage_consent: Optional[list[str]],
    is_diabetic: Optional[bool],
    diabetic_type: Optional[str],
    diabetes_duration: Optional[int | float],
    location: Optional[str],
    uses_cgm: Optional[bool],
    cgm_duration: Optional[int | float],
    wants_contact: bool,
    locale: Optional[str],
    current_step: int,
    prior_upload_consent: bool = False,
) -> Any:
    if 1 <= current_step <= 4:
        return validate_startup_form(
            email=email,
            age=age,
            gender=gender,
            format_value=format_value,
            data_usage_consent=data_usage_consent,
            is_diabetic=is_diabetic,
            diabetic_type=diabetic_type,
            diabetes_duration=diabetes_duration,
            location=location,
            uses_cgm=uses_cgm,
            cgm_duration=cgm_duration,
            wants_contact=wants_contact,
            locale=locale,
            wizard_step=current_step,
            prior_upload_consent=prior_upload_consent,
        ).hint_children()
    if current_step == 0:
        return ""
    return validate_startup_form(
        email=email,
        age=age,
        gender=gender,
        format_value=format_value,
        data_usage_consent=data_usage_consent,
        is_diabetic=is_diabetic,
        diabetic_type=diabetic_type,
        diabetes_duration=diabetes_duration,
        location=location,
        uses_cgm=uses_cgm,
        cgm_duration=cgm_duration,
        wants_contact=wants_contact,
        prior_upload_consent=prior_upload_consent,
        locale=locale,
    ).hint_children()



def _compute_format_options(
    uses_cgm: Optional[bool],
    interface_language: Optional[str],
    current_format: Optional[str],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Return the dropdown options list and the desired selected value.

    Keeping the option ordering consistent (A, B, C) is important for the
dropdown scroller.  Formats B and C are disabled unless ``uses_cgm`` is True.
    The returned ``value`` is used to update the component's value according to
eligibility and previous selection.
    """
    allow_all = uses_cgm is True
    options: list[dict[str, Any]] = [
        {
            'label': t("ui.startup.format_a_label", locale=interface_language),
            'value': 'A',
        },
        {
            'label': t("ui.startup.format_b_label", locale=interface_language),
            'value': 'B',
            'disabled': not allow_all,
        },
        {
            'label': t("ui.startup.format_c_label", locale=interface_language),
            'value': 'C',
            'disabled': not allow_all,
        },
    ]

    if not current_format:
        return options, ('C' if allow_all else 'A')
    if allow_all and current_format == 'A':
        # Encourage option C once eligible.
        return options, 'C'
    if not allow_all and current_format in ('B', 'C'):
        return options, 'A'
    return options, current_format


class StartupPage(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self.component_id: str = 'startup-page'
        self._locale: str = locale
        
        # Create the layout
        layout = [
            html.H1(t("ui.common.app_title", locale=locale), 
                style={
                    'textAlign': 'center', 
                    'marginBottom': '30px', 
                    'fontSize': '48px',
                    'fontWeight': 'bold',
                    'color': '#2c5282'  # Match the prediction page color
                }
            ),
            html.Div([
                html.Div([
                    html.Div([
                        html.P(t("ui.startup.required_fields_note", locale=locale), style={'color': '#666', 'fontSize': '16px', 'fontStyle': 'italic', 'marginBottom': '20px', 'textAlign': 'right'})
                    ]),
                    
                    html.Div([
                        html.Label(t("ui.startup.email_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='email-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='email-input',
                        type='email',
                        placeholder=t("ui.startup.email_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Label(t("ui.startup.age_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='age-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='age-input',
                        type='number',
                        placeholder=t("ui.startup.age_placeholder", locale=locale),
                        min=0,
                        max=130,
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    html.Div(
                        id='age-error',
                        children='',
                        style={'color': '#d32f2f', 'fontSize': '16px', 'marginTop': '-12px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Label(t("ui.startup.gender_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='gender-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='gender-dropdown',
                        options=[
                            {'label': t("ui.startup.gender_male", locale=locale), 'value': 'M'},
                            {'label': t("ui.startup.gender_female", locale=locale), 'value': 'F'},
                            {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'}
                        ],
                        placeholder=t("ui.startup.gender_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),

                    html.Label(t("ui.startup.cgm_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a'}),
                    dcc.Dropdown(
                        id='cgm-dropdown',
                        options=[
                            {'label': t("ui.startup.yes", locale=locale), 'value': True},
                            {'label': t("ui.startup.no", locale=locale), 'value': False}
                        ],
                        placeholder=t("ui.startup.cgm_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),

                    html.Div(id='cgm-details', children=[
                        html.Label(t("ui.startup.cgm_duration_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a'}),
                        dcc.Input(
                            id='cgm-duration-input',
                            type='number',
                            placeholder=t("ui.startup.cgm_duration_placeholder", locale=locale),
                            min=0,
                            max=130,
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '8px'}
                        ),
                        html.Div(
                            id='cgm-duration-error',
                            children='',
                            style={'color': '#d32f2f', 'fontSize': '16px', 'marginBottom': '20px'}
                        )
                    ]),

                    html.Div([
                        html.Div([
                            html.Label(t("ui.startup.format_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                            html.Span(id='format-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='format-dropdown',
                            options=[
                                {'label': t("ui.startup.format_a_label", locale=locale), 'value': 'A'},
                                {'label': t("ui.startup.format_b_label", locale=locale), 'value': 'B', 'disabled': True},
                                {'label': t("ui.startup.format_c_label", locale=locale), 'value': 'C', 'disabled': True},
                            ],
                            placeholder=t("ui.startup.format_placeholder", locale=locale),
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'fontSize': '20px', 'marginBottom': '10px'}
                        ),
                        html.Div(
                            [
                                html.Small(t("ui.startup.format_help_a", locale=locale)),
                                html.Br(),
                                html.Small(t("ui.startup.format_help_b", locale=locale)),
                                html.Br(),
                                html.Small(t("ui.startup.format_help_c", locale=locale)),
                            ],
                            style={'color': '#666', 'fontSize': '14px', 'marginBottom': '20px', 'lineHeight': '1.4'}
                        ),
                        html.Div(
                            id='data-usage-consent-container',
                            children=[
                                dcc.Checklist(
                                    id='data-usage-consent',
                                    options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
                                    value=[],
                                    persistence=True,
                                    persistence_type=STORAGE_TYPE,
                                    style={'fontSize': '16px'}
                                ),
                                html.Div(id='data-usage-error', style={'marginTop': '8px', 'color': '#d32f2f', 'fontSize': '16px'}),
                                *import_controls_children(locale),
                            ],
                            style={'display': 'none', 'marginBottom': '20px'}
                        ),
                    ], style={'marginBottom': '10px'}),
                    
                    html.Div([
                        html.Label(t("ui.startup.diabetic_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='diabetic-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='diabetic-dropdown',
                        options=[
                            {'label': t("ui.startup.yes", locale=locale), 'value': True},
                            {'label': t("ui.startup.no", locale=locale), 'value': False}
                        ],
                        placeholder=t("ui.startup.diabetic_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(id='diabetic-details', children=[
                        html.Div([
                            html.Label(t("ui.startup.diabetes_type_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                            html.Span(id='diabetic-type-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='diabetic-type-dropdown',
                            options=[
                                {'label': t("ui.startup.diabetes_type_1", locale=locale), 'value': 'Type 1'},
                                {'label': t("ui.startup.diabetes_type_2", locale=locale), 'value': 'Type 2'},
                                {'label': t("ui.startup.diabetes_type_gestational", locale=locale), 'value': 'Gestational'},
                                {'label': t("ui.startup.diabetes_type_lada", locale=locale), 'value': 'LADA'},
                                {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'}
                            ],
                            placeholder=t("ui.startup.diabetes_type_placeholder", locale=locale),
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'fontSize': '20px', 'marginBottom': '20px'}
                        ),
                        
                        html.Div([
                            html.Label(t("ui.startup.diabetes_duration_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                            html.Span(id='diabetes-duration-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Input(
                            id='diabetes-duration-input',
                            type='number',
                            placeholder=t("ui.startup.diabetes_duration_placeholder", locale=locale),
                            min=0,
                            max=130,
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '8px'}
                        ),
                        html.Div(
                            id='diabetes-duration-error',
                            children='',
                            style={'color': '#d32f2f', 'fontSize': '16px', 'marginBottom': '20px'}
                        )
                    ]),
                    
                    html.Div([
                        html.Label(t("ui.startup.location_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='location-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    html.Div(
                        dcc.Input(
                            id='location-input',
                            type='text',
                            placeholder=t("ui.startup.location_placeholder", locale=locale),
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px'}
                        ),
                        className='location-autocomplete-host',
                        style={'marginBottom': '20px'},
                    ),
                    
                    html.Div(
                        [
                            html.H3(
                                t("ui.startup.contact_prefs_title", locale=locale),
                                style={'fontSize': '24px', 'marginBottom': '12px', 'color': '#2c5282'}
                            ),
                            html.P(
                                t("ui.startup.contact_prefs_text", locale=locale),
                                style={'fontSize': '18px', 'lineHeight': '1.6', 'marginBottom': '0', 'color': '#555'}
                            ),
                        ],
                        style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}
                    ),
                    
                    # <!-- START INSERTION: Just Test Me Button (Debug Mode Only) --> 
                    html.Div([
                        html.Button(
                            t("ui.startup.just_test_me", locale=locale),
                            id='test-me-button',
                            className="ui blue-action button",
                            style={
                                'backgroundColor': '#1976D2',
                                'color': 'white',
                                'padding': '15px 25px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'fontSize': '18px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'height': '60px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'lineHeight': '1.2',
                                'marginBottom': '15px'
                            }
                        )
                    ], style={
                        'textAlign': 'center', 
                        'marginTop': '30px',
                        'display': 'block' if self._get_debug_mode() else 'none'
                    }),
                    # <!-- END INSERTION: Just Test Me Button (Debug Mode Only) -->
                    
                    html.Div(
                        id='startup-missing-fields',
                        children='',
                        style={**_MISSING_FIELDS_STYLE, 'display': 'none', 'textAlign': 'center'},
                    ),
                    html.Div([
                        html.Button(
                            t("ui.startup.start_prediction", locale=locale),
                            id='start-button',
                            className="ui green button",
                            disabled=True,  # Initially disabled until consent is given
                            style={
                                'backgroundColor': '#cccccc',  # Gray when disabled
                                'color': 'white',
                                'padding': '20px 30px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'fontSize': '24px',
                                'cursor': 'not-allowed',  # Show not-allowed cursor when disabled
                                'width': '100%',
                                'height': '80px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'lineHeight': '1.2'
                            }
                        )
                    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
                    

                ], style={'maxWidth': '600px', 'margin': '0 auto', 'padding': '20px'})
            ], style={'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'})
        ]
        
        # Initialize the parent html.Div with the layout and styling
        super().__init__(
            children=layout,
            id=self.component_id,
            style={
                'padding': '20px', 
                'backgroundColor': '#f5f5f5', 
                'minHeight': '100vh',
                'display': 'flex',
                'flexDirection': 'column'
            }
        )

    def _get_debug_mode(self) -> bool:
        """Dynamically get the current DEBUG_MODE value."""
        try:
            from sugar_sugar.config import DEBUG_MODE
            return DEBUG_MODE
        except ImportError:
            return False

    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            [Output('format-dropdown', 'options'),
             Output('format-dropdown', 'value')],
            [Input('cgm-dropdown', 'value'),
             Input('interface-language', 'data')],
            [State('format-dropdown', 'value')]
        )
        def update_format_options(
            uses_cgm: Optional[bool],
            interface_language: Optional[str],
            current_format: Optional[str],
        ) -> tuple[list[dict[str, Any]], Optional[str]]:
            # delegate to helper so we can unit-test behaviour independently
            return _compute_format_options(uses_cgm, interface_language, current_format)

        @app.callback(
            [Output('data-usage-consent-container', 'style'),
             Output('data-usage-consent', 'value')],
            [Input('format-dropdown', 'value'),
             Input('user-info-store', 'data')],
            [State('data-usage-consent', 'value')],
        )
        def toggle_data_usage_consent(
            format_value: Optional[str],
            user_info: Optional[dict[str, Any]],
            current_value: Optional[list[str]],
        ) -> tuple[dict[str, str], list[str]]:
            if format_value in ('B', 'C'):
                # Landing/mobile may already have recorded upload consent — pre-tick
                # so the user is not asked twice before Start.
                if prior_upload_data_consent(user_info):
                    return {'display': 'block', 'marginBottom': '20px'}, ['agree']
                return {'display': 'block', 'marginBottom': '20px'}, list(current_value or [])
            return {'display': 'none', 'marginBottom': '20px'}, []

        # --- Startup-stage data import (CSV upload + Nightscout) --------------
        # Both write only `uploaded_data_path` / `uploaded_data_filename` into
        # user-info-store; handle_start_button carries them forward and the game
        # loads the window from that path (see resolve_dataset_identity). Consent
        # (for B/C) is required before we persist any personal data.
        def _consent_ok(
            format_value: Optional[str],
            consent_value: Optional[list[str]],
            user_info: Optional[dict[str, Any]] = None,
        ) -> bool:
            if format_value not in ('B', 'C'):
                return True
            return bool(consent_value and 'agree' in consent_value) or prior_upload_data_consent(user_info)

        @app.callback(
            [Output('user-info-store', 'data', allow_duplicate=True),
             Output('startup-import-status', 'children', allow_duplicate=True)],
            [Input('startup-upload-payload', 'data')],
            [State('startup-upload-data', 'filename'),
             State('user-info-store', 'data'),
             State('format-dropdown', 'value'),
             State('data-usage-consent', 'value'),
             State('interface-language', 'data')],
            prevent_initial_call=True,
        )
        def handle_startup_csv_upload(
            contents: Optional[str],
            filename: Optional[str],
            user_info: Optional[dict[str, Any]],
            format_value: Optional[str],
            consent_value: Optional[list[str]],
            interface_language: Optional[str],
        ) -> tuple[Any, Any]:
            from datetime import datetime
            from pathlib import Path
            from sugar_sugar.i18n import normalize_locale
            from sugar_sugar.data import load_glucose_data, decode_upload_bytes

            if not contents:
                raise dash.exceptions.PreventUpdate
            locale = normalize_locale(interface_language)
            if not _consent_ok(format_value, consent_value, user_info):
                return no_update, _import_status_msg(t("ui.startup.import_needs_consent", locale=locale), ok=False)
            decoded = decode_upload_bytes(contents)
            if decoded is None:
                return no_update, _import_status_msg(t("ui.startup.import_bad_file", locale=locale), ok=False)

            # Parse/save is fully guarded: a malformed file must never 500 the app.
            try:
                users_dir = Path('data/input/users')
                users_dir.mkdir(parents=True, exist_ok=True)
                safe = (filename or 'uploaded').replace(' ', '_').replace('/', '_')
                if not safe.lower().endswith('.csv'):
                    safe += '.csv'
                save_path = users_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"
                save_path.write_bytes(decoded)
                glucose_df, _events = load_glucose_data(save_path)
            except Exception:
                return no_update, _import_status_msg(t("ui.startup.import_bad_file", locale=locale), ok=False)

            if glucose_df is None or glucose_df.height == 0:
                return no_update, _import_status_msg(t("ui.startup.import_empty", locale=locale), ok=False)

            info = dict(user_info or {})
            info['uploaded_data_path'] = str(save_path)
            info['uploaded_data_filename'] = str(filename or 'uploaded.csv')
            return info, _import_status_msg(
                t("ui.startup.import_success", locale=locale, count=glucose_df.height, source=str(filename or 'CSV')),
                ok=True,
            )

        @app.callback(
            [Output('user-info-store', 'data', allow_duplicate=True),
             Output('startup-import-status', 'children', allow_duplicate=True)],
            [Input('startup-ns-import', 'n_clicks')],
            [State('startup-ns-url', 'value'),
             State('startup-ns-token', 'value'),
             State('user-info-store', 'data'),
             State('format-dropdown', 'value'),
             State('data-usage-consent', 'value'),
             State('interface-language', 'data')],
            prevent_initial_call=True,
        )
        def handle_startup_nightscout_import(
            n_clicks: Optional[int],
            url: Optional[str],
            token: Optional[str],
            user_info: Optional[dict[str, Any]],
            format_value: Optional[str],
            consent_value: Optional[list[str]],
            interface_language: Optional[str],
        ) -> tuple[Any, Any]:
            from pathlib import Path
            from sugar_sugar.i18n import normalize_locale

            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            locale = normalize_locale(interface_language)
            if not _consent_ok(format_value, consent_value, user_info):
                return no_update, _import_status_msg(t("ui.startup.import_needs_consent", locale=locale), ok=False)
            if not (url and url.strip()):
                return no_update, _import_status_msg(t("ui.startup.import_needs_url", locale=locale), ok=False)

            try:
                import httpx as _httpx
            except Exception:
                _httpx = None

            # The whole network path is guarded so a malformed/incomplete/offline
            # Nightscout never crashes the app -- each failure maps to a friendly,
            # non-leaking message (raw exception text can contain the URL/token).
            try:
                from sugar_sugar.data import load_glucose_data_from_nightscout
                glucose_df, _events, save_path = load_glucose_data_from_nightscout(
                    url.strip(), token=(token or None), save_dir=Path('data/input/users'),
                )
            except ImportError:
                return no_update, _import_status_msg(t("ui.startup.import_ns_unavailable", locale=locale), ok=False)
            except Exception as exc:  # network / HTTP / parse -- classify, don't leak
                name = type(exc).__name__
                if _httpx is not None and isinstance(exc, _httpx.HTTPStatusError):
                    code = exc.response.status_code if getattr(exc, 'response', None) is not None else '?'
                    return no_update, _import_status_msg(t("ui.startup.import_ns_http_error", locale=locale, code=code), ok=False)
                if _httpx is not None and isinstance(exc, _httpx.TimeoutException) or 'Timeout' in name:
                    return no_update, _import_status_msg(t("ui.startup.import_ns_timeout", locale=locale), ok=False)
                if (_httpx is not None and isinstance(exc, (_httpx.ConnectError, _httpx.NetworkError))) \
                        or any(k in name for k in ('Connect', 'Network', 'Resolve', 'Proxy')):
                    return no_update, _import_status_msg(t("ui.startup.import_ns_unreachable", locale=locale), ok=False)
                return no_update, _import_status_msg(t("ui.startup.import_failed", locale=locale), ok=False)

            if glucose_df is None or glucose_df.height == 0:
                return no_update, _import_status_msg(t("ui.startup.import_empty", locale=locale), ok=False)

            ns_label = url.strip().rstrip('/')
            info = dict(user_info or {})
            info['uploaded_data_path'] = str(save_path)
            info['uploaded_data_filename'] = ns_label
            info['nightscout_url'] = ns_label
            if token:
                info['nightscout_token'] = token
            return info, _import_status_msg(
                t("ui.startup.import_success", locale=locale, count=glucose_df.height, source='Nightscout'),
                ok=True,
            )

        @app.callback(
            [Output('diabetic-details', 'style'),
             Output('diabetic-type-dropdown', 'value'),
             Output('diabetes-duration-input', 'value')],
            [Input('diabetic-dropdown', 'value')],
            [State('test-me-button', 'n_clicks'),
             State('email-input', 'value')]
        )
        def update_diabetic_details(
            is_diabetic: Optional[bool],
            test_clicks: Optional[int],
            email: Optional[str]
        ) -> tuple[dict[str, str], Any, Any]:
            if is_diabetic is None:
                return {'display': 'none'}, dash.no_update, dash.no_update
            elif is_diabetic:
                # Check if this is from the test button (email will be test email)
                if test_clicks and email and 'test.user@example.com' in str(email):
                    return {'display': 'block'}, 'Type 1', 5
                else:
                    return {'display': 'block'}, dash.no_update, dash.no_update
            else:
                return {'display': 'none'}, 'N/A', 0

        @app.callback(
            [Output('cgm-details', 'style'),
             Output('cgm-duration-input', 'value')],
            [Input('cgm-dropdown', 'value')],
            [State('test-me-button', 'n_clicks'),
             State('email-input', 'value')]
        )
        def update_cgm_details(
            uses_cgm: Optional[bool],
            test_clicks: Optional[int],
            email: Optional[str],
        ) -> tuple[dict[str, str], Any]:
            if uses_cgm is True:
                if test_clicks and email and 'test.user@example.com' in str(email):
                    return {'display': 'block'}, 3
                return {'display': 'block'}, dash.no_update
            return {'display': 'none'}, dash.no_update

        @app.callback(
            [Output('start-button', 'disabled'),
             Output('start-button', 'style'),
             Output('email-required', 'style'),
             Output('age-required', 'style'),
             Output('gender-required', 'style'),
             Output('diabetic-required', 'style'),
             Output('diabetic-type-required', 'style'),
             Output('diabetes-duration-required', 'style'),
             Output('location-required', 'style'),
             Output('format-required', 'style'),
             Output('age-error', 'children'),
             Output('diabetes-duration-error', 'children'),
             Output('cgm-duration-error', 'children'),
             Output('data-usage-error', 'children'),
             Output('startup-missing-fields', 'children'),
             Output('startup-missing-fields', 'style')],
            [Input('email-input', 'value'),
             Input('age-input', 'value'),
             Input('gender-dropdown', 'value'),
             Input('cgm-dropdown', 'value'),
             Input('cgm-duration-input', 'value'),
             Input('format-dropdown', 'value'),
             Input('data-usage-consent', 'value'),
             Input('diabetic-dropdown', 'value'),
             Input('diabetic-type-dropdown', 'value'),
             Input('diabetes-duration-input', 'value'),
             Input('location-input', 'value'),
             Input('user-info-store', 'data'),
             Input('interface-language', 'data'),
             Input('startup-step', 'data')]
        )
        def update_form_validation(
            email: Optional[str],
            age: Optional[int | float],
            gender: Optional[str],
            uses_cgm: Optional[bool],
            cgm_duration: Optional[int | float],
            format_value: Optional[str],
            data_usage_consent: Optional[list[str]],
            is_diabetic: Optional[bool],
            diabetic_type: Optional[str],
            diabetes_duration: Optional[int | float],
            location: Optional[str],
            user_info: Optional[dict[str, Any]],
            interface_language: Optional[str],
            startup_step: Optional[int],
        ) -> tuple[Any, ...]:
            hidden_style = {'display': 'none'}
            required_style = {'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'}

            wants_contact = _wants_contact_from_user_info(user_info)
            current_step = int(startup_step or 0)
            is_mobile = _is_mobile_ua(
                flask_request.headers.get('User-Agent') if has_request_context() else None
            )
            full_validation = validate_startup_form(
                email=email,
                age=age,
                gender=gender,
                format_value=format_value,
                data_usage_consent=data_usage_consent,
                is_diabetic=is_diabetic,
                diabetic_type=diabetic_type,
                diabetes_duration=diabetes_duration,
                location=location,
                uses_cgm=uses_cgm,
                cgm_duration=cgm_duration,
                wants_contact=wants_contact,
                locale=interface_language,
                prior_upload_consent=prior_upload_data_consent(user_info),
            )
            if is_mobile:
                if 1 <= current_step <= 4:
                    hint = _step_hint_children(
                        email=email,
                        age=age,
                        gender=gender,
                        format_value=format_value,
                        data_usage_consent=data_usage_consent,
                        is_diabetic=is_diabetic,
                        diabetic_type=diabetic_type,
                        diabetes_duration=diabetes_duration,
                        location=location,
                        uses_cgm=uses_cgm,
                        cgm_duration=cgm_duration,
                        wants_contact=wants_contact,
                        locale=interface_language,
                        current_step=current_step,
                        prior_upload_consent=prior_upload_data_consent(user_info),
                    )
                elif current_step == 0:
                    hint = ""
                else:
                    hint = full_validation.hint_children()
            else:
                hint = full_validation.hint_children()
            validation = full_validation

            email_asterisk = hidden_style if (not wants_contact or email) else required_style
            age_asterisk = hidden_style if age else required_style
            gender_asterisk = hidden_style if gender else required_style
            format_asterisk = hidden_style if format_value else required_style
            diabetic_asterisk = hidden_style if is_diabetic is not None else required_style
            diabetic_type_asterisk = hidden_style if (not is_diabetic or diabetic_type) else required_style
            diabetes_duration_asterisk = (
                hidden_style if (not is_diabetic or diabetes_duration is not None) else required_style
            )
            location_asterisk = hidden_style if location else required_style

            hint_visible = bool(hint) if isinstance(hint, str) else bool(hint)
            missing_style = {
                **_MISSING_FIELDS_STYLE,
                'display': 'block' if hint_visible else 'none',
                'textAlign': 'center',
            }

            if full_validation.form_complete:
                button_style = {
                    'backgroundColor': '#4CBB17',
                    'color': 'white',
                    'padding': '20px 30px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': '24px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'height': '80px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2',
                }
                return (
                    False,
                    button_style,
                    email_asterisk,
                    age_asterisk,
                    gender_asterisk,
                    diabetic_asterisk,
                    diabetic_type_asterisk,
                    diabetes_duration_asterisk,
                    location_asterisk,
                    format_asterisk,
                    validation.age_error,
                    validation.diabetes_duration_error,
                    validation.cgm_duration_error,
                    validation.data_usage_error,
                    hint,
                    missing_style,
                )

            button_style = {
                'backgroundColor': '#555555',
                'color': 'white',
                'padding': '20px 30px',
                'border': 'none',
                'borderRadius': '5px',
                'fontSize': '24px',
                'cursor': 'not-allowed',
                'width': '100%',
                'height': '80px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'lineHeight': '1.2',
            }
            return (
                True,
                button_style,
                email_asterisk,
                age_asterisk,
                gender_asterisk,
                diabetic_asterisk,
                diabetic_type_asterisk,
                diabetes_duration_asterisk,
                location_asterisk,
                format_asterisk,
                validation.age_error,
                validation.diabetes_duration_error,
                validation.cgm_duration_error,
                validation.data_usage_error,
                hint,
                missing_style,
            )

        # <!-- START INSERTION: Test Me Button Callback -->
        # Callback for "Just Test Me" button
        # Note: diabetic-type-dropdown and diabetes-duration-input are handled
        # by their respective callback when diabetic-dropdown changes
        @app.callback(
            [Output('email-input', 'value'),
             Output('age-input', 'value'),
             Output('gender-dropdown', 'value'),
             Output('cgm-dropdown', 'value'),
             Output('diabetic-dropdown', 'value'),
             Output('location-input', 'value')],
            [Input('test-me-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def fill_form_data(n_clicks: Optional[int]) -> tuple[str, int, str, bool, bool, str]:
            if n_clicks:
                # Fill the form with realistic test data and tick consent checkbox
                # Note: diabetic-type and diabetes-duration will be auto-filled by existing callbacks
                return (
                    'test.user@example.com',  # email
                    28,                       # age
                    'F',                      # gender (Female)
                    True,                     # uses_cgm
                    True,                     # is_diabetic (Yes) - this will trigger diabetic details callback
                    'San Francisco, CA'       # location
                )

            return no_update, no_update, no_update, no_update, no_update, no_update

        # ---- Mobile startup wizard step navigation (StartupPageMobile) ----
        # The `mobile-step-*`, `startup-prev`, `startup-next`, `startup-progress`
        # ids exist ONLY in the mobile builder.  On desktop these buttons are
        # absent, so with prevent_initial_call=True this callback never fires
        # there.  Initial step visibility is baked into the mobile layout, so no
        # initial call is needed.  This callback only toggles the `mobile-step-*`
        # wrappers -- it never touches `cgm-details` / `diabetic-details` /
        # `data-usage-consent-container`, which keep their own conditional
        # callbacks (disjoint Outputs, no races).
        @app.callback(
            [Output('startup-step', 'data'),
             *[Output(f'mobile-step-{i}', 'style') for i in range(WIZARD_STEPS)],
             Output('startup-prev', 'style'),
             Output('startup-next', 'style'),
             Output('startup-progress', 'children')],
            [Input('startup-prev', 'n_clicks'),
             Input('startup-next', 'n_clicks')],
            [State('startup-step', 'data'),
             State('interface-language', 'data')],
            prevent_initial_call=True,
        )
        def navigate_startup_wizard(
            prev_clicks: Optional[int],
            next_clicks: Optional[int],
            current_step: Optional[int],
            interface_language: Optional[str],
        ) -> tuple[Any, ...]:
            trigger = dash.callback_context.triggered_id
            step = int(current_step or 0)
            if trigger == 'startup-next':
                step = min(step + 1, WIZARD_STEPS - 1)
            elif trigger == 'startup-prev':
                step = max(step - 1, 0)
            step_styles = [
                ({'display': 'block'} if i == step else {'display': 'none'})
                for i in range(WIZARD_STEPS)
            ]
            return (
                step,
                *step_styles,
                _wizard_nav_btn_style(visible=step > 0),
                _wizard_nav_btn_style(visible=step < WIZARD_STEPS - 1),
                _wizard_progress_children(step, interface_language),
            )

        # The consent gate must run on initial mobile render so the first Next
        # button starts disabled until the required consent actions are complete.
        @app.callback(
            [Output('startup-next', 'disabled'),
             Output('startup-next', 'className'),
             Output('startup-consent-hint', 'style')],
            [Input('consent-acknowledge', 'value'),
             Input('consent-gdpr', 'value'),
             Input('startup-step', 'data'),
             Input('email-input', 'value'),
             Input('age-input', 'value'),
             Input('gender-dropdown', 'value'),
             Input('cgm-dropdown', 'value'),
             Input('cgm-duration-input', 'value'),
             Input('diabetic-dropdown', 'value'),
             Input('diabetic-type-dropdown', 'value'),
             Input('diabetes-duration-input', 'value'),
             Input('location-input', 'value'),
             Input('format-dropdown', 'value'),
             Input('data-usage-consent', 'value'),
             Input('user-info-store', 'data'),
             Input('interface-language', 'data')],
            prevent_initial_call=False,
        )
        def gate_mobile_consent_step(
            acknowledge_value: Optional[list[str]],
            gdpr_value: Optional[list[str]],
            current_step: Optional[int],
            email: Optional[str],
            age: Optional[int | float],
            gender: Optional[str],
            uses_cgm: Optional[bool],
            cgm_duration: Optional[int | float],
            is_diabetic: Optional[bool],
            diabetic_type: Optional[str],
            diabetes_duration: Optional[int | float],
            location: Optional[str],
            format_value: Optional[str],
            data_usage_consent: Optional[list[str]],
            user_info: Optional[dict[str, Any]],
            interface_language: Optional[str],
        ) -> tuple[bool, str, dict[str, str]]:
            step = int(current_step or 0)
            hint_hidden = {'display': 'none'}
            hint_shown = {
                'display': 'block', 'marginTop': '10px', 'fontSize': '14px',
                'color': '#b45309', 'textAlign': 'center',
            }
            if step == 0:
                blocked = not (
                    bool(acknowledge_value and 'ack' in acknowledge_value) and
                    bool(gdpr_value and 'gdpr' in gdpr_value)
                )
                if blocked:
                    return True, "ui button startup-next-disabled", hint_shown
                return False, "ui blue button", hint_hidden

            if 1 <= step <= 4:
                wants_contact = _wants_contact_from_user_info(user_info)
                step_validation = validate_startup_form(
                    email=email,
                    age=age,
                    gender=gender,
                    format_value=format_value,
                    data_usage_consent=data_usage_consent,
                    is_diabetic=is_diabetic,
                    diabetic_type=diabetic_type,
                    diabetes_duration=diabetes_duration,
                    location=location,
                    uses_cgm=uses_cgm,
                    cgm_duration=cgm_duration,
                    wants_contact=wants_contact,
                    locale=interface_language,
                    wizard_step=step,
                    prior_upload_consent=prior_upload_data_consent(user_info),
                )
                if not step_validation.step_complete:
                    return True, "ui button startup-next-disabled", hint_hidden

            return False, "ui blue button", hint_hidden

        app.clientside_callback(
            """
            function(pathname) {
                if (window.sugarSugarLocationAutocomplete && window.sugarSugarLocationAutocomplete.refresh) {
                    window.sugarSugarLocationAutocomplete.refresh(pathname);
                }
                return Date.now();
            }
            """,
            Output('location-autocomplete-ping', 'data'),
            Input('url', 'pathname'),
            prevent_initial_call=False,
        )


# ---------------------------------------------------------------------------
# Mobile startup wizard (StartupPageMobile)
# ---------------------------------------------------------------------------
# Number of wizard steps.  Must match the number of `mobile-step-{i}` wrappers
# the builder renders and the Outputs in `navigate_startup_wizard`.
WIZARD_STEPS: int = 6

# Mobile field styling: big tap targets, 16px+ to avoid iOS zoom-on-focus.
_M_LABEL = {'fontSize': '18px', 'fontWeight': '800', 'marginBottom': '8px', 'color': '#0f172a', 'display': 'inline-block'}
_M_REQ = {'color': '#d32f2f', 'fontSize': '18px', 'fontWeight': 'bold'}
_M_INPUT = {'width': '100%', 'padding': '14px', 'fontSize': '17px', 'marginBottom': '6px', 'boxSizing': 'border-box'}
_M_DROPDOWN = {'fontSize': '17px', 'marginBottom': '6px'}
_M_ERROR = {'color': '#d32f2f', 'fontSize': '15px', 'marginTop': '2px', 'marginBottom': '10px', 'minHeight': '18px'}


def _wizard_nav_btn_style(*, visible: bool) -> dict[str, str]:
    """Style for a wizard Back/Next button; hidden via visibility to keep layout."""
    return {
        'flex': '1',
        'padding': '16px',
        'fontSize': '18px',
        'fontWeight': '700',
        'borderRadius': '10px',
        'border': 'none',
        'cursor': 'pointer',
        'visibility': 'visible' if visible else 'hidden',
    }


def _wizard_progress_children(step: int, locale: Optional[str]) -> html.Div:
    """Progress indicator: a row of dots plus 'Step X of N' text."""
    dots = [
        html.Span(
            style={
                'display': 'inline-block',
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'margin': '0 4px',
                'backgroundColor': '#2c5282' if i <= step else '#cbd5e1',
            }
        )
        for i in range(WIZARD_STEPS)
    ]
    return html.Div(
        [
            html.Div(dots, style={'textAlign': 'center', 'marginBottom': '6px'}, disable_n_clicks=True),
            html.Div(
                t("ui.startup.wizard_step", locale=locale, current=step + 1, total=WIZARD_STEPS),
                style={'textAlign': 'center', 'color': '#64748b', 'fontSize': '14px'},
                disable_n_clicks=True,
            ),
        ],
        disable_n_clicks=True,
    )


def _m_label(text: str, required_id: Optional[str] = None) -> html.Div:
    """A field label, optionally with a managed required-asterisk span."""
    children: list[Any] = [html.Label(text, style=_M_LABEL)]
    if required_id:
        children.append(html.Span(id=required_id, children=' *', style=_M_REQ))
    return html.Div(children, style={'marginBottom': '8px'}, disable_n_clicks=True)


class StartupPageMobile(html.Div):
    """Portrait-first multi-step wizard for the startup form.

    Renders EVERY input id of the desktop ``StartupPage`` (same ids, same
    persistence) so the existing validation / conditional callbacks in
    ``StartupPage.register_callbacks`` drive it unchanged -- and so no callback
    ever targets a missing component (the "nonexistent object" crash class).
    The fields are grouped into ``mobile-step-{i}`` wrappers shown one at a time;
    ``navigate_startup_wizard`` toggles their ``display``.  Each conditional
    parent lives in the SAME step as its dependents (CGM->duration,
    diabetic->type+duration, format B/C->data-usage-consent) so a hidden step
    never strands a half-revealed cascade.

    Only the first step is visible initially (baked into the layout); the step
    store is memory-backed and resets to 0 on load.  No callbacks are registered
    here.
    """

    def __init__(self, *, locale: str = "en") -> None:
        self.component_id = 'startup-page'
        self._locale = locale

        # --- Step 0: consent (mandatory gate before form fields) ---
        step_consent = [
            html.H2(
                t("ui.landing.patient_consent_form_title", locale=locale),
                style={'fontSize': '22px', 'fontWeight': '800', 'color': '#2c5282', 'marginBottom': '12px'},
                disable_n_clicks=True,
            ),
            html.Div(
                consent_controls_children(locale),
                id='consent-notice-scroll',
                disable_n_clicks=True,
            ),
            html.Div(
                t("ui.landing.next_hint", locale=locale),
                style={'color': '#64748b', 'marginTop': '10px', 'fontSize': '13px'},
                disable_n_clicks=True,
            ),
            # Explains why the Next button is disabled; toggled by
            # gate_mobile_consent_step (hidden once both required boxes are ticked).
            html.Div(
                t("ui.startup.consent_gate_hint", locale=locale),
                id='startup-consent-hint',
                style={'display': 'none'},
                disable_n_clicks=True,
            ),
            dcc.Store(id='consent-scroll-complete', data=False, storage_type=STORAGE_TYPE),
            dcc.Interval(id='consent-scroll-poll', interval=500, n_intervals=0),
        ]

        # --- Step 1: identity (pure required fields, no conditional cascade) ---
        step0 = [
            _m_label(t("ui.startup.email_label", locale=locale), 'email-required'),
            dcc.Input(
                id='email-input', type='email',
                placeholder=t("ui.startup.email_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
            ),
            _m_label(t("ui.startup.age_label", locale=locale), 'age-required'),
            dcc.Input(
                id='age-input', type='number',
                placeholder=t("ui.startup.age_placeholder", locale=locale),
                min=0, max=130, persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
            ),
            html.Div(id='age-error', children='', style=_M_ERROR, disable_n_clicks=True),
            _m_label(t("ui.startup.gender_label", locale=locale), 'gender-required'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': t("ui.startup.gender_male", locale=locale), 'value': 'M'},
                    {'label': t("ui.startup.gender_female", locale=locale), 'value': 'F'},
                    {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'},
                ],
                placeholder=t("ui.startup.gender_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(style={'height': '12px'}, disable_n_clicks=True),
            _m_label(t("ui.startup.location_label", locale=locale), 'location-required'),
            html.Div(
                dcc.Input(
                    id='location-input', type='text',
                    placeholder=t("ui.startup.location_placeholder", locale=locale),
                    persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
                ),
                className='location-autocomplete-host',
            ),
        ]

        # --- Step 2: CGM (cgm-dropdown -> cgm-details/duration) ---
        step1 = [
            _m_label(t("ui.startup.cgm_label", locale=locale)),
            dcc.Dropdown(
                id='cgm-dropdown',
                options=[
                    {'label': t("ui.startup.yes", locale=locale), 'value': True},
                    {'label': t("ui.startup.no", locale=locale), 'value': False},
                ],
                placeholder=t("ui.startup.cgm_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(id='cgm-details', children=[
                html.Div(style={'height': '12px'}, disable_n_clicks=True),
                _m_label(t("ui.startup.cgm_duration_label", locale=locale)),
                dcc.Input(
                    id='cgm-duration-input', type='number',
                    placeholder=t("ui.startup.cgm_duration_placeholder", locale=locale),
                    min=0, max=130, persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
                ),
                html.Div(id='cgm-duration-error', children='', style=_M_ERROR, disable_n_clicks=True),
            ]),
        ]

        # --- Step 3: diabetes (diabetic-dropdown -> type + duration) ---
        step2 = [
            _m_label(t("ui.startup.diabetic_label", locale=locale), 'diabetic-required'),
            dcc.Dropdown(
                id='diabetic-dropdown',
                options=[
                    {'label': t("ui.startup.yes", locale=locale), 'value': True},
                    {'label': t("ui.startup.no", locale=locale), 'value': False},
                ],
                placeholder=t("ui.startup.diabetic_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(id='diabetic-details', children=[
                html.Div(style={'height': '12px'}, disable_n_clicks=True),
                _m_label(t("ui.startup.diabetes_type_label", locale=locale), 'diabetic-type-required'),
                dcc.Dropdown(
                    id='diabetic-type-dropdown',
                    options=[
                        {'label': t("ui.startup.diabetes_type_1", locale=locale), 'value': 'Type 1'},
                        {'label': t("ui.startup.diabetes_type_2", locale=locale), 'value': 'Type 2'},
                        {'label': t("ui.startup.diabetes_type_gestational", locale=locale), 'value': 'Gestational'},
                        {'label': t("ui.startup.diabetes_type_lada", locale=locale), 'value': 'LADA'},
                        {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'},
                    ],
                    placeholder=t("ui.startup.diabetes_type_placeholder", locale=locale),
                    persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
                ),
                html.Div(style={'height': '12px'}, disable_n_clicks=True),
                _m_label(t("ui.startup.diabetes_duration_label", locale=locale), 'diabetes-duration-required'),
                dcc.Input(
                    id='diabetes-duration-input', type='number',
                    placeholder=t("ui.startup.diabetes_duration_placeholder", locale=locale),
                    min=0, max=130, persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
                ),
                html.Div(id='diabetes-duration-error', children='', style=_M_ERROR, disable_n_clicks=True),
            ]),
        ]

        # --- Step 4: format & data-usage consent ---
        step3 = [
            _m_label(t("ui.startup.format_label", locale=locale), 'format-required'),
            dcc.Dropdown(
                id='format-dropdown',
                options=[
                    {'label': t("ui.startup.format_a_label", locale=locale), 'value': 'A'},
                    {'label': t("ui.startup.format_b_label", locale=locale), 'value': 'B', 'disabled': True},
                    {'label': t("ui.startup.format_c_label", locale=locale), 'value': 'C', 'disabled': True},
                ],
                placeholder=t("ui.startup.format_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(
                [
                    html.Small(t("ui.startup.format_help_a", locale=locale)), html.Br(),
                    html.Small(t("ui.startup.format_help_b", locale=locale)), html.Br(),
                    html.Small(t("ui.startup.format_help_c", locale=locale)),
                ],
                style={'color': '#666', 'fontSize': '14px', 'margin': '8px 0 16px', 'lineHeight': '1.4'},
                disable_n_clicks=True,
            ),
            html.Div(
                id='data-usage-consent-container',
                children=[
                    dcc.Checklist(
                        id='data-usage-consent',
                        options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
                        value=[], persistence=True, persistence_type=STORAGE_TYPE, style={'fontSize': '16px'},
                    ),
                    html.Div(id='data-usage-error', style={'marginTop': '8px', 'color': '#d32f2f', 'fontSize': '15px'}, disable_n_clicks=True),
                    *import_controls_children(locale),
                ],
                style={'display': 'none', 'marginBottom': '20px'},
            ),
        ]

        # --- Step 5: contact prefs + submit (start-button driven by validation) ---
        step4 = [
            html.Div(
                [
                    html.H3(t("ui.startup.contact_prefs_title", locale=locale), style={'fontSize': '20px', 'marginBottom': '10px', 'color': '#2c5282'}),
                    html.P(t("ui.startup.contact_prefs_text", locale=locale), style={'fontSize': '15px', 'lineHeight': '1.6', 'margin': '0', 'color': '#555'}),
                ],
                style={'backgroundColor': '#f8f9fa', 'padding': '16px', 'borderRadius': '8px', 'marginBottom': '20px'},
                disable_n_clicks=True,
            ),
            html.Button(
                t("ui.startup.just_test_me", locale=locale),
                id='test-me-button',
                className="ui blue-action button",
                style={
                    'backgroundColor': '#1976D2', 'color': 'white', 'padding': '14px',
                    'border': 'none', 'borderRadius': '8px', 'fontSize': '16px',
                    'cursor': 'pointer', 'width': '100%', 'marginBottom': '14px',
                    'display': 'block' if self._get_debug_mode() else 'none',
                },
            ),
            html.Button(
                t("ui.startup.start_prediction", locale=locale),
                id='start-button',
                className="ui green button",
                disabled=True,
                style={
                    'backgroundColor': '#cccccc', 'color': 'white', 'padding': '18px',
                    'border': 'none', 'borderRadius': '8px', 'fontSize': '20px',
                    'cursor': 'not-allowed', 'width': '100%',
                },
            ),
        ]

        steps = [step_consent, step0, step1, step2, step3, step4]
        step_divs = [
            html.Div(
                children=content,
                id=f'mobile-step-{i}',
                style={'display': 'block' if i == 0 else 'none'},
                disable_n_clicks=True,
            )
            for i, content in enumerate(steps)
        ]

        nav = html.Div(
            [
                html.Button(
                    t("ui.startup.wizard_back", locale=locale),
                    id='startup-prev',
                    className="ui button",
                    style=_wizard_nav_btn_style(visible=False),
                ),
                html.Button(
                    t("ui.startup.wizard_next", locale=locale),
                    id='startup-next',
                    className="ui blue button",
                    disabled=True,
                    style=_wizard_nav_btn_style(visible=True),
                ),
            ],
            style={'display': 'flex', 'gap': '12px', 'marginTop': '20px'},
            disable_n_clicks=True,
        )

        card = html.Div(
            [
                html.H1(
                    t("ui.common.app_title", locale=locale),
                    style={'textAlign': 'center', 'marginBottom': '14px', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2c5282'},
                    disable_n_clicks=True,
                ),
                html.Div(id='startup-progress', children=_wizard_progress_children(0, locale), disable_n_clicks=True),
                html.Div(
                    t("ui.startup.required_fields_note", locale=locale),
                    style={'color': '#666', 'fontSize': '13px', 'fontStyle': 'italic', 'margin': '12px 0', 'textAlign': 'right'},
                    disable_n_clicks=True,
                ),
                *step_divs,
                html.Div(
                    id='startup-missing-fields',
                    children='',
                    style={**_MISSING_FIELDS_STYLE, 'display': 'none', 'textAlign': 'center'},
                    disable_n_clicks=True,
                ),
                nav,
            ],
            style={
                'backgroundColor': 'white', 'borderRadius': '12px',
                'boxShadow': '0 0 10px rgba(0,0,0,0.1)', 'padding': '18px 16px',
            },
            disable_n_clicks=True,
        )

        super().__init__(
            children=[card],
            id=self.component_id,
            style={'padding': '14px 12px 32px', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'},
            disable_n_clicks=True,
        )

    def _get_debug_mode(self) -> bool:
        try:
            from sugar_sugar.config import DEBUG_MODE
            return DEBUG_MODE
        except ImportError:
            return False

 