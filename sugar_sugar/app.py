from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import dash
from dash import dcc, html, Output, Input, State, no_update, dash_table, ctx
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import dash_bootstrap_components as dbc
import os
import sys
import typer
from flask import send_file as flask_send_file, request as flask_request
import uuid
from dotenv import load_dotenv
from eliot import start_action, start_task
from pycomfort.logging import to_nice_file, to_nice_stdout

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Ensure unicode (e.g. Ukrainian) is printable on Windows terminals.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)
to_nice_stdout()
to_nice_file(logs_dir / 'sugar_sugar.json', logs_dir / 'sugar_sugar.log')

from sugar_sugar.i18n import setup_i18n, normalize_locale, t, t_raw
setup_i18n()

from sugar_sugar.data import load_glucose_data
from sugar_sugar.config import (
    DEFAULT_POINTS,
    MIN_POINTS,
    MAX_POINTS,
    DOUBLE_CLICK_THRESHOLD,
    PREDICTION_HOUR_OFFSET,
    DASH_DEBUG,
    DASH_HOST,
    DASH_PORT,
    DEBUG_MODE,
    MAX_ROUNDS,
    STORAGE_TYPE,
)
import sugar_sugar.config as sugar_sugar_config
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.components.predictions import PredictionTableComponent
from sugar_sugar.components.startup import StartupPage
from sugar_sugar.components.landing import LandingPage
from sugar_sugar.components.consent_form import ConsentFormPage
from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.components.header import HeaderComponent
from sugar_sugar.components.ending import EndingPage
from sugar_sugar.components.navbar import NavBar
from sugar_sugar.generic_sources_metadata import load_generic_sources_metadata
from sugar_sugar.contact_info import load_contact_info
from sugar_sugar.static_markdown import static_markdown_autosize_iframe

# Type aliases for clarity
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type

GLUCOSE_MGDL_PER_MMOLL: float = 18.0

FORMAT_ORDER: dict[str, int] = {"C": 0, "B": 1, "A": 2}
GENERIC_SOURCES_METADATA = load_generic_sources_metadata()


def _format_label(format_code: str, *, locale: str) -> str:
    code = str(format_code or "").strip().upper()
    if code == "A":
        return t("ui.startup.format_a_label", locale=locale)
    if code == "B":
        return t("ui.startup.format_b_label", locale=locale)
    if code == "C":
        return t("ui.startup.format_c_label", locale=locale)
    return code

def dataframe_to_store_dict(df_in: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert a Polars DataFrame into a session-store friendly dictionary."""
    return {
        'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'gl': df_in.get_column('gl').to_list(),
        'prediction': df_in.get_column('prediction').to_list(),
        'age': df_in.get_column('age').to_list(),
        'user_id': df_in.get_column('user_id').to_list()
    }


def events_dataframe_to_store_dict(df_in: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert an events Polars DataFrame into a session-store dictionary."""
    return {
        'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'event_type': df_in.get_column('event_type').to_list(),
        'event_subtype': df_in.get_column('event_subtype').to_list(),
        'insulin_value': df_in.get_column('insulin_value').to_list()
    }


def get_random_data_window(
    full_df: pl.DataFrame,
    points: int,
    used_starts: Optional[set[int]] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Get a random window of data from the full DataFrame, avoiding previously
    used start positions when possible.
    """
    import random
    max_start_index = len(full_df) - points
    if max_start_index > 0:
        max_multiple = max_start_index // points
        candidates = [m * points for m in range(max_multiple + 1)]
        if used_starts:
            remaining = [s for s in candidates if s not in used_starts]
            if remaining:
                candidates = remaining
        if len(candidates) > 1 and 0 in candidates:
            candidates = [c for c in candidates if c != 0] or candidates
        random_start = random.choice(candidates)
    else:
        random_start = 0

    windowed_df = full_df.slice(random_start, points)
    return windowed_df, random_start

# Load initial data for session storage.
# When ``_CHART_FILE`` env var is set (by the ``chart`` CLI command), load from
# that file and optionally prefill predictions so the debug reloader preserves
# the state across forks.
_chart_file_env = os.environ.get("_CHART_FILE")
_chart_prefill = os.environ.get("_CHART_PREFILL") == "1"
_chart_noise = float(os.environ.get("_CHART_NOISE", "0.05"))
_chart_points = int(os.environ.get("_CHART_POINTS", str(DEFAULT_POINTS)))
_chart_start_env = os.environ.get("_CHART_START")

if _chart_file_env:
    _init_full_df, _init_events_df = load_glucose_data(Path(_chart_file_env))
else:
    _init_full_df, _init_events_df = load_glucose_data()

_init_full_df = _init_full_df.with_columns(pl.lit(0.0).alias("prediction"))

if _chart_start_env is not None:
    _init_start = max(0, min(int(_chart_start_env), len(_init_full_df) - _chart_points))
    _init_window_df = _init_full_df.slice(_init_start, _chart_points)
else:
    _init_window_df, _init_start = get_random_data_window(_init_full_df, _chart_points)

_init_window_df = _init_window_df.with_columns(pl.lit(0.0).alias("prediction"))

if _chart_prefill:
    import random as _rnd
    _n = len(_init_window_df)
    _visible = _n - PREDICTION_HOUR_OFFSET
    _gl_vals = _init_window_df.get_column("gl").to_list()
    _preds = [0.0] * _n
    for _i in range(_visible, _n):
        _gl = _gl_vals[_i]
        if _gl is not None:
            _preds[_i] = round(_gl * (1.0 + _rnd.uniform(-_chart_noise, _chart_noise)), 1)
    _init_window_df = _init_window_df.with_columns(pl.Series("prediction", _preds, dtype=pl.Float64))
    for _i in range(len(_init_window_df)):
        _pv = _init_window_df.get_column("prediction")[_i]
        if _pv != 0.0:
            _tv = _init_window_df.get_column("time")[_i]
            _init_full_df = _init_full_df.with_columns(
                pl.when(pl.col("time") == _tv).then(_pv).otherwise(pl.col("prediction")).alias("prediction")
            )

example_full_df_store = dataframe_to_store_dict(_init_full_df)
example_initial_df_store = dataframe_to_store_dict(_init_window_df)
example_events_df_store = events_dataframe_to_store_dict(_init_events_df)
example_initial_slider_value = _init_start

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP,
    'https://cdn.jsdelivr.net/npm/fomantic-ui@2.9.3/dist/semantic.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
]

# Dash defaults to width=device-width, which makes phones use a narrow layout viewport and
# breaks chart/drawing. A fixed layout width matches what mobile browsers do for "Desktop
# site": the page is laid out at desktop width and scaled to fit the screen.
_DESKTOP_LAYOUT_VIEWPORT_CSS_PX: int = 1280

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    assets_folder=str(project_root / 'assets'),
    suppress_callback_exceptions=True,
    meta_tags=[
        {
            "name": "viewport",
            "content": (
                f"width={_DESKTOP_LAYOUT_VIEWPORT_CSS_PX}, "
                "maximum-scale=5, user-scalable=yes"
            ),
        },
    ],
)
app.title = "Sugar Sugar - Glucose Prediction Game"

server = app.server

@server.route("/download-study-pdf")
def _download_study_pdf():
    locale = flask_request.args.get("locale", "en")
    pdf_path, _ = _study_design_pdf_info(locale)
    if pdf_path is not None:
        return flask_send_file(str(pdf_path), mimetype="application/pdf", as_attachment=True, download_name=pdf_path.name)
    return "PDF not found", 404

app.clientside_callback(
    """
    function(theme, _navbar, _page, _resume) {
          const activeTheme = theme || 'light';
          document.documentElement.setAttribute('data-bs-theme', activeTheme);
          if (activeTheme === 'dark') {
              document.body.classList.add('dark-mode');
          } else {
              document.body.classList.remove('dark-mode');
          }
          
          // Sync theme to all local iframes (e.g. for markdown)
          const frames = document.querySelectorAll('iframe');
          frames.forEach(f => {
              try {
                  const doc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
                  if (doc && doc.body) {
                      if (activeTheme === 'dark') {
                          doc.body.classList.add('dark-mode');
                      } else {
                          doc.body.classList.remove('dark-mode');
                      }
                  }
              } catch(e) {}
          });
          
          // Update icon if it exists
          const icon = document.getElementById('dark-mode-icon');
          if (icon) {
              icon.className = activeTheme === 'dark' ? 'sun icon' : 'moon icon';
          }
          return window.dash_clientside.no_update;
      }
    """,
    Output('theme-apply-dummy', 'data-theme'),
    [Input('theme-store', 'data'), Input('navbar-container', 'children'), Input('page-content', 'children'), Input('resume-dialog-container', 'children')],
    prevent_initial_call=False
)

app.clientside_callback(
    "function() { return window.navigator.userAgent || ''; }",
    Output('user-agent', 'data'),
    Input('url', 'href'),
    prevent_initial_call=False
)

@app.callback(
    [Output('consent-form-background', 'style'),
     Output('consent-form-card', 'style')],
    Input('theme-store', 'data'),
    State('url', 'pathname'),
    prevent_initial_call=True
)
def update_consent_form_background(theme: str, pathname: str):
    if pathname != '/consent-form':
        raise PreventUpdate
    background = (
        "linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 35%, #3a3a3a 100%)"
        if theme == "dark"
        else "linear-gradient(135deg, #eff6ff 0%, #f8fafc 35%, #fff7ed 100%)"
    )
    border_color = "rgba(255, 255, 255, 0.2)" if theme == "dark" else "rgba(15, 23, 42, 0.10)"
    card_bg = "transparent"
    return {
        "height": "100vh",
        "overflow": "hidden",
        "padding": "28px 18px",
        "background": background,
    }, {
        "borderRadius": "14px",
        "border": f"1px solid {border_color}",
        "backgroundColor": card_bg,
    }

app.clientside_callback(
    """
    function(n_clicks, current_theme) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        return current_theme === 'dark' ? 'light' : 'dark';
    }
    """,
    Output('theme-store', 'data'),
    Input('dark-mode-toggle', 'n_clicks'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_intervals, alreadyComplete) {
        // Guard: once complete, keep it disabled and stay complete.
        if (alreadyComplete) {
            return [true, true];
        }
        var el = document.getElementById('consent-notice-scroll');
        // Fix (original): previously this returned [false, false] when the element
        // was absent, writing `false` to consent-scroll-complete on every tick even
        // though the value hadn't changed. Because dcc.Store triggers downstream
        // server-side callbacks on every write (regardless of value equality), this
        // caused update_continue_button to POST at the full interval rate indefinitely.
        //
        // Fix (this revision): the previous attempt used `return no_update` (scalar)
        // for a multi-output callback. Dash's JS runtime does NOT treat a bare scalar
        // no_update as "suppress all outputs" for multi-output callbacks — the correct
        // API is `throw window.dash_clientside.PreventUpdate`, which is the JS
        // equivalent of Python's `raise PreventUpdate`. Background-tab timer throttling
        // (browsers slow setInterval to ~1-4s for inactive tabs) meant this kept
        // reaching the server at ~1 POST/2 s even after the apparent fix.
        if (!el) {
            throw window.dash_clientside.PreventUpdate;
        }
        var epsilon = 4;
        var atEnd = (el.scrollTop + el.clientHeight) >= (el.scrollHeight - epsilon);
        if (!atEnd) {
            throw window.dash_clientside.PreventUpdate;
        }
        return [true, true];
    }
    """,
    [
        Output("consent-scroll-complete", "data"),
        Output("consent-scroll-poll", "disabled"),
    ],
    Input("consent-scroll-poll", "n_intervals"),
    State("consent-scroll-complete", "data"),
    prevent_initial_call=False,
)



# Create component instances
glucose_chart = GlucoseChart(id='glucose-graph', hide_last_hour=True)  # Hide last hour in prediction page
prediction_table = PredictionTableComponent()
metrics_component = MetricsComponent()
submit_component = SubmitComponent()
header_component = HeaderComponent(show_time_slider=False, initial_slider_value=example_initial_slider_value)
# startup_page will be created in main() after debug mode is set
startup_page = None  # Will be initialized in main()
landing_page = None  # Will be initialized in main()
ending_page = EndingPage()

# When _CHART_MODE env var is set, pre-populate stores for the prediction page
# so the debug reloader preserves the state across forks.
_is_chart_mode = os.environ.get("_CHART_MODE") == "1"
_clean_storage = os.environ.get("_CLEAN_STORAGE") == "1"
_chart_source = os.environ.get("_CHART_SOURCE", "example.csv")
_chart_is_example = _chart_file_env is None
_chart_unit = os.environ.get("_CHART_UNIT", "mg/dL")
_chart_locale = os.environ.get("_CHART_LOCALE", "en")

if _is_chart_mode:
    _chart_user_info: Optional[Dict[str, Any]] = {
        "study_id": str(uuid.uuid4()),
        "email": "dev@chart.local",
        "age": 28,
        "gender": "F",
        "uses_cgm": True,
        "cgm_duration_years": 1,
        "format": "A",
        "run_format": "A",
        "consent_use_uploaded_data": False,
        "diabetic": True,
        "diabetic_type": "Type 1",
        "diabetes_duration": 5,
        "location": "Dev Machine",
        "rounds": [],
        "max_rounds": MAX_ROUNDS,
        "current_round_number": 1,
        "statistics_saved": False,
        "is_example_data": _chart_is_example,
        "data_source_name": _chart_source,
        "consent_play_only": True,
        "consent_participate_in_study": False,
        "consent_receive_results_later": False,
        "consent_keep_up_to_date": False,
        "consent_no_selection": False,
        "consent_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
else:
    _chart_user_info = None

app.layout = html.Div([
    dcc.Location(id='url', refresh=False, **({'pathname': '/prediction'} if _is_chart_mode else {})),
    dcc.Store(id='user-info-store', data=_chart_user_info, storage_type=STORAGE_TYPE),
    dcc.Store(id='last-click-time', data=0),
    dcc.Store(id='consent-scroll-request', data=0),
    dcc.Store(id='current-window-df', data=example_initial_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='full-df', data=example_full_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='events-df', data=example_events_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='is-example-data', data=_chart_is_example, storage_type=STORAGE_TYPE),
    dcc.Store(id='data-source-name', data=_chart_source if _is_chart_mode else "example.csv", storage_type=STORAGE_TYPE),
    dcc.Store(id='randomization-initialized', data=_is_chart_mode, storage_type=STORAGE_TYPE),
    dcc.Store(id='glucose-chart-mode', data={'hide_last_hour': True}, storage_type=STORAGE_TYPE),
    dcc.Store(id='glucose-unit', data=_chart_unit if _is_chart_mode else 'mg/dL', storage_type=STORAGE_TYPE),
    dcc.Store(id='interface-language', data=_chart_locale if _is_chart_mode else 'en', storage_type=STORAGE_TYPE),
    dcc.Store(id='user-agent', data=None, storage_type=STORAGE_TYPE),
    dcc.Store(id='initial-slider-value', data=example_initial_slider_value, storage_type=STORAGE_TYPE),
    dcc.Store(id='theme-store', data='light', storage_type=STORAGE_TYPE),
    # Tracks the last page the user reached so we can restore it on reload (local storage only).
    dcc.Store(id='last-visited-page', data=None, storage_type=STORAGE_TYPE),
    # One-shot flag: prevents the restore-redirect from firing more than once per session.
    dcc.Store(id='page-restore-done', data=False, storage_type='memory'),
    # Tracks whether the user has already interacted with the app in this browser tab.
    # Uses sessionStorage: survives full page reloads (navbar clicks) but clears when
    # the tab is closed.  restore_page_on_load uses this to decide whether to show the
    # resume dialog (fresh session) or silently redirect (tab-switch-back).
    dcc.Store(id='session-active', data=False, storage_type='session'),
    # Set to True by --clean flag; consumed once by a clientside callback to wipe localStorage.
    dcc.Store(id='clean-storage-flag', data=_clean_storage, storage_type='memory'),
    # Holds the target page for the resume dialog; set by restore_page_on_load.
    dcc.Store(id='resume-dialog-target', data=None, storage_type='memory'),
    dcc.ConfirmDialog(
        id='confirm-home-clear-session',
        message='Attention! This action will delete your current session data!',
    ),
    dcc.ConfirmDialog(
        id='confirm-ending-finish-exit',
        message='Attention! This action will delete your current session data!',
    ),

    html.Div(id='mobile-warning', style={'margin': '0'}),
    html.Div(id='theme-apply-dummy', style={'display': 'none'}),
    html.Div(id='scroll-to-top-trigger', style={'display': 'none'}),

    html.Div(id='resume-dialog-container', children=[], disable_n_clicks=True),

    html.Div(id='navbar-container', children=[], disable_n_clicks=True),
    
    html.Div(id='page-content', children=[], disable_n_clicks=True)
])


@app.callback(
    Output('glucose-unit', 'data', allow_duplicate=True),
    [Input('url', 'pathname')],
    prevent_initial_call='initial_duplicate'
)
def reset_glucose_unit_on_start_page(pathname: Optional[str]) -> str:
    """Always reset units to mg/dL on the start page to avoid carry-over between runs/users."""
    if pathname in ('/', '/startup'):
        return 'mg/dL'
    raise PreventUpdate


@app.callback(
    Output('interface-language', 'data'),
    [Input('lang-en', 'n_clicks'),
     Input('lang-de', 'n_clicks'),
     Input('lang-uk', 'n_clicks'),
     Input('lang-ro', 'n_clicks'),
     Input('lang-ru', 'n_clicks'),
     Input('lang-zh', 'n_clicks'),
     Input('lang-fr', 'n_clicks'),
     Input('lang-es', 'n_clicks')],
    [State('interface-language', 'data')],
    prevent_initial_call=True
)
def set_interface_language(
    n_en: Optional[int],
    n_de: Optional[int],
    n_uk: Optional[int],
    n_ro: Optional[int],
    n_ru: Optional[int],
    n_zh: Optional[int],
    n_fr: Optional[int],
    n_es: Optional[int],
    current_language: Optional[str],
) -> str:
    """Set the interface language from navbar flag buttons."""
    triggered = ctx.triggered_id
    if not triggered:
        raise PreventUpdate
    _clicks = {
        'lang-en': n_en, 'lang-de': n_de, 'lang-uk': n_uk, 'lang-ro': n_ro,
        'lang-ru': n_ru, 'lang-zh': n_zh, 'lang-fr': n_fr, 'lang-es': n_es,
    }
    if not _clicks.get(triggered):
        raise PreventUpdate
    _lang_map = {
        'lang-en': 'en', 'lang-de': 'de', 'lang-uk': 'uk', 'lang-ro': 'ro',
        'lang-ru': 'ru', 'lang-zh': 'zh', 'lang-fr': 'fr', 'lang-es': 'es',
    }
    new_lang = _lang_map.get(triggered)
    if not new_lang or new_lang == current_language:
        raise PreventUpdate
    return new_lang


@app.callback(
    [
        Output('prediction-data-usage-consent', 'style'),
        Output('prediction-data-usage-consent', 'options'),
        Output('prediction-data-usage-consent', 'value'),
        Output('prediction-data-usage-consent-status', 'children'),
    ],
    [Input('user-info-store', 'data'),
     Input('url', 'pathname'),
     Input('interface-language', 'data')],
    [State('prediction-data-usage-consent', 'value')],
    prevent_initial_call=False,
)
def update_prediction_uploaded_data_consent_ui(
    user_info: Optional[Dict[str, Any]],
    pathname: Optional[str],
    interface_language: Optional[str],
    current_value: Optional[list[str]],
) -> Tuple[Dict[str, str], list[dict[str, Any]], list[str], Optional[html.Div]]:
    if pathname != '/prediction':
        raise PreventUpdate
    if not user_info:
        raise PreventUpdate

    fmt = str(user_info.get("format") or "A")
    if fmt not in ("B", "C"):
        return {'display': 'none'}, [], [], None

    locale = normalize_locale(interface_language)
    base_label = t("ui.startup.data_usage_consent_label", locale=locale)
    if bool(user_info.get("consent_use_uploaded_data", False)):
        return (
            {'display': 'block', 'fontSize': '16px'},
            [{'label': base_label, 'value': 'agree', 'disabled': True}],
            ['agree'],
            dbc.Alert(
                t("ui.prediction.upload_consent_recorded", locale=locale),
                color="success",
                style={"marginTop": "8px"},
            ),
        )

    return (
        {'display': 'block', 'fontSize': '16px'},
        [{'label': base_label, 'value': 'agree', 'disabled': False}],
        list(current_value or []),
        dbc.Alert(
            t("ui.startup.data_usage_consent_required", locale=locale),
            color="warning",
            style={"marginTop": "8px"},
        ),
    )


_STATEFUL_PAGES = frozenset({'/prediction', '/ending'})


@app.callback(
    [Output('page-content', 'children', allow_duplicate=True),
     Output('mobile-warning', 'children', allow_duplicate=True),
     Output('navbar-container', 'children', allow_duplicate=True)],
    [Input('interface-language', 'data'),
     Input('theme-store', 'data')],
    [State('url', 'pathname'),
     State('user-info-store', 'data'),
     State('user-agent', 'data'),
     State('full-df', 'data'),
     State('glucose-unit', 'data')],
    prevent_initial_call=True,
)
def update_on_language_change(
    interface_language: Optional[str],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    user_agent: Optional[str],
    full_df_data: Optional[Dict],
    glucose_unit: Optional[str],
    theme: Optional[str],
) -> tuple:
    """Re-render page content and navbar when language changes.

    Pages with interactive state (prediction chart, ending) only get
    a navbar refresh -- page content is left untouched via per-element callbacks.
    """
    locale = normalize_locale(interface_language)
    theme = theme or 'light'
    navbar = NavBar(locale=locale, current_page=pathname or "/")

    if pathname in _STATEFUL_PAGES:
        return no_update, no_update, navbar

    warning_content = render_mobile_warning(user_agent, locale=locale)
    if pathname == '/final':
        if user_info:
            return create_final_layout(full_df_data, user_info, glucose_unit, locale=locale), warning_content, navbar
        return no_update, no_update, navbar
    if pathname == "/consent-form":
        return ConsentFormPage(locale=locale, theme=theme), warning_content, navbar
    if pathname == '/startup':
        return StartupPage(locale=locale, theme=theme), warning_content, navbar
    if pathname == '/about':
        return create_about_page(locale=locale), warning_content, navbar
    if pathname == '/contact':
        return create_contact_page(locale=locale), warning_content, navbar
    if pathname == '/demo':
        return create_demo_page(locale=locale), warning_content, navbar
    if pathname == '/faq':
        return create_faq_page(locale=locale), warning_content, navbar
    # Landing page
    return LandingPage(locale=locale), warning_content, navbar


@app.callback(
    [Output('header-app-title', 'children'),
     Output('header-description', 'children'),
     Output('header-how-to-play', 'children'),
     Output('header-data-source-label', 'children'),
     Output('header-upload-prompt', 'children'),
     Output('use-example-data-button', 'children'),
     Output('header-time-window-label', 'children'),
     Output('prediction-units-label', 'children'),
     Output('prediction-consent-label', 'children'),
     Output('submit-button', 'children'),
     Output('finish-study-button', 'children')],
    [Input('interface-language', 'data')],
    [State('url', 'pathname')],
    prevent_initial_call=True,
)
def update_prediction_text_on_language_change(
    interface_language: Optional[str],
    pathname: Optional[str],
) -> tuple:
    """Update translatable text on the prediction page when language changes mid-game."""
    if pathname != '/prediction':
        raise PreventUpdate

    locale = normalize_locale(interface_language)
    return (
        t("ui.common.app_title", locale=locale),
        [
            t("ui.header.description_1", locale=locale) + " ",
            html.Br(),
            t("ui.header.description_2", locale=locale) + " ",
            t("ui.header.description_3", locale=locale),
        ],
        [
            html.Strong(t("ui.header.how_to_play", locale=locale)),
            html.Br(),
            t("ui.header.how_to_play_1", locale=locale),
            html.Br(),
            t("ui.header.how_to_play_2", locale=locale),
            html.Br(),
            t("ui.header.how_to_play_3", locale=locale),
        ],
        t("ui.header.current_data_source", locale=locale),
        [
            t("ui.header.upload_prompt_1", locale=locale),
            html.A(t("ui.header.upload_prompt_2", locale=locale)),
        ],
        t("ui.header.use_example_data", locale=locale),
        t("ui.header.time_window_label", locale=locale),
        t("ui.prediction.units_label", locale=locale),
        t("ui.startup.data_usage_consent_label", locale=locale),
        t("ui.submit.submit", locale=locale),
        t("ui.common.finish_exit", locale=locale),
    )


@app.callback(
    [Output('ending-title', 'children'),
     Output('ending-disclaimer-line1', 'children'),
     Output('ending-disclaimer-line2', 'children'),
     Output('ending-disclaimer-line3', 'children'),
     Output('ending-round-info', 'children'),
     Output('ending-round-motivation', 'children'),
     Output('ending-units-line', 'children'),
     Output('ending-graph-explanation', 'children'),
     Output('ending-prediction-results-title', 'children'),
     Output('ending-prediction-table', 'data'),
     Output('ending-prediction-table', 'columns'),
     Output('ending-metrics-container', 'children'),
     Output('ending-local-storage-note', 'children'),
     Output('finish-study-button-ending', 'children'),
     Output('next-round-button', 'children'),
     Output('ending-switch-format-title', 'children'),
     Output('switch-format-c', 'children'),
     Output('switch-format-a', 'children'),
     Output('switch-format-b', 'children')],
    [Input('interface-language', 'data')],
    [State('url', 'pathname'),
     State('user-info-store', 'data'),
     State('glucose-unit', 'data')],
    prevent_initial_call=True,
)
def update_ending_text_on_language_change(
    interface_language: Optional[str],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    glucose_unit: Optional[str],
) -> tuple:
    """Update translatable text on the ending page when language changes."""
    if pathname != '/ending':
        raise PreventUpdate

    locale = normalize_locale(interface_language)
    unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'

    rounds_played = len(user_info.get('rounds') or []) if user_info else 0
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS) if user_info else MAX_ROUNDS
    current_round_number = int(user_info.get('current_round_number') or rounds_played) if user_info else rounds_played
    is_last_round = current_round_number >= max_rounds

    metric_label_map: dict[str, str] = {
        "Actual Glucose": t("ui.table.actual_glucose", locale=locale),
        "Predicted": t("ui.table.predicted", locale=locale),
        "Absolute Error": t("ui.table.absolute_error", locale=locale),
        "Relative Error (%)": t("ui.table.relative_error_pct", locale=locale, pct="%"),
    }

    table_data: list[dict[str, str]] = no_update
    table_columns: list[dict[str, str]] = no_update
    if user_info and 'prediction_table_data' in user_info:
        raw_table = _convert_table_data_units(user_info['prediction_table_data'], unit)
        table_data = []
        for row in raw_table:
            new_row = dict(row)
            new_row["metric"] = metric_label_map.get(str(row.get("metric", "")), str(row.get("metric", "")))
            table_data.append(new_row)
        table_columns = [{'name': t("ui.table.metric_header", locale=locale), 'id': 'metric'}] + [
            {'name': f'T{i}', 'id': f't{i}', 'type': 'text'}
            for i in range(len(raw_table[0]) - 1)
            if raw_table and raw_table[1].get(f't{i}', '-') != '-'
        ]

    metrics_display: Any = no_update
    if user_info and 'prediction_table_data' in user_info:
        raw_table = _convert_table_data_units(user_info['prediction_table_data'], unit)
        metrics_comp = MetricsComponent()
        stored_metrics = metrics_comp._calculate_metrics_from_table_data(raw_table) if len(raw_table) >= 2 else None
        metrics_display = MetricsComponent.create_ending_metrics_display(stored_metrics, locale=locale) if stored_metrics else [
            html.H3(t("ui.metrics.title_accuracy_metrics", locale=locale), style={'textAlign': 'center'}),
            html.Div(
                t("ui.metrics.no_metrics_available", locale=locale),
                style={'color': 'gray', 'fontStyle': 'italic', 'fontSize': '16px', 'padding': '10px', 'textAlign': 'center'}
            )
        ]

    finish_button_text = t("ui.ending.view_complete_analysis", locale=locale) if is_last_round else t("ui.common.finish_exit", locale=locale)

    return (
        t("ui.ending.title", locale=locale),
        t("ui.results_disclaimer.line1", locale=locale),
        t("ui.results_disclaimer.line2", locale=locale),
        t("ui.results_disclaimer.line3", locale=locale),
        t("ui.common.round_of", locale=locale, current=current_round_number, total=max_rounds),
        t("ui.ending.round_motivation", locale=locale, total=max_rounds, min_useful=max(1, max_rounds // 2)),
        t("ui.ending.units_line", locale=locale, unit=unit),
        t("ui.ending.graph_explanation", locale=locale),
        t("ui.ending.prediction_results", locale=locale),
        table_data,
        table_columns,
        metrics_display,
        t("ui.ending.local_storage_note", locale=locale),
        finish_button_text,
        t("ui.ending.next_round", locale=locale),
        t("ui.switch_format.title", locale=locale),
        t("ui.switch_format.try_c", locale=locale),
        t("ui.switch_format.try_a", locale=locale),
        t("ui.switch_format.try_b", locale=locale),
    )


@app.callback(
    [Output('page-content', 'children'),
     Output('mobile-warning', 'children'),
     Output('navbar-container', 'children')],
    [Input('url', 'pathname')],
    [State('interface-language', 'data'),
     State('user-info-store', 'data'),
     State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('events-df', 'data'),
     State('glucose-unit', 'data'),
     State('user-agent', 'data'),
     State('theme-store', 'data')],
    prevent_initial_call=False
)
def display_page(
    pathname: Optional[str],
    interface_language: Optional[str],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict],
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
    glucose_unit: Optional[str],
    user_agent: Optional[str],
    theme: Optional[str],
) -> tuple[html.Div, Optional[html.Div], html.Div]:
    has_ptd = bool(user_info and 'prediction_table_data' in user_info) if user_info else False
    has_full = bool(full_df_data)
    print(f"DEBUG display_page: pathname={pathname} has_user_info={user_info is not None} has_prediction_table_data={has_ptd} has_full_df={has_full}")
    locale = normalize_locale(interface_language)
    theme = theme or 'light'
    navbar = NavBar(locale=locale, current_page=pathname or "/")
    
    with start_action(action_type=u"display_page", pathname=pathname, locale=locale):
        warning_content = render_mobile_warning(user_agent, locale=locale)
        if pathname == "/consent-form":
            return ConsentFormPage(locale=locale, theme=theme), warning_content, navbar
        if pathname == '/prediction' and user_info:
            format_value = str(user_info.get("format") or "A")
            return create_prediction_layout(locale=locale, format_value=format_value, user_info=user_info), warning_content, navbar
        if pathname == '/startup':
            return (StartupPage(locale=locale, theme=theme), warning_content, navbar)
        if pathname == '/ending':
            # Check if we have the required data for ending page
            if not full_df_data or not user_info or 'prediction_table_data' not in user_info:
                return html.Div([
                    html.H2(t("ui.session_expired.title", locale=locale), style={'textAlign': 'center', 'marginTop': '50px'}),
                    html.P(t("ui.session_expired.text", locale=locale), style={'textAlign': 'center', 'marginBottom': '30px'}),
                    html.Div([
                        html.A(
                            t("ui.common.go_to_start", locale=locale),
                            href="/",
                            style={
                                'backgroundColor': '#007bff',
                                'color': 'white',
                                'padding': '15px 30px',
                                'textDecoration': 'none',
                                'borderRadius': '5px',
                                'fontSize': '18px'
                            }
                        )
                    ], style={'textAlign': 'center'})
                ]), warning_content, navbar
            return create_ending_layout(full_df_data, current_df_data, events_df_data, user_info, glucose_unit, locale=locale), warning_content, navbar
        if pathname == '/final':
            if not user_info:
                return html.Div([
                    html.H2(t("ui.session_expired.title", locale=locale), style={'textAlign': 'center', 'marginTop': '50px'}),
                    html.P(t("ui.session_expired.text", locale=locale), style={'textAlign': 'center', 'marginBottom': '30px'}),
                    html.Div([
                        html.A(
                            t("ui.common.go_to_start", locale=locale),
                            href="/",
                            style={
                                'backgroundColor': '#007bff',
                                'color': 'white',
                                'padding': '15px 30px',
                                'textDecoration': 'none',
                                'borderRadius': '5px',
                                'fontSize': '18px'
                            }
                        )
                    ], style={'textAlign': 'center'})
                ]), warning_content, navbar
            return create_final_layout(full_df_data, user_info, glucose_unit, locale=locale), warning_content, navbar
        if pathname == '/about':
            return create_about_page(locale=locale), warning_content, navbar
        if pathname == '/contact':
            return create_contact_page(locale=locale), warning_content, navbar
        if pathname == '/demo':
            return create_demo_page(locale=locale), warning_content, navbar
        if pathname == '/faq':
            return create_faq_page(locale=locale), warning_content, navbar
        # Default route: landing page
        return (LandingPage(locale=locale), warning_content, navbar)

from dash import html


def create_info_page(*, locale: str, title: str, body: str) -> html.Div:
    return html.Div(
        [
            html.H1(title, disable_n_clicks=True),
            html.Div(body, style={"marginBottom": "14px"}, disable_n_clicks=True),
        ],
        className="info-page",
        disable_n_clicks=True,
    )


def create_faq_page(*, locale: str) -> html.Div:
    sections: list[Any] = t_raw("ui.faq.sections", locale=locale)
    section_divs: list[Any] = []
    for section in sections:
        items: list[Any] = []
        for item in section.get("items", []):
            items.append(
                html.Div(
                    [
                        html.H3(
                            item["q"],
                            style={"marginBottom": "6px"},
                            disable_n_clicks=True,
                        ),
                        dcc.Markdown(
                            item["a"],
                            link_target="_blank",
                            style={"marginBottom": "0"},
                        ),
                    ],
                    className="ui segment",
                    style={"marginBottom": "8px"},
                    disable_n_clicks=True,
                )
            )
        section_divs.append(
            html.Div(
                [
                    html.H2(
                        section["title"],
                        style={"marginBottom": "12px", "marginTop": "24px"},
                        disable_n_clicks=True,
                    ),
                    html.Div(items, disable_n_clicks=True),
                ],
                disable_n_clicks=True,
            )
        )
    return html.Div(
        [
            html.H1(t("ui.faq.title", locale=locale), disable_n_clicks=True),
            html.Div(section_divs, disable_n_clicks=True),
        ],
        className="info-page",
        disable_n_clicks=True,
    )

@lru_cache(maxsize=4)
def _study_design_markdown(locale: str) -> str:
    loc = normalize_locale(locale)
    base = project_root / "data" / "input" / "study_design" / "The study - technical Guidebook.md"

    candidates: list[Path] = []
    if base.exists():
        candidates.append(base.with_name(f"{base.stem}.{loc}{base.suffix}"))
        candidates.append(base.with_name(f"{base.stem}_{loc}{base.suffix}"))
        candidates.append(base)

    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return ""


def _study_design_pdf_info(locale: str) -> tuple[Path | None, bool]:
    """Return (pdf_path, is_original_english).

    *is_original_english* is True when the PDF found is the base (English)
    file and the requested locale is not English — i.e. no locale-specific
    PDF exists.
    """
    loc = normalize_locale(locale)
    base_dir = project_root / "data" / "input" / "study_design"
    localized = base_dir / f"study_design.{loc}.pdf"
    if localized.exists():
        return localized, False
    base = base_dir / "study_design.pdf"
    if base.exists():
        return base, loc != "en"
    return None, False


def create_about_page(*, locale: str) -> html.Div:
    study_md = _study_design_markdown(locale)
    children: list[Any] = [
        html.H1(t("ui.about.title", locale=locale)),
        html.Div(t("ui.about.body", locale=locale), style={"marginBottom": "14px"}),
        html.Div(
            html.A(
                t("ui.about.github_link_label", locale=locale),
                href="https://github.com/GlucoseDAO/sugar-sugar",
                target="_blank",
                rel="noopener noreferrer",
                style={"fontWeight": "700"},
            ),
            style={"marginBottom": "10px"},
        ),
    ]
    if study_md:
        study_header_children: list[Any] = [
            html.H2(
                t("ui.about.study_design_title", locale=locale),
                style={"marginBottom": "16px"},
            ),
        ]
        pdf_path, pdf_is_english_original = _study_design_pdf_info(locale)
        if pdf_path is not None:
            pdf_children: list[Any] = [
                html.A(
                    t("ui.about.download_pdf_label", locale=locale),
                    href=f"/download-study-pdf?locale={normalize_locale(locale)}",
                    target="_blank",
                    rel="noopener noreferrer",
                    className="ui blue basic button",
                ),
            ]
            if pdf_is_english_original:
                pdf_children.append(
                    html.Span(
                        t("ui.about.pdf_original_english_note", locale=locale),
                        style={
                            "marginLeft": "10px",
                            "color": "#64748b",
                            "fontSize": "14px",
                            "fontStyle": "italic",
                        },
                    )
                )
            study_header_children.append(
                html.Div(
                    pdf_children,
                    style={
                        "marginBottom": "16px",
                        "display": "flex",
                        "alignItems": "center",
                        "flexWrap": "wrap",
                        "gap": "4px",
                    },
                    disable_n_clicks=True,
                )
            )

        if pdf_is_english_original:
            study_header_children.append(
                html.Div(
                    t("ui.about.translation_note", locale=locale),
                    style={
                        "color": "#64748b",
                        "fontSize": "14px",
                        "fontStyle": "italic",
                        "marginBottom": "12px",
                    },
                    disable_n_clicks=True,
                )
            )

        children.extend(
            [
                html.Hr(style={"margin": "24px 0"}),
                *study_header_children,
                static_markdown_autosize_iframe(
                    study_md,
                    title=t("ui.about.study_design_title", locale=locale),
                ),
            ]
        )
    return html.Div(children, className="info-page", disable_n_clicks=True)


def create_contact_page(*, locale: str) -> html.Div:
    info = load_contact_info()
    page_children: list[Any] = [
        html.H1(t("ui.contact.title", locale=locale)),
        html.Div(t("ui.contact.body", locale=locale), style={"marginBottom": "14px"}),
    ]

    def table_style() -> dict[str, Any]:
        return {
            "width": "100%",
            "borderCollapse": "collapse",
            "background": "rgba(255,255,255,0.75)",
        }

    def th_style() -> dict[str, Any]:
        return {"textAlign": "left", "padding": "8px 10px", "borderBottom": "1px solid rgba(15, 23, 42, 0.12)"}

    def td_style() -> dict[str, Any]:
        return {"textAlign": "left", "padding": "8px 10px", "verticalAlign": "top", "borderBottom": "1px solid rgba(15, 23, 42, 0.06)"}

    if info.study_contacts:
        page_children.extend(
            [
                html.H2(t("ui.contact.study_contacts_title", locale=locale)),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(t("ui.contact.col_name", locale=locale), style=th_style()),
                                    html.Th(t("ui.contact.col_email", locale=locale), style=th_style()),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(item.name, style=td_style()),
                                        html.Td(
                                            html.A(item.email, href=f"mailto:{item.email}"),
                                            style=td_style(),
                                        ),
                                    ]
                                )
                                for item in info.study_contacts
                            ]
                        ),
                    ],
                    style=table_style(),
                ),
                html.Hr(style={"margin": "18px 0"}),
            ]
        )

    if info.general_email:
        page_children.extend(
            [
                html.H2(t("ui.contact.general_email_title", locale=locale)),
                html.Div(
                    html.A(info.general_email, href=f"mailto:{info.general_email}", style={"fontWeight": "700"}),
                    style={"marginBottom": "18px"},
                ),
            ]
        )

    if info.social_links:
        page_children.append(html.H2(t("ui.contact.social_title", locale=locale)))
        page_children.append(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(t("ui.contact.col_platform", locale=locale), style=th_style()),
                                html.Th(t("ui.contact.col_link", locale=locale), style=th_style()),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(item.platform, style=td_style()),
                                    html.Td(
                                        html.A(item.label, href=item.url, target="_blank", rel="noopener noreferrer"),
                                        style=td_style(),
                                    ),
                                ]
                            )
                            for item in info.social_links
                        ]
                    ),
                ],
                style=table_style(),
            )
        )
        page_children.append(html.Hr(style={"margin": "18px 0"}))

    if info.platform_links:
        page_children.append(html.H2(t("ui.contact.platforms_title", locale=locale)))
        page_children.append(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(t("ui.contact.col_platform", locale=locale), style=th_style()),
                                html.Th(t("ui.contact.col_link", locale=locale), style=th_style()),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(item.platform, style=td_style()),
                                    html.Td(
                                        html.A(item.label, href=item.url, target="_blank", rel="noopener noreferrer"),
                                        style=td_style(),
                                    ),
                                ]
                            )
                            for item in info.platform_links
                        ]
                    ),
                ],
                style=table_style(),
            )
        )
        page_children.append(html.Hr(style={"margin": "18px 0"}))

    if info.linkedin_contacts:
        page_children.append(html.H2(t("ui.contact.linkedin_title", locale=locale)))
        page_children.append(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(t("ui.contact.col_name", locale=locale), style=th_style()),
                                html.Th(t("ui.contact.col_role", locale=locale), style=th_style()),
                                html.Th(t("ui.contact.col_link", locale=locale), style=th_style()),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(item.name, style=td_style()),
                                    html.Td(item.role, style=td_style()),
                                    html.Td(
                                        html.A(
                                            t("ui.contact.open_linkedin", locale=locale),
                                            href=item.url,
                                            target="_blank",
                                            rel="noopener noreferrer",
                                        ),
                                        style=td_style(),
                                    ),
                                ]
                            )
                            for item in info.linkedin_contacts
                        ]
                    ),
                ],
                style=table_style(),
            )
        )

    return html.Div(page_children, className="info-page", disable_n_clicks=True)


def create_demo_page(*, locale: str) -> html.Div:
    return create_info_page(
        locale=locale,
        title=t('ui.demo.title', locale=locale),
        body=t('ui.demo.body', locale=locale),
    )


def create_prediction_layout(*, locale: str, format_value: str, user_info: Dict[str, Any]) -> html.Div:
    """Create the prediction page layout"""
    show_upload = format_value in ("B", "C")
    consent_given = bool(user_info.get("consent_use_uploaded_data", False))
    consent_value = ['agree'] if consent_given else []
    return html.Div([
        HeaderComponent(
            show_time_slider=False,
            show_upload_section=show_upload,
            show_example_button=(format_value == "A"),
            initial_slider_value=example_initial_slider_value,
            locale=locale,
        ),
        html.Div(
            [
                html.Div(
                    t("ui.startup.data_usage_consent_label", locale=locale),
                    id='prediction-consent-label',
                    style={'fontWeight': '600', 'marginBottom': '8px'},
                ),
                dcc.Checklist(
                    id="prediction-data-usage-consent",
                    options=[
                        {
                            'label': t("ui.startup.data_usage_consent_label", locale=locale),
                            'value': 'agree',
                            'disabled': bool(consent_given),
                        }
                    ],
                    value=consent_value,
                    style={'fontSize': '16px'},
                ),
                html.Div(id="prediction-data-usage-consent-status"),
            ],
            style={
                'maxWidth': '900px',
                'margin': '0 auto',
                'padding': '12px 16px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.06)',
                'border': '1px solid #e5e7eb',
                'display': 'block' if show_upload else 'none',
            },
        ),
        html.Div(id="upload-required-alert", style={'margin': '0 auto', 'maxWidth': '900px'}),
        html.Div(id='round-indicator', style={
            'textAlign': 'center',
            'fontSize': '18px',
            'fontWeight': '600',
            'color': '#2c5282',
            'marginBottom': '10px'
        }),
        html.Div([
            html.Div(t("ui.prediction.units_label", locale=locale), id='prediction-units-label', style={'fontWeight': '600', 'marginRight': '10px'}),
            dbc.RadioItems(
                id='glucose-unit-selector',
                options=[
                    {'label': 'mg/dL', 'value': 'mg/dL'},
                    {'label': 'mmol/L', 'value': 'mmol/L'}
                ],
                value='mg/dL',
                inline=True
            ),
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'gap': '10px',
            'marginBottom': '10px'
        }),
        html.Div([
            html.Div(
                GlucoseChart(id='glucose-graph', hide_last_hour=True),
                id='prediction-glucose-chart-container'
            ),
            SubmitComponent(locale=locale)
        ], style={'flex': '1'})
    ], style={
        'margin': '0 auto',
        'padding': '0 20px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '20px'
    })


@app.callback(
    Output('glucose-unit', 'data', allow_duplicate=True),
    [Input('glucose-unit-selector', 'value')],
    [State('glucose-unit', 'data')],
    prevent_initial_call=True
)
def set_glucose_unit(unit_value: Optional[str], current_unit: Optional[str]) -> str:
    if unit_value not in ('mg/dL', 'mmol/L'):
        raise PreventUpdate
    # Fix: previously this always wrote to glucose-unit, which triggered
    # sync_glucose_unit_selector below, which then wrote back to glucose-unit-selector,
    # which triggered this callback again — an infinite ping-pong loop at network
    # round-trip speed. Break the cycle by suppressing the write when the store
    # already holds the same value the selector just reported.
    if unit_value == current_unit:
        raise PreventUpdate
    return unit_value


@app.callback(
    Output('glucose-unit-selector', 'value'),
    [Input('url', 'pathname'),
     Input('glucose-unit', 'data')],
    [State('glucose-unit-selector', 'value')],
    prevent_initial_call=False
)
def sync_glucose_unit_selector(
    pathname: Optional[str],
    glucose_unit: Optional[str],
    current_selector: Optional[str],
) -> str:
    if pathname != '/prediction':
        raise PreventUpdate
    resolved = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
    # Fix: same loop as above, other direction. If the selector already shows the
    # correct unit, skip the write so set_glucose_unit is not re-triggered needlessly.
    if resolved == current_selector:
        raise PreventUpdate
    return resolved

@app.callback(
    Output('round-indicator', 'children'),
    [Input('url', 'pathname'),
     Input('user-info-store', 'data'),
     Input('interface-language', 'data')],
    prevent_initial_call=False
)
def update_round_indicator(pathname: Optional[str], user_info: Optional[Dict[str, Any]], interface_language: Optional[str]) -> str:
    if pathname != '/prediction':
        raise PreventUpdate
    if not user_info:
        return ""
    rounds_played = len(user_info.get('rounds') or [])
    current_round = int(user_info.get('current_round_number') or (rounds_played + 1))
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    return t("ui.common.round_of", locale=normalize_locale(interface_language), current=current_round, total=max_rounds)


@app.callback(
    Output("upload-required-alert", "children"),
    [Input("url", "pathname"),
     Input("current-window-df", "data"),
     Input("user-info-store", "data"),
     Input("interface-language", "data")],
    prevent_initial_call=False,
)
def show_upload_required_alert(
    pathname: Optional[str],
    current_df_data: Optional[Dict[str, Any]],
    user_info: Optional[Dict[str, Any]],
    interface_language: Optional[str],
) -> Optional[html.Div]:
    if pathname != "/prediction":
        return None
    fmt = str((user_info or {}).get("format") or "A")
    if fmt not in ("B", "C"):
        return None
    if current_df_data:
        return None
    locale = normalize_locale(interface_language)
    has_prior_rounds = bool((user_info or {}).get("runs_by_format") or (user_info or {}).get("rounds"))
    consent_ok = bool((user_info or {}).get("consent_use_uploaded_data", False))
    children: list[Any] = [t("ui.prediction.upload_required_alert", locale=locale)]
    if not consent_ok:
        children += [
            html.Br(),
            html.Span(t("ui.startup.data_usage_consent_required", locale=locale)),
        ]
    if has_prior_rounds:
        children += [
            html.Br(),
            html.Button(
                t("ui.common.game", locale=locale) + " → " + t("ui.final.title", locale=locale),
                id="back-to-final-from-upload",
                className="ui small button",
                style={"paddingLeft": "0", "marginTop": "6px"},
            ),
        ]
    return dbc.Alert(children, color="info", style={"marginBottom": "10px"})

def create_ending_layout(
    full_df_data: Optional[Dict],
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
    user_info: Optional[Dict] = None,
    glucose_unit: Optional[str] = None,
    *,
    locale: str,
) -> html.Div:
    """Create the ending page layout"""
    if not full_df_data:
        print("DEBUG: No data available for ending page")
        return html.Div("No data available", style={'textAlign': 'center', 'padding': '50px'})
    
    print("DEBUG: Creating ending page with stored data")
    
    # Reconstruct DataFrames from stored data
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    events_df = reconstruct_events_dataframe_from_dict(events_df_data) if events_df_data else pl.DataFrame(
        {
            'time': [],
            'event_type': [],
            'event_subtype': [],
            'insulin_value': []
        }
    )
    
    # Check if we have stored prediction data from the submit button
    if user_info and 'prediction_table_data' in user_info:
        print("DEBUG: Using stored prediction table data from submit button")
        unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
        prediction_table_data = _convert_table_data_units(user_info['prediction_table_data'], unit)
        
        # Check if we have predictions in the stored data
        if len(prediction_table_data) >= 2:
            prediction_row = prediction_table_data[1]  # Second row contains predictions
            valid_predictions = sum(1 for key, value in prediction_row.items() 
                                  if key != 'metric' and value != "-")
            print(f"DEBUG: Found {valid_predictions} valid predictions in stored data")
            
            if valid_predictions == 0:
                print("DEBUG: No valid predictions in stored data")
                return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
        else:
            print("DEBUG: No prediction table data available")
            return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
        
        # Prefer the exact window with predictions as stored in session (fixes missing prediction traces).
        if current_df_data:
            df = reconstruct_dataframe_from_dict(current_df_data)
            print(f"DEBUG: Using current-window-df for ending chart (points={len(df)})")
        elif user_info and 'prediction_window_start' in user_info and 'prediction_window_size' in user_info:
            window_start = user_info['prediction_window_start']
            window_size = user_info['prediction_window_size']
            # Ensure we don't go beyond the available data
            max_start = len(full_df) - window_size
            safe_start = min(window_start, max_start)
            safe_start = max(0, safe_start)
            df = full_df.slice(safe_start, window_size)
            print(f"DEBUG: Using prediction window starting at {safe_start} with size {window_size}")
        else:
            # Fallback to first DEFAULT_POINTS for display
            df = full_df.slice(0, DEFAULT_POINTS)
            print("DEBUG: No prediction window info found, using default first 24 points")
    else:
        print("DEBUG: No stored prediction data found")
        return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
    
    # Calculate metrics directly from the stored prediction table data
    metrics_component_ending = MetricsComponent()
    stored_metrics = None
    
    if len(prediction_table_data) >= 2:  # Need at least actual and predicted rows
        stored_metrics = metrics_component_ending._calculate_metrics_from_table_data(prediction_table_data)
    
    def _translate_metric_label(metric: str) -> str:
        mapping: dict[str, str] = {
            "Actual Glucose": t("ui.table.actual_glucose", locale=locale),
            "Predicted": t("ui.table.predicted", locale=locale),
            "Absolute Error": t("ui.table.absolute_error", locale=locale),
            "Relative Error (%)": t("ui.table.relative_error_pct", locale=locale, pct="%"),
        }
        return mapping.get(metric, metric)

    prediction_table_data_display: list[dict[str, str]] = []
    for row in prediction_table_data:
        metric_val = str(row.get("metric", ""))
        new_row = dict(row)
        new_row["metric"] = _translate_metric_label(metric_val)
        prediction_table_data_display.append(new_row)

    # Create metrics display directly
    metrics_display = MetricsComponent.create_ending_metrics_display(stored_metrics, locale=locale) if stored_metrics else [
        html.H3(t("ui.metrics.title_accuracy_metrics", locale=locale), style={'textAlign': 'center'}),
        html.Div(
            t("ui.metrics.no_metrics_available", locale=locale),
            style={
                'color': 'gray',
                'fontStyle': 'italic',
                'fontSize': '16px',
                'padding': '10px',
                'textAlign': 'center'
            }
        )
    ]

    # Create the page content with metrics container that will be populated by the callback
    rounds_played = len(user_info.get('rounds') or []) if user_info else 0
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS) if user_info else MAX_ROUNDS
    current_round_number = int(user_info.get('current_round_number') or rounds_played) if user_info else rounds_played
    is_last_round = current_round_number >= max_rounds
    current_format = str((user_info or {}).get("format") or "A")
    uses_cgm = bool((user_info or {}).get("uses_cgm", False))
    allowed_formats: list[str] = (["C", "B", "A"] if uses_cgm else ["A"])
    runs_by_format: dict[str, list[dict[str, Any]]] = dict((user_info or {}).get("runs_by_format") or {})
    already_played: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
    if rounds_played > 0:
        already_played.add(current_format)
    switch_targets: list[str] = [f for f in allowed_formats if f not in already_played]
    # Consent is handled on the prediction page (B/C upload flow).
    show_switch_data_consent = False
    switch_data_consent_value: list[str] = []

    data_source_name = str(user_info.get('data_source_name') or '') if user_info else ''
    meta = GENERIC_SOURCES_METADATA.get(Path(data_source_name).name) if data_source_name else None

    subject_parts: list[str] = []
    if data_source_name:
        subject_parts.append(t("ui.ending.data_source_label", locale=locale, source=Path(data_source_name).name))
    if meta:
        gender_raw = str(meta.gender or "").strip().lower()
        gender_display = (
            t(f"ui.startup.gender_{gender_raw}", locale=locale)
            if gender_raw in ("male", "female", "na")
            else meta.gender
        )
        subject_parts.append(
            f"{t('ui.startup.age_label', locale=locale)}: {meta.age} · "
            f"{t('ui.startup.gender_label', locale=locale)}: {gender_display} · "
            f"{t('ui.header.weight_label', locale=locale)}: {meta.weight}"
        )
    elif user_info:
        age = user_info.get('age')
        gender_raw = str(user_info.get('gender') or "").strip().lower()
        if age:
            gender_display = (
                t(f"ui.startup.gender_{gender_raw}", locale=locale)
                if gender_raw in ("male", "female", "na")
                else (user_info.get('gender') or "")
            )
            parts = [f"{t('ui.startup.age_label', locale=locale)}: {age}"]
            if gender_display:
                parts.append(f"{t('ui.startup.gender_label', locale=locale)}: {gender_display}")
            subject_parts.append(" · ".join(parts))

    subject_info_line = " — ".join(subject_parts) if subject_parts else ""

    return html.Div([
        html.H1(t("ui.ending.title", locale=locale), id='ending-title', style={
            'textAlign': 'center', 
            'marginBottom': '20px',
            'fontSize': 'clamp(24px, 4vw, 48px)',
            'padding': '0 10px'
        }),
        html.Div(
            [
                html.P(t("ui.results_disclaimer.line1", locale=locale), id='ending-disclaimer-line1', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line2", locale=locale), id='ending-disclaimer-line2', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line3", locale=locale), id='ending-disclaimer-line3', style={'margin': '0'}),
            ],
            disable_n_clicks=True,
            style={
                'maxWidth': '900px',
                'margin': '0 auto 15px auto',
                'padding': '12px 16px',
                'backgroundColor': '#fff7ed',
                'border': '1px solid #fdba74',
                'borderRadius': '10px',
                'color': '#7c2d12',
                'fontSize': '14px',
                'lineHeight': '1.4',
                'boxSizing': 'border-box',
            },
        ),
        html.Div(
            t("ui.common.round_of", locale=locale, current=current_round_number, total=max_rounds),
            id='ending-round-info',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '2px',
                'fontSize': 'clamp(16px, 2.5vw, 22px)',
                'fontWeight': '600',
                'color': '#2c5282'
            }
        ),
        html.Div(
            t("ui.ending.round_motivation", locale=locale, total=max_rounds, min_useful=max(1, max_rounds // 2)),
            id='ending-round-motivation',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '5px',
                'color': '#4a5568',
                'fontSize': '13px',
                'fontStyle': 'italic',
                'display': 'none' if is_last_round else 'block',
            }
        ),
        html.Div(
            subject_info_line,
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '5px',
                'color': '#4a5568',
                'fontSize': '13px',
                'display': 'block' if subject_info_line else 'none',
            }
        ),
        html.Div(
            t("ui.ending.units_line", locale=locale, unit=unit),
            id='ending-units-line',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '5px',
                'color': '#4a5568',
                'fontSize': '14px'
            }
        ),
        # Graph section - full window with known + predicted lines
        html.Div([
            html.P(
                t("ui.ending.graph_explanation", locale=locale),
                id='ending-graph-explanation',
                style={
                    'textAlign': 'center',
                    'color': '#4a5568',
                    'fontSize': '14px',
                    'marginBottom': '8px',
                    'fontStyle': 'italic',
                },
            ),
            html.Div(
                id='ending-glucose-chart-container',
                children=dcc.Graph(
                    id='ending-static-graph',
                    figure=GlucoseChart.build_static_figure(
                        df,
                        events_df,
                        str(user_info.get('data_source_name') or '') if user_info else None,
                        unit=unit,
                        locale=locale,
                        prediction_boundary=len(df) - PREDICTION_HOUR_OFFSET,
                    ),
                    config={
                        'displayModeBar': False,
                        'scrollZoom': False,
                        'doubleClick': 'reset',
                        'showAxisDragHandles': False,
                        'displaylogo': False,
                        'editable': False,
                    },
                    style={'height': '400px'},
                ),
                disable_n_clicks=True,
            )
        ], disable_n_clicks=True, style={
            'marginBottom': '20px',
            'padding': 'clamp(10px, 2vw, 20px)',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '100%',
            'boxSizing': 'border-box'
        }),
        
        # Prediction table section - only columns with actual predictions
        html.Div([
            html.H3(t("ui.ending.prediction_results", locale=locale), id='ending-prediction-results-title', style={
                'textAlign': 'center', 
                'marginBottom': '15px',
                'fontSize': 'clamp(18px, 3vw, 24px)'
            }),
            dash_table.DataTable(
                id='ending-prediction-table',
                data=prediction_table_data_display,
                columns=[{'name': t("ui.table.metric_header", locale=locale), 'id': 'metric'}] + [
                    {'name': f'T{i}', 'id': f't{i}', 'type': 'text'}
                    for i in range(len(prediction_table_data[0]) - 1)
                    if prediction_table_data
                    and prediction_table_data[1].get(f't{i}', '-') != '-'
                ],
                cell_selectable=False,
                row_selectable=False,
                editable=False,
                style_table={
                    'width': '100%',
                    'height': 'auto',
                    'maxHeight': 'clamp(300px, 40vh, 500px)',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'tableLayout': 'fixed'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': 'clamp(2px, 1vw, 4px) clamp(1px, 0.5vw, 2px)',
                    'fontSize': 'clamp(8px, 1.5vw, 12px)',
                    'whiteSpace': 'nowrap',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'lineHeight': '1.2',
                    'minWidth': '40px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 0},
                        'backgroundColor': 'rgba(200, 240, 200, 0.5)'
                    },
                    {
                        'if': {'row_index': 1},
                        'backgroundColor': 'rgba(255, 200, 200, 0.5)'
                    }
                ]
            )
        ], style={
            'marginBottom': '20px',
            'padding': 'clamp(10px, 2vw, 20px)',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'display': 'flex',
            'flexDirection': 'column',
            'width': '100%',
            'boxSizing': 'border-box',
            'overflowX': 'auto'
        }),
        html.Div(
            metrics_display,
            id='ending-metrics-container',
            disable_n_clicks=True,
            style={
                'padding': 'clamp(10px, 2vw, 20px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'width': '100%',
                'boxSizing': 'border-box'
            }
        ),
        
        html.Div(
            t("ui.ending.local_storage_note", locale=locale),
            id='ending-local-storage-note',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '10px',
                'color': '#2d6a4f',
                'fontSize': '13px',
                'fontStyle': 'italic',
                'display': 'block' if STORAGE_TYPE == 'local' else 'none',
            }
        ),
        html.Div([
            html.Button(
                t("ui.ending.view_complete_analysis", locale=locale) if is_last_round else t("ui.common.finish_exit", locale=locale),
                id='finish-study-button-ending',
                autoFocus=False,
                style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'padding': 'clamp(12px, 2vw, 16px) clamp(18px, 3vw, 26px)',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': 'clamp(16px, 2.5vw, 22px)',
                    'cursor': 'pointer',
                    'minWidth': '200px',
                    'maxWidth': '400px',
                    'width': '100%',
                    'height': 'clamp(55px, 7vh, 70px)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2',
                    'margin': '0 clamp(5px, 1vw, 10px)',
                }
            ),
            html.Button(
                t("ui.ending.next_round", locale=locale),
                id='next-round-button',
                className="ui green button",
                disabled=is_last_round,
                style={
                    'backgroundColor': '#4CBB17' if not is_last_round else '#cccccc',
                    'color': 'white' if not is_last_round else '#666666',
                    'padding': 'clamp(12px, 2vw, 16px) clamp(18px, 3vw, 26px)',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': 'clamp(16px, 2.5vw, 22px)',
                    'cursor': 'pointer' if not is_last_round else 'not-allowed',
                    'minWidth': '200px',
                    'maxWidth': '400px',
                    'width': '100%',
                    'height': 'clamp(55px, 7vh, 70px)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2',
                    'margin': '0 clamp(5px, 1vw, 10px)',
                }
            ),
        ], disable_n_clicks=True, style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'stretch',
            'marginTop': '20px',
            'padding': '0 10px',
        }),
        html.Div(
            [
                html.H3(
                    t("ui.switch_format.title", locale=locale),
                    id='ending-switch-format-title',
                    style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px', 'fontSize': 'clamp(18px, 3vw, 24px)'},
                ),
                html.Div(id="switch-format-error", style={'marginBottom': '10px'}),
                dcc.Checklist(
                    id="switch-data-usage-consent",
                    options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
                    value=switch_data_consent_value,
                    style={'display': 'none'},
                ),
                html.Div(
                    [
                        html.Button(
                            t("ui.switch_format.try_c", locale=locale),
                            id="switch-format-c",
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'padding': '12px 18px',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'display': 'inline-block' if "C" in switch_targets else 'none',
                            },
                        ),
                        html.Button(
                            t("ui.switch_format.try_a", locale=locale),
                            id="switch-format-a",
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'padding': '12px 18px',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'display': 'inline-block' if "A" in switch_targets else 'none',
                            },
                        ),
                        html.Button(
                            t("ui.switch_format.try_b", locale=locale),
                            id="switch-format-b",
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'padding': '12px 18px',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'display': 'inline-block' if "B" in switch_targets else 'none',
                            },
                        ),
                    ],
                    style={'display': 'flex', 'justifyContent': 'center', 'gap': '12px', 'flexWrap': 'wrap'},
                ),
            ],
            disable_n_clicks=True,
            style={
                'marginTop': '10px',
                'padding': 'clamp(10px, 2vw, 20px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'width': '100%',
                'boxSizing': 'border-box',
                'display': 'block' if (is_last_round and switch_targets) else 'none',
            },
        ),
    ], disable_n_clicks=True, style={
        'maxWidth': '100%',
        'width': '100%',
        'margin': '0 auto',
        'padding': 'clamp(10px, 2vw, 20px)',
        'display': 'flex',
        'flexDirection': 'column',
        'minHeight': '100vh',
        'gap': 'clamp(10px, 2vh, 20px)',
        'boxSizing': 'border-box'
    })


def _count_valid_pairs_from_table_data(table_data: list[dict[str, str]]) -> int:
    if len(table_data) < 2:
        return 0
    actual_row = table_data[0]
    prediction_row = table_data[1]
    count = 0
    for key, actual_str in actual_row.items():
        if key == 'metric':
            continue
        pred_str = prediction_row.get(key, "-")
        if actual_str != "-" and pred_str != "-":
            count += 1
    return count


def _convert_table_data_units(table_data: list[dict[str, str]], glucose_unit: str) -> list[dict[str, str]]:
    """Convert table display values between mg/dL and mmol/L (display only)."""
    if glucose_unit != 'mmol/L':
        return table_data

    converted: list[dict[str, str]] = []
    for row in table_data:
        metric = row.get('metric', '')
        new_row: dict[str, str] = {'metric': metric}

        # Only convert numeric glucose-like rows. Keep % rows untouched.
        convert_row = metric in {'Actual Glucose', 'Predicted', 'Absolute Error'}

        for key, val in row.items():
            if key == 'metric':
                continue
            if not convert_row or val == "-" or val is None:
                new_row[key] = val
                continue
            if isinstance(val, str) and '%' in val:
                new_row[key] = val
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                new_row[key] = val
                continue
            new_row[key] = f"{(num / GLUCOSE_MGDL_PER_MMOLL):.1f}"

        converted.append(new_row)

    return converted


def _build_aggregate_table_data(rounds: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a synthetic table_data for aggregated metrics across rounds."""
    actual_row: dict[str, str] = {'metric': 'Actual Glucose'}
    prediction_row: dict[str, str] = {'metric': 'Predicted'}
    out_idx = 0

    for round_info in rounds:
        table_data = round_info.get('prediction_table_data') or []
        if len(table_data) < 2:
            continue

        round_actual = table_data[0]
        round_pred = table_data[1]

        # Ensure deterministic order t0..tN
        i = 0
        while True:
            key = f"t{i}"
            if key not in round_actual or key not in round_pred:
                break
            actual_row[f"t{out_idx}"] = round_actual.get(key, "-")
            prediction_row[f"t{out_idx}"] = round_pred.get(key, "-")
            out_idx += 1
            i += 1

    return [actual_row, prediction_row]


def create_final_layout(full_df_data: Optional[Dict], user_info: Dict[str, Any], glucose_unit: Optional[str], *, locale: str) -> html.Div:
    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    # If current rounds are empty (e.g. user just switched format), fall back to the
    # most recently archived run so results are still visible.
    if not rounds:
        runs_by_format: dict[str, list[dict[str, Any]]] = dict(user_info.get('runs_by_format') or {})
        all_archived: list[dict[str, Any]] = [run for runs in runs_by_format.values() for run in runs]
        if all_archived:
            latest_run = max(all_archived, key=lambda r: r.get('ended_at') or '')
            rounds = list(latest_run.get('rounds') or [])
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
    study_id = str(user_info.get('study_id') or '')
    current_format = str(user_info.get("format") or "A")
    uses_cgm = bool(user_info.get("uses_cgm", False))
    allowed_formats: list[str] = (["C", "B", "A"] if uses_cgm else ["A"])
    runs_by_format: dict[str, list[dict[str, Any]]] = dict(user_info.get("runs_by_format") or {})
    already_played: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
    if rounds:
        already_played.add(current_format)
    switch_targets: list[str] = [f for f in allowed_formats if f not in already_played]
    # Consent is handled on the prediction page (B/C upload flow).
    show_switch_data_consent = False
    switch_data_consent_value: list[str] = []
    played_formats: list[str] = sorted(already_played, key=lambda x: FORMAT_ORDER.get(str(x), 999))

    def _rank_info(
        ranking_path: Path,
        *,
        format_filter: Optional[str],
        mode: str,
    ) -> Optional[tuple[int, int]]:
        """Return (rank, total) by overall MAE (mg/dL) for this study_id."""
        if not study_id or not ranking_path.exists():
            return None
        try:
            ranking_df = pl.read_csv(ranking_path)
        except Exception:
            return None
        if 'study_id' not in ranking_df.columns or 'overall_mae_mgdl' not in ranking_df.columns:
            return None

        cols: list[str] = ['study_id', 'overall_mae_mgdl']
        if 'format' in ranking_df.columns:
            cols.append('format')
        if 'timestamp' in ranking_df.columns:
            cols.append('timestamp')
        df2 = ranking_df.select([c for c in cols if c in ranking_df.columns])
        df2 = df2.with_columns(pl.col('overall_mae_mgdl').cast(pl.Float64, strict=False)).filter(
            pl.col('overall_mae_mgdl').is_not_null()
        )
        if format_filter and 'format' in df2.columns:
            df2 = df2.filter(pl.col('format') == format_filter)

        if mode == "latest" and 'timestamp' in df2.columns:
            df2 = df2.with_columns(
                pl.col('timestamp').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False).alias('_ts')
            )
            df_pick = (
                df2.sort(['study_id', '_ts'])
                .group_by('study_id')
                .agg(pl.last('overall_mae_mgdl').alias('overall_mae_mgdl'))
            )
        else:
            # Default: keep the best (lowest MAE) per study_id.
            df_pick = df2.group_by('study_id').agg(pl.col('overall_mae_mgdl').min().alias('overall_mae_mgdl'))

        total = df_pick.height
        if total == 0:
            return None

        df_sorted = df_pick.sort(['overall_mae_mgdl', 'study_id'])
        matches = df_sorted.with_row_index('rank_idx').filter(pl.col('study_id') == study_id)
        if matches.height == 0:
            return None
        rank = int(matches.get_column('rank_idx')[0]) + 1
        return rank, total

    ranking_lines: list[str] = []
    for fmt in played_formats:
        if fmt not in ("A", "B", "C"):
            continue
        info = _rank_info(
            project_root / 'data' / 'input' / f'prediction_ranking_{fmt}.csv',
            format_filter=fmt,
            mode="best",
        )
        if info:
            rank, total = info
            ranking_lines.append(
                t(
                    "ui.final.ranking_format_line",
                    locale=locale,
                    format=_format_label(fmt, locale=locale),
                    rank=rank,
                    total=total,
                )
            )

    # Always show cumulative overall ranking ("ALL"), updated after each finished run.
    info = _rank_info(
        project_root / 'data' / 'input' / 'prediction_ranking.csv',
        format_filter="ALL",
        mode="latest",
    )
    if info:
        rank, total = info
        ranking_lines.append(t("ui.final.ranking_overall_line", locale=locale, rank=rank, total=total))

    metrics_component_final = MetricsComponent()
    aggregate_table_data = _convert_table_data_units(_build_aggregate_table_data(rounds), unit)
    overall_metrics = metrics_component_final._calculate_metrics_from_table_data(aggregate_table_data)
    overall_metrics_display = MetricsComponent.create_ending_metrics_display(overall_metrics, locale=locale) if overall_metrics else [
        html.H3(t("ui.metrics.title_accuracy_metrics", locale=locale), style={'textAlign': 'center'}),
        html.Div(
            t("ui.metrics.no_metrics_available", locale=locale),
            style={
                'color': 'gray',
                'fontStyle': 'italic',
                'fontSize': '16px',
                'padding': '10px',
                'textAlign': 'center'
            }
        )
    ]

    round_rows: list[dict[str, Any]] = []
    for round_info in rounds:
        round_number = int(round_info.get('round_number') or (len(round_rows) + 1))
        table_data_raw = round_info.get('prediction_table_data') or []
        table_data = _convert_table_data_units(table_data_raw, unit)
        valid_pairs = _count_valid_pairs_from_table_data(table_data)
        round_metrics = metrics_component_final._calculate_metrics_from_table_data(table_data) if len(table_data) >= 2 else {}

        def _metric_value(metric_name: str) -> Optional[float]:
            metric = round_metrics.get(metric_name)
            if not metric:
                return None
            val = metric.get('value')
            return float(val) if val is not None else None

        round_rows.append({
            'Round': round_number,
            'Pairs': valid_pairs,
            'MAE': _metric_value('MAE'),
            'MSE': _metric_value('MSE'),
            'RMSE': _metric_value('RMSE'),
            'MAPE': _metric_value('MAPE'),
        })

    return html.Div([
        html.H1(t("ui.final.title", locale=locale), id='final-title', style={
            'textAlign': 'center',
            'marginBottom': '10px',
            'fontSize': 'clamp(24px, 4vw, 48px)',
            'padding': '0 10px'
        }),
        html.Div(
            [
                html.P(t("ui.results_disclaimer.line1", locale=locale), id='final-disclaimer-line1', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line2", locale=locale), id='final-disclaimer-line2', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line3", locale=locale), id='final-disclaimer-line3', style={'margin': '0'}),
            ],
            disable_n_clicks=True,
            style={
                'maxWidth': '900px',
                'margin': '0 auto 15px auto',
                'padding': '12px 16px',
                'backgroundColor': '#fff7ed',
                'border': '1px solid #fdba74',
                'borderRadius': '10px',
                'color': '#7c2d12',
                'fontSize': '14px',
                'lineHeight': '1.4',
                'boxSizing': 'border-box',
            },
        ),
        html.Div(
            t("ui.final.rounds_played", locale=locale, played=len(rounds), total=max_rounds),
            id='final-rounds-played',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '20px',
                'fontSize': 'clamp(16px, 2.5vw, 22px)',
                'fontWeight': '600',
                'color': '#2c5282'
            }
        ),
        html.Div(
            [
                html.H3(t("ui.final.ranking_title", locale=locale), id='final-ranking-title', style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Ul([html.Li(line) for line in ranking_lines], id='final-ranking-list', style={'margin': '0 auto', 'maxWidth': '760px'}),
            ],
            disable_n_clicks=True,
            style={
                'marginBottom': '15px',
                'color': '#4a5568',
                'fontSize': '14px',
                'display': 'block' if ranking_lines else 'none',
                'padding': 'clamp(10px, 2vw, 16px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            },
        ),
        html.Div(
            (
                t(
                    "ui.final.played_formats",
                    locale=locale,
                    formats=", ".join(_format_label(f, locale=locale) for f in played_formats),
                )
                if played_formats
                else ""
            ),
            id='final-played-formats',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '12px',
                'color': '#4a5568',
                'fontSize': '14px',
                'display': 'block' if played_formats else 'none',
            },
        ),
        html.Div(
            overall_metrics_display,
            id='final-overall-metrics-container',
            disable_n_clicks=True,
            style={
                'padding': 'clamp(10px, 2vw, 20px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'width': '100%',
                'boxSizing': 'border-box'
            }
        ),
        html.Div([
            html.H3(t("ui.final.per_round_metrics", locale=locale), id='final-per-round-title', style={
                'textAlign': 'center',
                'marginBottom': '15px',
                'fontSize': 'clamp(18px, 3vw, 24px)'
            }),
            html.Div(
                t("ui.ending.units_line", locale=locale, unit=unit),
                id='final-units-line',
                style={
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'color': '#4a5568',
                    'fontSize': '14px'
                }
            ),
            dash_table.DataTable(
                id='final-rounds-table',
                data=round_rows,
                columns=[
                    {'name': 'Round', 'id': 'Round', 'type': 'numeric'},
                    {'name': 'Pairs', 'id': 'Pairs', 'type': 'numeric'},
                    {'name': 'MAE', 'id': 'MAE', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'MSE', 'id': 'MSE', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'RMSE', 'id': 'RMSE', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'MAPE', 'id': 'MAPE', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
                ],
                cell_selectable=False,
                row_selectable=False,
                editable=False,
                style_table={
                    'width': '100%',
                    'overflowX': 'auto'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '8px',
                    'fontSize': '14px',
                    'whiteSpace': 'nowrap'
                },
                style_header={
                    'backgroundColor': '#f8fafc',
                    'fontWeight': 'bold'
                }
            )
        ], disable_n_clicks=True, style={
            'marginBottom': '20px',
            'padding': 'clamp(10px, 2vw, 20px)',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '100%',
            'boxSizing': 'border-box'
        }),
        html.Div(
            [
                html.H3(
                    t("ui.switch_format.title", locale=locale),
                    id='final-switch-format-title',
                    style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': 'clamp(18px, 3vw, 24px)'},
                ),
                html.Div(id="switch-format-error", style={'marginBottom': '10px'}),
                dcc.Checklist(
                    id="switch-data-usage-consent",
                    options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
                    value=switch_data_consent_value,
                    style={'display': 'none'},
                ),
                html.Div(
                    [
                        html.Button(
                            t("ui.switch_format.try_a", locale=locale),
                            id="switch-format-a",
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'padding': '12px 18px',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'display': 'inline-block' if "A" in switch_targets else 'none',
                            },
                        ),
                        html.Button(
                            t("ui.switch_format.try_b", locale=locale),
                            id="switch-format-b",
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'padding': '12px 18px',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'display': 'inline-block' if "B" in switch_targets else 'none',
                            },
                        ),
                        html.Button(
                            t("ui.switch_format.try_c", locale=locale),
                            id="switch-format-c",
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'padding': '12px 18px',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'display': 'inline-block' if "C" in switch_targets else 'none',
                            },
                        ),
                    ],
                    disable_n_clicks=True,
                    style={'display': 'flex', 'justifyContent': 'center', 'gap': '12px', 'flexWrap': 'wrap'},
                ),
            ],
            disable_n_clicks=True,
            style={
                'marginBottom': '20px',
                'padding': 'clamp(10px, 2vw, 20px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'width': '100%',
                'boxSizing': 'border-box',
                'display': 'block' if switch_targets else 'none',
            },
        ),
        html.Div([
            html.Button(
                t("ui.final.start_over", locale=locale),
                id='restart-button',
                className="ui green button",
                style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'padding': 'clamp(15px, 2vw, 20px) clamp(20px, 3vw, 30px)',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': 'clamp(18px, 3vw, 24px)',
                    'cursor': 'pointer',
                    'minWidth': '200px',
                    'maxWidth': '400px',
                    'width': '100%',
                    'height': 'clamp(60px, 8vh, 80px)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2'
                }
            ),
            html.Button(
                t("ui.common.finish_exit", locale=locale),
                id='final-exit-button',
                className="ui red button",
                style={
                    'backgroundColor': '#dc3545',
                    'color': 'white',
                    'padding': 'clamp(15px, 2vw, 20px) clamp(20px, 3vw, 30px)',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': 'clamp(18px, 3vw, 24px)',
                    'cursor': 'pointer',
                    'minWidth': '200px',
                    'maxWidth': '400px',
                    'width': '100%',
                    'height': 'clamp(60px, 8vh, 80px)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2',
                    'marginLeft': '15px'
                }
            )
        ], disable_n_clicks=True, style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'marginTop': '20px',
            'padding': '0 10px'
        })
    ], disable_n_clicks=True, style={
        'maxWidth': '100%',
        'width': '100%',
        'margin': '0 auto',
        'padding': 'clamp(10px, 2vw, 20px)',
        'display': 'flex',
        'flexDirection': 'column'
    })

def render_mobile_warning(user_agent: Optional[str], *, locale: str) -> Optional[html.Div]:
    if not user_agent:
        return None
    ua = user_agent.lower()
    mobile_keywords = ("iphone", "android", "ipad", "mobile", "opera mini", "mobi")
    if any(keyword in ua for keyword in mobile_keywords):
        return html.Div(
            t("ui.mobile_warning.text", locale=locale),
            style={
                'backgroundColor': '#fff3cd',
                'border': '1px solid #ffeeba',
                'color': '#856404',
                'padding': '10px 14px',
                'borderRadius': '6px',
                'textAlign': 'center',
                'marginBottom': '12px',
                'fontWeight': '600'
            }
        )
    return None

def reconstruct_events_dataframe_from_dict(events_data: Dict[str, List[Any]]) -> pl.DataFrame:
    """Reconstruct the events DataFrame from stored data.""" 
    # Convert mixed types to strings first, then to float
    insulin_values = []
    for val in events_data['insulin_value']:
        if val is None or val == '':
            insulin_values.append(None)
        else:
            try:
                # Convert to float, handling both string and numeric inputs
                insulin_values.append(float(val))
            except (ValueError, TypeError):
                insulin_values.append(None)
    
    return pl.DataFrame({
        'time': pl.Series(events_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
        'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
        # Use pre-processed float values
        'insulin_value': pl.Series(insulin_values, dtype=pl.Float64)
    })

@app.callback(
    [Output('url', 'pathname'),
     Output('user-info-store', 'data')],
    [Input('start-button', 'n_clicks')],
    [State('email-input', 'value'),
     State('age-input', 'value'),
     State('gender-dropdown', 'value'),
     State('cgm-dropdown', 'value'),
     State('cgm-duration-input', 'value'),
     State('format-dropdown', 'value'),
     State('data-usage-consent', 'value'),
     State('diabetic-dropdown', 'value'),
     State('diabetic-type-dropdown', 'value'),
     State('diabetes-duration-input', 'value'),
     State('location-input', 'value'),
     State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_start_button(n_clicks: Optional[int], email: Optional[str], age: Optional[int | float], 
                       gender: Optional[str], uses_cgm: Optional[bool], cgm_duration_years: Optional[float],
                       format_value: Optional[str], data_usage_consent: Optional[list[str]],
                       diabetic: Optional[bool], diabetic_type: Optional[str], 
                       diabetes_duration: Optional[float], location: Optional[str],
                       existing_user_info: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """Handle start button on startup page"""
    if not n_clicks:
        return no_update, no_update

    is_adult = (age is not None) and (float(age) >= 18)
    has_data_consent = bool(data_usage_consent and "agree" in data_usage_consent)

    if age and gender and diabetic is not None and location and format_value and is_adult:
        from datetime import datetime
        from sugar_sugar.consent import ensure_consent_agreement_row, get_next_study_number

        info: Dict[str, Any] = dict(existing_user_info or {})
        study_id = info.get('study_id') or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        uses_cgm_bool = bool(uses_cgm) if uses_cgm is not None else False

        info.update({
            'study_id': study_id,
            'run_id': run_id,
            'email': email or info.get('email') or '',
            'age': age,
            'gender': gender,
            'uses_cgm': uses_cgm_bool,
            'cgm_duration_years': cgm_duration_years,
            'format': format_value,
            'run_format': format_value,
            # Optional consent for uploaded CGM data usage in study.
            # Only meaningful for B/C, but we store an explicit boolean for all formats.
            'consent_use_uploaded_data': bool(has_data_consent) if format_value in ("B", "C") else False,
            'diabetic': diabetic,
            'diabetic_type': diabetic_type,
            'diabetes_duration': diabetes_duration,
            'location': location,
            'rounds': info.get('rounds') or [],
            'max_rounds': int(info.get('max_rounds') or MAX_ROUNDS),
            'current_round_number': int(info.get('current_round_number') or 1),
            'statistics_saved': bool(info.get('statistics_saved') or False),
            'is_example_data': bool(info.get('is_example_data', True)),
            'data_source_name': str(info.get('data_source_name', 'example.csv')),
        })

        # Ensure stable "number" across consent + stats + ranking CSVs.
        if info.get("number") is None:
            info["number"] = get_next_study_number()

        # Ensure consent fields are explicit booleans (avoid null/missing keys in session storage).
        if "consent_play_only" not in info:
            info["consent_play_only"] = False
        if "consent_participate_in_study" not in info:
            info["consent_participate_in_study"] = False
        if "consent_receive_results_later" not in info:
            info["consent_receive_results_later"] = False
        if "consent_keep_up_to_date" not in info:
            info["consent_keep_up_to_date"] = False
        if "consent_no_selection" not in info:
            info["consent_no_selection"] = True
        if "consent_timestamp" not in info:
            info["consent_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure consent CSV always has a row for this study_id (even when users bypass landing).
        ensure_consent_agreement_row(
            {
                "study_id": info["study_id"],
                "number": info.get("number", ""),
                "timestamp": info.get("consent_timestamp", ""),
                "play_only": bool(info.get("consent_play_only", False)),
                "participate_in_study": bool(info.get("consent_participate_in_study", False)),
                "receive_results_later": bool(info.get("consent_receive_results_later", False)),
                "keep_up_to_date": bool(info.get("consent_keep_up_to_date", False)),
                "no_selection": bool(info.get("consent_no_selection", True)),
            }
        )
        return '/prediction', info
    return no_update, no_update


@app.callback(
    Output('user-info-store', 'data', allow_duplicate=True),
    [Input('data-source-name', 'data'),
     Input('is-example-data', 'data')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def sync_data_source_into_user_info(
    data_source_name: Optional[str],
    is_example_data: Optional[bool],
    user_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not user_info:
        raise PreventUpdate
    user_info['data_source_name'] = data_source_name or user_info.get('data_source_name') or 'example.csv'
    user_info['is_example_data'] = bool(is_example_data) if is_example_data is not None else bool(user_info.get('is_example_data', True))
    return user_info

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('time-slider', 'value')],
    prevent_initial_call=True
)
def handle_submit_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict],
    current_df_data: Optional[Dict],
    slider_value: Optional[int],
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool], Dict[str, List[Any]]]:
    """Handle submit button on prediction page"""
    print(f"DEBUG handle_submit_button FIRED: n_clicks={n_clicks}")
    # NOTE: Dash can re-trigger callbacks when components are re-mounted across pages.
    # Guard so we only process a *new* submit for the current round.
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    info_guard: Dict[str, Any] = dict(user_info or {})
    rounds_guard: list[dict[str, Any]] = info_guard.get('rounds') or []
    pending_round_number = int(len(rounds_guard) + 1)
    last_submit_round_number = int(info_guard.get("last_submit_round_number") or 0)
    last_submit_n_clicks = int(info_guard.get("last_submit_n_clicks") or 0)
    if pending_round_number == last_submit_round_number and int(n_clicks) <= last_submit_n_clicks:
        return no_update, no_update, no_update, no_update

    if full_df_data and current_df_data:
        print("DEBUG: Submit button clicked")
        
        # Reconstruct DataFrames from session storage
        current_full_df = reconstruct_dataframe_from_dict(full_df_data)
        current_df = reconstruct_dataframe_from_dict(current_df_data)
        
        # Update age and user_id from user_info
        if user_info and 'age' in user_info:
            current_full_df = current_full_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
            current_df = current_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
        
        # Generate prediction table data directly from DataFrame instead of relying on component
        if user_info is None:
            user_info = {}
        # Mark this round as submitted at this click-count. This prevents double-submits if the
        # callback is re-triggered due to component re-mounts/navigation.
        user_info["last_submit_round_number"] = pending_round_number
        user_info["last_submit_n_clicks"] = int(n_clicks)

        rounds: list[dict[str, Any]] = user_info.get('rounds') or []
        max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
        round_number = len(rounds) + 1
        
        # Store the window position information for the ending page
        user_info['prediction_window_start'] = slider_value or 0
        user_info['prediction_window_size'] = len(current_df)
        
        # Create a temporary prediction table component to generate the table data
        temp_prediction_table = PredictionTableComponent()
        prediction_table_data = temp_prediction_table._generate_table_data(current_df)
        user_info['prediction_table_data'] = prediction_table_data
        user_info['current_round_number'] = round_number

        round_info: dict[str, Any] = {
            'round_number': round_number,
            'prediction_window_start': user_info['prediction_window_start'],
            'prediction_window_size': user_info['prediction_window_size'],
            'prediction_table_data': prediction_table_data,
            'format': str(user_info.get('format') or ''),
            'is_example_data': bool(user_info.get('is_example_data', True)),
            'data_source_name': str(user_info.get('data_source_name', 'example.csv')),
        }
        rounds.append(round_info)
        user_info['rounds'] = rounds
        
        # Debug: Check what predictions we have
        prediction_count = current_df.filter(pl.col("prediction") != 0.0).height
        print(f"DEBUG: Submit button - Found {prediction_count} predictions in current_df")
        print(f"DEBUG: Submit button - Sample predictions: {current_df.filter(pl.col('prediction') != 0.0).select(['time', 'prediction']).head(5).to_dicts()}")

        # Save exactly once when finishing the study (round 12 or user exits early)
        play_only = bool(user_info.get('consent_play_only'))
        if (not play_only) and round_number >= max_rounds and not bool(user_info.get('statistics_saved')):
            submit_component.save_statistics(current_full_df, user_info)
            user_info['statistics_saved'] = True
        
        # Update chart mode to show ground truth and return the full window with ground truth
        chart_mode = {'hide_last_hour': False}
        
        # Convert the current DataFrame back to dict for the store
        def convert_df_to_dict(df_in: pl.DataFrame) -> Dict[str, List[Any]]:
            return {
                'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
                'gl': df_in.get_column('gl').to_list(),
                'prediction': df_in.get_column('prediction').to_list(),
                'age': df_in.get_column('age').to_list(),
                'user_id': df_in.get_column('user_id').to_list()
            }
        
        return '/ending', user_info, chart_mode, convert_df_to_dict(current_df)

    return no_update, no_update, no_update, no_update


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True)],
    [Input('next-round-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_next_round_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict]
) -> Tuple[str, Dict[str, Any], Dict[str, bool], Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int]:
    print(f"DEBUG handle_next_round_button FIRED: n_clicks={n_clicks}")
    if not n_clicks or not user_info:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    next_round_number = len(rounds) + 1
    if next_round_number > max_rounds:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    with start_action(action_type=u"handle_next_round_button", next_round=next_round_number):
        fmt = str(user_info.get("format") or "A")
        points = int(user_info.get('prediction_window_size') or DEFAULT_POINTS)
        points = max(MIN_POINTS, min(MAX_POINTS, points))

        # Choose dataset based on format.
        is_example: bool
        source_name: str
        if fmt == "A":
            full_df, events_df = load_glucose_data()
            is_example = True
            source_name = "example.csv"
        elif fmt == "B":
            uploaded_path = user_info.get("uploaded_data_path")
            if not uploaded_path:
                # Should not happen in normal flow, but keep safe empty state.
                return '/prediction', user_info, {'hide_last_hour': True}, no_update, no_update, no_update, False, "", False, 0
            full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
            is_example = False
            source_name = str(user_info.get("uploaded_data_filename") or user_info.get("data_source_name") or "uploaded.csv")
        else:
            # Format C: alternate between uploaded (odd rounds) and example (even rounds)
            uploaded_path = user_info.get("uploaded_data_path")
            if not uploaded_path:
                return '/prediction', user_info, {'hide_last_hour': True}, no_update, no_update, no_update, False, "", False, 0
            use_example = (next_round_number % 2 == 0)
            if use_example:
                full_df, events_df = load_glucose_data()
                is_example = True
                source_name = "example.csv"
            else:
                full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
                is_example = False
                source_name = str(user_info.get("uploaded_data_filename") or user_info.get("data_source_name") or "uploaded.csv")

        # Reset any previous predictions before starting a fresh round.
        full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))

        used_starts: set[int] = {
            int(r["prediction_window_start"])
            for r in rounds
            if r.get("prediction_window_start") is not None
        }
        new_df, random_start = get_random_data_window(full_df, points, used_starts=used_starts)
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))

        user_info['current_round_number'] = next_round_number
        user_info['is_example_data'] = is_example
        user_info['data_source_name'] = source_name
        chart_mode = {'hide_last_hour': True}

        return (
            '/prediction',
            user_info,
            chart_mode,
            convert_df_to_dict(full_df),
            convert_df_to_dict(new_df),
            convert_events_df_to_dict(events_df),
            is_example,
            source_name,
            False,  # let slider init set it from initial-slider-value
            random_start
        )


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True)],
    [Input('finish-study-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_finish_study_from_prediction(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict]
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool]]:
    print(f"DEBUG handle_finish_study_from_prediction FIRED: n_clicks={n_clicks}")
    if not n_clicks:
        return no_update, no_update, no_update

    with start_action(action_type=u"handle_finish_study_from_prediction", n_clicks=int(n_clicks)):
        pass

    if not user_info:
        return '/final', None, {'hide_last_hour': True}

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    if not rounds:
        return '/final', user_info, {'hide_last_hour': True}

    play_only = bool(user_info.get('consent_play_only')) if user_info else False
    if full_df_data and (not play_only) and not bool(user_info.get('statistics_saved')):
        with start_action(action_type=u"handle_finish_study_from_prediction"):
            full_df = reconstruct_dataframe_from_dict(full_df_data)
            submit_component.save_statistics(full_df, user_info)
            user_info['statistics_saved'] = True

    return '/final', user_info, {'hide_last_hour': False}


@app.callback(
    Output('confirm-ending-finish-exit', 'displayed'),
    Input('finish-study-button-ending', 'n_clicks'),
    prevent_initial_call=True
)
def confirm_finish_study_from_ending(n_clicks: Optional[int]) -> bool:
    """Ask for confirmation before Finish/Exit clears storage from ending."""
    if not n_clicks:
        raise PreventUpdate
    return True


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('clean-storage-flag', 'data', allow_duplicate=True)],
    [Input('confirm-ending-finish-exit', 'submit_n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_finish_study_from_ending(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict]
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool], bool]:
    print(f"DEBUG handle_finish_study_from_ending FIRED: n_clicks={n_clicks}")
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    with start_action(action_type=u"handle_finish_study_from_ending", n_clicks=int(n_clicks)):
        pass

    if not user_info:
        return '/', None, {'hide_last_hour': True}, True

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    rounds_played = len(rounds)
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    current_round_number = int(user_info.get('current_round_number') or rounds_played)
    is_last_round = current_round_number >= max_rounds

    play_only = bool(user_info.get('consent_play_only')) if user_info else False
    if full_df_data and (not play_only) and not bool(user_info.get('statistics_saved')):
        with start_action(action_type=u"handle_finish_study_from_ending"):
            full_df = reconstruct_dataframe_from_dict(full_df_data)
            submit_component.save_statistics(full_df, user_info)
            user_info['statistics_saved'] = True

    if is_last_round:
        return '/final', user_info, {'hide_last_hour': False}, False
    else:
        return '/', None, {'hide_last_hour': True}, True


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True)],
    Input('back-to-final-from-upload', 'n_clicks'),
    prevent_initial_call=True,
)
def handle_back_to_final_from_upload(n_clicks: Optional[int]) -> Tuple[str, Dict[str, bool]]:
    if n_clicks:
        return '/final', {'hide_last_hour': False}
    raise PreventUpdate


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('glucose-unit', 'data', allow_duplicate=True),
     Output('interface-language', 'data', allow_duplicate=True),
     Output('last-visited-page', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('clean-storage-flag', 'data', allow_duplicate=True),
     Output('session-active', 'data', allow_duplicate=True)],
    [Input('restart-button', 'n_clicks')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_restart_button(n_clicks: Optional[int], user_info: Optional[Dict[str, Any]]) -> tuple:
    """Handle restart button — keep consent but restart game from /prediction."""
    print(f"DEBUG handle_restart_button FIRED: n_clicks={n_clicks}")
    if not n_clicks:
        raise PreventUpdate
    with start_action(action_type=u"handle_restart_button") as action:
        action.log(message_type="restart_clicked")

    if not user_info:
        return (
            '/',                       # url pathname
            None,                      # user-info-store
            {'hide_last_hour': True},  # glucose-chart-mode
            False,                     # randomization-initialized
            no_update,                 # glucose-unit
            no_update,                 # interface-language
            None,                      # last-visited-page
            None,                      # full-df
            None,                      # current-window-df
            None,                      # events-df
            True,                      # is-example-data
            'example.csv',             # data-source-name
            None,                      # initial-slider-value
            True,                      # clean-storage-flag
            True,                      # session-active
        )

    info = dict(user_info)
    target_format = str(info.get("format") or "A")

    # Archive the current run so it appears in the final screen rankings
    current_fmt = target_format
    rounds_now = info.get("rounds") or []
    if current_fmt and rounds_now:
        runs_by_format = dict(info.get("runs_by_format") or {})
        runs_list = list(runs_by_format.get(current_fmt) or [])
        runs_list.append({
            "run_id": str(uuid.uuid4()),
            "format": current_fmt,
            "active_run_id": str(info.get("run_id") or ""),
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rounds": rounds_now,
            "rounds_played": int(len(rounds_now)),
            "uses_cgm": bool(info.get("uses_cgm", False)),
            "consent_use_uploaded_data": bool(info.get("consent_use_uploaded_data", False)),
            "is_example_data": bool(info.get("is_example_data", True)),
            "data_source_name": str(info.get("data_source_name") or ""),
        })
        runs_by_format[current_fmt] = runs_list
        info["runs_by_format"] = runs_by_format

    # Reset current run state
    info["run_id"] = str(uuid.uuid4())
    info["rounds"] = []
    info["current_round_number"] = 1
    info["last_submit_round_number"] = 0
    info["last_submit_n_clicks"] = 0
    info["prediction_table_data"] = None
    info["prediction_window_start"] = None
    info["prediction_window_size"] = None
    info["statistics_saved"] = False

    points = int(info.get("prediction_window_size") or DEFAULT_POINTS)
    points = max(MIN_POINTS, min(MAX_POINTS, points))
    uploaded_path = info.get("uploaded_data_path")
    chart_mode = {'hide_last_hour': True}

    if target_format == "A" or (target_format in ("B", "C") and not uploaded_path):
        full_df, events_df = load_glucose_data()
        full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
        new_df, random_start = get_random_data_window(full_df, points)
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
        info["is_example_data"] = True
        info["data_source_name"] = "example.csv"
        return (
            '/prediction',             # url pathname
            info,                      # user-info-store
            chart_mode,                # glucose-chart-mode
            False,                     # randomization-initialized
            no_update,                 # glucose-unit
            no_update,                 # interface-language
            '/prediction',             # last-visited-page
            convert_df_to_dict(full_df),      # full-df
            convert_df_to_dict(new_df),       # current-window-df
            convert_events_df_to_dict(events_df), # events-df
            True,                      # is-example-data
            'example.csv',             # data-source-name
            random_start,              # initial-slider-value
            False,                     # clean-storage-flag
            True,                      # session-active
        )
    else:
        full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
        full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
        new_df, random_start = get_random_data_window(full_df, points)
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
        source_name = str(info.get("uploaded_data_filename") or info.get("data_source_name") or "uploaded.csv")
        info["is_example_data"] = False
        info["data_source_name"] = source_name
        return (
            '/prediction',             # url pathname
            info,                      # user-info-store
            chart_mode,                # glucose-chart-mode
            False,                     # randomization-initialized
            no_update,                 # glucose-unit
            no_update,                 # interface-language
            '/prediction',             # last-visited-page
            convert_df_to_dict(full_df),      # full-df
            convert_df_to_dict(new_df),       # current-window-df
            convert_events_df_to_dict(events_df), # events-df
            False,                     # is-example-data
            source_name,               # data-source-name
            random_start,              # initial-slider-value
            False,                     # clean-storage-flag
            True,                      # session-active
        )


@app.callback(
    [
        Output('url', 'pathname', allow_duplicate=True),
        Output('user-info-store', 'data', allow_duplicate=True),
        Output('glucose-chart-mode', 'data', allow_duplicate=True),
        Output('full-df', 'data', allow_duplicate=True),
        Output('current-window-df', 'data', allow_duplicate=True),
        Output('events-df', 'data', allow_duplicate=True),
        Output('is-example-data', 'data', allow_duplicate=True),
        Output('data-source-name', 'data', allow_duplicate=True),
        Output('randomization-initialized', 'data', allow_duplicate=True),
        Output('initial-slider-value', 'data', allow_duplicate=True),
        Output('switch-format-error', 'children', allow_duplicate=True),
    ],
    [
        Input('switch-format-a', 'n_clicks'),
        Input('switch-format-b', 'n_clicks'),
        Input('switch-format-c', 'n_clicks'),
    ],
    [
        State('user-info-store', 'data'),
        State('interface-language', 'data'),
    ],
    prevent_initial_call=True,
)
def handle_switch_format(
    n_a: Optional[int],
    n_b: Optional[int],
    n_c: Optional[int],
    user_info: Optional[Dict[str, Any]],
    interface_language: Optional[str],
) -> Tuple[
    str,
    Dict[str, Any],
    Dict[str, bool],
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    bool,
    str,
    bool,
    int,
    Optional[Any],
]:
    print(f"DEBUG handle_switch_format FIRED: n_a={n_a} n_b={n_b} n_c={n_c} triggered={ctx.triggered_id}")
    triggered = ctx.triggered_id
    if triggered not in ('switch-format-a', 'switch-format-b', 'switch-format-c'):
        raise PreventUpdate

    triggered_nclicks = {'switch-format-a': n_a, 'switch-format-b': n_b, 'switch-format-c': n_c}[triggered]
    if not triggered_nclicks:
        raise PreventUpdate

    target_format = {'switch-format-a': 'A', 'switch-format-b': 'B', 'switch-format-c': 'C'}[triggered]
    locale = normalize_locale(interface_language)
    info: Dict[str, Any] = dict(user_info or {})

    # Switching into B/C is only available for participants who said they have CGM data.
    # Consent for uploaded CGM data usage is optional and stored as a boolean.
    if target_format in ("B", "C") and not bool(info.get("uses_cgm", False)):
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dbc.Alert(t("ui.switch_format.not_eligible_no_cgm", locale=locale), color="warning"),
        )

    def _archive_current_run(info_in: Dict[str, Any]) -> None:
        current_fmt = str(info_in.get("format") or "")
        rounds_now = info_in.get("rounds") or []
        if not current_fmt or not rounds_now:
            return
        runs_by_format: Dict[str, list[Dict[str, Any]]] = dict(info_in.get("runs_by_format") or {})
        runs_list = list(runs_by_format.get(current_fmt) or [])
        runs_list.append(
            {
                "run_id": str(uuid.uuid4()),
                "format": current_fmt,
                "active_run_id": str(info_in.get("run_id") or ""),
                "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rounds": rounds_now,
                "rounds_played": int(len(rounds_now)),
                "uses_cgm": bool(info_in.get("uses_cgm", False)),
                "consent_use_uploaded_data": bool(info_in.get("consent_use_uploaded_data", False)),
                "is_example_data": bool(info_in.get("is_example_data", True)),
                "data_source_name": str(info_in.get("data_source_name") or ""),
            }
        )
        runs_by_format[current_fmt] = runs_list
        info_in["runs_by_format"] = runs_by_format

    with start_action(action_type=u"handle_switch_format", target=target_format):
        _archive_current_run(info)

        # Reset current run state, keep participant + consent fields.
        info["format"] = target_format
        info["run_id"] = str(uuid.uuid4())
        info["run_format"] = target_format
        info["rounds"] = []
        info["current_round_number"] = 1
        # Reset submit de-dup guards; otherwise first submit in new format can be ignored.
        info["last_submit_round_number"] = 0
        info["last_submit_n_clicks"] = 0
        info["prediction_table_data"] = None
        info["prediction_window_start"] = None
        info["prediction_window_size"] = None
        info["statistics_saved"] = False

        chart_mode = {'hide_last_hour': True}

        points = int(info.get("prediction_window_size") or DEFAULT_POINTS)
        points = max(MIN_POINTS, min(MAX_POINTS, points))

        uploaded_path = info.get("uploaded_data_path")

        if target_format == "A":
            full_df, events_df = load_glucose_data()
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            new_df, random_start = get_random_data_window(full_df, points)
            new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
            info["is_example_data"] = True
            info["data_source_name"] = "example.csv"
            return (
                "/prediction",
                info,
                chart_mode,
                convert_df_to_dict(full_df),
                convert_df_to_dict(new_df),
                convert_events_df_to_dict(events_df),
                True,
                "example.csv",
                False,
                random_start,
                None,
            )

        if target_format in ("B", "C") and uploaded_path:
            full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            new_df, random_start = get_random_data_window(full_df, points)
            new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
            source_name = str(info.get("uploaded_data_filename") or info.get("data_source_name") or "uploaded.csv")
            info["is_example_data"] = False
            info["data_source_name"] = source_name
            return (
                "/prediction",
                info,
                chart_mode,
                convert_df_to_dict(full_df),
                convert_df_to_dict(new_df),
                convert_events_df_to_dict(events_df),
                False,
                source_name,
                False,
                random_start,
                None,
            )

        # Upload-required empty state for B/C.
        info["is_example_data"] = False
        info["data_source_name"] = ""
        return (
            "/prediction",
            info,
            chart_mode,
            None,
            None,
            None,
            False,
            "",
            False,
            0,
            None,
        )

# Add client-side callback to scroll to top when ending page loads
app.clientside_callback(
    """
    function(pathname, consentScrollRequest) {
        // Avoid repeated scrolls on unrelated pathname changes by tracking the last consent request.
        if (typeof window._lastConsentScrollRequest === 'undefined') {
            window._lastConsentScrollRequest = 0;
        }

        if (pathname === '/ending' || pathname === '/final' || pathname === '/startup' || pathname === '/prediction') {
            window.scrollTo(0, 0);
            return '';
        }

        // Only scroll on the *edge* of a consent request (when the value changes),
        // and only while on the prediction page.
        if (pathname === '/prediction' && consentScrollRequest && consentScrollRequest !== window._lastConsentScrollRequest) {
            window._lastConsentScrollRequest = consentScrollRequest;
            // Defer to next tick so layout updates don't immediately re-scroll.
            setTimeout(function() { window.scrollTo(0, 0); }, 0);
            return '';
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('scroll-to-top-trigger', 'children'),
    [Input('url', 'pathname'),
     Input('consent-scroll-request', 'data')]
)

# --- --clean flag: wipe localStorage on first connect ---
# The flag is set via env var by ``uv run start --clean``.  The clientside
# callback runs once (memory-backed store) and clears all Dash-persisted
# localStorage keys so the session starts fresh.  Subsequent tabs or reloads
# against the same running server will also clean, which is the intended
# behaviour: stop the server to stop cleaning.
app.clientside_callback(
    """
    function(shouldClean) {
        if (!shouldClean) { return false; }
        try { window.localStorage.clear(); } catch (e) {}
        return false;
    }
    """,
    Output('clean-storage-flag', 'data', allow_duplicate=True),
    [Input('clean-storage-flag', 'data')],
    prevent_initial_call='initial_duplicate',
)

# --- Page-restore logic for STORAGE_TYPE=local ---
#
# Two responsibilities:
#  1. *Persist* – write the current pathname into ``last-visited-page`` whenever
#     the user navigates to a main-flow page.  Done client-side for speed.
#     We skip the very first write if the pathname is ``/`` so the restore
#     callback (below) has a chance to redirect before the persisted value is
#     overwritten with ``/``.
#  2. *Restore* – on the very first page load, if ``last-visited-page`` holds a
#     non-landing value from a prior session (localStorage), redirect to that
#     page provided enough session state exists to render it.
#
# Ordering guarantee: Dash hydrates ``dcc.Store(storage_type='local')`` from
# the browser *after* the initial server-side render.  The hydration writes to
# the store's ``data`` property, which fires any ``Input`` callbacks.  We use
# ``prevent_initial_call=True`` on the restore callback so it only fires on
# the *hydrated* value, never on the server-default ``None``.

app.clientside_callback(
    """
    function(pathname) {
        // Only persist actual game-flow pages (never "/" – the landing page).
        // This ensures clicking the "Game" navbar link (href="/") does not
        // overwrite the stored last-game-page, so the redirect-back callback
        // can return the user to their in-progress game.
        var persistable = ["/startup", "/prediction", "/ending", "/final"];
        if (persistable.indexOf(pathname) !== -1) {
            return [pathname, true];
        }
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    }
    """,
    [Output('last-visited-page', 'data'),
     Output('session-active', 'data', allow_duplicate=True)],
    [Input('url', 'pathname')],
    prevent_initial_call='initial_duplicate',
)


@app.callback(
    [Output('resume-dialog-target', 'data'),
     Output('page-restore-done', 'data'),
     Output('url', 'pathname', allow_duplicate=True),
     Output('session-active', 'data')],
    [Input('last-visited-page', 'data'),
     Input('user-info-store', 'data'),
     Input('full-df', 'data')],
    [State('page-restore-done', 'data'),
     State('url', 'pathname'),
     State('session-active', 'data')],
    prevent_initial_call=True,
)
def restore_page_on_load(
    last_page: Optional[str],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict],
    already_done: Optional[bool],
    pathname: Optional[str],
    session_active: Optional[bool],
) -> Tuple[Optional[Dict[str, Any]], bool, str, bool]:
    """Restore the user's last game page on load.

    On a **fresh browser session** (``session-active`` is False in
    sessionStorage): show the resume-or-start-over dialog so the user can
    choose.

    On a **tab-switch-back** (``session-active`` is True — the user already
    interacted in this tab and just clicked a navbar link that caused a full
    reload): silently redirect to the last game page without a dialog.

    All three localStorage stores (last-visited-page, user-info-store, full-df)
    are Inputs so the callback re-fires as each store hydrates.  The
    ``page-restore-done`` memory flag prevents action after the first decision.
    """
    if already_done or _is_chart_mode:
        raise PreventUpdate

    if not last_page or last_page == "/":
        return no_update, True, no_update, True

    if pathname and pathname != "/":
        return no_update, True, no_update, True

    if last_page in ("/prediction", "/ending", "/final") and not user_info:
        raise PreventUpdate
    if last_page == "/ending" and not full_df_data:
        raise PreventUpdate

    rounds_played = 0
    current_round = 0
    if user_info:
        rounds_played = len(user_info.get('rounds') or [])
        current_round = int(user_info.get('current_round_number') or (rounds_played + 1))

    with start_action(action_type=u"restore_page_on_load", last_page=last_page, has_user_info=user_info is not None, session_active=bool(session_active)) as action:
        target: Optional[str] = None

        if last_page == "/startup":
            target = "/startup"

        elif last_page == "/prediction":
            target = "/prediction" if user_info else "/startup"

        elif last_page == "/ending":
            has_prediction_data = bool(user_info and "prediction_table_data" in user_info)
            if has_prediction_data and full_df_data:
                target = "/ending"
            elif user_info:
                target = "/prediction"

        elif last_page == "/final":
            if user_info:
                target = "/final"

        if target is None:
            action.log(message_type="no_restorable_target", last_page=last_page)
            return no_update, True, no_update, True

        if session_active:
            action.log(message_type="tab_switch_redirect", target=target)
            return no_update, True, target, True

        action.log(message_type="showing_resume_dialog", target=target, current_round=current_round)
        dialog_data = {
            "target": target,
            "current_round": current_round,
            "max_rounds": MAX_ROUNDS,
        }
        return dialog_data, True, no_update, True


# --- In-session redirect: "Game" navbar link → last game page ---
#
# With ``dcc.Link`` navigation (no full page reload), stores are already
# populated.  When the user clicks "Game" (href="/") while mid-game, this
# callback redirects them back to their last game page immediately.

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input('url', 'pathname')],
    [State('last-visited-page', 'data'),
     State('user-info-store', 'data'),
     State('full-df', 'data')],
    prevent_initial_call=True,
)
def redirect_landing_to_game(
    pathname: Optional[str],
    last_page: Optional[str],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict],
) -> str:
    """Redirect ``/`` → last game page when an active session exists.

    Only fires for in-session client-side navigation (stores are populated).
    On a fresh page load the stores are still ``None`` and ``PreventUpdate``
    lets ``restore_page_on_load`` handle the redirect via hydration.
    """
    if pathname != "/" or _is_chart_mode:
        raise PreventUpdate

    if not last_page or last_page == "/":
        raise PreventUpdate

    if last_page in ("/prediction", "/ending", "/final") and not user_info:
        raise PreventUpdate

    if last_page == "/ending":
        has_ptd = bool(user_info and "prediction_table_data" in user_info)
        if has_ptd and full_df_data:
            return "/ending"
        if user_info:
            return "/prediction"
        raise PreventUpdate

    if last_page == "/final":
        return "/final" if user_info else "/prediction"

    if last_page in ("/startup", "/prediction"):
        return last_page

    raise PreventUpdate


# --- Resume dialog: render, continue, start-over ---

@app.callback(
    Output('resume-dialog-container', 'children'),
    [Input('resume-dialog-target', 'data'),
     Input('interface-language', 'data')],
    prevent_initial_call=True,
)
def render_resume_dialog(
    dialog_data: Optional[Dict[str, Any]],
    interface_language: Optional[str],
) -> List:
    """Render the resume-or-start-over modal when a prior session is detected."""
    if not dialog_data or not dialog_data.get("target"):
        return []

    locale = normalize_locale(interface_language)
    current_round = dialog_data.get("current_round", 0)
    max_rounds = dialog_data.get("max_rounds", MAX_ROUNDS)

    if current_round > 0:
        message = t("ui.resume_dialog.message", locale=locale, round=current_round, total=max_rounds)
    else:
        message = t("ui.resume_dialog.message_no_round", locale=locale)

    overlay_style = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'width': '100vw',
        'height': '100vh',
        'backgroundColor': 'rgba(0,0,0,0.55)',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'zIndex': 10000,
    }
    card_style = {
        'backgroundColor': '#fff',
        'borderRadius': '12px',
        'padding': '36px 40px',
        'maxWidth': '480px',
        'width': '90vw',
        'boxShadow': '0 8px 32px rgba(0,0,0,0.25)',
        'textAlign': 'center',
    }
    title_style = {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'marginBottom': '16px',
        'color': '#333',
    }
    message_style = {
        'fontSize': '16px',
        'lineHeight': '1.5',
        'color': '#555',
        'marginBottom': '28px',
    }
    buttons_style = {
        'display': 'flex',
        'gap': '16px',
        'justifyContent': 'center',
    }

    warning_style = {
        'fontSize': '13px',
        'lineHeight': '1.4',
        'color': '#b5600a',
        'backgroundColor': '#fff8f0',
        'border': '1px solid #f0c88a',
        'borderRadius': '6px',
        'padding': '10px 14px',
        'marginBottom': '24px',
        'textAlign': 'left',
    }

    return [html.Div([
        html.Div([
            html.Div(
                t("ui.resume_dialog.title", locale=locale),
                style=title_style,
                disable_n_clicks=True,
            ),
            html.Div(message, style=message_style, disable_n_clicks=True),
            html.Div(
                t("ui.resume_dialog.warning", locale=locale),
                style=warning_style,
                disable_n_clicks=True,
            ),
            html.Div([
                html.Button(
                    t("ui.resume_dialog.start_over_btn", locale=locale),
                    id='resume-start-over-btn',
                    className='ui red button',
                    style={'minWidth': '140px'},
                ),
                html.Button(
                    t("ui.resume_dialog.continue_btn", locale=locale),
                    id='resume-continue-btn',
                    className='ui green button',
                    style={'minWidth': '140px'},
                ),
            ], style=buttons_style, disable_n_clicks=True),
        ], style=card_style, disable_n_clicks=True),
    ], style=overlay_style, disable_n_clicks=True)]


@app.callback(
    Output('confirm-home-clear-session', 'displayed'),
    Input('home-button', 'n_clicks'),
    prevent_initial_call=True,
)
def confirm_home_button_clear_session(n_clicks: Optional[int]) -> bool:
    """Ask for confirmation before clearing session via navbar Home."""
    if not n_clicks:
        raise PreventUpdate
    return True


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('resume-dialog-container', 'children', allow_duplicate=True),
     Output('resume-dialog-target', 'data', allow_duplicate=True),
     Output('session-active', 'data', allow_duplicate=True)],
    [Input('resume-continue-btn', 'n_clicks')],
    [State('resume-dialog-target', 'data')],
    prevent_initial_call=True,
)
def handle_resume_continue(
    n_clicks: Optional[int],
    dialog_data: Optional[Dict[str, Any]],
) -> Tuple[str, List, None, bool]:
    """Navigate to the saved page when the user clicks Continue."""
    if not n_clicks or not dialog_data:
        raise PreventUpdate
    target = dialog_data.get("target", "/")
    with start_action(action_type=u"resume_continue", target=target) as action:
        action.log(message_type="user_chose_continue")
    return target, [], None, True


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('resume-dialog-container', 'children', allow_duplicate=True),
     Output('resume-dialog-target', 'data', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('glucose-unit', 'data', allow_duplicate=True),
     Output('interface-language', 'data', allow_duplicate=True),
     Output('last-visited-page', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('clean-storage-flag', 'data', allow_duplicate=True),
     Output('session-active', 'data', allow_duplicate=True)],
    [Input('resume-start-over-btn', 'n_clicks'),
     Input('final-exit-button', 'n_clicks')],
    prevent_initial_call=True,
)
def handle_resume_start_over(
    n_resume: Optional[int],
    n_final: Optional[int],
) -> tuple:
    """Reset all in-memory stores and trigger the clean-storage-flag to wipe localStorage."""
    if not n_resume and not n_final:
        raise PreventUpdate
    with start_action(action_type=u"resume_start_over") as action:
        action.log(message_type="user_chose_start_over")
    return (
        "/",                       # url pathname
        [],                        # resume-dialog-container
        None,                      # resume-dialog-target
        None,                      # user-info-store
        {'hide_last_hour': True},  # glucose-chart-mode
        False,                     # randomization-initialized
        'mg/dL',                   # glucose-unit
        'en',                      # interface-language
        None,                      # last-visited-page
        None,                      # full-df
        None,                      # current-window-df
        None,                      # events-df
        True,                      # is-example-data
        'example.csv',             # data-source-name
        None,                      # initial-slider-value
        True,                      # clean-storage-flag (self-resets via clientside callback)
        True,                      # session-active (user made a choice in this tab)
    )


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('resume-dialog-container', 'children', allow_duplicate=True),
     Output('resume-dialog-target', 'data', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('glucose-unit', 'data', allow_duplicate=True),
     Output('interface-language', 'data', allow_duplicate=True),
     Output('last-visited-page', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('clean-storage-flag', 'data', allow_duplicate=True),
     Output('session-active', 'data', allow_duplicate=True)],
    [Input('confirm-home-clear-session', 'submit_n_clicks')],
    prevent_initial_call=True,
)
def handle_home_button(
    n_clicks: Optional[int],
) -> tuple:
    """Reset all in-memory stores and trigger the clean-storage-flag to wipe localStorage when Home is clicked."""
    if not n_clicks:
        raise PreventUpdate
    with start_action(action_type=u"home_button") as action:
        action.log(message_type="user_clicked_home")
    return (
        "/",                       # url pathname
        [],                        # resume-dialog-container
        None,                      # resume-dialog-target
        None,                      # user-info-store
        {'hide_last_hour': True},  # glucose-chart-mode
        False,                     # randomization-initialized
        'mg/dL',                   # glucose-unit
        'en',                      # interface-language
        None,                      # last-visited-page
        None,                      # full-df
        None,                      # current-window-df
        None,                      # events-df
        True,                      # is-example-data
        'example.csv',             # data-source-name
        None,                      # initial-slider-value
        True,                      # clean-storage-flag (self-resets via clientside callback)
        True,                      # session-active (user made a choice in this tab)
    )


## Removed URL-based data writer callback to enforce single-writer for data stores

# Data initialization callback (URL-based only)
@app.callback(
    [Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True)],
    [Input('url', 'pathname')],
    [State('full-df', 'data'),
     State('user-info-store', 'data')],
    prevent_initial_call=True
)
def initialize_data_on_url_change(
    pathname: Optional[str],
    full_df_data: Optional[Dict],
    user_info: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    bool,
    str,
    bool,
    int,
]:
    """Initialize data when URL changes to /prediction without existing data.

    Only loads fresh example data when navigating to /prediction and no data
    exists yet.  All other pages are left alone so that persisted localStorage
    stores are never overwritten (critical for the resume flow).
    """
    _no_change = (no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    if pathname != '/prediction':
        return _no_change

    # For format B/C: require upload, don't auto-load example dataset.
    fmt = str((user_info or {}).get("format") or "A")
    uploaded_path = (user_info or {}).get("uploaded_data_path")
    if fmt in ("B", "C") and not uploaded_path:
        return None, None, None, False, "", False, 0

    # Data already present — preserve (handles resume and round transitions).
    if full_df_data is not None:
        return _no_change

    # First visit to /prediction with no data: load fresh example data.
    full_df, events_df = load_glucose_data()
    df, random_start = get_random_data_window(full_df, DEFAULT_POINTS)
    full_df = full_df.with_columns(pl.lit(0.0).alias('prediction'))
    df = df.with_columns(pl.lit(0.0).alias('prediction'))

    with start_action(action_type=u"initialize_data_on_url_change") as action:
        action.log(message_type="new_random_start", random_start=random_start)

    return (
        convert_df_to_dict(full_df),
        convert_df_to_dict(df),
        convert_events_df_to_dict(events_df),
        True,
        'example.csv',
        False,
        random_start,
    )

# Separate callback for file upload handling
@app.callback(
    [Output('last-click-time', 'data'),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('consent-scroll-request', 'data')],
    [Input('upload-data', 'contents'),
     Input('prediction-data-usage-consent', 'value')],
    [State('upload-data', 'filename'),
     State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_file_upload(
    upload_contents: Optional[str],
    consent_value: Optional[list[str]],
    filename: Optional[str],
    user_info: Optional[Dict[str, Any]],
) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, Dict[str, Any], int]:
    """Handle file upload and data loading"""
    triggered = ctx.triggered_id
    if triggered not in ("upload-data", "prediction-data-usage-consent"):
        raise PreventUpdate

    info_pre: Dict[str, Any] = dict(user_info or {})
    fmt = str(info_pre.get("format") or "A")

    with start_action(action_type=u"handle_file_upload", triggered=str(triggered), filename=filename):
        current_time = int(time.time() * 1000)

        # If consent toggled on prediction page, persist it immediately (sticky),
        # then (optionally) process any cached/pending upload.
        if triggered == "prediction-data-usage-consent":
            if fmt not in ("B", "C"):
                raise PreventUpdate
            has_consent = bool(consent_value and "agree" in consent_value)
            if not has_consent:
                # Ignore attempts to uncheck.
                raise PreventUpdate

            prev_consent = bool(info_pre.get("consent_use_uploaded_data", False))
            pending = info_pre.get("pending_upload_contents")

            if not prev_consent:
                info_pre["consent_use_uploaded_data"] = True
                info_pre["blocked_upload_requires_consent"] = False

                study_id = str(info_pre.get("study_id") or "")
                if study_id:
                    from sugar_sugar.consent import upsert_consent_agreement_fields

                    upsert_consent_agreement_fields(
                        study_id,
                        {
                            "consent_use_uploaded_data": True,
                            "consent_use_uploaded_data_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )
            elif not pending:
                # Loop-breaker: consent was already recorded (prev_consent=True) and
                # there is no pending upload to process, so info_pre is identical to
                # user_info. Returning it would write the same value back to
                # user-info-store, re-triggering update_prediction_uploaded_data_consent_ui,
                # which re-writes prediction-data-usage-consent.value, which triggers
                # this callback again — an infinite server-side loop at ~2 req/s for
                # format B/C users who have already consented on the prediction page.
                raise PreventUpdate

            # If no pending upload, just persist consent in session storage.
            if not pending:
                return (
                    current_time,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    info_pre,
                    current_time,
                )

            # Process cached upload (browser may not re-fire upload for same file).
            upload_contents = str(pending)
            filename = str(info_pre.get("pending_upload_filename") or filename or "")

        if not upload_contents:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
        consent_ok = bool(info_pre.get("consent_use_uploaded_data", False)) or bool(consent_value and "agree" in consent_value)
        if fmt in ("B", "C") and not consent_ok:
            info_pre["blocked_upload_requires_consent"] = True
            # Cache the attempted upload so we can process it immediately after consent is given,
            # without forcing the user to re-upload (browsers often don't fire "change" for same file).
            info_pre["pending_upload_contents"] = upload_contents
            info_pre["pending_upload_filename"] = str(filename or "")
            return (
                current_time,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                info_pre,
                no_update,
            )
        
        # Parse upload contents
        if ',' not in upload_contents:
            print(f"ERROR: Invalid upload format for file {filename}")
            return (
                current_time,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                dict(user_info or {}),
                no_update,
            )
        
        content_type, content_string = upload_contents.split(',', 1)
        decoded = base64.b64decode(content_string)
        
        # Ensure user data directory exists under data/input/users
        users_data_dir = project_root / 'data' / 'input' / 'users'
        users_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = filename.replace(' ', '_').replace('/', '_') if filename else 'uploaded_data'
        if not safe_filename.endswith('.csv'):
            safe_filename += '.csv'
        unique_filename = f"{timestamp}_{safe_filename}"
        
        # Save file to the users data folder
        save_path = users_data_dir / unique_filename
        with open(save_path, 'wb') as f:
            f.write(decoded)
        
        print(f"DEBUG: saved uploaded file to {save_path}")
        
        # Load glucose data - let load_glucose_data handle its own error cases
        new_full_df, new_events_df = load_glucose_data(save_path)
        
        # Start at a random position for uploaded files too
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        new_df, random_start = get_random_data_window(new_full_df, points)
        
        info: Dict[str, Any] = dict(info_pre)
        info["uploaded_data_path"] = str(save_path)
        info["uploaded_data_filename"] = str(filename or "")
        info["is_example_data"] = False
        info["data_source_name"] = str(filename or "")
        info["blocked_upload_requires_consent"] = False
        info.pop("pending_upload_contents", None)
        info.pop("pending_upload_filename", None)

        return (
            current_time,
            convert_df_to_dict(new_full_df),
            convert_df_to_dict(new_df),
            convert_events_df_to_dict(new_events_df),
            False,  # is_example_data = False for uploaded files
            str(filename or ""),  # store the original filename
            False,  # reset randomization flag for new data
            random_start,  # Update initial slider value
            info,
            current_time if triggered == "prediction-data-usage-consent" else no_update,
        )


# Separate callback for example data button
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('time-slider', 'value', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True)],  # Add initial slider value update
    [Input('use-example-data-button', 'n_clicks')],
    prevent_initial_call=True
)
def handle_example_data_button(example_button_clicks: Optional[int]) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, int]:
    """Handle use example data button click"""
    if not example_button_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    with start_action(action_type=u"handle_example_data_button"):
        current_time = int(time.time() * 1000)
        
        # Load fresh example data
        new_full_df, new_events_df = load_glucose_data()
        
        # Start at a random position for example data too
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        new_df, random_start = get_random_data_window(new_full_df, points)
        
        # Reset predictions
        new_full_df = new_full_df.with_columns(pl.lit(0.0).alias("prediction"))
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
        
        print(f"DEBUG: Generated new random start position for example data: {random_start}")
        
        return (current_time, 
               convert_df_to_dict(new_full_df),
               convert_df_to_dict(new_df),
               convert_events_df_to_dict(new_events_df),
               True,  # is_example_data = True for example data
               "example.csv",  # data_source_name for example data
               False,  # reset randomization flag for new data
               random_start,  # Set slider to the random start position
               random_start)  # Update initial slider value


# Separate callback for time slider
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('time-slider', 'value')],
    [State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_time_slider(
    slider_value: Optional[int],
    full_df_data: Optional[Dict],
) -> Tuple[int, Dict[str, List[Any]]]:
    """Handle time slider changes"""
    if slider_value is None or not full_df_data:
        return no_update, no_update
    
    with start_action(action_type=u"handle_time_slider", slider_value=slider_value):
        current_time = int(time.time() * 1000)
        
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        
        # Ensure we don't go beyond the available data
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        max_start = len(full_df) - points
        safe_slider_value = min(slider_value, max_start)
        safe_slider_value = max(0, safe_slider_value)
        
        new_df = full_df.slice(safe_slider_value, points)
        
        return current_time, convert_df_to_dict(new_df)

# Separate callback for glucose graph interactions (only active on prediction page)
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('glucose-graph-graph', 'clickData'),
     Input('glucose-graph-graph', 'relayoutData')],
    [State('last-click-time', 'data'),
     State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('glucose-unit', 'data')],
    prevent_initial_call=True
)
def handle_graph_interactions(click_data: Optional[Dict], relayout_data: Optional[Dict],
                            last_click_time: int, full_df_data: Optional[Dict], 
                            current_df_data: Optional[Dict], glucose_unit: Optional[str]) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Handle glucose graph click and draw interactions"""
    if not full_df_data or not current_df_data:
        return no_update, no_update, no_update
    
    unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'

    def to_mgdl(y_value: float) -> float:
        if unit == 'mmol/L':
            return float(y_value) * GLUCOSE_MGDL_PER_MMOLL
        return float(y_value)

    current_time = int(time.time() * 1000)
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    df = reconstruct_dataframe_from_dict(current_df_data)
    predictions_values = df.get_column("prediction").to_list()
    visible_points = len(df) - PREDICTION_HOUR_OFFSET
    
    
    def snap_index(x_value: Optional[float]) -> Optional[int]:
        """Snap a drawn x-coordinate to the nearest data index while respecting prediction bounds."""
        if x_value is None:
            return None
        snapped_idx = int(round(float(x_value)))
        snapped_idx = max(0, min(snapped_idx, len(df) - 1))
        if snapped_idx < visible_points and predictions_values[snapped_idx] == 0.0:
            return None
        return snapped_idx
    
    if click_data:
        if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            df = df.with_columns(pl.lit(0.0).alias("prediction"))
            
            return (current_time,
                   convert_df_to_dict(full_df),
                   convert_df_to_dict(df))
        
        point_data = click_data['points'][0]
        click_x = point_data['x']
        click_y = point_data['y']
        snapped_idx = snap_index(float(click_x))
        if snapped_idx is None:
            return no_update, no_update, no_update
        nearest_time = df.get_column("time")[snapped_idx]
        
        # Check if this is the first prediction point at the boundary - snap to ground truth
        prediction_y = to_mgdl(float(click_y))
        if snapped_idx == visible_points:  # First point in hidden area
            # Check if this is the start of a new prediction sequence
            existing_predictions = df.filter(pl.col("prediction") != 0.0).height
            if existing_predictions == 0:  # No existing predictions, snap to ground truth
                ground_truth_y = df.get_column("gl")[snapped_idx]
                prediction_y = ground_truth_y
        
        full_df = full_df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(prediction_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        df = df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(prediction_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        
        return (current_time,
               convert_df_to_dict(full_df),
               convert_df_to_dict(df))
    
    elif relayout_data and 'shapes' in relayout_data:
        shapes = relayout_data['shapes']
        if shapes and len(shapes) > 0:
            latest_shape = shapes[-1]
            
            start_x = latest_shape.get('x0')
            end_x = latest_shape.get('x1')
            start_y = latest_shape.get('y0')
            end_y = latest_shape.get('y1')
            
            if all(v is not None for v in [start_x, end_x, start_y, end_y]):
                start_idx = snap_index(float(start_x))
                end_idx = snap_index(float(end_x))
                if start_idx is None or end_idx is None:
                    return (
                        last_click_time,
                        convert_df_to_dict(full_df),
                        convert_df_to_dict(df)
                    )
                
                start_time = df.get_column("time")[start_idx]
                
                # Check if this is the first prediction starting at the boundary - snap to ground truth
                actual_start_y = to_mgdl(float(start_y))
                if start_idx == visible_points:  # Starting at first point in hidden area
                    # Check if this is the start of a new prediction sequence
                    existing_predictions = df.filter(pl.col("prediction") != 0.0).height
                    if existing_predictions == 0:  # No existing predictions, snap to ground truth
                        ground_truth_y = df.get_column("gl")[start_idx]
                        actual_start_y = ground_truth_y
                
                # Calculate the intersection with the first vertical guideline after start
                actual_end_x, actual_end_y = calculate_first_guideline_intersection(
                    float(start_idx), float(actual_start_y), float(end_idx), to_mgdl(float(end_y)), df
                )
                snapped_end_idx = snap_index(actual_end_x)
                if snapped_end_idx is None:
                    return (
                        last_click_time,
                        convert_df_to_dict(full_df),
                        convert_df_to_dict(df)
                    )
                end_time = df.get_column("time")[snapped_end_idx]
                
                # Get intermediate prediction points every 5 minutes
                intermediate_points = create_intermediate_predictions(start_time, end_time, float(actual_start_y), float(actual_end_y), df)
                
                # Collect all times that need prediction values
                all_prediction_times = [start_time, end_time]
                all_prediction_values = [float(actual_start_y), float(actual_end_y)]
                
                # Add intermediate points
                for time_point, glucose_value in intermediate_points:
                    all_prediction_times.append(time_point)
                    all_prediction_values.append(glucose_value)
                
                # Create a mapping for the predictions
                time_to_value = dict(zip(all_prediction_times, all_prediction_values))
                
                # Update both DataFrames with all prediction points
                full_df = full_df.with_columns(
                    pl.when(pl.col("time").is_in(all_prediction_times))
                    .then(
                        # Use a series of when conditions to map each time to its value
                        pl.when(pl.col("time") == start_time)
                        .then(float(actual_start_y))
                        .when(pl.col("time") == end_time)
                        .then(float(actual_end_y))
                        .otherwise(
                            # For intermediate points, we need to match them individually
                            pl.col("time").map_elements(
                                lambda x: time_to_value.get(x, 0.0),
                                return_dtype=pl.Float64
                            )
                        )
                    )
                    .otherwise(pl.col("prediction"))
                    .alias("prediction")
                )
                df = df.with_columns(
                    pl.when(pl.col("time").is_in(all_prediction_times))
                    .then(
                        # Use a series of when conditions to map each time to its value
                        pl.when(pl.col("time") == start_time)
                        .then(float(actual_start_y))
                        .when(pl.col("time") == end_time)
                        .then(float(actual_end_y))
                        .otherwise(
                            # For intermediate points, we need to match them individually
                            pl.col("time").map_elements(
                                lambda x: time_to_value.get(x, 0.0),
                                return_dtype=pl.Float64
                            )
                        )
                    )
                    .otherwise(pl.col("prediction"))
                    .alias("prediction")
                )
                
                return (current_time,
                       convert_df_to_dict(full_df),
                       convert_df_to_dict(df))
    
    return no_update, no_update, no_update

@app.callback(
    Output('data-source-display', 'children'),
    [Input('url', 'pathname'),
     Input('data-source-name', 'data'),
     Input('user-info-store', 'data'),
     Input('interface-language', 'data')],
    prevent_initial_call=True
)
def update_data_source_display(
    pathname: str,
    source_name: Optional[str],
    user_info: Optional[Dict[str, Any]],
    interface_language: Optional[str],
) -> str:
    """Update the visible data source label only when on the prediction page."""
    if pathname != '/prediction':
        raise PreventUpdate
    if source_name:
        return source_name
    fmt = str((user_info or {}).get("format") or "A")
    if fmt in ("B", "C"):
        return t("ui.header.upload_required", locale=normalize_locale(interface_language))
    return "example.csv"


@app.callback(
    Output("generic-source-metadata-display", "children"),
    [
        Input("url", "pathname"),
        Input("data-source-name", "data"),
        Input("interface-language", "data"),
    ],
    prevent_initial_call=False,
)
def update_generic_source_metadata_display(
    pathname: str,
    source_name: Optional[str],
    interface_language: Optional[str],
) -> str:
    if pathname != "/prediction":
        return ""

    key = Path(str(source_name or "example.csv")).name
    meta = GENERIC_SOURCES_METADATA.get(key)
    if meta is None:
        return ""

    locale = normalize_locale(interface_language)
    gender_raw = str(meta.gender or "").strip().lower()
    if gender_raw in ("male", "female", "na"):
        gender_display = t(f"ui.startup.gender_{gender_raw}", locale=locale)
    else:
        gender_display = meta.gender

    return (
        f"{t('ui.startup.age_label', locale=locale)}: {meta.age} · "
        f"{t('ui.startup.gender_label', locale=locale)}: {gender_display} · "
        f"{t('ui.header.weight_label', locale=locale)}: {meta.weight}"
    )

# Add callback for random slider initialization when prediction page components are ready
@app.callback(
    [Output('time-slider', 'value', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True)],
    [Input('time-slider', 'max')],  # Triggers when slider is created and max is set
    [State('url', 'pathname'),
     State('full-df', 'data'),
     State('randomization-initialized', 'data'),
     State('initial-slider-value', 'data')],
    prevent_initial_call=True
)
def randomize_slider_on_prediction_page(slider_max: int, pathname: str, full_df_data: Optional[Dict], 
                                       randomization_initialized: bool, 
                                       initial_slider_value: Optional[int]) -> Tuple[int, bool]:
    """Set slider to a random valid window start when slider mounts on prediction page. Returns slider value and updated randomization flag."""
    if pathname == '/prediction' and full_df_data and slider_max is not None and not randomization_initialized:
        # Use the stored initial slider value if available
        if initial_slider_value is not None:
            return initial_slider_value, True
        # Otherwise generate a new random start
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        _, random_start = get_random_data_window(full_df, points)
        return random_start, True  # Set randomization flag to True after randomizing
    return no_update, no_update


# Separate UI callback for upload success message
@app.callback(
    Output('example-data-warning', 'children'),
    [Input('upload-data', 'contents'),
     Input('interface-language', 'data'),
     Input('user-info-store', 'data')],
    [State('upload-data', 'filename'),
     State('is-example-data', 'data')],
    prevent_initial_call=True
)
def update_upload_success_message(
    upload_contents: Optional[str],
    interface_language: Optional[str],
    filename: Optional[str],
    is_example_data: Optional[bool],
    user_info: Optional[Dict[str, Any]],
) -> Optional[html.Div]:
    """Show success message when file is uploaded"""
    if not upload_contents:
        return no_update

    info = dict(user_info or {})
    fmt = str(info.get("format") or "A")
    consent_ok = bool(info.get("consent_use_uploaded_data", False))
    if fmt in ("B", "C") and (not consent_ok):
        return html.Div(
            t("ui.startup.data_usage_consent_required", locale=normalize_locale(interface_language)),
            style={
                'color': '#7f1d1d',
                'backgroundColor': '#fee2e2',
                'padding': '10px',
                'borderRadius': '5px',
                'textAlign': 'center',
            },
        )
    
    if not is_example_data:  # File was successfully uploaded
        return html.Div([
            html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
            t("ui.header.upload_success", locale=normalize_locale(interface_language), filename=filename or "")
        ], style={
            'color': '#2f855a',
            'backgroundColor': '#c6f6d5',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
    return None


# Separate UI callback for example data button message and upload reset
@app.callback(
    [Output('example-data-warning', 'children', allow_duplicate=True),
     Output('time-slider', 'max', allow_duplicate=True),
     Output('upload-data', 'contents', allow_duplicate=True),  # Reset upload contents
     Output('upload-data', 'filename', allow_duplicate=True)],  # Reset filename
    [Input('use-example-data-button', 'n_clicks')],
    [State('full-df', 'data'),
     State('interface-language', 'data')],
    prevent_initial_call=True
)
def reset_upload_on_example_data(
    example_button_clicks: Optional[int],
    full_df_data: Optional[Dict],
    interface_language: Optional[str],
) -> Tuple[Optional[html.Div], int, None, None]:
    """Reset upload component and show message when example data button is clicked"""
    if not example_button_clicks or not full_df_data:
        return no_update, no_update, no_update, no_update
    
    with start_action(action_type=u"reset_upload_on_example_data"):
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        new_max = len(full_df) - points
        
        print("DEBUG: Resetting upload component to allow re-upload of same file")
        
        # Show message that we're now using example data
        example_msg = html.Div([
            html.I(className="fas fa-info-circle", style={'marginRight': '8px'}),
            t("ui.header.example_data_now_using", locale=normalize_locale(interface_language))
        ], style={
            'color': '#0c5460',
            'backgroundColor': '#d1ecf1',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
        
        # Reset upload component by clearing contents and filename
        # This allows the same file to be uploaded again after switching to example data
        return example_msg, new_max, None, None

def convert_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert a Polars DataFrame to a session-store dictionary."""
    return {
        'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'gl': df.get_column('gl').to_list(),
        'prediction': df.get_column('prediction').to_list(),
        'age': df.get_column('age').to_list(),
        'user_id': df.get_column('user_id').to_list()
    }

def convert_events_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert an events Polars DataFrame to a session-store dictionary."""
    return {
        'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'event_type': df.get_column('event_type').to_list(),
        'event_subtype': df.get_column('event_subtype').to_list(),
        'insulin_value': df.get_column('insulin_value').to_list()
    }

def reconstruct_dataframe_from_dict(df_data: Dict[str, List[Any]]) -> pl.DataFrame:
    """Safely reconstruct a Polars DataFrame from a dictionary with proper type handling."""
    return pl.DataFrame({
        'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
        'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
        'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
        'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
    })

def calculate_first_guideline_intersection(start_x: float, start_y: float, end_x: float, end_y: float, df: pl.DataFrame) -> Tuple[float, float]:
    """
    Calculate the intersection of the drawn line with the first vertical guideline after the start point.
    Returns the (x, y) coordinates of the intersection with the next time marker.
    """
    # Find the next integer x position (vertical guideline) after start_x
    next_x = int(start_x) + 1
    
    # If the line doesn't extend past the next guideline, use the original end point
    if next_x >= end_x:
        return end_x, end_y
    
    # Make sure the next_x is within the DataFrame bounds
    if next_x >= len(df):
        next_x = len(df) - 1
    
    # Calculate the y-value at the intersection using linear interpolation
    if end_x != start_x:  # Avoid division by zero
        slope = (end_y - start_y) / (end_x - start_x)
        intersect_y = start_y + slope * (next_x - start_x)
    else:
        intersect_y = start_y
    
    return float(next_x), float(intersect_y)


def create_intermediate_predictions(start_time: datetime, end_time: datetime, start_y: float, end_y: float, df: pl.DataFrame) -> List[Tuple[datetime, float]]:
    """
    Create intermediate prediction points every 5 minutes between start and end points.
    Returns a list of (time, glucose_value) tuples for intermediate points.
    """
    intermediate_points = []
    time_diff = end_time - start_time
    
    # Only create intermediate points if the difference is more than 5 minutes
    if time_diff.total_seconds() <= 5 * 60:  # 5 minutes in seconds
        return intermediate_points
    
    # Get all available times in the DataFrame between start and end
    available_times = (df
        .filter((pl.col("time") > start_time) & (pl.col("time") < end_time))
        .get_column("time")
        .to_list()
    )
    
    if not available_times:
        return intermediate_points
    
    # Calculate the total time range in minutes for interpolation
    total_minutes = time_diff.total_seconds() / 60
    
    # Create prediction points for times that are approximately every 5 minutes
    target_interval = 5  # minutes
    for time_point in available_times:
        # Calculate how far along we are in the time range (0 to 1)
        time_from_start = time_point - start_time
        progress = time_from_start.total_seconds() / time_diff.total_seconds()
        
        # Check if this time point is approximately at a 5-minute interval
        minutes_from_start = time_from_start.total_seconds() / 60
        
        # Add point if it's close to a 5-minute interval (within 2.5 minutes)
        nearest_interval = round(minutes_from_start / target_interval) * target_interval
        if abs(minutes_from_start - nearest_interval) <= 2.5 and nearest_interval > 0 and nearest_interval < total_minutes:
            # Interpolate the glucose value
            interpolated_value = start_y + (end_y - start_y) * progress
            intermediate_points.append((time_point, interpolated_value))
    
    return intermediate_points


def find_nearest_time(x: Union[str, float, datetime], df: pl.DataFrame) -> datetime:
    """
    Finds the nearest allowed time from the DataFrame 'df' for a given x-coordinate.
    x can be either an index (float) or a timestamp string.
    """
    if isinstance(x, (int, float)):
        # If x is a numerical index, round to nearest integer and get corresponding time
        idx = round(float(x))
        idx = max(0, min(idx, len(df) - 1))  # Ensure index is within bounds
        return df.get_column("time")[idx]
    
    # If x is a timestamp string, convert to datetime
    if isinstance(x, str):
        x_ts = datetime.fromisoformat(x.replace('Z', '+00:00'))
    else:
        x_ts = x
    time_diffs = df.select([
        (pl.col("time").cast(pl.Int64) - pl.lit(int(x_ts.timestamp() * 1000)))
        .abs()
        .alias("diff")
    ])
    nearest_idx = time_diffs.select(pl.col("diff").arg_min()).item()
    return df.get_column("time")[nearest_idx]



def _register_all_callbacks() -> None:
    """Register all Dash component callbacks (shared by ``main`` and ``chart``)."""
    global startup_page, landing_page
    landing_page = LandingPage()
    startup_page = StartupPage()

    prediction_table.register_callbacks(app)
    metrics_component.register_callbacks(app, prediction_table)
    glucose_chart.register_callbacks(app)
    submit_component.register_callbacks(app)
    landing_page.register_callbacks(app)
    startup_page.register_callbacks(app)
    ending_page.register_callbacks(app)


# Create typer app.  invoke_without_command + the @cli.callback default
# mean ``uv run start`` (no subcommand) still works, while ``uv run chart``
# routes to the ``chart`` subcommand via its own entrypoint.
cli = typer.Typer(invoke_without_command=True)

@cli.callback(invoke_without_command=True)
def main(
    typer_ctx: typer.Context,
    debug: Optional[bool] = typer.Option(None, "--debug", help="Enable debug mode to show test button"),
    host: Optional[str] = typer.Option(None, "--host", help="Host to run the server on"),
    port: Optional[int] = typer.Option(None, "--port", help="Port to run the server on"),
    clean: bool = typer.Option(False, "--clean", help="Clear browser localStorage on first connect so the session starts fresh"),
) -> None:
    """Start the Dash server.

    Defaults come from ``sugar_sugar.config`` (``DASH_*``, ``DEBUG_MODE``). If
    ``--debug`` / ``--no-debug`` is passed, Dash ``debug`` follows it and
    ``config.DEBUG_MODE`` is updated so in-app debug (e.g. test button) stays in sync.
    """
    if typer_ctx.invoked_subcommand is not None:
        return

    if clean:
        os.environ["_CLEAN_STORAGE"] = "1"
        for child in app.layout.children:
            if getattr(child, 'id', None) == 'clean-storage-flag':
                child.data = True
                break

    dash_host = DASH_HOST if host is None else (host or DASH_HOST)
    dash_port = DASH_PORT if port is None else port
    dash_debug = DASH_DEBUG if debug is None else debug
    if debug is not None:
        sugar_sugar_config.DEBUG_MODE = debug

    _register_all_callbacks()
    app.layout.children[-1].children = [landing_page]
    
    with start_action(
        action_type=u"start_dash_server",
        host=dash_host,
        port=dash_port,
        debug=dash_debug,
        clean=clean
    ):
        app.run(host=dash_host, port=dash_port, debug=dash_debug)

@cli.command()
def chart(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="CSV file to load (Dexcom/Libre/Medtronic/Nightscout). Default: built-in example."),
    points: int = typer.Option(DEFAULT_POINTS, "--points", "-p", help="Number of data points in the window"),
    start: Optional[int] = typer.Option(None, "--start", "-s", help="Start index for the data window (default: random)"),
    unit: str = typer.Option("mg/dL", "--unit", "-u", help="Glucose unit: mg/dL or mmol/L"),
    locale: str = typer.Option("en", "--locale", "-l", help="UI locale (en, de, uk, ro)"),
    prefill: bool = typer.Option(False, "--prefill", help="Pre-fill predictions with noisy ground truth so submit/ending can be tested immediately"),
    noise: float = typer.Option(0.05, "--noise", help="Noise level for --prefill (fraction of gl value, e.g. 0.05 = +/-5%%)"),
    host: Optional[str] = typer.Option(None, "--host", help="Host to run the server on"),
    port: Optional[int] = typer.Option(None, "--port", help="Port to run the server on"),
) -> None:
    """Dev shortcut: load data and jump straight to the prediction chart.

    Bypasses landing, startup, and consent pages. Equivalent to filling in the
    form, clicking "Just Test Me", and pressing "Start Prediction" -- but
    instant.  Accepts an external CSV so you can iterate on real data without
    uploading through the UI every time.

    With --prefill the prediction region is filled with noisy ground-truth
    values so you can test submit/ending/metrics without drawing.
    """
    # Set env vars so the module-level data loading picks them up on
    # Werkzeug debug-reloader re-imports.
    os.environ["_CHART_MODE"] = "1"
    if file:
        os.environ["_CHART_FILE"] = str(file)
    os.environ["_CHART_POINTS"] = str(points)
    if start is not None:
        os.environ["_CHART_START"] = str(start)
    os.environ["_CHART_UNIT"] = unit if unit in ("mg/dL", "mmol/L") else "mg/dL"
    os.environ["_CHART_LOCALE"] = normalize_locale(locale)
    os.environ["_CHART_SOURCE"] = file.name if file else "example.csv"
    if prefill:
        os.environ["_CHART_PREFILL"] = "1"
        os.environ["_CHART_NOISE"] = str(noise)

    sugar_sugar_config.DEBUG_MODE = True

    _register_all_callbacks()

    dash_host = DASH_HOST if host is None else (host or DASH_HOST)
    dash_port = DASH_PORT if port is None else port

    with start_action(
        action_type=u"start_chart_dev",
        file=str(file) if file else "example.csv",
        points=points,
        prefill=prefill,
        host=dash_host,
        port=dash_port,
    ):
        app.run(host=dash_host, port=dash_port, debug=True)


def cli_main() -> None:
    """CLI entry point"""
    cli()


def chart_main() -> None:
    """CLI entry point that defaults to the ``chart`` command."""
    cli(["chart"] + sys.argv[1:])


if __name__ == '__main__':
    cli()
