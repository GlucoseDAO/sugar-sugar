from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
from html import escape as html_escape
from io import BytesIO
import dash
from dash import dcc, html, Output, Input, State, no_update, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import polars as pl
from datetime import datetime
import time
from pathlib import Path
import math
import base64
import dash_bootstrap_components as dbc
import os
import sys
import typer
from flask import Response, send_file as flask_send_file, request as flask_request
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


def _configure_eliot_logging() -> None:
    """Install human-readable Eliot log renderers unless explicitly disabled."""
    if os.environ.get("SUGAR_SUGAR_DISABLE_NICE_LOGS") == "1":
        return

    to_nice_stdout()
    to_nice_file(logs_dir / 'sugar_sugar.json', logs_dir / 'sugar_sugar.log')


_configure_eliot_logging()

from sugar_sugar.i18n import setup_i18n, normalize_locale, t, t_list, t_raw
setup_i18n()

from sugar_sugar.data import load_glucose_data, load_glucose_data_from_nightscout
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
    DEPLOY_URL,
    DEPLOY_BUILD,
    MAX_ROUNDS,
    MIN_USEFUL_ROUNDS,
    SHARE_FORMATS,
    SHARE_NAME,
    SHARE_NOISE,
    SHARE_ROUNDS,
    STORAGE_TYPE,
    UMAMI_DOMAINS,
    UMAMI_HOST_URL,
    UMAMI_SCRIPT_URL,
    UMAMI_WEBSITE_ID,
)
import sugar_sugar.config as sugar_sugar_config
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.components.predictions import PredictionTableComponent
from sugar_sugar.components.ag_grid import build_readonly_ag_grid, build_readonly_column_defs
from sugar_sugar.components.startup import StartupPage, StartupPageMobile
from sugar_sugar.components.landing import LandingPage, LandingPageMobile
from sugar_sugar.components.consent_form import ConsentFormPage
from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.encouragement import pick_bracket
from sugar_sugar.components.header import HeaderComponent
from sugar_sugar.components.ending import EndingPage
from sugar_sugar.components.navbar import NavBar, MobileNavBar
from sugar_sugar.components.share import (
    build_share_card_figure,
    create_expired_layout,
    create_share_layout,
)
from sugar_sugar import share_store
from sugar_sugar import resume_store
from sugar_sugar.generic_sources_metadata import load_generic_sources_metadata
from sugar_sugar.contact_info import load_contact_info
from sugar_sugar.static_markdown import static_markdown_autosize_iframe

# Type aliases for clarity
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type

GLUCOSE_MGDL_PER_MMOLL: float = 18.0

FORMAT_ORDER: dict[str, int] = {"C": 0, "B": 1, "A": 2}
GENERIC_SOURCES_METADATA = load_generic_sources_metadata()
from sugar_sugar.ai_models.registry import get_model

def _compute_ai_forecast_if_needed(window_store: dict, user_info: dict) -> dict:
    """
    Verifică dacă utilizatorul joacă în modul VS AI și rulează 
    modelul GluMind pe baza istoricului glicemic vizibil.
    """
    if not user_info or user_info.get("mode") != "vs_ai":
        return {"glumind": {"predictions": [], "ready": False}}
        
    gl_values = window_store.get("gl", [])
    offset = PREDICTION_HOUR_OFFSET
        
    visible_cutoff = len(gl_values) - offset
    if visible_cutoff <= 0:
        return {"glumind": {"predictions": [], "ready": False}}
        
    # Extrage doar valorile din trecut, vizibile omului
    history = [g for g in gl_values[:visible_cutoff] if g is not None]
    
    try:
        model = get_model("glumind")
        preds = model.predict(history, prediction_steps=offset)
        return {
            "glumind": {
                "predictions": preds,
                "ready": True
            }
        }
    except Exception as e:
        # Failsafe în caz de eroare structurală
        fallback_val = history[-1] if history else 100.0
        return {
            "glumind": {
                "predictions": [round(fallback_val, 1)] * offset,
                "ready": True,
                "error": str(e)
            }
        }


def _build_ai_table_rows(
    gl_values: list,
    ai_predictions: list,
    offset: int,
) -> list[dict[str, str]]:
    n = len(gl_values)
    visible_cutoff = n - offset
    pred_row: dict[str, str] = {"metric": "Predicted (AI)"}
    abs_row: dict[str, str] = {"metric": "Absolute Error (AI)"}
    rel_row: dict[str, str] = {"metric": "Relative Error (AI) (%)"}

    for ti in range(n):
        idx_in_pred = ti - visible_cutoff
        a = gl_values[ti] if ti < len(gl_values) else None
        if idx_in_pred < 0 or idx_in_pred >= len(ai_predictions) or a is None:
            pred_row[f"t{ti}"] = abs_row[f"t{ti}"] = rel_row[f"t{ti}"] = "-"
            continue
        p = float(ai_predictions[idx_in_pred])
        pred_row[f"t{ti}"] = f"{p:.1f}"
        err = abs(float(a) - p)
        abs_row[f"t{ti}"] = f"{err:.1f}"
        rel_row[f"t{ti}"] = f"{(err / float(a) * 100):.1f}%" if a != 0 else "-"

    return [pred_row, abs_row, rel_row]


def _extract_metric_values(metrics_dict: Optional[dict]) -> dict[str, Optional[float]]:
    """Aduce {'MAE': {...,'value':x}, ...} la {'MAE': x, ...} simplu de salvat/serializat."""
    out: dict[str, Optional[float]] = {}
    for name in ("MAE", "MSE", "RMSE", "MAPE"):
        entry = (metrics_dict or {}).get(name)
        val = entry.get("value") if entry else None
        out[name] = float(val) if val is not None else None
    return out


def _compute_ai_comparison_scores(
    gl_values: list,
    human_predictions: list,
    ai_predictions: list,
    offset: int,
) -> dict[str, dict[str, Optional[float]]]:
    """Calculează cele 2 seturi de scoruri pentru GluMind: vs_reality și vs_human.

    Refolosește ``MetricsComponent()._calculate_metrics_from_table_data``, aceeași
    funcție folosită deja pentru scorurile omului (ending/final), construind un
    tabel temporar de 2 rânduri ("Actual Glucose" / "Predicted") de fiecare dată:
      - vs_reality: glicemia reală  vs  predicția AI
      - vs_human:   predicția omului  vs  predicția AI (cât de departe e AI de om)
    """
    n = len(gl_values)
    visible_cutoff = n - offset
    metrics_comp = MetricsComponent()

    def _table_for(reference_values: list) -> list[dict[str, str]]:
        actual_row: dict[str, str] = {"metric": "Actual Glucose"}
        pred_row: dict[str, str] = {"metric": "Predicted"}
        for ti in range(offset):
            src_idx = visible_cutoff + ti
            ref = reference_values[src_idx] if 0 <= src_idx < len(reference_values) else None
            p = ai_predictions[ti] if ti < len(ai_predictions) else None
            if ref is None or p is None:
                actual_row[f"t{ti}"] = pred_row[f"t{ti}"] = "-"
            else:
                actual_row[f"t{ti}"] = f"{float(ref):.1f}"
                pred_row[f"t{ti}"] = f"{float(p):.1f}"
        return [actual_row, pred_row]

    if visible_cutoff <= 0 or not ai_predictions:
        empty = {"MAE": None, "MSE": None, "RMSE": None, "MAPE": None}
        return {"vs_reality": dict(empty), "vs_human": dict(empty)}

    vs_reality_table = _table_for(gl_values)
    vs_human_table = _table_for(human_predictions)

    vs_reality = _extract_metric_values(metrics_comp._calculate_metrics_from_table_data(vs_reality_table))
    vs_human = _extract_metric_values(metrics_comp._calculate_metrics_from_table_data(vs_human_table))

    return {"vs_reality": vs_reality, "vs_human": vs_human}


SITE_TITLE: str = "Sugar Sugar"
SITE_DESCRIPTION: str = (
    "Test your glucose prediction skills, compare your forecasts with real CGM data, "
    "and help establish a human baseline for glucose forecasting research."
)
OG_PREVIEW_PATH: str = "/assets/og-card.png"
OG_PREVIEW_VERSION: int = 1
OG_PREVIEW_SIZE: tuple[int, int] = (1200, 630)
# Bump when the share-card PNG design changes so FB/X/LinkedIn re-fetch the
# image instead of serving a stale crop from their own caches.
SHARE_CARD_IMAGE_VERSION: int = 2
PUBLIC_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/", "Sugar Sugar", SITE_DESCRIPTION),
    ("/about", "About Sugar Sugar", "Learn why the Sugar Sugar glucose prediction study matters."),
    ("/faq", "Sugar Sugar FAQ", "Answers to common questions about the Sugar Sugar study and gameplay."),
    ("/demo", "Video Instructions", "Watch how to play the Sugar Sugar glucose prediction game."),
    ("/contact", "Contact GlucoseDAO", "Get in touch with the Sugar Sugar team."),
)


def canonical_base() -> str:
    """Configured public origin without a trailing slash, or empty in local dev."""
    return DEPLOY_URL.strip().rstrip("/")


def _site_og_image_url() -> str:
    """Site-wide OG image URL. Production should set DEPLOY_URL for absolute URLs."""
    path: str = f"{OG_PREVIEW_PATH}?v={OG_PREVIEW_VERSION}"
    base: str = canonical_base()
    return f"{base}{path}" if base else path


def _format_label(format_code: str, *, locale: str) -> str:
    code = str(format_code or "").strip().upper()
    if code == "A":
        return t("ui.startup.format_a_label", locale=locale)
    if code == "B":
        return t("ui.startup.format_b_label", locale=locale)
    if code == "C":
        return t("ui.startup.format_c_label", locale=locale)
    return code


def _rank_from_ranking_csv(
    ranking_path: Path,
    *,
    study_id: str,
    format_filter: Optional[str],
    mode: str,
) -> Optional[tuple[int, int]]:
    """Return ``(rank, total)`` for ``study_id`` against the ranking CSV.

    Extracted from ``create_final_layout`` so the share page can compute and
    freeze rankings into a share record at save time.  ``mode`` is either
    ``"best"`` (keep lowest MAE per study_id) or ``"latest"`` (keep most
    recent MAE by timestamp).  Ranks on ``overall_mae_mgdl`` ascending.
    """
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
        df_pick = df2.group_by('study_id').agg(pl.col('overall_mae_mgdl').min().alias('overall_mae_mgdl'))

    total = df_pick.height
    if total == 0:
        return None
    df_sorted = df_pick.sort(['overall_mae_mgdl', 'study_id'])
    matches = df_sorted.with_row_index('rank_idx').filter(pl.col('study_id') == study_id)
    if matches.height == 0:
        return None
    return int(matches.get_column('rank_idx')[0]) + 1, total


def compute_share_rankings(study_id: str, played_formats: list[str]) -> dict[str, Any]:
    """Freeze the per-format and overall rankings for a study_id.

    Returns a dict with:
      - ``per_format``: ``[{format, rank, total}, ...]`` in FORMAT_ORDER order
      - ``overall``: ``{rank, total}`` or ``None``
    Used by the share callback so the share URL always shows the ranks that
    existed at share time, even if the CSVs are appended to later.
    """
    per_format: list[dict[str, Any]] = []
    ordered: list[str] = sorted(
        {f for f in played_formats if f in ("A", "B", "C")},
        key=lambda x: FORMAT_ORDER.get(str(x), 999),
    )
    for fmt in ordered:
        info = _rank_from_ranking_csv(
            project_root / 'data' / 'input' / f'prediction_ranking_{fmt}.csv',
            study_id=study_id,
            format_filter=fmt,
            mode="best",
        )
        if info is not None:
            rank, total = info
            per_format.append({"format": fmt, "rank": rank, "total": total})

    overall: Optional[dict[str, int]] = None
    overall_info = _rank_from_ranking_csv(
        project_root / 'data' / 'input' / 'prediction_ranking.csv',
        study_id=study_id,
        format_filter="ALL",
        mode="latest",
    )
    if overall_info is not None:
        rank, total = overall_info
        overall = {"rank": rank, "total": total}

    return {"per_format": per_format, "overall": overall}


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

# ---------------------------------------------------------------------------
# Share-mode: generate fake multi-round data, persist a share record, and
# navigate directly to /share/<id> on startup.  Activated by _SHARE_MODE=1
# (set by the ``share`` CLI command).
# ---------------------------------------------------------------------------
_is_share_mode = os.environ.get("_SHARE_MODE") == "1"
_share_mode_id: Optional[str] = None

# Staging mode (prod+): when _STAGING_MODE=1, extra `/staging/*` test routes are
# exposed that jump straight to prefilled prediction / ending / final / share
# states for remote testing, without a full playthrough. The flag defaults off,
# so production is byte-identical. Set by `serve --staging` / `uv run serve-staging`.
_is_staging_mode = os.environ.get("_STAGING_MODE") == "1"

if _is_share_mode:
    import random as _share_rnd
    _share_rounds_n = int(os.environ.get("_SHARE_ROUNDS", str(SHARE_ROUNDS)))
    _share_noise = float(os.environ.get("_SHARE_NOISE", str(SHARE_NOISE)))
    _share_locale = os.environ.get("_SHARE_LOCALE", "en")
    _share_formats_env = os.environ.get("_SHARE_FORMATS", SHARE_FORMATS)
    _share_formats = [f.strip().upper() for f in _share_formats_env.split(",") if f.strip()]
    _share_source = os.environ.get("_SHARE_SOURCE", "example.csv")
    _share_is_example = os.environ.get("_CHART_FILE") is None

    _share_full_df = _init_full_df.clone()
    _share_used_starts: set[int] = set()
    _share_all_rounds: list[dict[str, Any]] = []

    for _ri in range(_share_rounds_n):
        _fmt = _share_formats[_ri % len(_share_formats)]
        _win_df, _win_start = get_random_data_window(
            _share_full_df, _chart_points, _share_used_starts,
        )
        _share_used_starts.add(_win_start)
        _win_df = _win_df.with_columns(pl.lit(0.0).alias("prediction"))

        _sn = len(_win_df)
        _s_visible = _sn - PREDICTION_HOUR_OFFSET
        _s_gl = _win_df.get_column("gl").to_list()
        _s_preds = [0.0] * _sn
        _s_pred_steps = _sn - _s_visible
        for _si in range(_s_visible, _sn):
            _sg = _s_gl[_si]
            if _sg is not None:
                _s_step_frac = ((_si - _s_visible) / max(_s_pred_steps - 1, 1)) ** 1.8
                _s_step_noise = _share_noise * _s_step_frac
                _s_preds[_si] = round(
                    _sg * (1.0 + _share_rnd.uniform(-_s_step_noise, _s_step_noise)), 1
                )
        _win_df = _win_df.with_columns(
            pl.Series("prediction", _s_preds, dtype=pl.Float64)
        )

        _s_actual_row: dict[str, str] = {"metric": "Actual Glucose"}
        _s_pred_row: dict[str, str] = {"metric": "Predicted"}
        _s_abs_err_row: dict[str, str] = {"metric": "Absolute Error"}
        _s_rel_err_row: dict[str, str] = {"metric": "Relative Error (%)"}
        for _ti in range(_sn):
            _a = _s_gl[_ti]
            _p = _s_preds[_ti]
            _s_actual_row[f"t{_ti}"] = "-" if _a is None else f"{float(_a):.1f}"
            if _p == 0.0 or _a is None:
                _s_pred_row[f"t{_ti}"] = "-"
                _s_abs_err_row[f"t{_ti}"] = "-"
                _s_rel_err_row[f"t{_ti}"] = "-"
            else:
                _s_pred_row[f"t{_ti}"] = f"{_p:.1f}"
                _s_err = abs(float(_a) - _p)
                _s_abs_err_row[f"t{_ti}"] = f"{_s_err:.1f}"
                _s_rel_err_row[f"t{_ti}"] = (
                    f"{(_s_err / float(_a) * 100):.1f}%" if _a != 0 else "-"
                )

        _share_all_rounds.append({
            "round_number": _ri + 1,
            "prediction_window_start": _win_start,
            "prediction_window_size": _sn,
            "prediction_table_data": [
                _s_actual_row, _s_pred_row, _s_abs_err_row, _s_rel_err_row,
            ],
            "format": _fmt,
            "is_example_data": _share_is_example,
            "data_source_name": _share_source,
        })

    _share_study_id = str(uuid.uuid4())
    _share_run_id = str(uuid.uuid4())
    _share_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _share_played_formats = sorted(
        {r["format"] for r in _share_all_rounds},
        key=lambda x: FORMAT_ORDER.get(str(x), 999),
    )

    def _metrics_from_rounds(rounds: list[dict[str, Any]]) -> dict[str, float]:
        all_actual: list[float] = []
        all_pred: list[float] = []
        for rnd in rounds:
            ptd = rnd.get("prediction_table_data", [])
            if len(ptd) < 2:
                continue
            actual_row, pred_row = ptd[0], ptd[1]
            for k in actual_row:
                if k == "metric":
                    continue
                a_s, p_s = actual_row[k], pred_row[k]
                if a_s != "-" and p_s != "-":
                    try:
                        all_actual.append(float(a_s))
                        all_pred.append(float(p_s))
                    except ValueError:
                        continue
        n = len(all_actual)
        if n == 0:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mape": 0.0}
        mae = sum(abs(a - p) for a, p in zip(all_actual, all_pred)) / n
        mse = sum((a - p) ** 2 for a, p in zip(all_actual, all_pred)) / n
        rmse = mse ** 0.5
        nonzero = sum(1 for a in all_actual if a != 0)
        mape = (sum(abs((a - p) / a) * 100 for a, p in zip(all_actual, all_pred) if a != 0) / nonzero) if nonzero else 0.0
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}

    import tempfile, shutil
    _ranking_header = "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,overall_mape\n"

    def _append_ranking_row(path: Path, fmt: str, rounds_for_fmt: list[dict[str, Any]]) -> None:
        m = _metrics_from_rounds(rounds_for_fmt)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_ranking_header, encoding="utf-8")
        row = (
            f"{_share_study_id},{_share_run_id},0,{_share_timestamp},{fmt},"
            f"{len(rounds_for_fmt)},{_share_is_example},{_share_source},"
            f"{m['mae']},{m['mse']},{m['rmse']},{m['mape']}\n"
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(row)

    _tmp_ranking_dir = Path(tempfile.mkdtemp(prefix="sugar_share_ranking_"))
    _real_ranking_dir = project_root / "data" / "input"

    _rounds_by_fmt: dict[str, list[dict[str, Any]]] = {}
    for _r in _share_all_rounds:
        _rounds_by_fmt.setdefault(_r["format"], []).append(_r)

    for _fmt_key, _fmt_rounds in _rounds_by_fmt.items():
        _real_fmt_csv = _real_ranking_dir / f"prediction_ranking_{_fmt_key}.csv"
        _tmp_fmt_csv = _tmp_ranking_dir / f"prediction_ranking_{_fmt_key}.csv"
        if _real_fmt_csv.exists():
            shutil.copy2(_real_fmt_csv, _tmp_fmt_csv)
        _append_ranking_row(_tmp_fmt_csv, _fmt_key, _fmt_rounds)

    _real_overall_csv = _real_ranking_dir / "prediction_ranking.csv"
    _tmp_overall_csv = _tmp_ranking_dir / "prediction_ranking.csv"
    if _real_overall_csv.exists():
        shutil.copy2(_real_overall_csv, _tmp_overall_csv)
    _append_ranking_row(_tmp_overall_csv, "ALL", _share_all_rounds)

    def _share_rank(fmt_filter: Optional[str], csv_name: str) -> Optional[tuple[int, int]]:
        return _rank_from_ranking_csv(
            _tmp_ranking_dir / csv_name,
            study_id=_share_study_id,
            format_filter=fmt_filter,
            mode="best",
        )

    _share_per_format: list[dict[str, Any]] = []
    for _fmt_key in sorted(_rounds_by_fmt, key=lambda x: FORMAT_ORDER.get(x, 999)):
        _info = _share_rank(_fmt_key, f"prediction_ranking_{_fmt_key}.csv")
        if _info:
            _share_per_format.append({"format": _fmt_key, "rank": _info[0], "total": _info[1]})
    _share_overall = _share_rank(None, "prediction_ranking.csv")
    _share_rankings: dict[str, Any] = {
        "per_format": _share_per_format,
        "overall": {"rank": _share_overall[0], "total": _share_overall[1]} if _share_overall else None,
    }
    shutil.rmtree(_tmp_ranking_dir, ignore_errors=True)

    _share_record: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "locale": normalize_locale(_share_locale),
        "rounds": _share_all_rounds,
        "played_formats": _share_played_formats,
        "rankings": _share_rankings,
        "user_info": {
            "name": os.environ.get("_SHARE_NAME", SHARE_NAME),
            "study_id": _share_study_id,
            "format": _share_formats[0],
            "uses_cgm": True,
            "max_rounds": MAX_ROUNDS,
        },
    }
    _share_mode_id = share_store.save_share(_share_record)
    with start_action(action_type=u"share_mode_setup") as _share_action:
        _share_action.add_success_fields(
            share_id=_share_mode_id,
            rounds=_share_rounds_n,
            rankings=str(_share_rankings),
        )

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP,
    'https://cdn.jsdelivr.net/npm/fomantic-ui@2.9.3/dist/semantic.min.css',
    'https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@7.2.0/css/all.min.css',
]

external_scripts: list[dict[str, str]] = []
if UMAMI_SCRIPT_URL and UMAMI_WEBSITE_ID:
    _umami_script_attrs: dict[str, str] = {
        "src": UMAMI_SCRIPT_URL,
        "defer": "defer",
        "data-website-id": UMAMI_WEBSITE_ID,
    }
    if UMAMI_DOMAINS:
        _umami_script_attrs["data-domains"] = UMAMI_DOMAINS
    if UMAMI_HOST_URL:
        _umami_script_attrs["data-host-url"] = UMAMI_HOST_URL
    external_scripts.append(_umami_script_attrs)

# Mobile-first: the STATIC default viewport is `device-width` so every page is
# correct in portrait from first paint, with no dependency on a JS reflow.  The
# ONE exception is `/prediction`, where Plotly drawline needs a wide layout
# viewport -- a clientside callback (see below) flips the <meta> to this fixed
# width only on that route.  Desktop browsers ignore the viewport meta entirely,
# so neither value affects desktop layout.
_DESKTOP_LAYOUT_VIEWPORT_CSS_PX: int = 1280

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    external_scripts=external_scripts,
    assets_folder=str(project_root / 'assets'),
    suppress_callback_exceptions=True,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes",
        },
        {"name": "robots", "content": "index, follow"},
        {"name": "description", "content": SITE_DESCRIPTION},
        {"property": "og:type", "content": "website"},
        {"property": "og:site_name", "content": SITE_TITLE},
        {"property": "og:title", "content": f"{SITE_TITLE} - Glucose Prediction Game"},
        {"property": "og:description", "content": SITE_DESCRIPTION},
        {"property": "og:url", "content": canonical_base() or "/"},
        {"property": "og:image", "content": _site_og_image_url()},
        {"property": "og:image:secure_url", "content": _site_og_image_url()},
        {"property": "og:image:type", "content": "image/png"},
        {"property": "og:image:width", "content": str(OG_PREVIEW_SIZE[0])},
        {"property": "og:image:height", "content": str(OG_PREVIEW_SIZE[1])},
        {"property": "og:image:alt", "content": "Sugar Sugar glucose prediction game preview card."},
        {"name": "twitter:card", "content": "summary_large_image"},
        {"name": "twitter:title", "content": f"{SITE_TITLE} - Glucose Prediction Game"},
        {"name": "twitter:description", "content": SITE_DESCRIPTION},
        {"name": "twitter:image", "content": _site_og_image_url()},
        {"name": "twitter:image:alt", "content": "Sugar Sugar glucose prediction game preview card."},
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


@server.route("/robots.txt")
def _robots_txt() -> Response:
    """Crawler policy with canonical sitemap and LLM overview links."""
    return Response(_build_robots_txt(), mimetype="text/plain; charset=utf-8")


@server.route("/sitemap.xml")
def _sitemap_xml() -> Response:
    """Canonical sitemap for public, non-stateful routes."""
    return Response(_build_sitemap_xml(), mimetype="application/xml; charset=utf-8")


@server.route("/llms.txt")
def _llms_txt() -> Response:
    """Short LLM-readable overview of the public site."""
    return Response(_build_llms_txt(), mimetype="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# Share routes
#
# Two routes complement the Dash page at /share/<id>:
#  * /share/<id>/image.png  -- PNG render of the share card, served by kaleido.
#    Cached in-process by share_id so repeated loads (crawler + human) don't
#    spin kaleido up twice.
#  * /share/<id>/og         -- tiny HTML shell with Open Graph meta tags for
#    crawlers that don't execute JavaScript (Facebook, X, LinkedIn, WhatsApp).
#    Humans who hit this URL get redirected to the real Dash page.
# ---------------------------------------------------------------------------

_SHARE_PNG_CACHE: dict[tuple[str, str], bytes] = {}

_SOCIAL_CRAWLER_USER_AGENT_TOKENS: tuple[str, ...] = (
    "facebookexternalhit",
    "facebot",
    "twitterbot",
    "linkedinbot",
    "whatsapp",
    "slackbot",
    "telegrambot",
    "discordbot",
    "pinterest",
    "skypeuripreview",
)


def _first_forwarded_header_value(value: Optional[str]) -> Optional[str]:
    """Return the first value from a comma-separated proxy header."""
    if not value:
        return None
    first: str = value.split(",", 1)[0].strip()
    return first or None


def _public_request_base_url() -> str:
    """Base URL as seen by users/crawlers, respecting reverse-proxy headers."""
    deploy_url: str = canonical_base()
    if deploy_url:
        return deploy_url

    configured: Optional[str] = _first_forwarded_header_value(
        os.environ.get("SUGAR_SUGAR_PUBLIC_BASE_URL")
    )
    if configured:
        return configured.rstrip("/")

    forwarded_host: Optional[str] = _first_forwarded_header_value(
        flask_request.headers.get("X-Forwarded-Host")
    )
    forwarded_proto: Optional[str] = _first_forwarded_header_value(
        flask_request.headers.get("X-Forwarded-Proto")
    )
    if forwarded_host:
        scheme: str = forwarded_proto or flask_request.scheme or "https"
        return f"{scheme}://{forwarded_host}".rstrip("/")
    return flask_request.host_url.rstrip("/")


def _public_base_url_for_crawler_file() -> str:
    """Public base URL for crawler files, falling back to the active request."""
    base: str = canonical_base()
    if base:
        return base
    return _public_request_base_url()


def _absolute_url_for_path(path: str) -> str:
    """Build an absolute URL for a root-relative path in the current public origin."""
    cleaned_path: str = "/" + str(path or "/").lstrip("/")
    return f"{_public_base_url_for_crawler_file()}{cleaned_path}"


def _build_robots_txt() -> str:
    sitemap_url: str = _absolute_url_for_path("/sitemap.xml")
    llms_url: str = _absolute_url_for_path("/llms.txt")
    return "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            "Allow: /llms.txt",
            "Disallow: /_dash-",
            "Disallow: /_reload-hash",
            # NOTE: do NOT Disallow /share/*/image.png here. Twitterbot honors
            # robots.txt, so a Disallow makes it skip the OG card image entirely
            # (FB/WhatsApp/LinkedIn/Telegram ignore robots.txt for OG fetches, so
            # they still showed it -- this is exactly why Twitter alone broke).
            # Search engines are kept from indexing the per-share PNGs via the
            # `X-Robots-Tag: noindex` response header on the image route instead,
            # which permits crawler *fetching* while blocking *indexing*.
            "",
            f"Sitemap: {sitemap_url}",
            f"# LLM-readable overview: {llms_url}",
            "",
        ]
    )


def _build_sitemap_xml() -> str:
    lastmod: str = datetime.utcnow().date().isoformat()
    entries: list[str] = []
    for route, _title, _description in PUBLIC_ROUTES:
        loc: str = html_escape(_absolute_url_for_path(route), quote=True)
        entries.append(
            "  <url>\n"
            f"    <loc>{loc}</loc>\n"
            f"    <lastmod>{lastmod}</lastmod>\n"
            "    <changefreq>weekly</changefreq>\n"
            f"    <priority>{'1.0' if route == '/' else '0.7'}</priority>\n"
            "  </url>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(entries)
        + "\n</urlset>\n"
    )


def _build_llms_txt() -> str:
    routes: str = "\n".join(
        f"- {_absolute_url_for_path(route)} — {description}"
        for route, _title, description in PUBLIC_ROUTES
    )
    return (
        "# Sugar Sugar\n\n"
        f"{SITE_DESCRIPTION}\n\n"
        "Sugar Sugar is a Dash research app from GlucoseDAO. Participants predict "
        "the next hour of CGM glucose values, compare predictions with ground truth, "
        "and can share a public performance summary.\n\n"
        "## Public Routes\n\n"
        f"{routes}\n\n"
        "## Crawl Guidance\n\n"
        "- Public informational routes are crawlable.\n"
        "- Game-flow routes such as /startup, /prediction, /ending, and /final are "
        "stateful participant flows and should not be treated as durable documents.\n"
        "- Share URLs under /share/<id> expose crawler-ready Open Graph metadata and "
        "redirect humans into the live Dash share page.\n"
    )


def _build_share_url(share_id: str) -> str:
    """Compose an absolute public URL for a share id based on the current request."""
    try:
        base: str = _public_request_base_url()
    except RuntimeError:
        # Not inside a Flask request context -- fall back to a relative path.
        return f"/share/{share_id}"
    return f"{base}/share/{share_id}"


def _share_id_from_public_path(path: str) -> Optional[str]:
    """Extract share id from the public Dash route, excluding image/OG assets."""
    if not path.startswith("/share/"):
        return None
    suffix: str = path.removeprefix("/share/").strip("/")
    if not suffix or "/" in suffix:
        return None
    return suffix


def _is_social_crawler(user_agent: str) -> bool:
    ua: str = str(user_agent or "").lower()
    return any(token in ua for token in _SOCIAL_CRAWLER_USER_AGENT_TOKENS)


def _share_card_og_response(share_id: str) -> Any:
    """HTML page with OG tags only, for social-platform crawlers."""
    from flask import Response, abort
    record = share_store.load_share(share_id)
    if record is None:
        abort(404)
    locale: str = str(record.get("locale") or "en")
    loc: str = normalize_locale(locale)
    share_url: str = _build_share_url(share_id)
    image_url: str = f"{share_url}/image.png?v={SHARE_CARD_IMAGE_VERSION}"

    from sugar_sugar.components.share import compute_aggregate_stats, _best_ranking_entry, _format_number
    og_stats = compute_aggregate_stats(list(record.get("rounds") or []))
    og_accuracy = og_stats.get("accuracy", float("nan"))
    og_accuracy_str = f"{_format_number(og_accuracy)}%" if not math.isnan(og_accuracy) else "?"
    og_best = _best_ranking_entry(record)
    og_percentile = og_best.get("percentile") if og_best else None
    if og_percentile is not None:
        title = html_escape(
            t("ui.share.og_title_ranked", locale=loc, percentile=f"{og_percentile}%", accuracy=og_accuracy_str),
            quote=True,
        )
    else:
        title = html_escape(
            t("ui.share.og_title_unranked", locale=loc, accuracy=og_accuracy_str),
            quote=True,
        )
    description: str = html_escape(t("ui.share.subtitle", locale=loc), quote=True)
    escaped_share_url: str = html_escape(share_url, quote=True)
    escaped_image_url: str = html_escape(image_url, quote=True)

    html_page: str = f"""<!doctype html>
<html lang="{html_escape(loc, quote=True)}">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="{escaped_image_url}">
<meta property="og:image:secure_url" content="{escaped_image_url}">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="{OG_PREVIEW_SIZE[0]}">
<meta property="og:image:height" content="{OG_PREVIEW_SIZE[1]}">
<meta property="og:image:alt" content="{description}">
<meta property="og:url" content="{escaped_share_url}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{escaped_image_url}">
<meta name="twitter:image:alt" content="{description}">
</head>
<body>
<p><a href="{escaped_share_url}">Open {title}</a>.</p>
</body>
</html>
"""
    return Response(html_page, mimetype="text/html; charset=utf-8")


@server.before_request
def _serve_share_og_to_social_crawlers() -> Optional[Any]:
    """Serve card metadata at the share URL crawlers actually request."""
    share_id: Optional[str] = _share_id_from_public_path(flask_request.path)
    if share_id is None:
        return None
    if not _is_social_crawler(flask_request.headers.get("User-Agent", "")):
        return None
    return _share_card_og_response(share_id)


@server.route("/share/<share_id>/image.png")
def _share_card_png(share_id: str) -> Any:
    from flask import abort, request as flask_req
    record = share_store.load_share(share_id)
    if record is None:
        abort(404)
    locale: str = flask_req.args.get("lang") or str(record.get("locale") or "en")
    cache_key = (share_id, locale)
    cached: Optional[bytes] = _SHARE_PNG_CACHE.get(cache_key)
    if cached is None:
        share_url: str = _build_share_url(share_id)
        from sugar_sugar.share_png import render_share_card_png_bytes

        cached = render_share_card_png_bytes(
            record,
            share_url=share_url,
            locale=locale,
            seed=share_id,
        )
        _SHARE_PNG_CACHE[cache_key] = cached
    # Serve INLINE (not as_attachment): a `Content-Disposition: attachment`
    # makes some social image consumers (Twitter among the pickier ones) refuse
    # to render the card. The human "Download" button on the share page forces a
    # download client-side via the HTML `download` attribute, so it doesn't need
    # the attachment disposition here.
    response = flask_send_file(
        BytesIO(cached),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"sugar-sugar-{share_id}.png",
        max_age=86400,
    )
    response.headers["Cache-Control"] = "public, max-age=86400"
    # Allow crawlers to FETCH the card (needed for Twitter/X OG) but keep the
    # per-share PNGs out of search indexes. Pairs with the robots.txt note.
    response.headers["X-Robots-Tag"] = "noindex"
    return response


@server.route("/share/<share_id>/og")
def _share_card_og(share_id: str) -> Any:
    # Crawlers get the OG metadata HTML; humans who land here are redirected
    # server-side to the real Dash share page (we dropped the meta-refresh,
    # which confused X/Twitter into showing the generic site card).
    if _is_social_crawler(flask_request.headers.get("User-Agent", "")):
        return _share_card_og_response(share_id)
    from flask import redirect
    return redirect(_build_share_url(share_id), code=302)


# ---------------------------------------------------------------------------
# Staging mode (prod+): synthetic prefilled nodes for remote testing.
#
# Every helper and route below is invoked ONLY when `_is_staging_mode` is True
# (set by `serve --staging` / `uv run serve-staging`). They are defined at
# module scope but never run at import time, so when the flag is off the app
# behaves identically to production. They reuse the real layout builders
# (create_ending_layout / create_final_layout / create_share_layout) and
# share_store; only the synthetic *input* data is generated here. This is
# deliberately additive: no production callback or builder is modified.
# ---------------------------------------------------------------------------
_STAGING_NODES: list[tuple[str, str]] = [
    ("/staging/prediction", "Prefilled prediction chart (rotate to landscape on mobile)"),
    ("/staging/ending", "Round-ending page with synthetic predictions + metrics"),
    ("/staging/final", "Final results page with several synthetic rounds"),
    ("/staging/share", "Generate a synthetic share record and open /share/<id>"),
]


def _staging_prefill_window(full_df: pl.DataFrame, *, noise: float = 0.05) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Pick a random window and fill its hidden region with noisy ground truth.

    Returns ``(full_df_with_predictions, window_df, window_start)``. Mirrors the
    ``--prefill`` logic used by chart mode at module import (lines ~327-344).
    """
    import random as _rnd
    window_df, start = get_random_data_window(full_df, _chart_points)
    window_df = window_df.with_columns(pl.lit(0.0).alias("prediction"))
    n = len(window_df)
    visible = n - PREDICTION_HOUR_OFFSET
    gl = window_df.get_column("gl").to_list()
    preds: list[float] = [0.0] * n
    for i in range(visible, n):
        if gl[i] is not None:
            preds[i] = round(gl[i] * (1.0 + _rnd.uniform(-noise, noise)), 1)
    window_df = window_df.with_columns(pl.Series("prediction", preds, dtype=pl.Float64))
    for i in range(n):
        pv = window_df.get_column("prediction")[i]
        if pv != 0.0:
            tv = window_df.get_column("time")[i]
            full_df = full_df.with_columns(
                pl.when(pl.col("time") == tv).then(pv).otherwise(pl.col("prediction")).alias("prediction")
            )
    return full_df, window_df, start


def _staging_ptd_from_window(window_df: pl.DataFrame) -> list[dict[str, str]]:
    """Build the 4-row ``prediction_table_data`` from a prefilled window.

    Mirrors the table construction in the share-mode block (lines ~398-416).
    """
    n = len(window_df)
    gl = window_df.get_column("gl").to_list()
    preds = window_df.get_column("prediction").to_list()
    actual_row: dict[str, str] = {"metric": "Actual Glucose"}
    pred_row: dict[str, str] = {"metric": "Predicted"}
    abs_row: dict[str, str] = {"metric": "Absolute Error"}
    rel_row: dict[str, str] = {"metric": "Relative Error (%)"}
    for ti in range(n):
        a = gl[ti]
        p = preds[ti]
        actual_row[f"t{ti}"] = "-" if a is None else f"{float(a):.1f}"
        if not p or a is None:
            pred_row[f"t{ti}"] = abs_row[f"t{ti}"] = rel_row[f"t{ti}"] = "-"
        else:
            pred_row[f"t{ti}"] = f"{p:.1f}"
            err = abs(float(a) - p)
            abs_row[f"t{ti}"] = f"{err:.1f}"
            rel_row[f"t{ti}"] = f"{(err / float(a) * 100):.1f}%" if a != 0 else "-"
    return [actual_row, pred_row, abs_row, rel_row]


def _staging_base_user_info() -> dict[str, Any]:
    """A synthetic, already-consented user_info for staging nodes."""
    return {
        "study_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "email": "staging@vanilla-sugar.local",
        "age": 30, "gender": "F", "uses_cgm": True, "cgm_duration_years": 2,
        "format": "A", "run_format": "A",
        "diabetic": True, "diabetic_type": "Type 1", "diabetes_duration": 6,
        "location": "Staging",
        "max_rounds": MAX_ROUNDS, "current_round_number": 1, "statistics_saved": False,
        "is_example_data": True, "data_source_name": "example.csv",
        "consent_completed": True, "consent_no_selection": False,
        "consent_play_only": False, "consent_participate_in_study": True,
        "consent_receive_results_later": False, "consent_keep_up_to_date": False,
        "consent_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resume_code": resume_store.new_code(),
    }


def _staging_ending_args() -> tuple[dict, dict, dict, dict[str, Any]]:
    """Build (full_df_store, window_store, events_store, user_info) for /staging/ending."""
    full_df, events_df = load_glucose_data()
    full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
    full_df, window_df, start = _staging_prefill_window(full_df)
    info = _staging_base_user_info()
    info.update({
        "prediction_window_start": start,
        "prediction_window_size": len(window_df),
        "prediction_table_data": _staging_ptd_from_window(window_df),
    })
    return (
        dataframe_to_store_dict(full_df),
        dataframe_to_store_dict(window_df),
        events_dataframe_to_store_dict(events_df),
        info,
    )


def _staging_final_user_info(*, rounds_n: int = 3, formats: Optional[list[str]] = None) -> tuple[dict, dict[str, Any]]:
    """Build (full_df_store, user_info-with-rounds) for /staging/final."""
    fmts = formats or ["A", "B", "C"]
    full_df, _events = load_glucose_data()
    full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
    rounds: list[dict[str, Any]] = []
    for ri in range(rounds_n):
        fmt = fmts[ri % len(fmts)]
        _f, window_df, start = _staging_prefill_window(full_df.clone())
        rounds.append({
            "round_number": ri + 1,
            "prediction_window_start": start,
            "prediction_window_size": len(window_df),
            "prediction_table_data": _staging_ptd_from_window(window_df),
            "format": fmt,
            "is_example_data": True,
            "data_source_name": "example.csv",
        })
    info = _staging_base_user_info()
    info.update({
        "rounds": rounds,
        "current_round_number": rounds_n,
        "format": rounds[-1]["format"] if rounds else "A",
    })
    return dataframe_to_store_dict(full_df), info


def _staging_build_share_record(*, rounds_n: int = 6, formats: Optional[list[str]] = None, locale: str = "en") -> str:
    """Generate a synthetic share record on disk and return its share id."""
    fmts = formats or ["A", "B", "C"]
    full_df, _events = load_glucose_data()
    full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
    rounds: list[dict[str, Any]] = []
    for ri in range(rounds_n):
        fmt = fmts[ri % len(fmts)]
        _f, window_df, start = _staging_prefill_window(full_df.clone())
        rounds.append({
            "round_number": ri + 1,
            "prediction_window_start": start,
            "prediction_window_size": len(window_df),
            "prediction_table_data": _staging_ptd_from_window(window_df),
            "format": fmt,
            "is_example_data": True,
            "data_source_name": "example.csv",
        })
    played_formats = sorted({r["format"] for r in rounds}, key=lambda x: FORMAT_ORDER.get(str(x), 999))
    record: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "locale": normalize_locale(locale),
        "rounds": rounds,
        "played_formats": played_formats,
        "rankings": {"per_format": [], "overall": None},
        "user_info": {
            "name": "Staging Tester",
            "study_id": str(uuid.uuid4()),
            "format": played_formats[0] if played_formats else "A",
            "uses_cgm": True,
            "max_rounds": MAX_ROUNDS,
        },
    }
    return share_store.save_share(record)


def _staging_index_layout(*, locale: str) -> html.Div:
    """A simple index of the available staging test nodes."""
    return html.Div(
        [
            html.H1("Staging test nodes", disable_n_clicks=True),
            html.P(
                "Prod+ test routes (active only when _STAGING_MODE=1). Each node "
                "jumps straight to a prefilled state for remote/visual testing.",
                disable_n_clicks=True,
            ),
            html.Ul(
                [
                    html.Li(
                        [dcc.Link(path, href=path), html.Span(f" — {desc}")],
                        disable_n_clicks=True,
                    )
                    for path, desc in _STAGING_NODES
                ],
                disable_n_clicks=True,
            ),
        ],
        className="info-page",
        disable_n_clicks=True,
    )


def _staging_display(pathname: str, *, locale: str, glucose_unit: Optional[str]) -> Optional[html.Div]:
    """Render a staging node layout, or None to fall through (e.g. /staging/prediction)."""
    if pathname in ("/staging", "/staging/"):
        return _staging_index_layout(locale=locale)
    if pathname == "/staging/ending":
        full_store, window_store, events_store, info = _staging_ending_args()
        return create_ending_layout(full_store, window_store, events_store, info, glucose_unit, locale=locale)
    if pathname == "/staging/final":
        full_store, info = _staging_final_user_info()
        return create_final_layout(full_store, info, glucose_unit, locale=locale)
    # /staging/prediction is handled by the _staging_seed_prediction callback,
    # which seeds the stores and redirects to /prediction. Fall through here.
    return None


if _is_staging_mode:
    @server.before_request
    def _guard_staging_routes() -> Any:
        """Optional Basic-Auth gate for the /staging/* test routes.

        Activates only when STAGING_AUTH ("user:password") is set, so local
        `serve --staging` and the screenshot harness stay open, while the public
        staging origin (vanilla-sugar.glucosedao.org) can lock the test nodes
        down. Behind a TLS reverse proxy, Basic Auth over HTTPS is sufficient.
        Read live each request so the credential can be rotated without code
        changes. The /staging callback content arrives via /_dash-update-component
        once the browser has authenticated for the realm, so gating the /staging*
        GETs is enough to keep anonymous users out.
        """
        if not flask_request.path.startswith("/staging"):
            return None
        credential = os.environ.get("STAGING_AUTH")
        if not credential:
            return None  # unconfigured -> open (local dev / harness)
        from flask import Response
        auth = flask_request.authorization
        if auth and f"{auth.username}:{auth.password}" == credential:
            return None
        return Response(
            "Staging area requires authentication.",
            401,
            {"WWW-Authenticate": 'Basic realm="sugar-sugar staging"'},
        )

    @server.route("/staging/share")
    def _staging_share_route() -> Any:
        """Generate a synthetic share record and 302-redirect to /share/<id>."""
        from flask import redirect, request as flask_req
        locale = flask_req.args.get("lang") or "en"
        formats_arg = flask_req.args.get("formats")
        formats = [f.strip().upper() for f in formats_arg.split(",")] if formats_arg else None
        share_id = _staging_build_share_record(locale=locale, formats=formats)
        return redirect(f"/share/{share_id}", code=302)

    @app.callback(
        [Output('url', 'pathname', allow_duplicate=True),
         Output('user-info-store', 'data', allow_duplicate=True),
         Output('full-df', 'data', allow_duplicate=True),
         Output('current-window-df', 'data', allow_duplicate=True),
         Output('events-df', 'data', allow_duplicate=True),
         Output('randomization-initialized', 'data', allow_duplicate=True),
         Output('is-example-data', 'data', allow_duplicate=True),
         Output('data-source-name', 'data', allow_duplicate=True)],
        Input('url', 'pathname'),
        prevent_initial_call=True,
    )
    def _staging_seed_prediction(pathname: Optional[str]) -> tuple[Any, ...]:
        """Seed the prediction stores with a prefilled window, then route to /prediction."""
        if pathname != '/staging/prediction':
            raise PreventUpdate
        full_store, window_store, events_store, info = _staging_ending_args()
        return ('/prediction', info, full_store, window_store, events_store, True, True, "example.csv")

app.clientside_callback(
    "function() { return window.navigator.userAgent || ''; }",
    Output('user-agent', 'data'),
    Input('url', 'href'),
    prevent_initial_call=False
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
_callbacks_registered: bool = False

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
        # Synthetic dev session is treated as already-consented so the
        # display_page consent guard lets chart mode render /prediction.
        "consent_completed": True,
        # Resume code so the /ending copy-resume-link button works in dev/testing.
        "resume_code": resume_store.new_code(),
    }
else:
    _chart_user_info = None

app.layout = html.Div([
    dcc.Store(id='ai-prediction-store', data={"glumind": {"predictions": [], "ready": False}}, storage_type='session'),
    dcc.Store(id='session-store', storage_type=STORAGE_TYPE),
    dcc.Store(id='window-store', storage_type='session'),
    dcc.Location(id='url', refresh=False, **(
        {'pathname': f'/share/{_share_mode_id}'} if _is_share_mode and _share_mode_id
        else {'pathname': '/prediction'} if _is_chart_mode
        else {}
    )),
    dcc.Store(id='user-info-store', data=_chart_user_info, storage_type=STORAGE_TYPE),
    dcc.Store(id='last-click-time', data=0),
    # Fingerprint sentinel: value must equal DEPLOY_BUILD in config.py.
    # Dash fingerprints the layout JSON, not clientside callback JS, so a JS-only
    # change survives a server restart and old browsers keep their cached
    # /_dash-dependencies. Bumping DEPLOY_BUILD changes the layout hash, forcing
    # every reconnecting browser to do a full reload and pick up the new JS.
    dcc.Store(id='_build', data=DEPLOY_BUILD),
    dcc.Store(id='consent-scroll-request', data=0),
    dcc.Store(id='current-window-df', data=example_initial_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='full-df', data=example_full_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='events-df', data=example_events_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='is-example-data', data=_chart_is_example, storage_type=STORAGE_TYPE),
    dcc.Store(id='data-source-name', data=_chart_source if _is_chart_mode else "example.csv", storage_type=STORAGE_TYPE),
    dcc.Store(id='randomization-initialized', data=_is_chart_mode, storage_type=STORAGE_TYPE),
    dcc.Store(id='glucose-chart-mode', data={'hide_last_hour': True}, storage_type='memory'),
    dcc.Store(id='glucose-unit', data=_chart_unit if _is_chart_mode else 'mg/dL', storage_type=STORAGE_TYPE),
    dcc.Store(id='interface-language', data=_chart_locale if _is_chart_mode else 'en', storage_type=STORAGE_TYPE),
    dcc.Store(id='user-agent', data=None, storage_type=STORAGE_TYPE),
    dcc.Store(id='initial-slider-value', data=example_initial_slider_value, storage_type=STORAGE_TYPE),
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
    # Current step index for the mobile startup wizard (StartupPageMobile).
    # Memory: wizard position resets per page load, like page-restore-done.
    dcc.Store(id='startup-step', data=0, storage_type='memory'),

    html.Div(id='mobile-warning', style={'display': 'none'}),
    html.Div(id='scroll-to-top-trigger', style={'display': 'none'}),
    html.Div(id='demo-video-sink', style={'display': 'none'}),
    # Throwaway sink for the per-page viewport / route-class clientside callback
    # (there is no real Dash Output for the <meta viewport> tag or <html> class).
    html.Div(id='viewport-sink', style={'display': 'none'}),
    # Throwaway sink for the cross-device auto-snapshot callback (writes the live
    # session to resume_store keyed by user_info['resume_code']).
    dcc.Store(id='resume-sync', data=None, storage_type='memory'),
    # One-shot guard so the ?resume=<code> redeem callback acts at most once.
    dcc.Store(id='resume-redeem-done', data=False, storage_type='memory'),
    # Throwaway sink for the clientside callback that strips ?resume= from the URL.
    html.Div(id='resume-clean-sink', style={'display': 'none'}),

    html.Div(id='resume-dialog-container', children=[], disable_n_clicks=True),

    html.Div(id='navbar-container', children=[], disable_n_clicks=True),

    html.Div(id='page-content', children=[], disable_n_clicks=True),

    # Throwaway sinks for the clientside immersive handlers.
    html.Div(id="immersive-sink", style={"display": "none"}),
    html.Div(id="prediction-fullscreen-sink", style={"display": "none"}),
    html.Div(id="copy-link-sink", style={"display": "none"}),
])


# Add a global `mobile-device` class to <html> based on the browser
# user-agent.  This lets the CSS in assets/mobile.css scope all mobile
# overrides without touching the desktop path.  The class is also removed
# on non-mobile user agents, so CSS selectors are stable across hot-reload.
app.clientside_callback(
    """
    function(ua) {
        if (!document || !document.documentElement) {
            return window.dash_clientside.no_update;
        }
        var root = document.documentElement;
        var isMobile = false;
        if (ua && typeof ua === 'string') {
            var lc = ua.toLowerCase();
            var keywords = ['iphone', 'android', 'ipad', 'mobile', 'opera mini', 'mobi'];
            for (var i = 0; i < keywords.length; i++) {
                if (lc.indexOf(keywords[i]) !== -1) { isMobile = true; break; }
            }
        }
        // Touch-capable + coarse pointer is a reliable tablet fallback.
        if (!isMobile && window.matchMedia) {
            try {
                if (window.matchMedia('(pointer: coarse)').matches &&
                    window.matchMedia('(max-device-width: 1024px)').matches) {
                    isMobile = true;
                }
            } catch (e) { /* ignore */ }
        }
        if (isMobile) {
            root.classList.add('mobile-device');
        } else {
            root.classList.remove('mobile-device');
        }
        return {'display': 'none'};
    }
    """,
    Output('mobile-warning', 'style'),
    Input('user-agent', 'data'),
    prevent_initial_call=False,
)


# Per-page layout viewport + route class.  The chart-drawing page keeps a wide
# layout viewport only in landscape, where drawing is the primary mode.  In
# portrait it stays mobile-width and CSS puts the wide chart inside a horizontal
# scroller so the surrounding UI remains readable.
app.clientside_callback(
    """
    function(pathname) {
        var root = document.documentElement;
        var isPrediction = (pathname === '/prediction');
        if (root) {
            if (isPrediction) { root.classList.add('route-prediction'); }
            else { root.classList.remove('route-prediction'); }
        }
        function scrollPredictionChartToDrawArea() {
            var scroller = document.getElementById('prediction-glucose-chart-container');
            if (!scroller) { return; }
            if (window.matchMedia && window.matchMedia('(orientation: portrait)').matches) {
                scroller.scrollLeft = Math.max(0, scroller.scrollWidth - scroller.clientWidth);
            }
        }
        function applyViewport() {
            var m = document.querySelector('meta[name="viewport"]');
            if (!m) { return; }
            // ALWAYS device-width on /prediction. We used to force width=1280 in
            // landscape, but in real fullscreen landscape the browser does NOT
            // auto-scale the 1280 layout to fit, so the right ~30% (incl. Submit)
            // overflowed off-screen. The real landscape device-width (~800-900px)
            // is plenty for drawing, and portrait uses a horizontal-scroll chart.
            var fluid = 'width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes';
            m.setAttribute('content', fluid);
            window.setTimeout(scrollPredictionChartToDrawArea, 250);
            window.setTimeout(scrollPredictionChartToDrawArea, 900);
        }
        applyViewport();
        if (window.__sugarPredictionViewportHandler) {
            window.removeEventListener('resize', window.__sugarPredictionViewportHandler);
            window.removeEventListener('orientationchange', window.__sugarPredictionViewportHandler);
        }
        window.__sugarPredictionViewportHandler = applyViewport;
        window.addEventListener('resize', applyViewport);
        window.addEventListener('orientationchange', applyViewport);
        return window.dash_clientside.no_update;
    }
    """,
    Output('viewport-sink', 'children'),
    Input('url', 'pathname'),
    prevent_initial_call=False,
)


# Mobile burger menu: toggle the nav drawer open/closed.  n_clicks parity is
# fine because the navbar is re-rendered fresh on every page navigation (which
# resets n_clicks and closes the drawer).  These ids exist only in MobileNavBar.
app.clientside_callback(
    """
    function(n) {
        var open = (n || 0) % 2 === 1;
        return {'display': open ? 'block' : 'none'};
    }
    """,
    Output('mobile-nav-drawer', 'style'),
    Input('mobile-nav-toggle', 'n_clicks'),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(openClicks, closeClicks, currentStyle) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) {
            return currentStyle || {'display': 'none'};
        }
        var prop = ctx.triggered[0].prop_id || '';
        if (prop.indexOf('header-how-to-play-close') === 0) {
            return {'display': 'none'};
        }
        if (prop.indexOf('header-how-to-play-toggle') === 0) {
            var visible = currentStyle && currentStyle.display !== 'none';
            return {'display': visible ? 'none' : 'block'};
        }
        return currentStyle || {'display': 'none'};
    }
    """,
    Output('header-how-to-play-bubble', 'style'),
    [Input('header-how-to-play-toggle', 'n_clicks'),
     Input('header-how-to-play-close', 'n_clicks')],
    [State('header-how-to-play-bubble', 'style')],
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(n) {
        if (!n) {
            return window.dash_clientside.no_update;
        }

        var shell = document.getElementById('demo-video-shell');
        var frame = document.getElementById('demo-video-frame');
        var youtubeUrl = 'https://www.youtube.com/watch?v=M9JDhLFfFbA';

        function openYoutubeFallback() {
            window.location.href = youtubeUrl;
        }

        if (!shell || !frame) {
            openYoutubeFallback();
            return window.dash_clientside.no_update;
        }

        var requestFullscreen = (
            shell.requestFullscreen ||
            shell.webkitRequestFullscreen ||
            shell.msRequestFullscreen
        );

        if (!requestFullscreen) {
            openYoutubeFallback();
            return window.dash_clientside.no_update;
        }

        shell.classList.add('demo-video-immersive');

        function clearImmersiveClass() {
            if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                shell.classList.remove('demo-video-immersive');
                document.removeEventListener('fullscreenchange', clearImmersiveClass);
                document.removeEventListener('webkitfullscreenchange', clearImmersiveClass);
            }
        }

        document.addEventListener('fullscreenchange', clearImmersiveClass);
        document.addEventListener('webkitfullscreenchange', clearImmersiveClass);

        try {
            var result = requestFullscreen.call(shell);
            if (result && result.catch) {
                result.catch(openYoutubeFallback);
            }
        } catch (e) {
            openYoutubeFallback();
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('demo-video-sink', 'children'),
    Input('demo-fullscreen-button', 'n_clicks'),
    prevent_initial_call=True,
)


# Immersive entry: when the user clicks the wizard's final Start button on a
# mobile device, request fullscreen on the whole page (the same Fullscreen API
# the demo video uses successfully) and best-effort lock to landscape, so they
# land directly in the immersive chart. Triggered by the Start-button gesture so
# the browser honours requestFullscreen (a route-change callback would lose the
# user-activation and be rejected). screen.orientation.lock() needs the fullscreen
# we just entered; it works on Android Chrome/Vivaldi and rejects on iOS Safari
# (where the user rotates manually -- the immersive landscape CSS still applies).
# Desktop is excluded via the mobile-device class check.
app.clientside_callback(
    """
    function(n) {
        if (!n) { return window.dash_clientside.no_update; }
        if (!document.documentElement.classList.contains('mobile-device')) {
            return window.dash_clientside.no_update;
        }
        var el = document.documentElement;
        var requestFullscreen = (
            el.requestFullscreen ||
            el.webkitRequestFullscreen ||
            el.msRequestFullscreen
        );
        function lockLandscape() {
            try {
                if (screen.orientation && screen.orientation.lock) {
                    var p = screen.orientation.lock('landscape');
                    if (p && p.catch) { p.catch(function(){}); }
                }
            } catch (e) { /* unsupported (iOS Safari) -- ignore */ }
            setTimeout(function(){
                window.dispatchEvent(new Event('resize'));
                if (window.Plotly) {
                    document.querySelectorAll('.js-plotly-plot').forEach(function(g){
                        try { window.Plotly.Plots.resize(g); } catch(e){}
                    });
                }
            }, 400);
        }
        if (!requestFullscreen) { lockLandscape(); return window.dash_clientside.no_update; }
        try {
            var result = requestFullscreen.call(el);
            if (result && result.then) { result.then(lockLandscape).catch(lockLandscape); }
            else { lockLandscape(); }
        } catch (e) { lockLandscape(); }
        return window.dash_clientside.no_update;
    }
    """,
    Output('immersive-sink', 'children'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True,
)


# Persistent "Go fullscreen" button on /prediction: same fullscreen + landscape
# lock as the Start-button path, but available any time (gesture-reliable). The
# button is CSS-hidden off mobile/non-prediction, so this only fires where it
# should.
app.clientside_callback(
    """
    function(n) {
        if (!n) { return window.dash_clientside.no_update; }
        var el = document.documentElement;
        var requestFullscreen = (
            el.requestFullscreen ||
            el.webkitRequestFullscreen ||
            el.msRequestFullscreen
        );
        function lockLandscape() {
            try {
                if (screen.orientation && screen.orientation.lock) {
                    var p = screen.orientation.lock('landscape');
                    if (p && p.catch) { p.catch(function(){}); }
                }
            } catch (e) { /* iOS Safari -- ignore */ }
            setTimeout(function(){
                window.dispatchEvent(new Event('resize'));
                if (window.Plotly) {
                    document.querySelectorAll('.js-plotly-plot').forEach(function(g){
                        try { window.Plotly.Plots.resize(g); } catch(e){}
                    });
                }
            }, 400);
        }
        if (!requestFullscreen) { lockLandscape(); return window.dash_clientside.no_update; }
        try {
            var result = requestFullscreen.call(el);
            if (result && result.then) { result.then(lockLandscape).catch(lockLandscape); }
            else { lockLandscape(); }
        } catch (e) { lockLandscape(); }
        return window.dash_clientside.no_update;
    }
    """,
    Output('prediction-fullscreen-sink', 'children'),
    Input('prediction-fullscreen-button', 'n_clicks'),
    prevent_initial_call=True,
)


# Copy a cross-device resume link (?resume=<code>) to the clipboard from the
# between-rounds /ending summary (the in-round chart page has no screen budget).
# The code lives in user-info-store (assigned at consent). Shows transient "copied"
# feedback in the button text (reverted via setTimeout), localized through the
# button's data-copied-text attribute. Falls back to execCommand on non-secure
# contexts where navigator.clipboard is unavailable.
app.clientside_callback(
    """
    function(n, userInfo) {
        if (!n) { return window.dash_clientside.no_update; }
        var btn = document.getElementById('ending-copy-link-button');
        if (!btn) { return window.dash_clientside.no_update; }
        var code = userInfo && userInfo.resume_code;
        if (!code) { return window.dash_clientside.no_update; }
        var url = window.location.origin + '/?resume=' + encodeURIComponent(code);
        var original = btn.getAttribute('data-label') || btn.textContent;
        btn.setAttribute('data-label', original);
        var copiedMsg = btn.getAttribute('data-copied-text') || 'Copied!';
        function feedback() {
            btn.textContent = copiedMsg;
            setTimeout(function(){ btn.textContent = original; }, 2200);
        }
        function fallbackCopy() {
            try {
                var ta = document.createElement('textarea');
                ta.value = url; ta.style.position = 'fixed'; ta.style.opacity = '0';
                document.body.appendChild(ta); ta.focus(); ta.select();
                document.execCommand('copy'); document.body.removeChild(ta);
            } catch (e) {}
            feedback();
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url).then(feedback).catch(fallbackCopy);
        } else {
            fallbackCopy();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('copy-link-sink', 'children'),
    Input('ending-copy-link-button', 'n_clicks'),
    State('user-info-store', 'data'),
    prevent_initial_call=True,
)


@app.callback(
    Output('prediction-fullscreen-button', 'children'),
    [Input('interface-language', 'data')],
    [State('url', 'pathname')],
    prevent_initial_call=False,
)
def update_fullscreen_button_text(interface_language: Optional[str], pathname: Optional[str]) -> str:
    """Keep the 'Go fullscreen' button translated as the language changes."""
    if pathname != '/prediction':
        raise PreventUpdate
    return t("ui.orientation.go_fullscreen", locale=normalize_locale(interface_language))


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

# Keyword list mirrors the clientside `mobile-device` class setter.  Kept here so
# the server-side layout branch (display_page / update_on_language_change) can pick
# the mobile builder for structurally-different pages (startup wizard, landing).
_MOBILE_UA_KEYWORDS: tuple[str, ...] = (
    'iphone', 'android', 'ipad', 'mobile', 'mobi', 'opera mini',
)


def _is_mobile_ua(ua: Optional[str]) -> bool:
    """True if the User-Agent string looks like a phone/tablet.

    We read the live request header (request-scoped, always present) rather than
    the async-hydrating ``user-agent`` dcc.Store, because the layout branch must
    be correct on the very first render.  This is intentionally coarse: it only
    decides *which layout* to serve; the clientside class-setter still owns the
    finer ``(pointer: coarse)`` CSS gating.
    """
    if not ua:
        return False
    lc = ua.lower()
    return any(kw in lc for kw in _MOBILE_UA_KEYWORDS)


def _is_mobile_request() -> bool:
    """Detect a mobile client from the current Flask request's User-Agent."""
    return _is_mobile_ua(flask_request.headers.get('User-Agent', ''))


def _startup_builder(*, locale: str) -> html.Div:
    """Return the startup page builder appropriate for the requesting device."""
    if _is_mobile_request():
        return StartupPageMobile(locale=locale)
    return StartupPage(locale=locale)


def _landing_builder(*, locale: str) -> html.Div:
    """Return the landing page builder appropriate for the requesting device."""
    if _is_mobile_request():
        return LandingPageMobile(locale=locale)
    return LandingPage(locale=locale)


def _navbar(*, locale: str, pathname: Optional[str]) -> html.Div:
    """Return the compact mobile burger navbar or the desktop tabular menu."""
    current = pathname or "/"
    if _is_mobile_request():
        return MobileNavBar(locale=locale, current_page=current)
    return NavBar(locale=locale, current_page=current)


@app.callback(
    [Output('page-content', 'children', allow_duplicate=True),
     Output('mobile-warning', 'children', allow_duplicate=True),
     Output('navbar-container', 'children', allow_duplicate=True)],
    [Input('interface-language', 'data')],
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
) -> tuple:
    """Re-render page content and navbar when language changes.

    Pages with interactive state (prediction chart, ending) only get
    a navbar refresh -- page content is left untouched via per-element callbacks.
    """
    locale = normalize_locale(interface_language)
    navbar = _navbar(locale=locale, pathname=pathname)

    if pathname in _STATEFUL_PAGES:
        return no_update, no_update, navbar

    warning_content = render_mobile_warning(user_agent, locale=locale)
    if _is_staging_mode and pathname and pathname.startswith('/staging'):
        staging_layout = _staging_display(pathname, locale=locale, glucose_unit=glucose_unit)
        if staging_layout is not None:
            return staging_layout, warning_content, navbar
    if pathname == '/final':
        if user_info:
            return create_final_layout(full_df_data, user_info, glucose_unit, locale=locale), warning_content, navbar
        return no_update, no_update, navbar
    if pathname and pathname.startswith('/share/'):
        share_id = pathname.split('/share/', 1)[1].strip('/').split('/', 1)[0]
        record = share_store.load_share(share_id) if share_id else None
        if record is None:
            return create_expired_layout(locale=locale), warning_content, navbar
        share_url = _build_share_url(share_id)
        return create_share_layout(
            record, share_id=share_id, share_url=share_url, locale=locale,
        ), warning_content, navbar
    if pathname == "/consent-form":
        return ConsentFormPage(locale=locale), warning_content, navbar
    if pathname == '/startup':
        return _startup_builder(locale=locale), warning_content, navbar
    if pathname == '/about':
        return create_about_page(locale=locale), warning_content, navbar
    if pathname == '/contact':
        return create_contact_page(locale=locale), warning_content, navbar
    if pathname == '/demo':
        return create_demo_page(locale=locale), warning_content, navbar
    if pathname == '/faq':
        return create_faq_page(locale=locale), warning_content, navbar
    # Landing page
    return _landing_builder(locale=locale), warning_content, navbar


@app.callback(
    [Output('header-app-title', 'children'),
     Output('header-description', 'children'),
     Output('header-how-to-play', 'children'),
     Output('prediction-round-tagline', 'children'),
     Output('header-data-source-label', 'children'),
     Output('header-upload-prompt', 'children'),
     Output('use-example-data-button', 'children'),
     Output('header-time-window-label', 'children'),
     Output('prediction-units-label', 'children'),
     Output('prediction-consent-label', 'children'),
     Output('finish-study-button', 'children'),
     Output('nightscout-load-button', 'children')],
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
        "Prediction" if locale == "en" else t("ui.header.description_1", locale=locale),
        [
            html.Button(
                t("ui.header.how_to_play", locale=locale),
                id="header-how-to-play-toggle",
                className="header-how-to-play-toggle",
                type="button",
            ),
            html.Div(
                [
                    html.Button("×", id="header-how-to-play-close", className="header-how-to-play-close", type="button"),
                    html.Div(
                        [
                            t("ui.header.description_2", locale=locale) + " ",
                            t("ui.header.description_3", locale=locale),
                            html.Br(),
                            t("ui.header.how_to_play_1", locale=locale),
                            html.Br(),
                            t("ui.header.how_to_play_2", locale=locale),
                            html.Br(),
                            t("ui.header.how_to_play_3", locale=locale),
                        ],
                        className="header-how-to-play-body",
                    ),
                ],
                id="header-how-to-play-bubble",
                className="header-how-to-play-bubble",
                style={"display": "none"},
            ),
        ],
        t("ui.header.description_1", locale=locale),
        "Source:" if locale == "en" else t("ui.header.current_data_source", locale=locale),
        [
            t("ui.header.upload_prompt_1", locale=locale),
            html.A(t("ui.header.upload_prompt_2", locale=locale)),
        ],
        t("ui.header.use_example_data", locale=locale),
        t("ui.header.time_window_label", locale=locale),
        t("ui.prediction.units_label", locale=locale),
        t("ui.startup.data_usage_consent_label", locale=locale),
        t("ui.common.finish_exit", locale=locale),
        t("ui.header.nightscout_load_button", locale=locale),
    )


@app.callback(
    [Output('ending-title', 'children'),
     Output('ending-disclaimer-line1', 'children'),
     Output('ending-disclaimer-line2', 'children'),
     Output('ending-disclaimer-line3', 'children'),
     Output('ending-round-info', 'children'),
     Output('ending-gamification', 'children'),
     Output('ending-units-line', 'children'),
     Output('ending-graph-explanation', 'children'),
     Output('ending-prediction-results-title', 'children'),
     Output('ending-prediction-table', 'rowData'),
     Output('ending-prediction-table', 'columnDefs'),
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
    min_useful = int(user_info.get('min_useful_rounds') or MIN_USEFUL_ROUNDS) if user_info else MIN_USEFUL_ROUNDS
    prediction_table_data = user_info.get('prediction_table_data') if user_info else None
    current_mae = _compute_round_mae(prediction_table_data) if prediction_table_data else None
    all_rounds: list[dict[str, Any]] = (user_info.get('rounds') or []) if user_info else []

    metric_label_map: dict[str, str] = {
        "Actual Glucose": t("ui.table.actual_glucose", locale=locale),
        "Predicted": t("ui.table.predicted", locale=locale),
        "Absolute Error": t("ui.table.absolute_error", locale=locale),
        "Relative Error (%)": t("ui.table.relative_error_pct", locale=locale, pct="%"),
    }

    table_data: list[dict[str, str]] = no_update
    table_columns: list[dict[str, Any]] = no_update
    if user_info and 'prediction_table_data' in user_info:
        raw_table = _convert_table_data_units(user_info['prediction_table_data'], unit)
        table_data = []
        for row in raw_table:
            new_row = dict(row)
            new_row["metric"] = metric_label_map.get(str(row.get("metric", "")), str(row.get("metric", "")))
            table_data.append(new_row)
        table_columns = build_readonly_column_defs([{'name': t("ui.table.metric_header", locale=locale), 'id': 'metric'}] + [
            {'name': f'T{i}', 'id': f't{i}', 'type': 'text'}
            for i in range(len(raw_table[0]) - 1)
            if raw_table and raw_table[1].get(f't{i}', '-') != '-'
        ])

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
        _build_gamification_section(
            current_round=current_round_number,
            max_rounds=max_rounds,
            min_useful=min_useful,
            mae=current_mae,
            rounds=all_rounds,
            locale=locale,
            is_last_round=is_last_round,
        ).children,
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
     State('ai-prediction-store', 'data')],
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
    ai_data: Optional[str],
) -> tuple[html.Div, Optional[html.Div], html.Div]:
    has_ptd = bool(user_info and 'prediction_table_data' in user_info) if user_info else False
    has_full = bool(full_df_data)
    print(f"DEBUG display_page: pathname={pathname} has_user_info={user_info is not None} has_prediction_table_data={has_ptd} has_full_df={has_full}")
    locale = normalize_locale(interface_language)
    navbar = _navbar(locale=locale, pathname=pathname)
    
    with start_action(action_type=u"display_page", pathname=pathname, locale=locale):
        warning_content = render_mobile_warning(user_agent, locale=locale)
        if _is_staging_mode and pathname and pathname.startswith('/staging'):
            staging_layout = _staging_display(pathname, locale=locale, glucose_unit=glucose_unit)
            if staging_layout is not None:
                return staging_layout, warning_content, navbar
        if pathname == "/consent-form":
            return ConsentFormPage(locale=locale), warning_content, navbar
        # Consent guard: mandatory consent (acknowledge + GDPR) must have been
        # recorded before the game flow is reachable. `consent_completed` is set
        # by handle_landing_continue (desktop) and the mobile wizard's Start
        # handler. Without it, a direct-URL / burger-menu visit could otherwise
        # bypass the consent gate (desktop /startup omits the consent fields, so
        # handle_start_button would skip its own check).
        consent_done = bool(user_info and user_info.get('consent_completed'))
        if pathname == '/prediction' and user_info:
            if not consent_done:
                # No consent on record -> send the user to the consent entry.
                return (_landing_builder(locale=locale), warning_content, navbar)
            format_value = str(user_info.get("format") or "A")
            return create_prediction_layout(locale=locale, format_value=format_value, user_info=user_info), warning_content, navbar
        if pathname == '/startup':
            # On mobile, /startup IS the consent entry (wizard step 0), so it
            # must stay reachable without a prior consent record. On desktop,
            # consent lives on the landing page, so require it first.
            if not _is_mobile_request() and not consent_done:
                return (_landing_builder(locale=locale), warning_content, navbar)
            return (_startup_builder(locale=locale), warning_content, navbar)
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
            return create_ending_layout(full_df_data, current_df_data, events_df_data, user_info, glucose_unit, ai_data, locale=locale), warning_content, navbar
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
        if pathname and pathname.startswith('/share/'):
            share_id = pathname.split('/share/', 1)[1].strip('/').split('/', 1)[0]
            record = share_store.load_share(share_id) if share_id else None
            if record is None:
                return create_expired_layout(locale=locale), warning_content, navbar
            share_url = _build_share_url(share_id)
            return create_share_layout(
                record, share_id=share_id, share_url=share_url, locale=locale,
            ), warning_content, navbar
        if pathname == '/about':
            return create_about_page(locale=locale), warning_content, navbar
        if pathname == '/contact':
            return create_contact_page(locale=locale), warning_content, navbar
        if pathname == '/demo':
            return create_demo_page(locale=locale), warning_content, navbar
        if pathname == '/faq':
            return create_faq_page(locale=locale), warning_content, navbar
        # Default route: landing page
        return (_landing_builder(locale=locale), warning_content, navbar)

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
        html.Div(
            t("ui.contact.body", locale=locale),
            style={"marginBottom": "14px"},
            className="contact-intro",
        ),
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
                    className="contact-table",
                ),
                html.Hr(style={"margin": "18px 0"}),
            ]
        )

    if info.general_email:
        page_children.extend(
            [
                html.H2(t("ui.contact.general_email_title", locale=locale)),
                html.Div(
                    html.A(
                        info.general_email,
                        href=f"mailto:{info.general_email}",
                        style={"fontWeight": "700"},
                        className="contact-general-email",
                    ),
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
                className="contact-table",
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
                className="contact-table",
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
                className="contact-table",
            )
        )

    return html.Div(page_children, className="info-page contact-page", disable_n_clicks=True)


def create_demo_page(*, locale: str) -> html.Div:
    return html.Div(
        [
            html.H1(t("ui.common.video_instructions", locale=locale), disable_n_clicks=True),
            html.Div(
                t("ui.demo.body", locale=locale),
                style={"marginBottom": "18px"},
                disable_n_clicks=True,
            ),
            html.Div(
                html.Iframe(
                    id="demo-video-frame",
                    src="https://www.youtube.com/embed/M9JDhLFfFbA",
                    title=t("ui.common.video_instructions", locale=locale),
                    allow=(
                        "accelerometer; autoplay; clipboard-write; encrypted-media; fullscreen; "
                        "gyroscope; picture-in-picture; web-share"
                    ),
                    style={
                        "position": "absolute",
                        "top": "0",
                        "left": "0",
                        "width": "100%",
                        "height": "100%",
                        "border": "0",
                    },
                ),
                id="demo-video-shell",
                style={
                    "position": "relative",
                    "width": "100%",
                    "maxWidth": "960px",
                    "paddingBottom": "56.25%",
                    "height": "0",
                    "overflow": "hidden",
                    "borderRadius": "8px",
                    "backgroundColor": "#000",
                },
                disable_n_clicks=True,
            ),
            html.Button(
                t("ui.demo.fullscreen_video", locale=locale),
                id="demo-fullscreen-button",
                className="ui blue button demo-fullscreen-button",
                n_clicks=0,
                **{"aria-label": t("ui.demo.fullscreen_video", locale=locale)},
            ),
        ],
        className="info-page demo-page",
        disable_n_clicks=True,
    )


def create_prediction_layout(*, locale: str, format_value: str, user_info: Dict[str, Any]) -> html.Div:
    """Create the prediction page layout"""
    show_upload = format_value in ("B", "C")
    consent_given = bool(user_info.get("consent_use_uploaded_data", False))
    consent_value = ['agree'] if consent_given else []
    data_source_name = str(user_info.get("data_source_name") or "")
    if data_source_name:
        data_source_display = data_source_name
    elif format_value in ("B", "C"):
        data_source_display = t("ui.header.upload_required", locale=locale)
    else:
        data_source_display = "example.csv"
    return html.Div([
        HeaderComponent(
            show_time_slider=False,
            show_upload_section=show_upload,
            show_example_button=(format_value == "A"),
            show_data_source_section=False,
            initial_slider_value=example_initial_slider_value,
            locale=locale,
            data_source_name=data_source_display,
            className="prediction-header",
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
        html.Div(
            [
                html.Span(
                    t("ui.common.round_of", locale=locale, current=1, total=user_info.get("max_rounds", MAX_ROUNDS)).replace("Round", "Prediction Round", 1),
                    id="prediction-round-tagline",
                    className="prediction-round-tagline",
                    style={"display": "none"},
                ),
                html.Div(id='round-indicator', style={
                    'textAlign': 'center',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': '#2c5282',
                    'marginBottom': '10px'
                }),
            ],
            id="prediction-round-summary",
            disable_n_clicks=True,
        ),
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
        }, id='prediction-units-row'),
        html.Div([
            html.Div(
                GlucoseChart(id='glucose-graph', hide_last_hour=True),
                id='prediction-glucose-chart-container'
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Source:" if locale == "en" else t("ui.header.current_data_source", locale=locale),
                                id='header-data-source-label',
                                className="prediction-source-label",
                            ),
                            html.Div(id='data-source-display', children=data_source_display, className="prediction-source-name"),
                            html.Div(id='prediction-chart-meta', className="prediction-source-time"),
                        ],
                        className="prediction-source-line",
                    ),
                    html.Div(id='generic-source-metadata-display', children="", className="prediction-source-metadata"),
                ],
                id="prediction-source-plaque",
                disable_n_clicks=True,
            ),
            html.Div(
                [
                    html.Button(
                        t("ui.orientation.go_fullscreen", locale=locale),
                        id="prediction-fullscreen-button",
                        className="prediction-fullscreen-button",
                        type="button",
                    ),
                    SubmitComponent(locale=locale),
                ],
                id="prediction-mobile-actions",
            ),
        ], id='prediction-chart-submit-wrap', style={'flex': '1'})
    ], style={
        'margin': '0 auto',
        'padding': '0 20px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '20px'
    })


@app.callback(
    Output('prediction-chart-meta', 'children'),
    [Input('current-window-df', 'data'),
     Input('data-source-name', 'data')],
    [State('url', 'pathname')],
    prevent_initial_call=False
)
def update_prediction_chart_meta(
    current_df_data: Optional[dict[str, Any]],
    source_name: Optional[str],
    pathname: Optional[str],
) -> str:
    if pathname != '/prediction' or not current_df_data:
        raise PreventUpdate

    time_values = current_df_data.get('time') or []
    if not time_values:
        raise PreventUpdate

    start_time = datetime.fromisoformat(str(time_values[0])).strftime('%H:%M')
    end_time = datetime.fromisoformat(str(time_values[-1])).strftime('%H:%M')
    return f"{start_time}-{end_time}"


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
     Input('interface-language', 'data'),
     Input('user-agent', 'data')],
    prevent_initial_call=False
)
def update_round_indicator(
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    interface_language: Optional[str],
    user_agent: Optional[str],
) -> str:
    if pathname != '/prediction':
        raise PreventUpdate
    if not user_info:
        return ""
    rounds_played = len(user_info.get('rounds') or [])
    current_round = int(user_info.get('current_round_number') or (rounds_played + 1))
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    locale = normalize_locale(interface_language)
    round_text = t("ui.common.round_of", locale=locale, current=current_round, total=max_rounds)
    if _is_mobile_ua(user_agent):
        return round_text
    if locale == "en":
        return round_text.replace("Round", "Prediction Round", 1)
    return round_text


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
                t("ui.prediction.no_upload_back_to_final", locale=locale),
                id="back-to-final-from-upload",
                className="ui small button",
                style={"paddingLeft": "0", "marginTop": "6px"},
            ),
        ]
    return dbc.Alert(children, color="info", style={"marginBottom": "10px"})

def _compute_round_mae(prediction_table_data: list[dict[str, str]]) -> Optional[float]:
    """Extract MAE from raw prediction table data (always in mg/dL)."""
    if len(prediction_table_data) < 2:
        return None
    actual_row = prediction_table_data[0]
    pred_row = prediction_table_data[1]
    errors: list[float] = []
    for key in actual_row:
        if key == "metric":
            continue
        try:
            a, p = float(actual_row[key]), float(pred_row[key])
            errors.append(abs(a - p))
        except (ValueError, TypeError):
            continue
    return sum(errors) / len(errors) if errors else None


def _pick_reaction(mae: Optional[float], round_number: int, locale: str) -> str:
    bracket = pick_bracket(mae)
    pool = t_list(f"ui.ending.reaction.{bracket}", locale=locale)
    if not pool:
        return ""
    return pool[(round_number - 1) % len(pool)]


def _is_personal_best(mae: Optional[float], rounds: list[dict[str, Any]]) -> bool:
    if mae is None or not rounds:
        return False
    for r in rounds[:-1]:
        prev_mae = _compute_round_mae(r.get("prediction_table_data") or [])
        if prev_mae is not None and prev_mae <= mae:
            return False
    return len(rounds) > 1


def _pick_milestone(current_round: int, max_rounds: int, min_useful: int, locale: str) -> Optional[str]:
    if current_round == 1:
        return t("ui.ending.milestone.first_round", locale=locale)
    if current_round == min_useful:
        return t("ui.ending.milestone.minimum_reached", locale=locale)
    if current_round == max_rounds:
        return t("ui.ending.milestone.all_complete", locale=locale)
    return None


def _build_progress_bar(current_round: int, max_rounds: int, min_useful: int, locale: str) -> html.Div:
    """Two-phase segmented progress bar: green up to min_useful, gold for stretch."""
    segments: list[html.Div] = []
    for i in range(1, max_rounds + 1):
        is_min_phase = i <= min_useful
        filled = i <= current_round
        if filled:
            bg = "#4CBB17" if is_min_phase else "#D4A017"
        else:
            bg = "#e0e0e0" if is_min_phase else "#f5f0e0"
        border_right = "2px solid white" if i < max_rounds else "none"
        border_left = "3px solid #888" if i == min_useful + 1 else "none"
        segments.append(html.Div(
            disable_n_clicks=True,
            style={
                "flex": "1",
                "height": "22px",
                "backgroundColor": bg,
                "borderRight": border_right,
                "borderLeft": border_left,
                "transition": "background-color 0.3s",
            },
        ))

    labels = html.Div([
        html.Span(
            t("ui.ending.progress.minimum_goal", locale=locale, min_useful=min_useful),
            style={"fontSize": "13px", "color": "#4a5568", "fontWeight": "600"},
        ),
        html.Span(
            t("ui.ending.progress.stretch_goal", locale=locale, total=max_rounds),
            style={"fontSize": "13px", "color": "#9e7c16", "fontWeight": "600"},
        ),
    ], disable_n_clicks=True, style={
        "display": "flex",
        "justifyContent": "space-between",
        "marginTop": "4px",
    })

    return html.Div([
        html.Div(
            segments,
            disable_n_clicks=True,
            style={
                "display": "flex",
                "borderRadius": "10px",
                "overflow": "hidden",
                "border": "1px solid #bbb",
                "boxShadow": "inset 0 1px 3px rgba(0,0,0,0.15)",
            },
        ),
        labels,
    ], id="ending-progress-bar", disable_n_clicks=True, style={
        "maxWidth": "550px",
        "margin": "0 auto 10px auto",
    })


def _build_gamification_section(
    current_round: int,
    max_rounds: int,
    min_useful: int,
    mae: Optional[float],
    rounds: list[dict[str, Any]],
    locale: str,
    *,
    is_last_round: bool = False,
) -> html.Div:
    """Assemble progress bar, reaction line, milestone, motivation inside one card."""
    children: list[Any] = []

    children.append(_build_progress_bar(current_round, max_rounds, min_useful, locale))

    reaction = _pick_reaction(mae, current_round, locale)
    personal_best = _is_personal_best(mae, rounds)

    reaction_parts: list[Any] = []
    if reaction:
        reaction_parts.append(html.Span(reaction, id="ending-reaction-text"))
    if personal_best:
        if reaction_parts:
            reaction_parts.append("  ")
        reaction_parts.append(html.Span(
            t("ui.ending.personal_best", locale=locale),
            id="ending-personal-best",
            style={
                "fontWeight": "bold",
                "color": "#b8860b",
                "backgroundColor": "#fff8e1",
                "padding": "2px 10px",
                "borderRadius": "12px",
                "border": "1px solid #f0d060",
                "fontSize": "clamp(14px, 2vw, 17px)",
            },
        ))
    if not reaction_parts:
        reaction_parts.append(html.Span("", id="ending-reaction-text"))
        reaction_parts.append(html.Span("", id="ending-personal-best"))

    children.append(html.Div(
        reaction_parts,
        id="ending-reaction-line",
        disable_n_clicks=True,
        style={
            "textAlign": "center",
            "fontSize": "clamp(16px, 2.2vw, 20px)",
            "color": "#2c5282",
            "fontWeight": "500",
            "marginBottom": "6px",
            "minHeight": "28px",
            "lineHeight": "1.5",
        },
    ))

    milestone = _pick_milestone(current_round, max_rounds, min_useful, locale)
    children.append(html.Div(
        milestone or "",
        id="ending-milestone",
        disable_n_clicks=True,
        style={
            "textAlign": "center",
            "fontSize": "clamp(15px, 2vw, 18px)",
            "color": "#1b5e20",
            "fontWeight": "700",
            "marginBottom": "4px",
            "minHeight": "24px",
            "display": "block" if milestone else "none",
        },
    ))

    children.append(html.Div(
        t("ui.ending.round_motivation", locale=locale, total=max_rounds, min_useful=min_useful),
        id='ending-round-motivation',
        disable_n_clicks=True,
        style={
            'textAlign': 'center',
            'color': '#4a5568',
            'fontSize': '13px',
            'fontStyle': 'italic',
            'marginTop': '6px',
            'display': 'none' if is_last_round else 'block',
        }
    ))

    return html.Div(children, id="ending-gamification", disable_n_clicks=True, style={
        "maxWidth": "900px",
        "margin": "10px auto 12px auto",
        "padding": "16px 24px 12px 24px",
        "backgroundColor": "#f0f7ff",
        "borderRadius": "12px",
        "border": "1px solid #c5d9f0",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    })


def create_ending_layout(
    full_df_data: Optional[Dict],
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
    user_info: Optional[Dict] = None,
    glucose_unit: Optional[str] = None,
    ai_data: Optional[Dict] = None,
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

    
    # --- AI comparison (only when the user chose "Human vs AI" mode) ---
    ai_predicted_row: Optional[dict[str, str]] = None
    ai_metrics_display: list = []

    is_vs_ai_mode = bool(user_info and user_info.get("mode") == "vs_ai")
    glumind_data = (ai_data or {}).get("glumind") if ai_data else None

    if is_vs_ai_mode and glumind_data and glumind_data.get("ready"):
        ai_predictions = glumind_data["predictions"]
        actual_row = prediction_table_data[0]
        predicted_om_row = prediction_table_data[1]

        # Build the AI row using the same t0..t11 keys as the human row.
        # AI predictions only cover the hidden hour, so we align them to the
        # last len(ai_predictions) keys of the table.
        value_keys = [k for k in actual_row if k != "metric"]
        ai_value_keys = value_keys[-len(ai_predictions):]

        ai_predicted_row = {"metric": "Predicted (AI)"}
        ai_abs_error_row = {"metric": "Absolute Error (AI)"}
        ai_rel_error_row = {"metric": "Relative Error (AI) (%)"}

        for key, value in zip(ai_value_keys, ai_predictions):
            actual_str = actual_row.get(key, "-")
            ai_predicted_row[key] = f"{value:.1f}"
            if actual_str != "-":
                actual_val = float(actual_str)
                err = abs(actual_val - value)
                ai_abs_error_row[key] = f"{err:.1f}"
                ai_rel_error_row[key] = f"{(err / actual_val * 100):.1f}%" if actual_val != 0 else "-"
            else:
                ai_abs_error_row[key] = "-"
                ai_rel_error_row[key] = "-"

        for key in value_keys:
            if key not in ai_predicted_row:
                ai_predicted_row[key] = "-"
                ai_abs_error_row[key] = "-"
                ai_rel_error_row[key] = "-"

        # Three metric comparisons, reusing the existing calculation function.
        ai_vs_reality = metrics_component_ending._calculate_metrics_from_table_data(
            [actual_row, ai_predicted_row]
        )
        ai_vs_human = metrics_component_ending._calculate_metrics_from_table_data(
            [predicted_om_row, ai_predicted_row]
        )

        if ai_vs_reality:
            ai_metrics_display.append(
                html.H3("Accuracy Metrics - AI", style={
                    'textAlign': 'center',
                    'fontSize': 'clamp(20px, 3vw, 28px)',
                    'marginTop': '20px',
                    'marginBottom': 'clamp(10px, 2vh, 20px)'
                })
            )
            ai_metrics_display.extend(
                MetricsComponent.create_ending_metrics_display(ai_vs_reality, locale=locale)[1:]
            )
        if ai_vs_human:
            ai_metrics_display.append(
                html.H3("Accuracy Metrics - Human vs AI", style={
                    'textAlign': 'center',
                    'fontSize': 'clamp(20px, 3vw, 28px)',
                    'marginTop': '20px',
                    'marginBottom': 'clamp(10px, 2vh, 20px)'
                })
            )
            ai_metrics_display.extend(
                MetricsComponent.create_ending_metrics_display(ai_vs_human, locale=locale)[1:]
            )

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
    min_useful = int(user_info.get('min_useful_rounds') or MIN_USEFUL_ROUNDS) if user_info else MIN_USEFUL_ROUNDS
    current_mae = _compute_round_mae(prediction_table_data) if prediction_table_data else None
    all_rounds: list[dict[str, Any]] = (user_info.get('rounds') or []) if user_info else []
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
        meta_line = (
            f"{t('ui.startup.age_label', locale=locale)}: {meta.age} · "
            f"{t('ui.startup.gender_label', locale=locale)}: {gender_display} · "
            f"{t('ui.header.weight_label', locale=locale)}: {meta.weight}"
        )
        if meta.sensor:
            meta_line += f" · {t('ui.ending.sensor_label', locale=locale)}: {meta.sensor}"
        subject_parts.append(meta_line)
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
                html.I(className="close icon"),
                html.P(t("ui.results_disclaimer.line1", locale=locale), id='ending-disclaimer-line1', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line2", locale=locale), id='ending-disclaimer-line2', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line3", locale=locale), id='ending-disclaimer-line3', style={'margin': '0'}),
            ],
            className='ui warning message',
            disable_n_clicks=True,
            style={
                'maxWidth': '900px',
                'margin': '0 auto 15px auto',
                'fontSize': '14px',
                'lineHeight': '1.4',
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
        _build_gamification_section(
            current_round=current_round_number,
            max_rounds=max_rounds,
            min_useful=min_useful,
            mae=current_mae,
            rounds=all_rounds,
            locale=locale,
            is_last_round=is_last_round,
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
                        ai_data if is_vs_ai_mode else None,
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
            build_readonly_ag_grid(
                table_id='ending-prediction-table',
                row_data=prediction_table_data_display,
                column_defs=build_readonly_column_defs(
                    [{'name': t("ui.table.metric_header", locale=locale), 'id': 'metric'}] + [
                        {'name': f'T{i}', 'id': f't{i}', 'type': 'text'}
                        for i in range(len(prediction_table_data[0]) - 1)
                        if prediction_table_data
                        and prediction_table_data[1].get(f't{i}', '-') != '-'
                    ]
                ),
                style={
                    'width': '100%',
                    'height': 'auto',
                    'maxHeight': 'clamp(300px, 40vh, 500px)',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                },
                highlight_first_two_rows=True,
                has_ai_rows=is_vs_ai_mode,
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
            metrics_display + ai_metrics_display,
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
        # Cross-device resume: copy a "?resume=<code>" link from the between-rounds
        # summary (there's screen budget here, unlike the in-round chart page) so
        # the player can continue this session on another device.
        html.Div(
            html.Button(
                t("ui.resume_code.copy_link", locale=locale),
                id='ending-copy-link-button',
                type='button',
                **{"data-copied-text": t("ui.resume_code.copied", locale=locale)},
                style={
                    'backgroundColor': '#ffffff',
                    'color': '#2185d0',
                    'padding': 'clamp(10px, 1.6vw, 14px) clamp(16px, 2.4vw, 22px)',
                    'border': '1px solid #2185d0',
                    'borderRadius': '8px',
                    'fontSize': 'clamp(14px, 2vw, 18px)',
                    'fontWeight': '700',
                    'cursor': 'pointer',
                    'maxWidth': '400px',
                    'width': '100%',
                },
            ),
            id='ending-copy-link-row',
            disable_n_clicks=True,
            style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '12px', 'padding': '0 10px'},
        ),
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


def _build_ai_aggregate_table_data(rounds: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a synthetic table_data for aggregated AI metrics across rounds.

    Mirrors ``_build_aggregate_table_data`` above, but stitches together the
    "Actual Glucose" values against the AI's ``ai_predicted_row`` (saved per
    round by ``handle_submit_button`` when the round was played in "vs_ai"
    mode). Rounds without an AI prediction (e.g. played in "solo" mode) are
    skipped, so mixed-mode sessions only aggregate the rounds that actually
    have an AI comparison.
    """
    actual_row: dict[str, str] = {'metric': 'Actual Glucose'}
    ai_row: dict[str, str] = {'metric': 'Predicted (AI)'}
    out_idx = 0

    for round_info in rounds:
        table_data = round_info.get('prediction_table_data') or []
        ai_pred_row = round_info.get('ai_predicted_row')
        if len(table_data) < 1 or not ai_pred_row:
            continue

        round_actual = table_data[0]

        i = 0
        while True:
            key = f"t{i}"
            if key not in round_actual or key not in ai_pred_row:
                break
            av = round_actual.get(key, "-")
            pv = ai_pred_row.get(key, "-")
            if av != "-" and pv != "-":
                actual_row[f"t{out_idx}"] = av
                ai_row[f"t{out_idx}"] = pv
                out_idx += 1
            i += 1

    return [actual_row, ai_row]


def _average_ai_comparison_scores(rounds: list[dict[str, Any]]) -> Optional[dict[str, dict[str, Optional[float]]]]:
    """Average the per-round ``ai_comparison`` (vs_reality / vs_human) scores.

    Each round that was played in "vs_ai" mode already carries its own set of
    computed metrics (see ``_compute_ai_comparison_scores`` in
    ``handle_submit_button``), rather than raw point-by-point predictions for
    ``vs_human`` -- the human's drawn predictions aren't preserved per-round in
    a form we can re-aggregate directly. So instead of recomputing MAE/MSE/
    RMSE/MAPE from scratch (like the human's own aggregate does), we average
    the metric values already stored per round. This is a reasonable
    approximation across rounds of equal length (fixed 12-point horizon), and
    keeps the "AI vs Human" comparison consistent with what was shown on each
    round's own /ending page.
    """
    per_round: list[dict[str, dict[str, Optional[float]]]] = [
        r["ai_comparison"]["glumind"]
        for r in rounds
        if r.get("ai_comparison", {}).get("glumind")
    ]
    if not per_round:
        return None

    def _avg(kind: str) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for metric_name in ("MAE", "MSE", "RMSE", "MAPE"):
            values = [
                r[kind][metric_name]
                for r in per_round
                if r.get(kind, {}).get(metric_name) is not None
            ]
            out[metric_name] = (sum(values) / len(values)) if values else None
        return out

    return {"vs_reality": _avg("vs_reality"), "vs_human": _avg("vs_human")}


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

    # --- Human vs AI (issue #49) — aggregated across all rounds played in
    # "vs_ai" mode, mirroring the 3-set layout already on /ending (Livia's
    # feedback: "on the final page is a bit weird to have only one set of
    # accuracy when on every in between page i had the whole 3 sets"). ---
    is_vs_ai_session = any(r.get("ai_comparison", {}).get("glumind") for r in rounds)
    ai_final_metrics_display: list = []

    if is_vs_ai_session:
        # AI vs reality: same aggregate-then-score approach as the human's
        # own overall metrics above, reusing the same calculation function.
        ai_aggregate_table_data = _convert_table_data_units(_build_ai_aggregate_table_data(rounds), unit)
        ai_vs_reality_overall = metrics_component_final._calculate_metrics_from_table_data(ai_aggregate_table_data)

        # AI vs human: point-by-point human predictions aren't retained per
        # round, so we average the per-round vs_human scores instead (see
        # docstring on _average_ai_comparison_scores for why).
        averaged = _average_ai_comparison_scores(rounds)

        def _as_metrics_dict(values: Optional[dict[str, Optional[float]]]) -> Optional[dict[str, Any]]:
            if not values or all(v is None for v in values.values()):
                return None
            return {name: {"value": val, "description": ""} for name, val in values.items() if val is not None}

        ai_vs_reality_display_metrics = ai_vs_reality_overall or _as_metrics_dict(
            (averaged or {}).get("vs_reality")
        )
        ai_vs_human_display_metrics = _as_metrics_dict((averaged or {}).get("vs_human"))

        if ai_vs_reality_display_metrics:
            ai_final_metrics_display.append(
                html.H3("Accuracy Metrics - AI", style={
                    'textAlign': 'center',
                    'fontSize': 'clamp(20px, 3vw, 28px)',
                    'marginTop': '20px',
                    'marginBottom': 'clamp(10px, 2vh, 20px)',
                })
            )
            ai_final_metrics_display.extend(
                MetricsComponent.create_ending_metrics_display(ai_vs_reality_display_metrics, locale=locale)[1:]
            )
        if ai_vs_human_display_metrics:
            ai_final_metrics_display.append(
                html.H3("Accuracy Metrics - Human vs AI", style={
                    'textAlign': 'center',
                    'fontSize': 'clamp(20px, 3vw, 28px)',
                    'marginTop': '20px',
                    'marginBottom': 'clamp(10px, 2vh, 20px)',
                })
            )
            ai_final_metrics_display.extend(
                MetricsComponent.create_ending_metrics_display(ai_vs_human_display_metrics, locale=locale)[1:]
            )

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
                html.I(className="close icon"),
                html.P(t("ui.results_disclaimer.line1", locale=locale), id='final-disclaimer-line1', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line2", locale=locale), id='final-disclaimer-line2', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line3", locale=locale), id='final-disclaimer-line3', style={'margin': '0'}),
            ],
            className='ui warning message',
            disable_n_clicks=True,
            style={
                'maxWidth': '900px',
                'margin': '0 auto 15px auto',
                'fontSize': '14px',
                'lineHeight': '1.4',
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
            overall_metrics_display + ai_final_metrics_display,
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
            build_readonly_ag_grid(
                table_id='final-rounds-table',
                row_data=round_rows,
                column_defs=build_readonly_column_defs(
                    [
                        {'name': 'Round', 'id': 'Round', 'type': 'numeric'},
                        {'name': 'Pairs', 'id': 'Pairs', 'type': 'numeric'},
                        {'name': 'MAE', 'id': 'MAE', 'type': 'numeric'},
                        {'name': 'MSE', 'id': 'MSE', 'type': 'numeric'},
                        {'name': 'RMSE', 'id': 'RMSE', 'type': 'numeric'},
                        {'name': 'MAPE', 'id': 'MAPE', 'type': 'numeric'},
                    ],
                    fixed_decimal_fields={'MAE', 'MSE', 'RMSE', 'MAPE'},
                ),
                style={
                    'width': '100%',
                    'overflowX': 'auto',
                },
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
                t("ui.share.button_share", locale=locale),
                id='share-results-button',
                n_clicks=0,
                className="ui green button",
                style={
                    'backgroundColor': '#4CBB17',
                    'color': 'white',
                    'padding': 'clamp(15px, 2vw, 20px) clamp(20px, 3vw, 30px)',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': 'clamp(18px, 3vw, 24px)',
                    'fontWeight': '700',
                    'cursor': 'pointer',
                    'minWidth': '200px',
                    'maxWidth': '400px',
                    'width': '100%',
                    'height': 'clamp(60px, 8vh, 80px)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2',
                    'marginBottom': '14px',
                },
            ),
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
            )
        ], disable_n_clicks=True, style={
            'display': 'flex',
            'flexDirection': 'column',
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
    """Deprecated: the yellow mobile banner has been replaced by the
    orientation-prompt overlay (see `assets/orientation.css` and the
    `orientation-overlay` div in `app.layout`).  We keep the function and
    its call sites returning ``None`` to avoid churn in every page-render
    callback; the `mobile-warning` div stays in the DOM purely as a
    throwaway Output for the clientside `mobile-device` class setter.
    """
    _ = user_agent, locale
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
        'time': pl.Series(events_data['time'], dtype=pl.String).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
        'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
        # Use pre-processed float values
        'insulin_value': pl.Series(insulin_values, dtype=pl.Float64)
    })

@app.callback(
    [Output('url', 'pathname'),
     Output('user-info-store', 'data')],
    [Input('start-button', 'n_clicks'),
    Input('start-vs-ai-button', 'n_clicks')],
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
def handle_start_button(n_clicks: Optional[int], n_clicks_vs_ai: Optional[int], email: Optional[str], age: Optional[int | float],
                       gender: Optional[str], uses_cgm: Optional[bool], cgm_duration_years: Optional[float],
                       format_value: Optional[str], data_usage_consent: Optional[list[str]],
                       diabetic: Optional[bool], diabetic_type: Optional[str],
                       diabetes_duration: Optional[float], location: Optional[str],
                       existing_user_info: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """Handle start button on startup page.

    Consent is recorded BEFORE this callback runs -- on desktop by
    handle_landing_continue, on mobile by record_mobile_consent -- and arrives
    here via `existing_user_info`. This callback must NOT take the landing-only
    consent components (`consent-acknowledge`, `consent-gdpr`, ...) as State:
    those components live only in the desktop landing page and the mobile wizard
    step 0, so they are absent from the *desktop* /startup DOM. Dash refuses to
    fire a callback whose Input/State components aren't all in the layout, so
    referencing them left the desktop Start button inert (it activated but
    navigated nowhere). See record_mobile_consent below for the mobile path.
    """
    if not n_clicks and not n_clicks_vs_ai:
        return no_update, no_update

    triggered_id = ctx.triggered_id
    selected_mode = "vs_ai" if triggered_id == "start-vs-ai-button" else "solo"

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
            # Stable cross-device resume code (server-side savegame key).
            'resume_code': info.get('resume_code') or resume_store.new_code(),
            'email': email or info.get('email') or '',
            'age': age,
            'gender': gender,
            'uses_cgm': uses_cgm_bool,
            'cgm_duration_years': cgm_duration_years,
            'format': format_value,
            'run_format': format_value,
            'mode': selected_mode,
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
        consent_row: Dict[str, Any] = {
            "study_id": info["study_id"],
            "number": info.get("number", ""),
            "timestamp": info.get("consent_timestamp", ""),
            "gdpr_consent": bool(info.get("consent_gdpr", False)),
            "upload_own_data": bool(info.get("consent_upload_own_data", False)),
            "play_only": bool(info.get("consent_play_only", False)),
            "participate_in_study": bool(info.get("consent_participate_in_study", False)),
            "receive_results_later": bool(info.get("consent_receive_results_later", False)),
            "keep_up_to_date": bool(info.get("consent_keep_up_to_date", False)),
            "no_selection": bool(info.get("consent_no_selection", True)),
        }
        ensure_consent_agreement_row(consent_row)
        return '/prediction', info
    return no_update, no_update


@app.callback(
    Output('user-info-store', 'data', allow_duplicate=True),
    [Input('consent-acknowledge', 'value'),
     Input('consent-gdpr', 'value'),
     Input('consent-upload-own-data', 'value'),
     Input('consent-play-only', 'value'),
     Input('consent-receive-results', 'value'),
     Input('consent-keep-updated', 'value')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def record_mobile_consent(
    acknowledge_value: Optional[list[str]],
    gdpr_value: Optional[list[str]],
    upload_own_data_value: Optional[list[str]],
    play_only_value: Optional[list[str]],
    receive_results_value: Optional[list[str]],
    keep_updated_value: Optional[list[str]],
    existing_user_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mirror the mobile wizard's consent choices into `user-info-store`.

    On mobile, consent lives in StartupPageMobile wizard step 0 (the consent
    components are imported from landing.py). Nothing else records consent on
    mobile before Start, so this callback does it -- writing `consent_completed`
    once the two mandatory boxes are ticked, exactly like handle_landing_continue
    does on desktop. That lets handle_start_button read consent from
    `user-info-store` on BOTH platforms instead of taking these landing-only
    components as State (which broke desktop -- see that callback).

    The same consent components also render on the desktop landing page, so a
    UA guard below restricts this callback to mobile; desktop consent stays
    owned by handle_landing_continue.

    No CSV row is written here (this fires on every checkbox toggle);
    handle_start_button writes the consent-agreement row once on Start.
    """
    from datetime import datetime

    # The consent components also exist on the *desktop* landing page, so this
    # callback would otherwise fire there too and race handle_landing_continue.
    # Restrict it to mobile (where it is the only consent recorder); on desktop,
    # handle_landing_continue owns consent recording untouched.
    if not _is_mobile_request():
        raise PreventUpdate

    acknowledged = bool(acknowledge_value and "ack" in acknowledge_value)
    gdpr_consented = bool(gdpr_value and "gdpr" in gdpr_value)
    if not (acknowledged and gdpr_consented):
        # Mandatory consent not yet complete; don't mark it done. The wizard's
        # gate_mobile_consent_step keeps the Next button disabled until then.
        raise PreventUpdate

    info: Dict[str, Any] = dict(existing_user_info or {})
    if not info.get("study_id"):
        info["study_id"] = str(uuid.uuid4())

    any_selected = bool(play_only_value) or bool(receive_results_value) or bool(keep_updated_value)
    no_selection = not any_selected
    upload_own_data = bool(upload_own_data_value and "upload_own_data" in upload_own_data_value)
    play_only = bool(play_only_value and "play_only" in play_only_value)
    receive_results = bool(receive_results_value and "receive_results" in receive_results_value)
    keep_updated = bool(keep_updated_value and "keep_updated" in keep_updated_value)

    info.update({
        "consent_gdpr": gdpr_consented,
        "consent_upload_own_data": upload_own_data,
        "consent_play_only": play_only,
        "consent_participate_in_study": (not play_only) and (not no_selection),
        "consent_receive_results_later": receive_results,
        "consent_keep_up_to_date": keep_updated,
        "consent_no_selection": no_selection,
        "consent_timestamp": info.get("consent_timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "consent_completed": True,
        # Stable cross-device resume code, assigned at consent like on desktop.
        "resume_code": info.get("resume_code") or resume_store.new_code(),
    })
    return info


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
     State('time-slider', 'value'),
     State('ai-prediction-store', 'data')],
    prevent_initial_call=True
)
def handle_submit_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict],
    current_df_data: Optional[Dict],
    slider_value: Optional[int],
    ai_prediction_data: Optional[Dict[str, Any]],
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

        # --- Human vs AI (issue #49) — Persoana A: backend/AI ---
        # Dacă runda a fost jucată în modul "vs_ai" și predicția GluMind e gata
        # în ai-prediction-store (pornită încă de la începutul rundei, vezi cele
        # 3 locuri unde se generează fereastra), adăugăm rândurile GluMind în
        # tabel și calculăm cele 2 seturi de scoruri (vs_reality, vs_human).
        ai_comparison_round: dict[str, Any] = {}
        # Rândul brut de predicție AI, salvat separat lângă restul rundei, ca să
        # poată fi reagregat pe /final la fel cum se agregă rândul omului (vezi
        # _build_ai_aggregate_table_data mai jos în fișier) — ai_comparison
        # ține doar scorurile deja calculate, nu și predicțiile punct-cu-punct.
        ai_predicted_row_for_round: Optional[dict[str, str]] = None
        if user_info.get("mode") == "vs_ai":
            glumind_data = (ai_prediction_data or {}).get("glumind") or {}
            ai_predictions = glumind_data.get("predictions") or []
            ai_ready = bool(glumind_data.get("ready")) and bool(ai_predictions)

            if not ai_ready:
                # Contract-ul cu Persoana B: dacă modelul n-a apucat să răspundă
                # până la submit, recalculăm sincron aici ca fallback, ca omul
                # să nu rămână fără comparație doar din cauza unei curse cu UI-ul.
                fallback = _compute_ai_forecast_if_needed(convert_df_to_dict(current_df), user_info)
                glumind_data = fallback.get("glumind") or {}
                ai_predictions = glumind_data.get("predictions") or []
                ai_ready = bool(glumind_data.get("ready")) and bool(ai_predictions)

            if ai_ready:
                gl_values = current_df.get_column("gl").to_list()
                human_predictions = current_df.get_column("prediction").to_list()
                offset = PREDICTION_HOUR_OFFSET

                ai_rows = _build_ai_table_rows(gl_values, ai_predictions, offset)
                prediction_table_data = prediction_table_data + ai_rows
                ai_predicted_row_for_round = ai_rows[0]  # "Predicted (GluMind)" row

                ai_comparison_round = {
                    "glumind": _compute_ai_comparison_scores(
                        gl_values, human_predictions, ai_predictions, offset
                    )
                }

        user_info['prediction_table_data'] = prediction_table_data
        user_info['current_round_number'] = round_number
        if ai_comparison_round:
            user_info['ai_comparison'] = ai_comparison_round

        round_info: dict[str, Any] = {
            'round_number': round_number,
            'prediction_window_start': user_info['prediction_window_start'],
            'prediction_window_size': user_info['prediction_window_size'],
            'prediction_table_data': prediction_table_data,
            'ai_predicted_row': ai_predicted_row_for_round,
            'format': str(user_info.get('format') or ''),
            'is_example_data': bool(user_info.get('is_example_data', True)),
            'data_source_name': str(user_info.get('data_source_name', 'example.csv')),
        }
        if ai_comparison_round:
            # Salvate per rundă, lângă restul rezultatelor rundei — folosesc
            # exact același loc/format de fișier unde se salvează deja rundele
            # omului (submit_component.save_statistics(...) mai jos serializează
            # tot user_info/rounds, deci intră automat în același sistem).
            round_info['ai_comparison'] = ai_comparison_round
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
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('ai-prediction-store', 'data', allow_duplicate=True)],
    [Input('next-round-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_next_round_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict]
) -> Tuple[str, Dict[str, Any], Dict[str, bool], Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, Dict[str, Any]]:
    print(f"DEBUG handle_next_round_button FIRED: n_clicks={n_clicks}")
    if not n_clicks or not user_info:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    next_round_number = len(rounds) + 1
    if next_round_number > max_rounds:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

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
                return '/prediction', user_info, {'hide_last_hour': True}, no_update, no_update, no_update, False, "", False, 0, no_update
            full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
            is_example = False
            source_name = str(user_info.get("uploaded_data_filename") or user_info.get("data_source_name") or "uploaded.csv")
        else:
            # Format C: alternate between uploaded (odd rounds) and example (even rounds)
            uploaded_path = user_info.get("uploaded_data_path")
            if not uploaded_path:
                return '/prediction', user_info, {'hide_last_hour': True}, no_update, no_update, no_update, False, "", False, 0, no_update
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

        # 3rd of the 3 places a fresh window is generated: kick off GluMind
        # for this new round right away, in parallel with the human playing.
        new_window_dict = convert_df_to_dict(new_df)
        ai_forecast_data = _compute_ai_forecast_if_needed(new_window_dict, user_info)

        return (
            '/prediction',
            user_info,
            chart_mode,
            convert_df_to_dict(full_df),
            new_window_dict,
            convert_events_df_to_dict(events_df),
            is_example,
            source_name,
            False,  # let slider init set it from initial-slider-value
            random_start,
            ai_forecast_data,
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
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True)],
    [Input('finish-study-button-ending', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_finish_study_from_ending(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    full_df_data: Optional[Dict]
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool]]:
    print(f"DEBUG handle_finish_study_from_ending FIRED: n_clicks={n_clicks}")
    if not n_clicks:
        return no_update, no_update, no_update

    with start_action(action_type=u"handle_finish_study_from_ending", n_clicks=int(n_clicks)):
        pass

    if not user_info:
        return '/final', None, {'hide_last_hour': True}

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    if not rounds:
        return '/final', user_info, {'hide_last_hour': True}

    play_only = bool(user_info.get('consent_play_only')) if user_info else False
    if full_df_data and (not play_only) and not bool(user_info.get('statistics_saved')):
        with start_action(action_type=u"handle_finish_study_from_ending"):
            full_df = reconstruct_dataframe_from_dict(full_df_data)
            submit_component.save_statistics(full_df, user_info)
            user_info['statistics_saved'] = True

    return '/final', user_info, {'hide_last_hour': False}


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
    prevent_initial_call=True
)
def handle_restart_button(n_clicks: Optional[int]) -> tuple:
    """Reset session state for the "Exit" button on ``/final``."""
    if not n_clicks:
        raise PreventUpdate
    with start_action(action_type=u"handle_restart_button") as action:
        action.log(message_type="restart_clicked")
    return _full_session_reset()


def _full_session_reset() -> tuple:
    """Return the tuple consumed by the restart / play-again callbacks.

    Mirrors every ``Output`` in the decorators below: navigates to ``/``,
    nulls persisted session stores, and raises ``clean-storage-flag=True``
    so the clientside hook wipes ``localStorage`` too.
    """
    return (
        '/',                       # url pathname
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
        True,                      # clean-storage-flag
        True,                      # session-active
    )


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
    [Input('share-play-again-button', 'n_clicks')],
    prevent_initial_call=True,
)
def handle_share_play_again(n_clicks: Optional[int]) -> tuple:
    """Reset session state for "Play again" on ``/share/<id>``.

    The share page is dynamic -- it only mounts when a user is on
    ``/share/<id>``. `suppress_callback_exceptions=True` on the Dash app lets
    us register this callback anyway; it fires only when the button actually
    exists in the DOM.  Using a dedicated callback (rather than adding this
    input to ``handle_restart_button``) keeps each handler's input list
    stable for Dash's initial-layout validation.
    """
    if not n_clicks:
        raise PreventUpdate
    with start_action(action_type=u"handle_share_play_again") as action:
        action.log(message_type="share_play_again_clicked")
    return _full_session_reset()


@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input('share-results-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('interface-language', 'data')],
    prevent_initial_call=True,
)
def handle_share_results_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    interface_language: Optional[str],
) -> str:
    """Persist a share record and navigate the user to the public share page.

    The share record MUST capture every round the user has played across
    every format they've tried, not just the currently-active run.  The
    final page shows both; the share page must do the same or it'd hide
    prior achievements.
    """
    if not n_clicks or not user_info:
        raise PreventUpdate
    with start_action(action_type=u"handle_share_results_button") as action:
        current_rounds: list[dict[str, Any]] = list(user_info.get("rounds") or [])
        current_format: str = str(user_info.get("format") or "")

        # Tag currently-playing rounds with their format if missing, so the
        # share page can split them by format even after we merge archives.
        tagged_current: list[dict[str, Any]] = []
        for rnd in current_rounds:
            r = dict(rnd)
            if not r.get("format"):
                r["format"] = current_format
            tagged_current.append(r)

        # Merge archived runs (one key per previously-completed format run).
        # Each archived run is already a list of round dicts with its own format.
        archived_rounds: list[dict[str, Any]] = []
        runs_by_format: dict[str, list[dict[str, Any]]] = dict(user_info.get("runs_by_format") or {})
        for fmt_key, runs in runs_by_format.items():
            for run in (runs or []):
                for rnd in (run.get("rounds") or []):
                    r = dict(rnd)
                    if not r.get("format"):
                        r["format"] = fmt_key
                    archived_rounds.append(r)

        all_rounds: list[dict[str, Any]] = archived_rounds + tagged_current
        if not all_rounds:
            action.log(message_type=u"no_rounds_to_share")
            raise PreventUpdate

        # Figure out which formats the user has actually played (for the
        # ranking block).  Include the current format if it has rounds.
        played_formats: set[str] = {str(r.get("format") or "") for r in all_rounds}
        played_formats.discard("")

        study_id: str = str(user_info.get("study_id") or "")
        rankings: dict[str, Any] = compute_share_rankings(study_id, sorted(played_formats))

        # Strip the share record to JSON-safe primitives so it survives a
        # round-trip through JSON on disk.  `prediction_table_data` is already
        # a list of {str: str}; round_info is shallow dicts of primitives.
        share_record: dict[str, Any] = {
            "schema_version": 2,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "locale": normalize_locale(interface_language),
            "rounds": all_rounds,
            "played_formats": sorted(played_formats, key=lambda x: FORMAT_ORDER.get(str(x), 999)),
            "rankings": rankings,
            "user_info": {
                "name": str(user_info.get("name") or ""),
                "study_id": study_id,
                "format": current_format,
                "uses_cgm": bool(user_info.get("uses_cgm", False)),
                "max_rounds": int(user_info.get("max_rounds") or MAX_ROUNDS),
            },
        }
        share_id: str = share_store.save_share(share_record)
        action.log(
            message_type=u"share_saved",
            share_id=share_id,
            total_rounds=len(all_rounds),
            archived_rounds=len(archived_rounds),
            current_rounds=len(tagged_current),
            played_formats=sorted(played_formats),
        )
    return f"/share/{share_id}"


# Clientside: clipboard copy for the "Copy link" button on the share page.
app.clientside_callback(
    """
    function(n_clicks, url) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        if (!url) { return window.dash_clientside.no_update; }
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(url);
            } else {
                var ta = document.createElement('textarea');
                ta.value = url;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
        } catch (e) { /* ignore */ }
        var feedback = document.getElementById('share-copy-link-feedback');
        if (feedback) {
            feedback.style.opacity = '1';
            setTimeout(function() { feedback.style.opacity = '0'; }, 1800);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('share-copy-link-feedback', 'children'),
    Input('share-copy-link-button', 'n_clicks'),
    State('share-url-value', 'children'),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(n_clicks, url) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        if (!url) { return window.dash_clientside.no_update; }
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(url);
            } else {
                var ta = document.createElement('textarea');
                ta.value = url;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
        } catch (e) { /* ignore */ }
        window.open('https://discord.com/channels/@me', '_blank', 'noopener,noreferrer,width=980,height=720');
        var feedback = document.getElementById('share-copy-link-feedback');
        if (feedback) {
            feedback.style.opacity = '1';
            setTimeout(function() { feedback.style.opacity = '0'; }, 1800);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('share-copy-link-feedback', 'style'),
    Input('share-discord-button', 'n_clicks'),
    State('share-url-value', 'children'),
    prevent_initial_call=True,
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
        Output('ai-prediction-store', 'data', allow_duplicate=True),
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
    Optional[Dict[str, Any]],
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
            no_update,
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
            # Same as the other 2 window-generation sites: kick off GluMind for
            # this freshly-switched round right away, so it's ready by submit.
            new_window_dict = convert_df_to_dict(new_df)
            ai_forecast_data = _compute_ai_forecast_if_needed(new_window_dict, info)
            return (
                "/prediction",
                info,
                chart_mode,
                convert_df_to_dict(full_df),
                new_window_dict,
                convert_events_df_to_dict(events_df),
                True,
                "example.csv",
                False,
                random_start,
                None,
                ai_forecast_data,
            )

        if target_format in ("B", "C") and uploaded_path:
            full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            new_df, random_start = get_random_data_window(full_df, points)
            new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
            source_name = str(info.get("uploaded_data_filename") or info.get("data_source_name") or "uploaded.csv")
            info["is_example_data"] = False
            info["data_source_name"] = source_name
            new_window_dict = convert_df_to_dict(new_df)
            ai_forecast_data = _compute_ai_forecast_if_needed(new_window_dict, info)
            return (
                "/prediction",
                info,
                chart_mode,
                convert_df_to_dict(full_df),
                new_window_dict,
                convert_events_df_to_dict(events_df),
                False,
                source_name,
                False,
                random_start,
                None,
                ai_forecast_data,
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
            no_update,
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
            "resume_code": (user_info or {}).get("resume_code"),
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
            *([
                html.Div(
                    t("ui.resume_dialog.your_code", locale=locale, code=dialog_data.get("resume_code")),
                    style={
                        'fontSize': '13px', 'color': '#2b6cb0', 'backgroundColor': '#ebf4ff',
                        'border': '1px solid #bcd4f0', 'borderRadius': '6px',
                        'padding': '8px 12px', 'marginBottom': '20px', 'wordBreak': 'break-all',
                    },
                    disable_n_clicks=True,
                )
            ] if dialog_data.get("resume_code") else []),
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
    [Input('resume-start-over-btn', 'n_clicks')],
    prevent_initial_call=True,
)
def handle_resume_start_over(
    n_clicks: Optional[int],
) -> tuple:
    """Reset all in-memory stores and trigger the clean-storage-flag to wipe localStorage."""
    if not n_clicks:
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


# ---------------------------------------------------------------------------
# Cross-device resume (server-side savegame keyed by a short resume code).
#
# localStorage is per-device, so a session started on a phone can't be continued
# on a desktop. We keep a server-side snapshot of the live session (resume_store)
# keyed by user_info['resume_code'] and let the user re-enter that code on another
# device to restore it. Entirely additive: the auto-snapshot only reads stores,
# and the redeem callbacks are gated on an explicit code so they never perturb the
# normal in-tab persistence/resume flow.
# ---------------------------------------------------------------------------
def _resume_payload(
    user_info: Optional[Dict[str, Any]],
    full_df: Optional[Dict],
    current_df: Optional[Dict],
    events_df: Optional[Dict],
    last_page: Optional[str],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
) -> Dict[str, Any]:
    """Thin JSON-serialisable snapshot of the stores needed to restore a game."""
    return {
        "user_info": user_info,
        "full_df": full_df,
        "current_window_df": current_df,
        "events_df": events_df,
        "last_visited_page": last_page,
        "glucose_unit": glucose_unit,
        "interface_language": interface_language,
    }


@app.callback(
    Output('resume-sync', 'data'),
    [Input('user-info-store', 'data'),
     Input('last-visited-page', 'data'),
     Input('glucose-unit', 'data'),
     Input('interface-language', 'data')],
    [State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('events-df', 'data')],
    prevent_initial_call=True,
)
def auto_snapshot_session(
    user_info: Optional[Dict[str, Any]],
    last_page: Optional[str],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
    full_df: Optional[Dict],
    current_df: Optional[Dict],
    events_df: Optional[Dict],
) -> Any:
    """Persist the live session to resume_store at meaningful boundaries.

    Triggers on user_info / navigation / unit / language changes (round
    completions and page moves) and captures the dataframes via State, so it does
    NOT fire on every in-progress drawline (current-window-df) update. Keyed by
    user_info['resume_code']; only runs for consented sessions. Reads stores only
    (the Output is a throwaway sink), so it cannot disturb the in-browser
    persistence/resume contract.
    """
    if not user_info or not user_info.get('consent_completed'):
        raise PreventUpdate
    code = user_info.get('resume_code')
    if not code:
        raise PreventUpdate
    resume_store.save_session(
        code,
        _resume_payload(user_info, full_df, current_df, events_df, last_page, glucose_unit, interface_language),
    )
    return code


def _restore_outputs_from_code(code: Optional[str]) -> Optional[tuple]:
    """Load a session by code and return the store-output tuple, or None if missing.

    Output order matches the redeem callbacks:
    (pathname, user_info, full_df, current_window_df, events_df, glucose_unit,
     interface_language, last_visited_page, randomization_initialized,
     is_example_data, data_source_name, session_active).
    """
    payload = resume_store.load_session(code)
    if not payload:
        return None
    user_info = payload.get("user_info") or {}
    last_page = payload.get("last_visited_page") or "/prediction"
    return (
        last_page,
        user_info,
        payload.get("full_df"),
        payload.get("current_window_df"),
        payload.get("events_df"),
        payload.get("glucose_unit") or "mg/dL",
        normalize_locale(payload.get("interface_language")),
        last_page,
        True,   # randomization-initialized: data already chosen, don't re-roll
        bool(user_info.get("is_example_data", True)),
        str(user_info.get("data_source_name", "example.csv")),
        True,   # session-active
    )


_RESUME_RESTORE_OUTPUTS = [
    Output('url', 'pathname', allow_duplicate=True),
    Output('user-info-store', 'data', allow_duplicate=True),
    Output('full-df', 'data', allow_duplicate=True),
    Output('current-window-df', 'data', allow_duplicate=True),
    Output('events-df', 'data', allow_duplicate=True),
    Output('glucose-unit', 'data', allow_duplicate=True),
    Output('interface-language', 'data', allow_duplicate=True),
    Output('last-visited-page', 'data', allow_duplicate=True),
    Output('randomization-initialized', 'data', allow_duplicate=True),
    Output('is-example-data', 'data', allow_duplicate=True),
    Output('data-source-name', 'data', allow_duplicate=True),
    Output('session-active', 'data', allow_duplicate=True),
]


@app.callback(
    [Output('resume-redeem-done', 'data'),
     *_RESUME_RESTORE_OUTPUTS],
    [Input('url', 'search')],
    [State('resume-redeem-done', 'data')],
    prevent_initial_call='initial_duplicate',
)
def redeem_resume_from_url(search: Optional[str], done: Optional[bool]) -> tuple:
    """Restore a session from a ``?resume=<code>`` URL (universal cross-device entry).

    Runs on the initial load too (``initial_duplicate``) so a fresh device opening
    ``https://.../?resume=CODE`` restores immediately. The one-shot
    ``resume-redeem-done`` guard (read via State) stops it re-firing. We route via
    ``url.pathname`` (a different property from the ``url.search`` Input, so there
    is no self-cycle); a clientside callback strips the ``?resume=`` query.
    """
    if done:
        raise PreventUpdate
    code: Optional[str] = None
    if search:
        from urllib.parse import parse_qs
        code = (parse_qs(search.lstrip("?")).get("resume") or [None])[0]
    if not code:
        raise PreventUpdate
    restored = _restore_outputs_from_code(code)
    if restored is None:
        # Invalid/expired code: mark done, leave stores alone.
        return (True, *([no_update] * len(_RESUME_RESTORE_OUTPUTS)))
    with start_action(action_type=u"redeem_resume_from_url", code=str(code)):
        pass
    return (True, *restored)


# Strip the ?resume=<code> query from the URL after a successful redeem so the
# transfer token doesn't linger in the address bar / browser history.
app.clientside_callback(
    """
    function(done) {
        if (done && window.history && window.location.search.indexOf('resume=') !== -1) {
            window.history.replaceState({}, '', window.location.pathname);
        }
        return '';
    }
    """,
    Output('resume-clean-sink', 'children'),
    [Input('resume-redeem-done', 'data')],
    prevent_initial_call=True,
)


@app.callback(
    [Output('resume-redeem-error', 'children'),
     *_RESUME_RESTORE_OUTPUTS],
    [Input('resume-redeem-btn', 'n_clicks')],
    [State('resume-redeem-input', 'value'),
     State('interface-language', 'data')],
    prevent_initial_call=True,
)
def redeem_resume_from_input(
    n_clicks: Optional[int],
    code: Optional[str],
    interface_language: Optional[str],
) -> tuple:
    """Restore a session from a code typed into the landing-page resume box."""
    if not n_clicks:
        raise PreventUpdate
    locale = normalize_locale(interface_language)
    restored = _restore_outputs_from_code(code)
    if restored is None:
        return (
            t("ui.resume_code.not_found", locale=locale),
            *([no_update] * len(_RESUME_RESTORE_OUTPUTS)),
        )
    with start_action(action_type=u"redeem_resume_from_input", code=str(code)):
        pass
    return ("", *restored)


## Removed URL-based data writer callback to enforce single-writer for data stores

# Data initialization callback (URL-based only)
@app.callback(
    [
        Output('full-df', 'data', allow_duplicate=True),
        Output('current-window-df', 'data', allow_duplicate=True),
        Output('events-df', 'data', allow_duplicate=True),
        Output('is-example-data', 'data', allow_duplicate=True),
        Output('data-source-name', 'data', allow_duplicate=True),
        Output('randomization-initialized', 'data', allow_duplicate=True),
        Output('initial-slider-value', 'data', allow_duplicate=True),
        Output('ai-prediction-store', 'data'),  # <-- 1. Adăugat corect în lista de Outputs
    ],
    [Input('url', 'pathname')],
    [
        State('full-df', 'data'),
        State('user-info-store', 'data'),
        State('session-store', 'data'),          # <-- 2. Adăugat corect în lista de States
    ],
    prevent_initial_call=True,
)
def initialize_data_on_url_change(
    pathname: Optional[str],
    full_df_data: Optional[Dict],
    user_info: Optional[Dict[str, Any]],
    session_store_data: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    bool,
    str,
    bool,
    int,
    Dict[str, Any],
]:
    """Initialize data when URL changes to /prediction without existing data.

    Only loads fresh example data when navigating to /prediction and no data
    exists yet.  All other pages are left alone so that persisted localStorage
    stores are never overwritten (critical for the resume flow).

    Also the 1st of the 3 places where a fresh window is generated: as soon as
    the round starts, we kick off the GluMind forecast (if the user picked
    "Human vs AI") so it's ready in ``ai-prediction-store`` by the time they
    submit, instead of waiting until submit to compute it.
    """
    _no_ai_default = {"glumind": {"predictions": [], "ready": False}}
    _no_change = (no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    if pathname != '/prediction':
        return _no_change

    # For format B/C: require upload, don't auto-load example dataset.
    fmt = str((user_info or {}).get("format") or "A")
    uploaded_path = (user_info or {}).get("uploaded_data_path")
    if fmt in ("B", "C") and not uploaded_path:
        return None, None, None, False, "", False, 0, _no_ai_default

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

    window_dict = convert_df_to_dict(df)
    ai_forecast_data = _compute_ai_forecast_if_needed(window_dict, user_info)

    return (
        convert_df_to_dict(full_df),
        window_dict,
        convert_events_df_to_dict(events_df),
        True,
        'example.csv',
        False,
        random_start,
        ai_forecast_data,
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
     Output('consent-scroll-request', 'data'),
     Output('ai-prediction-store', 'data', allow_duplicate=True)],
    [Input('upload-data', 'contents'),
     Input('prediction-data-usage-consent', 'value')],
    [State('upload-data', 'filename'),
     State('user-info-store', 'data'),
     State('session-store', 'data')],
    prevent_initial_call=True,
)
def handle_file_upload(
    upload_contents: Optional[str],
    consent_value: Optional[list[str]],
    filename: Optional[str],
    user_info: Optional[Dict[str, Any]],
    session_store_data: Optional[Dict[str, Any]],
) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, Dict[str, Any], int, Dict[str, Any]]:
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
                    no_update,
                )

            # Process cached upload (browser may not re-fire upload for same file).
            upload_contents = str(pending)
            filename = str(info_pre.get("pending_upload_filename") or filename or "")

        if not upload_contents:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
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

        # 2nd of the 3 places a fresh window is generated: kick off GluMind
        # here too, so it's ready by the time the user submits this round.
        new_window_dict = convert_df_to_dict(new_df)
        ai_forecast_data = _compute_ai_forecast_if_needed(new_window_dict, info)

        return (
            current_time,
            convert_df_to_dict(new_full_df),
            new_window_dict,
            convert_events_df_to_dict(new_events_df),
            False,  # is_example_data = False for uploaded files
            str(filename or ""),  # store the original filename
            False,  # reset randomization flag for new data
            random_start,  # Update initial slider value
            info,
            current_time if triggered == "prediction-data-usage-consent" else no_update,
            ai_forecast_data,
        )


# Nightscout data load callback
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('nightscout-status', 'children')],
    [Input('nightscout-load-button', 'n_clicks')],
    [State('nightscout-url-input', 'value'),
     State('nightscout-token-input', 'value'),
     State('user-info-store', 'data'),
     State('prediction-data-usage-consent', 'value')],
    prevent_initial_call=True,
)
def handle_nightscout_load(
    n_clicks: Optional[int],
    nightscout_url: Optional[str],
    nightscout_token: Optional[str],
    user_info: Optional[Dict[str, Any]],
    consent_value: Optional[list[str]],
) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, Dict[str, Any], Any]:
    """Load CGM data from a Nightscout server URL."""
    if not n_clicks:
        raise PreventUpdate

    _no = (no_update,) * 9

    info_pre: Dict[str, Any] = dict(user_info or {})
    fmt = str(info_pre.get("format") or "A")
    locale = normalize_locale(info_pre.get("interface_language"))

    def _error(msg: str) -> Any:
        return html.Div(msg, style={
            'color': '#7f1d1d',
            'backgroundColor': '#fee2e2',
            'padding': '8px 10px',
            'borderRadius': '4px',
            'marginTop': '6px',
        })

    if not nightscout_url or not nightscout_url.strip():
        return _no + (_error(t("ui.header.nightscout_url_required", locale=locale)),)

    consent_ok = bool(info_pre.get("consent_use_uploaded_data", False)) or bool(consent_value and "agree" in consent_value)
    if fmt in ("B", "C") and not consent_ok:
        return _no + (_error(t("ui.header.nightscout_consent_required", locale=locale)),)

    with start_action(action_type=u"handle_nightscout_load", url=nightscout_url.strip()):
        users_data_dir = project_root / 'data' / 'input' / 'users'
        try:
            new_full_df, new_events_df, save_path = load_glucose_data_from_nightscout(
                nightscout_url.strip(),
                token=nightscout_token or None,
                save_dir=users_data_dir,
            )
        except Exception as exc:
            return _no + (_error(t("ui.header.nightscout_error", locale=locale, error=str(exc))),)

        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        new_df, random_start = get_random_data_window(new_full_df, points)

        ns_label = nightscout_url.strip().rstrip('/')
        current_time = int(time.time() * 1000)

        info: Dict[str, Any] = dict(info_pre)
        info["uploaded_data_path"] = str(save_path)
        info["uploaded_data_filename"] = ns_label
        info["is_example_data"] = False
        info["data_source_name"] = ns_label
        info["nightscout_url"] = ns_label
        if nightscout_token:
            info["nightscout_token"] = nightscout_token
        info["blocked_upload_requires_consent"] = False
        info.pop("pending_upload_contents", None)
        info.pop("pending_upload_filename", None)

        count = len(new_full_df)
        success_div = html.Div(
            [
                html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
                t("ui.header.nightscout_success", locale=locale, count=count),
            ],
            style={
                'color': '#2f855a',
                'backgroundColor': '#c6f6d5',
                'padding': '10px',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginTop': '6px',
            },
        )

        return (
            current_time,
            convert_df_to_dict(new_full_df),
            convert_df_to_dict(new_df),
            convert_events_df_to_dict(new_events_df),
            False,
            ns_label,
            False,
            random_start,
            info,
            success_div,
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
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('glucose-graph-graph', 'clickData'),
     Input('glucose-graph-graph', 'relayoutData')],
    [State('last-click-time', 'data'),
     State('current-window-df', 'data'),
     State('glucose-unit', 'data')],
    prevent_initial_call=True
)
def handle_graph_interactions(click_data: Optional[Dict], relayout_data: Optional[Dict],
                            last_click_time: int,
                            current_df_data: Optional[Dict], glucose_unit: Optional[str]) -> Tuple[int, Dict[str, List[Any]]]:
    """Handle glucose graph click and draw interactions.

    PERFORMANCE: predictions are a property of the CURRENT WINDOW only, so this
    hot-path callback (fires on every click / drawline stroke) updates ONLY
    `current-window-df` (~tens of rows). It deliberately does NOT touch
    `full-df`. For an uploaded multi-month CGM export full-df is tens of
    thousands of rows; reconstructing it from JSON and re-serialising it back on
    every stroke (the old behaviour) made Plotly resolve each drawn line after a
    long lag -- the reported "background hog". This is safe because full-df's
    prediction column is never consumed: save_statistics derives predictions
    from `prediction_table_data` (built from the window) and uses full-df only
    for window times + age/user_id; the chart figure renders from
    current-window-df; and the window is re-sliced from full-df only by the
    hidden, round-start-only time slider (when predictions are legitimately 0).
    """
    if not current_df_data:
        return no_update, no_update

    unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'

    def to_mgdl(y_value: float) -> float:
        if unit == 'mmol/L':
            return float(y_value) * GLUCOSE_MGDL_PER_MMOLL
        return float(y_value)

    current_time = int(time.time() * 1000)
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
            df = df.with_columns(pl.lit(0.0).alias("prediction"))
            return current_time, convert_df_to_dict(df)

        point_data = click_data['points'][0]
        click_x = point_data['x']
        click_y = point_data['y']
        snapped_idx = snap_index(float(click_x))
        if snapped_idx is None:
            return no_update, no_update
        nearest_time = df.get_column("time")[snapped_idx]

        # Check if this is the first prediction point at the boundary - snap to ground truth
        prediction_y = to_mgdl(float(click_y))
        if snapped_idx == visible_points:  # First point in hidden area
            # Check if this is the start of a new prediction sequence
            existing_predictions = df.filter(pl.col("prediction") != 0.0).height
            if existing_predictions == 0:  # No existing predictions, snap to ground truth
                ground_truth_y = df.get_column("gl")[snapped_idx]
                prediction_y = ground_truth_y

        df = df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(prediction_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )

        return current_time, convert_df_to_dict(df)

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
                    return last_click_time, convert_df_to_dict(df)

                start_time = df.get_column("time")[start_idx]

                # Check if this is the first prediction starting at the boundary - snap to ground truth
                actual_start_y = to_mgdl(float(start_y))
                if start_idx == visible_points:  # Starting at first point in hidden area
                    # Check if this is the start of a new prediction sequence
                    existing_predictions = df.filter(pl.col("prediction") != 0.0).height
                    if existing_predictions == 0:  # No existing predictions, snap to ground truth
                        ground_truth_y = df.get_column("gl")[start_idx]
                        actual_start_y = ground_truth_y

                # Use the full extent of the drawn line (end_idx already snapped above)
                actual_end_y = to_mgdl(float(end_y))
                end_time = df.get_column("time")[end_idx]

                # Get intermediate prediction points for every grid point along the line
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

                # Update the window DataFrame with all prediction points
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

                return current_time, convert_df_to_dict(df)

    return no_update, no_update

@app.callback(
    Output('data-source-display', 'children'),
    [Input('url', 'pathname'),
     Input('data-source-name', 'data'),
     Input('user-info-store', 'data'),
     Input('interface-language', 'data')],
    prevent_initial_call=False
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

    age_display = (
        str(meta.age)
        .replace("years old", "")
        .replace("year old", "")
        .strip()
    )
    weight_display = str(meta.weight).replace(" ", "")
    if locale == "en":
        return f"{age_display} yr old {gender_display}, weight {weight_display}"

    return (
        f"{age_display} · {gender_display} · "
        f"{t('ui.header.weight_label', locale=locale)} {weight_display}"
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


def create_intermediate_predictions(start_time: datetime, end_time: datetime, start_y: float, end_y: float, df: pl.DataFrame) -> List[Tuple[datetime, float]]:
    """
    Create linearly-interpolated prediction points for every dataframe row
    between start_time and end_time (exclusive of both endpoints).
    """
    available_times = (df
        .filter((pl.col("time") > start_time) & (pl.col("time") < end_time))
        .get_column("time")
        .to_list()
    )
    total_seconds = (end_time - start_time).total_seconds()
    if total_seconds == 0:
        return []
    return [
        (t, start_y + (end_y - start_y) * ((t - start_time).total_seconds() / total_seconds))
        for t in available_times
    ]


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
    global startup_page, landing_page, _callbacks_registered
    if _callbacks_registered:
        return
    landing_page = LandingPage()
    startup_page = StartupPage()

    prediction_table.register_callbacks(app)
    metrics_component.register_callbacks(app, prediction_table)
    glucose_chart.register_callbacks(app)
    submit_component.register_callbacks(app)
    landing_page.register_callbacks(app)
    startup_page.register_callbacks(app)
    ending_page.register_callbacks(app)
    _callbacks_registered = True


def bootstrap_wsgi_application() -> Any:
    """Prepare callbacks and initial layout for WSGI servers.

    Each gunicorn worker imports this module, so provision Chrome here too:
    the parent ``serve`` process may have downloaded it already (shared user
    cache), but doing it per worker makes share-card export robust regardless
    of how the app is launched.
    """
    _register_all_callbacks()
    _ensure_chrome()
    return server


def _ensure_chrome() -> None:
    """Ensure a Chromium browser is available for kaleido image export.

    Checks choreographer's browser search first; if nothing is found,
    downloads Chrome for Testing via ``kaleido.get_chrome_sync()``. The
    downloaded binary is self-contained, but Chromium still links system
    shared libraries (libatk-1.0.so.0, libnss3, libgbm1 …) — on a slim/bare
    host those must be installed or Chrome dies on launch with
    ``BrowserFailedError`` (see README for the apt lib set). The actual launch
    failure is surfaced loudly by the render path in ``share_png.py``.
    """
    from choreographer.browsers.chromium import (
        get_chrome_download_path,
        get_old_chrome_download_path,
    )
    with start_action(action_type="ensure_chrome") as action:
        # Ensure the *managed* download exists rather than trusting
        # find_browser(): a present-but-broken system/snap chromium would
        # otherwise satisfy find_browser, win over the managed binary kaleido
        # prefers, and crash on launch (BrowserFailedError).
        managed = get_chrome_download_path(mkdir=False)
        old = get_old_chrome_download_path()
        have_managed: bool = bool((managed and managed.exists()) or (old and old.exists()))
        if not have_managed:
            import kaleido
            with start_action(action_type="ensure_chrome_download"):
                kaleido.get_chrome_sync()
            managed = get_chrome_download_path(mkdir=False)
        action.log(message_type="chrome_resolved", path=str(managed))


import socket as _socket


def _find_free_port(host: str, preferred: int, max_tries: int = 20) -> int:
    """Return *preferred* if available, otherwise increment until a free port is found."""
    for offset in range(max_tries):
        candidate = preferred + offset
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
                return candidate
        except OSError:
            continue
    return preferred


# Create typer app.  invoke_without_command + the @cli.callback default
# mean ``uv run start`` (no subcommand) still works, while ``uv run chart``
# routes to the ``chart`` subcommand via its own entrypoint.
cli = typer.Typer(invoke_without_command=True)


def _arg_value(argv: list[str], *names: str) -> Optional[str]:
    """Return the value for a CLI option without fully invoking Typer."""
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name and index + 1 < len(argv):
                return argv[index + 1]
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix):]
    return None


def _arg_present(argv: list[str], *names: str) -> bool:
    """Check whether any CLI flag is present."""
    return any(arg in names for arg in argv)


def _seed_chart_env_from_argv(argv: list[str], env: Dict[str, str]) -> None:
    """Seed chart-mode env vars before the Dash app module is imported.

    Python console-script entry points import this module before Typer dispatches
    to ``chart()``.  Re-execing with these values already in the environment
    lets module-level layout/store initialization see chart mode immediately.
    """
    env["_CHART_MODE"] = "1"

    file_arg = _arg_value(argv, "--file", "-f")
    if file_arg:
        env["_CHART_FILE"] = file_arg
        env["_CHART_SOURCE"] = Path(file_arg).name
    else:
        env.pop("_CHART_FILE", None)
        env["_CHART_SOURCE"] = "example.csv"

    env["_CHART_POINTS"] = _arg_value(argv, "--points", "-p") or str(DEFAULT_POINTS)

    start_arg = _arg_value(argv, "--start", "-s")
    if start_arg is not None:
        env["_CHART_START"] = start_arg
    else:
        env.pop("_CHART_START", None)

    unit_arg = _arg_value(argv, "--unit", "-u")
    env["_CHART_UNIT"] = unit_arg if unit_arg in ("mg/dL", "mmol/L") else "mg/dL"
    env["_CHART_LOCALE"] = normalize_locale(_arg_value(argv, "--locale", "-l") or "en")

    if _arg_present(argv, "--prefill"):
        env["_CHART_PREFILL"] = "1"
        env["_CHART_NOISE"] = _arg_value(argv, "--noise") or "0.05"
    else:
        env.pop("_CHART_PREFILL", None)
        env.pop("_CHART_NOISE", None)

    if _arg_present(argv, "--clean"):
        env["_CLEAN_STORAGE"] = "1"

    if _arg_present(argv, "--debug"):
        env["DASH_DEBUG"] = "1"
        env["DEBUG_MODE"] = "1"
    if _arg_present(argv, "--no-debug"):
        env["DASH_DEBUG"] = "0"
        env["DEBUG_MODE"] = "0"


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

    _ensure_chrome()

    dash_host = DASH_HOST if host is None else (host or DASH_HOST)
    dash_port = _find_free_port(dash_host, DASH_PORT if port is None else port)
    dash_debug = DASH_DEBUG if debug is None else debug
    if debug is not None:
        sugar_sugar_config.DEBUG_MODE = debug

    _register_all_callbacks()

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
    clean: bool = typer.Option(False, "--clean", help="Clear browser localStorage on first connect so the session starts fresh"),
    debug: Optional[bool] = typer.Option(None, "--debug/--no-debug", help="Override Dash debug mode for this chart server"),
    reloader: bool = typer.Option(False, "--reloader/--no-reloader", help="Enable Werkzeug's debug reloader. Disabled by default so chart-mode stores are deterministic."),
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

    if clean:
        os.environ["_CLEAN_STORAGE"] = "1"
        for child in app.layout.children:
            if getattr(child, 'id', None) == 'clean-storage-flag':
                child.data = True
                break

    dash_debug = (os.getenv("DASH_DEBUG", "").lower() not in ("0", "false", "no")) if debug is None else debug
    sugar_sugar_config.DEBUG_MODE = dash_debug

    _ensure_chrome()
    _register_all_callbacks()

    dash_host = DASH_HOST if host is None else (host or DASH_HOST)
    dash_port = _find_free_port(dash_host, DASH_PORT if port is None else port)

    with start_action(
        action_type=u"start_chart_dev",
        file=str(file) if file else "example.csv",
        points=points,
        prefill=prefill,
        host=dash_host,
        port=dash_port,
        debug=dash_debug,
        reloader=reloader,
    ):
        app.run(host=dash_host, port=dash_port, debug=dash_debug, use_reloader=dash_debug and reloader)


@cli.command()
def share(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="CSV file to load. Default: built-in example."),
    rounds: int = typer.Option(SHARE_ROUNDS, "--rounds", "-r", help="Number of fake rounds to generate"),
    formats: str = typer.Option(SHARE_FORMATS, "--formats", help="Comma-separated format letters to cycle through (e.g. 'A,B,C')"),
    noise: float = typer.Option(SHARE_NOISE, "--noise", help="Max noise at last prediction step (fraction, e.g. 0.30 = +/-30%%)"),
    points: int = typer.Option(DEFAULT_POINTS, "--points", "-p", help="Number of data points per window"),
    locale: str = typer.Option("en", "--locale", "-l", help="UI locale (en, de, uk, ro)"),
    name: str = typer.Option(SHARE_NAME, "--name", "-n", help="Player name shown on the share card"),
    host: Optional[str] = typer.Option(None, "--host", help="Host to run the server on"),
    port: Optional[int] = typer.Option(None, "--port", help="Port to run the server on"),
) -> None:
    """Dev shortcut: generate fake multi-round data and open the share page.

    Bypasses the entire game flow.  Generates N rounds of noisy predictions
    from the example data (or a custom CSV), saves a share record to disk,
    and starts Dash at /share/<id> so you can iterate on the share page
    layout, card rendering, and social-sharing flow.
    """
    os.environ["_SHARE_MODE"] = "1"
    os.environ["_SHARE_ROUNDS"] = str(max(1, rounds))
    os.environ["_SHARE_FORMATS"] = formats
    os.environ["_SHARE_NOISE"] = str(noise)
    os.environ["_SHARE_LOCALE"] = normalize_locale(locale)
    os.environ["_SHARE_NAME"] = name
    os.environ["_SHARE_SOURCE"] = file.name if file else "example.csv"
    os.environ["_CHART_POINTS"] = str(points)
    if file:
        os.environ["_CHART_FILE"] = str(file)

    sugar_sugar_config.DEBUG_MODE = True

    _ensure_chrome()
    _register_all_callbacks()

    dash_host = DASH_HOST if host is None else (host or DASH_HOST)
    dash_port = _find_free_port(dash_host, DASH_PORT if port is None else port)

    with start_action(
        action_type=u"start_share_dev",
        rounds=rounds,
        formats=formats,
        noise=noise,
        host=dash_host,
        port=dash_port,
    ):
        app.run(host=dash_host, port=dash_port, debug=True)


@cli.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Host gunicorn should bind"),
    port: Optional[int] = typer.Option(None, "--port", help="Port gunicorn should bind"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Gunicorn worker count"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Gunicorn worker timeout in seconds"),
    staging: bool = typer.Option(False, "--staging", help="Enable prod+ staging test routes under /staging/*"),
) -> None:
    """Run the Dash app with gunicorn for production/staging deployments."""
    if staging:
        # Set before exec so every gunicorn worker re-reads it at import.
        os.environ["_STAGING_MODE"] = "1"
    _ensure_chrome()
    bind_host: str = DASH_HOST if host is None else (host or DASH_HOST)
    bind_port: int = DASH_PORT if port is None else port
    worker_count: int = workers if workers is not None else int(os.getenv("WEB_CONCURRENCY", os.getenv("GUNICORN_WORKERS", "2")))
    worker_timeout: int = timeout if timeout is not None else int(os.getenv("GUNICORN_TIMEOUT", "120"))
    bind: str = f"{bind_host}:{bind_port}"
    command: list[str] = [
        "gunicorn",
        "sugar_sugar.wsgi:application",
        "--bind",
        bind,
        "--workers",
        str(worker_count),
        "--timeout",
        str(worker_timeout),
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "--forwarded-allow-ips",
        os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "*"),
    ]
    with start_action(
        action_type=u"serve_gunicorn",
        host=bind_host,
        port=bind_port,
        workers=worker_count,
        timeout=worker_timeout,
    ):
        os.execvp(command[0], command)


def cli_main() -> None:
    """CLI entry point"""
    cli()


def chart_main() -> None:
    """CLI entry point that defaults to the ``chart`` command."""
    argv = sys.argv[1:]
    if os.environ.get("_CHART_REEXECED") != "1":
        env = {**os.environ}
        _seed_chart_env_from_argv(argv, env)
        env["_CHART_REEXECED"] = "1"
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "sugar_sugar.app", "chart", *argv],
            env,
        )
    cli(["chart"] + argv)


def share_main() -> None:
    """CLI entry point that defaults to the ``share`` command."""
    cli(["share"] + sys.argv[1:])


def serve_main() -> None:
    """CLI entry point that defaults to the ``serve`` command."""
    typer.run(serve)


def serve_staging_main() -> None:
    """CLI entry point: ``serve`` with the staging test routes enabled.

    Equivalent to ``uv run serve --staging`` but available as its own command so
    the staging deployment (https://vanilla-sugar.glucosedao.org/) can run the
    dev branch with prod+ test nodes without remembering the flag.
    """
    os.environ["_STAGING_MODE"] = "1"
    if "--staging" not in sys.argv:
        sys.argv.append("--staging")
    typer.run(serve)


def setup_chrome_main() -> None:
    """Download Chrome for Testing if no Chromium browser is available."""
    _ensure_chrome()
    from choreographer.browsers.chromium import Chromium
    path = Chromium.find_browser(skip_local=False)
    print(f"Chrome ready: {path}")


if __name__ == '__main__':
    cli()