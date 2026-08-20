from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
from html import escape as html_escape
from io import BytesIO
import dash
from dash import ALL, dcc, html, Output, Input, State, no_update, ctx
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
from flask import Response, has_request_context, send_file as flask_send_file, request as flask_request
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

from sugar_sugar.data import load_glucose_data, load_glucose_data_from_nightscout, decode_upload_bytes
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
from sugar_sugar.nickname import (
    MAX_NICKNAME_LENGTH,
    email_key,
    identity_key,
    normalize_nickname,
)
from sugar_sugar.consent import (
    apply_optional_consent_choices,
    reconcile_stored_consents,
    should_persist_study_data,
    upsert_consent_agreement_fields,
)
from sugar_sugar.components.glucose import (
    FOOD_COMPOSITE_MAX,
    GlucoseChart,
    meal_food_bubble_children,
)
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.components.predictions import PredictionTableComponent
from sugar_sugar.components.ag_grid import build_readonly_ag_grid, build_readonly_column_defs
from sugar_sugar.components.startup import StartupPage, StartupPageMobile
from sugar_sugar.components.landing import LandingPage, LandingPageMobile
from sugar_sugar.components.consent_form import ConsentFormPage
from sugar_sugar.components.submit import (
    FINISH_EXIT_BUTTON_CLASS,
    WINDOWS_CLOSE_RED,
    SubmitComponent,
    finish_confirm_message,
    finish_confirm_overlay,
    finish_exit_button_style,
    hidden_area_is_complete,
)
from sugar_sugar.encouragement import pick_bracket
from sugar_sugar.components.header import HeaderComponent, make_csv_upload
from sugar_sugar.components.ending import EndingPage
from sugar_sugar.components.navbar import NavBar, MobileNavBar
from sugar_sugar.components.share import (
    build_share_card_figure,
    build_share_panel,
    build_synthesis_card,
    collect_playable_rounds,
    create_expired_layout,
    create_share_layout,
)
from sugar_sugar import share_store
from sugar_sugar import resume_store
from sugar_sugar.generic_sources_metadata import (
    GenericSourceMetadata,
    format_generic_source_metadata,
    format_participant_demographics,
    format_source_notes,
    load_generic_sources_metadata,
    resolve_source_metadata,
)
from sugar_sugar.prediction_window_context import should_show_no_carbs_note
from sugar_sugar.subject_sources import (
    collect_generic_round_history,
    generic_round_window_from_df,
    generic_window_slice_key,
    load_random_generic_dataset,
    generic_intervention_for_user,
    pick_unique_generic_window,
    resolve_generic_source_path,
    window_is_continuous,
)
from sugar_sugar.contact_info import load_contact_info
from sugar_sugar.static_markdown import static_markdown_autosize_iframe

# Type aliases for clarity
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type

GLUCOSE_MGDL_PER_MMOLL: float = 18.0

FORMAT_ORDER: dict[str, int] = {"C": 0, "B": 1, "A": 2}
GENERIC_SOURCES_METADATA = load_generic_sources_metadata()


def _build_source_metadata_line(
    *,
    source_name: str,
    user_info: Optional[Dict[str, Any]],
    is_example_data: bool,
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
    locale: str,
) -> str:
    key = Path(str(source_name or "example.csv")).name
    meta = GENERIC_SOURCES_METADATA.get(key) or resolve_source_metadata(key)
    show_no_carbs = should_show_no_carbs_note(window_df, events_df)
    show_carbs_info = not is_example_data

    if meta is not None:
        return format_generic_source_metadata(
            meta,
            locale=locale,
            show_no_carbs_note=show_no_carbs,
            show_carbs_info_note=show_carbs_info,
        )

    if not is_example_data and user_info and user_info.get("age"):
        diabetic_raw = user_info.get("diabetic")
        diabetic = diabetic_raw if isinstance(diabetic_raw, bool) else None
        return format_participant_demographics(
            user_info["age"],
            str(user_info.get("gender") or ""),
            locale=locale,
            weight=str(user_info.get("weight") or ""),
            diabetic=diabetic,
            show_no_carbs_note=show_no_carbs,
            show_carbs_info_note=True,
        )

    if show_carbs_info or show_no_carbs:
        return format_source_notes(
            locale=locale,
            show_no_carbs_note=show_no_carbs,
            show_carbs_info_note=show_carbs_info,
        )
    return ""


def _source_window_time_range(window_df: pl.DataFrame) -> str:
    if window_df.is_empty() or "time" not in window_df.columns:
        return ""
    times = window_df.get_column("time")
    start = times[0]
    end = times[-1]

    def _hhmm(value: Any) -> str:
        if hasattr(value, "strftime"):
            return value.strftime("%H:%M")
        return datetime.fromisoformat(str(value)).strftime("%H:%M")

    return f"{_hhmm(start)}-{_hhmm(end)}"


def _source_plaque_label(locale: str) -> str:
    """Short plaque label. `current_data_source` is the long form used elsewhere."""
    return t("ui.header.source_short", locale=locale)


def _ending_source_plaque_children(
    *,
    user_info: Optional[Dict[str, Any]],
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
    locale: str,
) -> list[Any]:
    """Same plaque as /prediction: bold metadata, then Source + file + time."""
    data_source_name = Path(str((user_info or {}).get("data_source_name") or "")).name
    is_example = bool((user_info or {}).get("is_example_data", True))
    metadata_line = _build_source_metadata_line(
        source_name=data_source_name,
        user_info=user_info,
        is_example_data=is_example,
        window_df=window_df,
        events_df=events_df,
        locale=locale,
    )
    return [
        html.Div(
            metadata_line,
            id="ending-source-metadata",
            className="prediction-source-metadata",
            disable_n_clicks=True,
        ),
        html.Div(
            [
                html.Label(
                    _source_plaque_label(locale),
                    id="ending-source-label",
                    className="prediction-source-label",
                ),
                html.Div(
                    data_source_name,
                    id="ending-source-name",
                    className="prediction-source-name",
                    disable_n_clicks=True,
                ),
                html.Div(
                    _source_window_time_range(window_df),
                    id="ending-source-time",
                    className="prediction-source-time",
                    disable_n_clicks=True,
                ),
            ],
            className="prediction-source-line",
            disable_n_clicks=True,
        ),
    ]


def _load_generic_round_window(
    points: int,
    rounds: list[dict[str, Any]] | None = None,
    user_info: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, str, int, str, str, str, str]:
    history = collect_generic_round_history(rounds, user_info)
    intervention = generic_intervention_for_user(user_info)
    if user_info is not None:
        user_info["generic_intervention"] = intervention
    selection = pick_unique_generic_window(points, history, intervention=intervention)
    round_window = generic_round_window_from_df(
        selection.window_df,
        source_name=selection.source.source_name,
    )
    return (
        selection.window_df,
        selection.events_df,
        selection.source.source_name,
        selection.start_index,
        selection.slice_key,
        round_window.window_start.isoformat(sep=" "),
        round_window.window_end.isoformat(sep=" "),
        round_window.anchor_time.isoformat(sep=" "),
    )


def _store_current_generic_window(
    user_info: Dict[str, Any],
    *,
    source_name: str,
    slice_key: str,
    window_start: str,
    window_end: str,
    anchor_time: str,
) -> None:
    user_info['current_generic_slice_key'] = slice_key
    user_info['current_generic_window_start'] = window_start
    user_info['current_generic_window_end'] = window_end
    user_info['current_generic_anchor_time'] = anchor_time
    user_info['data_source_name'] = source_name


def _apply_generic_round_selection(
    user_info: Dict[str, Any],
    rounds: list[dict[str, Any]] | None,
    points: int,
) -> tuple[pl.DataFrame, pl.DataFrame, str, int]:
    new_df, events_df, source_name, random_start, slice_key, window_start, window_end, anchor_time = (
        _load_generic_round_window(points, rounds, user_info)
    )
    _store_current_generic_window(
        user_info,
        source_name=source_name,
        slice_key=slice_key,
        window_start=window_start,
        window_end=window_end,
        anchor_time=anchor_time,
    )
    return new_df, events_df, source_name, random_start


def _upload_data_consent_given(user_info: Optional[Dict[str, Any]]) -> bool:
    """True once upload/CGM-data usage consent is known for this session."""
    from sugar_sugar.components.startup import prior_upload_data_consent

    return prior_upload_data_consent(user_info)


def _show_prediction_upload_consent(user_info: Optional[Dict[str, Any]], *, show_upload: bool) -> bool:
    """Show prediction consent only for B/C when consent was never given anywhere."""
    if not show_upload:
        return False
    # Consent form, startup checkbox, prior B/C rounds, or an existing upload —
    # any of these means do not ask again (including after C→B format switches).
    return not _upload_data_consent_given(user_info)


SITE_TITLE: str = "Sugar-Sugar"
SITE_DESCRIPTION: str = (
    "Test your glucose prediction skills, compare your forecasts with real CGM data, "
    "and help establish a human baseline for glucose forecasting research."
)
OG_PREVIEW_PATH: str = "/assets/og-card.png"
OG_PREVIEW_VERSION: int = 1
OG_PREVIEW_SIZE: tuple[int, int] = (1200, 630)
# Bump when the share-card PNG design changes so FB/X/LinkedIn re-fetch the
# image instead of serving a stale crop from their own caches.
SHARE_CARD_IMAGE_VERSION: int = 5
PUBLIC_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/", "Sugar-Sugar", SITE_DESCRIPTION),
    ("/about", "About Sugar-Sugar", "Learn why the Sugar-Sugar glucose prediction study matters."),
    ("/faq", "Sugar-Sugar FAQ", "Answers to common questions about the Sugar-Sugar study and gameplay."),
    ("/demo", "Video Instructions", "Watch how to play the Sugar-Sugar glucose prediction game."),
    ("/contact", "Contact GlucoseDAO", "Get in touch with the Sugar-Sugar team."),
    (
        "/highscore",
        "Sugar-Sugar Highscore",
        "Anonymous leaderboard of human glucose prediction accuracy, ranked by mean absolute error.",
    ),
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


def _identity_expr() -> pl.Expr:
    """Dataframe form of :func:`sugar_sugar.nickname.identity_key`.

    Single definition so the arcade board (`_ranking_entries`) and the per-player
    aggregation (`_ranking_identities`) can never drift apart on what counts as
    "the same person".
    """
    return (
        pl.when(pl.col('email_key') != "")
        .then(pl.format("e:{}", pl.col('email_key')))
        .otherwise(pl.format("s:{}", pl.col('study_id')))
        .alias('identity')
    )


def _ranking_entries(
    ranking_path: Path,
    *,
    format_filter: Optional[str],
) -> Optional[pl.DataFrame]:
    """Every finished game in a ranking CSV as its own board slot, best-first.

    Arcade rules: a score you set keeps its place forever.  Nothing is merged or
    collapsed, so a player who finished several games occupies several slots and an
    earlier, worse score is still visible below a later, better one.  Runs with
    fewer than ``MIN_USEFUL_ROUNDS`` (default 6) are excluded -- a short lucky
    MAE must not outrank a full attempt.  Completers stay unranked on ``/final``
    and the share card until they reach that floor.

    Returns ``None`` when the CSV is missing, unreadable or has nothing rankable.
    Otherwise: ``identity``, ``study_id``, ``email_key``, ``mae``, ``rounds``,
    ``nickname``, ``rank_idx`` (0-based).

    Ties break on the earlier timestamp, so whoever got there first sits higher.
    Each slot keeps the nickname stored on its own row -- the name that score was
    set under, not whatever the player is called now.
    """
    if not ranking_path.exists():
        return None
    try:
        ranking_df = pl.read_csv(ranking_path)
    except Exception:
        return None
    if 'study_id' not in ranking_df.columns or 'overall_mae_mgdl' not in ranking_df.columns:
        return None

    has_format = 'format' in ranking_df.columns

    def _optional_text(column: str) -> pl.Expr:
        """The column as text, or an empty literal for pre-nickname CSV schemas."""
        source = pl.col(column) if column in ranking_df.columns else pl.lit("")
        return source.cast(pl.String, strict=False).fill_null("").alias(column)

    def _optional_int(column: str) -> pl.Expr:
        source = pl.col(column) if column in ranking_df.columns else pl.lit(None)
        return source.cast(pl.Int64, strict=False).alias(column)

    df = ranking_df.select(
        pl.col('study_id').cast(pl.String, strict=False).fill_null("").alias('study_id'),
        pl.col('overall_mae_mgdl').cast(pl.Float64, strict=False).alias('mae'),
        _optional_int('rounds_played').alias('rounds'),
        _optional_text('format'),
        _optional_text('email_key'),
        _optional_text('nickname'),
        _optional_text('timestamp'),
    ).filter(pl.col('mae').is_not_null())

    if format_filter and has_format:
        df = df.filter(pl.col('format') == format_filter)
    # Below the useful-round floor (exclusive): 1–5 round rows stay in the CSV
    # for history but are not ranked. Null rounds (pre-column schemas) stay.
    df = df.filter(
        pl.col('rounds').is_null() | (pl.col('rounds') >= MIN_USEFUL_ROUNDS)
    )
    if df.height == 0:
        return None

    df = df.with_columns(
        # Identity does not merge slots here; it only answers "is this row mine?" and
        # feeds the nickname suggestion, which is what carries a name across devices.
        _identity_expr(),
        pl.col('timestamp')
        .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False)
        .alias('_ts'),
    )

    return df.sort(['mae', '_ts', 'study_id'], nulls_last=True).with_row_index('rank_idx')


def _own_entries(
    entries: Optional[pl.DataFrame],
    *,
    study_id: str,
    key: str,
) -> Optional[pl.DataFrame]:
    """The slots belonging to the current player -- there may be several.

    Matches on the hashed email (so slots set on another device are recognised as
    yours) *or* on ``study_id``, which also covers rows written before the player
    supplied an email.
    """
    if entries is None or entries.height == 0:
        return None
    predicate: Optional[pl.Expr] = None
    if study_id:
        predicate = pl.col('study_id') == study_id
    if key:
        by_key = pl.col('email_key') == key
        predicate = by_key if predicate is None else (predicate | by_key)
    if predicate is None:
        return None
    mine = entries.filter(predicate)
    return mine if mine.height > 0 else None


def _ranking_identities(
    ranking_path: Path,
    *,
    format_filter: Optional[str],
    mode: str,
) -> Optional[pl.DataFrame]:
    """Collapse a ranking CSV to one row per *player*.

    NOT used by `/highscore` or `/final`: those boards are arcade-style, one slot per
    finished game (`_ranking_entries`).  This is the per-player rollup kept for the
    planned individual stats page -- "your best, across every device you played on".
    Covered by ``tests/test_ranking_identity.py`` so it cannot rot before then.

    Columns: ``identity``, ``mae``, ``rounds``, ``nickname``, ``games``,
    ``study_ids``.  Rows are grouped in two stages:

    1. Within one ``study_id``, ``mode`` picks the representative score --
       ``"latest"`` (newest row by timestamp; the overall CSV's rows are *cumulative*,
       so only the newest covers the whole play) or ``"best"`` (lowest MAE).
    2. Across the ``study_id``s sharing an ``identity``, the **best** of those wins.

    The reported nickname is the newest non-empty one anywhere in the identity, so a
    later blank run never erases a name the player chose.
    """
    entries = _ranking_entries(ranking_path, format_filter=format_filter)
    if entries is None:
        return None

    if mode == "latest":
        per_study = (
            entries.sort(['identity', 'study_id', '_ts'], nulls_last=False)
            .group_by(['identity', 'study_id'])
            .agg(
                pl.last('mae').alias('mae'),
                pl.last('rounds').alias('rounds'),
                pl.len().alias('games'),
            )
        )
    else:
        per_study = (
            entries.sort(['identity', 'study_id', 'mae'], nulls_last=True)
            .group_by(['identity', 'study_id'])
            .agg(
                pl.first('mae').alias('mae'),
                pl.first('rounds').alias('rounds'),
                pl.len().alias('games'),
            )
        )

    identities = (
        per_study.sort(['identity', 'mae'], nulls_last=True)
        .group_by('identity')
        .agg(
            pl.first('mae').alias('mae'),
            pl.first('rounds').alias('rounds'),
            pl.col('games').sum().alias('games'),
            pl.col('study_id').unique().alias('study_ids'),
        )
    )

    named = (
        entries.filter(pl.col('nickname') != "")
        .sort(['identity', '_ts'], nulls_last=False)
        .group_by('identity')
        .agg(pl.last('nickname').alias('nickname'))
    )
    identities = identities.join(named, on='identity', how='left').with_columns(
        pl.col('nickname').cast(pl.String, strict=False).fill_null("")
    )

    return identities.sort(['mae', 'identity'], nulls_last=True).with_row_index('rank_idx')


def _match_identity(
    identities: Optional[pl.DataFrame],
    *,
    study_id: str,
    key: str,
) -> Optional[dict[str, Any]]:
    """The aggregated row for one player in a :func:`_ranking_identities` frame.

    Companion to that helper, so likewise not wired into the boards yet.  Tries the
    derived identity first, then falls back to ``study_id`` membership so rows
    written before the player supplied an email still resolve to their owner.
    """
    if identities is None or identities.height == 0:
        return None
    if key or study_id:
        hit = identities.filter(pl.col('identity') == identity_key(key=key, study_id=study_id))
        if hit.height > 0:
            return hit.row(0, named=True)
    if study_id:
        hit = identities.filter(pl.col('study_ids').list.contains(study_id))
        if hit.height > 0:
            return hit.row(0, named=True)
    return None


def _rank_from_ranking_csv(
    ranking_path: Path,
    *,
    study_id: str,
    key: str = "",
    format_filter: Optional[str],
) -> Optional[tuple[int, int]]:
    """Return ``(best_rank, total_slots)`` for one player against the ranking CSV.

    Extracted from ``create_final_layout`` so the share page can compute and freeze
    rankings into a share record at save time.  ``total`` counts board slots (one
    per finished game); the rank is the player's *best* slot.
    """
    if not study_id and not key:
        return None
    entries = _ranking_entries(ranking_path, format_filter=format_filter)
    if entries is None:
        return None
    mine = _own_entries(entries, study_id=study_id, key=key)
    if mine is None:
        return None
    return int(mine.get_column('rank_idx').min()) + 1, entries.height


def _leaderboard_snapshot(
    ranking_path: Path,
    *,
    study_id: str,
    key: str = "",
    format_filter: Optional[str],
    mode: str = "",
    top_n: int = 5,
) -> Optional[dict[str, Any]]:
    """Build a compact leaderboard view for one player.

    Returns ``None`` when the CSV is missing/empty. Otherwise:
      ``{rank, total, players, mae, top: [{rank, mae, rounds, nickname, is_you}]}``

    ``total`` is the number of board slots and ``rank`` the player's best one;
    ``players`` counts distinct people, which is only used for the stat chips.
    ``nickname`` is ``""`` for slots set anonymously -- the caller falls back to the
    ``Player N`` label.  ``study_id`` and the hashed ``key`` are never returned.
    """
    entries = _ranking_entries(ranking_path, format_filter=format_filter)
    if entries is None:
        return None

    mine = _own_entries(entries, study_id=study_id, key=key)
    my_ranks: set[int] = (
        set(mine.get_column('rank_idx').to_list()) if mine is not None else set()
    )

    top: list[dict[str, Any]] = []
    for row in entries.head(max(1, top_n)).iter_rows(named=True):
        rounds = row["rounds"]
        top.append(
            {
                "rank": int(row["rank_idx"]) + 1,
                "mae": float(row["mae"]),
                "rounds": int(rounds) if rounds is not None else None,
                "nickname": str(row["nickname"] or ""),
                "is_you": int(row["rank_idx"]) in my_ranks,
            }
        )

    user_rank: Optional[int] = None
    user_mae: Optional[float] = None
    if mine is not None:
        best = mine.sort('rank_idx').row(0, named=True)
        user_rank = int(best['rank_idx']) + 1
        user_mae = float(best['mae'])
        if not any(entry["is_you"] for entry in top):
            top.append(
                {
                    "rank": user_rank,
                    "mae": user_mae,
                    "rounds": int(best['rounds']) if best['rounds'] is not None else None,
                    "nickname": str(best['nickname'] or ""),
                    "is_you": True,
                }
            )

    return {
        "rank": user_rank,
        "total": entries.height,
        "players": int(entries.get_column('identity').n_unique()),
        "mae": user_mae,
        "top": top,
    }


def stored_nickname(*, study_id: str, key: str) -> str:
    """The nickname this player last set, ``""`` when there is none.

    Used to pre-fill the `/final` box as a *suggestion*: a player returning on a new
    device (fresh localStorage, same email) sees the name they used last time instead
    of a blank field.  Existing board slots keep the name they were set under.
    """
    if not study_id and not key:
        return ""
    sources: list[tuple[Path, Optional[str]]] = [
        (project_root / 'data' / 'input' / 'prediction_ranking.csv', "ALL")
    ]
    sources.extend(
        (project_root / 'data' / 'input' / f'prediction_ranking_{fmt}.csv', fmt)
        for fmt in ("A", "B", "C")
    )
    newest: Optional[str] = None
    newest_ts: Any = None
    for path, fmt_filter in sources:
        mine = _own_entries(
            _ranking_entries(path, format_filter=fmt_filter), study_id=study_id, key=key
        )
        if mine is None:
            continue
        named = mine.filter(pl.col('nickname') != "").sort('_ts', nulls_last=False)
        if named.height == 0:
            continue
        candidate = named.row(named.height - 1, named=True)
        if newest_ts is None or (candidate['_ts'] is not None and candidate['_ts'] > newest_ts):
            newest, newest_ts = str(candidate['nickname']), candidate['_ts']
    return normalize_nickname(newest) if newest else ""


def compute_share_rankings(
    study_id: str,
    played_formats: list[str],
    *,
    key: str = "",
) -> dict[str, Any]:
    """Freeze the per-format and overall rankings for one player.

    Returns a dict with:
      - ``per_format``: ``[{format, rank, total}, ...]`` in FORMAT_ORDER order
      - ``overall``: ``{rank, total}`` or ``None``
    Used by the share callback so the share URL always shows the ranks that
    existed at share time, even if the CSVs are appended to later.  ``key`` is the
    hashed email identity, so a player's rows from other devices count as theirs.
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
            key=key,
            format_filter=fmt,
        )
        if info is not None:
            rank, total = info
            per_format.append({"format": fmt, "rank": rank, "total": total})

    overall: Optional[dict[str, int]] = None
    overall_info = _rank_from_ranking_csv(
        project_root / 'data' / 'input' / 'prediction_ranking.csv',
        study_id=study_id,
        key=key,
        format_filter="ALL",
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
    payload: Dict[str, List[Any]] = {
        'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'event_type': df_in.get_column('event_type').to_list(),
        'event_subtype': df_in.get_column('event_subtype').to_list(),
        'insulin_value': df_in.get_column('insulin_value').to_list()
    }
    if 'photo_path' in df_in.columns:
        payload['photo_path'] = [
            str(value or '') for value in df_in.get_column('photo_path').to_list()
        ]
    if 'food_note' in df_in.columns:
        payload['food_note'] = [
            str(value or '') for value in df_in.get_column('food_note').to_list()
        ]
    return payload


def events_within_window(events_df: pl.DataFrame, window_df: pl.DataFrame) -> pl.DataFrame:
    """Trim events to the time span of the window that is being played.

    PERFORMANCE (production freeze, 2026-07-28): the `events-df` store used to
    receive the WHOLE subject's event log -- 62k rows / 3.4 MB of JSON for
    `loop_467`, the largest generic source. It is a `storage_type='local'` store,
    so that payload was written to localStorage and, worse, re-uploaded by the
    browser with every callback request that lists `events-df` as Input/State
    (`auto_snapshot_session`, the chart figure, the source-metadata line). On a
    normal home uplink 3.4 MB is seconds per click, which is exactly what
    players saw the moment a round landed on a big subject: "extremely slow, it
    takes many seconds for every click", then a hard timeout.

    Every consumer already filters events to the window's first/last timestamp
    (`GlucoseChart._add_event_markers`, `window_has_carb_events`,
    `create_ending_layout`), so trimming here is lossless.
    """
    if events_df.height == 0 or 'time' not in events_df.columns:
        return events_df
    if window_df.height == 0 or 'time' not in window_df.columns:
        return events_df.clear()
    window_times = window_df.get_column('time')
    return events_df.filter(
        (pl.col('time') >= window_times[0]) & (pl.col('time') <= window_times[-1])
    )


def events_store_for_window(events_df: pl.DataFrame, window_df: pl.DataFrame) -> Dict[str, List[Any]]:
    """Window-trimmed events as a session-store dictionary (see events_within_window)."""
    return events_dataframe_to_store_dict(events_within_window(events_df, window_df))


def compacted_events_store(
    events_data: Optional[Dict[str, List[Any]]],
    window_data: Optional[Dict[str, List[Any]]],
) -> Any:
    """Trim a whole-subject `events-df` store left in localStorage by an older build.

    Returns ``no_update`` when there is nothing to shrink, so a caller can hand
    the result straight to an ``Output``.

    New rounds already store only the window's events, but a session that was
    mid-game across the deploy keeps the multi-megabyte store in localStorage and
    re-uploads it with every callback that reads `events-df` -- the slowness this
    fixes would otherwise persist for the rest of that round. Navigation is a
    cheap, always reached moment to shrink it; trimmed sessions no-op.

    Works on the store dicts rather than DataFrames because that is what
    localStorage holds; `events_within_window` is the DataFrame equivalent.
    """
    if not events_data or not window_data:
        return no_update
    event_times: list[Any] = events_data.get('time') or []
    window_times: list[Any] = window_data.get('time') or []
    if not event_times or not window_times:
        return no_update
    if any(len(column) != len(event_times) for column in events_data.values()):
        return no_update
    # Store timestamps are fixed-width ISO strings, so string order is chronological.
    start, end = window_times[0], window_times[-1]
    keep = [i for i, stamp in enumerate(event_times) if start <= stamp <= end]
    if len(keep) == len(event_times):
        return no_update
    return {key: [values[i] for i in keep] for key, values in events_data.items()}


def get_random_data_window(
    full_df: pl.DataFrame,
    points: int,
    used_starts: Optional[set[int]] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Get a random window of data from the full DataFrame, avoiding previously
    used start positions when possible.

    Windows that straddle a sensor gap are skipped: ``assign_sequence_ids``
    stamps a new ``sequence_id`` either side of every break, so a window
    touching two of them would ask the player to continue an hour that is not
    actually adjacent to the one they were shown. If no candidate is gap-free
    (a heavily fragmented dataset), the original choice stands rather than
    failing the round.
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
        continuous = [
            start
            for start in candidates
            if window_is_continuous(full_df.slice(start, points))
        ]
        random_start = random.choice(continuous or candidates)
    else:
        random_start = 0

    windowed_df = full_df.slice(random_start, points)
    return windowed_df, random_start


# ---------------------------------------------------------------------------
# Server-side dataset access.
#
# The full CGM dataset is NOT shipped to the browser. Instead it is loaded from
# its on-disk path on the server and cached per-worker; callbacks slice the small
# window they need. Every dataset has a stable file: the example ships in the
# repo (`data/example.csv`), uploads/nightscout are saved under
# `data/input/users/` and their path is kept in `user_info['uploaded_data_path']`.
# See docs / the "stop hauling the whole dataset" refactor.
# ---------------------------------------------------------------------------
EXAMPLE_DATASET_PATH: Path = Path("data/example.csv")


def _dataset_path_for(is_example: bool, uploaded_path: Optional[str]) -> Path:
    """Pick the on-disk dataset path for the example vs an uploaded file."""
    if is_example or not uploaded_path:
        return EXAMPLE_DATASET_PATH
    return Path(str(uploaded_path))


def _resolve_generic_dataset_path(user_info: Dict[str, Any]) -> Path:
    """Resolve the current generic source path from ``data_source_name``."""
    source_name = Path(str(user_info.get("data_source_name") or "")).name
    if not source_name or source_name == EXAMPLE_DATASET_PATH.name:
        return EXAMPLE_DATASET_PATH
    resolved = resolve_generic_source_path(source_name)
    if resolved is not None:
        return resolved
    return EXAMPLE_DATASET_PATH


def resolve_dataset_identity(
    user_info: Optional[Dict[str, Any]], *, round_number: Optional[int] = None
) -> Path:
    """Resolve which on-disk dataset a session (or a specific round) uses.

    Without ``round_number`` this returns the *current window's* dataset, trusting
    the per-round ``is_example_data`` flag in ``user_info`` (set by
    ``handle_next_round_button`` / format switches). With ``round_number`` it
    mirrors ``handle_next_round_button``'s per-format choice so per-round stats can
    resolve the correct dataset even for format C (which alternates datasets).
    Generic rounds use the selected subject file (via ``data_source_name``), not
    always ``example.csv``.
    """
    info = user_info or {}
    fmt = str(info.get("format") or "A")
    uploaded = info.get("uploaded_data_path")
    if round_number is not None:
        if fmt == "B":
            return _dataset_path_for(False, uploaded)
        if fmt == "C":
            # Mirror handle_next_round_button: ODD round -> generic,
            # EVEN round -> uploaded. Round 1 is the generic warm-up once a file
            # exists; before any upload the session is gated (see _is_upload_gated).
            use_generic = round_number % 2 == 1 or not uploaded
            if use_generic:
                return _resolve_generic_dataset_path(info)
            return Path(str(uploaded))
        return _resolve_generic_dataset_path(info)  # format A
    if bool(info.get("is_example_data", True)) or not uploaded:
        return _resolve_generic_dataset_path(info)
    return Path(str(uploaded))


def _load_round_one_stores(
    info: Dict[str, Any],
) -> tuple[Optional[Dict[str, List[Any]]], Optional[Dict[str, List[Any]]], bool, str, int, bool]:
    """First /prediction window after Start: own file for B, generic otherwise.

    Returns ``(window, events, is_example, source_name, slider, randomization_initialized)``.
    B/C without a file stay empty so the upload gate can show.
    """
    if _is_upload_gated(info):
        return None, None, False, "", 0, False
    fmt = str(info.get("format") or "A")
    uploaded_path = info.get("uploaded_data_path")
    if fmt == "B" and uploaded_path:
        full_df, events_df = load_dataset(Path(str(uploaded_path)))
        is_example = False
        source_name = str(
            info.get("uploaded_data_filename") or info.get("data_source_name") or "uploaded.csv"
        )
        df, random_start = get_random_data_window(full_df, DEFAULT_POINTS)
    else:
        df, events_df, source_name, random_start = _apply_generic_round_selection(
            info,
            info.get("rounds"),
            DEFAULT_POINTS,
        )
        is_example = True
    df = df.with_columns(pl.lit(0.0).alias("prediction"))
    return (
        convert_df_to_dict(df),
        events_store_for_window(events_df, df),
        is_example,
        source_name,
        random_start,
        False,
    )


def _is_upload_gated(user_info: Optional[Dict[str, Any]]) -> bool:
    """Whether /prediction must block on an upload right now (hide the chart).

    Formats B and C need a CGM file. If it was imported at startup, never gate.
    If the player arrives without one (skipped startup import, or switched from
    another format), show the gate *before* the graph — then hide it for good
    once ``uploaded_data_path`` is set. Format A never gates.
    """
    info = user_info or {}
    if info.get("uploaded_data_path"):
        return False
    fmt = str(info.get("format") or "A")
    return fmt in ("B", "C")


def _prediction_upload_strip_visible(user_info: Optional[Dict[str, Any]]) -> bool:
    """Show the action-strip Upload control only while a file is still required."""
    return _is_upload_gated(user_info)


def _upload_gate_text(user_info: Optional[Dict[str, Any]], locale: str) -> str:
    """Gate message tailored to the chosen format (own-only vs mixed)."""
    fmt = str((user_info or {}).get("format") or "A")
    key = "ui.prediction.upload_mixed_gate" if fmt == "C" else "ui.prediction.upload_only_gate"
    return t(key, locale=locale)


@lru_cache(maxsize=32)
def _load_dataset_cached(path_str: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    glucose_df, events_df = load_glucose_data(Path(path_str))
    # Match the store schema callers expect (they all reset predictions to 0.0
    # right after loading). The cached frame is treated as immutable: callers
    # only `.slice(...)` it and add predictions to the *window*, never here.
    glucose_df = glucose_df.with_columns(pl.lit(0.0).alias("prediction"))
    return glucose_df, events_df


def load_dataset(path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load + adapt a dataset by path, cached per-worker. Returns (glucose, events).

    Files are immutable (the example is in the repo; uploads have timestamped
    names), so the resolved absolute path is a safe cache key with no mtime check.
    """
    return _load_dataset_cached(str(Path(path).resolve()))


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
    _init_generic_source_name = Path(_chart_file_env).name
else:
    _init_full_df, _init_events_df, _init_generic_source = load_random_generic_dataset()
    _init_generic_source_name = _init_generic_source.source_name

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

example_initial_df_store = dataframe_to_store_dict(_init_window_df)
example_events_df_store = events_store_for_window(_init_events_df, _init_window_df)
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
            "nickname": os.environ.get("_SHARE_NAME", SHARE_NAME),
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
        {"property": "og:image:alt", "content": "Sugar-Sugar glucose prediction game preview card."},
        {"name": "twitter:card", "content": "summary_large_image"},
        {"name": "twitter:title", "content": f"{SITE_TITLE} - Glucose Prediction Game"},
        {"name": "twitter:description", "content": SITE_DESCRIPTION},
        {"name": "twitter:image", "content": _site_og_image_url()},
        {"name": "twitter:image:alt", "content": "Sugar-Sugar glucose prediction game preview card."},
    ],
)
app.title = "Sugar-Sugar - Glucose Prediction Game"

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
        "# Sugar-Sugar\n\n"
        f"{SITE_DESCRIPTION}\n\n"
        "Sugar-Sugar is a Dash research app from GlucoseDAO. Participants predict "
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


@server.route("/cgmacros/<subject>/photo/<path:rel_path>")
def _cgmacros_meal_photo(subject: str, rel_path: str) -> Any:
    """Serve a window-local CGMacros meal JPEG. Path is subject-relative."""
    from flask import abort

    from sugar_sugar.cgmacros import resolve_served_photo

    photo_path = resolve_served_photo(subject, rel_path)
    if photo_path is None:
        abort(404)
    suffix = photo_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
    }.get(suffix, "application/octet-stream")
    response = flask_send_file(
        photo_path,
        mimetype=mime,
        as_attachment=False,
        max_age=86400,
    )
    response.headers["X-Robots-Tag"] = "noindex"
    return response


@server.route("/d1namo/<subject>/photo/<path:rel_path>")
def _d1namo_meal_photo(subject: str, rel_path: str) -> Any:
    """Serve a window-local D1NAMO meal JPEG. Path is subject-relative."""
    from flask import abort

    from sugar_sugar.d1namo import resolve_served_photo as resolve_d1namo_photo

    photo_path = resolve_d1namo_photo(subject, rel_path)
    if photo_path is None:
        abort(404)
    suffix = photo_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
    }.get(suffix, "application/octet-stream")
    response = flask_send_file(
        photo_path,
        mimetype=mime,
        as_attachment=False,
        max_age=86400,
    )
    response.headers["X-Robots-Tag"] = "noindex"
    return response


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
        "age": 30, "gender": "F", "uses_cgm": True,
        "cgm_duration": [2, "years"], "cgm_duration_years": 2,
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


def _staging_ending_args() -> tuple[dict, dict, dict[str, Any]]:
    """Build (window_store, events_store, user_info) for /staging/ending.

    The full dataset is not shipped to the client; create_ending_layout reloads
    it server-side if needed (here current-window-df is always present).
    """
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
        dataframe_to_store_dict(window_df),
        events_store_for_window(events_df, window_df),
        info,
    )


def _staging_final_user_info(*, rounds_n: int = 3, formats: Optional[list[str]] = None) -> dict[str, Any]:
    """Build user_info-with-rounds for /staging/final."""
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
    return info


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
            "nickname": "Staging Tester",
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
        window_store, events_store, info = _staging_ending_args()
        return create_ending_layout(window_store, events_store, info, glucose_unit, locale=locale)
    if pathname == "/staging/final":
        info = _staging_final_user_info()
        return create_final_layout(info, glucose_unit, locale=locale)
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
        window_store, events_store, info = _staging_ending_args()
        return ('/prediction', info, window_store, events_store, True, True, "example.csv")

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
glucose_chart = GlucoseChart(
    id='glucose-graph',
    hide_last_hour=True,  # Hide last hour in prediction page
)
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
# Data-source format for chart mode (A=generic, B=own only, C=mixed). Lets the
# screenshot harness render the B upload-gate and the C prediction page.
_chart_format = os.environ.get("_CHART_FORMAT", "A")
if _chart_format not in ("A", "B", "C"):
    _chart_format = "A"

if _is_chart_mode:
    _chart_user_info: Optional[Dict[str, Any]] = {
        "study_id": str(uuid.uuid4()),
        "email": "dev@chart.local",
        "age": 28,
        "gender": "F",
        "uses_cgm": True,
        "cgm_duration": [1, "years"],
        "cgm_duration_years": 1,
        "format": _chart_format,
        "run_format": _chart_format,
        # B/C in the real flow always reach /prediction already data-consented
        # (update_form_validation gates Start on it), so mirror that here.
        "consent_use_uploaded_data": _chart_format in ("B", "C"),
        "diabetic": True,
        "diabetic_type": "Type 1",
        "diabetes_duration": 5,
        "location": "Dev Machine",
        "rounds": [],
        "max_rounds": MAX_ROUNDS,
        "current_round_number": 1,
        "statistics_saved": False,
        "is_example_data": _chart_is_example,
        # Format B starts with no data (upload gate); keep Source empty until upload.
        "data_source_name": "" if _chart_format == "B" else _chart_source,
        "consent_play_only": False,
        "consent_participate_in_study": True,
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
    if _chart_format == "C":
        # Mixed gates until a file exists. Seed one so `uv run chart --format C`
        # still opens a playable chart (startup import / first-round upload in prod).
        _seed_upload = str(Path(_chart_file_env) if _chart_file_env else EXAMPLE_DATASET_PATH)
        _chart_user_info["uploaded_data_path"] = _seed_upload
        _chart_user_info["uploaded_data_filename"] = Path(_seed_upload).name
else:
    _chart_user_info = None

app.layout = html.Div([
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
    # Client-compressed upload payloads (a clientside callback gzips dcc.Upload
    # contents into these so large CGM files survive the mobile upload path).
    dcc.Store(id='upload-data-payload', storage_type='memory'),
    dcc.Store(id='startup-upload-payload', storage_type='memory'),
    dcc.Store(id='current-window-df', data=example_initial_df_store, storage_type=STORAGE_TYPE),
    # NOTE: there is intentionally no 'full-df' client store. The full dataset is
    # never shipped to the browser -- it is loaded server-side on demand from its
    # on-disk path (see load_dataset / resolve_dataset_identity) and only the small
    # current-window-df is kept client-side. This removes the per-interaction
    # whole-dataset round-trip (lag) and the localStorage-quota risk.
    dcc.Store(id='events-df', data=example_events_df_store, storage_type=STORAGE_TYPE),
    dcc.Store(id='is-example-data', data=_chart_is_example, storage_type=STORAGE_TYPE),
    dcc.Store(id='data-source-name', data=("" if _chart_format == "B" else _chart_source) if _is_chart_mode else _init_generic_source_name, storage_type=STORAGE_TYPE),
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
    # One-shot signal that the localStorage-backed game stores have hydrated, so
    # `display_page` can re-render a game route it first rendered too early (a
    # cold load lands before hydration -- see _restoring_layout). Memory: a fresh
    # page load must start over at False.
    dcc.Store(id='game-stores-hydrated', data=False, storage_type='memory'),
    # Two-step /final fill: display_page writes phase 1 with the shell, then
    # fill_final_leaderboard advances to 2 and fill_final_share paints the card.
    # Memory so a reload starts over. Never put an Interval in page-content —
    # a dcc.Interval created as a callback output does not tick (01:53 /final).
    dcc.Store(id='final-fill-step', data=None, storage_type='memory'),
    # Server-side truth about whether the drawing chart is actually on screen.
    # The `route-prediction` <html> class (and every prediction-only CSS rule,
    # incl. the `:not(.route-prediction)` mobile overflow/tap-reliability
    # releases) is keyed on this, NOT on the pathname alone -- the URL can say
    # /prediction while the rendered content is something else entirely.
    dcc.Store(id='prediction-chart-rendered', data=False, storage_type='memory'),
    # Clientside location-autocomplete init ping (see assets/location-autocomplete.js).
    dcc.Store(id='location-autocomplete-ping', data=0, storage_type='memory'),

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
    # Must sit immediately after #page-content: CSS :has(#final-title) hides it
    # once display_page replaces ending with /final. Shown clientside on the
    # first Results click so the still-visible ending button is not clicked again
    # while create_final_layout runs (~3s).
    html.Div(
        html.Div("…", className="results-loading-card", disable_n_clicks=True),
        id="results-loading-overlay",
        className="results-loading-overlay",
        disable_n_clicks=True,
    ),

    html.Div(
        [
            html.Div(
                id='meal-food-lightbox-backdrop',
                className='meal-food-lightbox-backdrop',
                n_clicks=0,
            ),
            html.Img(
                id='meal-food-lightbox-image',
                className='meal-food-lightbox-image',
                src='',
                alt='',
                disable_n_clicks=True,
            ),
            html.Div(
                [
                    html.Img(
                        id={'type': 'meal-food-lightbox-tile', 'index': tile_i},
                        className='meal-food-lightbox-tile',
                        src='',
                        alt='',
                        disable_n_clicks=True,
                    )
                    for tile_i in range(FOOD_COMPOSITE_MAX)
                ],
                id='meal-food-lightbox-gallery',
                className='meal-food-lightbox-gallery',
                disable_n_clicks=True,
            ),
            html.Div(
                id='meal-food-lightbox-note',
                className='meal-food-lightbox-note',
                children='',
                disable_n_clicks=True,
            ),
        ],
        id='meal-food-lightbox',
        className='meal-food-lightbox',
        disable_n_clicks=True,
        **{"aria-hidden": "true"},
    ),

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
    function(pathname, chartRendered) {
        var root = document.documentElement;
        // Both conditions, deliberately: the pathname drops the class the instant
        // we navigate away, and `chartRendered` (server-side truth, see
        // mark_prediction_chart_rendered) withholds it until the chart is really
        // on screen. The URL alone once stamped it onto a consent form rendered
        // at /prediction, releasing the mobile overflow cap and tap-reliability
        // rules that are scoped `:not(.route-prediction)`.
        var isPrediction = (pathname === '/prediction') && !!chartRendered;
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
        if (pathname === '/faq' && window.location && window.location.hash) {
            var targetId = window.location.hash.replace('#', '');
            window.setTimeout(function() {
                var el = document.getElementById(targetId);
                if (el && el.scrollIntoView) {
                    el.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
            }, 80);
        }
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
    [Input('url', 'pathname'),
     Input('prediction-chart-rendered', 'data')],
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
def update_fullscreen_button_text(interface_language: Optional[str], pathname: Optional[str]) -> Any:
    """Keep the 'Go fullscreen' button (expand icon + label) translated."""
    if pathname != '/prediction':
        raise PreventUpdate
    return _fullscreen_button_children(normalize_locale(interface_language))


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


_PREDICTION_CONSENT_WRAP_VISIBLE: Dict[str, str] = {
    'maxWidth': '900px',
    'margin': '0 auto',
    'padding': '12px 16px',
    'backgroundColor': 'white',
    'borderRadius': '10px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.06)',
    'border': '1px solid #e5e7eb',
    'display': 'block',
}
_PREDICTION_CONSENT_WRAP_HIDDEN: Dict[str, str] = {
    **_PREDICTION_CONSENT_WRAP_VISIBLE,
    'display': 'none',
}


@app.callback(
    [
        Output('prediction-data-usage-consent-wrap', 'style'),
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
) -> Tuple[Dict[str, str], Dict[str, str], list[dict[str, Any]], list[str], Optional[html.Div]]:
    if pathname != '/prediction':
        raise PreventUpdate
    if not user_info:
        raise PreventUpdate

    fmt = str(user_info.get("format") or "A")
    locale = normalize_locale(interface_language)
    base_label = t("ui.startup.data_usage_consent_label", locale=locale)
    consented = _upload_data_consent_given(user_info)
    # Already consented (form / prior B/C play / upload) or not a B/C session —
    # keep the checklist in the DOM for upload wiring, but never show it again.
    if fmt not in ("B", "C") or consented:
        return (
            _PREDICTION_CONSENT_WRAP_HIDDEN,
            {'display': 'none'},
            [{'label': base_label, 'value': 'agree', 'disabled': True}],
            ['agree'] if consented else [],
            None,
        )

    return (
        _PREDICTION_CONSENT_WRAP_VISIBLE,
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

# Game routes whose content is rebuilt from localStorage-backed stores, so they
# cannot be rendered correctly until those stores have hydrated.
_GAME_ROUTES: frozenset[str] = frozenset({'/prediction', '/ending', '/final'})

# How long the restoring placeholder waits for hydration before giving up and
# routing to landing (ticks of `session-restore-poll`, 250 ms each).
_RESTORE_POLL_MS: int = 250
_RESTORE_GIVE_UP_TICKS: int = 16  # ~4 s; hydration normally lands in well under one tick.


def _game_stores_ready(
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict],
) -> bool:
    """True when the stores needed to rebuild *pathname* have hydrated.

    ``dcc.Store(storage_type='local')`` values arrive **after** the first server
    render, and `display_page` reads them as ``State`` (it must -- making
    user-info-store an Input would re-render, and so destroy, the live chart on
    every round). So a *full page load* whose URL is already a game route (an
    Android tab restore days later, a pull-to-refresh on the chart, a bookmark,
    F5) renders with ``user_info=None``.

    That used to fall through `display_page` to the **landing page** -- which on
    mobile leads straight into the consent wizard -- and nothing ever re-rendered,
    because `page-content` only changes when `url.pathname` changes and
    `restore_page_on_load` bails out for any pathname other than ``/``. A player
    resuming mid-study was silently dumped back into the consent form with her
    session intact in localStorage (reported: Samsung/Android portrait, 3 days
    after saving a session).
    """
    if pathname not in _GAME_ROUTES:
        return True
    if not user_info:
        return False
    if pathname == '/ending':
        # /ending also reads the played window back out of the client store.
        return bool(current_df_data)
    return True


def _renders_prediction_chart(
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
) -> bool:
    """True when `display_page` will actually render the drawing chart.

    Single source of truth shared with `mark_prediction_chart_rendered`, which
    drives the ``route-prediction`` <html> class. Keep the two in step: the class
    must never be stamped on non-chart content (see that callback for why).
    """
    return bool(
        pathname == '/prediction'
        and user_info
        and user_info.get('consent_completed')
    )


def _restoring_layout(*, locale: str) -> html.Div:
    """Neutral "restoring your session" placeholder for an un-hydrated game route.

    Rendered instead of the landing page / "session expired" screen while the
    localStorage stores catch up (see `_game_stores_ready`). The embedded
    `session-restore-poll` interval is what re-renders the page once they do --
    deliberately scoped to this layout, since a Dash callback only fires while
    every component it references is in the DOM. That makes it impossible for the
    re-render to fire mid-game and reset the chart.
    """
    return html.Div(
        [
            html.H2(
                t("ui.restoring.title", locale=locale),
                style={'textAlign': 'center', 'marginTop': '50px'},
                disable_n_clicks=True,
            ),
            html.P(
                t("ui.restoring.text", locale=locale),
                style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#475569'},
                disable_n_clicks=True,
            ),
            html.Div(
                html.A(
                    t("ui.common.go_to_start", locale=locale),
                    href="/",
                    style={
                        'backgroundColor': '#007bff',
                        'color': 'white',
                        'padding': '15px 30px',
                        'textDecoration': 'none',
                        'borderRadius': '5px',
                        'fontSize': '18px',
                    },
                ),
                style={'textAlign': 'center'},
                disable_n_clicks=True,
            ),
            dcc.Interval(
                id='session-restore-poll',
                interval=_RESTORE_POLL_MS,
                n_intervals=0,
                max_intervals=_RESTORE_GIVE_UP_TICKS,
            ),
        ],
        id='session-restoring',
        disable_n_clicks=True,
    )

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
    """Detect a mobile client from the current Flask request's User-Agent.

    Tests and CLI builders have no request: treat them as desktop so layout
    construction never depends on a live HTTP context.
    """
    if not has_request_context():
        return False
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
     Output('navbar-container', 'children', allow_duplicate=True),
     Output('final-fill-step', 'data', allow_duplicate=True)],
    [Input('interface-language', 'data')],
    [State('url', 'pathname'),
     State('user-info-store', 'data'),
     State('user-agent', 'data'),
     State('glucose-unit', 'data')],
    prevent_initial_call=True,
)
def update_on_language_change(
    interface_language: Optional[str],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    user_agent: Optional[str],
    glucose_unit: Optional[str],
) -> tuple:
    page, warning, navbar = _update_language_page(
        interface_language, pathname, user_info, user_agent, glucose_unit
    )
    kick: Any = (
        {"phase": 1, "nonce": time.time_ns()}
        if pathname == "/final" and user_info and page is not no_update
        else no_update
    )
    return page, warning, navbar, kick


def _update_language_page(
    interface_language: Optional[str],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    user_agent: Optional[str],
    glucose_unit: Optional[str],
) -> tuple:
    """Re-render page content and navbar when language changes.

    Pages with interactive state (prediction chart, ending) only get
    a navbar refresh -- page content is left untouched via per-element callbacks.
    """
    locale = normalize_locale(interface_language)
    navbar = _navbar(locale=locale, pathname=pathname)

    if pathname in _STATEFUL_PAGES:
        # /prediction still renders the landing/consent page when consent is
        # missing, so a language switch there has to rebuild that page rather
        # than leave it half-translated. Skipping the rebuild updates the navbar
        # only, which is what keeps the live chart (and the drawn line) alive.
        #
        # `user_info` is State, and `interface-language` is a localStorage store
        # like any other: on a cold load onto /prediction it hydrates while
        # user-info-store may still read None, and any stored locale other than
        # the layout default fires this callback. None there means "the stores
        # have not arrived yet", NOT "this player never consented" -- rebuilding
        # landing on it unmounts `_restoring_layout`, and with it the
        # session-restore-poll that is the only component able to re-render the
        # route. The player is then stranded on the consent form with the URL
        # still /prediction (the August 2026 report this placeholder fixed).
        # Today user-info-store happens to hydrate first because it sits earlier
        # in the layout; that ordering is not guaranteed and must not be relied on.
        if (
            pathname == '/prediction'
            and user_info is not None
            and not _renders_prediction_chart(pathname, user_info)
        ):
            warning_content = render_mobile_warning(user_agent, locale=locale)
            return _landing_builder(locale=locale), warning_content, navbar
        return no_update, no_update, navbar

    warning_content = render_mobile_warning(user_agent, locale=locale)
    if _is_staging_mode and pathname and pathname.startswith('/staging'):
        staging_layout = _staging_display(pathname, locale=locale, glucose_unit=glucose_unit)
        if staging_layout is not None:
            return staging_layout, warning_content, navbar
    if pathname == '/final':
        if user_info:
            return create_final_layout(user_info, glucose_unit, locale=locale, eager=False), warning_content, navbar
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
    if pathname == '/highscore':
        return create_highscore_page(user_info, glucose_unit, locale=locale), warning_content, navbar
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
     Output('finish-study-button', 'title'),
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
        t("ui.header.upload_button", locale=locale),
        t("ui.header.use_example_data", locale=locale),
        t("ui.header.time_window_label", locale=locale),
        t("ui.chart.y_axis_label", locale=locale),
        t("ui.startup.data_usage_consent_label", locale=locale),
        t("ui.submit.finish_game", locale=locale),
        t("ui.submit.finish_game", locale=locale),
        t("ui.header.nightscout_load_button", locale=locale),
    )


@app.callback(
    [Output('ending-title', 'children'),
     Output('ending-disclaimer-line1', 'children'),
     Output('ending-disclaimer-line2', 'children'),
     Output('ending-disclaimer-line3', 'children'),
     Output('ending-gamification', 'children'),
     Output('ending-units-line', 'children'),
     Output('ending-source-info', 'children'),
     Output('ending-graph-explanation', 'children'),
     Output('ending-prediction-results-title', 'children'),
     Output('ending-prediction-details-toggle', 'children'),
     Output('ending-prediction-table', 'rowData'),
     Output('ending-prediction-table', 'columnDefs'),
     Output('ending-metrics-summary', 'children'),
     Output('ending-metrics-details-toggle', 'children'),
     Output('ending-metrics-container', 'children'),
     Output('ending-local-storage-note', 'children'),
     Output('finish-study-button-ending', 'children'),
     Output('next-round-button', 'children'),
     Output('ending-switch-format-title', 'children'),
     Output('switch-format-c', 'children'),
     Output('switch-format-a', 'children'),
     Output('switch-format-b', 'children'),
     Output('ending-copy-link-button', 'children')],
    [Input('interface-language', 'data')],
    [State('url', 'pathname'),
     State('user-info-store', 'data'),
     State('glucose-unit', 'data'),
     State('current-window-df', 'data'),
     State('events-df', 'data')],
    prevent_initial_call=True,
)
def update_ending_text_on_language_change(
    interface_language: Optional[str],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    glucose_unit: Optional[str],
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
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
        metrics_display = MetricsComponent.create_ending_metrics_display(
            stored_metrics, locale=locale, include_title=False,
        ) if stored_metrics else [
            html.Div(
                t("ui.metrics.no_metrics_available", locale=locale),
                style={'color': 'gray', 'fontStyle': 'italic', 'fontSize': '16px', 'padding': '10px', 'textAlign': 'center'},
            )
        ]

    finish_button_text = t("ui.ending.results", locale=locale) if is_last_round else t("ui.submit.finish_game", locale=locale)

    empty_events = pl.DataFrame(
        {"time": [], "event_type": [], "event_subtype": [], "insulin_value": []}
    )
    if current_df_data:
        window_df = reconstruct_dataframe_from_dict(current_df_data)
    else:
        window_df = pl.DataFrame(
            {"time": [], "gl": [], "prediction": [], "age": [], "user_id": []}
        )
    events_df = (
        reconstruct_events_dataframe_from_dict(events_df_data)
        if events_df_data
        else empty_events
    )
    source_plaque = _ending_source_plaque_children(
        user_info=user_info,
        window_df=window_df,
        events_df=events_df,
        locale=locale,
    )

    return (
        t("ui.ending.title", locale=locale),
        t("ui.results_disclaimer.line1", locale=locale),
        t("ui.results_disclaimer.line2", locale=locale),
        t("ui.results_disclaimer.line3", locale=locale),
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
        source_plaque,
        t("ui.ending.graph_explanation", locale=locale),
        t("ui.ending.prediction_results", locale=locale),
        t("ui.ending.click_here_for_details", locale=locale),
        table_data,
        table_columns,
        t("ui.metrics.title_accuracy_metrics", locale=locale),
        t("ui.ending.click_here_for_details", locale=locale),
        metrics_display,
        t("ui.ending.local_storage_note", locale=locale),
        finish_button_text,
        t("ui.ending.next_round", locale=locale),
        t("ui.switch_format.title", locale=locale),
        t("ui.switch_format.try_c_short", locale=locale),
        t("ui.switch_format.try_a_short", locale=locale),
        t("ui.switch_format.try_b_short", locale=locale),
        t("ui.resume_code.copy_link", locale=locale),
    )


@app.callback(
    [Output('page-content', 'children'),
     Output('mobile-warning', 'children'),
     Output('navbar-container', 'children'),
     Output('final-fill-step', 'data')],
    [Input('url', 'pathname'),
     Input('game-stores-hydrated', 'data')],
    [State('interface-language', 'data'),
     State('user-info-store', 'data'),
     State('current-window-df', 'data'),
     State('events-df', 'data'),
     State('glucose-unit', 'data'),
     State('user-agent', 'data')],
    prevent_initial_call=False
)
def display_page(
    pathname: Optional[str],
    stores_hydrated: Optional[bool],
    interface_language: Optional[str],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
    glucose_unit: Optional[str],
    user_agent: Optional[str],
) -> tuple[html.Div, Optional[html.Div], html.Div, Any]:
    page, warning, navbar = _render_page(
        pathname,
        stores_hydrated,
        interface_language,
        user_info,
        current_df_data,
        events_df_data,
        glucose_unit,
        user_agent,
    )
    ready = _game_stores_ready(pathname, user_info, current_df_data)
    # ``no_update`` on every other route: writing None here would re-fire the
    # /final fill callbacks after Exit has already unmounted their outputs.
    kick: Any = (
        {"phase": 1, "nonce": time.time_ns()}
        if pathname == "/final" and user_info and ready
        else no_update
    )
    return page, warning, navbar, kick


def _render_page(
    pathname: Optional[str],
    stores_hydrated: Optional[bool],
    interface_language: Optional[str],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
    glucose_unit: Optional[str],
    user_agent: Optional[str],
) -> tuple[html.Div, Optional[html.Div], html.Div]:
    has_ptd = bool(user_info and 'prediction_table_data' in user_info) if user_info else False
    print(f"DEBUG display_page: pathname={pathname} has_user_info={user_info is not None} has_prediction_table_data={has_ptd}")
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
        # Hydration guard: on a full page load the localStorage stores arrive after
        # this first render, so a game route would otherwise be rebuilt from empty
        # state -- landing/consent for /prediction, "session expired" for
        # /ending and /final. Hold a neutral placeholder instead; its poll flips
        # `game-stores-hydrated`, which re-runs this callback with real data.
        if not _game_stores_ready(pathname, user_info, current_df_data):
            return _restoring_layout(locale=locale), warning_content, navbar
        # Consent guard: mandatory consent (acknowledge + GDPR) must have been
        # recorded before the game flow is reachable. `consent_completed` is set
        # by handle_landing_continue (desktop) and the mobile wizard's Start
        # handler. Without it, a direct-URL / burger-menu visit could otherwise
        # bypass the consent gate (desktop /startup omits the consent fields, so
        # handle_start_button would skip its own check).
        consent_done = bool(user_info and user_info.get('consent_completed'))
        if pathname == '/prediction' and user_info:
            # Same predicate as `mark_prediction_chart_rendered`, so the
            # `route-prediction` CSS class can never disagree with what is drawn.
            if not _renders_prediction_chart(pathname, user_info):
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
            # Check if we have the required data for ending page. Keyed on the
            # small window store (full-df is no longer shipped to the client);
            # create_ending_layout reloads the dataset server-side if it needs a
            # fallback window.
            if not current_df_data or not user_info or 'prediction_table_data' not in user_info:
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
            return create_ending_layout(current_df_data, events_df_data, user_info, glucose_unit, locale=locale), warning_content, navbar
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
            return create_final_layout(user_info, glucose_unit, locale=locale, eager=False), warning_content, navbar
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
        if pathname == '/highscore':
            return create_highscore_page(user_info, glucose_unit, locale=locale), warning_content, navbar
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


def _faq_tag_options(locale: str) -> list[dict[str, str]]:
    from sugar_sugar.faq_board import allowed_faq_tags
    return [
        {"label": t(f"ui.faq.tag_{tag}", locale=locale), "value": tag}
        for tag in allowed_faq_tags()
    ]


def _faq_section_options(locale: str) -> list[dict[str, str]]:
    return [
        {"label": t("ui.faq.section_participant", locale=locale), "value": "participant"},
        {"label": t("ui.faq.section_developer", locale=locale), "value": "developer"},
    ]


def _faq_post_card(item: dict[str, Any], *, locale: str) -> html.Div:
    qid = str(item.get("id") or "")
    tags = [t(f"ui.faq.tag_{tag}", locale=locale) for tag in (item.get("tags") or [])]
    replies = item.get("replies") or []
    reply_nodes: list[Any] = []
    for reply in replies:
        who = t(
            "ui.faq.section_developer" if reply.get("section") == "developer" else "ui.faq.section_participant",
            locale=locale,
        )
        name = str(reply.get("name") or "").strip()
        label = f"{who} · {name}" if name else who
        reply_nodes.append(
            html.Div(
                [
                    html.Div(label, style={"fontWeight": "700", "marginBottom": "4px"}, disable_n_clicks=True),
                    html.Div(str(reply.get("text") or ""), disable_n_clicks=True),
                ],
                className="faq-reply",
                disable_n_clicks=True,
            )
        )
    name = str(item.get("name") or "").strip()
    who = t(
        "ui.faq.section_developer" if item.get("section") == "developer" else "ui.faq.section_participant",
        locale=locale,
    )
    return html.Div(
        [
            html.Div(
                f"{who}" + (f" · {name}" if name else ""),
                style={"fontWeight": "700", "color": "#1e3a5f"},
                disable_n_clicks=True,
            ),
            html.Div(
                " · ".join(tags),
                style={"color": "#64748b", "fontSize": "13px", "margin": "4px 0 8px"},
                disable_n_clicks=True,
            ),
            html.Div(str(item.get("text") or ""), style={"whiteSpace": "pre-wrap"}, disable_n_clicks=True),
            html.Div(reply_nodes, disable_n_clicks=True),
            html.Div(
                [
                    dcc.Textarea(
                        id={"type": "faq-reply-text", "index": qid},
                        placeholder=t("ui.faq.reply_placeholder", locale=locale),
                        style={"width": "100%", "minHeight": "70px", "marginTop": "10px"},
                    ),
                    dcc.RadioItems(
                        id={"type": "faq-reply-section", "index": qid},
                        options=_faq_section_options(locale),
                        value="developer" if item.get("section") == "participant" else "participant",
                        inline=True,
                        style={"margin": "8px 0"},
                    ),
                    html.Button(
                        t("ui.faq.reply_button", locale=locale),
                        id={"type": "faq-reply-submit", "index": qid},
                        className="ui blue button",
                        n_clicks=0,
                    ),
                ],
                className="faq-reply-form",
            ),
        ],
        className="ui segment faq-question",
        disable_n_clicks=True,
    )


def faq_board_children(*, locale: str) -> list[Any]:
    from sugar_sugar.faq_board import load_faq_questions
    items = load_faq_questions()
    participants = [item for item in items if item.get("section") != "developer"]
    developers = [item for item in items if item.get("section") == "developer"]
    empty = html.Div(
        t("ui.faq.board_empty", locale=locale),
        style={"color": "#64748b", "fontStyle": "italic"},
        disable_n_clicks=True,
    )
    return [
        html.H2(t("ui.faq.board_participant_title", locale=locale), disable_n_clicks=True),
        html.Div(
            [_faq_post_card(item, locale=locale) for item in reversed(participants)] or [empty],
            disable_n_clicks=True,
        ),
        html.H2(t("ui.faq.board_developer_title", locale=locale), disable_n_clicks=True),
        html.Div(
            [_faq_post_card(item, locale=locale) for item in reversed(developers)] or [empty],
            disable_n_clicks=True,
        ),
    ]


def create_faq_page(*, locale: str) -> html.Div:
    from sugar_sugar.faq_board import faq_board_enabled

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
        section_id = str(section.get("id") or "").strip()
        section_kwargs: dict[str, Any] = {"disable_n_clicks": True}
        if section_id:
            section_kwargs["id"] = section_id
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
                **section_kwargs,
            )
        )
    children: list[Any] = [
        html.H1(t("ui.faq.title", locale=locale), disable_n_clicks=True),
        html.Div(section_divs, disable_n_clicks=True),
    ]
    if faq_board_enabled():
        children.extend(_faq_board_form_children(locale=locale))
    return html.Div(
        children,
        className="info-page",
        disable_n_clicks=True,
    )


def _faq_board_form_children(*, locale: str) -> list[Any]:
    return [
        html.Div(
            [
                html.H2(t("ui.faq.ask_title", locale=locale), disable_n_clicks=True),
                html.Div(
                    t("ui.faq.ask_intro", locale=locale),
                    style={"marginBottom": "10px", "color": "#334155"},
                    disable_n_clicks=True,
                ),
                dcc.Textarea(
                    id="faq-ask-text",
                    placeholder=t("ui.faq.ask_placeholder", locale=locale),
                    style={"width": "100%", "minHeight": "110px"},
                ),
                dcc.Input(
                    id="faq-ask-name",
                    type="text",
                    placeholder=t("ui.faq.name_placeholder", locale=locale),
                    style={"width": "100%", "marginTop": "8px"},
                ),
                html.Div(
                    t("ui.faq.tags_label", locale=locale),
                    style={"fontWeight": "700", "margin": "10px 0 6px"},
                    disable_n_clicks=True,
                ),
                dcc.Checklist(
                    id="faq-ask-tags",
                    options=_faq_tag_options(locale),
                    value=[],
                    inline=True,
                ),
                dcc.RadioItems(
                    id="faq-ask-section",
                    options=_faq_section_options(locale),
                    value="participant",
                    inline=True,
                    style={"margin": "10px 0"},
                ),
                html.Button(
                    t("ui.faq.ask_button", locale=locale),
                    id="faq-ask-submit",
                    className="ui green button",
                    n_clicks=0,
                ),
                html.Div(id="faq-ask-status", style={"marginTop": "8px", "color": "#1b5e20"}),
            ],
            id="faq-ask-form",
            className="ui segment",
            style={"marginTop": "28px"},
        ),
        html.Div(faq_board_children(locale=locale), id="faq-board"),
    ]

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


def _fullscreen_button_children(locale: str) -> list[Any]:
    """Fullscreen button content: FontAwesome four-corners expand icon + label."""
    return [
        html.I(className="fas fa-expand", style={"marginRight": "8px"}),
        t("ui.orientation.go_fullscreen", locale=locale),
    ]


def create_prediction_layout(*, locale: str, format_value: str, user_info: Dict[str, Any]) -> html.Div:
    """Create the prediction page layout"""
    show_upload = format_value in ("B", "C")
    # Gate B/C until a file exists (startup import or first-round upload).
    b_gated = _is_upload_gated(user_info)
    upload_strip_visible = _prediction_upload_strip_visible(user_info)
    consent_given = _upload_data_consent_given(user_info)
    consent_value = ['agree'] if consent_given else []
    show_consent_ui = _show_prediction_upload_consent(user_info, show_upload=show_upload)
    data_source_name = str(user_info.get("data_source_name") or "")
    if b_gated:
        # While gated (awaiting upload) keep the Source blank -- any stale value
        # refers to the generic warm-up, not the user's data.
        data_source_display = ""
    elif data_source_name:
        data_source_display = data_source_name
    elif format_value == "B":
        # "My data only": keep the Source blank until the user uploads their file.
        data_source_display = ""
    else:
        # A / C start on the generic example.
        data_source_display = "example.csv"
    return html.Div([
        HeaderComponent(
            show_time_slider=False,
            # The CSV upload now lives in the always-visible action strip (so it is
            # reachable in landscape for B/C); the header no longer renders it.
            show_upload_section=False,
            show_example_button=(format_value == "A"),
            show_data_source_section=False,
            render_csv_upload=(format_value == "A"),
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
            id="prediction-data-usage-consent-wrap",
            # Consent is recorded on landing/startup for B/C — only show this
            # fallback when it is somehow missing. Kept in the DOM (display:none)
            # so handle_file_upload's Input('prediction-data-usage-consent') stays wired.
            style=(
                _PREDICTION_CONSENT_WRAP_VISIBLE
                if show_consent_ui
                else _PREDICTION_CONSENT_WRAP_HIDDEN
            ),
        ),
        html.Div(id="upload-required-alert", style={'margin': '0 auto', 'maxWidth': '900px'}),
        html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            t("ui.common.round_of", locale=locale, current=1, total=user_info.get("max_rounds", MAX_ROUNDS)).replace("Round", "Prediction Round", 1),
                            id="prediction-round-tagline",
                            className="prediction-round-tagline",
                            style={"display": "none"},
                        ),
                        html.Div(id='round-indicator', style={
                            'textAlign': 'left',
                            'fontSize': '18px',
                            'fontWeight': '600',
                            'color': '#2c5282',
                            'marginBottom': '0'
                        }),
                    ],
                    id="prediction-round-summary",
                    disable_n_clicks=True,
                ),
                html.Div(
                    [
                        html.Div(id='generic-source-metadata-display', children="", className="prediction-source-metadata"),
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
                    ],
                    id="prediction-source-plaque",
                    disable_n_clicks=True,
                ),
            ],
            id="prediction-meta-row",
            className="prediction-meta-row",
            disable_n_clicks=True,
        ),
        html.Div([
            html.Div(
                [
                    # Sits on the chart near the y-axis: horizontal "Glucose Level"
                    # continued by the mg/dL ↔ mmol/L toggle (same ids/callbacks).
                    html.Div(
                        [
                            html.Span(
                                t("ui.chart.y_axis_label", locale=locale),
                                id='prediction-units-label',
                                className='chart-yaxis-units-label',
                            ),
                            dbc.RadioItems(
                                id='glucose-unit-selector',
                                options=[
                                    {'label': 'mg/dL', 'value': 'mg/dL'},
                                    {'label': 'mmol/L', 'value': 'mmol/L'},
                                ],
                                value='mg/dL',
                                inline=True,
                                className='chart-unit-toggle',
                            ),
                        ],
                        id='prediction-units-row',
                        className='chart-yaxis-units',
                    ),
                    GlucoseChart(
                        id='glucose-graph',
                        hide_last_hour=True,
                    ),
                ],
                id='prediction-glucose-chart-container',
                style={'display': 'none'} if b_gated else None,
            ),
            # Format B/C upload gate: shown in place of the chart until the user
            # uploads their file. The Upload button itself lives in the action strip
            # below (reachable in both portrait and landscape) and is hidden once
            # a file is stored.
            html.Div(
                _upload_gate_text(user_info, locale),
                id="prediction-upload-gate",
                className="prediction-upload-gate",
                disable_n_clicks=True,
                style={'display': 'block'} if b_gated else {'display': 'none'},
            ),
            html.Div(
                [
                    html.Button(
                        _fullscreen_button_children(locale),
                        id="prediction-fullscreen-button",
                        className="prediction-fullscreen-button",
                        type="button",
                    ),
                    # B/C: CSV upload lives in the action strip so it stays
                    # reachable in landscape. Shown only while a file is still
                    # required (gate); hidden once uploaded so it does not steal
                    # chart space. Wrapper stays in the DOM so callbacks keep
                    # `upload-data` / `header-upload-prompt` wired.
                    html.Div(
                        make_csv_upload(
                            locale, style={}, className="prediction-upload-button"
                        ) if show_upload else None,
                        id="prediction-upload-slot",
                        className=(
                            "prediction-upload-visible"
                            if upload_strip_visible
                            else "prediction-upload-hidden"
                        ),
                        disable_n_clicks=True,
                    ),
                    SubmitComponent(locale=locale),
                ],
                id="prediction-mobile-actions",
                className="has-upload" if upload_strip_visible else "",
            ),
        ], id='prediction-chart-submit-wrap', style={'flex': '1'}),
        dcc.Store(id='finish-confirm-context-prediction', data=None, storage_type='memory'),
        finish_confirm_overlay(locale, source="prediction"),
    ], id="prediction-page", className="prediction-page", style={
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
    # Superseded by the dedicated upload gate (prediction-upload-gate), which now
    # handles B and C until a file is uploaded. Kept as an inert Output sink so the
    # 'upload-required-alert' element stays wired; the Finish/Exit button in the
    # action strip remains available for users who don't want to upload.
    return None


@app.callback(
    [Output('prediction-glucose-chart-container', 'style', allow_duplicate=True),
     Output('prediction-upload-gate', 'style'),
     Output('prediction-upload-gate', 'children'),
     Output('prediction-chart-submit-wrap', 'className'),
     Output('prediction-meta-row', 'className'),
     Output('prediction-upload-slot', 'className'),
     Output('prediction-mobile-actions', 'className')],
    [Input('url', 'pathname'),
     Input('user-info-store', 'data'),
     Input('interface-language', 'data')],
    prevent_initial_call='initial_duplicate',
)
def toggle_upload_gate(
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    interface_language: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, str], Any, str, str, str, str]:
    """Show the B/C upload gate (and hide the chart) until a file is uploaded.

    Runs on load as well as navigation so it also catches direct loads / resumes
    onto /prediction, where the persisted example window would otherwise leak a
    playable generic chart into a "my data only" session. Keyed on user_info (not
    the window store) so it never clobbers a just-uploaded window and never loops.

    The ``b-gated`` class on the chart/submit wrapper and the meta row lets CSS
    force-hide the Source plaque (whose time/metadata reflect the stale example
    seed) even in immersive landscape, where the plaque style is otherwise
    re-shown with ``!important`` and would beat an inline ``display:none``.

    The strip Upload button and ``has-upload`` class follow the same gate: once
    a file is stored they disappear so the chart keeps the format-A layout.
    """
    if pathname != '/prediction':
        raise PreventUpdate
    locale = normalize_locale(interface_language)
    if _is_upload_gated(user_info):
        return (
            {'display': 'none'},
            {'display': 'block'},
            _upload_gate_text(user_info, locale),
            'b-gated',
            'prediction-meta-row b-gated',
            'prediction-upload-visible',
            'has-upload',
        )
    return (
        {'display': 'block'},
        {'display': 'none'},
        no_update,
        '',
        'prediction-meta-row',
        'prediction-upload-hidden',
        '',
    )


@app.callback(
    Output('startup-step', 'data', allow_duplicate=True),
    Input('url', 'pathname'),
    State('startup-step', 'data'),
    prevent_initial_call=True,
)
def reset_startup_wizard_step(pathname: Optional[str], current_step: Optional[int]) -> Any:
    """Reset the mobile startup wizard to step 0 whenever /startup is (re)entered.

    `startup-step` is a memory store that survives client-side (SPA) navigation --
    it is only reset by a full browser reload, and nothing else resets it on
    arrival. So after leaving mid-wizard and returning (resume "Continue",
    tab-switch, or a play-again reset that routes through /), the store kept a
    stale index while the layout re-baked step 0 visible; the next Back/Next then
    jumped several steps in (surfaced as "resumed at Step 5 of 6"). Snapping it
    back to 0 on entry keeps the store in sync with the freshly rendered layout.
    """
    if pathname != '/startup':
        raise PreventUpdate
    if int(current_step or 0) == 0:
        raise PreventUpdate
    return 0


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
    """Segmented progress bar; filled segments stay green for every round."""
    segments: list[html.Div] = []
    for i in range(1, max_rounds + 1):
        filled = i <= current_round
        bg = "#4CBB17" if filled else "#e0e0e0"
        border_right = "2px solid white" if i < max_rounds else "none"
        border_left = "3px solid #888" if i == min_useful + 1 else "none"
        segments.append(html.Div(
            disable_n_clicks=True,
            style={
                "flex": "1",
                "height": "12px",
                "backgroundColor": bg,
                "borderRight": border_right,
                "borderLeft": border_left,
                "transition": "background-color 0.3s",
            },
        ))

    labels = html.Div([
        html.Span(
            t("ui.ending.progress.minimum_goal", locale=locale, min_useful=min_useful),
            style={"fontSize": "11px", "color": "#4a5568", "fontWeight": "600"},
        ),
        html.Span(
            t("ui.ending.progress.stretch_goal", locale=locale, total=max_rounds),
            style={"fontSize": "11px", "color": "#4a5568", "fontWeight": "600"},
        ),
    ], disable_n_clicks=True, style={
        "display": "flex",
        "justifyContent": "space-between",
        "marginTop": "2px",
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
        "flex": "1",
        "minWidth": "0",
        "margin": "0",
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
    """Assemble progress bar, reaction, and best-round tag inside the blue card."""
    children: list[Any] = []

    children.append(html.Div(
        [
            html.Div(
                t("ui.common.round_of", locale=locale, current=current_round, total=max_rounds),
                id='ending-round-info',
                disable_n_clicks=True,
                style={
                    'textAlign': 'left',
                    'marginBottom': '0',
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#2c5282',
                    'whiteSpace': 'nowrap',
                    'flexShrink': '0',
                },
            ),
            _build_progress_bar(current_round, max_rounds, min_useful, locale),
        ],
        disable_n_clicks=True,
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'alignItems': 'flex-start',
            'gap': '12px',
            'width': '100%',
            'marginBottom': '4px',
        },
    ))

    reaction = _pick_reaction(mae, current_round, locale)
    personal_best = _is_personal_best(mae, rounds)
    celebrate_class = "ending-celebrate" if is_last_round else ""

    reaction_parts: list[Any] = []
    if reaction:
        reaction_parts.append(html.Span(reaction, id="ending-reaction-text"))
    if personal_best:
        if reaction_parts:
            reaction_parts.append("  ")
        reaction_parts.append(html.Span(
            t("ui.ending.personal_best", locale=locale),
            id="ending-personal-best",
            className=celebrate_class,
            style={
                "fontWeight": "bold",
                "color": "#b8860b",
                "backgroundColor": "#fff8e1",
                "padding": "4px 12px" if is_last_round else "2px 10px",
                "borderRadius": "12px",
                "border": "1px solid #f0d060",
                "fontSize": "18px" if is_last_round else "13px",
            },
        ))
    if not reaction_parts:
        reaction_parts.append(html.Span("", id="ending-reaction-text"))
        reaction_parts.append(html.Span("", id="ending-personal-best"))

    children.append(html.Div(
        reaction_parts,
        id="ending-reaction-line",
        className=celebrate_class,
        disable_n_clicks=True,
        style={
            "textAlign": "center",
            "fontSize": "24px" if is_last_round else "14px",
            "color": "#2c5282",
            "fontWeight": "700" if is_last_round else "500",
            "marginBottom": "0",
            "minHeight": "0",
            "lineHeight": "1.35",
        },
    ))

    milestone = _pick_milestone(current_round, max_rounds, min_useful, locale)
    children.append(html.Div(
        milestone or "",
        id="ending-milestone",
        className=celebrate_class,
        disable_n_clicks=True,
        style={
            "textAlign": "center",
            "fontSize": "22px" if is_last_round else "13px",
            "color": "#1b5e20",
            "fontWeight": "800" if is_last_round else "700",
            "marginTop": "6px" if is_last_round else "2px",
            "minHeight": "0",
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
            'fontSize': '12px',
            'fontStyle': 'italic',
            'marginTop': '2px',
            'lineHeight': '1.3',
            'display': 'none',
        }
    ))

    return html.Div(
        children,
        id="ending-gamification",
        className="ending-gamification-complete" if is_last_round else "",
        disable_n_clicks=True,
        style={
            "maxWidth": "100%",
            "margin": "0 auto",
            "padding": "12px 16px 10px 16px" if is_last_round else "8px 14px 6px 14px",
            "backgroundColor": "#fff8e1" if is_last_round else "#f0f7ff",
            "borderRadius": "10px",
            "border": "1px solid #f0d060" if is_last_round else "1px solid #c5d9f0",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
            "width": "100%",
            "boxSizing": "border-box",
            "flexShrink": "0",
        },
    )


def _switch_format_button(
    format_code: str,
    *,
    locale: str,
    visible: bool,
    short: bool,
) -> html.Button:
    """Format-switch CTA. `short` is the compact last-round submit-row label."""
    code = str(format_code).strip().upper()
    long_key = f"ui.switch_format.try_{code.lower()}"
    label_key = f"{long_key}_short" if short else long_key
    return html.Button(
        t(label_key, locale=locale),
        id=f"switch-format-{code.lower()}",
        className="ui blue button ending-switch-format-btn",
        title=t(long_key, locale=locale),
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
            "lineHeight": "1.2",
            "margin": "0",
            "whiteSpace": "nowrap",
            "flex": "1 1 auto",
            "maxWidth": "280px",
        },
    )


def create_ending_layout(
    current_df_data: Optional[Dict],
    events_df_data: Optional[Dict],
    user_info: Optional[Dict] = None,
    glucose_unit: Optional[str] = None,
    *,
    locale: str,
) -> html.Div:
    """Create the ending page layout"""
    print("DEBUG: Creating ending page with stored data")

    # The full dataset is NOT shipped to the client. Load it server-side lazily,
    # only as a fallback window source when current-window-df is absent.
    _full_df_cache: list[pl.DataFrame] = []
    def _ending_full_df() -> pl.DataFrame:
        if not _full_df_cache:
            glucose_df, _ = load_dataset(resolve_dataset_identity(user_info))
            _full_df_cache.append(glucose_df)
        return _full_df_cache[0]

    # Events for the markers: prefer the small events store; else load from the
    # dataset server-side (the chart filters events to the window anyway).
    if events_df_data:
        events_df = reconstruct_events_dataframe_from_dict(events_df_data)
    elif user_info:
        _, events_df = load_dataset(resolve_dataset_identity(user_info))
    else:
        events_df = pl.DataFrame(
            {'time': [], 'event_type': [], 'event_subtype': [], 'insulin_value': []}
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
            full_df = _ending_full_df()
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
            df = _ending_full_df().slice(0, DEFAULT_POINTS)
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
    metrics_display = MetricsComponent.create_ending_metrics_display(
        stored_metrics, locale=locale, include_title=False,
    ) if stored_metrics else [
        html.Div(
            t("ui.metrics.no_metrics_available", locale=locale),
            style={
                'color': 'gray',
                'fontStyle': 'italic',
                'fontSize': '16px',
                'padding': '10px',
                'textAlign': 'center',
            },
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

    source_plaque = _ending_source_plaque_children(
        user_info=user_info,
        window_df=df,
        events_df=events_df,
        locale=locale,
    )

    _fold_box_style: dict[str, Any] = {
        'marginBottom': '20px',
        'padding': 'clamp(10px, 2vw, 20px)',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'width': '100%',
        'boxSizing': 'border-box',
    }
    _fold_title_style: dict[str, Any] = {
        'textAlign': 'center',
        'marginBottom': '12px',
        'fontSize': 'clamp(18px, 3vw, 24px)',
    }
    _details_button_label = t("ui.ending.click_here_for_details", locale=locale)

    _finish_label = (
        t("ui.ending.results", locale=locale)
        if is_last_round
        else t("ui.submit.finish_game", locale=locale)
    )
    _finish_class = (
        "ui huge green button finish-study-exit finish-study-results"
        if is_last_round
        else FINISH_EXIT_BUTTON_CLASS
    )
    _finish_style: dict[str, Any] = {
        'backgroundColor': '#4CBB17',
        'color': 'white',
        'padding': '0 48px',
        'border': 'none',
        'borderRadius': '14px',
        'cursor': 'pointer',
        'width': 'auto',
        'minWidth': '320px',
        'height': '80px',
        'display': 'inline-flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'lineHeight': '1.2',
        'margin': '0',
        'flexShrink': '0',
        'fontSize': '32px',
        'fontWeight': '800',
    } if is_last_round else finish_exit_button_style()
    return html.Div([
        html.H1(
            t("ui.ending.title", locale=locale),
            id='ending-title',
            disable_n_clicks=True,
            **{"aria-hidden": "true"},
            style={'display': 'none'},
        ),
        html.Div([
        _build_gamification_section(
            current_round=current_round_number,
            max_rounds=max_rounds,
            min_useful=min_useful,
            mae=current_mae,
            rounds=all_rounds,
            locale=locale,
            is_last_round=is_last_round,
        ),
        html.Div([
            html.Div(
                id='ending-glucose-chart-container',
                className='glucose-chart-shell',
                children=[
                    html.Div(
                        meal_food_bubble_children(
                            df,
                            events_df,
                            source_name=str(user_info.get('data_source_name') or '') if user_info else '',
                            locale=locale,
                        ),
                        id='ending-food-bubbles',
                        className='meal-food-bubble-strip',
                        disable_n_clicks=True,
                    ),
                    dcc.Graph(
                        id='ending-static-graph',
                        # Same compact figure + resize contract as /prediction
                        # (GlucoseChart._COMPACT_MARGIN, assets/compact-chart.js).
                        figure=GlucoseChart.build_static_figure(
                            df,
                            events_df,
                            str(user_info.get('data_source_name') or '') if user_info else None,
                            unit=unit,
                            locale=locale,
                            prediction_boundary=len(df) - PREDICTION_HOUR_OFFSET,
                            compact=_is_mobile_request(),
                        ),
                        config={
                            'displayModeBar': False,
                            'scrollZoom': False,
                            'doubleClick': 'reset',
                            'showAxisDragHandles': False,
                            'displaylogo': False,
                            'editable': False,
                        },
                        style={'height': '100%', 'flex': '1', 'minHeight': '0'},
                        responsive=True,
                    ),
                ],
                disable_n_clicks=True,
                style={'height': '100%', 'flex': '1', 'minHeight': '0', 'display': 'flex', 'flexDirection': 'column'},
            ),
            html.P(
                t("ui.ending.graph_explanation", locale=locale),
                id='ending-graph-explanation',
                style={
                    'textAlign': 'center',
                    'color': '#4a5568',
                    'fontSize': '12px',
                    'margin': '0',
                    'fontStyle': 'italic',
                    'lineHeight': '1.3',
                    'display': 'none',
                },
            ),
        ], id='ending-graph-card', disable_n_clicks=True, style={
            'marginBottom': '0',
            'padding': '0',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '100%',
            'boxSizing': 'border-box',
            'flex': '1 1 auto',
            'minHeight': '0',
            'display': 'flex',
            'flexDirection': 'column',
        }),
        html.Div(
            [
                html.Button(
                    _finish_label,
                    id='finish-study-button-ending',
                    className=_finish_class,
                    autoFocus=False,
                    title=_finish_label,
                    style=_finish_style,
                ),
                html.Button(
                    t("ui.ending.next_round", locale=locale),
                    id='next-round-button',
                    className="ui green button",
                    disabled=is_last_round,
                    style={
                        'backgroundColor': '#4CBB17' if not is_last_round else '#cccccc',
                        'color': 'white' if not is_last_round else '#666666',
                        'padding': '12px 24px',
                        'border': 'none',
                        'borderRadius': '8px',
                        'fontSize': '22px',
                        'cursor': 'pointer' if not is_last_round else 'not-allowed',
                        'width': '300px',
                        'height': '60px',
                        'display': 'none' if is_last_round else 'inline-flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'lineHeight': '1.2',
                        'margin': '0',
                    }
                ),
                _switch_format_button(
                    "A", locale=locale, short=True,
                    visible=False,
                ),
                _switch_format_button(
                    "B", locale=locale, short=True,
                    visible=False,
                ),
                _switch_format_button(
                    "C", locale=locale, short=True,
                    visible=False,
                ),
            ],
            id='ending-submit-row',
            className='ending-submit-row-last' if is_last_round else '',
            disable_n_clicks=True,
            style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'alignItems': 'center',
                'flexWrap': 'nowrap',
                'gap': '10px',
                'marginTop': '4px',
                'padding': '0 10px',
                'flexShrink': '0',
            },
        ),
        html.Div(
            source_plaque,
            id='ending-source-info',
            className='prediction-source-plaque',
            disable_n_clicks=True,
        ),
        html.Div(id="switch-format-error", disable_n_clicks=True, style={'margin': '0'}),
        dcc.Checklist(
            id="switch-data-usage-consent",
            options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
            value=switch_data_consent_value,
            style={'display': 'none'},
        ),
        html.H3(
            t("ui.switch_format.title", locale=locale),
            id='ending-switch-format-title',
            disable_n_clicks=True,
            style={'display': 'none'},
        ),
        ], id='ending-primary', disable_n_clicks=True),
        html.Div(
            html.Button(
                t("ui.resume_code.copy_link", locale=locale),
                id='ending-copy-link-button',
                type='button',
                **{"data-copied-text": t("ui.resume_code.copied", locale=locale)},
                style={
                    'backgroundColor': '#ffffff',
                    'color': '#2185d0',
                    'padding': '10px 22px',
                    'border': '1px solid #2185d0',
                    'borderRadius': '8px',
                    'fontSize': '16px',
                    'fontWeight': '700',
                    'cursor': 'pointer',
                    'width': 'auto',
                    'maxWidth': '100%',
                    'boxSizing': 'border-box',
                    'display': 'inline-flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'textAlign': 'center',
                    'lineHeight': '1.3',
                },
            ),
            id='ending-copy-link-row',
            disable_n_clicks=True,
            style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '4px', 'padding': '0 10px', 'flexShrink': '0'},
        ),
        html.Div(
            t("ui.ending.local_storage_note", locale=locale),
            id='ending-local-storage-note',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'margin': '6px 0 0 0',
                'color': '#2d6a4f',
                'fontSize': '13px',
                'fontStyle': 'italic',
                'display': 'block' if STORAGE_TYPE == 'local' else 'none',
            }
        ),
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
                'margin': '12px auto 8px auto',
                'fontSize': '14px',
                'lineHeight': '1.4',
            },
        ),
        html.Div(
            t("ui.ending.units_line", locale=locale, unit=unit),
            id='ending-units-line',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '8px',
                'color': '#4a5568',
                'fontSize': '14px',
            },
        ),
        html.Div(
            [
                html.H3(
                    t("ui.ending.prediction_results", locale=locale),
                    id='ending-prediction-results-title',
                    style=_fold_title_style,
                ),
                html.Details(
                    [
                        html.Summary(
                            _details_button_label,
                            id='ending-prediction-details-toggle',
                            className='ending-fold-button',
                        ),
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
                                'marginTop': '12px',
                            },
                            highlight_first_two_rows=True,
                        ),
                    ],
                    className='ending-fold',
                ),
            ],
            disable_n_clicks=True,
            style={**_fold_box_style, 'overflowX': 'auto'},
        ),
        html.Div(
            [
                html.H3(
                    t("ui.metrics.title_accuracy_metrics", locale=locale),
                    id='ending-metrics-summary',
                    style=_fold_title_style,
                ),
                html.Details(
                    [
                        html.Summary(
                            _details_button_label,
                            id='ending-metrics-details-toggle',
                            className='ending-fold-button',
                        ),
                        html.Div(
                            metrics_display,
                            id='ending-metrics-container',
                            disable_n_clicks=True,
                            style={'marginTop': '8px'},
                        ),
                    ],
                    className='ending-fold',
                ),
            ],
            disable_n_clicks=True,
            style=_fold_box_style,
        ),
        dcc.Store(id='finish-confirm-context-ending', data=None, storage_type='memory'),
        finish_confirm_overlay(locale, source="ending"),
    ], id="ending-page", className="ending-page", disable_n_clicks=True, style={
        'maxWidth': '100%',
        'width': '100%',
        'margin': '0 auto',
        'padding': '0 20px 16px 20px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '8px',
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


def _format_mae_for_unit(mae_mgdl: float, *, unit: str) -> str:
    """Format an MAE stored in mg/dL for the active display unit."""
    value = mae_mgdl / 18.0 if unit == "mmol/L" else mae_mgdl
    return f"{value:.2f}"


def _leaderboard_hero_children(
    overall: Optional[dict[str, Any]],
    *,
    locale: str,
    unit: str,
) -> list[Any]:
    """"Your place" card contents for a leaderboard snapshot.

    Shared by ``/final`` and ``/highscore`` so both read identically; returns an
    empty list when there is no snapshot at all (nothing to show yet).
    """
    from sugar_sugar.components.share import compute_percentile

    hero_inner: list[Any] = []
    if overall and overall.get("rank") is not None and overall.get("total"):
        rank = int(overall["rank"])
        total = int(overall["total"])
        pct = compute_percentile(rank, total)
        hero_inner.extend(
            [
                html.Div(
                    t("ui.final.your_place", locale=locale),
                    className="final-leaderboard-hero-label",
                    disable_n_clicks=True,
                ),
                html.Div(
                    t("ui.final.your_place_value", locale=locale, rank=rank, total=total),
                    className="final-leaderboard-hero-rank",
                    disable_n_clicks=True,
                ),
            ]
        )
        if pct is not None:
            hero_inner.append(
                html.Div(
                    t("ui.final.top_percentile", locale=locale, pct=pct),
                    className="final-leaderboard-hero-pct",
                    disable_n_clicks=True,
                )
            )
        if overall.get("mae") is not None:
            hero_inner.append(
                html.Div(
                    t(
                        "ui.final.your_mae",
                        locale=locale,
                        mae=_format_mae_for_unit(float(overall["mae"]), unit=unit),
                        unit=unit,
                    ),
                    className="final-leaderboard-hero-mae",
                    disable_n_clicks=True,
                )
            )
    elif overall and overall.get("total"):
        hero_inner.append(
            html.Div(
                t("ui.final.no_ranking_yet", locale=locale, min=MIN_USEFUL_ROUNDS),
                className="final-leaderboard-empty",
                disable_n_clicks=True,
            )
        )
    return hero_inner


def _leaderboard_board(
    entries: list[dict[str, Any]],
    *,
    title: str,
    locale: str,
    unit: str,
    subtitle: Optional[str] = None,
) -> Optional[html.Div]:
    """Anonymous ``# / Player / MAE`` table for a leaderboard snapshot's ``top``.

    Returns ``None`` when there are no entries, so callers can simply drop the
    board.  Shared by ``/final`` and ``/highscore``.
    """
    rows: list[Any] = []
    for entry in entries:
        is_you = bool(entry.get("is_you"))
        # Optional nickname replaces the anonymous rank-derived label. Slots set
        # anonymously keep "Player N" exactly as before.
        nickname = normalize_nickname(entry.get("nickname"))
        if is_you:
            player_label = (
                t("ui.final.you_named", locale=locale, name=nickname)
                if nickname
                else t("ui.final.you", locale=locale)
            )
        else:
            player_label = nickname or t("ui.final.player_n", locale=locale, n=int(entry["rank"]))
        rounds = entry.get("rounds")
        rows.append(
            html.Div(
                [
                    html.Span(str(int(entry["rank"])), className="final-leaderboard-cell rank"),
                    html.Span(
                        player_label,
                        className=(
                            "final-leaderboard-cell player you-label"
                            if is_you
                            else "final-leaderboard-cell player"
                        ),
                    ),
                    # Arcade boards keep every score, so one player can hold several
                    # slots; the round count is what tells those slots apart.
                    html.Span(
                        "-" if rounds is None else str(int(rounds)),
                        className="final-leaderboard-cell rounds",
                    ),
                    html.Span(
                        _format_mae_for_unit(float(entry["mae"]), unit=unit),
                        className="final-leaderboard-cell mae",
                    ),
                ],
                className="final-leaderboard-row you" if is_you else "final-leaderboard-row",
                disable_n_clicks=True,
            )
        )
    if not rows:
        return None

    children: list[Any] = [
        html.Div(title, className="final-leaderboard-board-title", disable_n_clicks=True),
    ]
    if subtitle:
        children.append(
            html.Div(subtitle, className="final-leaderboard-board-subtitle", disable_n_clicks=True)
        )
    children.extend(
        [
            html.Div(
                [
                    html.Span(t("ui.final.col_rank", locale=locale), className="final-leaderboard-cell rank"),
                    html.Span(t("ui.final.col_player", locale=locale), className="final-leaderboard-cell player"),
                    html.Span(
                        t("ui.final.col_rounds", locale=locale),
                        className="final-leaderboard-cell rounds",
                    ),
                    html.Span(
                        t("ui.final.col_mae", locale=locale, unit=unit),
                        className="final-leaderboard-cell mae",
                    ),
                ],
                className="final-leaderboard-row head",
                disable_n_clicks=True,
            ),
            html.Div(rows, className="final-leaderboard-rows", disable_n_clicks=True),
        ]
    )
    return html.Div(children, className="final-leaderboard-board", disable_n_clicks=True)


def _nickname_editor_children(
    user_info: Optional[Dict[str, Any]],
    *,
    locale: str,
) -> list[Any]:
    """The "your name on the leaderboard" box shown on ``/final``.

    Only mounted when the player has board presence *and* did not already type
    a nickname at startup.  Deliberately absent from the public ``/highscore``
    page, which is session-free.
    Its ids are ``final-nickname-*`` rather than the ``/startup`` ``nickname-input``:
    a Dash callback only fires when *every* one of its components is in the layout,
    so reusing the id would drag the startup validation callback onto ``/final`` and
    crash on its missing Outputs.

    The value is a *suggestion* -- the player's own nickname when they have one,
    otherwise the last name recorded against their hashed email identity, so someone
    returning on a new device does not have to retype it.
    """
    info: Dict[str, Any] = user_info or {}
    study_id: str = str(info.get('study_id') or '')
    suggestion: str = normalize_nickname(info.get('nickname')) or stored_nickname(
        study_id=study_id, key=email_key(info.get('email'))
    )
    return [
        html.Div(
            t("ui.final.nickname_title", locale=locale),
            className="final-nickname-title",
            disable_n_clicks=True,
        ),
        html.Div(
            [
                dcc.Input(
                    id='final-nickname-input',
                    type='text',
                    value=suggestion,
                    maxLength=MAX_NICKNAME_LENGTH,
                    placeholder=t("ui.startup.nickname_placeholder", locale=locale),
                    # No persistence: a persisted client value would override the
                    # server-rendered suggestion and defeat the cross-device prefill.
                    className="final-nickname-input",
                ),
                html.Button(
                    t("ui.final.nickname_save", locale=locale),
                    id='final-nickname-save',
                    className="ui small green button final-nickname-save",
                ),
            ],
            className="final-nickname-row",
            disable_n_clicks=True,
        ),
        html.Div(
            t("ui.final.nickname_hint", locale=locale),
            className="final-nickname-hint",
            disable_n_clicks=True,
        ),
        html.Div(
            "",
            id='final-nickname-status',
            className="final-nickname-status",
            disable_n_clicks=True,
        ),
    ]


def _final_leaderboard_children(
    *,
    overall: Optional[dict[str, Any]],
    per_format: list[tuple[str, dict[str, Any]]],
    locale: str,
    unit: str,
    user_info: Optional[Dict[str, Any]] = None,
    offer_nickname: Optional[bool] = None,
) -> list[Any]:
    """Inner children of the ``final-ranking-list`` wrapper.

    Split out of :func:`_build_final_leaderboard` so ``save_final_nickname`` can
    re-render the board in place after the player renames themselves.

    The nickname box is skipped when they already typed one at startup
    (``user_info['nickname']``).  ``save_final_nickname`` passes
    ``offer_nickname=True`` so the just-saved editor (and its Dash ids) stay
    mounted for the status line.
    """
    hero_inner: list[Any] = _leaderboard_hero_children(overall, locale=locale, unit=unit)
    if not hero_inner:
        hero_inner = [
            html.Div(
                t("ui.final.no_ranking_yet", locale=locale, min=MIN_USEFUL_ROUNDS),
                className="final-leaderboard-empty",
                disable_n_clicks=True,
            )
        ]

    left_children: list[Any] = []
    if hero_inner:
        left_children.append(
            html.Div(hero_inner, className="final-leaderboard-hero", disable_n_clicks=True)
        )

    format_chips: list[Any] = []
    for fmt, board in per_format:
        if not board or board.get("rank") is None:
            continue
        format_chips.append(
            html.Div(
                t(
                    "ui.final.ranking_format_line",
                    locale=locale,
                    format=_format_label(fmt, locale=locale),
                    rank=int(board["rank"]),
                    total=int(board["total"]),
                ),
                className="final-leaderboard-format-chip",
                disable_n_clicks=True,
            )
        )
    if format_chips:
        left_children.append(
            html.Div(
                [
                    html.Div(
                        t("ui.final.format_ranks", locale=locale),
                        className="final-leaderboard-format-label",
                        disable_n_clicks=True,
                    ),
                    html.Div(format_chips, className="final-leaderboard-format-chips", disable_n_clicks=True),
                ],
                className="final-leaderboard-formats",
                disable_n_clicks=True,
            )
        )

    board = _leaderboard_board(
        list((overall or {}).get("top") or []),
        title=t("ui.final.top_predictors", locale=locale),
        locale=locale,
        unit=unit,
    )
    right_children: list[Any] = [board] if board is not None else []

    # Only offer the rename box once the player actually has a board presence --
    # short runs stay off the ranking CSVs, so a nickname would have nothing to
    # label.  Skip it entirely when they already picked a name at startup.
    already_named: bool = bool(normalize_nickname((user_info or {}).get("nickname")))
    show_editor: bool = not already_named if offer_nickname is None else offer_nickname
    has_rank: bool = bool(overall and overall.get("rank") is not None) or any(
        board and board.get("rank") is not None for _, board in per_format
    )
    if (left_children or right_children) and show_editor and has_rank:
        left_children.append(
            html.Div(
                _nickname_editor_children(user_info, locale=locale),
                className="final-nickname-editor",
                disable_n_clicks=True,
            )
        )

    # Two columns (inline styles so layout does not depend on asset cache):
    # left  = your placement / #rank / top% / MAE / data-source ranks + rename box
    # right = full player list (nicknames or anonymous Player N + highlighted You)
    split: Optional[html.Div] = None
    if left_children or right_children:
        split = html.Div(
            [
                html.Div(
                    left_children,
                    className="final-leaderboard-left",
                    disable_n_clicks=True,
                    style={
                        "flex": "0 1 380px",
                        "minWidth": "240px",
                        "maxWidth": "420px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                ),
                html.Div(
                    right_children,
                    className="final-leaderboard-right",
                    disable_n_clicks=True,
                    style={"flex": "1 1 320px", "minWidth": "260px"},
                ),
            ],
            className="final-leaderboard-split",
            disable_n_clicks=True,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "22px",
                "alignItems": "flex-start",
                "justifyContent": "center",
            },
        )

    children: list[Any] = [
        html.H3(
            t("ui.final.ranking_title", locale=locale),
            id="final-ranking-title",
            className="final-leaderboard-title",
        ),
    ]
    if split is not None:
        children.append(split)
    return children


def _build_final_leaderboard(
    *,
    overall: Optional[dict[str, Any]],
    per_format: list[tuple[str, dict[str, Any]]],
    locale: str,
    unit: str,
    user_info: Optional[Dict[str, Any]] = None,
) -> html.Div:
    """Leaderboard: previous card look on the left, top table on the right.

    The sole place the ``final-ranking-list`` id and its visibility style are set --
    ``save_final_nickname`` swaps only this wrapper's ``children``.
    """
    children = _final_leaderboard_children(
        overall=overall,
        per_format=per_format,
        locale=locale,
        unit=unit,
        user_info=user_info,
    )
    # children[0] is always the title; anything beyond it is real content.
    return html.Div(
        children,
        id="final-ranking-list",
        className="final-leaderboard",
        disable_n_clicks=True,
        style={"display": "block" if len(children) > 1 else "none"},
    )


def _final_ranking_snapshots(
    user_info: Dict[str, Any],
) -> tuple[Optional[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Read ranking CSVs for the formats this player actually ran."""
    rounds: list[dict[str, Any]] = user_info.get("rounds") or []
    current_format = str(user_info.get("format") or "A")
    runs_by_format: dict[str, list[dict[str, Any]]] = dict(user_info.get("runs_by_format") or {})
    already_played: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
    if rounds:
        already_played.add(current_format)
    played_formats: list[str] = sorted(already_played, key=lambda x: FORMAT_ORDER.get(str(x), 999))
    study_id = str(user_info.get("study_id") or "")
    leaderboard_key = email_key(user_info.get("email"))
    per_format_boards: list[tuple[str, dict[str, Any]]] = []
    for fmt in played_formats:
        if fmt not in ("A", "B", "C"):
            continue
        board = _leaderboard_snapshot(
            project_root / "data" / "input" / f"prediction_ranking_{fmt}.csv",
            study_id=study_id,
            key=leaderboard_key,
            format_filter=fmt,
        )
        if board is not None:
            per_format_boards.append((fmt, board))
    overall_board = _leaderboard_snapshot(
        project_root / "data" / "input" / "prediction_ranking.csv",
        study_id=study_id,
        key=leaderboard_key,
        format_filter="ALL",
    )
    return overall_board, per_format_boards


def _final_share_section_children(
    user_info: Dict[str, Any],
    *,
    locale: str,
) -> list[Any]:
    """Persist the share record and return the /final share panel children."""
    share_record: Optional[dict[str, Any]] = build_final_share_record(user_info, locale=locale)
    if share_record is None:
        return []
    final_share_id: str = share_store.ensure_share(share_record)
    return [
        html.H3(
            t("ui.share.button_share", locale=locale),
            style={
                "textAlign": "center",
                "marginBottom": "4px",
                "fontSize": "clamp(18px, 3vw, 24px)",
            },
        ),
        build_share_panel(
            share_record,
            share_id=final_share_id,
            share_url=_build_share_url(final_share_id),
            locale=locale,
        ),
    ]


def _final_synthesis_inner(
    user_info: Dict[str, Any],
    *,
    locale: str,
) -> list[Any]:
    """Inner nodes of the synthesis card (the wrapper already has the id)."""
    synthesis_rounds: list[dict[str, Any]] = collect_playable_rounds(user_info)
    if not synthesis_rounds:
        return []
    card = build_synthesis_card(
        {"rounds": synthesis_rounds},
        locale=locale,
        graph_id="final-synthesis-graph",
    )
    kids = card.children
    if isinstance(kids, (list, tuple)):
        return list(kids)
    if kids is None:
        return []
    return [kids]


HIGHSCORE_TOP_N: int = 20
HIGHSCORE_FORMAT_TOP_N: int = 10


def create_highscore_page(
    user_info: Optional[Dict[str, Any]],
    glucose_unit: Optional[str],
    *,
    locale: str,
) -> html.Div:
    """Public highscore page reachable from the navbar (desktop) / burger menu (mobile).

    Reads the same ranking CSVs as ``/final`` (``data/input/prediction_ranking*.csv``),
    so it renders for a first-time visitor with no session at all.  When a session
    exists, the visitor's own row is highlighted and a "your place" hero is shown.
    Players who picked a nickname are shown by it; the rest stay anonymous
    (``Player N``) exactly as on ``/final``.
    """
    from sugar_sugar.components.landing import (
        count_people_who_accessed,
        count_people_who_completed,
    )

    input_dir: Path = project_root / "data" / "input"
    accessed: int = count_people_who_accessed(input_dir / "prediction_statistics.csv")
    completed: int = count_people_who_completed(
        stats_path=input_dir / "prediction_statistics.csv",
        ranking_paths=(
            input_dir / "prediction_ranking.csv",
            input_dir / "prediction_ranking_A.csv",
            input_dir / "prediction_ranking_B.csv",
            input_dir / "prediction_ranking_C.csv",
        ),
    )

    unit: str = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
    study_id: str = str((user_info or {}).get('study_id') or '')
    key: str = email_key((user_info or {}).get('email'))

    overall: Optional[dict[str, Any]] = _leaderboard_snapshot(
        project_root / 'data' / 'input' / 'prediction_ranking.csv',
        study_id=study_id,
        key=key,
        format_filter="ALL",
        top_n=HIGHSCORE_TOP_N,
    )
    per_format: list[tuple[str, dict[str, Any]]] = []
    for fmt in ("A", "B", "C"):
        board = _leaderboard_snapshot(
            project_root / 'data' / 'input' / f'prediction_ranking_{fmt}.csv',
            study_id=study_id,
            key=key,
            format_filter=fmt,
            top_n=HIGHSCORE_FORMAT_TOP_N,
        )
        if board is not None:
            per_format.append((fmt, board))

    children: list[Any] = [
        html.H1(t("ui.highscore.title", locale=locale), disable_n_clicks=True),
        html.Div(
            t("ui.highscore.subtitle", locale=locale, min=MIN_USEFUL_ROUNDS),
            className="highscore-subtitle",
            disable_n_clicks=True,
        ),
        html.Div(
            [
                html.Div(
                    t("ui.highscore.games_played", locale=locale, count=accessed),
                    className="highscore-stat",
                    disable_n_clicks=True,
                ),
                html.Div(
                    t(
                        "ui.highscore.completed_the_task",
                        locale=locale,
                        count=completed,
                    ),
                    className="highscore-stat",
                    disable_n_clicks=True,
                ),
                # `total` counts board slots (one per finished game); `players` is
                # the distinct-people count among ranked completers.
                html.Div(
                    t("ui.highscore.players_count", locale=locale, total=int((overall or {}).get("players") or 0)),
                    className="highscore-stat",
                    disable_n_clicks=True,
                ),
            ],
            className="highscore-stats",
            disable_n_clicks=True,
        ),
    ]

    hero_inner: list[Any] = _leaderboard_hero_children(overall, locale=locale, unit=unit)
    if hero_inner:
        children.append(
            html.Div(
                html.Div(hero_inner, className="final-leaderboard-hero", disable_n_clicks=True),
                className="highscore-hero-wrap",
                disable_n_clicks=True,
            )
        )

    overall_board = _leaderboard_board(
        list((overall or {}).get("top") or []),
        title=t("ui.highscore.overall_board", locale=locale),
        subtitle=t("ui.highscore.lower_is_better", locale=locale),
        locale=locale,
        unit=unit,
    )
    if overall_board is not None:
        children.append(
            html.Div(
                overall_board,
                className="final-leaderboard highscore-card",
                disable_n_clicks=True,
            )
        )

    format_cards: list[Any] = []
    for fmt, board in per_format:
        card = _leaderboard_board(
            list(board.get("top") or []),
            title=_format_label(fmt, locale=locale),
            # One slot per finished game on this source, so count scores not players.
            subtitle=t("ui.highscore.scores_count", locale=locale, total=int(board.get("total") or 0)),
            locale=locale,
            unit=unit,
        )
        if card is not None:
            format_cards.append(
                html.Div(card, className="final-leaderboard highscore-card", disable_n_clicks=True)
            )
    if format_cards:
        children.append(
            html.Div(
                [
                    html.H3(
                        t("ui.highscore.by_data_source", locale=locale),
                        className="highscore-section-title",
                        disable_n_clicks=True,
                    ),
                    html.Div(format_cards, className="highscore-format-grid", disable_n_clicks=True),
                ],
                className="highscore-formats",
                disable_n_clicks=True,
            )
        )

    if overall_board is None and not format_cards:
        children.append(
            html.Div(
                t("ui.highscore.empty", locale=locale, min=MIN_USEFUL_ROUNDS),
                className="final-leaderboard-empty highscore-empty",
                disable_n_clicks=True,
            )
        )

    children.append(
        html.Div(
            dcc.Link(
                t("ui.highscore.play_now", locale=locale),
                href="/",
                className="ui blue button highscore-play-button",
            ),
            className="highscore-actions",
            disable_n_clicks=True,
        )
    )

    # Spells out what this board does and does not keep: an optional public
    # nickname plus a one-way hash of the email (only if one was given), never the
    # address and never anything from the study record.
    children.append(
        html.Div(
            t("ui.highscore.privacy_note", locale=locale),
            className="highscore-privacy-note",
            disable_n_clicks=True,
        )
    )

    return html.Div(children, className="info-page highscore-page", disable_n_clicks=True)


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


def build_final_share_record(user_info: Dict[str, Any], *, locale: str) -> Optional[dict[str, Any]]:
    """JSON-safe share record for the player's full game state, or None without rounds.

    Captures every round across every format (current run + archived
    `runs_by_format`), the frozen rankings, and a trimmed `user_info`.  The
    record intentionally drops heavyweight stores (`full-df`, `events-df`) --
    everything the share page needs already lives in `prediction_table_data`.
    """
    all_rounds: list[dict[str, Any]] = collect_playable_rounds(user_info)
    if not all_rounds:
        return None
    played_formats: set[str] = {str(r.get("format") or "") for r in all_rounds}
    played_formats.discard("")
    study_id: str = str(user_info.get("study_id") or "")
    rankings: dict[str, Any] = compute_share_rankings(
        study_id, sorted(played_formats), key=email_key(user_info.get("email"))
    )
    return {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "locale": normalize_locale(locale),
        "rounds": all_rounds,
        "played_formats": sorted(played_formats, key=lambda x: FORMAT_ORDER.get(str(x), 999)),
        "rankings": rankings,
        "user_info": {
            "name": str(user_info.get("name") or ""),
            "nickname": normalize_nickname(user_info.get("nickname")),
            "study_id": study_id,
            "format": str(user_info.get("format") or ""),
            "uses_cgm": bool(user_info.get("uses_cgm", False)),
            "max_rounds": int(user_info.get("max_rounds") or MAX_ROUNDS),
            "challenge_unknown": bool(user_info.get("challenge_unknown", False)),
            "challenge_unknown_pct": user_info.get("challenge_unknown_pct", ""),
        },
    }


def create_final_layout(
    user_info: Dict[str, Any],
    glucose_unit: Optional[str],
    *,
    locale: str,
    eager: bool = True,
) -> html.Div:
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

    if eager:
        overall_board, per_format_boards = _final_ranking_snapshots(user_info)
        leaderboard = _build_final_leaderboard(
            overall=overall_board,
            per_format=per_format_boards,
            locale=locale,
            unit=unit,
            user_info=user_info,
        )
    else:
        leaderboard = html.Div(
            [
                html.H3(
                    t("ui.final.ranking_title", locale=locale),
                    id="final-ranking-title",
                    className="final-leaderboard-title",
                ),
            ],
            id="final-ranking-list",
            className="final-leaderboard",
            disable_n_clicks=True,
            style={"display": "block"},
        )

    metrics_component_final = MetricsComponent()
    aggregate_table_data = _convert_table_data_units(_build_aggregate_table_data(rounds), unit)
    overall_metrics = metrics_component_final._calculate_metrics_from_table_data(aggregate_table_data)
    # Player-facing cards show only MAE and MAPE, with lay titles and units;
    # MSE/RMSE are statistician detail and live inside the per-round fold.
    overall_metrics_display = MetricsComponent.create_ending_metrics_display(
        overall_metrics,
        locale=locale,
        metrics_subset=['MAE', 'MAPE'],
        unit=unit,
    ) if overall_metrics else [
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

    def _overall_metric(name: str) -> Optional[float]:
        metric = (overall_metrics or {}).get(name) or {}
        value = metric.get('value')
        return float(value) if value is not None else None

    # Plain-language headline: the one number a regular player actually wants,
    # in glucose units, before any acronym appears on the page.
    hero_mae: Optional[float] = _overall_metric('MAE')
    hero_mape: Optional[float] = _overall_metric('MAPE')
    hero_line: Optional[html.Div] = None
    if hero_mae is not None and hero_mape is not None:
        hero_line = html.Div(
            t(
                "ui.final.hero_summary",
                locale=locale,
                mae=f"{hero_mae:.1f}",
                unit=unit,
                mape=f"{hero_mape:.1f}",
            ),
            id='final-hero-summary',
            disable_n_clicks=True,
        )
    stats_mse: Optional[float] = _overall_metric('MSE')
    stats_rmse: Optional[float] = _overall_metric('RMSE')

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

    synthesis_rounds: list[dict[str, Any]] = collect_playable_rounds(user_info)
    if eager:
        synthesis_card = (
            build_synthesis_card(
                {"rounds": synthesis_rounds},
                locale=locale,
                card_id="final-synthesis-card",
                graph_id="final-synthesis-graph",
            )
            if synthesis_rounds
            else None
        )
        share_kids = _final_share_section_children(user_info, locale=locale)
        share_section: Optional[html.Div] = (
            html.Div(
                share_kids,
                id="final-share-panel",
                disable_n_clicks=True,
                style={"width": "100%", "boxSizing": "border-box"},
            )
            if share_kids
            else None
        )
    else:
        # Stable ids so the deferred callback can fill them after first paint.
        # Empty nodes stay hidden via :empty CSS until the tick writes children.
        share_section = html.Div(
            [],
            id="final-share-panel",
            disable_n_clicks=True,
            style={"width": "100%", "boxSizing": "border-box"},
        )
        synthesis_card = html.Div(
            [],
            id="final-synthesis-card",
            className="results-synthesis-card",
            disable_n_clicks=True,
        )

    _exit_label = t("ui.final.start_over", locale=locale)
    # "Want to start another?" is only asked when there is a format left to
    # start; otherwise the sentence would dangle with no "yes" button.
    journey_key: str = "ui.final.journey_title" if switch_targets else "ui.final.journey_title_done"
    return html.Div([
        html.H1(t("ui.final.title", locale=locale), id='final-title', style={
            'textAlign': 'center',
            'marginBottom': '8px',
            'fontSize': 'clamp(24px, 4vw, 48px)',
            'padding': '0 10px'
        }),
        # Medical disclaimer FIRST: for a diabetic audience "do not change
        # medical decisions based on this app" must be seen, so it sits under
        # the title — compact and without a dismiss icon — instead of two
        # screens below the fold.
        html.Div(
            [
                html.P(t("ui.results_disclaimer.line1", locale=locale), id='final-disclaimer-line1', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line2", locale=locale), id='final-disclaimer-line2', style={'margin': '0'}),
                html.P(t("ui.results_disclaimer.line3", locale=locale), id='final-disclaimer-line3', style={'margin': '0'}),
            ],
            id='final-disclaimer',
            className='ui warning message final-disclaimer-compact',
            disable_n_clicks=True,
            style={
                'display': 'block',
                'maxWidth': '900px',
                'margin': '0 auto 14px auto',
                'fontSize': '13px',
                'lineHeight': '1.45',
                'padding': '10px 16px',
                'textAlign': 'center',
            },
        ),
        html.Div(
            t("ui.final.rounds_played", locale=locale, played=len(rounds), total=max_rounds),
            id='final-rounds-played',
            disable_n_clicks=True,
            style={
                'textAlign': 'center',
                'marginBottom': '10px',
                'fontSize': 'clamp(16px, 2.5vw, 22px)',
                'fontWeight': '600',
                'color': '#2c5282'
            }
        ),
        *([hero_line] if hero_line is not None else []),
        leaderboard,
        *([share_section] if share_section is not None else []),
        *([synthesis_card] if synthesis_card is not None else []),
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
        html.Div(
            [
                html.H3(
                    t("ui.final.per_round_metrics", locale=locale),
                    id='final-per-round-title',
                    style={
                        'textAlign': 'center',
                        'marginBottom': '12px',
                        'fontSize': 'clamp(18px, 3vw, 24px)',
                    },
                ),
                html.Details(
                    [
                        html.Summary(
                            [
                                html.Span(
                                    t("ui.ending.click_here_for_details", locale=locale),
                                    className='fold-label-show',
                                ),
                                html.Span(
                                    t("ui.ending.hide_details", locale=locale),
                                    className='fold-label-hide',
                                ),
                            ],
                            id='final-per-round-details-toggle',
                            className='ending-fold-button',
                        ),
                        *([
                            html.Div(
                                t(
                                    "ui.final.stats_note",
                                    locale=locale,
                                    mse=f"{stats_mse:.2f}",
                                    rmse=f"{stats_rmse:.2f}",
                                    unit=unit,
                                ),
                                id='final-stats-note',
                                style={
                                    'textAlign': 'center',
                                    'margin': '12px 0 0 0',
                                    'color': '#64748b',
                                    'fontSize': '13px',
                                },
                            )
                        ] if stats_mse is not None and stats_rmse is not None else []),
                        html.Div(
                            t("ui.ending.units_line", locale=locale, unit=unit),
                            id='final-units-line',
                            style={
                                'textAlign': 'center',
                                'margin': '12px 0 10px 0',
                                'color': '#4a5568',
                                'fontSize': '14px',
                            },
                        ),
                        build_readonly_ag_grid(
                            table_id='final-rounds-table',
                            row_data=round_rows,
                            column_defs=build_readonly_column_defs(
                                [
                                    {'name': t("ui.final.col_round", locale=locale), 'id': 'Round', 'type': 'numeric'},
                                    {'name': t("ui.final.col_points", locale=locale), 'id': 'Pairs', 'type': 'numeric'},
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
                                'marginTop': '8px',
                            },
                        ),
                    ],
                    className='ending-fold',
                ),
            ],
            id='final-per-round-section',
            disable_n_clicks=True,
            style={
                'marginBottom': '20px',
                'padding': 'clamp(10px, 2vw, 20px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'width': '100%',
                'boxSizing': 'border-box',
            },
        ),
        # Actions LAST: you leave after seeing everything, not before. The
        # destructive Exit (wipes the session) is a labelled button, never a
        # bare icon, and never the first interactive element on the page.
        html.Div(
            t(journey_key, locale=locale),
            id='final-journey-title',
            disable_n_clicks=True,
        ),
        html.H3(
            t("ui.switch_format.title", locale=locale),
            id='final-switch-format-title',
            className='final-switch-format-title-visible' if switch_targets else '',
            disable_n_clicks=True,
            style={'display': 'block' if switch_targets else 'none'},
        ),
        html.Div(
            [
                _switch_format_button(
                    "A", locale=locale, short=True, visible="A" in switch_targets,
                ),
                _switch_format_button(
                    "B", locale=locale, short=True, visible="B" in switch_targets,
                ),
                _switch_format_button(
                    "C", locale=locale, short=True, visible="C" in switch_targets,
                ),
                html.Button(
                    _exit_label,
                    id='restart-button',
                    className='ui button final-exit-button',
                    title=_exit_label,
                    style={
                        'backgroundColor': WINDOWS_CLOSE_RED,
                        'color': 'white',
                        'height': '48px',
                        'padding': '0 28px',
                        'border': 'none',
                        'borderRadius': '8px',
                        'fontSize': '17px',
                        'fontWeight': '700',
                        'cursor': 'pointer',
                        'display': 'inline-flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'lineHeight': '1.2',
                        'margin': '0',
                        'flexShrink': '0',
                    },
                ),
            ],
            id='final-action-row',
            disable_n_clicks=True,
            style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'alignItems': 'center',
                'flexWrap': 'nowrap',
                'gap': '10px',
                'margin': '0 auto 16px auto',
                'padding': '0 10px',
                'width': '100%',
                'boxSizing': 'border-box',
            },
        ),
        html.Div(id="switch-format-error", disable_n_clicks=True, style={'margin': '0 0 8px 0'}),
        dcc.Checklist(
            id="switch-data-usage-consent",
            options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
            value=switch_data_consent_value,
            style={'display': 'none'},
        ),
        html.Div(
            "",
            id='final-played-formats',
            disable_n_clicks=True,
            style={'display': 'none'},
        ),
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
    
    reconstructed = {
        'time': pl.Series(events_data['time'], dtype=pl.String).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
        'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
        # Use pre-processed float values
        'insulin_value': pl.Series(insulin_values, dtype=pl.Float64)
    }
    photo_paths = events_data.get('photo_path')
    if photo_paths is not None:
        reconstructed['photo_path'] = pl.Series(
            [str(value or '') for value in photo_paths],
            dtype=pl.String,
        )
    food_notes = events_data.get('food_note')
    if food_notes is not None:
        reconstructed['food_note'] = pl.Series(
            [str(value or '') for value in food_notes],
            dtype=pl.String,
        )
    return pl.DataFrame(reconstructed)

@app.callback(
    [Output('url', 'pathname'),
     Output('user-info-store', 'data'),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True)],
    [Input('start-button', 'n_clicks')],
    [State('nickname-input', 'value'),
     State('email-input', 'value'),
     State('age-input', 'value'),
     State('gender-dropdown', 'value'),
     State('cgm-dropdown', 'value'),
     State('cgm-duration-input', 'value'),
     State('cgm-duration-unit', 'value'),
     State('format-dropdown', 'value'),
     State('data-usage-consent', 'value'),
     State('diabetic-dropdown', 'value'),
     State('diabetic-type-dropdown', 'value'),
     State('diabetes-duration-input', 'value'),
     State('challenge-unknown-check', 'value'),
     State('paper-mention-check', 'value'),
     State('paper-full-name-input', 'value'),
     State('location-input', 'value'),
     State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_start_button(n_clicks: Optional[int], nickname: Optional[str],
                       email: Optional[str], age: Optional[int | float],
                       gender: Optional[str], uses_cgm: Optional[bool], cgm_duration_value: Optional[float],
                       cgm_duration_unit: Optional[str],
                       format_value: Optional[str], data_usage_consent: Optional[list[str]],
                       diabetic: Optional[bool], diabetic_type: Optional[str],
                       diabetes_duration: Optional[float],
                       challenge_unknown: Optional[list[str] | bool],
                       paper_mention_check: Optional[list[str] | bool],
                       paper_full_name: Optional[str],
                       location: Optional[str],
                       existing_user_info: Optional[Dict[str, Any]] = None) -> tuple[Any, ...]:
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
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    from sugar_sugar.components.startup import (
        _wants_contact_from_user_info,
        prior_upload_data_consent,
        stamp_upload_data_consent,
        validate_startup_form,
    )
    from sugar_sugar.cgm_duration import cgm_duration_to_years, normalize_cgm_duration_unit
    from sugar_sugar.challenge_unknown import (
        CHALLENGE_UNKNOWN_PCT,
        challenge_unknown_checked,
        challenge_unknown_eligible,
    )
    from sugar_sugar.paper_mention import resolve_paper_mention

    wants_contact = _wants_contact_from_user_info(existing_user_info)
    already_upload_consent = prior_upload_data_consent(existing_user_info)
    duration_unit = normalize_cgm_duration_unit(cgm_duration_unit)
    validation = validate_startup_form(
        email=email,
        age=age,
        gender=gender,
        format_value=format_value,
        data_usage_consent=data_usage_consent,
        is_diabetic=diabetic,
        diabetic_type=diabetic_type,
        diabetes_duration=diabetes_duration,
        location=location,
        uses_cgm=uses_cgm,
        cgm_duration=cgm_duration_value,
        wants_contact=wants_contact,
        locale=None,
        prior_upload_consent=already_upload_consent,
        cgm_duration_unit=duration_unit,
    )
    _start_idle = (no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)
    if not validation.form_complete:
        return _start_idle

    if age and gender and diabetic is not None and location and format_value:
        from datetime import datetime
        from sugar_sugar.consent import ensure_consent_agreement_row, get_next_study_number

        has_data_consent = (
            bool(data_usage_consent and "agree" in data_usage_consent)
            or already_upload_consent
        )
        info: Dict[str, Any] = dict(existing_user_info or {})
        study_id = info.get('study_id') or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        uses_cgm_bool = bool(uses_cgm) if uses_cgm is not None else False
        challenge_on = (
            challenge_unknown_checked(challenge_unknown)
            and challenge_unknown_eligible(
                {'diabetic': diabetic, 'diabetic_type': diabetic_type, 'format': format_value},
                format_value,
            )
        )
        wants_paper_mention, paper_name = resolve_paper_mention(
            paper_mention_check, paper_full_name
        )

        info.update({
            'study_id': study_id,
            'run_id': run_id,
            # Stable cross-device resume code (server-side savegame key).
            'resume_code': info.get('resume_code') or resume_store.new_code(),
            'email': email or info.get('email') or '',
            # Optional public leaderboard label -- display only, never study data.
            'nickname': normalize_nickname(nickname) or info.get('nickname') or '',
            'age': age,
            'gender': gender,
            'uses_cgm': uses_cgm_bool,
            'cgm_duration': (
                [cgm_duration_value, duration_unit]
                if uses_cgm_bool and cgm_duration_value is not None
                else None
            ),
            'cgm_duration_years': (
                cgm_duration_to_years(cgm_duration_value, duration_unit)
                if uses_cgm_bool
                else None
            ),
            'format': format_value,
            'run_format': format_value,
            # Startup B/C checkbox and/or landing upload consent — stamp both flags.
            'consent_use_uploaded_data': bool(has_data_consent),
            'consent_upload_own_data': bool(
                has_data_consent or info.get("consent_upload_own_data")
            ),
            'diabetic': diabetic,
            'diabetic_type': diabetic_type,
            'diabetes_duration': diabetes_duration,
            'challenge_unknown': challenge_on,
            'challenge_unknown_pct': CHALLENGE_UNKNOWN_PCT if challenge_on else '',
            'generic_intervention': generic_intervention_for_user(
                {
                    'diabetic': diabetic,
                    'diabetic_type': diabetic_type,
                    'format': format_value,
                    'challenge_unknown': challenge_on,
                }
            ),
            'paper_mention': wants_paper_mention,
            'paper_full_name': paper_name,
            'location': location,
            'rounds': info.get('rounds') or [],
            'max_rounds': int(info.get('max_rounds') or MAX_ROUNDS),
            'current_round_number': int(info.get('current_round_number') or 1),
            'statistics_saved': bool(info.get('statistics_saved') or False),
        })
        stamp_upload_data_consent(info)

        # Round-1 data-source identity, accounting for a file imported at startup
        # (uploaded_data_path set by handle_startup_csv_upload / _nightscout_import):
        #  - B with an import -> round 1 is the user's own data.
        #  - B/C without an import -> gated on /prediction (blank Source until upload).
        #  - A, and C with an import -> round 1 is the generic warm-up.
        _startup_uploaded = info.get('uploaded_data_path')
        if format_value == "B" and _startup_uploaded:
            info['is_example_data'] = False
            info['data_source_name'] = str(info.get('uploaded_data_filename') or 'uploaded.csv')
        elif format_value in ("B", "C") and not _startup_uploaded:
            info['is_example_data'] = True
            info['data_source_name'] = ""
        else:  # A, or C with a startup import -> generic warm-up
            info['is_example_data'] = True
            info['data_source_name'] = "example.csv"

        # Ensure stable "number" across consent + stats + ranking CSVs.
        if info.get("number") is None:
            info["number"] = get_next_study_number()

        # Ensure consent fields are explicit booleans (avoid null/missing keys in session storage).
        if "consent_play_only" not in info:
            info["consent_play_only"] = False
        if "consent_participate_in_study" not in info:
            info["consent_participate_in_study"] = True
        if "consent_receive_results_later" not in info:
            info["consent_receive_results_later"] = False
        if "consent_keep_up_to_date" not in info:
            info["consent_keep_up_to_date"] = False
        if "consent_no_selection" not in info:
            info["consent_no_selection"] = True
        if "consent_timestamp" not in info:
            info["consent_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Leftover localStorage may still have consent_play_only=True from the
        # removed checkbox; force it off so this Start persists the game.
        reconcile_stored_consents(info)

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
            "consent_use_uploaded_data": bool(info.get("consent_use_uploaded_data", False)),
            "paper_mention": bool(info.get("paper_mention", False)),
            "paper_full_name": str(info.get("paper_full_name") or ""),
        }
        ensure_consent_agreement_row(consent_row)
        # Stamp resolved flags so the consent CSV matches what we will persist
        # (clears leftover play_only=True from older sessions).
        upsert_consent_agreement_fields(
            str(info["study_id"]),
            {
                "play_only": bool(info.get("consent_play_only", False)),
                "participate_in_study": bool(info.get("consent_participate_in_study", False)),
                "no_selection": bool(info.get("consent_no_selection", True)),
                "receive_results_later": bool(info.get("consent_receive_results_later", False)),
                "keep_up_to_date": bool(info.get("consent_keep_up_to_date", False)),
                "upload_own_data": bool(info.get("consent_upload_own_data", False)),
                "paper_mention": bool(info.get("paper_mention", False)),
                "paper_full_name": str(info.get("paper_full_name") or ""),
            },
        )
        # Capture the starter immediately -- people who open /prediction and
        # never submit (or never hit Exit) still leave a study row.
        if should_persist_study_data(info):
            submit_component.save_statistics(info)
            info["statistics_saved"] = True
        window, events, is_example, source_name, slider, rand_init = _load_round_one_stores(info)
        info["is_example_data"] = is_example
        if source_name:
            info["data_source_name"] = source_name
        return '/prediction', info, window, events, is_example, source_name, slider, rand_init
    return _start_idle


@app.callback(
    Output('user-info-store', 'data', allow_duplicate=True),
    [Input('consent-acknowledge', 'value'),
     Input('consent-gdpr', 'value'),
     Input('consent-upload-own-data', 'value'),
     Input('consent-receive-results', 'value'),
     Input('consent-keep-updated', 'value')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def record_mobile_consent(
    acknowledge_value: Optional[list[str]],
    gdpr_value: Optional[list[str]],
    upload_own_data_value: Optional[list[str]],
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

    upload_own_data = bool(upload_own_data_value and "upload_own_data" in upload_own_data_value)
    receive_results = bool(receive_results_value and "receive_results" in receive_results_value)
    keep_updated = bool(keep_updated_value and "keep_updated" in keep_updated_value)

    info["consent_gdpr"] = gdpr_consented
    apply_optional_consent_choices(
        info,
        receive_results=receive_results,
        keep_updated=keep_updated,
        upload_own_data=upload_own_data or bool(info.get("consent_use_uploaded_data")),
    )
    info["consent_timestamp"] = info.get("consent_timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["consent_completed"] = True
    # Stable cross-device resume code, assigned at consent like on desktop.
    info["resume_code"] = info.get("resume_code") or resume_store.new_code()
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


def append_round_from_window(
    user_info: Dict[str, Any],
    current_df: pl.DataFrame,
    slider_value: Optional[int],
) -> Dict[str, Any]:
    """Append the current chart window as a completed round on ``user_info``.

    Shared by Submit (shows results) and Finish/Exit (stores the round, then
    routes to ``/ending`` or ``/final``).
    """
    info: Dict[str, Any] = dict(user_info)
    if 'age' in info:
        current_df = current_df.with_columns(pl.lit(int(info['age'])).alias("age"))
    rounds: list[dict[str, Any]] = list(info.get('rounds') or [])
    round_number = len(rounds) + 1
    info['prediction_window_start'] = slider_value or 0
    info['prediction_window_size'] = len(current_df)
    prediction_table_data = PredictionTableComponent()._generate_table_data(current_df)
    info['prediction_table_data'] = prediction_table_data
    info['current_round_number'] = round_number
    window_times = current_df.get_column('time').dt.strftime('%Y-%m-%d %H:%M:%S').to_list()
    info['window_times'] = window_times
    round_info: dict[str, Any] = {
        'round_number': round_number,
        'prediction_window_start': info['prediction_window_start'],
        'prediction_window_size': info['prediction_window_size'],
        'prediction_table_data': prediction_table_data,
        'window_times': window_times,
        'format': str(info.get('format') or ''),
        'is_example_data': bool(info.get('is_example_data', True)),
        'data_source_name': str(info.get('data_source_name', 'example.csv')),
    }
    # Window fingerprint for every format: A (generic subject), B (own file),
    # and C (alternating). Prefer the key stamped when a generic window was
    # selected; never reuse that key on an own-data round (stale after A→B).
    if round_info['is_example_data'] and info.get('current_generic_slice_key'):
        round_info['generic_slice_key'] = str(info['current_generic_slice_key'])
    else:
        round_info['generic_slice_key'] = generic_window_slice_key(current_df)
    rounds.append(round_info)
    info['rounds'] = rounds
    return info


def capture_complete_round_on_exit(
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
    slider_value: Optional[int],
) -> Dict[str, Any]:
    """If the hidden hour is fully drawn, persist that round onto ``user_info``."""
    info: Dict[str, Any] = dict(user_info or {})
    if not current_df_data:
        return info
    current_df = reconstruct_dataframe_from_dict(current_df_data)
    if not hidden_area_is_complete(current_df):
        return info
    return append_round_from_window(info, current_df, slider_value)


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('current-window-df', 'data'),
     State('time-slider', 'value')],
    prevent_initial_call=True
)
def handle_submit_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
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

    if current_df_data:
        print("DEBUG: Submit button clicked")

        # Only the small window is needed here; the full dataset is no longer
        # round-tripped through the client (it lives server-side, sliced on demand).
        current_df = reconstruct_dataframe_from_dict(current_df_data)

        # Update age from user_info on the window.
        if user_info and 'age' in user_info:
            current_df = current_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))

        # Generate prediction table data directly from DataFrame instead of relying on component
        if user_info is None:
            user_info = {}
        # Mark this round as submitted at this click-count. This prevents double-submits if the
        # callback is re-triggered due to component re-mounts/navigation.
        user_info["last_submit_round_number"] = pending_round_number
        user_info["last_submit_n_clicks"] = int(n_clicks)

        user_info = append_round_from_window(user_info, current_df, slider_value)
        
        # Debug: Check what predictions we have
        prediction_count = current_df.filter(pl.col("prediction") != 0.0).height
        print(f"DEBUG: Submit button - Found {prediction_count} predictions in current_df")
        print(f"DEBUG: Submit button - Sample predictions: {current_df.filter(pl.col('prediction') != 0.0).select(['time', 'prediction']).head(5).to_dicts()}")

        # Upsert after every submitted round so a closed tab without Exit
        # still leaves the incomplete game in the study CSVs.
        if should_persist_study_data(user_info):
            submit_component.save_statistics(user_info)
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
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True)],
    [Input('next-round-button', 'n_clicks')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_next_round_button(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], Dict[str, bool], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int]:
    print(f"DEBUG handle_next_round_button FIRED: n_clicks={n_clicks}")
    if not n_clicks or not user_info:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    max_rounds = int(user_info.get('max_rounds') or MAX_ROUNDS)
    next_round_number = len(rounds) + 1
    if next_round_number > max_rounds:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    with start_action(action_type=u"handle_next_round_button", next_round=next_round_number):
        fmt = str(user_info.get("format") or "A")
        points = int(user_info.get('prediction_window_size') or DEFAULT_POINTS)
        points = max(MIN_POINTS, min(MAX_POINTS, points))

        # Choose dataset based on format.
        is_example: bool
        source_name: str
        random_start: int
        if fmt == "A":
            new_df, events_df, source_name, random_start = _apply_generic_round_selection(
                user_info, rounds, points,
            )
            is_example = True
        elif fmt == "B":
            uploaded_path = user_info.get("uploaded_data_path")
            if not uploaded_path:
                # Should not happen in normal flow, but keep safe empty state.
                return '/prediction', user_info, {'hide_last_hour': True}, no_update, no_update, False, "", False, 0
            full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
            is_example = False
            source_name = str(user_info.get("uploaded_data_filename") or user_info.get("data_source_name") or "uploaded.csv")
            used_starts: set[int] = {
                int(r["prediction_window_start"])
                for r in rounds
                if not r.get("is_example_data", True)
                and r.get("prediction_window_start") is not None
            }
            new_df, random_start = get_random_data_window(full_df, points, used_starts=used_starts)
        else:
            # Format C ("mixed"): a file is required before any graph. Once
            # uploaded, interleave: ODD round -> generic, EVEN round -> own data.
            uploaded_path = user_info.get("uploaded_data_path")
            if not uploaded_path:
                # next_round_number is always >= 2 here (round 1 came from init).
                user_info['current_round_number'] = next_round_number
                return '/prediction', user_info, {'hide_last_hour': True}, None, None, False, "", False, 0
            use_example = (next_round_number % 2 == 1)
            if use_example:
                new_df, events_df, source_name, random_start = _apply_generic_round_selection(
                    user_info, rounds, points,
                )
                is_example = True
            else:
                full_df, events_df = load_glucose_data(Path(str(uploaded_path)))
                is_example = False
                source_name = str(user_info.get("uploaded_data_filename") or user_info.get("data_source_name") or "uploaded.csv")
                used_starts = {
                    int(r["prediction_window_start"])
                    for r in rounds
                    if not r.get("is_example_data", True)
                    and r.get("prediction_window_start") is not None
                }
                new_df, random_start = get_random_data_window(full_df, points, used_starts=used_starts)

        # Reset any previous predictions before starting a fresh round.
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))

        user_info['current_round_number'] = next_round_number
        user_info['is_example_data'] = is_example
        user_info['data_source_name'] = source_name
        chart_mode = {'hide_last_hour': True}

        return (
            '/prediction',
            user_info,
            chart_mode,
            convert_df_to_dict(new_df),
            events_store_for_window(events_df, new_df),
            is_example,
            source_name,
            False,  # let slider init set it from initial-slider-value
            random_start
        )


def _finish_rounds_if_exiting(
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
    *,
    count_current_drawing: bool,
) -> int:
    rounds_played = len((user_info or {}).get("rounds") or [])
    if (
        count_current_drawing
        and current_df_data
        and hidden_area_is_complete(reconstruct_dataframe_from_dict(current_df_data))
    ):
        rounds_played += 1
    return rounds_played


def _finish_confirmation_context(
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
    *,
    count_current_drawing: bool,
) -> Dict[str, int]:
    return {
        "rounds_played": _finish_rounds_if_exiting(
            user_info,
            current_df_data,
            count_current_drawing=count_current_drawing,
        ),
        "max_rounds": int((user_info or {}).get("max_rounds") or MAX_ROUNDS),
        "min_useful": int((user_info or {}).get("min_useful_rounds") or MIN_USEFUL_ROUNDS),
    }


def _open_finish_confirmation(
    *,
    count_current_drawing: bool,
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, str], str, Dict[str, int]]:
    return (
        {"display": "flex"},
        "finish-confirm-overlay is-open",
        _finish_confirmation_context(
            user_info,
            current_df_data,
            count_current_drawing=count_current_drawing,
        ),
    )


def _finish_confirmation_text(
    context: Optional[Dict[str, int]],
    interface_language: Optional[str],
) -> tuple[str, str, str, str]:
    """Translate confirmation copy without changing overlay or navigation state."""
    locale = normalize_locale(interface_language)
    message = ""
    if context:
        message = finish_confirm_message(
            rounds_played=int(context.get("rounds_played") or 0),
            max_rounds=int(context.get("max_rounds") or MAX_ROUNDS),
            min_useful=int(context.get("min_useful") or MIN_USEFUL_ROUNDS),
            locale=locale,
        )
    return (
        t("ui.submit.finish_confirm_title", locale=locale),
        message,
        t("ui.submit.finish_anyway", locale=locale),
        t("ui.submit.keep_playing", locale=locale),
    )


@app.callback(
    [Output('finish-confirm-title-prediction', 'children'),
     Output('finish-confirm-message-prediction', 'children'),
     Output('finish-confirm-button-prediction', 'children'),
     Output('finish-keep-playing-button-prediction', 'children')],
    [Input('interface-language', 'data'),
     Input('finish-confirm-context-prediction', 'data')],
)
def update_prediction_finish_confirm_text(
    interface_language: Optional[str],
    context: Optional[Dict[str, int]],
) -> tuple[str, str, str, str]:
    return _finish_confirmation_text(context, interface_language)


@app.callback(
    [Output('finish-confirm-title-ending', 'children'),
     Output('finish-confirm-message-ending', 'children'),
     Output('finish-confirm-button-ending', 'children'),
     Output('finish-keep-playing-button-ending', 'children')],
    [Input('interface-language', 'data'),
     Input('finish-confirm-context-ending', 'data')],
)
def update_ending_finish_confirm_text(
    interface_language: Optional[str],
    context: Optional[Dict[str, int]],
) -> tuple[str, str, str, str]:
    return _finish_confirmation_text(context, interface_language)


@app.callback(
    [Output('finish-confirm-overlay-prediction', 'style'),
     Output('finish-confirm-overlay-prediction', 'className'),
     Output('finish-confirm-context-prediction', 'data')],
    [Input('finish-study-button', 'n_clicks'),
     Input('finish-keep-playing-button-prediction', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('current-window-df', 'data')],
    prevent_initial_call=True,
)
def toggle_finish_confirmation_from_prediction(
    finish_clicks: Optional[int],
    keep_playing_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, str], str, Optional[Dict[str, int]]]:
    if ctx.triggered_id == 'finish-keep-playing-button-prediction':
        if not keep_playing_clicks:
            raise PreventUpdate
        return {"display": "none"}, "finish-confirm-overlay", None
    if not finish_clicks:
        raise PreventUpdate
    return _open_finish_confirmation(
        count_current_drawing=True,
        user_info=user_info,
        current_df_data=current_df_data,
    )


@app.callback(
    [Output('finish-confirm-overlay-ending', 'style'),
     Output('finish-confirm-overlay-ending', 'className'),
     Output('finish-confirm-context-ending', 'data')],
    [Input('finish-study-button-ending', 'n_clicks'),
     Input('finish-keep-playing-button-ending', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('current-window-df', 'data')],
    prevent_initial_call=True,
)
def toggle_finish_confirmation_from_ending(
    ending_clicks: Optional[int],
    keep_playing_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, str], str, Optional[Dict[str, int]]]:
    if ctx.triggered_id == 'finish-keep-playing-button-ending':
        if not keep_playing_clicks:
            raise PreventUpdate
        return {"display": "none"}, "finish-confirm-overlay", None
    if not ending_clicks:
        raise PreventUpdate
    max_rounds = int((user_info or {}).get("max_rounds") or MAX_ROUNDS)
    current_round = int(
        (user_info or {}).get("current_round_number")
        or len((user_info or {}).get("rounds") or [])
    )
    if current_round >= max_rounds:
        raise PreventUpdate
    return _open_finish_confirmation(
        count_current_drawing=False,
        user_info=user_info,
        current_df_data=current_df_data,
    )


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('last-visited-page', 'data', allow_duplicate=True)],
    [Input('finish-confirm-button-prediction', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('current-window-df', 'data'),
     State('time-slider', 'value')],
    prevent_initial_call=True
)
def handle_finish_study_from_prediction(
    n_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, Any]],
    slider_value: Optional[int],
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool], Optional[str]]:
    """Exit from the chart: keep any fully drawn round, then show results.

    A complete current drawing goes to ``/ending`` (this round vs actual).
    Prior rounds with an incomplete current drawing go to ``/final`` (overall).
    Only a session with nothing to show returns to landing.
    """
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    with start_action(action_type=u"handle_finish_study_from_prediction", n_clicks=int(n_clicks)):
        pass

    before = len((user_info or {}).get("rounds") or [])
    info = capture_complete_round_on_exit(user_info, current_df_data, slider_value)
    rounds: list[dict[str, Any]] = info.get("rounds") or []
    current_complete = False
    if current_df_data:
        current_complete = hidden_area_is_complete(
            reconstruct_dataframe_from_dict(current_df_data)
        )
    just_captured = len(rounds) > before
    show_this_round = bool(
        (just_captured or current_complete)
        and info.get("prediction_table_data")
        and current_df_data
    )

    if should_persist_study_data(info) and rounds:
        with start_action(action_type=u"handle_finish_study_from_prediction"):
            submit_component.save_statistics(info, write_ranking=not show_this_round)
            info["statistics_saved"] = True

    if show_this_round:
        return "/ending", info, {"hide_last_hour": False}, "/ending"
    if rounds:
        return "/final", info, {"hide_last_hour": False}, "/final"
    return "/", info, {"hide_last_hour": True}, None


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True)],
    [Input('finish-study-button-ending', 'n_clicks'),
     Input('finish-confirm-button-ending', 'n_clicks')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_finish_study_from_ending(
    finish_clicks: Optional[int],
    confirm_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool]]:
    n_clicks = (
        confirm_clicks
        if ctx.triggered_id == "finish-confirm-button-ending"
        else finish_clicks
    )
    if not n_clicks:
        return no_update, no_update, no_update
    if ctx.triggered_id == "finish-study-button-ending" and user_info:
        max_rounds = int(user_info.get("max_rounds") or MAX_ROUNDS)
        current_round = int(
            user_info.get("current_round_number")
            or len(user_info.get("rounds") or [])
        )
        if current_round < max_rounds:
            return no_update, no_update, no_update

    with start_action(action_type=u"handle_finish_study_from_ending", n_clicks=int(n_clicks)):
        pass

    if not user_info:
        return '/final', None, {'hide_last_hour': True}

    rounds: list[dict[str, Any]] = user_info.get('rounds') or []
    if not rounds:
        return '/final', user_info, {'hide_last_hour': True}

    # Last Submit already wrote this run. Do not rewrite on Results — that
    # blocks the click for a second and is why the button feels dead.
    if should_persist_study_data(user_info) and not user_info.get("statistics_saved"):
        with start_action(action_type=u"handle_finish_study_from_ending"):
            submit_component.save_statistics(user_info)
            user_info['statistics_saved'] = True

    return '/final', user_info, {'hide_last_hour': False}


def _final_fill_locale_unit(
    interface_language: Optional[str],
    glucose_unit: Optional[str],
) -> tuple[str, str]:
    locale = normalize_locale(interface_language)
    unit = glucose_unit if glucose_unit in ("mg/dL", "mmol/L") else "mg/dL"
    return locale, unit


@app.callback(
    [
        Output("final-ranking-list", "children", allow_duplicate=True),
        Output("final-fill-step", "data", allow_duplicate=True),
    ],
    Input("final-fill-step", "data"),
    [
        State("url", "pathname"),
        State("user-info-store", "data"),
        State("glucose-unit", "data"),
        State("interface-language", "data"),
    ],
    prevent_initial_call=True,
)
def fill_final_leaderboard(
    kick: Optional[Any],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
) -> tuple[Any, Any]:
    """Phase 1: ranking CSVs. Advances the store so the share callback can run.

    A single callback that Output its own Input did not re-enter for phase 2
    (seen 10:18:40 — leaderboard landed, graph never did).
    """
    if pathname != "/final" or not user_info or not isinstance(kick, dict):
        raise PreventUpdate
    if kick.get("phase") != 1:
        raise PreventUpdate
    locale, unit = _final_fill_locale_unit(interface_language, glucose_unit)
    with start_action(action_type=u"fill_final_leaderboard"):
        overall, per_format = _final_ranking_snapshots(user_info)
        ranking = _final_leaderboard_children(
            overall=overall,
            per_format=per_format,
            locale=locale,
            unit=unit,
            user_info=user_info,
        )
    return ranking, {**kick, "phase": 2}


@app.callback(
    [
        Output("final-share-panel", "children", allow_duplicate=True),
        Output("final-synthesis-card", "children", allow_duplicate=True),
    ],
    Input("final-fill-step", "data"),
    [
        State("url", "pathname"),
        State("user-info-store", "data"),
        State("glucose-unit", "data"),
        State("interface-language", "data"),
    ],
    prevent_initial_call=True,
)
def fill_final_share(
    kick: Optional[Any],
    pathname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
) -> tuple[Any, Any]:
    """Phase 2: share record + synthesis graph, after the leaderboard store write."""
    if pathname != "/final" or not user_info or not isinstance(kick, dict):
        raise PreventUpdate
    if kick.get("phase") != 2:
        raise PreventUpdate
    locale, _unit = _final_fill_locale_unit(interface_language, glucose_unit)
    with start_action(action_type=u"fill_final_share"):
        share = _final_share_section_children(user_info, locale=locale)
        synthesis = _final_synthesis_inner(user_info, locale=locale)
    return share, synthesis


app.clientside_callback(
    """
    function(resultsClicks, confirmClicks, userInfo) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        var prop = ctx.triggered[0].prop_id || '';
        var isConfirm = prop.indexOf('finish-confirm-button-ending') === 0;
        var isResults = prop.indexOf('finish-study-button-ending') === 0;
        if (isConfirm && confirmClicks) {
            return 'results-loading-overlay is-open';
        }
        if (!isResults || !resultsClicks) {
            return window.dash_clientside.no_update;
        }
        var info = userInfo || {};
        var maxRounds = Number(info.max_rounds || 12);
        var current = Number(info.current_round_number || 0);
        var nRounds = (info.rounds && info.rounds.length) ? info.rounds.length : 0;
        if (Math.max(current, nRounds) < maxRounds) {
            return window.dash_clientside.no_update;
        }
        var btn = document.getElementById('finish-study-button-ending');
        if (btn) {
            btn.disabled = true;
            btn.style.pointerEvents = 'none';
        }
        return 'results-loading-overlay is-open';
    }
    """,
    Output("results-loading-overlay", "className"),
    Input("finish-study-button-ending", "n_clicks"),
    Input("finish-confirm-button-ending", "n_clicks"),
    State("user-info-store", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(pathname) {
        if (pathname === '/ending') {
            return window.dash_clientside.no_update;
        }
        return 'results-loading-overlay';
    }
    """,
    Output("results-loading-overlay", "className", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)


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
    [Output('final-nickname-status', 'children'),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('final-ranking-list', 'children', allow_duplicate=True)],
    Input('final-nickname-save', 'n_clicks'),
    [State('final-nickname-input', 'value'),
     State('user-info-store', 'data'),
     State('glucose-unit', 'data'),
     State('interface-language', 'data')],
    prevent_initial_call=True,
)
def save_final_nickname(
    n_clicks: Optional[int],
    raw_nickname: Optional[str],
    user_info: Optional[Dict[str, Any]],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
) -> Tuple[str, Dict[str, Any], list[Any]]:
    """Persist the leaderboard nickname typed on ``/final`` and re-render the board.

    Writes it onto **this study's** ranking rows only, so a returning player who picks
    a different name leaves their earlier study entries as they were.  The nickname is
    a public display label and never enters the study record.
    """
    if not n_clicks or not user_info:
        raise PreventUpdate

    locale = normalize_locale(interface_language)
    unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
    nickname = normalize_nickname(raw_nickname)
    study_id = str(user_info.get('study_id') or '')
    key = email_key(user_info.get('email'))

    with start_action(action_type=u"save_final_nickname", named=bool(nickname)) as action:
        updated_info: Dict[str, Any] = dict(user_info)
        updated_info['nickname'] = nickname
        rows_changed = submit_component.set_study_nickname(
            study_id=study_id, key=key, nickname=nickname
        )
        action.log(message_type=u"nickname_saved", rows_changed=rows_changed)

    # Re-read the CSVs so the board shows the new name without a page reload.
    runs_by_format: dict[str, list[dict[str, Any]]] = dict(updated_info.get('runs_by_format') or {})
    already_played: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
    if updated_info.get('rounds'):
        already_played.add(str(updated_info.get('format') or 'A'))

    per_format_boards: list[tuple[str, dict[str, Any]]] = []
    for fmt in sorted(already_played, key=lambda x: FORMAT_ORDER.get(str(x), 999)):
        if fmt not in ("A", "B", "C"):
            continue
        board = _leaderboard_snapshot(
            project_root / 'data' / 'input' / f'prediction_ranking_{fmt}.csv',
            study_id=study_id,
            key=key,
            format_filter=fmt,
        )
        if board is not None:
            per_format_boards.append((fmt, board))

    overall_board = _leaderboard_snapshot(
        project_root / 'data' / 'input' / 'prediction_ranking.csv',
        study_id=study_id,
        key=key,
        format_filter="ALL",
    )

    return (
        t("ui.final.nickname_saved", locale=locale),
        updated_info,
        _final_leaderboard_children(
            overall=overall_board,
            per_format=per_format_boards,
            locale=locale,
            unit=unit,
            user_info=updated_info,
            offer_nickname=True,
        ),
    )


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('glucose-unit', 'data', allow_duplicate=True),
     Output('interface-language', 'data', allow_duplicate=True),
     Output('last-visited-page', 'data', allow_duplicate=True),
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


# NOTE: there is deliberately no share callback anymore.  The share flow is
# part of /final itself: `create_final_layout` builds the record via
# `build_final_share_record`, persists it with `share_store.ensure_share`
# (content-addressed, so re-renders reuse the file), and renders
# `build_share_panel` eagerly.  /share/<id> remains the recipient-facing page.

# Clientside: clipboard copy for the "Copy link" button on the share panel
# (present on both /final and /share/<id>).
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
        from sugar_sugar.components.startup import stamp_upload_data_consent

        _archive_current_run(info)
        # C→B (etc.): if consent/upload already happened in a prior format, keep it.
        stamp_upload_data_consent(info)

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
            new_df, events_df, source_name, random_start = _apply_generic_round_selection(info, [], points)
            new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
            info["is_example_data"] = True
            info["data_source_name"] = source_name
            return (
                "/prediction",
                info,
                chart_mode,
                convert_df_to_dict(new_df),
                events_store_for_window(events_df, new_df),
                True,
                source_name,
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
                convert_df_to_dict(new_df),
                events_store_for_window(events_df, new_df),
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
     Input('current-window-df', 'data')],
    [State('page-restore-done', 'data'),
     State('url', 'pathname'),
     State('session-active', 'data')],
    prevent_initial_call=True,
)
def restore_page_on_load(
    last_page: Optional[str],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict],
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

    All three localStorage stores (last-visited-page, user-info-store,
    current-window-df) are Inputs so the callback re-fires as each store
    hydrates.  The ``page-restore-done`` memory flag prevents action after the
    first decision. (full-df is no longer a client store; the small window store
    is the "game in progress" signal now.)
    """
    if already_done or _is_chart_mode:
        raise PreventUpdate

    if not last_page or last_page == "/":
        # Hydration ORDER, not just hydration: `last-visited-page` sits *after*
        # user-info-store / current-window-df in the layout, so it hydrates later
        # and this callback's first firing routinely carries a populated session
        # with `last_page` still None. Marking the restore "done" there burned the
        # one-shot guard: when last-visited-page finally arrived the callback was
        # already spent, so no dialog and no redirect ever happened and the player
        # was left on the landing page -- whose only mobile CTA walks into the
        # consent wizard. Wait for the next firing instead whenever the session
        # stores already hold data; only a genuinely empty localStorage (nothing
        # hydrated anywhere) means "fresh visitor, nothing to restore".
        if user_info or current_df_data:
            raise PreventUpdate
        return no_update, True, no_update, True

    if pathname and pathname != "/":
        return no_update, True, no_update, True

    if last_page in ("/prediction", "/ending", "/final") and not user_info:
        raise PreventUpdate
    if last_page == "/ending" and not current_df_data:
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
            if has_prediction_data and current_df_data:
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


@app.callback(
    [Output('game-stores-hydrated', 'data'),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input('session-restore-poll', 'n_intervals')],
    [State('user-info-store', 'data'),
     State('current-window-df', 'data'),
     State('url', 'pathname')],
    prevent_initial_call=True,
)
def resolve_session_restore(
    n_intervals: Optional[int],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict],
    pathname: Optional[str],
) -> Tuple[Any, Any]:
    """Finish (or abandon) the restore that `_restoring_layout` is showing.

    `session-restore-poll` lives **only inside that placeholder**, so this
    callback is inert on every other page -- in particular it can never fire
    while the chart is up and force a `display_page` re-render mid-round.

    As soon as the stores this route needs have hydrated, flip
    `game-stores-hydrated` so `display_page` re-renders the real page. If they
    never arrive (localStorage genuinely empty -- e.g. someone deep-links to
    /prediction without a session), give up after `_RESTORE_GIVE_UP_TICKS` and
    route to landing rather than spin forever.
    """
    if _game_stores_ready(pathname, user_info, current_df_data):
        return True, no_update
    if int(n_intervals or 0) >= _RESTORE_GIVE_UP_TICKS:
        with start_action(action_type=u"session_restore_gave_up", pathname=pathname):
            return no_update, "/"
    raise PreventUpdate


@app.callback(
    Output('prediction-chart-rendered', 'data'),
    [Input('url', 'pathname'),
     Input('game-stores-hydrated', 'data')],
    [State('user-info-store', 'data')],
    prevent_initial_call=False,
)
def mark_prediction_chart_rendered(
    pathname: Optional[str],
    stores_hydrated: Optional[bool],
    user_info: Optional[Dict[str, Any]],
) -> bool:
    """Publish whether the drawing chart is really on screen (route-class truth).

    The `route-prediction` class used to be stamped from the pathname alone. When
    the URL said /prediction but `display_page` had rendered something else (the
    un-hydrated cold load above, or the consent bounce), every prediction-only
    mobile rule then applied to that foreign content -- including the two
    `:not(.route-prediction)` *releases*: the `#page-content *  { max-width:100% }`
    overflow cap (without it a form page overflows and the browser zooms the whole
    page out) and `touch-action: manipulation` (without it Android waits ~300 ms
    per tap for a double-tap-zoom and swallows taps, the documented "Next worked
    on the 4th click"). Net effect: a consent form the player could tick but not
    submit. Keyed on the render decision, that combination cannot recur.
    """
    return _renders_prediction_chart(pathname, user_info)


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
     State('current-window-df', 'data')],
    prevent_initial_call=True,
)
def redirect_landing_to_game(
    pathname: Optional[str],
    last_page: Optional[str],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict],
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

    # Exit / Start Over nulls user-info. Do not bounce a cleared session back
    # onto /startup or /prediction (that reused the old form and a leftover
    # generic window). In-session "Game" still has user_info populated.
    if not user_info:
        raise PreventUpdate

    if last_page == "/ending":
        has_ptd = bool(user_info and "prediction_table_data" in user_info)
        if has_ptd and current_df_data:
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
        # Wrap on narrow phones: two 140px buttons + the gap don't fit inside the
        # 90vw card on a ~360px portrait screen, and an overflowing element makes
        # the browser expand the layout viewport and zoom the whole page out.
        'flexWrap': 'wrap',
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
    current_df: Optional[Dict],
    events_df: Optional[Dict],
    last_page: Optional[str],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
) -> Dict[str, Any]:
    """Thin JSON-serialisable snapshot of the stores needed to restore a game.

    The full dataset is NOT snapshotted -- only its identity (in user_info:
    is_example_data / uploaded_data_path) plus the current window + its events.
    On restore the dataset is reloaded server-side from that identity, so the
    file must persist on the server (uploads under data/input/users do).
    """
    return {
        "user_info": user_info,
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
    [State('current-window-df', 'data'),
     State('events-df', 'data')],
    prevent_initial_call=True,
)
def auto_snapshot_session(
    user_info: Optional[Dict[str, Any]],
    last_page: Optional[str],
    glucose_unit: Optional[str],
    interface_language: Optional[str],
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
        _resume_payload(user_info, current_df, events_df, last_page, glucose_unit, interface_language),
    )
    return code


def _restore_outputs_from_code(code: Optional[str]) -> Optional[tuple]:
    """Load a session by code and return the store-output tuple, or None if missing.

    Output order matches the redeem callbacks:
    (pathname, user_info, current_window_df, events_df, glucose_unit,
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
    [Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True)],
    [Input('url', 'pathname')],
    [State('current-window-df', 'data'),
     State('user-info-store', 'data'),
     State('data-source-name', 'data'),
     State('events-df', 'data')],
    prevent_initial_call=True
)
def initialize_data_on_url_change(
    pathname: Optional[str],
    current_df_data: Optional[Dict],
    user_info: Optional[Dict[str, Any]],
    source_name_store: Optional[str] = None,
    events_data: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    bool,
    str,
    bool,
    int,
]:
    """Initialize the window when URL changes to /prediction without existing data.

    Only loads a fresh example window when navigating to /prediction and no
    window exists yet.  All other pages are left alone so that persisted
    localStorage stores are never overwritten (critical for the resume flow).
    The full dataset is sliced server-side; only the window is shipped.

    Compacting an oversized `events-df` (`compacted_events_store`) is folded into
    this callback rather than living in its own: Dash derives the `allow_duplicate`
    suffix from the INPUTS alone (`create_callback_id._hash_inputs`), so a second
    callback writing `events-df` off the same `Input('url', 'pathname')` hashes to
    the same output id and the renderer aborts the page with "Duplicate callback
    outputs". Two writers of one store on one trigger were also a last-writer-wins
    race. Any future `events-df` writer keyed on the pathname belongs here too.
    """
    compacted = compacted_events_store(events_data, current_df_data)
    _no_change = (no_update, compacted, no_update, no_update, no_update, no_update)

    if pathname != '/prediction':
        return _no_change

    # Upload gate (B/C until a file exists): don't auto-load a generic window
    # when the session must block on an upload -- leave the window empty so the
    # gate shows (handles direct load / resume mid-session).
    if _is_upload_gated(user_info):
        return None, None, False, "", False, 0

    info = dict(user_info or {})
    fmt = str(info.get("format") or "A")
    uploaded_path = info.get("uploaded_data_path")
    expected_own = ""
    if fmt == "B" and uploaded_path:
        expected_own = str(info.get("uploaded_data_filename") or info.get("data_source_name") or "uploaded.csv")

    # Keep an existing window (resume / next-round) unless it is a leftover
    # generic slice while this session is My Data with a file already imported.
    if current_df_data is not None:
        if not expected_own or str(source_name_store or "") == expected_own:
            return _no_change

    window, events, is_example, source_name, random_start, rand_init = _load_round_one_stores(info)

    with start_action(action_type=u"initialize_data_on_url_change") as action:
        action.log(message_type="new_random_start", random_start=random_start, is_example=is_example)

    return (window, events, is_example, source_name, rand_init, random_start)

# Client-side upload compression. dcc.Upload hands us the file as a base64 data
# URL; we gzip it in the browser (CompressionStream, Safari 16.4+/Chrome/Firefox)
# and hand a "gzip:<base64>" string to the server store instead. A ~2.4 MB CSV
# shrinks to ~300-400 KB, well under whatever ceiling the mobile browser hits.
# Any failure (no CompressionStream, exception) returns the original contents so
# the server's data-URL path still works -- desktop behaviour is unchanged.
_UPLOAD_COMPRESS_JS = """
async function(contents, filename) {
    if (!contents) { return window.dash_clientside.no_update; }
    try {
        if (typeof CompressionStream === 'undefined') { return contents; }
        var comma = contents.indexOf(',');
        if (comma < 0) { return contents; }
        var b64 = contents.slice(comma + 1);
        var raw = Uint8Array.from(atob(b64), function(c){ return c.charCodeAt(0); });
        var cs = new CompressionStream('gzip');
        var buf = await new Response(new Blob([raw]).stream().pipeThrough(cs)).arrayBuffer();
        var comp = new Uint8Array(buf);
        var bin = '';
        var CH = 0x8000;
        for (var i = 0; i < comp.length; i += CH) {
            bin += String.fromCharCode.apply(null, comp.subarray(i, i + CH));
        }
        return 'gzip:' + btoa(bin);
    } catch (e) {
        return contents;
    }
}
"""

app.clientside_callback(
    _UPLOAD_COMPRESS_JS,
    Output('upload-data-payload', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)

app.clientside_callback(
    _UPLOAD_COMPRESS_JS,
    Output('startup-upload-payload', 'data'),
    Input('startup-upload-data', 'contents'),
    State('startup-upload-data', 'filename'),
    prevent_initial_call=True,
)


# Separate callback for file upload handling
@app.callback(
    [Output('last-click-time', 'data'),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('consent-scroll-request', 'data')],
    [Input('upload-data-payload', 'data'),
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
) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, Dict[str, Any], int]:
    """Handle file upload and data loading"""
    triggered = ctx.triggered_id
    if triggered not in ("upload-data-payload", "prediction-data-usage-consent"):
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

            prev_consent = _upload_data_consent_given(info_pre)
            pending = info_pre.get("pending_upload_contents")

            if not prev_consent:
                from sugar_sugar.components.startup import stamp_upload_data_consent

                info_pre["consent_use_uploaded_data"] = True
                info_pre["consent_upload_own_data"] = True
                stamp_upload_data_consent(info_pre)
                info_pre["blocked_upload_requires_consent"] = False

                study_id = str(info_pre.get("study_id") or "")
                if study_id:
                    from sugar_sugar.consent import upsert_consent_agreement_fields

                    upsert_consent_agreement_fields(
                        study_id,
                        {
                            "consent_use_uploaded_data": True,
                            "upload_own_data": True,
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
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        consent_ok = _upload_data_consent_given(info_pre) or bool(consent_value and "agree" in consent_value)
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
                info_pre,
                no_update,
            )
        
        # Parse upload contents (gzip-compressed by the client, or a raw data URL)
        decoded = decode_upload_bytes(upload_contents)
        if decoded is None:
            print(f"ERROR: Invalid upload format for file {filename}")
            return (
                current_time,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                dict(user_info or {}),
                no_update,
            )

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
            convert_df_to_dict(new_df),
            events_store_for_window(new_events_df, new_df),
            False,  # is_example_data = False for uploaded files
            str(filename or ""),  # store the original filename
            False,  # reset randomization flag for new data
            random_start,  # Update initial slider value
            info,
            current_time if triggered == "prediction-data-usage-consent" else no_update,
        )


# Nightscout data load callback
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
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
) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, Dict[str, Any], Any]:
    """Load CGM data from a Nightscout server URL."""
    if not n_clicks:
        raise PreventUpdate

    _no = (no_update,) * 8

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

    consent_ok = _upload_data_consent_given(info_pre) or bool(consent_value and "agree" in consent_value)
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
            convert_df_to_dict(new_df),
            events_store_for_window(new_events_df, new_df),
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
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('time-slider', 'value', allow_duplicate=True),
     Output('initial-slider-value', 'data', allow_duplicate=True)],  # Add initial slider value update
    [Input('use-example-data-button', 'n_clicks')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_example_data_button(
    example_button_clicks: Optional[int],
    user_info: Optional[Dict[str, Any]],
) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int, int]:
    """Handle use example data button click"""
    if not example_button_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    with start_action(action_type=u"handle_example_data_button"):
        current_time = int(time.time() * 1000)
        
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        info_dict: Dict[str, Any] = dict(user_info or {})
        new_df, new_events_df, source_name, random_start = _apply_generic_round_selection(
            info_dict,
            info_dict.get("rounds"),
            points,
        )
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
        
        print(f"DEBUG: Generated new random start position for example data: {random_start}")
        
        return (current_time,
               convert_df_to_dict(new_df),
               events_store_for_window(new_events_df, new_df),
               True,  # is_example_data = True for public data
               source_name,
               False,  # reset randomization flag for new data
               random_start,  # Set slider to the random start position
               random_start)  # Update initial slider value


# Separate callback for time slider
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True)],
    [Input('time-slider', 'value')],
    [State('user-info-store', 'data'),
     State('current-window-df', 'data')],
    prevent_initial_call=True
)
def handle_time_slider(
    slider_value: Optional[int],
    user_info: Optional[Dict[str, Any]],
    current_df_data: Optional[Dict[str, List[Any]]],
) -> Tuple[Any, Any, Any]:
    """Handle time slider changes (slices the window server-side from the dataset).

    The re-slice comes from ``load_dataset`` which zeroes the ``prediction`` column
    (datasets are immutable; predictions live only in the window). The slider is
    ``persistence=True`` and always mounted, so it re-fires on every layout rebuild
    (page nav, resume, language change) with its persisted value -- which re-slices
    the SAME window and would overwrite ``current-window-df``, wiping any
    predictions already there (a resumed in-progress round, or the ``--prefill``
    seed). Guard: if the freshly-sliced window covers the same timestamps as the
    current one, no-op so those predictions survive. A genuine move to a different
    window still returns a fresh (legitimately zeroed) window.

    Moving the window also re-trims the events store to it (see
    ``events_within_window``) so the two never drift apart.
    """
    if slider_value is None or not user_info:
        return no_update, no_update, no_update

    with start_action(action_type=u"handle_time_slider", slider_value=slider_value):
        current_time = int(time.time() * 1000)

        full_df, events_df = load_dataset(resolve_dataset_identity(user_info))

        # Ensure we don't go beyond the available data
        points = max(MIN_POINTS, min(MAX_POINTS, DEFAULT_POINTS))
        max_start = len(full_df) - points
        safe_slider_value = min(slider_value, max_start)
        safe_slider_value = max(0, safe_slider_value)

        new_df = full_df.slice(safe_slider_value, points)
        new_dict = convert_df_to_dict(new_df)

        # Same window as already displayed (slider mount / persistence re-fire) ->
        # preserve the current window (and its predictions) instead of clobbering.
        if current_df_data and current_df_data.get('time') == new_dict.get('time'):
            return no_update, no_update, no_update

        return current_time, new_dict, events_store_for_window(events_df, new_df)


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

        return current_time, convert_df_to_dict(anchor_predictions_at_boundary(df))

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

                return current_time, convert_df_to_dict(anchor_predictions_at_boundary(df))

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
    # B/C keep the Source blank until a file is uploaded.
    if _is_upload_gated(user_info):
        return ""
    if fmt == "B":
        return ""
    return "example.csv"


@app.callback(
    Output("generic-source-metadata-display", "children"),
    [
        Input("url", "pathname"),
        Input("data-source-name", "data"),
        Input("current-window-df", "data"),
        Input("events-df", "data"),
        Input("interface-language", "data"),
        Input("user-info-store", "data"),
        Input("is-example-data", "data"),
    ],
    prevent_initial_call=False,
)
def update_generic_source_metadata_display(
    pathname: str,
    source_name: Optional[str],
    window_df_data: Optional[Dict[str, List[Any]]],
    events_df_data: Optional[Dict[str, List[Any]]],
    interface_language: Optional[str],
    user_info: Optional[Dict[str, Any]],
    is_example_data: Optional[bool],
) -> str:
    if pathname != "/prediction":
        return ""

    locale = normalize_locale(interface_language)
    is_example = bool(is_example_data) if is_example_data is not None else bool(
        (user_info or {}).get("is_example_data", True)
    )
    empty_events = pl.DataFrame(
        {
            "time": [],
            "event_type": [],
            "event_subtype": [],
            "insulin_value": [],
        }
    )
    if not window_df_data or not window_df_data.get("time"):
        if not is_example:
            return format_source_notes(locale=locale, show_carbs_info_note=True)
        return ""

    window_df = reconstruct_dataframe_from_dict(window_df_data)
    events_df = (
        reconstruct_events_dataframe_from_dict(events_df_data)
        if events_df_data
        else empty_events
    )
    return _build_source_metadata_line(
        source_name=str(source_name or ""),
        user_info=user_info,
        is_example_data=is_example,
        window_df=window_df,
        events_df=events_df,
        locale=locale,
    )

# Add callback for random slider initialization when prediction page components are ready
@app.callback(
    [Output('time-slider', 'value', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True)],
    [Input('time-slider', 'max')],  # Triggers when slider is created and max is set
    [State('url', 'pathname'),
     State('user-info-store', 'data'),
     State('randomization-initialized', 'data'),
     State('initial-slider-value', 'data')],
    prevent_initial_call=True
)
def randomize_slider_on_prediction_page(slider_max: int, pathname: str, user_info: Optional[Dict[str, Any]],
                                       randomization_initialized: bool,
                                       initial_slider_value: Optional[int]) -> Tuple[int, bool]:
    """Set slider to a random valid window start when slider mounts on prediction page. Returns slider value and updated randomization flag."""
    if pathname == '/prediction' and user_info and slider_max is not None and not randomization_initialized:
        # Use the stored initial slider value if available
        if initial_slider_value is not None:
            return initial_slider_value, True
        # Otherwise generate a new random start (dataset loaded server-side)
        full_df, _ = load_dataset(resolve_dataset_identity(user_info))
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
    consent_ok = _upload_data_consent_given(info)
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
    [State('interface-language', 'data')],
    prevent_initial_call=True
)
def reset_upload_on_example_data(
    example_button_clicks: Optional[int],
    interface_language: Optional[str],
) -> Tuple[Optional[html.Div], int, None, None]:
    """Reset upload component and show message when example data button is clicked"""
    if not example_button_clicks:
        return no_update, no_update, no_update, no_update

    with start_action(action_type=u"reset_upload_on_example_data"):
        # This button switches to the example dataset; size the slider to it.
        full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
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

def anchor_predictions_at_boundary(df: pl.DataFrame) -> pl.DataFrame:
    """Make the drawn prediction path start where the known glucose line ends.

    The boundary slot (first index of the hidden hour) is shared by the last
    known glucose point and the first predictable point. A stroke that starts
    later leaves that slot empty, so the red prediction line looked detached
    from the blue line for some draws and joined for others. Anchoring fills
    the boundary slot with the ground truth there -- the same rule the very
    first stroke already applies -- and linearly interpolates the slots up to
    the user's first drawn point, exactly like ``create_intermediate_predictions``
    does inside a stroke.
    """
    boundary_idx = len(df) - PREDICTION_HOUR_OFFSET
    if not 0 <= boundary_idx < len(df):
        return df
    predictions = df.get_column("prediction").to_list()
    if predictions[boundary_idx] not in (0.0, None):
        return df
    first_drawn_idx = next(
        (i for i in range(boundary_idx + 1, len(predictions))
         if predictions[i] not in (0.0, None)),
        None,
    )
    if first_drawn_idx is None:
        return df
    anchor_value = df.get_column("gl")[boundary_idx]
    if anchor_value is None:
        return df
    anchor_value = float(anchor_value)
    first_value = float(predictions[first_drawn_idx])
    steps = first_drawn_idx - boundary_idx
    for offset in range(steps):
        predictions[boundary_idx + offset] = (
            anchor_value + (first_value - anchor_value) * (offset / steps)
        )
    return df.with_columns(pl.Series("prediction", predictions, dtype=pl.Float64))


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
    """Convert an events Polars DataFrame to a session-store dictionary.

    Do NOT use this to write the `events-df` store -- it would ship the frame as
    given, and a whole-subject event log there is what froze production on
    2026-07-28. Use ``events_store_for_window`` so the payload stays window-sized.
    """
    return events_dataframe_to_store_dict(df)

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



def register_faq_callbacks(app_instance: dash.Dash) -> None:
    from sugar_sugar.faq_board import add_faq_question, add_faq_reply, faq_board_enabled

    @app_instance.callback(
        [Output("faq-board", "children"),
         Output("faq-ask-status", "children"),
         Output("faq-ask-text", "value")],
        Input("faq-ask-submit", "n_clicks"),
        [State("faq-ask-text", "value"),
         State("faq-ask-name", "value"),
         State("faq-ask-tags", "value"),
         State("faq-ask-section", "value"),
         State("interface-language", "data")],
        prevent_initial_call=True,
    )
    def submit_faq_question(
        n_clicks: Optional[int],
        text: Optional[str],
        name: Optional[str],
        tags: Optional[list[str]],
        section: Optional[str],
        interface_language: Optional[str],
    ) -> tuple[Any, str, str]:
        if not n_clicks or not faq_board_enabled():
            raise PreventUpdate
        locale = normalize_locale(interface_language)
        posted = add_faq_question(text=text or "", section=section or "participant", tags=tags, name=name or "")
        if posted is None:
            return no_update, t("ui.faq.ask_empty", locale=locale), no_update
        return faq_board_children(locale=locale), t("ui.faq.ask_thanks", locale=locale), ""

    @app_instance.callback(
        Output("faq-board", "children", allow_duplicate=True),
        Input({"type": "faq-reply-submit", "index": ALL}, "n_clicks"),
        [State({"type": "faq-reply-text", "index": ALL}, "value"),
         State({"type": "faq-reply-section", "index": ALL}, "value"),
         State({"type": "faq-reply-submit", "index": ALL}, "id"),
         State("interface-language", "data")],
        prevent_initial_call=True,
    )
    def submit_faq_reply(
        n_clicks: list[Optional[int]],
        texts: list[Optional[str]],
        sections: list[Optional[str]],
        ids: list[dict[str, str]],
        interface_language: Optional[str],
    ) -> Any:
        if not n_clicks or not any(n_clicks) or not faq_board_enabled():
            raise PreventUpdate
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        qid = str(triggered.get("index") or "")
        locale = normalize_locale(interface_language)
        for index, button_id in enumerate(ids or []):
            if str(button_id.get("index")) == qid:
                add_faq_reply(
                    qid,
                    text=str(texts[index] or ""),
                    section=str(sections[index] or "developer"),
                )
                break
        return faq_board_children(locale=locale)


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
    register_faq_callbacks(app)
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

    format_arg = _arg_value(argv, "--format")
    env["_CHART_FORMAT"] = format_arg if format_arg in ("A", "B", "C") else "A"

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
    format: str = typer.Option("A", "--format", help="Data-source format: A=generic, B=my data only (upload gate), C=mixed"),
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
    os.environ["_CHART_FORMAT"] = format if format in ("A", "B", "C") else "A"
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
    threads: Optional[int] = typer.Option(None, "--threads", help="Threads per worker (>1 switches gunicorn to gthread)"),
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
    # Sync workers serve exactly one request each, start to finish -- including the
    # time spent reading a slow client's request body. So `workers` is the hard
    # ceiling on concurrent players, and one slow upload stalls everyone (a public
    # monitor timed out during the 2026-07-28 slowdown, not just the player).
    # >1 makes gunicorn use its gthread worker; default stays 1 so behaviour is
    # unchanged unless the deployment opts in via --threads / GUNICORN_THREADS.
    thread_count: int = threads if threads is not None else int(os.getenv("GUNICORN_THREADS", "1"))
    bind: str = f"{bind_host}:{bind_port}"
    command: list[str] = [
        "gunicorn",
        "sugar_sugar.wsgi:application",
        "--bind",
        bind,
        "--workers",
        str(worker_count),
        "--threads",
        str(thread_count),
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
        threads=thread_count,
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
