"""Share-page component.

Renders a Dash page at `/share/<share_id>` that lets the user show off
their Sugar Sugar performance on social networks.

Public API
----------
- ``create_share_layout(share_record, share_id, share_url, *, locale)``:
    Returns a ``html.Div`` for the Dash page.  ``share_record`` is what
    ``share_store.load_share`` returned; it must contain at minimum a
    ``rounds`` list matching the shape used on ``/final``.
- ``build_synthesis_figure(share_record, *, locale)``:
    Builds the grey/blue synthesis ``go.Figure`` shown in the page and
    embedded inside the share card.
- ``build_share_card_figure(share_record, share_url, *, locale)``:
    Builds the 1200x630 composite Plotly figure used for the downloadable
    PNG and the Open Graph preview.
- ``compute_aggregate_stats(rounds)``:
    Returns a dict with ``mae_mgdl``, ``rmse_mgdl``, ``mape``,
    ``rounds_played``, ``pairs`` used by both the layout and the LLM hook.
"""
from __future__ import annotations

import math
import urllib.parse
from typing import Any, Optional

import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.encouragement import encouragement_text, pick_bracket
from sugar_sugar.i18n import normalize_locale, t


# ---------------------------------------------------------------------------
# Pure data helpers
# ---------------------------------------------------------------------------

def _parse_float(cell: Any) -> Optional[float]:
    """Robust float parse for prediction_table_data cells (they're strings)."""
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        value: float = float(cell)
        return None if math.isnan(value) else value
    if isinstance(cell, str):
        if cell.strip() in ("", "-"):
            return None
        try:
            return float(cell)
        except ValueError:
            return None
    return None


def _collect_aligned_series(rounds: list[dict[str, Any]]) -> tuple[list[list[float]], list[list[float]], int]:
    """Group actual/predicted values across rounds by 5-min slot index.

    Returns ``(actual_by_slot, predicted_by_slot, max_slots)``.
    ``actual_by_slot[i]`` is every non-missing ground-truth value observed
    at slot i across all rounds; likewise for predictions.
    """
    actual_by_slot: list[list[float]] = []
    predicted_by_slot: list[list[float]] = []

    for round_info in rounds:
        table = round_info.get("prediction_table_data") or []
        if len(table) < 2:
            continue
        actual_row = table[0] or {}
        pred_row = table[1] or {}
        window_size: int = int(round_info.get("prediction_window_size") or 0)
        slots: int = window_size if window_size > 0 else max(
            (int(k[1:]) for k in actual_row.keys() if isinstance(k, str) and k.startswith("t") and k[1:].isdigit()),
            default=-1,
        ) + 1

        while len(actual_by_slot) < slots:
            actual_by_slot.append([])
            predicted_by_slot.append([])

        for i in range(slots):
            key: str = f"t{i}"
            a: Optional[float] = _parse_float(actual_row.get(key))
            if a is not None:
                actual_by_slot[i].append(a)
            p: Optional[float] = _parse_float(pred_row.get(key))
            if p is not None:
                predicted_by_slot[i].append(p)

    return actual_by_slot, predicted_by_slot, len(actual_by_slot)


def compute_aggregate_stats(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across every (actual, predicted) pair in every round."""
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    pct_errors: list[float] = []

    for round_info in rounds:
        table = round_info.get("prediction_table_data") or []
        if len(table) < 2:
            continue
        actual_row = table[0] or {}
        pred_row = table[1] or {}
        for key, raw_actual in actual_row.items():
            if key == "metric" or not (isinstance(key, str) and key.startswith("t")):
                continue
            a = _parse_float(raw_actual)
            p = _parse_float(pred_row.get(key))
            if a is None or p is None:
                continue
            diff: float = a - p
            abs_errors.append(abs(diff))
            sq_errors.append(diff * diff)
            if a != 0:
                pct_errors.append(abs(diff / a) * 100.0)

    if abs_errors:
        mae: float = sum(abs_errors) / len(abs_errors)
        rmse: float = math.sqrt(sum(sq_errors) / len(sq_errors))
        mape: float = (sum(pct_errors) / len(pct_errors)) if pct_errors else float("nan")
    else:
        mae = rmse = mape = float("nan")

    accuracy: float = max(0.0, 100.0 - mape) if not math.isnan(mape) else float("nan")

    return {
        "mae_mgdl": mae,
        "rmse_mgdl": rmse,
        "mape": mape,
        "accuracy": accuracy,
        "rounds_played": len(rounds),
        "pairs": len(abs_errors),
    }


def _best_ranking(share_record: dict[str, Any]) -> Optional[int]:
    """Pick the best (lowest) ranking across all formats the user has played.

    Reads the schema-2 ``rankings`` block baked into the share record at
    save time.  Falls back to the legacy schema-1 ``user_info.rankings``
    dict so older share URLs still render.
    """
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    best: Optional[int] = None

    per_format: list[dict[str, Any]] = list(rankings.get("per_format") or [])
    for entry in per_format:
        try:
            rank = int(entry.get("rank"))
        except (TypeError, ValueError):
            continue
        if rank <= 0:
            continue
        if best is None or rank < best:
            best = rank

    overall = rankings.get("overall")
    if isinstance(overall, dict):
        try:
            r_overall = int(overall.get("rank"))
            if r_overall > 0 and (best is None or r_overall < best):
                best = r_overall
        except (TypeError, ValueError):
            pass

    if best is not None:
        return best

    # Legacy fallback for schema 1.
    legacy: dict[str, Any] = dict((share_record.get("user_info") or {}).get("rankings") or {})
    for value in legacy.values():
        try:
            rank = int(value)
        except (TypeError, ValueError):
            continue
        if rank <= 0:
            continue
        if best is None or rank < best:
            best = rank
    return best


def _round_metrics(round_info: dict[str, Any]) -> dict[str, float]:
    """Compute MAE/RMSE/MAPE for a single round from its prediction_table_data."""
    table = round_info.get("prediction_table_data") or []
    if len(table) < 2:
        return {}
    actual_row = table[0] or {}
    pred_row = table[1] or {}
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    pct_errors: list[float] = []
    for key, raw_actual in actual_row.items():
        if key == "metric" or not (isinstance(key, str) and key.startswith("t")):
            continue
        a = _parse_float(raw_actual)
        p = _parse_float(pred_row.get(key))
        if a is None or p is None:
            continue
        d = a - p
        abs_errors.append(abs(d))
        sq_errors.append(d * d)
        if a != 0:
            pct_errors.append(abs(d / a) * 100.0)
    if not abs_errors:
        return {}
    return {
        "pairs": float(len(abs_errors)),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(sq_errors) / len(sq_errors)),
        "mape": (sum(pct_errors) / len(pct_errors)) if pct_errors else float("nan"),
    }


# Format code -> human label.  Kept local to avoid a cyclic app->share->app
# import; the strings mirror ``ui.startup.format_{a,b,c}_label`` in the
# translation YAMLs.  ``_resolve_format_label`` wraps ``t()`` and handles
# unknown codes gracefully.
def _resolve_format_label(code: str, *, locale: str) -> str:
    code = str(code or "").strip().upper()
    if code == "A":
        return t("ui.startup.format_a_label", locale=locale)
    if code == "B":
        return t("ui.startup.format_b_label", locale=locale)
    if code == "C":
        return t("ui.startup.format_c_label", locale=locale)
    return code


def _format_number(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_synthesis_figure(share_record: dict[str, Any], *, locale: str) -> go.Figure:
    """Build the grey/blue synthesis chart across all played rounds."""
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    actual_slots, predicted_slots, n_slots = _collect_aligned_series(rounds)

    xs: list[int] = list(range(n_slots))
    actual_mean: list[Optional[float]] = [
        sum(v) / len(v) if v else None for v in actual_slots
    ]
    actual_min: list[Optional[float]] = [min(v) if v else None for v in actual_slots]
    actual_max: list[Optional[float]] = [max(v) if v else None for v in actual_slots]

    pred_mean: list[Optional[float]] = [
        sum(v) / len(v) if v else None for v in predicted_slots
    ]
    pred_min: list[Optional[float]] = [min(v) if v else None for v in predicted_slots]
    pred_max: list[Optional[float]] = [max(v) if v else None for v in predicted_slots]

    # Range bars: use Scatter with mode='lines' per slot -> cheaper and renders
    # cleanly in kaleido without the multi-bar layering quirks.
    fig: go.Figure = go.Figure()

    # Grey "actual glucose" range shading drawn per slot.
    for x, lo, hi in zip(xs, actual_min, actual_max):
        if lo is None or hi is None or hi == lo:
            continue
        fig.add_shape(
            type="rect",
            x0=x - 0.35, x1=x + 0.35,
            y0=lo, y1=hi,
            fillcolor="rgba(120,120,120,0.35)",
            line=dict(width=0),
            layer="below",
        )

    # Blue "your prediction" range shading (narrower, rendered on top of grey).
    for x, lo, hi in zip(xs, pred_min, pred_max):
        if lo is None or hi is None or hi == lo:
            continue
        fig.add_shape(
            type="rect",
            x0=x - 0.2, x1=x + 0.2,
            y0=lo, y1=hi,
            fillcolor="rgba(33,133,208,0.30)",
            line=dict(width=0),
            layer="below",
        )

    # Actual mean: black line + dots.
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=actual_mean,
            mode="lines+markers",
            name=t("ui.share.synthesis.legend_actual", locale=loc),
            line=dict(color="rgba(20,20,20,0.9)", width=2),
            marker=dict(color="rgba(20,20,20,0.9)", size=7),
            connectgaps=False,
        )
    )

    # Prediction mean: blue, semi-transparent line + dots.
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=pred_mean,
            mode="lines+markers",
            name=t("ui.share.synthesis.legend_prediction", locale=loc),
            line=dict(color="rgba(21,101,192,0.80)", width=3),
            marker=dict(color="rgba(21,101,192,0.95)", size=7),
            connectgaps=False,
        )
    )

    # Prediction-zone separator: dashed vertical line where the hidden area begins.
    if n_slots > PREDICTION_HOUR_OFFSET:
        sep_x: float = n_slots - PREDICTION_HOUR_OFFSET - 0.5
        fig.add_shape(
            type="line",
            x0=sep_x, x1=sep_x,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="rgba(21,101,192,0.55)", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=sep_x, y=1.02,
            xref="x", yref="paper",
            text=t("ui.share.synthesis.prediction_region", locale=loc),
            showarrow=False,
            font=dict(color="rgba(21,101,192,0.85)", size=12),
            align="left",
        )

    fig.update_layout(
        title=dict(
            text=t("ui.share.synthesis.title", locale=loc),
            x=0.02, xanchor="left",
            font=dict(size=18),
        ),
        xaxis=dict(
            title=t("ui.share.synthesis.x_axis", locale=loc),
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(15,23,42,0.08)",
        ),
        yaxis=dict(
            title=t("ui.share.synthesis.y_axis", locale=loc),
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(15,23,42,0.08)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,
            xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    return fig


def build_share_card_figure(
    share_record: dict[str, Any],
    *,
    share_url: str,
    locale: str,
) -> go.Figure:
    """Build a single 1200x630 figure used as the social-share PNG."""
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    stats: dict[str, Any] = compute_aggregate_stats(rounds)
    user_info: dict[str, Any] = dict(share_record.get("user_info") or {})
    name: str = str(user_info.get("name") or user_info.get("study_id") or "").strip()

    # Two-column subplot: left is a text column (invisible axes, annotations),
    # right is the synthesis chart.
    fig: go.Figure = make_subplots(
        rows=1, cols=2,
        column_widths=[0.35, 0.65],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.04,
    )

    # Embed the synthesis chart (without its own title/legend) into col 2.
    syn: go.Figure = build_synthesis_figure(share_record, locale=loc)
    for trace in syn.data:
        fig.add_trace(trace, row=1, col=2)
    for shape in (syn.layout.shapes or []):
        new_shape = shape.to_plotly_json()
        new_shape["xref"] = "x2"
        # y-refs that were 'paper' stay paper; explicit 'y' becomes 'y2'.
        if new_shape.get("yref") == "y":
            new_shape["yref"] = "y2"
        fig.add_shape(**new_shape)

    # Left panel: stats as annotations on hidden axes.
    mae: float = stats.get("mae_mgdl") or float("nan")
    rmse: float = stats.get("rmse_mgdl") or float("nan")
    accuracy: float = stats.get("accuracy") or float("nan")
    rounds_played: int = int(stats.get("rounds_played") or 0)
    best_rank: Optional[int] = _best_ranking(share_record)
    bracket: str = pick_bracket(mae)
    encourage: str = encouragement_text(stats, loc)

    label_title: str = t("ui.share.title", locale=loc)
    stats_heading: str = t("ui.share.stats_heading", locale=loc)
    mae_label: str = t("ui.share.stat_mae", locale=loc)
    rmse_label: str = t("ui.share.stat_rmse", locale=loc)
    rounds_label: str = t("ui.share.stat_rounds", locale=loc)
    acc_label: str = t("ui.share.stat_accuracy", locale=loc)
    rank_label: str = t("ui.share.stat_ranking", locale=loc)
    no_rank: str = t("ui.share.stat_no_ranking", locale=loc)
    footer: str = t("ui.share.card_footer", locale=loc, url=share_url)

    # Build the per-format / overall ranking lines so they appear in the PNG.
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    per_format_entries: list[dict[str, Any]] = list(rankings.get("per_format") or [])
    overall_entry = rankings.get("overall") if isinstance(rankings.get("overall"), dict) else None
    played_formats: list[str] = list(share_record.get("played_formats") or [])

    ranking_strings: list[str] = []
    for entry in per_format_entries:
        try:
            r = int(entry.get("rank"))
            total = int(entry.get("total"))
        except (TypeError, ValueError):
            continue
        fmt = str(entry.get("format") or "")
        label = _resolve_format_label(fmt, locale=loc)
        ranking_strings.append(f"{label}: <b>#{r}</b> / {total}")
    if overall_entry is not None:
        try:
            o_rank = int(overall_entry.get("rank"))
            o_total = int(overall_entry.get("total"))
            ranking_strings.append(f"<b>Overall: #{o_rank}</b> / {o_total}")
        except (TypeError, ValueError):
            pass

    # Compact the layout: header block at the top, stats in a dense grid,
    # rankings underneath, encouragement at the bottom.  The share card is
    # 1200x630 so we have to be disciplined with vertical space.
    text_lines: list[tuple[float, str, int, str]] = [
        (0.97, f"<b>{label_title}</b>", 24, "rgba(15,23,42,1)"),
    ]
    if name:
        text_lines.append((0.91, name, 16, "rgba(71,85,105,1)"))

    # Stats mini-grid (one line).
    stats_inline: str = (
        f"<b>{_format_number(mae)}</b> mg/dL  {mae_label}   "
        f"<b>{_format_number(rmse)}</b> mg/dL  {rmse_label}   "
        f"<b>{_format_number(accuracy)}%</b>  {acc_label}   "
        f"<b>{rounds_played}</b>  {rounds_label}"
    )
    text_lines.append((0.83, stats_inline, 13, "rgba(15,23,42,1)"))

    # Ranking block.
    text_lines.append((0.74, f"<b>{t('ui.final.ranking_title', locale=loc)}</b>",
                       15, "rgba(21,101,192,1)"))
    base_y: float = 0.68
    for idx, rline in enumerate(ranking_strings):
        text_lines.append((base_y - idx * 0.065, rline, 14, "rgba(15,23,42,1)"))

    # Played formats.
    if played_formats:
        played_y = base_y - len(ranking_strings) * 0.065 - 0.02
        played_text: str = t(
            "ui.final.played_formats",
            locale=loc,
            formats=", ".join(_resolve_format_label(f, locale=loc) for f in played_formats),
        )
        text_lines.append((played_y, f"<i>{played_text}</i>", 12, "rgba(71,85,105,1)"))

    # Encouragement message at the bottom of the left column.
    text_lines.append((0.10, encourage, 14, "rgba(30,58,138,1)"))

    for y, text, size, color in text_lines:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=y,
            xanchor="left", yanchor="top",
            text=text,
            showarrow=False,
            font=dict(size=size, color=color),
            align="left",
        )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.02,
        xanchor="center", yanchor="bottom",
        text=f"<i>{footer}</i>",
        showarrow=False,
        font=dict(size=14, color="rgba(71,85,105,0.9)"),
    )

    # Hide the left-panel axes so it reads as a text block.
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Style the embedded synthesis axes (right col).
    fig.update_xaxes(
        title=t("ui.share.synthesis.x_axis", locale=loc),
        showgrid=True, gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
        row=1, col=2,
    )
    fig.update_yaxes(
        title=t("ui.share.synthesis.y_axis", locale=loc),
        showgrid=True, gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
        row=1, col=2,
    )

    fig.update_layout(
        width=1200, height=630,
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=0.98,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        margin=dict(l=40, r=40, t=60, b=60),
        title=dict(
            text=f"<b>Sugar Sugar - {t('ui.share.subtitle', locale=loc)}</b>",
            x=0.5, xanchor="center",
            y=0.98, yanchor="top",
            font=dict(size=16, color="rgba(15,23,42,0.9)"),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Dash layout
# ---------------------------------------------------------------------------

def _share_button(label: str, href: str, *, color: str, icon: str) -> html.A:
    """Render a pill-style social-share button."""
    return html.A(
        [html.I(className=f"fab {icon}", style={"marginRight": "8px"}),
         label],
        href=href,
        target="_blank",
        rel="noopener noreferrer",
        className="share-btn",
        style={
            "backgroundColor": color,
            "color": "white",
            "padding": "12px 20px",
            "borderRadius": "999px",
            "textDecoration": "none",
            "fontWeight": "600",
            "fontSize": "15px",
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "6px",
        },
    )


def create_expired_layout(*, locale: str) -> html.Div:
    """Minimal page shown when a share URL does not resolve."""
    loc: str = normalize_locale(locale)
    return html.Div(
        [
            html.H1(t("ui.share.expired_title", locale=loc),
                    style={"fontSize": "32px", "marginBottom": "16px", "color": "#0f172a"}),
            html.P(t("ui.share.expired_body", locale=loc),
                   style={"fontSize": "18px", "color": "#475569", "maxWidth": "560px",
                          "marginBottom": "28px", "lineHeight": "1.6"}),
            html.A(
                t("ui.share.expired_cta", locale=loc),
                href="/",
                className="ui green button",
                style={"padding": "14px 28px", "fontSize": "18px"},
            ),
        ],
        className="info-page",
        disable_n_clicks=True,
        style={"textAlign": "center"},
    )


def create_share_layout(
    share_record: dict[str, Any],
    *,
    share_id: str,
    share_url: str,
    locale: str,
) -> html.Div:
    """Render the full share page for a valid share record."""
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    stats: dict[str, Any] = compute_aggregate_stats(rounds)
    user_info: dict[str, Any] = dict(share_record.get("user_info") or {})
    name: str = str(user_info.get("name") or user_info.get("study_id") or "").strip()

    mae: float = stats.get("mae_mgdl") or float("nan")
    rmse: float = stats.get("rmse_mgdl") or float("nan")
    accuracy: float = stats.get("accuracy") or float("nan")
    rounds_played: int = int(stats.get("rounds_played") or 0)
    best_rank: Optional[int] = _best_ranking(share_record)
    invite_text: str = t(
        "ui.share.invite_text",
        locale=loc,
        mae=_format_number(mae),
        rounds=rounds_played,
    )
    encourage: str = encouragement_text(stats, loc)

    encoded_url: str = urllib.parse.quote(share_url, safe="")
    encoded_text: str = urllib.parse.quote(invite_text, safe="")

    share_buttons: html.Div = html.Div(
        [
            _share_button(
                t("ui.share.share_on_x", locale=loc),
                f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}",
                color="#000000", icon="fa-x-twitter",
            ),
            _share_button(
                t("ui.share.share_on_facebook", locale=loc),
                f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}",
                color="#1877F2", icon="fa-facebook",
            ),
            _share_button(
                t("ui.share.share_on_whatsapp", locale=loc),
                f"https://api.whatsapp.com/send?text={urllib.parse.quote(invite_text + ' ' + share_url, safe='')}",
                color="#25D366", icon="fa-whatsapp",
            ),
            _share_button(
                t("ui.share.share_on_linkedin", locale=loc),
                f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}",
                color="#0A66C2", icon="fa-linkedin",
            ),
        ],
        className="share-buttons",
        style={"display": "flex", "flexWrap": "wrap", "gap": "10px",
               "justifyContent": "center", "marginTop": "16px"},
    )

    # Stat card -- rendered with inline styles so it looks decent on both desktop and mobile.
    def stat_tile(label: str, value: str, sub: str = "") -> html.Div:
        return html.Div(
            [
                html.Div(value, style={"fontSize": "32px", "fontWeight": "800",
                                        "color": "#0f172a", "lineHeight": "1.1"}),
                html.Div(label, style={"fontSize": "13px", "fontWeight": "600",
                                        "color": "#64748b", "letterSpacing": "0.04em",
                                        "textTransform": "uppercase", "marginTop": "4px"}),
                html.Div(sub, style={"fontSize": "12px", "color": "#94a3b8",
                                      "marginTop": "2px"}) if sub else None,
            ],
            style={
                "background": "white",
                "borderRadius": "14px",
                "padding": "16px 20px",
                "boxShadow": "0 4px 14px rgba(15,23,42,0.08)",
                "minWidth": "140px",
                "flex": "1 1 140px",
                "textAlign": "left",
            },
            disable_n_clicks=True,
        )

    stats_row: html.Div = html.Div(
        [
            stat_tile(t("ui.share.stat_mae", locale=loc), f"{_format_number(mae)} mg/dL"),
            stat_tile(t("ui.share.stat_rmse", locale=loc), f"{_format_number(rmse)} mg/dL"),
            stat_tile(t("ui.share.stat_accuracy", locale=loc), f"{_format_number(accuracy)}%"),
            stat_tile(t("ui.share.stat_rounds", locale=loc), str(rounds_played)),
            stat_tile(
                t("ui.share.stat_ranking", locale=loc),
                f"#{best_rank}" if best_rank else t("ui.share.stat_no_ranking", locale=loc),
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "14px",
               "marginTop": "20px", "justifyContent": "center"},
        disable_n_clicks=True,
    )

    # ---------- Ranking block (per-format + overall) ----------
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    per_format_entries: list[dict[str, Any]] = list(rankings.get("per_format") or [])
    overall_entry: Optional[dict[str, Any]] = rankings.get("overall") if isinstance(rankings.get("overall"), dict) else None

    ranking_lines: list[Any] = []
    for entry in per_format_entries:
        fmt = str(entry.get("format") or "")
        try:
            rank = int(entry.get("rank"))
            total = int(entry.get("total"))
        except (TypeError, ValueError):
            continue
        ranking_lines.append(
            html.Li(
                t(
                    "ui.final.ranking_format_line",
                    locale=loc,
                    format=_resolve_format_label(fmt, locale=loc),
                    rank=rank,
                    total=total,
                ),
                style={"marginBottom": "4px"},
                disable_n_clicks=True,
            )
        )
    if overall_entry is not None:
        try:
            o_rank = int(overall_entry.get("rank"))
            o_total = int(overall_entry.get("total"))
            ranking_lines.append(
                html.Li(
                    t("ui.final.ranking_overall_line", locale=loc, rank=o_rank, total=o_total),
                    style={"marginBottom": "4px", "fontWeight": "700"},
                    disable_n_clicks=True,
                )
            )
        except (TypeError, ValueError):
            pass

    ranking_card: Optional[html.Div] = None
    if ranking_lines:
        ranking_card = html.Div(
            [
                html.H3(
                    t("ui.final.ranking_title", locale=loc),
                    style={"margin": "0 0 10px 0", "color": "#1565c0",
                           "fontSize": "20px", "fontWeight": "700"},
                    disable_n_clicks=True,
                ),
                html.Ul(
                    ranking_lines,
                    style={"listStyle": "none", "padding": "0", "margin": "0",
                           "fontSize": "16px", "color": "#0f172a"},
                    disable_n_clicks=True,
                ),
            ],
            style={
                "background": "white",
                "borderRadius": "14px",
                "padding": "18px 22px",
                "boxShadow": "0 4px 14px rgba(15,23,42,0.08)",
                "marginTop": "20px",
                "maxWidth": "760px",
                "marginLeft": "auto",
                "marginRight": "auto",
            },
            disable_n_clicks=True,
        )

    # ---------- Played formats line ----------
    played_formats: list[str] = list(share_record.get("played_formats") or [])
    # Fallback: if the record doesn't carry played_formats (older schema),
    # derive from round dicts.
    if not played_formats:
        derived: set[str] = {str(r.get("format") or "") for r in rounds}
        derived.discard("")
        played_formats = sorted(derived, key=lambda x: {"C": 0, "B": 1, "A": 2}.get(x, 999))

    played_line: Optional[html.Div] = None
    if played_formats:
        played_line = html.Div(
            t(
                "ui.final.played_formats",
                locale=loc,
                formats=", ".join(_resolve_format_label(f, locale=loc) for f in played_formats),
            ),
            style={"marginTop": "14px", "textAlign": "center",
                   "fontSize": "15px", "color": "#475569", "fontStyle": "italic"},
            disable_n_clicks=True,
        )

    # ---------- Per-round summary table ----------
    per_round_rows: list[Any] = []
    for idx, rnd in enumerate(rounds, start=1):
        m = _round_metrics(rnd)
        if not m:
            continue
        fmt = str(rnd.get("format") or "")
        per_round_rows.append(
            html.Tr(
                [
                    html.Td(str(rnd.get("round_number") or idx),
                            style={"padding": "6px 12px"}),
                    html.Td(_resolve_format_label(fmt, locale=loc) if fmt else "-",
                            style={"padding": "6px 12px"}),
                    html.Td(f"{int(m['pairs'])}", style={"padding": "6px 12px"}),
                    html.Td(f"{m['mae']:.2f}", style={"padding": "6px 12px", "fontWeight": "600"}),
                    html.Td(f"{m['rmse']:.2f}", style={"padding": "6px 12px"}),
                    html.Td(f"{m['mape']:.2f}" if not math.isnan(m['mape']) else "-",
                            style={"padding": "6px 12px"}),
                ],
                disable_n_clicks=True,
            )
        )

    per_round_card: Optional[html.Div] = None
    if per_round_rows:
        header_cells = [
            html.Th("#", style={"padding": "8px 12px", "textAlign": "left"}),
            html.Th(t("ui.startup.format_heading", locale=loc) if False else "Format",  # keep "Format" literal; same across locales
                    style={"padding": "8px 12px", "textAlign": "left"}),
            html.Th("Pairs", style={"padding": "8px 12px", "textAlign": "left"}),
            html.Th("MAE", style={"padding": "8px 12px", "textAlign": "left"}),
            html.Th("RMSE", style={"padding": "8px 12px", "textAlign": "left"}),
            html.Th("MAPE", style={"padding": "8px 12px", "textAlign": "left"}),
        ]
        per_round_card = html.Div(
            [
                html.H3(
                    t("ui.final.per_round_metrics", locale=loc),
                    style={"margin": "0 0 10px 0", "color": "#1565c0",
                           "fontSize": "20px", "fontWeight": "700"},
                    disable_n_clicks=True,
                ),
                html.Table(
                    [
                        html.Thead(html.Tr(header_cells, disable_n_clicks=True),
                                   disable_n_clicks=True),
                        html.Tbody(per_round_rows, disable_n_clicks=True),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse",
                           "fontSize": "15px", "color": "#0f172a"},
                    disable_n_clicks=True,
                ),
            ],
            style={
                "background": "white",
                "borderRadius": "14px",
                "padding": "18px 22px",
                "boxShadow": "0 4px 14px rgba(15,23,42,0.08)",
                "marginTop": "20px",
                "maxWidth": "760px",
                "marginLeft": "auto",
                "marginRight": "auto",
                "overflowX": "auto",
            },
            disable_n_clicks=True,
        )

    synthesis_card: html.Div = html.Div(
        [
            dcc.Graph(
                figure=build_synthesis_figure(share_record, locale=loc),
                config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False},
                style={"height": "440px"},
            ),
            html.Div(
                t(
                    "ui.share.synthesis.caption_close" if not math.isnan(mae) and mae < 10
                    else "ui.share.synthesis.caption_far",
                    locale=loc,
                ),
                style={"fontSize": "14px", "color": "#475569", "textAlign": "center",
                       "padding": "6px 16px 16px 16px", "fontStyle": "italic"},
                disable_n_clicks=True,
            ),
        ],
        style={"background": "white", "borderRadius": "18px",
               "boxShadow": "0 8px 24px rgba(15,23,42,0.08)",
               "overflow": "hidden", "marginTop": "24px"},
        disable_n_clicks=True,
    )

    download_href: str = f"/share/{share_id}/image.png"

    action_buttons: html.Div = html.Div(
        [
            html.A(
                [html.I(className="fas fa-download", style={"marginRight": "8px"}),
                 t("ui.share.download_png", locale=loc)],
                href=download_href,
                download=f"sugar-sugar-{share_id}.png",
                className="ui green button",
                style={"padding": "14px 24px", "fontSize": "16px", "marginRight": "8px"},
            ),
            html.Button(
                [html.I(className="fas fa-link", style={"marginRight": "8px"}),
                 t("ui.share.copy_link", locale=loc)],
                id="share-copy-link-button",
                n_clicks=0,
                className="ui button",
                style={"padding": "14px 24px", "fontSize": "16px"},
            ),
            html.Span(
                t("ui.share.copy_link_success", locale=loc),
                id="share-copy-link-feedback",
                style={"marginLeft": "12px", "color": "#16a34a",
                       "fontWeight": "600", "opacity": "0",
                       "transition": "opacity 0.2s ease-in"},
                disable_n_clicks=True,
            ),
            html.Button(
                [html.I(className="fas fa-play", style={"marginRight": "8px"}),
                 t("ui.share.play_again", locale=loc)],
                id="share-play-again-button",
                n_clicks=0,
                className="ui button",
                style={"padding": "14px 24px", "fontSize": "16px", "marginLeft": "8px"},
            ),
        ],
        style={"marginTop": "20px", "textAlign": "center"},
        disable_n_clicks=True,
    )

    # Hidden span that holds the canonical share URL for the copy-link callback.
    url_store: html.Div = html.Div(
        share_url,
        id="share-url-value",
        style={"display": "none"},
        disable_n_clicks=True,
    )

    return html.Div(
        [
            url_store,
            html.Div(
                [
                    html.H1(
                        t("ui.share.title", locale=loc),
                        style={"fontSize": "clamp(28px,4vw,44px)", "margin": "0 0 4px 0",
                               "color": "#0f172a", "textAlign": "center"},
                    ),
                    html.P(
                        t("ui.share.subtitle", locale=loc),
                        style={"fontSize": "clamp(16px,2.5vw,20px)",
                               "color": "#475569", "textAlign": "center",
                               "margin": "0 0 4px 0"},
                        disable_n_clicks=True,
                    ),
                    html.P(
                        name,
                        style={"fontSize": "15px", "color": "#1e3a8a",
                               "textAlign": "center", "fontWeight": "600",
                               "margin": "0"},
                        disable_n_clicks=True,
                    ) if name else None,
                ],
                style={"paddingTop": "20px"},
                disable_n_clicks=True,
            ),

            stats_row,
            ranking_card,
            played_line,
            synthesis_card,
            per_round_card,

            html.Div(
                encourage,
                style={"marginTop": "22px", "padding": "18px 22px",
                       "background": "linear-gradient(135deg,#eff6ff,#ede9fe)",
                       "borderRadius": "14px", "fontSize": "17px",
                       "color": "#1e3a8a", "textAlign": "center",
                       "boxShadow": "0 3px 10px rgba(30,64,175,0.08)"},
                disable_n_clicks=True,
            ),

            action_buttons,
            share_buttons,

            html.Div(
                t("ui.share.download_png_hint", locale=loc),
                style={"fontSize": "13px", "color": "#94a3b8",
                       "textAlign": "center", "marginTop": "14px"},
                disable_n_clicks=True,
            ),
        ],
        className="share-page info-page",
        id="share-page",
        disable_n_clicks=True,
        style={
            "background": "linear-gradient(135deg,#eff6ff 0%,#f8fafc 40%,#fdf2f8 100%)",
            "maxWidth": "1100px",
        },
    )
