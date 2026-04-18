"""Share-page component.

Renders a Dash page at ``/share/<share_id>`` that lets the user show off
their Sugar Sugar performance on social networks.

Public API
----------
- ``create_share_layout(share_record, share_id, share_url, *, locale)``:
    Returns a ``html.Div`` for the Dash page.  ``share_record`` is what
    ``share_store.load_share`` returned; it must contain at minimum a
    ``rounds`` list matching the shape used on ``/final``.
- ``build_synthesis_figure(share_record, *, locale)``:
    Builds the grey/blue synthesis ``go.Figure`` shown in the page and
    embedded inside the square share card.  Grey (actual) and blue
    (prediction) rectangles are drawn per ``(slot, format)`` with opacity
    encoding the format bucket: A=0.25, B=0.50, C=0.75.
- ``build_share_card_figure(share_record, share_url, *, locale)``:
    Builds the 1080x1080 composite Plotly figure used for the downloadable
    PNG and the Open Graph preview.  Three vertical bands:
    header+stats / chart / ranking+footer.
- ``compute_aggregate_stats(rounds)``:
    Returns a dict with ``mae_mgdl``, ``rmse_mgdl``, ``mape``,
    ``rounds_played``, ``pairs`` used by both the layout and the LLM hook.
"""
from __future__ import annotations

import math
import re
import urllib.parse
from typing import Any, Optional

import plotly.graph_objects as go
from dash import dcc, html
from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.encouragement import encouragement_text
from sugar_sugar.i18n import normalize_locale, t


# Opacity by format bucket.  A = Generic Data, B = My Data, C = Mixed.
# Values chosen so ALL THREE are visible even on white, while the jump
# from 25 to 65 still reads as a clear gradient.  Spacing (25/40/65) is
# intentionally non-linear: C bars are the most prominent because "play
# your own data in both categories" is the hardest / most interesting
# round to have completed.
_FORMAT_OPACITY: dict[str, float] = {"A": 0.25, "B": 0.40, "C": 0.65}
_FORMAT_DRAW_ORDER: list[str] = ["A", "B", "C"]

# Colours used by the synthesis chart.  Actual glucose draws in blue so
# it matches the brand colour used across the app; predicted draws in
# a saturated pure orange -- the earlier ochre tone was too washed out
# at low opacities (25% / 40%) to stay visible against the blue bars
# when they overlapped.  The two hues stay accessible for users with
# mild colour-vision differences.
_COLOR_ACTUAL_BASE: tuple[int, int, int] = (21, 101, 192)    # material-blue 800
_COLOR_PRED_BASE: tuple[int, int, int] = (255, 140, 0)       # pure orange
_COLOR_ACTUAL_LINE: str = "rgba(13, 71, 161, 0.95)"          # darker blue
_COLOR_PRED_LINE: str = "rgba(214, 100, 0, 0.95)"            # saturated dark orange

# Prediction rectangles bump their opacity by +0.05 relative to actual
# rectangles so that where the two overlap (same slot played in the
# same format) the orange prediction bar still reads over the blue.
# Kept as a single constant so the page-chart and the PNG-card share the
# same rule.
_PRED_OPACITY_BOOST: float = 0.05

_UUID_RE: re.Pattern[str] = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


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


def _collect_aligned_series_by_format(
    rounds: list[dict[str, Any]],
) -> tuple[
    dict[str, list[list[float]]],
    dict[str, list[list[float]]],
    int,
]:
    """Group actual/predicted values by ``(format, slot)`` across rounds.

    Returns ``(actual_by_format, predicted_by_format, n_slots)`` where each
    inner list has length ``n_slots`` and each cell is the list of
    non-missing observations at that slot from rounds of that format.
    Formats not present in the rounds simply don't appear as keys.
    """
    n_slots: int = 0
    # Pre-compute n_slots so we can size per-format bucket lists once.
    for r in rounds:
        table = r.get("prediction_table_data") or []
        if len(table) < 1:
            continue
        actual_row = table[0] or {}
        window_size: int = int(r.get("prediction_window_size") or 0)
        slots: int = window_size if window_size > 0 else max(
            (int(k[1:]) for k in actual_row.keys() if isinstance(k, str) and k.startswith("t") and k[1:].isdigit()),
            default=-1,
        ) + 1
        if slots > n_slots:
            n_slots = slots

    actual_by_format: dict[str, list[list[float]]] = {}
    predicted_by_format: dict[str, list[list[float]]] = {}

    for r in rounds:
        table = r.get("prediction_table_data") or []
        if len(table) < 2:
            continue
        actual_row = table[0] or {}
        pred_row = table[1] or {}
        fmt: str = str(r.get("format") or "").strip().upper() or "?"

        if fmt not in actual_by_format:
            actual_by_format[fmt] = [[] for _ in range(n_slots)]
            predicted_by_format[fmt] = [[] for _ in range(n_slots)]

        for i in range(n_slots):
            key: str = f"t{i}"
            a = _parse_float(actual_row.get(key))
            if a is not None:
                actual_by_format[fmt][i].append(a)
            p = _parse_float(pred_row.get(key))
            if p is not None:
                predicted_by_format[fmt][i].append(p)

    return actual_by_format, predicted_by_format, n_slots


def _flatten_slots(by_format: dict[str, list[list[float]]], n_slots: int) -> list[list[float]]:
    """Union per-format slot lists into one flat per-slot list."""
    out: list[list[float]] = [[] for _ in range(n_slots)]
    for slots in by_format.values():
        for i, values in enumerate(slots):
            if values:
                out[i].extend(values)
    return out


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


def _best_ranking_entry(share_record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the single best (lowest-rank) entry across per-format + overall.

    Output shape: ``{"rank": int, "total": int, "scope": "overall"|fmt}``.
    Returns None when no ranks are available.
    """
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    best: Optional[dict[str, Any]] = None

    for entry in list(rankings.get("per_format") or []):
        try:
            rank = int(entry.get("rank"))
            total = int(entry.get("total"))
        except (TypeError, ValueError):
            continue
        if rank <= 0:
            continue
        if best is None or rank < best["rank"]:
            best = {"rank": rank, "total": total, "scope": str(entry.get("format") or "")}

    overall = rankings.get("overall")
    if isinstance(overall, dict):
        try:
            o_rank = int(overall.get("rank"))
            o_total = int(overall.get("total"))
            if o_rank > 0 and (best is None or o_rank < best["rank"]):
                best = {"rank": o_rank, "total": o_total, "scope": "overall"}
        except (TypeError, ValueError):
            pass

    return best


def _resolve_format_label(code: str, *, locale: str) -> str:
    """Format code -> localised human label.  Unknown codes pass through."""
    code = str(code or "").strip().upper()
    if code == "A":
        return t("ui.startup.format_a_label", locale=locale)
    if code == "B":
        return t("ui.startup.format_b_label", locale=locale)
    if code == "C":
        return t("ui.startup.format_c_label", locale=locale)
    return code


def _resolve_format_label_short(code: str, *, locale: str) -> str:
    """Shorter label used on the crowded gradient-bar tick row.

    Falls back to the long label when no short form is defined; this
    keeps translations optional-friendly.  Short forms are critical on
    the tick row where three labels share ~300 px horizontally and the
    full "Generic + My Data" string collides with its neighbours.
    """
    code = str(code or "").strip().upper()
    if code == "A":
        key = "ui.startup.format_a_label_short"
    elif code == "B":
        key = "ui.startup.format_b_label_short"
    elif code == "C":
        key = "ui.startup.format_c_label_short"
    else:
        return code
    short: str = t(key, locale=locale)
    # i18nice returns the key itself when missing -- treat that as "no
    # short form provided" and fall back to the long label.
    if not short or short.startswith("ui.startup.format_"):
        return _resolve_format_label(code, locale=locale)
    return short


def _format_number(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _format_opacity_tick_positions() -> list[tuple[str, float]]:
    """Return [(format_code, fractional_x), ...] for the gradient-bar ticks.

    The x positions are literal percentages (0-1), matching each
    format's opacity value in _FORMAT_OPACITY.  Using opacity-as-position
    makes the bar read as "darker = higher percentage of this shade" --
    the same mental model the actual chart uses.
    """
    return [(fmt, _FORMAT_OPACITY[fmt]) for fmt in _FORMAT_DRAW_ORDER]


def _gradient_half(
    *,
    title: str,
    title_color: str,
    base: tuple[int, int, int],
    locale: str,
) -> html.Div:
    """Render one coloured half of the gradient legend (title + bar + ticks).

    The ticks live inside this container so ``left: 25%`` / ``left: 40%``
    / ``left: 65%`` map exactly to that percentage **of this half's
    width**, which is what makes the labels line up under the shading
    intensity they describe.
    """
    loc: str = normalize_locale(locale)
    tick_nodes: list[Any] = []
    for fmt, pos in _format_opacity_tick_positions():
        tick_nodes.append(
            html.Div(
                [
                    html.Div(
                        style={
                            "width": "1px",
                            "height": "8px",
                            "background": "#475569",
                            "margin": "0 auto 2px auto",
                        },
                        disable_n_clicks=True,
                    ),
                    html.Div(
                        _resolve_format_label_short(fmt, locale=loc),
                        style={
                            "fontSize": "11px",
                            "color": "#475569",
                            "textAlign": "center",
                            "fontWeight": "600",
                            "whiteSpace": "nowrap",
                        },
                        disable_n_clicks=True,
                    ),
                ],
                style={
                    "position": "absolute",
                    "left": f"{pos * 100:.2f}%",
                    "transform": "translateX(-50%)",
                    "top": "0",
                    "paddingTop": "4px",
                },
                disable_n_clicks=True,
            )
        )

    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "color": title_color,
                    "textAlign": "center",
                    "marginBottom": "4px",
                },
                disable_n_clicks=True,
            ),
            html.Div(
                style={
                    "height": "14px",
                    "borderRadius": "7px",
                    "background": (
                        "linear-gradient(to right, "
                        f"{_rgba(base, 0.08)} 0%, "
                        f"{_rgba(base, 1.0)} 100%)"
                    ),
                },
                disable_n_clicks=True,
            ),
            # Tick anchor row: positioned relative container so child
            # ticks can use absolute "left: %" within this half's width.
            html.Div(
                tick_nodes,
                style={"position": "relative", "height": "34px"},
                disable_n_clicks=True,
            ),
        ],
        style={"flex": "1", "minWidth": "0"},
        disable_n_clicks=True,
    )


def build_gradient_legend_bar(*, locale: str) -> html.Div:
    """Build a horizontal gradient bar for the share page.

    Layout:

        +-------------------------+     +-------------------------+
        |  Actual glucose         | gap |  Your prediction        |
        |  (blue 8% -> 100%)      |     |  (orange 8% -> 100%)    |
        +-------------------------+     +-------------------------+
           |        |         |            |        |         |
         25%      40%       65%          25%      40%       65%
         Generic  My Data   G+M          Generic  My Data   G+M

    Each half owns its own tick row so the labels line up under the
    shading intensity they describe, rather than being spread across
    a synthetic combined scale that would misrepresent the layout.
    """
    loc: str = normalize_locale(locale)
    blue_base: str = _rgba(_COLOR_ACTUAL_BASE, 1.0)
    orange_base: str = _rgba(_COLOR_PRED_BASE, 1.0)
    actual_label: str = t("ui.share.synthesis.legend_actual_shade", locale=loc)
    pred_label: str = t("ui.share.synthesis.legend_predicted_shade", locale=loc)

    return html.Div(
        [
            _gradient_half(
                title=actual_label,
                title_color=blue_base,
                base=_COLOR_ACTUAL_BASE,
                locale=loc,
            ),
            html.Div(style={"width": "12px"}, disable_n_clicks=True),
            _gradient_half(
                title=pred_label,
                title_color=orange_base,
                base=_COLOR_PRED_BASE,
                locale=loc,
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "stretch",
            "maxWidth": "620px",
            "margin": "18px auto 16px auto",
        },
        disable_n_clicks=True,
    )


def _add_card_gradient_bar(fig: go.Figure, *, locale: str) -> None:
    """Render the gradient legend bar onto the square PNG card.

    Draws two half-bars side by side between y=0.275 and y=0.305 in
    paper coordinates -- blue gradient on the left (actual glucose),
    ochre gradient on the right (predicted).  Each half is built as 20
    stacked rectangles with steadily increasing opacity to fake a CSS
    linear-gradient (kaleido cannot render SVG gradient fills
    reliably).  Three tick labels (Generic / My Data / Combined) land
    underneath at the x position corresponding to each format's
    opacity.
    """
    loc: str = normalize_locale(locale)

    bar_left: float = 0.12
    bar_right: float = 0.88
    bar_mid: float = (bar_left + bar_right) / 2
    gap: float = 0.010  # small gap between the two halves
    half_left_span: tuple[float, float] = (bar_left, bar_mid - gap)
    half_right_span: tuple[float, float] = (bar_mid + gap, bar_right)
    # Positioned in the dedicated gap between the chart-axis labels
    # (~y 0.37) and the ranking heading (~y 0.25).  Bar itself is a
    # thin strip; the "Actual glucose" / "Your prediction" titles sit
    # just above it, tick labels sit just below.
    bar_y0: float = 0.30
    bar_y1: float = 0.325

    def _draw_gradient_half(span: tuple[float, float], base: tuple[int, int, int]) -> None:
        n_steps: int = 24
        step_width = (span[1] - span[0]) / n_steps
        for i in range(n_steps):
            # Opacity climbs from 0.08 to 1.0 so both endpoints are visible.
            alpha = 0.08 + (i / max(1, n_steps - 1)) * (1.0 - 0.08)
            x0 = span[0] + i * step_width
            x1 = span[0] + (i + 1) * step_width
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=x0, x1=x1, y0=bar_y0, y1=bar_y1,
                fillcolor=_rgba(base, alpha),
                line=dict(width=0),
                layer="above",
            )

    _draw_gradient_half(half_left_span, _COLOR_ACTUAL_BASE)
    _draw_gradient_half(half_right_span, _COLOR_PRED_BASE)

    # Title labels for each half.
    blue_base = _rgba(_COLOR_ACTUAL_BASE, 1.0)
    ochre_base = _rgba(_COLOR_PRED_BASE, 1.0)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=(half_left_span[0] + half_left_span[1]) / 2, y=bar_y1 + 0.012,
        xanchor="center", yanchor="bottom",
        text=f"<b>{t('ui.share.synthesis.legend_actual_shade', locale=loc)}</b>",
        showarrow=False,
        font=dict(size=13, color=blue_base),
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=(half_right_span[0] + half_right_span[1]) / 2, y=bar_y1 + 0.012,
        xanchor="center", yanchor="bottom",
        text=f"<b>{t('ui.share.synthesis.legend_predicted_shade', locale=loc)}</b>",
        showarrow=False,
        font=dict(size=13, color=ochre_base),
    )

    # Three tick marks + labels UNDER EACH HALF.  Each half owns its
    # own 0-100% gradient, so ticks belong to their half's local span:
    # opacity 0.25 sits a quarter of the way into the blue half AND a
    # quarter of the way into the orange half, matching the shading
    # intensity the user actually sees at those positions.
    for (half_start, half_end) in (half_left_span, half_right_span):
        half_width_paper: float = half_end - half_start
        for fmt, pos in _format_opacity_tick_positions():
            tick_x = half_start + pos * half_width_paper
            fig.add_shape(
                type="line",
                xref="paper", yref="paper",
                x0=tick_x, x1=tick_x, y0=bar_y0 - 0.008, y1=bar_y0,
                line=dict(color="rgba(71,85,105,0.8)", width=1),
                layer="above",
            )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=tick_x, y=bar_y0 - 0.012,
                xanchor="center", yanchor="top",
                text=_resolve_format_label_short(fmt, locale=loc),
                showarrow=False,
                font=dict(size=11, color="rgba(71,85,105,1)"),
            )


def _safe_display_name(user_info: dict[str, Any]) -> str:
    """Return a human-readable display name, never a UUID.

    Study IDs generated server-side are UUIDs, which are ugly on the share
    card.  If the explicit ``name`` is missing we fall back to an empty
    string (callers omit the whole line) rather than surfacing the UUID.
    """
    raw_name = str(user_info.get("name") or "").strip()
    if raw_name and not _UUID_RE.match(raw_name):
        return raw_name
    return ""


def _rgba(base: tuple[int, int, int], alpha: float) -> str:
    return f"rgba({base[0]},{base[1]},{base[2]},{alpha:.3f})"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _slot_rect_bounds(format_code: str, slot: int, *, half_width: float) -> tuple[float, float]:
    """Compute the horizontal bounds of the per-format bar inside a slot.

    Each slot is conceptually a column of width ``2 * half_width``
    centered on the slot index.  Within that column the three format
    buckets are drawn side-by-side in FORMAT_DRAW_ORDER so darker bars
    don't fully hide lighter ones.  When a format hasn't been played the
    column simply shows fewer bars -- this is the right behaviour because
    the legend only lists formats that actually appear in the record.
    """
    try:
        idx = _FORMAT_DRAW_ORDER.index(format_code)
    except ValueError:
        idx = 0
    n_bins = len(_FORMAT_DRAW_ORDER)
    bin_width = (2 * half_width) / n_bins
    x0 = slot - half_width + idx * bin_width
    x1 = x0 + bin_width
    return x0, x1


def build_synthesis_figure(share_record: dict[str, Any], *, locale: str) -> go.Figure:
    """Build the grey/blue synthesis chart across all played rounds.

    Layout:
      - Per-slot grey rectangles (actual-glucose range), one per format
        present in the record, opacity from _FORMAT_OPACITY.
      - Per-slot blue rectangles (predicted range), narrower, same
        opacity encoding.
      - One black line+dots for the cross-format actual mean per slot.
      - One blue line+dots for the cross-format predicted mean per slot.
      - Dashed vertical separator at the start of the prediction zone.
      - Legend includes invisible sentinel scatter traces that carry the
        format-opacity keys so the reader can decode the shading.
    """
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    actual_by_format, predicted_by_format, n_slots = _collect_aligned_series_by_format(rounds)

    # Cross-format flat aggregates for the central mean lines.
    actual_flat: list[list[float]] = _flatten_slots(actual_by_format, n_slots)
    pred_flat: list[list[float]] = _flatten_slots(predicted_by_format, n_slots)

    xs: list[int] = list(range(n_slots))
    actual_mean: list[Optional[float]] = [
        sum(v) / len(v) if v else None for v in actual_flat
    ]
    pred_mean: list[Optional[float]] = [
        sum(v) / len(v) if v else None for v in pred_flat
    ]

    fig: go.Figure = go.Figure()

    # ---- Per-slot, per-format rectangles ----
    for fmt in _FORMAT_DRAW_ORDER:
        if fmt not in actual_by_format:
            continue
        opacity = _FORMAT_OPACITY.get(fmt, 0.4)
        slots = actual_by_format[fmt]
        for i, values in enumerate(slots):
            if not values:
                continue
            lo, hi = min(values), max(values)
            if hi == lo:
                # Widen a flat slot slightly so it's visible.
                hi = lo + 0.4
            x0, x1 = _slot_rect_bounds(fmt, i, half_width=0.38)
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=lo, y1=hi,
                fillcolor=_rgba(_COLOR_ACTUAL_BASE, opacity),
                line=dict(width=0),
                layer="below",
            )

    for fmt in _FORMAT_DRAW_ORDER:
        if fmt not in predicted_by_format:
            continue
        # Boost prediction opacity slightly above actual's so orange stays
        # visible when its rectangle overlaps a blue one at the same slot.
        opacity = min(1.0, _FORMAT_OPACITY.get(fmt, 0.4) + _PRED_OPACITY_BOOST)
        slots = predicted_by_format[fmt]
        for i, values in enumerate(slots):
            if not values:
                continue
            lo, hi = min(values), max(values)
            if hi == lo:
                hi = lo + 0.4
            x0, x1 = _slot_rect_bounds(fmt, i, half_width=0.25)
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=lo, y1=hi,
                fillcolor=_rgba(_COLOR_PRED_BASE, opacity),
                line=dict(width=0),
                layer="below",
            )

    # ---- Cross-format mean lines ----
    fig.add_trace(
        go.Scatter(
            x=xs, y=actual_mean,
            mode="lines+markers",
            name=t("ui.share.synthesis.legend_actual", locale=loc),
            line=dict(color=_COLOR_ACTUAL_LINE, width=2),
            marker=dict(color=_COLOR_ACTUAL_LINE, size=7),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=pred_mean,
            mode="lines+markers",
            name=t("ui.share.synthesis.legend_prediction", locale=loc),
            line=dict(color=_COLOR_PRED_LINE, width=3),
            marker=dict(color=_COLOR_PRED_LINE, size=7),
            connectgaps=False,
        )
    )

    # Per-format shading is explained by the gradient bar rendered below
    # the chart on the page and above the chart on the PNG card -- we do
    # NOT spam the Plotly legend with three more format entries.

    # ---- Prediction-zone separator ----
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
            zeroline=False, showgrid=True,
            gridcolor="rgba(15,23,42,0.08)",
        ),
        yaxis=dict(
            title=t("ui.share.synthesis.y_axis", locale=loc),
            zeroline=False, showgrid=True,
            gridcolor="rgba(15,23,42,0.08)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Square 1080x1080 share card
# ---------------------------------------------------------------------------

def build_share_card_figure(
    share_record: dict[str, Any],
    *,
    share_url: str,
    locale: str,
    seed: Optional[str] = None,
) -> go.Figure:
    """Build the 1080x1080 composite share card.

    Simpler approach than subplots: a single ``go.Figure`` where the
    synthesis chart occupies the middle ``[0.32, 0.74]`` vertical band
    (via ``xaxis.domain`` / ``yaxis.domain``), and header / stats /
    ranking / tagline / footer live as paper-coordinate annotations in
    explicit bands above and below.  Nothing overlaps because every
    annotation has a hand-tuned y fraction that doesn't collide with
    the chart domain or with sibling annotations.

    Band layout (top -> bottom):
      0.90 .. 1.00  title + name
      0.78 .. 0.88  stats line (MAE / RMSE / Accuracy / Rounds)
      0.74 .. 0.76  legend (set by legend.y below)
      0.32 .. 0.74  synthesis chart (yaxis.domain)
      0.20 .. 0.30  ranking heading + lines
      0.10 .. 0.18  tagline
      0.03 .. 0.08  footer url
    """
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    stats: dict[str, Any] = compute_aggregate_stats(rounds)
    user_info: dict[str, Any] = dict(share_record.get("user_info") or {})
    name: str = _safe_display_name(user_info)

    mae: float = stats.get("mae_mgdl") or float("nan")
    rmse: float = stats.get("rmse_mgdl") or float("nan")
    accuracy: float = stats.get("accuracy") or float("nan")
    rounds_played: int = int(stats.get("rounds_played") or 0)
    encourage: str = encouragement_text(stats, loc, seed=seed)

    # Build the synthesis figure, then flatten it into this figure by
    # copying traces and shapes.  The chart occupies a sub-rectangle of
    # the paper via xaxis.domain / yaxis.domain.
    syn: go.Figure = build_synthesis_figure(share_record, locale=loc)
    fig: go.Figure = go.Figure()
    for trace in syn.data:
        fig.add_trace(trace)
    # Shapes in the synthesis fig are a mix of x/y-ref rectangles and a
    # paper-ref separator line + annotation.  Rectangles use plain x/y
    # refs so they follow the chart domain we set below automatically.
    # Paper-ref shapes (the vertical separator) have y0=0, y1=1 covering
    # the whole paper; we clamp them to the chart band [0.32, 0.74] so
    # they don't crash through the ranking block underneath the chart.
    chart_band_bottom: float = 0.40
    chart_band_top: float = 0.76
    for shape in (syn.layout.shapes or []):
        new_shape = shape.to_plotly_json()
        if new_shape.get("yref") == "paper":
            new_shape["y0"] = chart_band_bottom
            new_shape["y1"] = chart_band_top
        fig.add_shape(**new_shape)
    # Copy the "Prediction zone" annotation.  Its yref=paper so we
    # position it *inside* the chart band, near the top, so it doesn't
    # collide with the Plotly legend that sits just above the chart.
    for ann in (syn.layout.annotations or []):
        ad = ann.to_plotly_json()
        if ad.get("yref") == "paper":
            ad["y"] = chart_band_top - 0.01
            ad["yanchor"] = "top"
        fig.add_annotation(**ad)

    # ---- Header ----
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.975,
        xanchor="center", yanchor="top",
        text=f"<b>{t('ui.share.title', locale=loc)}</b>",
        showarrow=False,
        font=dict(size=30, color="rgba(15,23,42,1)"),
    )
    if name:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.910,
            xanchor="center", yanchor="top",
            text=name,
            showarrow=False,
            font=dict(size=18, color="rgba(71,85,105,1)"),
        )

    # ---- Stats line ----
    mae_label: str = t("ui.share.stat_mae", locale=loc)
    rmse_label: str = t("ui.share.stat_rmse", locale=loc)
    rounds_label: str = t("ui.share.stat_rounds", locale=loc)
    acc_label: str = t("ui.share.stat_accuracy", locale=loc)
    stats_line: str = (
        f"<b>{_format_number(mae)}</b> mg/dL {mae_label}"
        f"   \u2022   "
        f"<b>{_format_number(rmse)}</b> mg/dL {rmse_label}"
        f"   \u2022   "
        f"<b>{_format_number(accuracy)}%</b> {acc_label}"
        f"   \u2022   "
        f"<b>{rounds_played}</b> {rounds_label}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=name and 0.855 or 0.895,
        xanchor="center", yanchor="top",
        text=stats_line,
        showarrow=False,
        font=dict(size=17, color="rgba(15,23,42,1)"),
    )

    # ---- Gradient legend bar (between chart and ranking) ----
    # The bar is drawn as two half-bars side by side (blue gradient on
    # the left = actual, ochre gradient on the right = predicted), with
    # three tick labels underneath that map shading intensity to the
    # format the user played.  Rendered via Plotly shapes so kaleido can
    # reproduce it in the PNG.
    _add_card_gradient_bar(fig, locale=loc)

    # ---- Ranking block (below the gradient bar) ----
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    per_format_entries: list[dict[str, Any]] = list(rankings.get("per_format") or [])
    overall_entry = rankings.get("overall") if isinstance(rankings.get("overall"), dict) else None

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.25,
        xanchor="center", yanchor="top",
        text=f"<b>{t('ui.final.ranking_title', locale=loc)}</b>",
        showarrow=False,
        font=dict(size=18, color="rgba(21,101,192,1)"),
    )

    ranking_lines: list[str] = []
    if overall_entry is not None:
        try:
            o_rank = int(overall_entry.get("rank"))
            o_total = int(overall_entry.get("total"))
            ranking_lines.append(
                t(
                    "ui.final.ranking_overall_line",
                    locale=loc,
                    rank=o_rank, total=o_total,
                )
            )
        except (TypeError, ValueError):
            pass
    for entry in per_format_entries:
        try:
            r = int(entry.get("rank"))
            total = int(entry.get("total"))
        except (TypeError, ValueError):
            continue
        fmt = str(entry.get("format") or "")
        label = _resolve_format_label(fmt, locale=loc)
        ranking_lines.append(
            t("ui.final.ranking_format_line", locale=loc,
              format=label, rank=r, total=total)
        )

    # Render each ranking line at a fixed paper-y with tight spacing.
    # Cap at 4 lines so we don't run into the tagline band.
    rank_base_y: float = 0.215
    rank_step: float = 0.028
    for idx, line in enumerate(ranking_lines[:4]):
        emphasis_open, emphasis_close = ("<b>", "</b>") if idx == 0 else ("", "")
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=rank_base_y - idx * rank_step,
            xanchor="center", yanchor="top",
            text=f"{emphasis_open}{line}{emphasis_close}",
            showarrow=False,
            font=dict(size=14, color="rgba(15,23,42,1)"),
        )

    # ---- Tagline ----
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.085,
        xanchor="center", yanchor="top",
        text=f"<i>{encourage}</i>",
        showarrow=False,
        font=dict(size=15, color="rgba(30,58,138,1)"),
    )

    # ---- Footer ----
    footer: str = t("ui.share.card_footer", locale=loc, url=share_url)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.035,
        xanchor="center", yanchor="top",
        text=footer,
        showarrow=False,
        font=dict(size=13, color="rgba(71,85,105,0.9)"),
    )

    # ---- Chart axis config: constrain the synthesis to the middle band ----
    fig.update_xaxes(
        domain=[0.08, 0.96],
        showgrid=True, gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
        title=None,
    )
    fig.update_yaxes(
        domain=[0.40, 0.76],
        showgrid=True, gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
        title=t("ui.share.synthesis.y_axis", locale=loc),
    )

    fig.update_layout(
        width=1080, height=1080,
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=0.78,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        margin=dict(l=30, r=30, t=30, b=30),
        title=None,
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
    name: str = _safe_display_name(user_info)

    mae: float = stats.get("mae_mgdl") or float("nan")
    rmse: float = stats.get("rmse_mgdl") or float("nan")
    accuracy: float = stats.get("accuracy") or float("nan")
    rounds_played: int = int(stats.get("rounds_played") or 0)
    best_entry: Optional[dict[str, Any]] = _best_ranking_entry(share_record)

    invite_text: str = t(
        "ui.share.invite_text",
        locale=loc,
        mae=_format_number(mae),
        rounds=rounds_played,
    )
    encourage: str = encouragement_text(stats, loc, seed=share_id)

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

    # Stat tile with optional subline (used for Best rank -> category).
    def stat_tile(label: str, value: str, sub: str = "") -> html.Div:
        children: list[Any] = [
            html.Div(value, style={"fontSize": "32px", "fontWeight": "800",
                                    "color": "#0f172a", "lineHeight": "1.1"}),
            html.Div(label, style={"fontSize": "13px", "fontWeight": "600",
                                    "color": "#64748b", "letterSpacing": "0.04em",
                                    "textTransform": "uppercase", "marginTop": "4px"}),
        ]
        if sub:
            children.append(
                html.Div(sub, style={"fontSize": "12px", "color": "#94a3b8",
                                     "marginTop": "2px", "fontStyle": "italic"})
            )
        return html.Div(
            children,
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

    # Best-rank tile: value + category subline.
    if best_entry:
        best_value: str = f"#{best_entry['rank']}"
        scope: str = best_entry["scope"]
        if scope == "overall":
            best_sub: str = t("ui.share.best_rank_scope_overall", locale=loc)
        else:
            best_sub = _resolve_format_label(scope, locale=loc)
    else:
        best_value = t("ui.share.stat_no_ranking", locale=loc)
        best_sub = ""

    stats_row: html.Div = html.Div(
        [
            stat_tile(t("ui.share.stat_mae", locale=loc), f"{_format_number(mae)} mg/dL"),
            stat_tile(t("ui.share.stat_rmse", locale=loc), f"{_format_number(rmse)} mg/dL"),
            stat_tile(t("ui.share.stat_accuracy", locale=loc), f"{_format_number(accuracy)}%"),
            stat_tile(t("ui.share.stat_rounds", locale=loc), str(rounds_played)),
            stat_tile(t("ui.share.stat_ranking", locale=loc), best_value, sub=best_sub),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "14px",
               "marginTop": "20px", "justifyContent": "center"},
        disable_n_clicks=True,
    )

    # ---------- Ranking block: Overall first, per-format after ----------
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    per_format_entries: list[dict[str, Any]] = list(rankings.get("per_format") or [])
    overall_entry: Optional[dict[str, Any]] = (
        rankings.get("overall") if isinstance(rankings.get("overall"), dict) else None
    )

    ranking_lines: list[Any] = []
    if overall_entry is not None:
        try:
            o_rank = int(overall_entry.get("rank"))
            o_total = int(overall_entry.get("total"))
            ranking_lines.append(
                html.Li(
                    t("ui.final.ranking_overall_line", locale=loc, rank=o_rank, total=o_total),
                    style={"marginBottom": "6px", "fontWeight": "700"},
                    disable_n_clicks=True,
                )
            )
        except (TypeError, ValueError):
            pass
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
                    rank=rank, total=total,
                ),
                style={"marginBottom": "4px"},
                disable_n_clicks=True,
            )
        )

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

    synthesis_card: html.Div = html.Div(
        [
            dcc.Graph(
                figure=build_synthesis_figure(share_record, locale=loc),
                config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False},
                style={"height": "460px"},
            ),
            # Gradient bar replaces the old text hint.  It visually shows
            # how blue vs ochre maps to actual vs predicted, and the
            # three ticks underneath (Generic / My Data / Combined) map
            # shading intensity back to the format that produced it.
            build_gradient_legend_bar(locale=loc),
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

    url_store: html.Div = html.Div(
        share_url, id="share-url-value",
        style={"display": "none"}, disable_n_clicks=True,
    )

    header_children: list[Any] = [
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
    ]
    if name:
        header_children.append(
            html.P(
                name,
                style={"fontSize": "15px", "color": "#1e3a8a",
                       "textAlign": "center", "fontWeight": "600",
                       "margin": "0"},
                disable_n_clicks=True,
            )
        )

    return html.Div(
        [
            url_store,
            html.Div(
                header_children,
                style={"paddingTop": "20px"},
                disable_n_clicks=True,
            ),

            stats_row,
            ranking_card,
            played_line,
            synthesis_card,

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
