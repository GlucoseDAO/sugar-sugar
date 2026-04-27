"""Share-page component.

Renders a Dash page at ``/share/<share_id>`` that lets the user show off
their Sugar Sugar performance on social networks.

Public API
----------
- ``create_share_layout(share_record, share_id, share_url, *, locale)``:
    Returns a ``html.Div`` for the Dash page.  ``share_record`` is what
    ``share_store.load_share`` returned; it must contain at minimum a
    ``rounds`` list matching the shape used on ``/final``.
- ``build_synthesis_figure(share_record, *, locale, show_title=False, figure_height=None, show_legend_in_figure=True)``:
    One stacked chart per data source format the user played (Generic,
    My data, or Mixed).  Each row is the **next prediction hour** only
    (every 5-min step; **Y** = percent error from actual, **X** = time
    in that hour).  Per round, translucent hatch appears **only** between the
    0% baseline and the variability curve (not the full panel), with a
    **gradient** (saturated near the baseline, greyer toward the curve).
    A solid **black** 0% line is the reference (drawn above the variability
    curves, under point markers).  If the first predicted point in that hour is
    missing, **actual** at that time is used so the line still starts.
    Colours desaturate as |error| grows.
- ``build_share_card_figure(share_record, share_url, *, locale)``:
    Builds the 1080x1080 composite Plotly figure used for the downloadable
    PNG and the Open Graph preview.  Vertical stack: title, stats, quote,
    legend, chart (per-panel format labels), ranking (left) and QR (right)
    in the footer band, QR linking to the play URL.
- ``compute_aggregate_stats(rounds)``:
    Returns a dict with ``mae_mgdl``, ``rmse_mgdl``, ``mape``,
    ``rounds_played``, ``pairs`` used by both the layout and the LLM hook.
"""
from __future__ import annotations

import base64
import io
import math
import re
import urllib.parse
from typing import Any, Optional

import plotly.graph_objects as go
import segno
from plotly.subplots import make_subplots
from dash import dcc, html
from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.encouragement import encouragement_text
from sugar_sugar.i18n import normalize_locale, t


# Draw order: Generic (A), My data (B), Mixed (C) — one subplot per format present.
_FORMAT_DRAW_ORDER: list[str] = ["A", "B", "C"]

# Per-format line colour (fills use the same hue with alpha, only under the curve).
_FORMAT_PANEL: dict[str, dict[str, str]] = {
    "A": {"line": "rgba(21, 101, 192, 0.92)"},
    "B": {"line": "rgba(234, 88, 12, 0.92)"},
    "C": {"line": "rgba(91, 33, 182, 0.88)"},
}
# Solid black baseline (scatter so markers can stay on top; drawn after variability lines)
_ZERO_LINE: dict[str, Any] = {"color": "black", "width": 3.5}
# Percent-error curve segments (between markers)
_VARIABILITY_LINE_WIDTH: float = 1.35
_LEGEND_VARIABILITY_SAMPLE_WIDTH: float = 1.0
_DEFAULT_FORMAT_STYLE: dict[str, str] = {"line": "rgba(255, 140, 0, 0.92)"}

_UUID_RE: re.Pattern[str] = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_RG_RGBA: re.Pattern[str] = re.compile(
    r"^rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([0-9.]+)\s*\)\s*$",
    re.IGNORECASE,
)
# Hatch: horizontal slices from baseline toward chord; alpha + grey increase toward the curve
_N_FILL_SLICES: int = 8
_FILL_ALPHA_TOP: float = 0.11
_FILL_ALPHA_BOTTOM: float = 0.22
# |y|/ref < 1 → blend toward full grey; ref = row max |% error|;
# Exp < 1: stronger greying so mid/outer range moves toward #808080, not just faint hue
_DESAT_GAMMA: float = 0.38
# Literal neutral used at t=1.0
_GREY_RGB: tuple[int, int, int] = (128, 128, 128)


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


def _window_size_for_round(r: dict[str, Any]) -> int:
    ws: int = int(r.get("prediction_window_size") or 0)
    if ws > 0:
        return ws
    table: list[Any] = list(r.get("prediction_table_data") or [])
    if not table:
        return 0
    row0: dict[str, Any] = table[0] or {}
    max_t: int = -1
    for key in row0:
        if isinstance(key, str) and key.startswith("t") and key[1:].isdigit():
            max_t = max(max_t, int(key[1:]))
    return max_t + 1 if max_t >= 0 else 0


def _prediction_next_hour_range(ws: int) -> Optional[tuple[int, int]]:
    """Return ``[start, end)`` slot indices for the *next* prediction hour."""
    if ws <= 0:
        return None
    n_visible: int = min(PREDICTION_HOUR_OFFSET, ws)
    start: int = max(0, ws - n_visible)
    return (start, ws)


def _minutes_tickvals(n_points: int) -> list[int]:
    """Wall-clock style offset within the next hour: 0, 5, 10, … (one per 5-min step)."""
    return [5 * i for i in range(n_points)]


def _format_figure_styling(format_code: str) -> dict[str, str]:
    return _FORMAT_PANEL.get(
        str(format_code or "").strip().upper(),
        _DEFAULT_FORMAT_STYLE,
    )


def _round_sort_key(ri: dict[str, Any]) -> tuple[int, int]:
    n: int = int(ri.get("round_number") or 0)
    return (n, id(ri))


def _t_blend_to_grey(
    abs_mag: float, ref_max: float, *, gamma: float = _DESAT_GAMMA
) -> float:
    """0 = full line colour, 1 = exact neutral grey, scaled by *this* subplot's data range.

    ``ref_max`` = max |percent error| in the row (or 0 if none).
    """
    if ref_max < 1e-12:
        return 0.0
    t_lin: float = min(1.0, abs(float(abs_mag)) / ref_max)
    return float(min(1.0, t_lin ** float(gamma)))


def _blend_toward_grey(
    line_rgba: str, t: float, alpha_override: Optional[float] = None
) -> str:
    """Linear blend of base RGB to ``_GREY_RGB``; *t* in [0,1]."""
    t2: float = max(0.0, min(1.0, t))
    m = _RG_RGBA.match(str(line_rgba).strip())
    if not m:
        return line_rgba
    r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    a = float(m.group(4)) if alpha_override is None else float(alpha_override)
    gr, gg, gb = _GREY_RGB
    r2 = int(r * (1.0 - t2) + gr * t2)
    g2 = int(g * (1.0 - t2) + gg * t2)
    b2 = int(b * (1.0 - t2) + gb * t2)
    return f"rgba({r2},{g2},{b2},{a:.3f})"


def _add_gradient_hatch_axis_to_chord(
    fig: go.Figure,
    *,
    row_idx: int,
    col: int,
    x0: float,
    x1: float,
    p0: float,
    p1: float,
    line_c: str,
    row_y_max: float,
    legend_rn: int,
) -> None:
    """Stacked quads between y=0 and chord *(x0,p0)–(x1,p1)*: vivid near baseline, grey near chord.

    If the chord crosses y=0, splits at the crossing and recurses so the hatch
    never sits outside the axis–curve strip.
    """
    if math.isclose(p0, 0.0, abs_tol=1e-12) and math.isclose(p1, 0.0, abs_tol=1e-12):
        return

    if p0 * p1 < 0.0:
        denom: float = p0 - p1
        if abs(denom) < 1e-15:
            return
        tn: float = p0 / denom
        if 0.0 < tn < 1.0:
            xc: float = x0 + tn * (x1 - x0)
            _add_gradient_hatch_axis_to_chord(
                fig,
                row_idx=row_idx,
                col=col,
                x0=x0,
                x1=xc,
                p0=p0,
                p1=0.0,
                line_c=line_c,
                row_y_max=row_y_max,
                legend_rn=legend_rn,
            )
            _add_gradient_hatch_axis_to_chord(
                fig,
                row_idx=row_idx,
                col=col,
                x0=xc,
                x1=x1,
                p0=0.0,
                p1=p1,
                line_c=line_c,
                row_y_max=row_y_max,
                legend_rn=legend_rn,
            )
            return

    for k in range(_N_FILL_SLICES):
        f0: float = k / _N_FILL_SLICES
        f1: float = (k + 1) / _N_FILL_SLICES
        yl0, yl1 = f0 * p0, f1 * p0
        yr0, yr1 = f0 * p1, f1 * p1
        m_band: float = max(abs(yl0), abs(yl1), abs(yr0), abs(yr1))
        t_mid: float = 0.5 * (f0 + f1)
        a_fill: float = (
            _FILL_ALPHA_BOTTOM * (1.0 - t_mid) + _FILL_ALPHA_TOP * t_mid
        )
        fill_t: float = _t_blend_to_grey(m_band, row_y_max)
        fill_rgba: str = _blend_toward_grey(line_c, fill_t, a_fill)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[yl0, yr0, yr1, yl1, yl0],
                fill="toself",
                fillcolor=fill_rgba,
                line=dict(width=0),
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"fill{legend_rn}",
            ),
            row=row_idx,
            col=col,
        )


def _formats_played_in_order(rounds: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for r in rounds:
        f: str = str(r.get("format") or "").strip().upper()
        if f in _FORMAT_DRAW_ORDER:
            seen.add(f)
    return [f for f in _FORMAT_DRAW_ORDER if f in seen]


def _percent_error_series_for_round(
    r: dict[str, Any], hour_range: tuple[int, int]
) -> list[Optional[float]]:
    """``(pred - actual) / actual * 100`` at each 5-min step; None when undefined.

    If the **first** point of the next-hour window is missing a prediction, use
    the actual at that time as the prediction (0% error) so the graph still runs.
    """
    start, end = hour_range
    table: list[Any] = list(r.get("prediction_table_data") or [])
    if len(table) < 2:
        return []
    actual_row: dict[str, Any] = table[0] or {}
    pred_row: dict[str, Any] = table[1] or {}
    out: list[Optional[float]] = []
    for i in range(start, end):
        key: str = f"t{i}"
        a = _parse_float(actual_row.get(key))
        p = _parse_float(pred_row.get(key))
        if i == start and p is None and a is not None and a != 0.0:
            p = a
        if a is None or p is None or a == 0.0:
            out.append(None)
        else:
            out.append((p - a) / a * 100.0)
    return out


def _max_abs_percent_for_format(rounds_f: list[dict[str, Any]]) -> float:
    """Largest |percent error| across all rounds in this format row (for colour scale)."""
    m: float = 0.0
    for r in rounds_f:
        ws: int = _window_size_for_round(r)
        h: Optional[tuple[int, int]] = _prediction_next_hour_range(ws)
        if h is None:
            continue
        ys: list[Optional[float]] = _percent_error_series_for_round(r, h)
        for v in ys:
            if v is not None:
                m = max(m, abs(float(v)))
    return m


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


def _format_number(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


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


def _play_url_from_share(share_url: str) -> str:
    """Absolute site root for the same host as ``share_url`` (where the game is played)."""
    s: str = (share_url or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        p = urllib.parse.urlsplit(s)
        return urllib.parse.urlunsplit((p.scheme, p.netloc, "/", "", ""))
    return "/"


def _qrcode_png_data_uri(target_url: str) -> str:
    """PNG QR as a data URI for Plotly ``layout.images`` (pure-Python writer, no Pillow)."""
    buf: io.BytesIO = io.BytesIO()
    q = segno.make(target_url, error="m")
    q.save(buf, kind="png", scale=5, border=2)
    b64: str = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# Share card PNG: Plotly ``xref/ yref: paper`` is the *inner* plot, not the full figure,
# so large margins do not reserve “safe” space — text must be placed in y-bands the
# traces do not use, by shrinking ``yaxis.domain`` for every stacked subplot.
_CARD_TRACE_Y0: float = 0.29
_CARD_TRACE_Y1: float = 0.71


def _constrain_synthesis_panels_to_vertical_band(
    fig: go.Figure,
    n_rows: int,
    *,
    y_lo: float,
    y_hi: float,
) -> None:
    """Force stacked subplots to use only *y* in [y_lo, y_hi] (plot / paper fraction)."""
    if n_rows < 1:
        return
    span: float = y_hi - y_lo
    step: float = span / float(n_rows)
    for row in range(1, n_rows + 1):
        d_bottom: float = y_lo + (n_rows - row) * step
        d_top: float = d_bottom + step
        fig.update_yaxes(domain=(d_bottom, d_top), row=row, col=1)


# ---------------------------------------------------------------------------
# Synthesis: HTML (share page) — legend cannot overlap the plot
# ---------------------------------------------------------------------------


def _synthesis_legend_row_html(share_record: dict[str, Any], *, locale: str) -> html.Div:
    """In-DOM legend for /share: matches stroke colours, never drawn on the canvas."""
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    formats: list[str] = _formats_played_in_order(rounds)
    fmt0: str = str(formats[0] or "A").strip().upper() or "A"
    line_c: str = _format_figure_styling(fmt0)["line"]
    z_label: str = t("ui.share.synthesis.legend_zero", locale=loc)
    v_label: str = t("ui.share.synthesis.legend_variability", locale=loc)
    item_style: dict[str, str] = {
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "8px",
    }
    swatch: dict[str, str] = {
        "display": "inline-block",
        "width": "28px",
        "verticalAlign": "middle",
        "marginRight": "0",
    }
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        style={**swatch, "borderBottom": "2.5px solid #0f0f0f", "alignSelf": "center"},
                    ),
                    html.Span(
                        z_label,
                        style={"color": "#334155", "fontSize": "13px", "lineHeight": "1.2"},
                    ),
                ],
                style=item_style,
            ),
            html.Div(
                [
                    html.Span(
                        style={
                            **swatch,
                            "borderBottom": f"2px solid {line_c}",
                            "alignSelf": "center",
                        },
                    ),
                    html.Span(
                        v_label,
                        style={"color": "#334155", "fontSize": "13px", "lineHeight": "1.2"},
                    ),
                ],
                style=item_style,
            ),
        ],
        className="share-synthesis-legend",
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "columnGap": "28px",
            "rowGap": "4px",
            "padding": "0 4px 4px 4px",
        },
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_synthesis_figure(
    share_record: dict[str, Any],
    *,
    locale: str,
    show_title: bool = False,
    figure_height: Optional[int] = None,
    show_legend_in_figure: bool = True,
    show_format_row_annotations: bool = True,
) -> go.Figure:
    """Per-slot percent error in the next hour (one subplot per data source.

    The share page passes ``show_legend_in_figure=False`` and renders the legend
    in HTML (see ``_synthesis_legend_row_html``) so Plotly never places it on
    the data. For PNG/OG, keep ``show_legend_in_figure=True`` unless the card
    draws its own legend. Both the downloadable share card and the interactive
    ``/share`` page use ``show_format_row_annotations=True`` by default so each
    panel shows its data-source label (Generic / My data / Mixed).

    ``figure_height`` should match the Dash ``dcc.Graph`` pixel height. The
    share page uses ``show_title=False`` and renders the chart title in HTML;
    Plotly ``layout.title`` is not used. ``autosize`` keeps the plot width in
    the container.

    Per format row, ``ref = max |%|`` in that row drives a blend to **#808080**
    at the largest errors; the curve uses the same scale.  Fill is only the
    region between y=0 and the curve, as stacked bands: **saturated near the
    baseline**, **greyer toward the variability chord**.  A solid **black** 0%
    line is drawn on top of the variability curves.
    """
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    formats: list[str] = _formats_played_in_order(rounds)
    n_fmt: int = len(formats)

    if n_fmt < 1 or not rounds:
        eff_empty: int = (
            figure_height
            if figure_height is not None
            else 420
        )
        fig = go.Figure()
        fig.update_layout(
            height=eff_empty,
            margin=dict(l=48, r=8, t=32, b=40),
            autosize=True,
            annotations=[
                dict(
                    text=t("ui.share.synthesis.empty", locale=loc),
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=15, color="#64748b"),
                )
            ],
        )
        return fig

    # Width of the x axis = length of the prediction window slice (all rounds aligned).
    n_points: int = 0
    for r in rounds:
        ws = _window_size_for_round(r)
        h = _prediction_next_hour_range(ws)
        if h is None:
            continue
        n_points = max(n_points, h[1] - h[0])
    if n_points < 1:
        n_points = PREDICTION_HOUR_OFFSET
    x_vals: list[int] = _minutes_tickvals(n_points)
    x_ticktext: list[str] = [str(m) for m in x_vals]
    y_zero: list[float] = [0.0] * n_points

    # No subplot_titles: they sit in the panel and crowd the main title / legend into the data.
    fig: go.Figure = make_subplots(
        rows=n_fmt,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
    )

    leg_zero: str = t("ui.share.synthesis.legend_zero", locale=loc)
    leg_variability: str = t("ui.share.synthesis.legend_variability", locale=loc)

    first_zero: bool = True

    for row_idx, fmt in enumerate(formats, start=1):
        sty: dict[str, str] = _format_figure_styling(fmt)
        line_c: str = sty["line"]

        rounds_f: list[dict[str, Any]] = [
            r
            for r in rounds
            if str(r.get("format") or "").strip().upper() == fmt
        ]
        rounds_f.sort(key=_round_sort_key)
        row_y_max: float = _max_abs_percent_for_format(rounds_f)

        y_dom: str = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        if show_format_row_annotations:
            # Upper-left **inside** the panel (x domain 0–1) so we keep margin.l=0; no left clip.
            fig.add_annotation(
                xref="x domain",
                yref=y_dom,
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                text=f"<b>{_resolve_format_label(fmt, locale=loc)}</b>",
                showarrow=False,
                font=dict(size=12, color="#475569"),
                bgcolor="rgba(255,255,255,0.88)",
                borderpad=3,
            )

        # 1) Gradient hatch only between y=0 and the variability chord
        for r in rounds_f:
            ws: int = _window_size_for_round(r)
            hrange: Optional[tuple[int, int]] = _prediction_next_hour_range(ws)
            if hrange is None:
                continue
            y_pct: list[Optional[float]] = _percent_error_series_for_round(r, hrange)
            if not y_pct or not any(x is not None for x in y_pct):
                continue
            n_r: int = len(y_pct)
            x_r: list[int] = _minutes_tickvals(n_r)
            rn: int = int(r.get("round_number") or 0)
            for j in range(0, n_r - 1):
                p0o, p1o = y_pct[j], y_pct[j + 1]
                if p0o is None or p1o is None:
                    continue
                p0: float = float(p0o)
                p1: float = float(p1o)
                xa: float = float(x_r[j])
                xb: float = float(x_r[j + 1])
                _add_gradient_hatch_axis_to_chord(
                    fig,
                    row_idx=row_idx,
                    col=1,
                    x0=xa,
                    x1=xb,
                    p0=p0,
                    p1=p1,
                    line_c=line_c,
                    row_y_max=row_y_max,
                    legend_rn=rn,
                )

        # 2) Variability line segments (thin); black 0% line and markers stacked above
        for r in rounds_f:
            ws2: int = _window_size_for_round(r)
            h2: Optional[tuple[int, int]] = _prediction_next_hour_range(ws2)
            if h2 is None:
                continue
            y_pct2: list[Optional[float]] = _percent_error_series_for_round(r, h2)
            if not y_pct2 or not any(x is not None for x in y_pct2):
                continue
            n2: int = len(y_pct2)
            x2: list[int] = _minutes_tickvals(n2)
            rn2: int = int(r.get("round_number") or 0)
            for j in range(0, n2 - 1):
                v0, v1 = y_pct2[j], y_pct2[j + 1]
                if v0 is None or v1 is None:
                    continue
                x0, x1 = x2[j], x2[j + 1]
                m_seg: float = max(abs(v0), abs(v1))
                seg_c: str = _blend_toward_grey(
                    line_c, _t_blend_to_grey(m_seg, row_y_max), None
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[v0, v1],
                        mode="lines",
                        line=dict(color=seg_c, width=_VARIABILITY_LINE_WIDTH),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=f"ln{rn2}",
                    ),
                    row=row_idx, col=1,
                )

        # 3) Black 0% line (above variability traces, below markers)
        show_lz: bool = show_legend_in_figure and first_zero
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_zero,
                mode="lines",
                name=leg_zero,
                line=_ZERO_LINE,
                connectgaps=True,
                legendgroup="zero",
                showlegend=show_lz,
                hoverinfo="skip",
            ),
            row=row_idx,
            col=1,
        )
        first_zero = False

        for r in rounds_f:
            ws3: int = _window_size_for_round(r)
            h3: Optional[tuple[int, int]] = _prediction_next_hour_range(ws3)
            if h3 is None:
                continue
            y_pct3: list[Optional[float]] = _percent_error_series_for_round(r, h3)
            if not y_pct3 or not any(x is not None for x in y_pct3):
                continue
            n3: int = len(y_pct3)
            x3: list[int] = _minutes_tickvals(n3)
            rn3: int = int(r.get("round_number") or 0)
            m_x: list[int] = []
            m_y: list[float] = []
            m_col: list[str] = []
            m_cd: list[list[int]] = []
            for j in range(n3):
                yy = y_pct3[j]
                if yy is None:
                    continue
                m_x.append(x3[j])
                m_y.append(yy)
                m_col.append(
                    _blend_toward_grey(
                        line_c, _t_blend_to_grey(abs(yy), row_y_max), None
                    )
                )
                m_cd.append([rn3])
            if m_x:
                fig.add_trace(
                    go.Scatter(
                        x=m_x,
                        y=m_y,
                        mode="markers",
                        marker=dict(size=5, color=m_col, line=dict(width=0, color="rgba(0,0,0,0)")),
                        legendgroup="err",
                        showlegend=False,
                        customdata=m_cd,
                        hovertemplate=(
                            f"{t('ui.share.synthesis.round_hover', locale=loc)} %{{customdata[0]}} · "
                            f"{t('ui.share.synthesis.y_axis_short', locale=loc)} %{{y:.1f}}%"
                            f"<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=1,
                )

    x_title: str = t("ui.share.synthesis.x_axis_time", locale=loc)
    y_name: str = t("ui.share.synthesis.y_axis_short", locale=loc)
    y_title_row: int = max(1, (n_fmt + 1) // 2)
    for row in range(1, n_fmt + 1):
        fig.update_yaxes(
            title_text=y_name if row == y_title_row else "",
            row=row, col=1,
            automargin=True,
            gridcolor="rgba(15,23,42,0.08)",
            ticksuffix="%",
        )
    for row in range(1, n_fmt + 1):
        x_kw: dict[str, Any] = {
            "row": row,
            "col": 1,
            "tickmode": "array",
            "tickvals": x_vals,
            "ticktext": x_ticktext,
            "zeroline": False,
            "showgrid": True,
            "gridcolor": "rgba(15,23,42,0.08)",
        }
        if row == n_fmt:
            x_kw["title_text"] = x_title
            x_kw["automargin"] = True
        fig.update_xaxes(**x_kw)

    if show_legend_in_figure and formats:
        var_leg_color: str = _format_figure_styling(formats[0])["line"]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=var_leg_color, width=_LEGEND_VARIABILITY_SAMPLE_WIDTH),
                name=leg_variability,
                showlegend=True,
                visible="legendonly",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    eff_h: int = (
        figure_height
        if figure_height is not None
        else max(620, 120 * n_fmt + 500)
    )
    # When show_legend_in_figure is False (web /share), no Plotly legend: it cannot
    # be kept reliably out of the drawable without wasting huge top margins.
    if show_title:
        margin_top = 102
    elif show_legend_in_figure:
        margin_top = 56
    else:
        margin_top = 8

    layout_extra: dict[str, Any] = {
        "autosize": True,
        "height": eff_h,
        "width": None,
        "margin": dict(
            l=0,
            r=4,
            t=margin_top,
            b=48,
        ),
        "showlegend": show_legend_in_figure,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "hovermode": "x unified",
        "hoverlabel": dict(
            namelength=-1,
            align="left",
            font=dict(size=13),
        ),
    }
    if show_legend_in_figure:
        legend_leg: dict[str, Any] = {
            "orientation": "h",
            "xref": "paper",
            "yref": "paper",
            "xanchor": "center",
            "x": 0.5,
            "yanchor": "top",
            "y": 1.0,
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
            "itemsizing": "constant",
            "itemwidth": 36,
            "font": dict(size=12, color="#334155"),
        }
        if show_title:
            fig.add_annotation(
                x=0.5,
                y=1.0,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="top",
                text=t("ui.share.synthesis.title", locale=loc),
                showarrow=False,
                font=dict(size=20, color="#0f172a"),
            )
            legend_leg["y"] = 0.89
        layout_extra["legend"] = legend_leg
    elif show_title:
        fig.add_annotation(
            x=0.5,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            text=t("ui.share.synthesis.title", locale=loc),
            showarrow=False,
            font=dict(size=20, color="#0f172a"),
        )
    fig.update_layout(**layout_extra)
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
    """Build the 1080x1080 composite share card for kaleido/OG.

    In Plotly, ``paper`` is the *inner* plot, so we shrink every subplot
    ``yaxis.domain`` to a middle band and place title, metrics, quote, legend
    entirely above, and ranking + QR entirely below, the traces.
    """
    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    stats: dict[str, Any] = compute_aggregate_stats(rounds)
    user_info: dict[str, Any] = dict(share_record.get("user_info") or {})
    name: str = _safe_display_name(user_info)

    mae: float = stats.get("mae_mgdl") or float("nan")
    rmse: float = stats.get("rmse_mgdl") or float("nan")
    rounds_played: int = int(stats.get("rounds_played") or 0)
    encourage: str = encouragement_text(stats, loc, seed=seed)
    play_url: str = _play_url_from_share(share_url)
    qri: str = _qrcode_png_data_uri(play_url)

    fig: go.Figure = build_synthesis_figure(
        share_record,
        locale=loc,
        show_title=False,
        show_legend_in_figure=False,
        show_format_row_annotations=True,
    )
    fig.update_layout(
        width=1080, height=1080,
        margin=dict(l=56, r=28, t=20, b=20),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="white",
        showlegend=False,
    )
    n_fmt: int = len(_formats_played_in_order(rounds))
    if n_fmt >= 1:
        _constrain_synthesis_panels_to_vertical_band(
            fig, n_fmt, y_lo=_CARD_TRACE_Y0, y_hi=_CARD_TRACE_Y1
        )
    mae_label: str = t("ui.share.stat_mae", locale=loc)
    rmse_label: str = t("ui.share.stat_rmse", locale=loc)
    rounds_label: str = t("ui.share.stat_rounds", locale=loc)
    leg_zero: str = t("ui.share.synthesis.legend_zero", locale=loc)
    leg_var: str = t("ui.share.synthesis.legend_variability", locale=loc)
    # Paper y: 0=bottom, 1=top. Traces use [_CARD_TRACE_Y0, _CARD_TRACE_Y1] only.
    y_legend: float = 0.755
    y_stats: float
    y_quote: float
    if name:
        y_stats = 0.888
        y_quote = 0.828
    else:
        y_stats = 0.908
        y_quote = 0.848

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.987,
        xanchor="center", yanchor="top",
        text=f"<b>{t('ui.share.title', locale=loc)}</b>",
        showarrow=False,
        font=dict(size=36, color="rgba(15,23,42,1)"),
    )
    if name:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.928,
            xanchor="center", yanchor="top",
            text=name,
            showarrow=False,
            font=dict(size=18, color="rgba(71,85,105,1)"),
        )
    stats_line: str = (
        f"<b>{_format_number(mae)}</b> mg/dL {mae_label}"
        f"   \u2022   "
        f"<b>{_format_number(rmse)}</b> mg/dL {rmse_label}"
        f"   \u2022   "
        f"<b>{rounds_played}</b> {rounds_label}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=y_stats,
        xanchor="center", yanchor="top",
        text=stats_line,
        showarrow=False,
        font=dict(size=17, color="rgba(15,23,42,1)"),
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=y_quote,
        xanchor="center", yanchor="top",
        text=f"<i>{encourage}</i>",
        showarrow=False,
        font=dict(size=16, color="rgba(30,58,138,1)"),
    )

    formats_played: list[str] = _formats_played_in_order(rounds)
    fmt0: str = str(formats_played[0]).strip().upper() if formats_played else "A"
    if fmt0 not in ("A", "B", "C"):
        fmt0 = "A"
    var_color: str = _format_figure_styling(fmt0)["line"]
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.28, x1=0.35, y0=y_legend, y1=y_legend,
        line={"color": "#0f0f0f", "width": 3.2},
        layer="above",
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.355, y=y_legend, xanchor="left", yanchor="middle",
        text=leg_zero,
        showarrow=False,
        font=dict(size=13, color="rgba(51,65,85,1)"),
    )
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.55, x1=0.62, y0=y_legend, y1=y_legend,
        line={"color": var_color, "width": 2.0},
        layer="above",
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.625, y=y_legend, xanchor="left", yanchor="middle",
        text=leg_var,
        showarrow=False,
        font=dict(size=13, color="rgba(51,65,85,1)"),
    )

    # ---- Ranking (large, bottom) ----
    rankings: dict[str, Any] = dict(share_record.get("rankings") or {})
    per_format_entries: list[dict[str, Any]] = list(rankings.get("per_format") or [])
    overall_entry: Optional[dict[str, Any]] = (
        rankings.get("overall") if isinstance(rankings.get("overall"), dict) else None
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
            t(
                "ui.final.ranking_format_line",
                locale=loc,
                format=label, rank=r, total=total,
            )
        )
    # Footer band: left column = ranking; right = scan hint + QR. Same top/bottom on both columns.
    rank_left_x: float = 0.04
    _rank_title_y_top: float = 0.205
    _rank_first_line_y_top: float = 0.176
    _rank_step: float = 0.026
    # One line of body text in paper fraction (~px/fig_height for 14–17px on 1080)
    _rank_line_paper_h: float = 0.022
    _scan_font_pt: int = 12
    _scan_line_paper_h: float = 0.02
    _scan_qr_gap: float = 0.003
    qr_col_x: float = 0.86

    footer_y_top: float
    footer_y_bottom: float
    if ranking_lines:
        footer_y_top = _rank_title_y_top
        n_body: int = len(ranking_lines[:4])
        y_last_line_top: float = (
            _rank_first_line_y_top - max(0, n_body - 1) * _rank_step
        )
        footer_y_bottom = y_last_line_top - _rank_line_paper_h
        fig.add_annotation(
            xref="paper", yref="paper",
            x=rank_left_x, y=footer_y_top,
            xanchor="left", yanchor="top",
            text=f"<b>{t('ui.final.ranking_title', locale=loc)}</b>",
            showarrow=False,
            font=dict(size=22, color="rgba(21,101,192,1)"),
        )
        for idx, line in enumerate(ranking_lines[:4]):
            em_o, em_c = ("<b>", "</b>") if idx == 0 else ("", "")
            fig.add_annotation(
                xref="paper", yref="paper",
                x=rank_left_x, y=_rank_first_line_y_top - idx * _rank_step,
                xanchor="left", yanchor="top",
                text=f"{em_o}{line}{em_c}",
                showarrow=False,
                font=dict(
                    size=17 if idx == 0 else 14,
                    color="rgba(15,23,42,1)",
                ),
            )
    else:
        footer_y_top, footer_y_bottom = 0.18, 0.05

    col_h: float = footer_y_top - footer_y_bottom
    qr_h: float = max(0.0, col_h - _scan_line_paper_h - _scan_qr_gap)
    # Square QR: same paper span on x and y (1080 square figure)
    qr_w: float = qr_h

    fig.add_annotation(
        xref="paper", yref="paper",
        x=qr_col_x, y=footer_y_top,
        xanchor="center", yanchor="top",
        text=t("ui.share.qr_scan_to_play", locale=loc),
        showarrow=False,
        font=dict(size=_scan_font_pt, color="rgba(71,85,105,0.95)"),
    )
    fig.add_layout_image(
        dict(
            source=qri,
            xref="paper",
            yref="paper",
            x=qr_col_x,
            y=footer_y_bottom,
            xanchor="center",
            yanchor="bottom",
            sizex=qr_w,
            sizey=qr_h,
            layer="above",
        )
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
            stat_tile(t("ui.share.stat_rounds", locale=loc), str(rounds_played)),
            stat_tile(t("ui.share.stat_ranking", locale=loc), best_value, sub=best_sub),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "14px",
               "marginTop": "8px", "justifyContent": "center"},
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
                    className="share-ranking-hero-title",
                    disable_n_clicks=True,
                ),
                html.Ul(
                    ranking_lines,
                    className="share-ranking-hero-list",
                    disable_n_clicks=True,
                ),
            ],
            className="share-ranking-hero",
            style={
                "background": "white",
                "borderRadius": "14px",
                "padding": "18px 22px",
                "boxShadow": "0 4px 14px rgba(15,23,42,0.08)",
                "flex": "1 1 280px",
                "minWidth": "0",
                "maxWidth": "540px",
                "textAlign": "left",
            },
            disable_n_clicks=True,
        )

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

    n_panels: int = len(_formats_played_in_order(rounds))
    # Match dcc.Graph height to build_synthesis_figure(figure_height=...). Title and legend
    # are in HTML; the figure has an ~8px top margin so nothing clips at the top edge.
    graph_height: int = max(620, 130 * n_panels + 520)

    play_url: str = _play_url_from_share(share_url)
    quote_block: html.Div = html.Div(
        encourage,
        className="share-encouragement-quote",
        style={
            "padding": "4px 16px 2px 16px",
            "fontSize": "clamp(18px, 2.2vw, 22px)",
            "lineHeight": "1.38",
            "color": "#1e3a8a",
            "textAlign": "center",
            "fontWeight": "600",
        },
        disable_n_clicks=True,
    )
    qr_block: html.Div = html.Div(
        [
            html.Div(
                t("ui.share.qr_scan_to_play", locale=loc),
                className="share-play-qr-hint",
                disable_n_clicks=True,
            ),
            html.A(
                html.Img(
                    src=_qrcode_png_data_uri(play_url),
                    className="share-play-qr-img",
                    alt=t("ui.share.qr_scan_to_play", locale=loc),
                ),
                href=play_url,
                className="share-play-qr-link",
            ),
        ],
        className="share-play-qr",
        disable_n_clicks=True,
    )

    synthesis_card: html.Div = html.Div(
        [
            html.Div(
                t("ui.share.synthesis.title", locale=loc),
                className="share-synthesis-headline",
                disable_n_clicks=True,
            ),
            *(
                (_synthesis_legend_row_html(share_record, locale=loc),)
                if n_panels > 0
                else ()
            ),
            dcc.Graph(
                figure=build_synthesis_figure(
                    share_record,
                    locale=loc,
                    show_title=False,
                    show_legend_in_figure=False,
                    figure_height=graph_height,
                ),
                className="share-synthesis-graph",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "staticPlot": False,
                    "responsive": True,
                },
                style={
                    "height": f"{graph_height}px",
                    "minHeight": "400px",
                    "width": "100%",
                },
            ),
            html.Div(
                t("ui.share.synthesis.caption", locale=loc),
                style={"fontSize": "14px", "color": "#475569", "textAlign": "center",
                       "padding": "6px 16px 16px 16px", "fontStyle": "italic"},
                disable_n_clicks=True,
            ),
        ],
        style={"background": "white", "borderRadius": "18px",
               "boxShadow": "0 8px 24px rgba(15,23,42,0.08)",
               "overflow": "hidden", "marginTop": "14px", "width": "100%"},
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
            style={"fontSize": "clamp(32px,4.5vw,52px)", "margin": "0 0 2px 0",
                   "lineHeight": "1.12",
                   "color": "#0f172a", "textAlign": "center"},
        ),
        html.P(
            t("ui.share.subtitle", locale=loc),
            style={"fontSize": "clamp(16px,2.5vw,20px)",
                   "color": "#475569", "textAlign": "center",
                   "margin": "0 0 2px 0"},
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

    main_stack: list[Any] = [
        url_store,
        html.Div(
            header_children,
            style={"paddingTop": "8px"},
            disable_n_clicks=True,
        ),
        stats_row,
        quote_block,
    ]
    if played_line is not None:
        main_stack.append(played_line)
    main_stack.append(synthesis_card)
    if ranking_card is not None:
        main_stack.append(
            html.Div(
                [ranking_card, qr_block],
                className="share-ranking-qr-row",
                disable_n_clicks=True,
            )
        )
    else:
        main_stack.append(qr_block)
    main_stack.extend(
        [
            action_buttons,
            share_buttons,
            html.Div(
                t("ui.share.download_png_hint", locale=loc),
                style={"fontSize": "13px", "color": "#94a3b8",
                       "textAlign": "center", "marginTop": "14px"},
                disable_n_clicks=True,
            ),
        ]
    )

    return html.Div(
        main_stack,
        className="share-page info-page",
        id="share-page",
        disable_n_clicks=True,
        style={
            "background": "linear-gradient(135deg,#eff6ff 0%,#f8fafc 40%,#fdf2f8 100%)",
            "maxWidth": "1100px",
        },
    )
