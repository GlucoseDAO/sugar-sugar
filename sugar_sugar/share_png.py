"""Share-card PNG rendering (kaleido primary, matplotlib Agg fallback).

Primary path uses Plotly + kaleido v1, which needs a Chromium-based browser
(``choreo_get_chrome`` download or a compatible system binary). When that is
unavailable, :func:`render_share_card_png_bytes` falls back to a simplified
1080×1080 card built with matplotlib's Agg backend (no Chrome).
"""
from __future__ import annotations

import io
import math
from typing import Any, Optional

import plotly.graph_objects as go
from eliot import start_action

from sugar_sugar.components.share import (
    _best_ranking_entry,
    _format_number,
    _formats_played_in_order,
    _percent_error_series_for_round,
    _play_url_from_share,
    _prediction_next_hour_range,
    _resolve_format_label,
    _safe_display_name,
    _window_size_for_round,
    build_share_card_figure,
    compute_aggregate_stats,
)
from sugar_sugar.encouragement import encouragement_text
from sugar_sugar.i18n import normalize_locale, t

PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"
DEFAULT_SHARE_PNG_WIDTH: int = 1080
DEFAULT_SHARE_PNG_HEIGHT: int = 1080
DEFAULT_SHARE_PNG_SCALE: int = 1


class SharePngKaleidoUnavailableError(RuntimeError):
    """Raised when kaleido cannot render and fallback is disabled."""


def _kaleido_failure_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [RuntimeError, OSError]
    try:
        from choreographer.browsers._errors import BrowserFailedError
    except ImportError:
        pass
    else:
        types.append(BrowserFailedError)
    try:
        from kaleido.errors import ChromeNotFoundError as KaleidoChromeNotFoundError
    except ImportError:
        pass
    else:
        types.append(KaleidoChromeNotFoundError)
    try:
        from choreographer.errors import ChromeNotFoundError as ChoreoChromeNotFoundError
    except ImportError:
        pass
    else:
        types.append(ChoreoChromeNotFoundError)
    return tuple(types)


_KALEIDO_FAILURES: tuple[type[BaseException], ...] = _kaleido_failure_types()


def render_plotly_figure_to_png(
    fig: go.Figure,
    *,
    width: int = DEFAULT_SHARE_PNG_WIDTH,
    height: int = DEFAULT_SHARE_PNG_HEIGHT,
    scale: int = DEFAULT_SHARE_PNG_SCALE,
    allow_fallback: bool = False,
    share_record: Optional[dict[str, Any]] = None,
    share_url: str = "",
    locale: str = "en",
    seed: Optional[str] = None,
) -> bytes:
    """Render a Plotly figure to PNG bytes via kaleido, optionally with fallback."""
    try:
        return _render_with_kaleido(fig, width=width, height=height, scale=scale)
    except _KALEIDO_FAILURES as err:
        if not allow_fallback or share_record is None:
            raise SharePngKaleidoUnavailableError(
                "Plotly kaleido PNG export failed; no Chromium browser available."
            ) from err
        with start_action(action_type=u"share_png_fallback_matplotlib") as action:
            action.log(message_type="kaleido_failed", error_type=type(err).__name__)
            return _render_share_png_matplotlib_fallback(
                share_record,
                share_url=share_url,
                locale=locale,
                seed=seed,
                width=width,
                height=height,
            )


def _render_with_kaleido(
    fig: go.Figure,
    *,
    width: int,
    height: int,
    scale: int,
) -> bytes:
    """Plotly static export through kaleido (requires Chromium)."""
    return fig.to_image(
        format="png",
        width=width,
        height=height,
        scale=scale,
    )


def render_share_card_png_bytes(
    share_record: dict[str, Any],
    *,
    share_url: str,
    locale: str,
    seed: Optional[str] = None,
    width: int = DEFAULT_SHARE_PNG_WIDTH,
    height: int = DEFAULT_SHARE_PNG_HEIGHT,
    scale: int = DEFAULT_SHARE_PNG_SCALE,
    allow_fallback: bool = True,
) -> bytes:
    """Build the share card figure and export PNG bytes."""
    loc: str = normalize_locale(locale)
    fig: go.Figure = build_share_card_figure(
        share_record,
        share_url=share_url,
        locale=loc,
        seed=seed,
    )
    return render_plotly_figure_to_png(
        fig,
        width=width,
        height=height,
        scale=scale,
        allow_fallback=allow_fallback,
        share_record=share_record,
        share_url=share_url,
        locale=loc,
        seed=seed,
    )


def _render_share_png_matplotlib_fallback(
    share_record: dict[str, Any],
    *,
    share_url: str,
    locale: str,
    seed: Optional[str],
    width: int,
    height: int,
) -> bytes:
    """Simplified share card without Chrome (matplotlib Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import segno

    loc: str = normalize_locale(locale)
    rounds: list[dict[str, Any]] = list(share_record.get("rounds") or [])
    stats: dict[str, Any] = compute_aggregate_stats(rounds)
    name: str = _safe_display_name(dict(share_record.get("user_info") or {}))
    play_url: str = _play_url_from_share(share_url)

    mae: float = float(stats.get("mae_mgdl", float("nan")))
    rmse: float = float(stats.get("rmse_mgdl", float("nan")))
    accuracy: float = float(stats.get("accuracy", float("nan")))
    accuracy_str: str = (
        f"{_format_number(accuracy)}%" if not math.isnan(accuracy) else "?"
    )
    rounds_played: int = int(stats.get("rounds_played") or 0)
    best_entry: Optional[dict[str, Any]] = _best_ranking_entry(share_record)
    percentile: Optional[int] = (
        int(best_entry["percentile"]) if best_entry and best_entry.get("percentile") is not None else None
    )
    encourage: str = encouragement_text(stats, loc, seed=seed)

    formats: list[str] = _formats_played_in_order(rounds)
    n_panels: int = max(1, len(formats))

    dpi: int = 100
    fig_w_in: float = width / dpi
    fig_h_in: float = height / dpi
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,
        squeeze=False,
    )
    fig.patch.set_facecolor("#f8fafc")

    title: str = t("ui.share.subtitle", locale=loc)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    header_lines: list[str] = []
    if name:
        header_lines.append(name)
    header_lines.append(
        f"{t('ui.share.stat_mae', locale=loc)} {_format_number(mae)} · "
        f"{t('ui.share.stat_rmse', locale=loc)} {_format_number(rmse)} · "
        f"{t('ui.share.stat_rounds', locale=loc)} {rounds_played} · "
        f"{accuracy_str}"
    )
    if percentile is not None:
        header_lines.append(
            t("ui.share.stat_percentile", locale=loc, percentile=str(percentile))
        )
    fig.text(0.5, 0.92, "\n".join(header_lines), ha="center", va="top", fontsize=9)
    fig.text(
        0.5,
        0.86,
        encourage[:220] + ("…" if len(encourage) > 220 else ""),
        ha="center",
        va="top",
        fontsize=8,
        color="#334155",
        wrap=True,
    )

    panel_axes: list[Any] = list(axes.flat)
    if not formats:
        ax0 = panel_axes[0]
        ax0.set_axis_off()
        ax0.text(
            0.5,
            0.5,
            t("ui.share.synthesis.empty", locale=loc),
            ha="center",
            va="center",
            transform=ax0.transAxes,
        )
    else:
        for ax, fmt in zip(panel_axes, formats, strict=False):
            fmt_rounds: list[dict[str, Any]] = [
                r for r in rounds if str(r.get("format") or "").upper() == fmt
            ]
            ax.set_facecolor("white")
            ax.axhline(0.0, color="black", linewidth=1.2, zorder=1)
            ax.set_title(_resolve_format_label(fmt, locale=loc), fontsize=9)
            ax.set_ylabel(t("ui.share.synthesis.y_axis_short", locale=loc), fontsize=8)
            for rnd in fmt_rounds:
                ws: int = _window_size_for_round(rnd)
                hour_range: Optional[tuple[int, int]] = _prediction_next_hour_range(ws)
                if hour_range is None:
                    continue
                ys: list[Optional[float]] = _percent_error_series_for_round(rnd, hour_range)
                xs: list[int] = list(range(len(ys)))
                y_vals: list[float] = [float(y) for y in ys if y is not None]
                x_vals: list[int] = [x for x, y in zip(xs, ys, strict=True) if y is not None]
                if x_vals:
                    ax.plot(x_vals, y_vals, linewidth=1.0, alpha=0.55)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(t("ui.share.synthesis.x_axis_time", locale=loc), fontsize=8)

    qr_ax = fig.add_axes([0.82, 0.02, 0.15, 0.15])
    qr_ax.set_axis_off()
    buf = io.BytesIO()
    segno.make(play_url, error="m").save(buf, kind="png", scale=4, border=1)
    buf.seek(0)
    qr_img = plt.imread(buf)
    qr_ax.imshow(qr_img)

    fig.subplots_adjust(left=0.08, right=0.78, top=0.84, bottom=0.08, hspace=0.35)
    out = io.BytesIO()
    fig.savefig(out, format="png", facecolor=fig.patch.get_facecolor())
    plt.close(fig)
    png_bytes: bytes = out.getvalue()
    if not png_bytes.startswith(PNG_SIGNATURE):
        raise RuntimeError("matplotlib fallback did not produce a valid PNG")
    return png_bytes
