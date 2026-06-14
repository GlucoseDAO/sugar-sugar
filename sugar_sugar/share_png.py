"""Share-card PNG rendering.

The share card is built **once** as a Plotly figure
(:func:`sugar_sugar.components.share.build_share_card_figure`). Only the
**image-export step** has a fallback chain — we never redraw the card in a
second library:

1. kaleido v1 (Plotly static export; needs a Chromium binary).
2. on failure: self-provision Chrome (``kaleido.get_chrome_sync()``) and
   retry once.
3. last resort: serve the static branded site card (``assets/og-card.png``)
   so the OG image is always a valid 1200x630 raster, and log the real
   underlying error loudly.

Why no matplotlib fallback: rendering a Plotly ``go.Figure`` to a raster
*requires* a browser engine (kaleido/orca are both Chromium); a parallel
matplotlib card inevitably drifts from the real design (tiny fonts,
overlapping titles) and is what shipped a broken card to production. The
correct fix is to make the Chromium export reliable and fail *loudly* — not
to silently swap in a different-looking image.

Note on the Chromium binary: ``kaleido.get_chrome_sync()`` downloads a
self-contained Chrome-for-Testing into the user cache (no systemwide Chrome
needed), but Chromium still links against system shared libraries
(``libatk-1.0.so.0``, ``libnss3``, ``libgbm1`` …). On a slim/bare host those
must be installed or Chrome dies on launch with ``BrowserFailedError`` (not
the more helpful ``BrowserDepsError``). See README for the apt lib set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go
from eliot import start_action

from sugar_sugar.components.share import build_share_card_figure
from sugar_sugar.i18n import normalize_locale

PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"
# 1.91:1 large-image card — the size FB / LinkedIn / X / Telegram all accept.
DEFAULT_SHARE_PNG_WIDTH: int = 1200
DEFAULT_SHARE_PNG_HEIGHT: int = 630
DEFAULT_SHARE_PNG_SCALE: int = 1

# Static branded card used as the absolute last resort so the OG image URL
# always returns a real 1200x630 raster, even if Chrome cannot launch.
_STATIC_FALLBACK_CARD: Path = Path(__file__).resolve().parent.parent / "assets" / "og-card.png"


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
        from choreographer.browsers._errors import BrowserDepsError
    except ImportError:
        pass
    else:
        types.append(BrowserDepsError)
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


def _render_with_kaleido(
    fig: go.Figure,
    *,
    width: int,
    height: int,
    scale: int,
) -> bytes:
    """Plotly static export through kaleido (requires a Chromium binary)."""
    return fig.to_image(
        format="png",
        width=width,
        height=height,
        scale=scale,
    )


def _ensure_chrome_available() -> None:
    """Self-provision the choreographer-managed Chrome for kaleido.

    We check the *managed* download path, not ``find_browser`` — a
    present-but-broken system/snap chromium otherwise satisfies ``find_browser``
    and wins over (and prevents) the reliable managed download. kaleido prefers
    the managed binary, so ensuring it exists makes export deterministic.

    Kept local to avoid importing the heavy app module from the render path.
    """
    from choreographer.browsers.chromium import (
        get_chrome_download_path,
        get_old_chrome_download_path,
    )

    managed = get_chrome_download_path(mkdir=False)
    old = get_old_chrome_download_path()
    if (managed and managed.exists()) or (old and old.exists()):
        return
    import kaleido

    kaleido.get_chrome_sync()


def _static_fallback_card_bytes() -> bytes:
    """Read the static branded card; raise if it is missing/corrupt."""
    data: bytes = _STATIC_FALLBACK_CARD.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise SharePngKaleidoUnavailableError(
            f"Static fallback card is not a valid PNG: {_STATIC_FALLBACK_CARD}"
        )
    return data


def render_plotly_figure_to_png(
    fig: go.Figure,
    *,
    width: int = DEFAULT_SHARE_PNG_WIDTH,
    height: int = DEFAULT_SHARE_PNG_HEIGHT,
    scale: int = DEFAULT_SHARE_PNG_SCALE,
    allow_fallback: bool = True,
) -> bytes:
    """Render a Plotly figure to PNG bytes via kaleido, with an export-only fallback.

    The fallback branches on the *export engine*, never on the figure: on a
    kaleido/Chromium failure we self-provision Chrome and retry once, then (if
    ``allow_fallback``) serve the static branded card. The real underlying
    error is always logged so a missing-lib regression fails loudly instead of
    silently degrading.
    """
    try:
        return _render_with_kaleido(fig, width=width, height=height, scale=scale)
    except _KALEIDO_FAILURES as err:
        with start_action(action_type=u"share_png_kaleido_failed") as action:
            action.log(
                message_type=u"kaleido_failed",
                error_type=type(err).__name__,
                error=str(err),
            )
            # Self-heal: make sure a Chromium binary exists, then retry once.
            try:
                _ensure_chrome_available()
                return _render_with_kaleido(fig, width=width, height=height, scale=scale)
            except _KALEIDO_FAILURES as retry_err:
                action.log(
                    message_type=u"kaleido_failed_after_retry",
                    error_type=type(retry_err).__name__,
                    error=str(retry_err),
                )
                if not allow_fallback:
                    raise SharePngKaleidoUnavailableError(
                        "Plotly kaleido PNG export failed; Chromium could not "
                        "launch (often missing system libraries such as "
                        "libatk-1.0.so.0/libnss3 — see README)."
                    ) from retry_err
                action.log(message_type=u"share_png_static_fallback")
                return _static_fallback_card_bytes()


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
    """Build the share card figure once and export it to PNG bytes."""
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
    )
