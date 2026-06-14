"""Integration tests for share-card PNG rendering."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pytest

from sugar_sugar.components.share import build_share_card_figure
from sugar_sugar.i18n import setup_i18n
from sugar_sugar.share_png import (
    DEFAULT_SHARE_PNG_HEIGHT,
    DEFAULT_SHARE_PNG_WIDTH,
    PNG_SIGNATURE,
    SharePngKaleidoUnavailableError,
    _STATIC_FALLBACK_CARD,
    _render_with_kaleido,
    render_plotly_figure_to_png,
    render_share_card_png_bytes,
)
from tests.share_fixtures import make_test_share_record


@pytest.fixture
def share_record() -> dict[str, Any]:
    setup_i18n()
    return make_test_share_record()


def _png_dimensions(png: bytes) -> tuple[int, int]:
    """Read PNG dimensions from the IHDR chunk."""
    assert png.startswith(PNG_SIGNATURE)
    return struct.unpack(">II", png[16:24])


def test_share_card_figure_is_social_landscape(
    share_record: dict[str, Any],
) -> None:
    """The Plotly card is the 1.91:1 OG/Twitter/LinkedIn landscape size."""
    fig: go.Figure = build_share_card_figure(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    assert fig.layout.width == DEFAULT_SHARE_PNG_WIDTH
    assert fig.layout.height == DEFAULT_SHARE_PNG_HEIGHT


def test_export_without_fallback_raises_after_retry(
    share_record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kaleido cannot export and fallback is disabled, raise loudly."""
    monkeypatch.setattr(
        "sugar_sugar.share_png._render_with_kaleido",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced kaleido failure for test")
        ),
    )
    monkeypatch.setattr("sugar_sugar.share_png._ensure_chrome_available", lambda: None)
    fig: go.Figure = build_share_card_figure(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    with pytest.raises(SharePngKaleidoUnavailableError):
        render_plotly_figure_to_png(fig, allow_fallback=False)


def test_static_card_fallback_produces_valid_png(
    share_record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback path returns the static branded card, not a redraw in another library."""
    monkeypatch.setattr(
        "sugar_sugar.share_png._render_with_kaleido",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced kaleido failure for test")
        ),
    )
    monkeypatch.setattr("sugar_sugar.share_png._ensure_chrome_available", lambda: None)
    png: bytes = render_share_card_png_bytes(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
        allow_fallback=True,
    )
    assert png == Path(_STATIC_FALLBACK_CARD).read_bytes()
    assert _png_dimensions(png) == (DEFAULT_SHARE_PNG_WIDTH, DEFAULT_SHARE_PNG_HEIGHT)


def test_render_share_card_png_bytes_valid_png(share_record: dict[str, Any]) -> None:
    """End-to-end export returns PNG bytes (kaleido or fallback)."""
    png: bytes = render_share_card_png_bytes(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    assert _png_dimensions(png) == (DEFAULT_SHARE_PNG_WIDTH, DEFAULT_SHARE_PNG_HEIGHT)
    assert len(png) > 8_000


def test_kaleido_direct_can_render_small_png(share_record: dict[str, Any]) -> None:
    """Low-level kaleido call can render once Chrome is provisioned."""
    fig: go.Figure = build_share_card_figure(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    png: bytes = _render_with_kaleido(fig, width=400, height=210, scale=1)
    assert _png_dimensions(png) == (400, 210)
