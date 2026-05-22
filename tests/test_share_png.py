"""Integration tests for share-card PNG rendering."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import pytest

from sugar_sugar.components.share import build_share_card_figure
from sugar_sugar.share_png import (
    PNG_SIGNATURE,
    SharePngKaleidoUnavailableError,
    _render_with_kaleido,
    render_plotly_figure_to_png,
    render_share_card_png_bytes,
)
from tests.share_fixtures import make_test_share_record


def _hide_all_browsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a prod host with no Chrome/Chromium on PATH or in choreographer cache."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    from choreographer.browsers.chromium import Chromium

    @classmethod
    def _no_browser(cls, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(Chromium, "find_browser", _no_browser)


@pytest.fixture
def share_record() -> dict[str, Any]:
    return make_test_share_record()


def test_kaleido_primary_fails_without_browser(
    share_record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kaleido-only export must fail when no Chromium binary is available."""
    _hide_all_browsers(monkeypatch)
    fig: go.Figure = build_share_card_figure(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    with pytest.raises(SharePngKaleidoUnavailableError):
        render_plotly_figure_to_png(
            fig,
            allow_fallback=False,
            share_record=share_record,
            share_url="http://localhost/share/test-id",
            locale="en",
            seed="test-id",
        )


def test_matplotlib_fallback_produces_valid_png(
    share_record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback path must return a real PNG without invoking kaleido."""
    _hide_all_browsers(monkeypatch)
    monkeypatch.setattr(
        "sugar_sugar.share_png._render_with_kaleido",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SharePngKaleidoUnavailableError("forced kaleido failure for test")
        ),
    )
    png: bytes = render_share_card_png_bytes(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
        allow_fallback=True,
    )
    assert png[: len(PNG_SIGNATURE)] == PNG_SIGNATURE
    assert len(png) > 8_000


def test_render_share_card_png_bytes_valid_png(share_record: dict[str, Any]) -> None:
    """End-to-end export returns PNG bytes (kaleido or fallback)."""
    png: bytes = render_share_card_png_bytes(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    assert png[: len(PNG_SIGNATURE)] == PNG_SIGNATURE
    assert len(png) > 8_000


def test_kaleido_direct_raises_when_browser_hidden(
    share_record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-level kaleido call fails on browser-less hosts (documents prod risk)."""
    _hide_all_browsers(monkeypatch)
    fig: go.Figure = build_share_card_figure(
        share_record,
        share_url="http://localhost/share/test-id",
        locale="en",
        seed="test-id",
    )
    with pytest.raises(Exception):
        _render_with_kaleido(fig, width=400, height=300, scale=1)
