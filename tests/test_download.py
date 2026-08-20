from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sugar_sugar import download
from sugar_sugar.download import fetch_public_datasets


def _record(
    called: list[tuple[str, Path, dict[str, Any]]], name: str
) -> object:
    def _fetch(dest: Path, **kwargs: Any) -> Path:
        called.append((name, dest, kwargs))
        return dest

    return _fetch


def test_fetch_public_datasets_downloads_format_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[str, Path, dict[str, Any]]] = []
    monkeypatch.setattr(download, "fetch_bigideas", _record(called, "bigideas"))
    monkeypatch.setattr(download, "fetch_d1namo", _record(called, "d1namo"))
    monkeypatch.setattr(download, "fetch_cgmacros", _record(called, "cgmacros"))

    fetch_public_datasets()

    names = [name for name, _, _ in called]
    assert names == ["bigideas", "d1namo"]
    bigideas_kwargs = called[0][2]
    d1namo_kwargs = called[1][2]
    assert bigideas_kwargs == {"force": False}
    assert d1namo_kwargs["include_photos"] is True
    assert d1namo_kwargs["force"] is False
    assert d1namo_kwargs["keep_zip"] is False


def test_fetch_public_datasets_all_includes_cgmacros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[str, Path, dict[str, Any]]] = []
    monkeypatch.setattr(download, "fetch_bigideas", _record(called, "bigideas"))
    monkeypatch.setattr(download, "fetch_d1namo", _record(called, "d1namo"))
    monkeypatch.setattr(download, "fetch_cgmacros", _record(called, "cgmacros"))

    fetch_public_datasets(
        force=True,
        include_photos=False,
        keep_zip=True,
        include_cgmacros=True,
    )

    names = [name for name, _, _ in called]
    assert names == ["bigideas", "d1namo", "cgmacros"]
    assert called[0][2] == {"force": True}
    assert called[1][2] == {
        "include_photos": False,
        "force": True,
        "keep_zip": True,
    }
    assert called[2][2] == {
        "include_photos": False,
        "force": True,
        "keep_zip": True,
    }
