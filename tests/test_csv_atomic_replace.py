"""Retry a brief Windows lock on the statistics CSV replace."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sugar_sugar.components.submit import _replace_atomically


def test_replace_atomically_retries_permission_error(tmp_path: Path) -> None:
    dest = tmp_path / "prediction_statistics.csv"
    dest.write_text("old", encoding="utf-8")
    tmp = tmp_path / "prediction_statistics.csv.tmp"
    tmp.write_text("new", encoding="utf-8")

    calls = {"n": 0}
    real_replace = Path.replace

    def flaky(self: Path, target: Path | str) -> Path:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Access is denied")
        return real_replace(self, target)

    with (
        patch.object(Path, "replace", flaky),
        patch("sugar_sugar.components.submit.time.sleep"),
    ):
        _replace_atomically(tmp, dest)

    assert dest.read_text(encoding="utf-8") == "new"
    assert calls["n"] == 3


def test_replace_atomically_raises_after_retries(tmp_path: Path) -> None:
    dest = tmp_path / "prediction_statistics.csv"
    dest.write_text("old", encoding="utf-8")
    tmp = tmp_path / "prediction_statistics.csv.tmp"
    tmp.write_text("new", encoding="utf-8")

    def always_locked(self: Path, target: Path | str) -> Path:
        raise PermissionError(13, "Access is denied")

    with (
        patch.object(Path, "replace", always_locked),
        patch("sugar_sugar.components.submit.time.sleep"),
        pytest.raises(PermissionError),
    ):
        _replace_atomically(tmp, dest)
