from __future__ import annotations

import os
from pathlib import Path

import pytest


# The app's pretty Eliot renderer goes through eliottree, which currently emits
# Python 3.14 deprecation warnings for every rendered action. Tests assert
# behavior, not terminal log formatting, so keep the renderer out of pytest.
os.environ.setdefault("SUGAR_SUGAR_DISABLE_NICE_LOGS", "1")


@pytest.fixture(autouse=True)
def _isolated_share_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep share records out of the repo's data/shares.

    `create_final_layout` persists a content-addressed share record on every
    render (the /final share panel is eager), so any test that builds the
    final page would otherwise litter the developer's `data/shares/`.
    """
    monkeypatch.setenv("SUGAR_SHARE_DIR", str(tmp_path / "shares"))
