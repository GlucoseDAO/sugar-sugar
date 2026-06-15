"""Server-side cross-device resume store.

Backing: one JSON file per session under ``data/resume/<code>.json``.

Why this exists: all in-browser session state lives in ``localStorage``, which is
**per-device**. A game started on a phone therefore cannot be continued on a
desktop. This store keeps a server-side copy of the live session keyed by a short,
human-typeable **resume code**, so the user can re-enter that code on another
device and pick up where they left off. It deliberately mirrors
``share_store.py`` (atomic writes, file-per-record, env override for the dir).

The payload is a thin, JSON-serialisable snapshot of the Dash stores needed to
restore the game (``user_info``, ``full_df``, ``current_window_df``, ``events_df``,
``last_visited_page``, ``glucose_unit``). It is overwritten in place each time the
same code is re-saved, so the server copy tracks the latest state.

Resume codes are session-transfer tokens: anyone with the code can resume the
session, so treat them like a one-time login link, not a public id.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Optional

from eliot import start_action


_RESUME_DIR_ENV: str = "SUGAR_RESUME_DIR"
# Human-typeable alphabet: uppercase, no ambiguous glyphs (O/0, I/1, L).
_RESUME_ALPHABET: str = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_RESUME_CODE_LEN: int = 6  # ~30 bits; fine for a short-lived transfer token.


def _resume_dir() -> Path:
    """Resolve the resume-store directory. Defaults to repo-root/data/resume."""
    override: Optional[str] = os.environ.get(_RESUME_DIR_ENV)
    root: Path = Path(override) if override else Path(__file__).resolve().parent.parent / "data" / "resume"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_code(code: Optional[str]) -> Optional[str]:
    """Uppercase + strip a user-entered code; reject anything unsafe/malformed."""
    if not code:
        return None
    c = code.strip().upper().replace("-", "").replace(" ", "")
    if not c or not all(ch in _RESUME_ALPHABET for ch in c):
        return None
    return c


def new_code() -> str:
    """Generate a resume code that does not collide with an existing file."""
    directory = _resume_dir()
    for _ in range(8):
        candidate = "".join(secrets.choice(_RESUME_ALPHABET) for _ in range(_RESUME_CODE_LEN))
        if not (directory / f"{candidate}.json").exists():
            return candidate
    return "".join(secrets.choice(_RESUME_ALPHABET) for _ in range(_RESUME_CODE_LEN * 2))


def save_session(code: str, payload: dict[str, Any]) -> Optional[str]:
    """Persist `payload` under `code` (atomic). Returns the normalized code or None.

    Overwrites any existing record for the same code so the server copy always
    reflects the latest session state.
    """
    norm = _normalize_code(code)
    if norm is None:
        return None
    directory: Path = _resume_dir()
    target: Path = directory / f"{norm}.json"
    with start_action(action_type=u"save_resume_session", code=norm):
        fd, tmp_path = tempfile.mkstemp(prefix=f".{norm}.", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, target)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    return norm


def load_session(code: Optional[str]) -> Optional[dict[str, Any]]:
    """Load a session payload by code (case-insensitive). None if missing/malformed."""
    norm = _normalize_code(code)
    if norm is None:
        return None
    target: Path = _resume_dir() / f"{norm}.json"
    if not target.is_file():
        return None
    with start_action(action_type=u"load_resume_session", code=norm) as action:
        text = target.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            action.log(message_type=u"resume_record_corrupt")
            return None
        if not isinstance(data, dict):
            return None
        return data
