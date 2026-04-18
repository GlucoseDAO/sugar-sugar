"""Persistent share-record store.

Backing: one JSON file per share under `data/shares/<share_id>.json`.

Why a file and not an in-memory dict:
- Dash's debug reloader forks a child process on every reload; an in-memory
  dict would get recreated and invalidate share URLs mid-session.
- Multi-worker / container deploys (gunicorn with >1 worker) need a shared
  source of truth.  A plain JSON file trivially satisfies that without
  pulling in a database.

The schema is intentionally a thin dict (no pydantic) -- everything is
JSON-serialisable because the caller hands us a trimmed-down `user_info`
plus a handful of precomputed stats.  If the schema grows, upgrade here.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Optional

from eliot import start_action


_SHARE_DIR_ENV: str = "SUGAR_SHARE_DIR"
_SHARE_ID_LEN: int = 10  # ~60 bits of entropy; short, URL-safe.


def _share_dir() -> Path:
    """Resolve the share-record directory.  Defaults to repo-root/data/shares.

    The env var `SUGAR_SHARE_DIR` can override the location for tests or
    multi-worker deployments that want to point at shared storage.
    """
    override: Optional[str] = os.environ.get(_SHARE_DIR_ENV)
    root: Path = Path(override) if override else Path(__file__).resolve().parent.parent / "data" / "shares"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_share_id() -> str:
    """Generate a URL-safe share id that does not collide with an existing file."""
    directory = _share_dir()
    for _ in range(8):
        candidate = secrets.token_urlsafe(_SHARE_ID_LEN)[:_SHARE_ID_LEN]
        if not (directory / f"{candidate}.json").exists():
            return candidate
    # Extremely unlikely; fall back to a longer token.
    return secrets.token_urlsafe(_SHARE_ID_LEN * 2)


def save_share(record: dict[str, Any]) -> str:
    """Persist `record` to disk and return the generated share id.

    Writes are atomic: serialise to a temp file in the same directory then
    rename over the target so concurrent readers never observe a partial
    JSON document.
    """
    share_id: str = _new_share_id()
    directory: Path = _share_dir()
    target: Path = directory / f"{share_id}.json"

    with start_action(action_type=u"save_share", share_id=share_id):
        fd, tmp_path = tempfile.mkstemp(prefix=f".{share_id}.", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, target)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    return share_id


def load_share(share_id: str) -> Optional[dict[str, Any]]:
    """Load a share record by id.  Returns None if missing or malformed."""
    if not share_id or "/" in share_id or "\\" in share_id or ".." in share_id:
        return None
    target: Path = _share_dir() / f"{share_id}.json"
    if not target.is_file():
        return None
    with start_action(action_type=u"load_share", share_id=share_id) as action:
        text = target.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            action.log(message_type=u"share_record_corrupt")
            return None
        if not isinstance(data, dict):
            action.log(message_type=u"share_record_not_dict")
            return None
        return data
