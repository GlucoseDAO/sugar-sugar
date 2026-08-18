"""Public FAQ question board (participant + developer posts and replies)."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from eliot import start_action

_FAQ_DIR_ENV: str = "SUGAR_FAQ_DIR"
_FAQ_BOARD_ENABLED_ENV: str = "FAQ_BOARD_ENABLED"
_ALLOWED_SECTIONS: frozenset[str] = frozenset({"participant", "developer"})
_ALLOWED_TAGS: tuple[str, ...] = ("gameplay", "data", "privacy", "download", "other")
_MAX_TEXT: int = 2000
_MAX_NAME: int = 40
_MAX_ITEMS: int = 400


def faq_board_enabled() -> bool:
    """Public ask/reply board. Off until bot protection exists; set FAQ_BOARD_ENABLED=1 to restore."""
    return os.getenv(_FAQ_BOARD_ENABLED_ENV, "0").lower() in ("1", "true", "yes")


def faq_board_path() -> Path:
    override = os.environ.get(_FAQ_DIR_ENV)
    root = Path(override) if override else Path(__file__).resolve().parent.parent / "data" / "faq"
    root.mkdir(parents=True, exist_ok=True)
    return root / "questions.json"


def allowed_faq_tags() -> tuple[str, ...]:
    return _ALLOWED_TAGS


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_text(raw: Any, *, limit: int) -> str:
    return str(raw or "").strip()[:limit]


def _clean_section(raw: Any) -> str:
    text = str(raw or "participant").strip().lower()
    return text if text in _ALLOWED_SECTIONS else "participant"


def _clean_tags(raw: Any) -> list[str]:
    values = raw if isinstance(raw, (list, tuple)) else []
    cleaned: list[str] = []
    for item in values:
        tag = str(item or "").strip().lower()
        if tag in _ALLOWED_TAGS and tag not in cleaned:
            cleaned.append(tag)
    return cleaned or ["other"]


def load_faq_questions() -> list[dict[str, Any]]:
    path = faq_board_path()
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _write_questions(items: list[dict[str, Any]]) -> None:
    path = faq_board_path()
    directory = path.parent
    fd, tmp_path = tempfile.mkstemp(prefix=".faq.", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"items": items}, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def add_faq_question(
    *,
    text: str,
    section: str,
    tags: Optional[list[str]] = None,
    name: str = "",
) -> Optional[dict[str, Any]]:
    if not faq_board_enabled():
        return None
    body = _clean_text(text, limit=_MAX_TEXT)
    if not body:
        return None
    item = {
        "id": secrets.token_urlsafe(8),
        "section": _clean_section(section),
        "name": _clean_text(name, limit=_MAX_NAME),
        "text": body,
        "tags": _clean_tags(tags),
        "created_at": _now(),
        "replies": [],
    }
    with start_action(action_type="faq_add_question", section=item["section"]):
        items = load_faq_questions()
        items.append(item)
        _write_questions(items[-_MAX_ITEMS:])
    return item


def add_faq_reply(
    question_id: str,
    *,
    text: str,
    section: str,
    name: str = "",
) -> Optional[dict[str, Any]]:
    if not faq_board_enabled():
        return None
    body = _clean_text(text, limit=_MAX_TEXT)
    qid = str(question_id or "").strip()
    if not body or not qid:
        return None
    reply = {
        "id": secrets.token_urlsafe(6),
        "section": _clean_section(section),
        "name": _clean_text(name, limit=_MAX_NAME),
        "text": body,
        "created_at": _now(),
    }
    with start_action(action_type="faq_add_reply", question_id=qid):
        items = load_faq_questions()
        for item in items:
            if str(item.get("id")) == qid:
                replies = list(item.get("replies") or [])
                replies.append(reply)
                item["replies"] = replies
                _write_questions(items)
                return reply
    return None
