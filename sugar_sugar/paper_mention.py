"""Optional scientific-paper acknowledgment opt-in.

A player may ask to be named in a paper published after the study. That is
publication consent, not a leaderboard label: the full name is stored on the
study/consent records so authors can build an acknowledgments list. The
published mention is only used when the player completed enough rounds
(``MAX_ROUNDS`` or more) — the form copy states that; enforcement is at
write-up time, not at save time.
"""
from __future__ import annotations

import unicodedata
from typing import Any, Final

from sugar_sugar.challenge_unknown import challenge_unknown_checked

MAX_PAPER_NAME_LENGTH: Final[int] = 80
# Acknowledgments only include people who completed a full 12-round game (or more).
PAPER_MENTION_MIN_ROUNDS: Final[int] = 12


def normalize_paper_full_name(raw: Any) -> str:
    """Collapse whitespace and drop control characters; empty when nothing usable."""
    if not raw:
        return ""
    cleaned = "".join(
        " " if ch.isspace() else ch
        for ch in str(raw)
        if not unicodedata.category(ch).startswith("C")
    )
    return " ".join(cleaned.split())[:MAX_PAPER_NAME_LENGTH].strip()


def resolve_paper_mention(checked: Any, full_name: Any) -> tuple[bool, str]:
    """Return ``(opted_in, name)``. Both the box and a real name are required."""
    name = normalize_paper_full_name(full_name)
    if challenge_unknown_checked(checked) and name:
        return True, name
    return False, ""
