from __future__ import annotations

from sugar_sugar.config import MAX_ROUNDS
from sugar_sugar.paper_mention import (
    MAX_PAPER_NAME_LENGTH,
    PAPER_MENTION_MIN_ROUNDS,
    normalize_paper_full_name,
    resolve_paper_mention,
)


def test_paper_mention_requires_checkbox_and_name() -> None:
    assert resolve_paper_mention(["on"], "Ada Lovelace") == (True, "Ada Lovelace")
    assert resolve_paper_mention(True, "Ada Lovelace") == (True, "Ada Lovelace")
    assert resolve_paper_mention([], "Ada Lovelace") == (False, "")
    assert resolve_paper_mention(["on"], "   ") == (False, "")
    assert resolve_paper_mention(None, None) == (False, "")


def test_paper_full_name_is_normalized_and_capped() -> None:
    assert normalize_paper_full_name("  Ada   Lovelace  ") == "Ada Lovelace"
    assert len(normalize_paper_full_name("A" * 200)) == MAX_PAPER_NAME_LENGTH
    opted, name = resolve_paper_mention(["on"], "  Ada   Lovelace  ")
    assert opted is True
    assert name == "Ada Lovelace"


def test_paper_mention_min_rounds_is_a_full_game() -> None:
    """The floor tracks MAX_ROUNDS: the form copy renders this exact number."""
    assert PAPER_MENTION_MIN_ROUNDS == MAX_ROUNDS
