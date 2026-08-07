"""Nickname sanitising and the one-way email key that groups leaderboard rows.

The nickname is rendered on a public page, so `normalize_nickname` is the only thing
standing between a typed string and the leaderboard. The `email_key` must be stable
(it is the grouping key stored in the ranking CSVs) and must never be the address.
"""

from __future__ import annotations

import pytest

from sugar_sugar import nickname as nickname_module
from sugar_sugar.nickname import (
    MAX_NICKNAME_LENGTH,
    email_key,
    identity_key,
    normalize_email,
    normalize_nickname,
)


@pytest.fixture(autouse=True)
def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the salt so hashes are deterministic and no salt file is touched."""
    monkeypatch.setattr(nickname_module, "RANKING_EMAIL_SALT", "test-salt")
    nickname_module._salt.cache_clear()
    yield
    nickname_module._salt.cache_clear()


# --- normalize_nickname ------------------------------------------------------


def test_none_and_blank_become_empty() -> None:
    assert normalize_nickname(None) == ""
    assert normalize_nickname("") == ""
    assert normalize_nickname("   ") == ""


def test_surrounding_and_inner_whitespace_is_collapsed() -> None:
    assert normalize_nickname("  Sugar   Ninja \t\n ") == "Sugar Ninja"


def test_length_is_capped() -> None:
    assert normalize_nickname("N" * 100) == "N" * MAX_NICKNAME_LENGTH


def test_cap_does_not_leave_trailing_space() -> None:
    raw = ("a" * (MAX_NICKNAME_LENGTH - 1)) + " tail"
    assert normalize_nickname(raw) == "a" * (MAX_NICKNAME_LENGTH - 1)


def test_control_and_bidi_characters_are_dropped() -> None:
    """Zero-width joiners and bidi overrides are how one player spoofs another."""
    assert normalize_nickname("Ni​nja‮") == "Ninja"
    assert normalize_nickname("a\x00b\x1fc") == "abc"


def test_ordinary_unicode_survives() -> None:
    assert normalize_nickname("Zoë 🍬") == "Zoë 🍬"


# --- email_key ---------------------------------------------------------------


def test_no_email_yields_no_key() -> None:
    assert email_key(None) == ""
    assert email_key("") == ""
    assert email_key("   ") == ""


def test_key_is_case_and_whitespace_insensitive() -> None:
    assert email_key(" Ann@X.COM ") == email_key("ann@x.com")


def test_key_is_stable_across_calls() -> None:
    assert email_key("ann@x.com") == email_key("ann@x.com")


def test_different_addresses_differ() -> None:
    assert email_key("ann@x.com") != email_key("bob@x.com")


def test_key_never_contains_the_address() -> None:
    key = email_key("ann@x.com")
    assert "ann" not in key and "@" not in key
    assert len(key) == 16 and all(c in "0123456789abcdef" for c in key)


def test_salt_changes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents why the salt must not be rotated: it re-splits every player."""
    before = email_key("ann@x.com")
    monkeypatch.setattr(nickname_module, "RANKING_EMAIL_SALT", "other-salt")
    nickname_module._salt.cache_clear()
    assert email_key("ann@x.com") != before


def test_normalize_email_casefolds_and_trims() -> None:
    assert normalize_email("  Ann@X.COM ") == "ann@x.com"
    assert normalize_email(None) == ""


# --- identity_key ------------------------------------------------------------


def test_identity_prefers_the_email_key() -> None:
    assert identity_key(key="abc123", study_id="s1") == "e:abc123"


def test_identity_falls_back_to_study_id() -> None:
    """Anonymous players keep the old one-row-per-session behaviour."""
    assert identity_key(key="", study_id="s1") == "s:s1"
