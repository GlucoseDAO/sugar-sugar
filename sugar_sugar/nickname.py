"""Optional leaderboard nicknames and the identity key that groups ranking rows.

Two concerns live here, both pure (no Dash imports) so `app.py` and
`components/submit.py` can share them:

**Nicknames.** A player may pick an optional display name for `/highscore` and the
`/final` leaderboard.  It is a *public display label*, deliberately NOT part of the
research record: it is written only to `data/input/prediction_ranking*.csv`, never to
`prediction_statistics.csv` and never to `consent_agreement.csv`.  That is why it needs
no consent wording.  Players who pick nothing keep the anonymous ``Player N`` label.

**Identity.** Ranking rows are keyed by ``study_id``, but a new device or a wiped
localStorage mints a fresh one, so one person could occupy several board rows.  Rows are
therefore grouped by :func:`email_key` — a salted HMAC of the address, never the address
itself, because the ranking CSVs are read by a public page and the consent notice promises
that the study-ID-to-email mapping lives in a separate encrypted file.  Players without an
email fall back to grouping by ``study_id``, i.e. exactly today's behaviour.

.. warning::
   The salt must stay **stable for the lifetime of a deployment**.  Changing it changes
   every ``email_key``, which re-splits existing players into brand-new identities.  Set
   ``RANKING_EMAIL_SALT`` explicitly to pin it, or let this module persist a random salt
   once to ``data/.ranking_salt`` (gitignored) and leave that file alone.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Final, Optional

from sugar_sugar.config import RANKING_EMAIL_SALT

# Long enough for a real handle, short enough not to break the leaderboard's player cell.
MAX_NICKNAME_LENGTH: Final[int] = 24

# Truncated HMAC: 16 hex chars = 64 bits, ample to avoid collisions among study
# participants while keeping the CSV column narrow and eyeball-friendly.
_EMAIL_KEY_HEX_LENGTH: Final[int] = 16

_SALT_FILE: Final[Path] = Path(__file__).resolve().parents[1] / "data" / ".ranking_salt"


def normalize_nickname(raw: Optional[str]) -> str:
    """Return a safe public display name, or ``""`` when there is nothing to show.

    Collapses all whitespace runs to single spaces, drops Unicode control/format
    characters (category ``C*`` — this also removes the bidi overrides and zero-width
    joiners used to spoof other players' names), and caps the length.
    """
    if not raw:
        return ""
    cleaned = "".join(
        " " if ch.isspace() else ch
        for ch in str(raw)
        if not unicodedata.category(ch).startswith("C")
    )
    return " ".join(cleaned.split())[:MAX_NICKNAME_LENGTH].strip()


def normalize_email(email: Optional[str]) -> str:
    """Casefolded, trimmed address — the input to :func:`email_key`."""
    return str(email or "").strip().casefold()


@lru_cache(maxsize=1)
def _salt() -> bytes:
    """Deployment-stable HMAC salt.

    ``RANKING_EMAIL_SALT`` wins when set.  Otherwise a random salt is generated once and
    persisted to ``data/.ranking_salt`` with owner-only permissions, so a fresh install
    gets a real secret without any configuration step.
    """
    if RANKING_EMAIL_SALT:
        return RANKING_EMAIL_SALT.encode("utf-8")
    if _SALT_FILE.exists():
        stored = _SALT_FILE.read_text(encoding="utf-8").strip()
        if stored:
            return stored.encode("utf-8")
    generated = secrets.token_hex(32)
    _SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SALT_FILE.write_text(generated, encoding="utf-8")
    _SALT_FILE.chmod(0o600)
    return generated.encode("utf-8")


def deployment_salt() -> bytes:
    """Deployment-stable secret salt for HMAC-derived public identifiers.

    Shared by :func:`email_key` and the content-addressed share ids in
    ``share_store`` (each HMAC uses its own domain prefix, so the derived
    keys never collide across uses).  Same stability warning as the module
    docstring: never rotate it.
    """
    return _salt()


def email_key(email: Optional[str]) -> str:
    """Non-reversible grouping key for an address; ``""`` when there is no email.

    Case- and whitespace-insensitive, so ``" Ann@X.COM "`` and ``"ann@x.com"`` group
    together.  Stored in the ranking CSVs in place of the address itself.
    """
    normalized = normalize_email(email)
    if not normalized:
        return ""
    digest = hmac.new(_salt(), normalized.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:_EMAIL_KEY_HEX_LENGTH]


def identity_key(*, key: str, study_id: str) -> str:
    """The leaderboard identity a ranking row belongs to.

    ``e:<email_key>`` when the player gave an email — that merges their rows across
    devices and sessions — otherwise ``s:<study_id>``, which preserves the old
    one-row-per-session behaviour for anonymous players.
    """
    return f"e:{key}" if key else f"s:{study_id}"
