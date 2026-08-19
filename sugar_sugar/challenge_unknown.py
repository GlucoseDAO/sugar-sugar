"""Challenge the unknown: mix the opposite Format A pool on purpose.

Default Format A routing is fixed by diabetes status. Only players whose
default pool is already a single corpus can opt in: no diabetes (BIG IDEAs)
or type 1 (D1NAMO). Type 2 / prediabetes / LADA already mix both corpora, so
the checkbox is hidden. Gestational stays on BIG IDEAs and is not offered the
challenge. Half the rounds then come from the other corpus. Format B (own
data) never uses this.
"""
from __future__ import annotations

from typing import Any, Optional

POOL_BIGIDEAS: str = "bigideas"
POOL_D1NAMO: str = "d1namo"
CHALLENGE_FORMATS: frozenset[str] = frozenset({"A", "C"})
# Fixed opposite-pool share so every opted-in player is comparable.
CHALLENGE_UNKNOWN_PCT: int = 50
CHALLENGE_OPPOSITE_SHARE: float = CHALLENGE_UNKNOWN_PCT / 100.0
MIX_POLICY_PREFIX: str = "mix:"


def _diabetes_kind(user_info: dict[str, Any] | None) -> str:
    info = user_info or {}
    if info.get("diabetic") is not True:
        return "none"
    return str(info.get("diabetic_type") or "").strip().lower()


def is_type_1(user_info: dict[str, Any] | None) -> bool:
    return _diabetes_kind(user_info) in {"type 1", "type1", "t1"}


def challenge_unknown_checked(raw: Any) -> bool:
    """True when the startup checklist (or a leftover bool store) is opted in."""
    if raw is True:
        return True
    if isinstance(raw, (list, tuple, set)):
        return any(item in {"on", True, "true", 1} for item in raw)
    return False


def _format_code(
    user_info: dict[str, Any] | None,
    format_value: Optional[str] = None,
) -> str:
    info = user_info or {}
    return str(format_value if format_value is not None else info.get("format") or "").strip().upper()


def _pure_pool_player(user_info: dict[str, Any] | None) -> bool:
    """True for the two groups whose Format A default is one corpus, not a mix."""
    info = user_info or {}
    if info.get("diabetic") is True:
        return is_type_1(info)
    return info.get("diabetic") is False


def challenge_unknown_visible(
    user_info: dict[str, Any] | None = None,
    format_value: Optional[str] = None,
) -> bool:
    """Show only for non-diabetic or type 1, on Public / Public + My Data."""
    return challenge_unknown_eligible(user_info, format_value)


def challenge_unknown_eligible(
    user_info: dict[str, Any] | None = None,
    format_value: Optional[str] = None,
) -> bool:
    """True for Format A/C when the player is non-diabetic or type 1."""
    return _format_code(user_info, format_value) in CHALLENGE_FORMATS and _pure_pool_player(
        user_info
    )


def challenge_unknown_active(user_info: dict[str, Any] | None) -> bool:
    info = user_info or {}
    return bool(info.get("challenge_unknown")) and challenge_unknown_eligible(info)


def encode_mix_policy(weights: dict[str, float]) -> str:
    parts = [
        f"{pool}={weight:.2f}"
        for pool, weight in weights.items()
        if weight > 0
    ]
    return MIX_POLICY_PREFIX + ",".join(parts)


def parse_mix_policy(policy: str | None) -> dict[str, float] | None:
    text = str(policy or "").strip()
    if not text.startswith(MIX_POLICY_PREFIX):
        return None
    body = text[len(MIX_POLICY_PREFIX):]
    weights: dict[str, float] = {}
    for chunk in body.split(","):
        if "=" not in chunk:
            continue
        name, raw = chunk.split("=", 1)
        pool = name.strip().lower()
        try:
            weight = float(raw)
        except ValueError:
            continue
        if pool and weight > 0:
            weights[pool] = weight
    return weights or None


def challenge_unknown_weights(user_info: dict[str, Any] | None) -> dict[str, float]:
    """Half the opposite pool; the rest stays on the player's home corpus.

    Keyed on type 1, not on ``diabetic`` alone: only type 1 has D1NAMO as its home
    corpus. The two agree for every player who reaches here (``challenge_unknown_active``
    admits only non-diabetic and type 1), but a gestational player -- diabetic, yet
    routed to BIG IDEAs -- would come out inverted under the looser test.
    """
    unknown = CHALLENGE_OPPOSITE_SHARE
    known = 1.0 - unknown
    if is_type_1(user_info):
        return {
            POOL_D1NAMO: known,
            POOL_BIGIDEAS: unknown,
        }
    return {
        POOL_BIGIDEAS: known,
        POOL_D1NAMO: unknown,
    }
