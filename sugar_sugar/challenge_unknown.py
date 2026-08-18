"""Challenge the unknown: mix the opposite Format A pool on purpose.

Default Format A routing is fixed by diabetes status. Any player on Public or
Public + My Data can opt in: half the rounds then come from the other corpus
(D1NAMO vs BIG IDEAs). That is the hard direction — diabetic traces if the
player is not diabetic, non-diabetic traces if they are. Format B (own data)
never uses this.
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


def is_diabetic(user_info: dict[str, Any] | None) -> bool:
    return (user_info or {}).get("diabetic") is True


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


def challenge_unknown_visible(
    user_info: dict[str, Any] | None = None,
    format_value: Optional[str] = None,
) -> bool:
    """Show for every diabetes answer except Format B (own data)."""
    return _format_code(user_info, format_value) != "B"


def challenge_unknown_eligible(
    user_info: dict[str, Any] | None = None,
    format_value: Optional[str] = None,
) -> bool:
    """True for generic / generic+own play, any diabetes status."""
    return _format_code(user_info, format_value) in CHALLENGE_FORMATS


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
    """Half the opposite pool; the rest stays on the player's home corpus."""
    unknown = CHALLENGE_OPPOSITE_SHARE
    known = 1.0 - unknown
    if is_diabetic(user_info):
        return {
            POOL_D1NAMO: known,
            POOL_BIGIDEAS: unknown,
        }
    return {
        POOL_BIGIDEAS: known,
        POOL_D1NAMO: unknown,
    }
