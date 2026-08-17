"""Challenge the unknown: mix the opposite Format A pool on purpose.

Default Format A routing is fixed by diabetes status. Non-diabetic and type 1
players can opt into a slider that injects the other corpus (D1NAMO vs BIG
IDEAs) in 10% steps. Type 2 / prediabetes / LADA already have a mixture, so
the control does not apply. Gestational stays on BIG IDEAs only.
"""
from __future__ import annotations

from typing import Any, Optional

POOL_BIGIDEAS: str = "bigideas"
POOL_D1NAMO: str = "d1namo"
CHALLENGE_FORMATS: frozenset[str] = frozenset({"A", "C"})
CHALLENGE_PCT_MIN: int = 10
CHALLENGE_PCT_MAX: int = 100
CHALLENGE_PCT_STEP: int = 10
DEFAULT_CHALLENGE_PCT: int = 10
MIX_POLICY_PREFIX: str = "mix:"


def _diabetes_kind(user_info: dict[str, Any] | None) -> str:
    info = user_info or {}
    if info.get("diabetic") is not True:
        return "none"
    return str(info.get("diabetic_type") or "").strip().lower()


def is_type_1(user_info: dict[str, Any] | None) -> bool:
    return _diabetes_kind(user_info) in {"type 1", "type1", "t1"}


def challenge_unknown_diabetes_eligible(user_info: dict[str, Any] | None) -> bool:
    """True after the player picks no diabetes, or type 1."""
    info = user_info or {}
    if info.get("diabetic") is False:
        return True
    return is_type_1(info)


def challenge_unknown_checked(raw: Any) -> bool:
    """True when the startup checklist (or a leftover bool store) is opted in."""
    if raw is True:
        return True
    if isinstance(raw, (list, tuple, set)):
        return any(item in {"on", True, "true", 1} for item in raw)
    return False


def challenge_unknown_visible(
    user_info: dict[str, Any] | None = None,
    format_value: Optional[str] = None,
) -> bool:
    """Show after a no-diabetes or type-1 answer, unless Format B (own data)."""
    info = user_info or {}
    fmt = str(format_value if format_value is not None else info.get("format") or "").strip().upper()
    if fmt == "B":
        return False
    return challenge_unknown_diabetes_eligible(info)


def challenge_unknown_eligible(
    user_info: dict[str, Any] | None = None,
    format_value: Optional[str] = None,
) -> bool:
    """True for generic / generic+own play on a single-pool diabetes category."""
    info = user_info or {}
    fmt = str(format_value if format_value is not None else info.get("format") or "").strip().upper()
    return challenge_unknown_diabetes_eligible(info) and fmt in CHALLENGE_FORMATS


def snap_challenge_pct(raw: Any) -> int:
    try:
        value = int(round(float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_CHALLENGE_PCT
    stepped = int(round(value / CHALLENGE_PCT_STEP) * CHALLENGE_PCT_STEP)
    return max(CHALLENGE_PCT_MIN, min(CHALLENGE_PCT_MAX, stepped))


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
    """Percent of the *unknown* pool. The rest stays on the player's default corpus."""
    pct = snap_challenge_pct((user_info or {}).get("challenge_unknown_pct"))
    unknown = pct / 100.0
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
