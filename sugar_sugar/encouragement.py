"""Encouragement tagline selection for the share page.

Hybrid design: a pool of hand-written taglines per score bracket, with a
pluggable LLM hook for later.  Selection is deterministic on a caller-
supplied seed (the share id), so the same share URL always renders the
same tagline -- important because the PNG is cached by share id.

The public API is one function -- ``encouragement_text(stats, locale, *,
seed=None)`` -- so callers don't need to know whether the text came from
a template pool or an LLM.  To swap in a real LLM later, replace
``LLM_BACKEND`` with a callable of signature ``(stats, locale) -> str``.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional

from eliot import start_action

from sugar_sugar.i18n import normalize_locale, t, t_list


# Score brackets (mean absolute error in mg/dL).  A MAE < 10 mg/dL is
# genuinely excellent for a human forecaster; the upper band catches the
# "still learning" case without shaming the player.
BRACKET_THRESHOLDS_MGDL: list[tuple[float, str]] = [
    (10.0, "excellent"),
    (20.0, "good"),
    (35.0, "average"),
    (float("inf"), "keep_practicing"),
]


def pick_bracket(mae_mgdl: Optional[float]) -> str:
    """Map a MAE value (mg/dL) to a bracket key used in translations."""
    if mae_mgdl is None or mae_mgdl != mae_mgdl:  # NaN guard
        return "keep_practicing"
    for threshold, label in BRACKET_THRESHOLDS_MGDL:
        if mae_mgdl < threshold:
            return label
    return "keep_practicing"


# Optional pluggable backend.  If set, it is tried first; failures fall
# back to the template pool.
LLM_BACKEND: Optional[Callable[[dict, str], Optional[str]]] = None


def _stable_idx(seed: str, modulo: int) -> int:
    """Return a deterministic non-negative int in ``[0, modulo)`` for ``seed``."""
    if modulo <= 0:
        return 0
    digest: bytes = hashlib.sha1(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _tagline_pool(bracket: str, locale: str, rounds: int) -> list[str]:
    """Fetch the ``lines:`` array for a bracket, falling back gracefully.

    i18nice's ``t_list`` returns an empty list if the key resolves to a
    scalar or is missing -- in that case we try the legacy scalar key and
    wrap its value so old share records from before the pool landed still
    produce text.
    """
    pool: list[str] = t_list(
        f"ui.share.encouragement.{bracket}.lines",
        locale=locale,
        rounds=rounds,
    )
    if pool:
        return pool

    # Legacy scalar fallback.
    legacy: str = t(
        f"ui.share.encouragement.{bracket}",
        locale=locale,
        rounds=rounds,
    )
    return [legacy] if legacy and not legacy.startswith("ui.share.") else []


def encouragement_text(
    stats: dict[str, Any],
    locale: str,
    *,
    seed: Optional[str] = None,
) -> str:
    """Return a short, locale-appropriate tagline.

    ``stats`` is expected to contain at least ``mae_mgdl`` (float) and
    ``rounds_played`` (int).  ``seed`` controls which tagline is picked
    from the bracket pool; when omitted we fall back to a hash of the
    bracket+rounds so repeated calls still produce the same string.
    """
    loc: str = normalize_locale(locale)
    bracket: str = pick_bracket(stats.get("mae_mgdl"))
    rounds: int = int(stats.get("rounds_played") or 0)

    with start_action(
        action_type=u"encouragement_text",
        locale=loc,
        bracket=bracket,
        has_llm=LLM_BACKEND is not None,
        has_seed=seed is not None,
    ) as action:
        if LLM_BACKEND is not None:
            text: Optional[str] = LLM_BACKEND(stats, loc)
            if text:
                action.log(message_type=u"llm_ok")
                return text.strip()
            action.log(message_type=u"llm_empty_fallback_to_template")

        pool: list[str] = _tagline_pool(bracket, loc, rounds)
        if not pool:
            # Absolute last-ditch fallback -- should never hit in production
            # because the YAMLs always define all four brackets.
            action.log(message_type=u"empty_pool_fallback")
            return ""

        effective_seed: str = seed if seed is not None else f"{bracket}:{rounds}"
        idx: int = _stable_idx(effective_seed, len(pool))
        action.log(message_type=u"pool_pick", idx=idx, pool_size=len(pool))
        return pool[idx]
