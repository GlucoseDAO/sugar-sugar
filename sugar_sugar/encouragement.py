"""Encouragement text for the share page.

Hybrid design: templates now, pluggable LLM later.

The public API is one function -- ``encouragement_text(stats, locale)`` --
so callers never have to know whether the text comes from a template or an
LLM.  To swap in a real LLM (e.g. OpenAI's ``gpt-4o-mini``) you replace
``LLM_BACKEND`` at import time with a callable of signature
``(stats, locale) -> str``; the template fallback still runs if the backend
returns None or raises.
"""
from __future__ import annotations

from typing import Callable, Optional

from eliot import start_action

from sugar_sugar.i18n import normalize_locale, t


# Score brackets (mean absolute error in mg/dL).  These match the accuracy
# categories shown on the /final page and are deliberately generous on the
# low end: a MAE < 10 mg/dL is genuinely excellent for a human forecaster.
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


# Optional pluggable backend.  If set, it will be tried first; failures fall
# back to the template.  Keep the type explicit so the swap site is obvious.
LLM_BACKEND: Optional[Callable[[dict, str], Optional[str]]] = None


def encouragement_text(stats: dict, locale: str) -> str:
    """Return a short, locale-appropriate encouraging message.

    ``stats`` is expected to contain at least ``mae_mgdl`` (float) and
    ``rounds_played`` (int).  Other keys are accepted and forwarded to the
    LLM backend untouched.
    """
    loc: str = normalize_locale(locale)
    with start_action(action_type=u"encouragement_text", locale=loc, has_llm=LLM_BACKEND is not None) as action:
        if LLM_BACKEND is not None:
            text: Optional[str] = LLM_BACKEND(stats, loc)
            if text:
                action.log(message_type=u"llm_ok")
                return text.strip()
            action.log(message_type=u"llm_empty_fallback_to_template")

        bracket: str = pick_bracket(stats.get("mae_mgdl"))
        key: str = f"ui.share.encouragement.{bracket}"
        rounds: int = int(stats.get("rounds_played") or 0)
        return t(key, locale=loc, rounds=rounds)
