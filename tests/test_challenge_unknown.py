from __future__ import annotations

from sugar_sugar.challenge_unknown import (
    CHALLENGE_UNKNOWN_PCT,
    challenge_unknown_active,
    challenge_unknown_checked,
    challenge_unknown_eligible,
    challenge_unknown_visible,
    challenge_unknown_weights,
    encode_mix_policy,
)
from sugar_sugar.subject_sources import (
    GENERIC_INTERVENTION_BIGIDEAS,
    GENERIC_INTERVENTION_D1NAMO,
    GENERIC_INTERVENTION_MIX_T2,
    generic_intervention_for_user,
    intervention_pool_weights,
)


def test_challenge_for_any_diabetes_answer_except_own_data() -> None:
    assert challenge_unknown_eligible({"diabetic": False, "format": "A"})
    assert challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Type 1", "format": "C"})
    assert challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Type 2", "format": "A"})
    assert challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Prediabetes", "format": "A"})
    assert challenge_unknown_eligible({"diabetic": True, "diabetic_type": "LADA", "format": "C"})
    assert challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Gestational", "format": "A"})
    assert challenge_unknown_eligible({"format": "A"})
    assert not challenge_unknown_eligible({"diabetic": False, "format": "B"})
    assert not challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Type 1", "format": "B"})


def test_challenge_shows_for_every_answer_except_own_data() -> None:
    assert challenge_unknown_visible({})
    assert challenge_unknown_visible({"diabetic": False})
    assert challenge_unknown_visible({"diabetic": False, "format": "A"})
    assert challenge_unknown_visible({"diabetic": False, "format": "C"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 1"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 2"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "Prediabetes", "format": "A"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "LADA", "format": "C"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "Gestational", "format": "A"})
    assert not challenge_unknown_visible({"diabetic": False, "format": "B"})
    assert not challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 1", "format": "B"})
    assert not challenge_unknown_eligible({"diabetic": False})


def test_challenge_checkbox_reads_checklist_or_bool() -> None:
    assert challenge_unknown_checked(["on"])
    assert challenge_unknown_checked(True)
    assert not challenge_unknown_checked([])
    assert not challenge_unknown_checked(False)
    assert not challenge_unknown_checked(None)


def test_non_diabetic_challenge_is_half_d1namo() -> None:
    info = {
        "diabetic": False,
        "format": "A",
        "challenge_unknown": True,
    }
    assert challenge_unknown_active(info)
    assert CHALLENGE_UNKNOWN_PCT == 50
    assert challenge_unknown_weights(info) == {"bigideas": 0.5, "d1namo": 0.5}
    assert generic_intervention_for_user(info) == "mix:bigideas=0.50,d1namo=0.50"
    assert intervention_pool_weights(generic_intervention_for_user(info)) == {
        "bigideas": 0.5,
        "d1namo": 0.5,
    }


def test_any_diabetic_challenge_is_half_bigideas() -> None:
    for kind in ("Type 1", "Type 2", "Prediabetes", "LADA", "Gestational"):
        info = {
            "diabetic": True,
            "diabetic_type": kind,
            "format": "A",
            "challenge_unknown": True,
        }
        assert generic_intervention_for_user(info) == encode_mix_policy(
            {"d1namo": 0.5, "bigideas": 0.5}
        )
        assert intervention_pool_weights(generic_intervention_for_user(info)) == {
            "d1namo": 0.5,
            "bigideas": 0.5,
        }


def test_legacy_pct_is_ignored_when_challenge_is_on() -> None:
    info = {
        "diabetic": False,
        "format": "A",
        "challenge_unknown": True,
        "challenge_unknown_pct": 10,
    }
    assert challenge_unknown_weights(info) == {"bigideas": 0.5, "d1namo": 0.5}


def test_default_routing_unchanged_when_challenge_off() -> None:
    assert generic_intervention_for_user({"diabetic": False}) == GENERIC_INTERVENTION_BIGIDEAS
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 1"}) == (
        GENERIC_INTERVENTION_D1NAMO
    )
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 2"}) == (
        GENERIC_INTERVENTION_MIX_T2
    )
