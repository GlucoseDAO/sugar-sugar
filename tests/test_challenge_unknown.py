from __future__ import annotations

from sugar_sugar.challenge_unknown import (
    challenge_unknown_active,
    challenge_unknown_checked,
    challenge_unknown_eligible,
    challenge_unknown_visible,
    challenge_unknown_weights,
    encode_mix_policy,
    snap_challenge_pct,
)
from sugar_sugar.subject_sources import (
    GENERIC_INTERVENTION_BIGIDEAS,
    GENERIC_INTERVENTION_D1NAMO,
    GENERIC_INTERVENTION_MIX_T2,
    generic_intervention_for_user,
    intervention_pool_weights,
)


def test_snap_challenge_pct_uses_ten_percent_steps() -> None:
    assert snap_challenge_pct(None) == 10
    assert snap_challenge_pct(14) == 10
    assert snap_challenge_pct(16) == 20
    assert snap_challenge_pct(100) == 100
    assert snap_challenge_pct(3) == 10


def test_challenge_only_for_generic_formats_and_single_pool_types() -> None:
    assert challenge_unknown_eligible({"diabetic": False, "format": "A"})
    assert challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Type 1", "format": "C"})
    assert not challenge_unknown_eligible({"diabetic": False, "format": "B"})
    assert not challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Type 2", "format": "A"})
    assert not challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Prediabetes", "format": "A"})
    assert not challenge_unknown_eligible({"diabetic": True, "diabetic_type": "LADA", "format": "C"})
    assert not challenge_unknown_eligible({"diabetic": True, "diabetic_type": "Gestational", "format": "A"})
    assert not challenge_unknown_eligible({"format": "A"})


def test_challenge_shows_only_for_non_diabetic_or_type_1_except_own_data() -> None:
    assert not challenge_unknown_visible({})
    assert challenge_unknown_visible({"diabetic": False})
    assert challenge_unknown_visible({"diabetic": False, "format": "A"})
    assert challenge_unknown_visible({"diabetic": False, "format": "C"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 1"})
    assert challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 1", "format": "C"})
    assert not challenge_unknown_visible({"diabetic": False, "format": "B"})
    assert not challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 1", "format": "B"})
    assert not challenge_unknown_visible({"diabetic": True, "diabetic_type": "Type 2"})
    assert not challenge_unknown_visible({"diabetic": True, "diabetic_type": "Prediabetes", "format": "A"})
    assert not challenge_unknown_eligible({"diabetic": False})


def test_challenge_checkbox_reads_checklist_or_bool() -> None:
    assert challenge_unknown_checked(["on"])
    assert challenge_unknown_checked(True)
    assert not challenge_unknown_checked([])
    assert not challenge_unknown_checked(False)
    assert not challenge_unknown_checked(None)


def test_non_diabetic_challenge_mixes_in_d1namo() -> None:
    info = {
        "diabetic": False,
        "format": "A",
        "challenge_unknown": True,
        "challenge_unknown_pct": 10,
    }
    assert challenge_unknown_active(info)
    assert challenge_unknown_weights(info) == {"bigideas": 0.9, "d1namo": 0.1}
    assert generic_intervention_for_user(info) == "mix:bigideas=0.90,d1namo=0.10"
    assert intervention_pool_weights(generic_intervention_for_user(info)) == {
        "bigideas": 0.9,
        "d1namo": 0.1,
    }


def test_type_1_challenge_mixes_in_bigideas() -> None:
    info = {
        "diabetic": True,
        "diabetic_type": "Type 1",
        "format": "C",
        "challenge_unknown": True,
        "challenge_unknown_pct": 100,
    }
    assert generic_intervention_for_user(info) == encode_mix_policy({"d1namo": 0.0, "bigideas": 1.0})
    assert intervention_pool_weights(generic_intervention_for_user(info)) == {"bigideas": 1.0}


def test_default_routing_unchanged_when_challenge_off() -> None:
    assert generic_intervention_for_user({"diabetic": False}) == GENERIC_INTERVENTION_BIGIDEAS
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 1"}) == (
        GENERIC_INTERVENTION_D1NAMO
    )
    assert generic_intervention_for_user({"diabetic": True, "diabetic_type": "Type 2"}) == (
        GENERIC_INTERVENTION_MIX_T2
    )
