from __future__ import annotations

import pytest

from sugar_sugar.components.landing import how_it_works_children
from sugar_sugar.food_glossary import PHRASES, UNITS, WORDS
from sugar_sugar.food_note_i18n import translate_food_note
from sugar_sugar.i18n import SUPPORTED_LOCALES, setup_i18n, t, t_list


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


def test_english_food_note_stays_source_text() -> None:
    note = "Berry Smoothie (20.0 fluid ounce)\nChicken Leg (1.0)"
    assert translate_food_note(note, "en") == note


def test_romanian_translates_testdata_foods() -> None:
    note = "Berry Smoothie (20.0 fluid ounce)\nChicken Leg (1.0)\nAsparagus (4.0)"
    translated = translate_food_note(note, "ro")
    assert "Smoothie de fructe de pădure (20.0 uncie lichidă)" in translated
    assert "Pulpă de pui (1.0)" in translated
    assert "Sparanghel (4.0)" in translated


def test_unknown_food_is_left_as_written() -> None:
    assert translate_food_note("Mystery Zzxxy (2.0 cup)", "de") == "Mystery Zzxxy (2.0 Tasse)"


def test_romanian_translates_salad_notepad() -> None:
    note = (
        "Cucumber (0.5 cup)\n"
        "Celery (0.5 cup)\n"
        "Salad Greens (2.0 cup)\n"
        "Red Bell Pepper (0.5 cup)\n"
        "Balsamic Vinegar (2.0 tablespoon)"
    )
    translated = translate_food_note(note, "ro")
    assert "Castravete (0.5 cană)" in translated
    assert "Țelină (0.5 cană)" in translated
    assert "Frunze de salată (2.0 cană)" in translated
    assert "Ardei gras roșu (0.5 cană)" in translated
    assert "Oțet balsamic (2.0 lingură)" in translated


def test_compound_line_translates_each_food_word() -> None:
    note = "Salad - Shrimp, Lettuce, Tomato (1.0 cup)"
    translated = translate_food_note(note, "ro")
    assert "Salată" in translated
    assert "Crevete" in translated
    assert "Salată verde" in translated
    assert "Roșie" in translated
    assert "(1.0 cană)" in translated


def test_romanian_translates_extracted_leftover_foods() -> None:
    note = "Roast Beef (1.0)\nTzatziki (2.0 tablespoon)\nMacaroni (1.0 cup)\nStew (1.0 bowl)"
    translated = translate_food_note(note, "ro")
    assert "Roast beef" in translated or "roast beef" in translated.lower()
    assert "Tzatziki" in translated
    assert "Macaroane" in translated
    assert "Tocană" in translated
    assert "lingură" in translated
    assert "bol" in translated


def test_glossary_covers_salad_and_testdata_phrases() -> None:
    for phrase in (
        "berry smoothie",
        "chicken leg",
        "salad greens",
        "red bell pepper",
        "balsamic vinegar",
    ):
        assert phrase in PHRASES
        assert "ro" in PHRASES[phrase]
    for word in ("cucumber", "celery", "asparagus", "tomato", "lettuce", "shrimp"):
        assert word in WORDS
        assert "ro" in WORDS[word]
    for unit in ("cup", "tablespoon", "fluid ounce"):
        assert unit in UNITS
        assert "ro" in UNITS[unit]


def test_how_it_works_keeps_teaser_and_drops_heading() -> None:
    children = how_it_works_children("en")
    texts = [getattr(node, "children", "") for node in children]
    assert "How it works" not in texts
    assert "Three steps. Predict the next hour. Your gut feeling vs the sensor." in texts
    assert children[0].className == "how-it-works-teaser"
    assert children[1].className == "how-it-works-steps"
    assert len(children) == 2
    steps = children[1].children
    assert len(steps) == 3
    assert steps[0].className == "how-it-works-step"
    assert str(steps[0].children).startswith("1. ")


# Teaser must name the forecast ("predict the next hour"), not a one-hour session.
_TEASER_HORIZON_MARKERS: dict[str, str] = {
    "en": "Predict the next hour",
    "uk": "Передбач наступну годину",
    "ru": "Предскажи следующий час",
    "de": "Sag die nächste Stunde voraus",
    "fr": "Prédis la prochaine heure",
    "es": "Predice la próxima hora",
    "ro": "Prezice ora următoare",
    "zh": "预测下一小时",
}


@pytest.mark.parametrize("locale", list(SUPPORTED_LOCALES))
def test_how_it_works_teaser_and_step_three_exist(locale: str) -> None:
    teaser = t("ui.landing.how_it_works_teaser", locale=locale)
    steps = t_list("ui.landing.how_it_works_steps", locale=locale)
    assert teaser
    assert not teaser.startswith("ui.landing.")
    assert _TEASER_HORIZON_MARKERS[locale] in teaser
    assert len(steps) == 3
    assert steps[2]
    assert t("ui.header.non_diabetic", locale=locale)
    assert not t("ui.header.non_diabetic", locale=locale).startswith("ui.header.")
    assert t("ui.startup.challenge_unknown_button", locale=locale)
    help_nd = t("ui.startup.challenge_unknown_help_nd", locale=locale)
    help_t1 = t("ui.startup.challenge_unknown_help_t1", locale=locale)
    assert help_nd and not help_nd.startswith("ui.startup.")
    assert help_t1 and not help_t1.startswith("ui.startup.")
    assert "10%" not in help_nd and "10 %" not in help_nd
    assert "10%" not in help_t1 and "10 %" not in help_t1
    paper_label = t("ui.startup.paper_mention_label", locale=locale)
    paper_hint = t("ui.startup.paper_mention_hint", locale=locale, min_rounds=12)
    assert paper_label and not paper_label.startswith("ui.startup.")
    assert paper_hint and not paper_hint.startswith("ui.startup.")
    assert "12" in paper_hint
    assert t("ui.faq.ask_title", locale=locale)
    assert t("ui.faq.section_participant", locale=locale)
    assert t("ui.faq.section_developer", locale=locale)
    assert t("ui.landing.faq_button", locale=locale)
    assert not t("ui.landing.faq_button", locale=locale).startswith("ui.landing.")
