"""Localized study-design markdown and consent paragraphs exist for every locale.

Missing study-design files make /about show English. Missing consent dict
entries fall through to the German markdown — both look like a broken
translation, not a fallback.
"""
from __future__ import annotations

from pathlib import Path

from sugar_sugar.consent_notice_text import (
    _CONSENT_NOTICE_TRANSLATIONS,
    consent_notice_paragraphs,
)
from sugar_sugar.i18n import SUPPORTED_LOCALES

_STUDY_DIR = Path(__file__).parent.parent / "data" / "input" / "study_design"
_STUDY_EN = _STUDY_DIR / "The study - technical Guidebook.md"


def _study_design_path(locale: str) -> Path:
    if locale == "en":
        return _STUDY_EN
    dotted = _STUDY_EN.with_name(f"{_STUDY_EN.stem}.{locale}{_STUDY_EN.suffix}")
    if dotted.exists():
        return dotted
    return _STUDY_EN.with_name(f"{_STUDY_EN.stem}_{locale}{_STUDY_EN.suffix}")


def test_study_design_markdown_exists_for_every_locale() -> None:
    assert _STUDY_EN.exists()
    missing = sorted(
        loc
        for loc in SUPPORTED_LOCALES
        if loc != "en" and not _study_design_path(loc).exists()
    )
    assert not missing, f"missing study-design markdown for {missing}"


def test_study_design_locale_files_are_not_english_copies() -> None:
    english = _STUDY_EN.read_text(encoding="utf-8").strip()
    copied = [
        loc
        for loc in SUPPORTED_LOCALES
        if loc != "en"
        and _study_design_path(loc).exists()
        and _study_design_path(loc).read_text(encoding="utf-8").strip() == english
    ]
    assert not copied, f"study-design markdown is still English for {copied}"


def test_consent_translations_cover_every_non_german_locale() -> None:
    english = _CONSENT_NOTICE_TRANSLATIONS["en"]
    german = consent_notice_paragraphs("de")
    missing: list[str] = []
    fell_back: list[str] = []
    for loc in sorted(SUPPORTED_LOCALES):
        if loc == "de":
            continue
        if loc != "en" and loc not in _CONSENT_NOTICE_TRANSLATIONS:
            missing.append(loc)
            continue
        if consent_notice_paragraphs(loc) == german:
            fell_back.append(loc)
    assert not missing, f"consent dict missing {missing}"
    assert not fell_back, f"consent fell back to German for {fell_back}"
    for loc in ("bg", "ja", "ko"):
        assert len(_CONSENT_NOTICE_TRANSLATIONS[loc]) == len(english), loc
