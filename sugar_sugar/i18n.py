from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Iterable, Optional

import i18n

DEFAULT_LOCALE: Final[str] = "en"
SUPPORTED_LOCALES: Final[set[str]] = {"en", "de", "uk"}


def normalize_locale(locale: Optional[str]) -> str:
    if locale in SUPPORTED_LOCALES:
        return locale
    return DEFAULT_LOCALE


@lru_cache(maxsize=1)
def translations_dir() -> Path:
    return Path(__file__).resolve().parent / "translations"


def setup_i18n(*, extra_load_paths: Optional[Iterable[Path]] = None) -> None:
    """
    Configure i18nice and load translations once.

    We keep English as the default locale and fallback.
    """
    i18n.set("locale", DEFAULT_LOCALE)
    i18n.set("fallback", DEFAULT_LOCALE)

    # Default file format is YAML if PyYAML is installed (it is via i18nice[yaml]).
    base = str(translations_dir())
    if base not in i18n.load_path:
        i18n.load_path.append(base)
    for p in (extra_load_paths or []):
        sp = str(Path(p))
        if sp not in i18n.load_path:
            i18n.load_path.append(sp)

    # Load and cache all translations upfront.
    # Using reload ensures changes to translation files are picked up on process start.
    i18n.reload_everything()


def t(key: str, *, locale: Optional[str] = None, **kwargs: object) -> str:
    return str(i18n.t(key, locale=normalize_locale(locale), **kwargs))


def t_list(key: str, *, locale: Optional[str] = None, **kwargs: object) -> list[str]:
    """
    Return a fully-evaluated list of translated strings.

    i18nice returns a lazy tuple for lists; `[:]` forces evaluation.
    """
    value = i18n.t(key, locale=normalize_locale(locale), _list=True, **kwargs)
    try:
        evaluated = value[:]
    except Exception:
        evaluated = value
    return [str(x) for x in evaluated]

