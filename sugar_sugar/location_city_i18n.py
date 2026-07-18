"""Localized city-name overrides keyed by (English city, English country)."""

from __future__ import annotations

from typing import Final, TypedDict


class CityI18nOverride(TypedDict, total=False):
    locales: dict[str, str]
    extra_search: tuple[str, ...]


# Keys are (city English name, country English name).
CITY_I18N: Final[dict[tuple[str, str], CityI18nOverride]] = {
    ("Athens", "Greece"): {"extra_search": ("Αθήνα",)},
    ("Beijing", "China"): {"locales": {"de": "Peking", "zh": "北京"}},
    ("Belgrade", "Serbia"): {"extra_search": ("Beograd",)},
    ("Bogota", "Colombia"): {"locales": {"es": "Bogotá"}},
    ("Brasilia", "Brazil"): {"extra_search": ("Brasília",)},
    ("Brussels", "Belgium"): {"locales": {"de": "Brüssel", "fr": "Bruxelles"}, "extra_search": ("Brussel",)},
    ("Bucharest", "Romania"): {"locales": {"ro": "București"}},
    ("Cologne", "Germany"): {"locales": {"de": "Köln", "fr": "Cologne"}, "extra_search": ("Koln", "Koeln")},
    ("Copenhagen", "Denmark"): {"extra_search": ("København", "Kobenhavn")},
    ("Dusseldorf", "Germany"): {"locales": {"de": "Düsseldorf"}, "extra_search": ("Duesseldorf",)},
    ("Florence", "Italy"): {"locales": {"de": "Florenz", "es": "Florencia", "fr": "Florence", "it": "Firenze"}},
    ("Frankfurt", "Germany"): {"locales": {"de": "Frankfurt am Main"}},
    ("Geneva", "Switzerland"): {"locales": {"de": "Genf", "fr": "Genève"}},
    ("Guangzhou", "China"): {"locales": {"zh": "广州"}},
    ("Hanoi", "Vietnam"): {"locales": {"fr": "Hanoï"}, "extra_search": ("Hà Nội",)},
    ("Ho Chi Minh City", "Vietnam"): {
        "locales": {"fr": "Hô-Chi-Minh-Ville", "zh": "胡志明市"},
        "extra_search": ("Thành phố Hồ Chí Minh",),
    },
    ("Hong Kong", "China"): {"locales": {"zh": "香港"}},
    ("Istanbul", "Turkey"): {"extra_search": ("İstanbul",)},
    ("Kyiv", "Ukraine"): {"locales": {"ru": "Киев", "uk": "Київ"}, "extra_search": ("Kiev",)},
    ("Lisbon", "Portugal"): {"extra_search": ("Lisboa",)},
    ("Los Angeles", "United States"): {"locales": {"es": "Los Ángeles"}},
    ("Milan", "Italy"): {"locales": {"de": "Mailand", "es": "Milán", "fr": "Milan"}, "extra_search": ("Milano",)},
    ("Mexico City", "Mexico"): {"locales": {"es": "Ciudad de México"}},
    ("Montreal", "Canada"): {"locales": {"fr": "Montréal"}},
    ("Moscow", "Russia"): {"locales": {"ru": "Москва"}},
    ("Munich", "Germany"): {"locales": {"de": "München"}, "extra_search": ("Muenchen",)},
    ("Naples", "Italy"): {"locales": {"de": "Neapel", "es": "Nápoles", "fr": "Naples"}, "extra_search": ("Napoli",)},
    ("New York", "United States"): {"locales": {"es": "Nueva York"}},
    ("Osaka", "Japan"): {"locales": {"zh": "大阪"}, "extra_search": ("大阪",)},
    ("Prague", "Czech Republic"): {"locales": {"de": "Prag", "es": "Praga", "fr": "Prague"}, "extra_search": ("Praha",)},
    ("Prague", "Czechia"): {"locales": {"de": "Prag", "es": "Praga", "fr": "Prague"}, "extra_search": ("Praha",)},
    ("Reykjavik", "Iceland"): {"extra_search": ("Reykjavík",)},
    ("Rio de Janeiro", "Brazil"): {"locales": {"es": "Río de Janeiro"}},
    ("Rome", "Italy"): {"locales": {"de": "Rom", "es": "Roma", "fr": "Rome"}, "extra_search": ("Roma",)},
    ("Saint Petersburg", "Russia"): {"locales": {"ru": "Санкт-Петербург"}},
    ("Sao Paulo", "Brazil"): {"extra_search": ("São Paulo",)},
    ("Seoul", "South Korea"): {"locales": {"zh": "首尔"}, "extra_search": ("서울",)},
    ("Shanghai", "China"): {"locales": {"zh": "上海"}},
    ("Shenzhen", "China"): {"locales": {"zh": "深圳"}},
    ("Taipei", "Taiwan"): {"locales": {"zh": "台北"}},
    ("The Hague", "Netherlands"): {"locales": {"de": "Den Haag", "fr": "La Haye", "nl": "Den Haag"}},
    ("Tokyo", "Japan"): {"locales": {"zh": "东京"}, "extra_search": ("東京",)},
    ("Vienna", "Austria"): {"locales": {"de": "Wien", "es": "Viena", "fr": "Vienne"}},
    ("Warsaw", "Poland"): {"locales": {"de": "Warschau", "es": "Varsovia", "fr": "Varsovie"}, "extra_search": ("Warszawa",)},
    ("Zurich", "Switzerland"): {"locales": {"de": "Zürich", "fr": "Zurich"}},
    ("Delhi", "India"): {"locales": {"hi": "दिल्ली"}},
    ("Mumbai", "India"): {"locales": {"hi": "मुंबई"}, "extra_search": ("Bombay",)},
    ("Chennai", "India"): {"extra_search": ("Madras",)},
    ("Kolkata", "India"): {"extra_search": ("Calcutta",)},
}
