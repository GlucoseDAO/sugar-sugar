"""Localized country and city names for startup location autocomplete."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

from sugar_sugar.i18n import SUPPORTED_LOCALES
from sugar_sugar.location_city_i18n import CITY_I18N
from sugar_sugar.location_countries import COUNTRY_NAMES

LOCALES: Final[tuple[str, ...]] = tuple(sorted(SUPPORTED_LOCALES))
_DATA_DIR = Path(__file__).resolve().parent / "data"

# English canonical country name -> localized labels (en is always the key itself).
COUNTRY_I18N: Final[dict[str, dict[str, str]]] = {
    "Afghanistan": {"de": "Afghanistan", "fr": "Afghanistan", "es": "Afganistán", "ru": "Афганистан", "uk": "Афганістан", "ro": "Afganistan", "zh": "阿富汗"},
    "Albania": {"de": "Albanien", "fr": "Albanie", "es": "Albania", "ru": "Албания", "uk": "Албанія", "ro": "Albania", "zh": "阿尔巴尼亚"},
    "Algeria": {"de": "Algerien", "fr": "Algérie", "es": "Argelia", "ru": "Алжир", "uk": "Алжир", "ro": "Algeria", "zh": "阿尔及利亚"},
    "Argentina": {"de": "Argentinien", "fr": "Argentine", "es": "Argentina", "ru": "Аргентина", "uk": "Аргентина", "ro": "Argentina", "zh": "阿根廷"},
    "Australia": {"de": "Australien", "fr": "Australie", "es": "Australia", "ru": "Австралия", "uk": "Австралія", "ro": "Australia", "zh": "澳大利亚"},
    "Austria": {"de": "Österreich", "fr": "Autriche", "es": "Austria", "ru": "Австрия", "uk": "Австрія", "ro": "Austria", "zh": "奥地利"},
    "Azerbaijan": {"de": "Aserbaidschan", "fr": "Azerbaïdjan", "es": "Azerbaiyán", "ru": "Азербайджан", "uk": "Аzerbaidzhan", "ro": "Azerbaidjan", "zh": "阿塞拜疆"},
    "Bangladesh": {"de": "Bangladesch", "fr": "Bangladesh", "es": "Bangladés", "ru": "Бангладеш", "uk": "Бангladesh", "ro": "Bangladesh", "zh": "孟加拉国"},
    "Belarus": {"de": "Belarus", "fr": "Biélorussie", "es": "Bielorrusia", "ru": "Беларусь", "uk": "Білорусь", "ro": "Belarus", "zh": "白俄罗斯"},
    "Belgium": {"de": "Belgien", "fr": "Belgique", "es": "Bélgica", "ru": "Бельгия", "uk": "Бельгія", "ro": "Belgia", "zh": "比利时"},
    "Bosnia and Herzegovina": {"de": "Bosnien und Herzegowina", "fr": "Bosnie-Herzégovine", "es": "Bosnia y Herzegovina", "ru": "Босния и Герцеговина", "uk": "Боснія і Герцеговина", "ro": "Bosnia și Herțegovina", "zh": "波斯尼亚和黑塞哥维那"},
    "Brazil": {"de": "Brasilien", "fr": "Brésil", "es": "Brasil", "ru": "Бразилия", "uk": "Бразилія", "ro": "Brazilia", "zh": "巴西"},
    "Bulgaria": {"de": "Bulgarien", "fr": "Bulgarie", "es": "Bulgaria", "ru": "Болгария", "uk": "Болгарія", "ro": "Bulgaria", "zh": "保加利亚"},
    "Canada": {"de": "Kanada", "fr": "Canada", "es": "Canadá", "ru": "Канада", "uk": "Канада", "ro": "Canada", "zh": "加拿大"},
    "Chile": {"de": "Chile", "fr": "Chili", "es": "Chile", "ru": "Чили", "uk": "Чилі", "ro": "Chile", "zh": "智利"},
    "China": {"de": "China", "fr": "Chine", "es": "China", "ru": "Китай", "uk": "Китай", "ro": "China", "zh": "中国"},
    "Colombia": {"de": "Kolumbien", "fr": "Colombie", "es": "Colombia", "ru": "Колумбия", "uk": "Колумбія", "ro": "Columbia", "zh": "哥伦比亚"},
    "Congo": {"de": "Kongo", "fr": "Congo", "es": "Congo", "ru": "Конго", "uk": "Кongo", "ro": "Congo", "zh": "刚果"},
    "Croatia": {"de": "Kroatien", "fr": "Croatie", "es": "Croacia", "ru": "Хорватия", "uk": "Хорватія", "ro": "Croația", "zh": "克罗地亚"},
    "Cyprus": {"de": "Zypern", "fr": "Chypre", "es": "Chipre", "ru": "Кипр", "uk": "Кіпр", "ro": "Cipru", "zh": "塞浦路斯"},
    "Czech Republic": {"de": "Tschechien", "fr": "Tchéquie", "es": "República Checa", "ru": "Чехия", "uk": "Чехія", "ro": "Cehia", "zh": "捷克"},
    "Czechia": {"de": "Tschechien", "fr": "Tchéquie", "es": "Chequia", "ru": "Чехия", "uk": "Чехія", "ro": "Cehia", "zh": "捷克"},
    "Denmark": {"de": "Dänemark", "fr": "Danemark", "es": "Dinamarca", "ru": "Дания", "uk": "Данія", "ro": "Danemarca", "zh": "丹麦"},
    "Ecuador": {"de": "Ecuador", "fr": "Équateur", "es": "Ecuador", "ru": "Эквадор", "uk": "Еcuador", "ro": "Ecuador", "zh": "厄瓜多尔"},
    "Egypt": {"de": "Ägypten", "fr": "Égypte", "es": "Egipto", "ru": "Египет", "uk": "Єгипет", "ro": "Egipt", "zh": "埃及"},
    "Estonia": {"de": "Estland", "fr": "Estonie", "es": "Estonia", "ru": "Эстония", "uk": "Естонія", "ro": "Estonia", "zh": "爱沙尼亚"},
    "Ethiopia": {"de": "Äthiopien", "fr": "Éthiopie", "es": "Etiopía", "ru": "Эфиопия", "uk": "Ефіопія", "ro": "Etiopia", "zh": "埃塞俄比亚"},
    "Finland": {"de": "Finnland", "fr": "Finlande", "es": "Finlandia", "ru": "Финляндия", "uk": "Фінляндія", "ro": "Finlanda", "zh": "芬兰"},
    "France": {"de": "Frankreich", "fr": "France", "es": "Francia", "ru": "Франция", "uk": "Франція", "ro": "Franța", "zh": "法国"},
    "Germany": {"de": "Deutschland", "fr": "Allemagne", "es": "Alemania", "ru": "Германия", "uk": "Німеччина", "ro": "Germania", "zh": "德国"},
    "Ghana": {"de": "Ghana", "fr": "Ghana", "es": "Ghana", "ru": "Гана", "uk": "Гана", "ro": "Ghana", "zh": "加纳"},
    "Greece": {"de": "Griechenland", "fr": "Grèce", "es": "Grecia", "ru": "Греция", "uk": "Греція", "ro": "Grecia", "zh": "希腊"},
    "Hungary": {"de": "Ungarn", "fr": "Hongrie", "es": "Hungría", "ru": "Венгрия", "uk": "Угорщина", "ro": "Ungaria", "zh": "匈牙利"},
    "Iceland": {"de": "Island", "fr": "Islande", "es": "Islandia", "ru": "Исландия", "uk": "Ісландія", "ro": "Islanda", "zh": "冰岛"},
    "India": {"de": "Indien", "fr": "Inde", "es": "India", "ru": "Индия", "uk": "Індія", "ro": "India", "zh": "印度"},
    "Indonesia": {"de": "Indonesien", "fr": "Indonésie", "es": "Indonesia", "ru": "Индонезия", "uk": "Індонезія", "ro": "Indonezia", "zh": "印度尼西亚"},
    "Iran": {"de": "Iran", "fr": "Iran", "es": "Irán", "ru": "Иран", "uk": "Іран", "ro": "Iran", "zh": "伊朗"},
    "Iraq": {"de": "Irak", "fr": "Irak", "es": "Irak", "ru": "Ирак", "uk": "Ірак", "ro": "Irak", "zh": "伊拉克"},
    "Ireland": {"de": "Irland", "fr": "Irlande", "es": "Irlanda", "ru": "Ирландия", "uk": "Ірландія", "ro": "Irlanda", "zh": "爱尔兰"},
    "Israel": {"de": "Israel", "fr": "Israël", "es": "Israel", "ru": "Израиль", "uk": "Ізраїль", "ro": "Israel", "zh": "以色列"},
    "Italy": {"de": "Italien", "fr": "Italie", "es": "Italia", "ru": "Италия", "uk": "Італія", "ro": "Italia", "zh": "意大利"},
    "Ivory Coast": {"de": "Elfenbeinküste", "fr": "Côte d'Ivoire", "es": "Costa de Marfil", "ru": "Кот-д'Ивуар", "uk": "Кот-д'Івuar", "ro": "Coasta de Fildeș", "zh": "科特迪瓦"},
    "Japan": {"de": "Japan", "fr": "Japon", "es": "Japón", "ru": "Япония", "uk": "Японія", "ro": "Japonia", "zh": "日本"},
    "Jordan": {"de": "Jordanien", "fr": "Jordanie", "es": "Jordania", "ru": "Иордания", "uk": "Йорданія", "ro": "Iordania", "zh": "约旦"},
    "Kenya": {"de": "Kenia", "fr": "Kenya", "es": "Kenia", "ru": "Кения", "uk": "Кенія", "ro": "Kenya", "zh": "肯尼亚"},
    "Kosovo": {"de": "Kosovo", "fr": "Kosovo", "es": "Kosovo", "ru": "Косово", "uk": "Косово", "ro": "Kosovo", "zh": "科索沃"},
    "Kuwait": {"de": "Kuwait", "fr": "Koweït", "es": "Kuwait", "ru": "Кувейт", "uk": "Кувейт", "ro": "Kuweit", "zh": "科威特"},
    "Latvia": {"de": "Lettland", "fr": "Lettonie", "es": "Letonia", "ru": "Латвия", "uk": "Латвія", "ro": "Letonia", "zh": "拉脱维亚"},
    "Lebanon": {"de": "Libanon", "fr": "Liban", "es": "Líbano", "ru": "Ливан", "uk": "Ліван", "ro": "Liban", "zh": "黎巴嫩"},
    "Libya": {"de": "Libyen", "fr": "Libye", "es": "Libia", "ru": "Ливия", "uk": "Лівія", "ro": "Libia", "zh": "利比亚"},
    "Lithuania": {"de": "Litauen", "fr": "Lituanie", "es": "Lituania", "ru": "Литва", "uk": "Литва", "ro": "Lituania", "zh": "立陶宛"},
    "Luxembourg": {"de": "Luxemburg", "fr": "Luxembourg", "es": "Luxemburgo", "ru": "Люксембург", "uk": "Люксембург", "ro": "Luxemburg", "zh": "卢森堡"},
    "Malaysia": {"de": "Malaysia", "fr": "Malaisie", "es": "Malasia", "ru": "Малайзия", "uk": "Малайзія", "ro": "Malaysia", "zh": "马来西亚"},
    "Malta": {"de": "Malta", "fr": "Malte", "es": "Malta", "ru": "Мальта", "uk": "Мальта", "ro": "Malta", "zh": "马耳他"},
    "Mexico": {"de": "Mexiko", "fr": "Mexique", "es": "México", "ru": "Мексика", "uk": "Мексика", "ro": "Mexic", "zh": "墨西哥"},
    "Moldova": {"de": "Republik Moldau", "fr": "Moldavie", "es": "Moldavia", "ru": "Молдова", "uk": "Мoldova", "ro": "Moldova", "zh": "摩尔多瓦"},
    "Montenegro": {"de": "Montenegro", "fr": "Monténégro", "es": "Montenegro", "ru": "Черногория", "uk": "Чорногорія", "ro": "Muntenegru", "zh": "黑山"},
    "Morocco": {"de": "Marokko", "fr": "Maroc", "es": "Marruecos", "ru": "Марокко", "uk": "Марокко", "ro": "Maroc", "zh": "摩洛哥"},
    "Netherlands": {"de": "Niederlande", "fr": "Pays-Bas", "es": "Países Bajos", "ru": "Нидерланды", "uk": "Нідерланди", "ro": "Olanda", "zh": "荷兰"},
    "New Zealand": {"de": "Neuseeland", "fr": "Nouvelle-Zélande", "es": "Nueva Zelanda", "ru": "Новая Зеландия", "uk": "Нова Зеландія", "ro": "Noua Zeelandă", "zh": "新西兰"},
    "Nigeria": {"de": "Nigeria", "fr": "Nigeria", "es": "Nigeria", "ru": "Нигерия", "uk": "Нігерія", "ro": "Nigeria", "zh": "尼日利亚"},
    "North Korea": {"de": "Nordkorea", "fr": "Corée du Nord", "es": "Corea del Norte", "ru": "КНДР", "uk": "Північна Корея", "ro": "Coreea de Nord", "zh": "朝鲜"},
    "North Macedonia": {"de": "Nordmazedonien", "fr": "Macédoine du Nord", "es": "Macedonia del Norte", "ru": "Северная Македония", "uk": "Північна Македонія", "ro": "Macedonia de Nord", "zh": "北马其顿"},
    "Norway": {"de": "Norwegen", "fr": "Norvège", "es": "Noruega", "ru": "Норвегия", "uk": "Норвегія", "ro": "Norvegia", "zh": "挪威"},
    "Pakistan": {"de": "Pakistan", "fr": "Pakistan", "es": "Pakistán", "ru": "Пакистан", "uk": "Пакистан", "ro": "Pakistan", "zh": "巴基斯坦"},
    "Palestine": {"de": "Palästina", "fr": "Palestine", "es": "Palestina", "ru": "Палестина", "uk": "Палестина", "ro": "Palestina", "zh": "巴勒斯坦"},
    "Peru": {"de": "Peru", "fr": "Pérou", "es": "Perú", "ru": "Перу", "uk": "Перу", "ro": "Peru", "zh": "秘鲁"},
    "Philippines": {"de": "Philippinen", "fr": "Philippines", "es": "Filipinas", "ru": "Филиппины", "uk": "Філіппіни", "ro": "Filipine", "zh": "菲律宾"},
    "Poland": {"de": "Polen", "fr": "Pologne", "es": "Polonia", "ru": "Польша", "uk": "Польща", "ro": "Polonia", "zh": "波兰"},
    "Portugal": {"de": "Portugal", "fr": "Portugal", "es": "Portugal", "ru": "Португалия", "uk": "Португалія", "ro": "Portugalia", "zh": "葡萄牙"},
    "Qatar": {"de": "Katar", "fr": "Qatar", "es": "Catar", "ru": "Катар", "uk": "Кatar", "ro": "Qatar", "zh": "卡塔尔"},
    "Romania": {"de": "Rumänien", "fr": "Roumanie", "es": "Rumania", "ru": "Румыния", "uk": "Румунія", "ro": "România", "zh": "罗马尼亚"},
    "Russia": {"de": "Russland", "fr": "Russie", "es": "Rusia", "ru": "Россия", "uk": "Росія", "ro": "Rusia", "zh": "俄罗斯"},
    "Saudi Arabia": {"de": "Saudi-Arabien", "fr": "Arabie saoudite", "es": "Arabia Saudita", "ru": "Саудовская Аравия", "uk": "Саудівська Аравія", "ro": "Arabia Saudită", "zh": "沙特阿拉伯"},
    "Serbia": {"de": "Serbien", "fr": "Serbie", "es": "Serbia", "ru": "Сербия", "uk": "Сербія", "ro": "Serbia", "zh": "塞尔维亚"},
    "Singapore": {"de": "Singapur", "fr": "Singapour", "es": "Singapur", "ru": "Сингапур", "uk": "Сінгапур", "ro": "Singapore", "zh": "新加坡"},
    "Slovakia": {"de": "Slowakei", "fr": "Slovaquie", "es": "Eslovaquia", "ru": "Словакия", "uk": "Словаччина", "ro": "Slovacia", "zh": "斯洛伐克"},
    "Slovenia": {"de": "Slowenien", "fr": "Slovénie", "es": "Eslovenia", "ru": "Словения", "uk": "Словенія", "ro": "Slovenia", "zh": "斯洛文尼亚"},
    "South Africa": {"de": "Südafrika", "fr": "Afrique du Sud", "es": "Sudáfrica", "ru": "ЮАР", "uk": "ПАР", "ro": "Africa de Sud", "zh": "南非"},
    "South Korea": {"de": "Südkorea", "fr": "Corée du Sud", "es": "Corea del Sur", "ru": "Южная Корея", "uk": "Південна Корея", "ro": "Coreea de Sud", "zh": "韩国"},
    "Spain": {"de": "Spanien", "fr": "Espagne", "es": "España", "ru": "Испания", "uk": "Іспанія", "ro": "Spania", "zh": "西班牙"},
    "Sweden": {"de": "Schweden", "fr": "Suède", "es": "Suecia", "ru": "Швеция", "uk": "Швеція", "ro": "Suedia", "zh": "瑞典"},
    "Switzerland": {"de": "Schweiz", "fr": "Suisse", "es": "Suiza", "ru": "Швейцария", "uk": "Швейцарія", "ro": "Elveția", "zh": "瑞士"},
    "Syria": {"de": "Syrien", "fr": "Syrie", "es": "Siria", "ru": "Сирия", "uk": "Сирія", "ro": "Siria", "zh": "叙利亚"},
    "Taiwan": {"de": "Taiwan", "fr": "Taïwan", "es": "Taiwán", "ru": "Тайвань", "uk": "Тайвань", "ro": "Taiwan", "zh": "台湾"},
    "Tanzania": {"de": "Tansania", "fr": "Tanzanie", "es": "Tanzania", "ru": "Танзания", "uk": "Танзанія", "ro": "Tanzania", "zh": "坦桑尼亚"},
    "Thailand": {"de": "Thailand", "fr": "Thaïlande", "es": "Tailandia", "ru": "Таиланд", "uk": "Таїланд", "ro": "Thailanda", "zh": "泰国"},
    "Tunisia": {"de": "Tunesien", "fr": "Tunisie", "es": "Túnez", "ru": "Тунис", "uk": "Туніс", "ro": "Tunisia", "zh": "突尼斯"},
    "Turkey": {"de": "Türkei", "fr": "Turquie", "es": "Turquía", "ru": "Турция", "uk": "Туреччина", "ro": "Turcia", "zh": "土耳其"},
    "Ukraine": {"de": "Ukraine", "fr": "Ukraine", "es": "Ucrania", "ru": "Украина", "uk": "Україна", "ro": "Ucraina", "zh": "乌克兰"},
    "United Arab Emirates": {"de": "Vereinigte Arabische Emirate", "fr": "Émirats arabes unis", "es": "Emiratos Árabes Unidos", "ru": "ОАЭ", "uk": "ОАЕ", "ro": "Emiratele Arabe Unite", "zh": "阿联酋"},
    "United Kingdom": {"de": "Vereinigtes Königreich", "fr": "Royaume-Uni", "es": "Reino Unido", "ru": "Великобритания", "uk": "Велика Британія", "ro": "Regatul Unit", "zh": "英国"},
    "United States": {"de": "Vereinigte Staaten", "fr": "États-Unis", "es": "Estados Unidos", "ru": "США", "uk": "США", "ro": "Statele Unite", "zh": "美国"},
    "Vatican City": {"de": "Vatikanstadt", "fr": "Vatican", "es": "Ciudad del Vaticano", "ru": "Ватикан", "uk": "Ватикан", "ro": "Vatican", "zh": "梵蒂冈"},
    "Venezuela": {"de": "Venezuela", "fr": "Venezuela", "es": "Venezuela", "ru": "Венесуэла", "uk": "Венесуела", "ro": "Venezuela", "zh": "委内瑞拉"},
    "Vietnam": {"de": "Vietnam", "fr": "Viêt Nam", "es": "Vietnam", "ru": "Вьетнам", "uk": "Вʼєтнам", "ro": "Vietnam", "zh": "越南"},
}

def country_labels(en_name: str) -> dict[str, str]:
    labels = {"en": en_name}
    overrides = COUNTRY_I18N.get(en_name, {})
    for locale in LOCALES:
        if locale == "en":
            continue
        labels[locale] = overrides.get(locale, en_name)
    return labels


@dataclass(frozen=True)
class CitySpec:
    country: str
    city: dict[str, str]
    extra_search: tuple[str, ...] = field(default_factory=tuple)
    rank: int = 0


def _city(
    en: str,
    country: str,
    *,
    locales: dict[str, str] | None = None,
    extra_search: tuple[str, ...] = (),
    rank: int = 0,
) -> CitySpec:
    names = {"en": en}
    for locale in LOCALES:
        if locale == "en":
            continue
        names[locale] = (locales or {}).get(locale, en)
    return CitySpec(country=country, city=names, extra_search=extra_search, rank=rank)


@lru_cache(maxsize=1)
def _top_cities_by_country() -> dict[str, tuple[str, ...]]:
    path = _DATA_DIR / "top_cities_by_country.json"
    if path.is_file():
        raw: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        by_country = {country: tuple(cities[:10]) for country, cities in raw.items()}
    else:
        from sugar_sugar.build_city_data import TOP_CITIES_BY_COUNTRY

        by_country = {
            country: tuple(cities[:10]) for country, cities in TOP_CITIES_BY_COUNTRY.items()
        }
    for country in COUNTRY_NAMES:
        if country not in by_country or not by_country[country]:
            by_country[country] = (country,)
    return by_country


def _build_city_specs() -> tuple[CitySpec, ...]:
    top_cities = _top_cities_by_country()
    specs: list[CitySpec] = []
    seen: set[tuple[str, str]] = set()
    for country in COUNTRY_NAMES:
        cities = top_cities.get(country, (country,))[:10]
        for rank, city_en in enumerate(cities):
            key = (city_en, country)
            if key in seen:
                continue
            seen.add(key)
            override = CITY_I18N.get(key, {})
            specs.append(
                _city(
                    city_en,
                    country,
                    locales=override.get("locales"),
                    extra_search=tuple(override.get("extra_search", ())),
                    rank=rank,
                )
            )
    return tuple(specs)


CITY_SPECS: Final[tuple[CitySpec, ...]] = _build_city_specs()
