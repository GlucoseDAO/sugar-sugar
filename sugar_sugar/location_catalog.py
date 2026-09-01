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

# Bulgarian / Japanese / Korean names merged into COUNTRY_I18N below.
_COUNTRY_I18N_NEW: Final[dict[str, dict[str, str]]] = {
    "Afghanistan": {"bg": "Афганистан", "ja": "アフガニスタン", "ko": "아프가니스탄"},
    "Albania": {"bg": "Албания", "ja": "アルバニア", "ko": "알바니아"},
    "Algeria": {"bg": "Алжир", "ja": "アルジェリア", "ko": "알제리"},
    "Argentina": {"bg": "Аржентина", "ja": "アルゼンチン", "ko": "아르헨티나"},
    "Australia": {"bg": "Австралия", "ja": "オーストラリア", "ko": "오스트레일리아"},
    "Austria": {"bg": "Австрия", "ja": "オーストリア", "ko": "오스트리아"},
    "Azerbaijan": {"bg": "Азербайджан", "ja": "アゼルバイジャン", "ko": "아제르바이잔"},
    "Bangladesh": {"bg": "Бангладеш", "ja": "バングラデシュ", "ko": "방글라데시"},
    "Belarus": {"bg": "Беларус", "ja": "ベラルーシ", "ko": "벨라루스"},
    "Belgium": {"bg": "Белгия", "ja": "ベルギー", "ko": "벨기에"},
    "Bosnia and Herzegovina": {"bg": "Босна и Херцеговина", "ja": "ボスニア・ヘルツェゴビナ", "ko": "보스니아 헤르체고비나"},
    "Brazil": {"bg": "Бразилия", "ja": "ブラジル", "ko": "브라질"},
    "Bulgaria": {"bg": "България", "ja": "ブルガリア", "ko": "불가리아"},
    "Canada": {"bg": "Канада", "ja": "カナダ", "ko": "캐나다"},
    "Chile": {"bg": "Чили", "ja": "チリ", "ko": "칠레"},
    "China": {"bg": "Китай", "ja": "中国", "ko": "중국"},
    "Colombia": {"bg": "Колумбия", "ja": "コロンビア", "ko": "콜롬비아"},
    "Congo": {"bg": "Конго", "ja": "コンゴ", "ko": "콩고"},
    "Croatia": {"bg": "Хърватия", "ja": "クロアチア", "ko": "크로아티아"},
    "Cyprus": {"bg": "Кипър", "ja": "キプロス", "ko": "키프로스"},
    "Czech Republic": {"bg": "Чехия", "ja": "チェコ", "ko": "체코"},
    "Czechia": {"bg": "Чехия", "ja": "チェコ", "ko": "체코"},
    "Denmark": {"bg": "Дания", "ja": "デンマーク", "ko": "덴마크"},
    "Ecuador": {"bg": "Еквадор", "ja": "エクアドル", "ko": "에콰도르"},
    "Egypt": {"bg": "Египет", "ja": "エジプト", "ko": "이집트"},
    "Estonia": {"bg": "Естония", "ja": "エストニア", "ko": "에스토니아"},
    "Ethiopia": {"bg": "Етиопия", "ja": "エチオ피아", "ko": "에티오피아"},
    "Finland": {"bg": "Финландия", "ja": "フィンランド", "ko": "핀란드"},
    "France": {"bg": "Франция", "ja": "フランス", "ko": "프랑스"},
    "Germany": {"bg": "Германия", "ja": "ドイツ", "ko": "독일"},
    "Ghana": {"bg": "Гана", "ja": "ガーナ", "ko": "가나"},
    "Greece": {"bg": "Гърция", "ja": "ギリシャ", "ko": "그리스"},
    "Hungary": {"bg": "Унгария", "ja": "ハンガリー", "ko": "헝가리"},
    "Iceland": {"bg": "Исландия", "ja": "アイスランド", "ko": "아이슬란드"},
    "India": {"bg": "Индия", "ja": "インド", "ko": "인도"},
    "Indonesia": {"bg": "Индонезия", "ja": "インドネシア", "ko": "인도네시아"},
    "Iran": {"bg": "Иран", "ja": "イラン", "ko": "이란"},
    "Iraq": {"bg": "Ирак", "ja": "イラク", "ko": "이라크"},
    "Ireland": {"bg": "Ирландия", "ja": "アイルランド", "ko": "아일랜드"},
    "Israel": {"bg": "Израел", "ja": "イスラエル", "ko": "이스라엘"},
    "Italy": {"bg": "Италия", "ja": "イタリア", "ko": "이탈리아"},
    "Ivory Coast": {"bg": "Кот д'Ивоар", "ja": "コートジボワール", "ko": "코트디부아르"},
    "Japan": {"bg": "Япония", "ja": "日本", "ko": "일본"},
    "Jordan": {"bg": "Йордания", "ja": "ヨルダン", "ko": "요르단"},
    "Kenya": {"bg": "Кения", "ja": "ケニア", "ko": "케냐"},
    "Kosovo": {"bg": "Косово", "ja": "コソボ", "ko": "코소보"},
    "Kuwait": {"bg": "Кувейт", "ja": "クウェート", "ko": "쿠웨이트"},
    "Latvia": {"bg": "Латвия", "ja": "ラトビア", "ko": "라트비아"},
    "Lebanon": {"bg": "Ливан", "ja": "レバノン", "ko": "레바논"},
    "Libya": {"bg": "Либия", "ja": "リビア", "ko": "리비아"},
    "Lithuania": {"bg": "Литва", "ja": "リトアニア", "ko": "리투아니아"},
    "Luxembourg": {"bg": "Люксембург", "ja": "ルクセンブルク", "ko": "룩셈부르크"},
    "Malaysia": {"bg": "Малайзия", "ja": "マレーシア", "ko": "말레이시아"},
    "Malta": {"bg": "Малта", "ja": "マルタ", "ko": "몰타"},
    "Mexico": {"bg": "Мексико", "ja": "メキシコ", "ko": "멕시코"},
    "Moldova": {"bg": "Молдова", "ja": "モルドバ", "ko": "몰도바"},
    "Montenegro": {"bg": "Черна гора", "ja": "モンテネグロ", "ko": "몬테네그로"},
    "Morocco": {"bg": "Мароко", "ja": "モロッコ", "ko": "모로코"},
    "Netherlands": {"bg": "Нидерландия", "ja": "オランダ", "ko": "네덜란드"},
    "New Zealand": {"bg": "Нова Зеландия", "ja": "ニュージーランド", "ko": "뉴질랜드"},
    "Nigeria": {"bg": "Нигерия", "ja": "ナイジェリア", "ko": "나이지리아"},
    "North Korea": {"bg": "Северна Корея", "ja": "北朝鮮", "ko": "북한"},
    "North Macedonia": {"bg": "Северна Македония", "ja": "北マケドニア", "ko": "북마케도니아"},
    "Norway": {"bg": "Норвегия", "ja": "ノルウェー", "ko": "노르웨이"},
    "Pakistan": {"bg": "Пакистан", "ja": "パキスタン", "ko": "파키스탄"},
    "Palestine": {"bg": "Палестина", "ja": "パレスチナ", "ko": "팔레스타인"},
    "Peru": {"bg": "Перу", "ja": "ペルー", "ko": "페루"},
    "Philippines": {"bg": "Филипини", "ja": "フィリピン", "ko": "필리핀"},
    "Poland": {"bg": "Полша", "ja": "ポーランド", "ko": "폴란드"},
    "Portugal": {"bg": "Португалия", "ja": "ポルトガル", "ko": "포르투갈"},
    "Qatar": {"bg": "Катар", "ja": "カタール", "ko": "카타르"},
    "Romania": {"bg": "Румъния", "ja": "ルーマニア", "ko": "루마니아"},
    "Russia": {"bg": "Русия", "ja": "ロシア", "ko": "러시아"},
    "Saudi Arabia": {"bg": "Саудитска Арабия", "ja": "サウジアラビア", "ko": "사우디아라비아"},
    "Serbia": {"bg": "Сърбия", "ja": "セルビア", "ko": "세르비아"},
    "Singapore": {"bg": "Сингапур", "ja": "シンガポール", "ko": "싱가포르"},
    "Slovakia": {"bg": "Словакия", "ja": "スロバキア", "ko": "슬로바키아"},
    "Slovenia": {"bg": "Словения", "ja": "スロベニア", "ko": "슬로베니아"},
    "South Africa": {"bg": "Южна Африка", "ja": "南アフリカ", "ko": "남아프리카 공화국"},
    "South Korea": {"bg": "Южна Корея", "ja": "韓国", "ko": "대한민국"},
    "Spain": {"bg": "Испания", "ja": "スペイン", "ko": "스페인"},
    "Sweden": {"bg": "Швеция", "ja": "スウェーデン", "ko": "스웨덴"},
    "Switzerland": {"bg": "Швейцария", "ja": "スイス", "ko": "스위스"},
    "Syria": {"bg": "Сирия", "ja": "シリア", "ko": "시리아"},
    "Taiwan": {"bg": "Тайван", "ja": "台湾", "ko": "대만"},
    "Tanzania": {"bg": "Танзания", "ja": "タンザニア", "ko": "탄자니아"},
    "Thailand": {"bg": "Тайланд", "ja": "タイ", "ko": "태국"},
    "Tunisia": {"bg": "Тунис", "ja": "チュニジア", "ko": "튀니지"},
    "Turkey": {"bg": "Турция", "ja": "トルコ", "ko": "튀르키예"},
    "Ukraine": {"bg": "Украйна", "ja": "ウクライナ", "ko": "우크라이나"},
    "United Arab Emirates": {"bg": "ОАЕ", "ja": "アラブ首長国連邦", "ko": "아랍에미리트"},
    "United Kingdom": {"bg": "Обединено кралство", "ja": "イギリス", "ko": "영국"},
    "United States": {"bg": "САЩ", "ja": "アメリカ合衆国", "ko": "미국"},
    "Vatican City": {"bg": "Ватикан", "ja": "バチカン", "ko": "바티칸"},
    "Venezuela": {"bg": "Венецуела", "ja": "ベネズエラ", "ko": "베네수엘라"},
    "Vietnam": {"bg": "Виетнам", "ja": "ベトナム", "ko": "베트남"},
    "Andorra": {"bg": "Андора", "ja": "アンドラ", "ko": "안도라"},
    "Angola": {"bg": "Ангола", "ja": "アンゴラ", "ko": "앙골라"},
    "Antigua and Barbuda": {"bg": "Антигуа и Барбуда", "ja": "アンティグア・バーブーダ", "ko": "앤티가 바부다"},
    "Armenia": {"bg": "Армения", "ja": "アルメニア", "ko": "아르메니아"},
    "Bahamas": {"bg": "Бахами", "ja": "バハマ", "ko": "바하마"},
    "Bahrain": {"bg": "Бахрейн", "ja": "バーレーン", "ko": "바레인"},
    "Barbados": {"bg": "Барбадос", "ja": "バルバドス", "ko": "바베이도스"},
    "Belize": {"bg": "Белиз", "ja": "ベリーズ", "ko": "벨리즈"},
    "Benin": {"bg": "Бенин", "ja": "ベナン", "ko": "베냉"},
    "Bhutan": {"bg": "Бутан", "ja": "ブータン", "ko": "부탄"},
    "Bolivia": {"bg": "Боливия", "ja": "ボリビア", "ko": "볼리비아"},
    "Botswana": {"bg": "Ботсвана", "ja": "ボツワナ", "ko": "보츠와나"},
    "Brunei": {"bg": "Бруней", "ja": "ブルネイ", "ko": "브루나이"},
    "Burkina Faso": {"bg": "Буркина Фасо", "ja": "ブルキナファソ", "ko": "부르키나파소"},
    "Burundi": {"bg": "Бурунди", "ja": "ブルンジ", "ko": "부룬디"},
    "Cambodia": {"bg": "Камбоджа", "ja": "カンボジア", "ko": "캄보디아"},
    "Cameroon": {"bg": "Камерун", "ja": "カメルーン", "ko": "카메룬"},
    "Cape Verde": {"bg": "Кабо Верде", "ja": "カーボベルデ", "ko": "카보베르데"},
    "Central African Republic": {"bg": "Централноафриканска република", "ja": "中央アフリカ共和国", "ko": "중앙아프리카 공화국"},
    "Chad": {"bg": "Чад", "ja": "チャド", "ko": "차드"},
    "Comoros": {"bg": "Коморски острови", "ja": "コモロ", "ko": "코모로"},
    "Costa Rica": {"bg": "Коста Рика", "ja": "コスタリカ", "ko": "코스타리카"},
    "Cuba": {"bg": "Куба", "ja": "キューバ", "ko": "쿠바"},
    "Djibouti": {"bg": "Джибути", "ja": "ジブチ", "ko": "지부티"},
    "Dominica": {"bg": "Доминика", "ja": "ドミニカ国", "ko": "도미니카 연방"},
    "Dominican Republic": {"bg": "Доминиканска република", "ja": "ドミニカ共和国", "ko": "도미니카 공화국"},
    "El Salvador": {"bg": "Салвадор", "ja": "エルサルバドル", "ko": "엘살바도르"},
    "Equatorial Guinea": {"bg": "Екваториална Гвинея", "ja": "赤道ギニア", "ko": "적도 기니"},
    "Eritrea": {"bg": "Еритрея", "ja": "エリトリア", "ko": "에리트레아"},
    "Eswatini": {"bg": "Есватини", "ja": "エスワティニ", "ko": "에스와티니"},
    "Fiji": {"bg": "Фиджи", "ja": "フィジー", "ko": "피지"},
    "Gabon": {"bg": "Габон", "ja": "ガボン", "ko": "가봉"},
    "Gambia": {"bg": "Гамбия", "ja": "ガンビア", "ko": "감비아"},
    "Georgia": {"bg": "Грузия", "ja": "ジョージア", "ko": "조지아"},
    "Grenada": {"bg": "Гренада", "ja": "グレナダ", "ko": "그레나다"},
    "Guatemala": {"bg": "Гватемала", "ja": "グアテマラ", "ko": "과테말라"},
    "Guinea": {"bg": "Гвинея", "ja": "ギニア", "ko": "기니"},
    "Guinea-Bissau": {"bg": "Гвинея-Бисау", "ja": "ギニアビサウ", "ko": "기니비사우"},
    "Guyana": {"bg": "Гаяна", "ja": "ガイアナ", "ko": "가이아나"},
    "Haiti": {"bg": "Хаити", "ja": "ハイチ", "ko": "아이티"},
    "Honduras": {"bg": "Хондурас", "ja": "ホンジュラス", "ko": "온두라스"},
    "Jamaica": {"bg": "Ямайка", "ja": "ジャマイカ", "ko": "자메이카"},
    "Kazakhstan": {"bg": "Казахстан", "ja": "カザフスタン", "ko": "카자흐스탄"},
    "Kiribati": {"bg": "Кирибати", "ja": "キリバス", "ko": "키리바시"},
    "Kyrgyzstan": {"bg": "Киргизстан", "ja": "キルギス", "ko": "키르기스스탄"},
    "Laos": {"bg": "Лаос", "ja": "ラオス", "ko": "라오스"},
    "Lesotho": {"bg": "Лесото", "ja": "レソト", "ko": "레소토"},
    "Liberia": {"bg": "Либерия", "ja": "リベリア", "ko": "라이베리아"},
    "Liechtenstein": {"bg": "Лихтенщайн", "ja": "リヒテンシュタイン", "ko": "리히텐슈타인"},
    "Madagascar": {"bg": "Мадагаскар", "ja": "マダガスカル", "ko": "마다가스카르"},
    "Malawi": {"bg": "Малави", "ja": "マラウイ", "ko": "말라위"},
    "Maldives": {"bg": "Малдиви", "ja": "モルディブ", "ko": "몰디브"},
    "Mali": {"bg": "Мали", "ja": "マリ", "ko": "말리"},
    "Marshall Islands": {"bg": "Маршалови острови", "ja": "マーシャル諸島", "ko": "마셜 제도"},
    "Mauritania": {"bg": "Мавритания", "ja": "モーリタニア", "ko": "모리타니"},
    "Mauritius": {"bg": "Мавриций", "ja": "モーリシャス", "ko": "모리셔스"},
    "Micronesia": {"bg": "Микронезия", "ja": "ミクロネシア", "ko": "미크로네시아"},
    "Monaco": {"bg": "Монако", "ja": "モナコ", "ko": "모나코"},
    "Mongolia": {"bg": "Монголия", "ja": "モンゴル", "ko": "몽골"},
    "Mozambique": {"bg": "Мозамбик", "ja": "モザンビーク", "ko": "모잠비크"},
    "Myanmar": {"bg": "Мианмар", "ja": "ミャンマー", "ko": "미얀마"},
    "Namibia": {"bg": "Намибия", "ja": "ナミビア", "ko": "나미비아"},
    "Nauru": {"bg": "Науру", "ja": "ナウル", "ko": "나우루"},
    "Nepal": {"bg": "Непал", "ja": "ネパール", "ko": "네팔"},
    "Nicaragua": {"bg": "Никарагуа", "ja": "ニカラグア", "ko": "니카라과"},
    "Niger": {"bg": "Нигер", "ja": "ニジェール", "ko": "니제르"},
    "Oman": {"bg": "Оман", "ja": "オマーン", "ko": "오만"},
    "Palau": {"bg": "Палау", "ja": "パラオ", "ko": "팔라우"},
    "Panama": {"bg": "Панама", "ja": "パナマ", "ko": "파나마"},
    "Papua New Guinea": {"bg": "Папуа Нова Гвинея", "ja": "パプアニューギニア", "ko": "파푸아뉴기니"},
    "Paraguay": {"bg": "Парагвай", "ja": "パラグアイ", "ko": "파라과이"},
    "Rwanda": {"bg": "Руанда", "ja": "ルワンダ", "ko": "르완다"},
    "Saint Kitts and Nevis": {"bg": "Сейнт Китс и Невис", "ja": "セントクリストファー・ネイビス", "ko": "세인트키츠 네비스"},
    "Saint Lucia": {"bg": "Сейнт Лусия", "ja": "セントルシア", "ko": "세인트루시아"},
    "Saint Vincent and the Grenadines": {"bg": "Сейнт Винсент и Гренадини", "ja": "セントビンセントおよびグレナディーン諸島", "ko": "세인트빈센트 그레나딘"},
    "Samoa": {"bg": "Самоа", "ja": "サモア", "ko": "사모아"},
    "San Marino": {"bg": "Сан Марино", "ja": "サンマリノ", "ko": "산마리노"},
    "Sao Tome and Principe": {"bg": "Сао Томе и Принсипи", "ja": "サントメ・プリンシペ", "ko": "상투메 프린시페"},
    "Senegal": {"bg": "Сенегал", "ja": "セネガル", "ko": "세네갈"},
    "Seychelles": {"bg": "Сейшели", "ja": "セーシェル", "ko": "세이셸"},
    "Sierra Leone": {"bg": "Сиера Леоне", "ja": "シエラレオネ", "ko": "시에라리온"},
    "Solomon Islands": {"bg": "Соломонови острови", "ja": "ソロモン諸島", "ko": "솔로몬 제도"},
    "Somalia": {"bg": "Сомалия", "ja": "ソマリア", "ko": "소말리아"},
    "South Sudan": {"bg": "Южен Судан", "ja": "南スーダン", "ko": "남수단"},
    "Sri Lanka": {"bg": "Шри Ланка", "ja": "スリランカ", "ko": "스리랑카"},
    "Sudan": {"bg": "Судан", "ja": "スーダン", "ko": "수단"},
    "Suriname": {"bg": "Суринам", "ja": "スリナム", "ko": "수리남"},
    "Tajikistan": {"bg": "Таджикистан", "ja": "タジキスタン", "ko": "타지키스탄"},
    "Timor-Leste": {"bg": "Източен Тимор", "ja": "東ティモール", "ko": "동티모르"},
    "Togo": {"bg": "Того", "ja": "トーゴ", "ko": "토고"},
    "Tonga": {"bg": "Тонга", "ja": "トンガ", "ko": "통가"},
    "Trinidad and Tobago": {"bg": "Тринидад и Тобаго", "ja": "トリニダード・トバゴ", "ko": "트리니다드 토바고"},
    "Turkmenistan": {"bg": "Туркменистан", "ja": "トルクメニスタン", "ko": "투르크메니스탄"},
    "Tuvalu": {"bg": "Тувалу", "ja": "ツバル", "ko": "투발루"},
    "Uganda": {"bg": "Уганда", "ja": "ウガンダ", "ko": "우간다"},
    "Uruguay": {"bg": "Уругвай", "ja": "ウルグアイ", "ko": "우루과이"},
    "Uzbekistan": {"bg": "Узбекистан", "ja": "ウズベキスタン", "ko": "우즈베키스탄"},
    "Vanuatu": {"bg": "Вануату", "ja": "バヌアツ", "ko": "바누아투"},
    "Yemen": {"bg": "Йемен", "ja": "イエメン", "ko": "예멘"},
    "Zambia": {"bg": "Замбия", "ja": "ザンビア", "ko": "잠비아"},
    "Zimbabwe": {"bg": "Зимбабве", "ja": "ジンバブエ", "ko": "짐바브웨"},
}

for _country, _extra in _COUNTRY_I18N_NEW.items():
    COUNTRY_I18N.setdefault(_country, {}).update(_extra)

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
