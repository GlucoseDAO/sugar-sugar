from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Final, Optional

from sugar_sugar.i18n import normalize_locale, t
from sugar_sugar.static_markdown import static_markdown_iframe

_CONSENT_NOTICE_MD_RELATIVE_PATH: Final[Path] = Path(
    "data/input/patient_consent_form/Teilnehmerinformation_und_Einwilligungserklaerung_v2.docx.md"
)


def _project_root() -> Path:
    # sugar_sugar/consent_notice_text.py -> repo root
    return Path(__file__).resolve().parents[1]


def _consent_notice_md_path() -> Path:
    return _project_root() / _CONSENT_NOTICE_MD_RELATIVE_PATH


_CHECKBOX_GLYPHS_LINE_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^\s*[☐☑]\s*")
_CHECKBOX_GLYPHS_ANYWHERE_RE: Final[re.Pattern[str]] = re.compile(r"[☐☑]\s*")
_ITTEILUNGEN_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^(\*?)itteilungen\b")


def _sanitize_markdown(text: str) -> str:
    # Remove "checkbox" glyphs from the plain text/markdown (we use real UI checkboxes).
    text = _CHECKBOX_GLYPHS_LINE_RE.sub("", text)
    text = _CHECKBOX_GLYPHS_ANYWHERE_RE.sub("", text)

    # Fix a known conversion artifact present in the markdown export.
    text = _ITTEILUNGEN_RE.sub(r"\1Mitteilungen", text)

    # Minor cleanup for artifacts caused by docx->markdown conversion.
    text = text.replace("Ihnenzwei", "Ihnen zwei")
    text = text.replace("historischenDaten", "historischen Daten")
    text = text.replace("fürv", "für")
    return text.strip()


def _read_markdown_file(path: Path) -> str:
    if not path.exists():
        return ""
    return _sanitize_markdown(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def consent_notice_markdown_de() -> str:
    return _read_markdown_file(_consent_notice_md_path())


# NOTE: These translations are derived from the German source .docx.
# They are used when the UI language is not German.
_CONSENT_NOTICE_TRANSLATIONS: Final[dict[str, list[str]]] = {
    "en": [
        "Participant information",
        "Prediction of glucose trajectories based on previous value patterns: an online benchmarking study",
        "Short title: Sugar-Sugar glucose prediction study",
        "Please read the following text carefully before making a decision. If you have questions, please contact the study leadership.",
        "Dear participant,",
        "we would like to ask whether you would like to take part in a scientific study. This study aims to systematically investigate, for the first time, how accurately humans can predict short-term glucose trajectories based on CGM data (data from a continuous glucose monitoring device). So far, there is no such benchmark. Your participation helps establish a reference value for human prediction accuracy, which can also be used in the long term to evaluate computer-based prediction models. Such prediction models are increasingly used in diabetes research and diabetes care and can be validated and compared using the data collected here.",
        "The study is conducted at IBIMA (Institute for Biostatistics and Informatics in Medicine and Ageing Research) at University Medicine Rostock, in collaboration with HEALES (Healthy Life Extension Society, Brussels), which acts as the funder of the study. The study leadership is Anton Kulaga (University Medicine Rostock) and Livia Zaharia (HEALES). At least 200 participants are planned.",
        "Participation in the study has no influence on any medical treatment. This is a purely online study without in-person appointments or medical procedures.",
        "How does the study work?",
        "Participation takes place entirely online via the Sugar-Sugar web application and takes about 15–20 minutes.",
        "First, you will fill out a short registration questionnaire (age, gender, country of residence, diabetes status, CGM usage history; optional: weight and height). You will then complete 6 to 12 prediction tasks. Each task shows three hours of historical CGM data from a publicly known, anonymized dataset. You will be shown the earlier time points of time series collected in other studies; you are asked to guess the later time points. You will draw the expected glucose trajectory for the following 60 minutes by placing 12 points in a chart. Before the scored tasks, you will be offered two optional practice rounds.",
        "After each task, the actual glucose values are displayed and accuracy metrics are calculated. At the end, you will receive a summary of your prediction accuracy (determined from the historical data) as well as a percentile rank.",
        "Prediction accuracy is compared across four groups: people with diabetes and CGM use, people with diabetes without CGM use, people without diabetes with CGM use, and people without diabetes without CGM experience; based on the data collected at the beginning you will be assigned to one of these groups.",
        "Optional upload of your own CGM data",
        "If you use a CGM device yourself, you can optionally upload an export file (CSV or JSON). Short segments are extracted from it and used for additional prediction tasks. This makes it possible to compare whether you predict your own data more accurately than the historical data. Date and time are not displayed in the user interface. The uploaded raw file is deleted within 30 days; only the extracted segments are retained in pseudonymized form.",
        "Voluntary participation",
        "Participation in this study is voluntary. You will only be included if you declare your consent. If you do not participate in the study or later wish to withdraw from it, you will not suffer any disadvantages. You can end participation at any time without giving reasons. Partial data from completed rounds are only included if at least 6 valid rounds have been completed.",
        "Who can participate?",
        "Adults aged 18 and over with internet access can participate. Both people with and without diabetes are eligible.",
        "What possible risks, discomforts, or side effects are associated with your participation?",
        "Since our study only collects data, there are no medical risks associated with participation. The accuracy values are relevant exclusively for research and have no diagnostic or clinical meaning.",
        "What possible benefits result from your participation in the study?",
        "There is no personal health benefit for you from participating. Participation contributes to a systematic benchmark of human glucose-value prediction ability. You will receive a personal accuracy summary as well as a results card. There is no monetary compensation.",
        "What rights and conditions are associated with participation?",
        "You can end your participation in the study at any time, even without giving reasons. If you do not wish to participate or wish to withdraw, you will not suffer any disadvantages.",
        "Please note that no separate insurance has been taken out for this project and the accompanying study, as it is a purely online study without medical procedures and therefore involves minimal risk.",
        "The study received a positive opinion from the responsible ethics committee.",
        "Data protection",
        "Below we would like to inform you comprehensively about what happens to your personal data within the study and how we handle it.",
        "1. Who is responsible for data processing and storage?",
        "University Medicine Rostock is responsible for data processing and storage within the study. HEALES operates the Sugar-Sugar application as a processor within the framework of a data processing agreement (DPA) with University Medicine Rostock. You can reach the responsible institution at the following contact details:",
        "Responsible institution: University Medicine Rostock – legally capable sub-entity of the University of Rostock",
        "Schillingallee 35, 18057 Rostock",
        "www.med.uni-rostock.de",
        "2. Who can you contact with questions about the study or data protection?",
        "To best safeguard your interests, you can also contact the following persons at any time with any questions about your participation in the study or the general procedure:",
        "Study contacts:",
        "• Anton Kulaga: anton.kulaga@uni-rostock.de",
        "• Livia Zaharia: livia.zaharia@uni-rostock.de",
        "For data protection questions, the data protection officer of University Medicine Rostock is also available:",
        "Doberaner Str. 142, 18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "If you wish to exercise your right to complain about unlawful data processing, please contact the competent supervisory authority:",
        "State Commissioner for Data Protection and Freedom of Information M-V",
        "Schloss Schwerin, Lennéstr. 1, 19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. Which data do we need from you to conduct the study?",
        "The following data are collected within the study:",
        "Contact details: email address (for registration, withdrawal handling, and optional re-notification).",
        "Demographic data: age, gender, country of residence; optional: weight and height.",
        "Health-related data: diabetes status (yes/no, type, duration), CGM usage history.",
        "Study data: predicted and, if applicable, actual glucose values, accuracy metrics (MAE, MSE, RMSE, MAPE).",
        "Consent preferences: documentation of your consent decisions.",
        "Optional – own CGM data: if uploaded, only extracted segments are retained in pseudonymized form.",
        "4. Where do we obtain this data?",
        "All data are collected exclusively directly from you as a participant, through your inputs in the Sugar-Sugar web application (registration questionnaire, prediction tasks, optional CGM upload). No data are obtained from other sources. The legal basis for processing is your explicit consent pursuant to Art. 6(1)(a) GDPR and Art. 9(2)(a) GDPR.",
        "5. For what purposes do we need your data and how are they protected?",
        "The data listed above are needed exclusively for the scientific investigation of human glucose prediction ability and are used only for this purpose.",
        "Each participating person receives a random study ID during registration. All research data are linked exclusively to this ID. A separate encrypted mapping file linking the study ID to the email address is stored separately and is accessible only to the study leadership (Anton Kulaga and Livia Zaharia). This file is used exclusively to handle withdrawal requests and optional result notifications. For other study staff, it is therefore not possible to identify you.",
        "Data are encrypted during transmission and at rest. Storage takes place on servers within the European Union (GDPR-compliant hosting). Access is regulated based on roles.",
        "6. Where and for how long are the data stored?",
        "The data collected within the study are first pseudonymized. The mapping file (email – study ID) is expected to be destroyed 12 months after study completion. After that, the study data are anonymized, i.e., they no longer allow any inference about your identity.",
        "Pseudonymized research data are retained for 10 years in accordance with German research standards. Uploaded raw CGM files are deleted within 30 days.",
        "7. Who will learn about your data? Will your data be shared or published?",
        "Pseudonymized research data are accessible only to the study team at IBIMA. The mapping file is accessible only to the study leadership (Anton Kulaga and Livia Zaharia). Study results are published only in aggregated, anonymized form. It is not possible to infer individual persons from publications.",
        "8. What data protection rights do you have when participating in the study?",
        "In accordance with applicable data protection regulations (GDPR), you have the right of access to stored data, rectification of inaccurate data, erasure of data, restriction of processing, objection to unreasonable processing, and the right to data portability of the data stored about you. You also have the right to withdraw your consent. You can exercise these rights at any time with the responsible institution.",
        "Right of withdrawal",
        "In principle, you can withdraw your consent at any time without giving reasons. Please contact the study leadership (anton.kulaga@uni-rostock.de). Your withdrawal takes effect from the time you submit it. Processing of your data up to that point remains lawful. In the event of withdrawal, deletion of personal data will be initiated. Study data that have already been anonymized and can therefore no longer be linked to you will continue to be used within the study. The mapping file is expected to be destroyed 12 months after study completion; after that, an individual assignment and thus deletion are no longer possible.",
        "Legal basis: Art. 6(1)(a) GDPR and Art. 9(2)(a) GDPR.",
        "Declaration of consent for participation in the Sugar-Sugar glucose prediction study and the associated data processing",
        "Study ID of the participating person: _________________________________ (assigned at registration)",
        "The following consents are recorded electronically at the time of registration by checking the corresponding boxes.",
        "I participate in the study voluntarily. I have carefully read the participant information for the study and the included data protection notice and the associated rights.",
        "I have been informed that personal data will be collected, processed, and used within the study, in particular my health data (diabetes status, CGM usage history, prediction performance).",
        "As stated in the participant information, processing takes place in pseudonymized form by the study team at IBIMA, University Medicine Rostock. HEALES acts as a processor under a DPA.",
        "I am aware that if my data are stored anonymously, deletion at my request is no longer possible.",
        "Right of withdrawal",
        "I have been informed that I can end my participation at any time without giving reasons, without any disadvantages for me. I am also aware that I can withdraw my consent to data processing at any time with effect for the future, without affecting the lawfulness of processing carried out on the basis of consent until withdrawal. I have taken note of the information on withdrawing consent in the corresponding participant information.",
        "I hereby declare my voluntary agreement to participate in the Sugar-Sugar glucose prediction study. I understand that I can end my participation at any time without giving reasons, without any disadvantages for me.",
        "I hereby consent, including the corresponding participant information, to the processing of my personal and health data for the following purposes:",
        "☐  Collection, processing and use of my data listed in the participant information, in particular my health data (diabetes status, CGM usage history, prediction performance), in pseudonymized form for conducting the study by the study team at IBIMA, University Medicine Rostock, as well as HEALES as processor. Storage takes place on servers within the European Union. Research data are retained for up to 10 years after anonymization of the mapping file. Withdrawal is possible up to 12 months after study completion; afterwards identification is no longer possible. Legal basis: Art. 6(1)(a) GDPR and Art. 9(2)(a) GDPR.",
        "☐  The participating person confirms being 18 years of age or older.",
        "Optional consents (refusal has no influence on participation in the study):",
        "☐  I consent to uploading my own CGM data for the optional prediction task with my own data. Raw data will be deleted within 30 days; only pseudonymized extracted segments will be retained.",
        "☐  I agree to be informed by the study team via the registered email address about the study results after publication.",
        "Notifications from HEALES or related projects will be presented on a separate page after the study ends and are completely independent of participation in the study and consent.",
        "This study received clearance from the Ethics Committee of University Medicine Rostock. Registration number: Ref. number A 2026-0064. Received by the Ethics Committee: February 27, 2026.",
        "Questions or withdrawal requests: anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
    "ru": [
        "Информация для участников",
        "Прогнозирование траекторий глюкозы на основе предыдущих шаблонов значений: онлайн-бенчмаркинговое исследование",
        "Краткое название: исследование прогнозирования глюкозы Sugar-Sugar",
        "Пожалуйста, внимательно прочитайте следующий текст, прежде чем принимать решение. Если у вас есть вопросы, пожалуйста, свяжитесь с руководством исследования.",
        "Уважаемый участник,",
        "мы хотели бы спросить, хотите ли вы принять участие в научном исследовании. Целью данного исследования является систематическое изучение того, насколько точно люди могут предсказывать краткосрочные траектории глюкозы на основе данных НМГ (данных с устройства непрерывного мониторинга глюкозы). На данный момент такого эталона не существует. Ваше участие поможет установить справочное значение точности человеческого прогноза, которое также может быть использовано в долгосрочной перспективе для оценки компьютерных моделей прогнозирования. Такие модели прогнозирования все чаще используются в исследованиях диабета и при уходе за диабетиками и могут быть проверены и сопоставлены с использованием данных, собранных здесь.",
        "Исследование проводится в IBIMA (Институт биостатистики и информатики в медицине и исследованиях старения) в Университетской медицине Ростока в сотрудничестве с HEALES (Healthy Life Extension Society, Брюссель), которая выступает спонсором исследования. Руководство исследованием осуществляют Антон Кулага (Университетская медицина Ростока) и Ливия Захария (HEALES). Планируется участие не менее 200 человек.",
        "Участие в исследовании не влияет на какое-либо медицинское лечение. Это чисто онлайн-исследование без личных встреч или медицинских процедур.",
        "Как проходит исследование?",
        "Участие полностью онлайн через веб-приложение Sugar-Sugar и занимает около 15–20 минут.",
        "Сначала вы заполните короткую регистрационную анкету (возраст, пол, страна проживания, статус диабета, история использования НМГ; по желанию: вес и рост). Затем вы выполните от 6 до 12 заданий по прогнозированию. В каждом задании показаны три часа исторических данных НМГ из общеизвестного анонимного набора данных. Вам будут показаны ранние моменты времени временных рядов, собранных в других исследованиях; вам предлагается угадать более поздние моменты времени. Вы нарисуете ожидаемую траекторию глюкозы на следующие 60 минут, поставив 12 точек на графике. Перед оцениваемыми заданиями вам будут предложены два дополнительных тренировочных раунда.",
        "После каждого задания отображаются фактические значения глюкозы и рассчитываются показатели точности. В конце вы получите сводку вашей точности прогнозирования (определенную на основе исторических данных), а также процентильный ранг.",
        "Точность прогнозирования сравнивается в четырех группах: люди с диабетом и использованием НМГ, люди с диабетом без использования НМГ, люди без диабета с использованием НМГ и люди без диабета без опыта использования НМГ; на основе данных, собранных в начале, вы будете распределены в одну из этих групп.",
        "Дополнительная загрузка собственных данных НМГ",
        "Если вы сами используете устройство НМГ, вы можете по желанию загрузить файл экспорта (CSV или JSON). Из него извлекаются короткие сегменты и используются для дополнительных задач прогнозирования. Это позволяет сравнить, предсказываете ли вы свои собственные данные более точно, чем исторические данные. Дата и время не отображаются в пользовательском интерфейсе. Загруженный исходный файл удаляется в течение 30 дней; в псевдонимизированном виде сохраняются только извлеченные сегменты.",
        "Добровольное участие",
        "Участие в данном исследовании является добровольным. Вы будете включены только в том случае, если заявите о своем согласии. Если вы не участвуете в исследовании или позже пожелаете выйти из него, вы не понесете никаких убытков. Вы можете прекратить участие в любое время без объяснения причин. Частичные данные завершенных раундов включаются только в том случае, если было завершено не менее 6 действительных раундов.",
        "Кто может участвовать?",
        "Участвовать могут взрослые в возрасте 18 лет и старше, имеющие доступ к Интернету. Имеют право участвовать как люди с диабетом, так и без него.",
        "Какие возможные риски, неудобства или побочные эффекты связаны с вашим участием?",
        "Поскольку наше исследование только собирает данные, медицинских рисков, связанных с участием, нет. Значения точности имеют значение исключительно для исследований и не имеют диагностического или клинического смысла.",
        "Какие возможные выгоды дает ваше участие в исследовании?",
        "Участие не приносит вам личной пользы для здоровья. Участие вносит вклад в систематический эталон способности человека предсказывать значения глюкозы. Вы получите персональную сводку точности, а также карточку результатов. Денежная компенсация не предусмотрена.",
        "Какие права и условия связаны с участием?",
        "Вы можете прекратить свое участие в исследовании в любое время, даже без объяснения причин. Если вы не желаете участвовать или хотите отказаться, вы не понесете никаких убытков.",
        "Обратите внимание, что для этого проекта и сопутствующего исследования не было оформлено отдельное страхование, так как это чисто онлайн-исследование без медицинских процедур и, следовательно, сопряжено с минимальным риском.",
        "Исследование получило положительное заключение ответственного комитета по этике.",
        "Защита данных",
        "Ниже мы хотели бы подробно проинформировать вас о том, что происходит с вашими личными данными в рамках исследования и как мы с ними обращаемся.",
        "1. Кто несет ответственность за обработку и хранение данных?",
        "Университетская медицина Ростока несет ответственность за обработку и хранение данных в рамках исследования. HEALES управляет приложением Sugar-Sugar в качестве обработчика в рамках соглашения об обработке данных (DPA) с Университетской медициной Ростока. Вы можете связаться с ответственным учреждением по следующим контактным данным:",
        "Ответственное учреждение: Университетская медицина Ростока — правоспособное подразделение Университета Ростока",
        "Шиллингалли 35, 18057 Росток",
        "www.med.uni-rostock.de",
        "2. К кому вы можете обратиться с вопросами об исследовании или защите данных?",
        "Чтобы наилучшим образом защитить ваши интересы, вы также можете в любое время связаться со следующими лицами по любым вопросам, касающимся вашего участия в исследовании или общей процедуры:",
        "Контакты исследования:",
        "• Антон Кулага: anton.kulaga@uni-rostock.de",
        "• Ливия Захария: livia.zaharia@uni-rostock.de",
        "По вопросам защиты данных также доступен сотрудник по защите данных Университетской медицины Ростока:",
        "Doberaner Str. 142, 18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "Если вы хотите воспользоваться своим правом на жалобу на незаконную обработку данных, пожалуйста, свяжитесь с компетентным надзорным органом:",
        "Государственный комиссар по защите данных и свободе информации M-V",
        "Schloss Schwerin, Lennéstr. 1, 19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. Какие данные нам нужны от вас для проведения исследования?",
        "В рамках исследования собираются следующие данные:",
        "Контактные данные: адрес электронной почты (для регистрации, обработки отказа и дополнительного повторного уведомления).",
        "Демографические данные: возраст, пол, страна проживания; по желанию: вес и рост.",
        "Данные, связанные со здоровьем: статус диабета (да/нет, тип, стаж), история использования НМГ.",
        "Данные исследования: прогнозируемые и, если применимо, фактические значения глюкозы, показатели точности (MAE, MSE, RMSE, MAPE).",
        "Предпочтения по согласию: документирование ваших решений о согласии.",
        "По желанию — собственные данные НМГ: в случае загрузки сохраняются только извлеченные сегменты в псевдонимизированном виде.",
        "4. Откуда мы берем эти данные?",
        "Все данные собираются исключительно непосредственно от вас как участника через ваши данные в веб-приложении Sugar-Sugar (регистрационная анкета, задания по прогнозированию, дополнительная загрузка НМГ). Данные из других источников не поступают. Юридической основой для обработки является ваше явное согласие в соответствии со ст. 6(1)(a) GDPR и ст. 9(2)(a) GDPR.",
        "5. Для каких целей нам нужны ваши данные и как они защищены?",
        "Перечисленные выше данные необходимы исключительно для научного исследования способности человека предсказывать уровень глюкозы и используются только для этой цели.",
        "Каждое участвующее лицо получает случайный идентификатор исследования при регистрации. Все данные исследования связаны исключительно с этим идентификатором. Отдельный зашифрованный файл сопоставления, связывающий идентификатор исследования с адресом электронной почты, хранится отдельно и доступен только руководству исследования (Антону Кулаге и Ливии Захарии). Этот файл используется исключительно для обработки запросов на отказ и дополнительных уведомлений о результатах. Таким образом, для других сотрудников исследования идентификация вас невозможна.",
        "Данные шифруются при передаче и хранении. Хранение осуществляется на серверах в пределах Европейского Союза (хостинг, соответствующий требованиям GDPR). Доступ регулируется на основе ролей.",
        "6. Где и как долго хранятся данные?",
        "Данные, собранные в рамках исследования, сначала псевдонимизируются. Ожидается, что файл сопоставления (электронная почта — идентификатор исследования) будет уничтожен через 12 месяцев после завершения исследования. После этого данные исследования становятся анонимными, то есть они больше не позволяют сделать какой-либо вывод о вашей личности.",
        "Псевдонимизированные данные исследования хранятся в течение 10 лет в соответствии с немецкими исследовательскими стандартами. Загруженные исходные файлы НМГ удаляются в течение 30 дней.",
        "7. Кто узнает о ваших данных? Будут ли ваши данные переданы или опубликованы?",
        "Псевдонимизированные данные исследования доступны только исследовательской группе IBIMA. Файл сопоставления доступен только руководству исследования (Антону Кулаге и Ливии Захарии). Результаты исследования публикуются только в агрегированном анонимном виде. Из публикаций невозможно сделать вывод об отдельных лицах.",
        "8. Какие права на защиту данных у вас есть при участии в исследовании?",
        "В соответствии с применимыми правилами защиты данных (GDPR) вы имеете право на доступ к хранящимся данным, исправление неточных данных, удаление данных, ограничение обработки, возражение против необоснованной обработки и право на переносимость данных, хранящихся о вас. Вы также имеете право отозвать свое согласие. Вы можете воспользоваться этими правами в любое время в ответственном учреждении.",
        "Право на отказ",
        "В принципе, вы можете отозвать свое согласие в любое время без объяснения причин. Пожалуйста, свяжитесь с руководством исследования (anton.kulaga@uni-rostock.de). Ваш отказ вступает в силу с момента его подачи. Обработка ваших данных до этого момента остается законной. В случае отказа будет инициировано удаление персональных данных. Данные исследования, которые уже были анонимизированы и поэтому больше не могут быть связаны с вами, будут продолжать использоваться в рамках исследования. Ожидается, что файл сопоставления будет уничтожен через 12 месяцев после завершения исследования; после этого индивидуальное назначение и, следовательно, удаление станут невозможными.",
        "Юридическая основа: ст. 6(1)(a) GDPR и ст. 9(2)(a) GDPR.",
        "Заявление о согласии на участие в исследовании прогнозирования глюкозы Sugar-Sugar и связанную с этим обработку данных",
        "Идентификатор исследования участвующего лица: _________________________________ (назначается при регистрации)",
        "Следующие согласия фиксируются в электронном виде при регистрации путем установки соответствующих флажков.",
        "Я участвую в исследовании добровольно. Я внимательно прочитал информацию для участников исследования и включенное уведомление о защите данных и связанные с ним права.",
        "Меня проинформировали о том, что в рамках исследования будут собираться, обрабатываться и использоваться персональные данные, в частности мои данные о здоровье (статус диабета, история использования НМГ, результаты прогнозирования).",
        "Как указано в информации для участников, обработка осуществляется в псевдонимизированной форме исследовательской группой IBIMA Университетской медицины Ростока. HEALES выступает в качестве обработчика в рамках DPA.",
        "Я осознаю, что если мои данные хранятся анонимно, удаление по моему запросу более невозможно.",
        "Право на отказ",
        "Меня проинформировали о том, что я могу прекратить свое участие в любое время без объяснения причин, без каких-либо негативных последствий для меня. Я также осознаю, что могу отозвать свое согласие на обработку данных в любое время с действием на будущее, не затрагивая законность обработки, осуществляемой на основании согласия до момента отказа. Я принял к сведению информацию об отзыве согласия в соответствующей информации для участников.",
        "Настоящим я заявляю о своем добровольном согласии на участие в исследовании прогнозирования глюкозы Sugar-Sugar. Я понимаю, что могу прекратить свое участие в любое время без объяснения причин, без каких-либо негативных последствий для меня.",
        "Настоящим я даю согласие, включая соответствующую информацию для участников, на обработку моих личных данных и данных о здоровье для следующих целей:",
        "☐ Сбор, обработка и использование моих данных, перечисленных в информации для участников, в частности моих данных о здоровье (статус диабета, история использования НМГ, результаты прогнозирования), в псевдонимизированной форме для проведения исследования исследовательской группой IBIMA Университетской медицины Ростока, а также HEALES в качестве обработчика. Хранение осуществляется на серверах в пределах Европейского Союза. Исследовательские данные хранятся до 10 лет после анонимизации файла сопоставления. Отказ возможен в течение 12 месяцев после завершения исследования; после этого идентификация более невозможна. Юридическая основа: ст. 6(1)(a) GDPR и ст. 9(2)(a) GDPR.",
        "☐ Участвующее лицо подтверждает, что ему исполнилось 18 лет.",
        "Дополнительные согласия (отказ не влияет на участие в исследовании):",
        "☐ Я даю согласие на загрузку моих собственных данных НМГ для дополнительного задания по прогнозированию с моими собственными данными. Исходные данные будут удалены в течение 30 дней; будут сохранены только псевдонимизированные извлеченные сегменты.",
        "☐ Я согласен получать информацию от исследовательской группы по зарегистрированному адресу электронной почты о результатах исследования после публикации.",
        "Уведомления от HEALES или связанных проектов будут представлены на отдельной странице после окончания исследования и полностью независимы от участия в исследовании и согласия.",
        "Данное исследование получило одобрение Комитета по этике Университетской медицины Ростока. Регистрационный номер: Ref. number A 2026-0064. Получено Комитетом по этике: 27 февраля 2026.",
        "Вопросы или просьбы об отказах: anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
    "zh": [
        "参与者信息",
        "基于先前数值模式的血糖轨迹预测：一项在线基准研究",
        "简称：Sugar-Sugar 血糖预测研究",
        "在做出决定之前，请仔细阅读以下文字。如有疑问，请联系研究负责人。",
        "尊敬的参与者：",
        "我们想征求您是否愿意参加一项科学研究。本研究旨在首次系统地调查人类根据 CGM 数据（来自连续血糖监测设备的数据）预测短期血糖轨迹的准确性。到目前为止，还没有这样的基准。您的参与有助于为人类预测准确性建立参考值，从长远来看，该参考值也可用于评估基于计算机的预测模型。此类预测模型越来越多地用于糖尿病研究和糖尿病护理，并可以使用此处收集的数据进行验证和比较。",
        "本研究在罗斯托克大学医学中心 IBIMA（医学统计与信息学及老龄化研究所）进行，与作为研究资助者的 HEALES（健康延寿协会，布鲁塞尔）合作。研究负责人是 Anton Kulaga（罗斯托克大学医学中心）和 Livia Zaharia (HEALES)。计划至少有 200 名参与者。",
        "参加研究对任何医疗均无影响。这是一项纯粹的在线研究，无需面谈或医疗程序。",
        "研究是如何进行的？",
        "参与完全通过 Sugar-Sugar Web 应用程序在线进行，大约需要 15-20 分钟。",
        "首先，您将填写一份简短的注册问卷（年龄、性别、居住国、糖尿病状况、CGM 使用历史；可选：体重和身高）。然后，您将完成 6 到 12 个预测任务。每个任务显示来自公开已知的匿名数据集的三个小时历史 CGM 数据。您将看到在其他研究中收集的时间序列的早期时间点；您被要求猜测后期时间点。您将通过在图表中放置 12 个点来绘制接下来的 60 分钟的预期血糖轨迹。在计分任务之前，您将获得两轮可选的练习。",
        "在每个任务之后，将显示实际血糖值并计算准确性指标。最后，您将收到预测准确性的摘要（根据历史数据确定）以及百分位排名。",
        "预测准确性在四个组中进行比较：患有糖尿病并使用 CGM 的人、患有糖尿病但不使用 CGM 的人、不患有糖尿病但使用 CGM 的人、以及不患有糖尿病且没有 CGM 经验的人；根据开始时收集的数据，您将被分配到这些组之一。",
        "可选上传您自己的 CGM 数据",
        "如果您自己使用 CGM 设备，您可以选择上传导出文件（CSV 或 JSON）。从中提取短片段并用于额外的预测任务。这样就可以比较您预测自己数据的准确性是否高于历史数据。日期和时间不会显示在用户界面中。上传的原始文件将在 30 天内删除；仅以匿名形式保留提取的片段。",
        "自愿参与",
        "参加本研究是自愿的。只有在您声明同意的情况下，您才会被纳入。如果您不参加研究或稍后希望退出研究，您不会遭受任何损失。您可以随时终止参与，无需说明理由。只有在完成至少 6 个有效轮次的情况下，才会包含已完成轮次的部分数据。",
        "谁可以参加？",
        "18 岁及以上且具有互联网访问权限的成年人可以参加。患有和不患有糖尿病的人都有资格参加。",
        "您的参与可能存在哪些风险、不适或副作用？",
        "由于我们的研究仅收集数据，因此参与没有医疗风险。准确度值仅与研究相关，没有诊断或临床意义。",
        "参与研究可能带来哪些好处？",
        "参加研究对您没有个人的健康益处。参与有助于建立人类血糖值预测能力的系统基准。您将收到一份个人准确度摘要以及一份结果卡。没有金钱补偿。",
        "与参与相关的权利和条件有哪些？",
        "您可以随时终止参加研究，甚至无需说明理由。如果您不希望参加或希望退出，您不会遭受任何损失。",
        "请注意，由于这是一项纯粹的在线研究，没有医疗程序，因此风险极小，尚未为本项目和随附的研究购买单独的保险。",
        "该研究已获得负责伦理委员会的正式批准。",
        "数据保护",
        "下面我们想向您全面介绍研究中您的个人数据会发生什么以及我们如何处理这些数据。",
        "1. 谁负责数据处理和存储？",
        "罗斯托克大学医学中心负责研究内的数据处理和存储。HEALES 作为处理者，在与罗斯托克大学医学中心签署的数据处理协议 (DPA) 框架内运营 Sugar-Sugar 应用程序。您可以通过以下联系方式联系负责机构：",
        "负责机构：罗斯托克大学医学中心——罗斯托克大学的法人分实体",
        "Schillingallee 35, 18057 Rostock",
        "www.med.uni-rostock.de",
        "2. 如果有关于研究或数据保护的问题，您可以联系谁？",
        "为了最好地维护您的利益，您还可以随时就您参与研究或一般程序的问题联系以下人员：",
        "研究联系人：",
        "• Anton Kulaga: anton.kulaga@uni-rostock.de",
        "• Livia Zaharia: livia.zaharia@uni-rostock.de",
        "对于数据保护问题，罗斯托克大学医学中心的数据保护官也可以提供帮助：",
        "Doberaner Str. 142, 18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "如果您希望行使对非法数据处理提出投诉的权利，请联系主管监管机构：",
        "M-V 州数据保护和信息自由专员",
        "Schloss Schwerin, Lennéstr. 1, 19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. 我们需要您的哪些数据来进行研究？",
        "研究中收集以下数据：",
        "联系方式：电子邮件地址（用于注册、退出处理和可选的重新通知）。",
        "人口统计数据：年龄、性别、居住国；可选：体重和身高。",
        "健康相关数据：糖尿病状况（是/否、类型、病程）、CGM 使用历史。",
        "研究数据：预测的以及（如果适用）实际的血糖值、准确性指标 (MAE、MSE、RMSE, MAPE)。",
        "同意偏好：记录您的同意决定。",
        "可选——自己的 CGM 数据：如果上传，仅以匿名形式保留提取的片段。",
        "4. 我们从哪里获得这些数据？",
        "所有数据均由您作为参与者直接收集，通过您在 Sugar-Sugar Web 应用程序中的输入（注册问卷、预测任务、可选的 CGM 上传）。不从其他来源获得数据。处理的法律依据是您根据 GDPR 第 6(1)(a) 条和第 9(2)(a) 条表示的明确同意。",
        "5. 我们出于什么目的需要您的数据以及如何保护这些数据？",
        "上述数据仅用于人类血糖预测能力的科学调查，且仅用于此目的。",
        "每个参与者在注册期间都会收到一个随机的研究 ID。所有研究数据仅链接到此 ID。链接研究 ID 和电子邮件地址的单独加密映射文件单独存储，仅研究负责人（Anton Kulaga 和 Livia Zaharia）可以访问。此文件仅用于处理退出请求和可选的结果通知。对于其他研究人员，无法识别您的身份。",
        "数据在传输和存储过程中均经过加密。存储发生在欧盟境内的服务器上（符合 GDPR 的托管）。访问权限基于角色进行监管。",
        "6. 数据存储在何处以及存储多长时间？",
        "研究中收集的数据首先经过匿名化处理。映射文件（电子邮件 – 研究 ID）预计在研究完成后 12 个月内销毁。在那之后，研究数据被彻底匿名化，即它们不再允许任何关于您身份的推断。",
        "匿名化研究数据根据德国研究标准保留 10 年。上传的原始 CGM 文件将在 30 天内删除。",
        "7. 谁会了解您的数据？您的数据会被共享或发布吗？",
        "匿名研究数据仅供 IBIMA 的研究团队访问。映射文件仅供研究负责人（Anton Kulaga 和 Livia Zaharia）访问。研究结果仅以汇总、匿名形式发布。无法从个人出版物中推断。",
        "8. 参加研究时您享有哪些数据保护权利？",
        "根据适用的数据保护条例 (GDPR)，您有权访问存储的数据、纠正不准确的数据、删除数据、限制处理、反对不合理的处理，以及对关于您的存储数据的数据可移植性的权利。您也有权撤回您的同意。您可以随时向负责机构行使这些权利。",
        "撤回权",
        "原则上，您可以随时撤回您的同意，无需说明理由。请联系研究负责人 (anton.kulaga@uni-rostock.de)。您的撤回自提交之日起生效。在那之前对您数据的处理保持合法。如果撤回，将启动删除个人数据。已经匿名化且因此不再能链接到您的研究数据将继续在研究中使用。映射文件预计在研究完成后 12 个月销毁；之后无法再进行个人分配，因此也无法删除。",
        "法律依据：GDPR 第 6(1)(a) 条和 第 9(2)(a) 条。",
        "同意参加 Sugar-Sugar 血糖预测研究及相关数据处理的声明",
        "参与者的研究 ID：_________________________________（注册时分配）",
        "以下同意书在注册时通过勾选相应的框进行电子记录。",
        "我自愿参加研究。我已仔细阅读本研究的参与者信息以及包含的数据保护通知和相关权利。",
        "我已被告知，研究中将收集、处理和使用个人数据，特别是我的健康数据（糖尿病状况、CGM 使用历史、预测表现）。",
        "如参与者信息所述，由罗斯托克大学医学中心 IBIMA 的研究团队以匿名形式进行处理。HEALES 作为 DPA 下的处理者。",
        "我意识到，如果我的数据被匿名存储，我请求的删除将不再可能。",
        "撤回权",
        "我已被告知，我可以随时终止参加，无需说明理由，对我没有任何不利。我也意识到，我可以随时撤回对数据处理的同意，自撤回之日起生效，且不影响撤回前基于同意进行的处理的合法性。我已注意到相应参与者信息中关于撤回同意的信息。",
        "我特此声明自愿同意参加 Sugar-Sugar 血糖预测研究。我明白我可以随时终止参加，无需说明理由，对我没有任何不利。",
        "我特此同意（包括相应的参与者信息）出于以下目的处理我的个人和健康数据：",
        "☐ 收集、处理和使用参与者信息中列出的我的数据，特别是我的健康数据（糖尿病状况、CGM 使用历史、预测表现），由罗斯托克大学医学中心 IBIMA 的研究团队以及作为处理者的 HEALES 以匿名形式进行研究。存储发生在欧盟境内的服务器上。研究数据在映射文件匿名化后保留长达 10 年。在研究完成后 12 个月内可以撤回；之后无法再进行身份识别。法律依据：GDPR 第 6(1)(a) 条和 第 9(2)(a) 条。",
        "☐ 参与者确认已年满 18 岁。",
        "可选同意（拒绝不影响参加研究）：",
        "☐ 我同意上传我自己的 CGM 数据，以便使用我自己的数据完成可选的预测任务。原始数据将在 30 天内删除；仅保留匿名提取的片段。",
        "☐ 我同意研究团队在研究结果发布后通过注册的电子邮件地址告知我研究结果。",
        "来自 HEALES 或相关项目的通知将在研究结束后呈现在一个单独的页面上，并完全独立于本研究的参与和同意。",
        "本研究已获得罗斯托克大学医学中心伦理委员会的批准。注册号：Ref. number A 2026-0064。伦理委员会受理日期：2026年2月27日。",
        "问题或撤回请求：anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
    "uk": [
        "Інформація для учасників",
        "Прогнозування перебігу глюкози на основі попередніх шаблонів значень: онлайн‑бенчмаркінг‑дослідження",
        "Коротка назва: дослідження прогнозування глюкози Sugar‑Sugar",
        "Будь ласка, уважно прочитайте наведений нижче текст, перш ніж приймати рішення. Якщо у вас є запитання, зверніться до керівництва дослідження.",
        "Шановна учаснице, шановний учаснику,",
        "ми хотіли б запитати, чи бажаєте ви взяти участь у науковому дослідженні. Метою цього дослідження є вперше систематично вивчити, наскільки точно люди можуть прогнозувати короткострокові зміни глюкози на основі даних CGM (дані безперервного моніторингу глюкози). Наразі такого еталонного порівняння не існує. Ваша участь допоможе створити референтний показник точності людського прогнозу, який у майбутньому також можна буде використовувати для оцінювання комп’ютерних моделей прогнозування. Такі моделі дедалі частіше застосовують у дослідженнях діабету та в догляді за людьми з діабетом, і їх можна валідовувати та порівнювати за допомогою даних, зібраних у цьому дослідженні.",
        "Дослідження проводиться в IBIMA (Інститут біостатистики та інформатики в медицині й дослідженнях старіння) Університетської медицини Ростока у співпраці з HEALES (Healthy Life Extension Society, Брюссель), яка виступає спонсором/фундатором дослідження. Керівництво дослідження: Антон Кулага (Університетська медицина Ростока) та Лівія Захарія (HEALES). Планується щонайменше 200 учасників.",
        "Участь у дослідженні не впливає на будь‑яке медичне лікування. Це суто онлайн‑дослідження без очних зустрічей чи медичних процедур.",
        "Як відбувається дослідження?",
        "Участь повністю онлайн через вебзастосунок Sugar‑Sugar і триває приблизно 15–20 хвилин.",
        "Спочатку ви заповните коротку реєстраційну анкету (вік, стать, країна проживання, статус діабету, історія використання CGM; опційно: вага та зріст). Потім ви виконаєте 6–12 завдань на прогноз. Кожне завдання показує три години історичних даних CGM з публічно відомого, анонімізованого набору даних. Вам буде показано ранні часові точки часових рядів, зібраних у рамках інших досліджень; пізніші точки вам потрібно вгадати. Ви намалюєте очікувану траєкторію глюкози на наступні 60 хвилин, поставивши 12 точок на діаграмі. Перед оцінюваними завданнями вам запропонують дві необов’язкові тренувальні раунди.",
        "Після кожного завдання показуються фактичні значення глюкози та обчислюються метрики точності. Наприкінці ви отримаєте підсумок своєї точності прогнозування (за історичними даними) та перцентильний ранг.",
        "Точність прогнозу порівнюється у чотирьох групах: люди з діабетом і з використанням CGM, люди з діабетом без CGM, люди без діабету з CGM та люди без діабету без досвіду CGM; на основі початкових даних вас буде віднесено до однієї з цих груп.",
        "Опційне завантаження власних CGM‑даних",
        "Якщо ви користуєтеся CGM‑пристроєм, ви можете опційно завантажити файл експорту (CSV або JSON). З нього будуть виділені короткі сегмененти та використані для додаткових завдань прогнозування. Це дозволяє порівняти, чи прогнозуєте ви власні дані точніше, ніж історичні. Дата і час у інтерфейсі не показуються. Завантажений сирий файл буде видалено протягом 30 днів; зберігатимуться лише виділені сегменти у псевдонімізованому вигляді.",
        "Добровільність участі",
        "Участь у цьому дослідженні є добровільною. Вас включать лише тоді, коли ви надасте згоду. Якщо ви не братимете участі або згодом захочете вийти з дослідження, це не матиме для вас жодних негативних наслідків. Ви можете припинити участь у будь‑який момент без пояснення причин. Часткові дані з завершених раундів враховуються лише якщо завершено щонайменше 6 валідних раундів.",
        "Хто може брати участь?",
        "Брати участь можуть повнолітні особи від 18 років з доступом до інтернету. До участі допускаються як люди з діабетом, так і без діабету.",
        "Які можливі ризики, дискомфорт чи побічні ефекти пов’язані з участю?",
        "Оскільки в рамках нашого дослідження збираються лише дані, участь не пов’язана з медичними ризиками. Показники точності мають значення виключно для науки і не мають діагностичного чи клінічного значення.",
        "Яка можлива користь від участі в дослідженні?",
        "Особистої користі для здоров’я від участі немає. Участь робить внесок у систематичний бенчмарк здатності людини прогнозувати значення глюкози. Ви отримаєте персональний підсумок точності та картку результатів. Грошової винагороди немає.",
        "Які права та умови пов’язані з участю?",
        "Ви можете припинити участь у дослідженні будь‑коли, навіть без пояснення причин. Якщо ви не хочете брати участь або бажаєте вийти, це не спричинить для вас жодних негативних наслідків.",
        "Звертаємо увагу, що для цього проєкту та супровідного дослідження не було оформлено окремого страхування, оскільки це суто онлайн‑дослідження без медичних процедур і, відповідно, з мінімальним ризиком.",
        "Дослідження отримало позитивний висновок відповідальної етичної комісії.",
        "Захист даних",
        "Нижче ми хочемо детально поінформувати вас, що відбувається з вашими персональними даними в межах дослідження і як ми з ними поводимося.",
        "1. Хто відповідає за обробку та зберігання даних?",
        "Відповідальною за обробку та зберігання даних у межах дослідження є Університетська медицина Ростока. HEALES експлуатує застосунок Sugar‑Sugar як обробник даних на підставі договору про обробку даних (AVV/DPA) з Університетською медициною Ростока. З відповідальною установою можна зв’язатися за такими контактами:",
        "Відповідальна установа",
        "Університетська медицина Ростока – правоздатна підструктура Університету Ростока",
        "Schillingallee 35",
        "18057 Rostock",
        "www.med.uni-rostock.de",
        "2. До кого звертатися з питаннями щодо дослідження або захисту даних?",
        "Щоб максимально захистити ваші інтереси, ви можете будь‑коли звертатися з усіма питаннями щодо вашого супроводу в межах дослідження або загального перебігу за такими контактами:",
        "Контакти щодо дослідження",
        "Anton Kulaga: anton.kulaga@uni-rostock.de",
        "Livia Zaharia: livia.zaharia@uni-rostock.de",
        "З питань захисту даних вам також може допомогти уповноважений із захисту даних Університетської медицини Ростока:",
        "Уповноважений із захисту даних Університетської медицини Ростока",
        "Уповноважений із захисту даних",
        "Doberaner Str. 142",
        "18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "Якщо ви хочете скористатися правом на скаргу щодо незаконної обробки даних, зверніться до компетентного наглядового органу:",
        "Компетентний наглядовий орган із захисту даних",
        "Landesbeauftragte für Datenschutz und Informationsfreiheit M‑V",
        "Schloss Schwerin, Lennéstr. 1",
        "19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. Які дані ми потребуємо від вас для проведення дослідження?",
        "У межах дослідження збираються такі дані:",
        "Контактні дані: адреса електронної пошти (для реєстрації, опрацювання відкликання та опційного повторного повідомлення).",
        "Демографічні дані: вік, стать, країна проживання; опційно: вага та зріст.",
        "Дані про здоров’я: статус діабету (так/ні, тип, тривалість), історія використання CGM.",
        "Дані дослідження: спрогнозовані та, за потреби, фактичні значення глюкози, метрики точності (MAE, MSE, RMSE, MAPE).",
        "Налаштування згоди: документування ваших рішень щодо згоди.",
        "Опційно — власні CGM‑дані: у разі завантаження зберігаються лише виділені сегменти у псевдонімізованому вигляді.",
        "4. Звідки ми отримуємо ці дані?",
        "Усі дані збираються виключно безпосередньо від вас як учасника — через ваші введення у вебзастосунку Sugar‑Sugar (реєстраційна анкета, завдання прогнозування, опційне завантаження CGM). Дані з інших джерел не отримуються. Правова підстава обробки — ваша явна згода відповідно до ст. 6(1)(a) GDPR та ст. 9(2)(a) GDPR.",
        "5. Для яких цілей потрібні ваші дані та як вони захищені?",
        "Зазначені вище дані потрібні виключно для наукового дослідження здатності людини прогнозувати глюкозу і використовуються лише для цієї мети.",
        "Під час реєстрації кожному учаснику присвоюється випадковий ID дослідження. Усі наукові дані пов’язуються лише з цим ID. Окремий зашифрований файл відповідності, який зіставляє ID дослідження з адресою електронної пошти, зберігається окремо і доступний лише керівництву дослідження (Антон Кулага та Лівія Захарія). Цей файл використовується виключно для опрацювання запитів на відкликання та опційного сповіщення про результати. Для інших співробітників дослідження встановлення вашої особи неможливе.",
        "Дані шифруються під час передавання та у стані зберігання. Зберігання відбувається на серверах у межах Європейського Союзу (GDPR‑сумісний хостинг). Доступ регулюється за ролями.",
        "6. Де і як довго зберігаються дані?",
        "Дані, зібрані в межах дослідження, спочатку псевдонімізуються. Файл відповідності (email – ID дослідження) планується знищити приблизно через 12 місяців після завершення дослідження. Після цього дані дослідження стають анонімізованими, тобто не дозволяють встановити вашу особу.",
        "Псевдонімізовані наукові дані зберігаються 10 років відповідно до німецьких наукових стандартів. Завантажені сирі CGM‑файли видаляються протягом 30 днів.",
        "7. Хто матиме доступ до ваших даних? Чи передаються та публікуються дані?",
        "Псевдонімізовані наукові дані доступні лише команді дослідження в IBIMA. Файл відповідності доступний лише керівництву дослідження (Антон Кулага та Лівія Захарія). Результати дослідження публікуються лише у агрегованому, анонімізованому вигляді. З публікацій неможливо зробити висновки про окремих осіб.",
        "8. Які права на захист даних ви маєте під час участі в дослідженні?",
        "Відповідно до чинних норм захисту даних (GDPR) ви маєте право на доступ до збережених даних, виправлення неточних даних, видалення даних, обмеження обробки даних, заперечення проти неприйнятної обробки та право на перенесення даних, що зберігаються про вас. Також ви маєте право відкликати надану згоду. Ці права можна реалізувати будь‑коли щодо відповідальної установи.",
        "Право на відкликання",
        "Загалом ви можете відкликати свою згоду будь‑коли без пояснення причин. Для цього зверніться до керівництва дослідження (anton.kulaga@uni-rostock.de). Ваше відкликання діє з моменту його подання. Обробка даних до цього моменту залишається правомірною. У разі відкликання буде ініційовано видалення персональних даних. Дані дослідження, які вже були анонімізовані і тому більше не можуть бути пов’язані з вами, продовжуватимуть використовуватися в межах дослідження. Файл відповідності планується знищити приблизно через 12 місяців після завершення; після цього індивідуальне зіставлення і, відповідно, видалення стають неможливими.",
        "Правова підстава: ст. 6(1)(a) GDPR та ст. 9(2)(a) GDPR.",
        "Заява про згоду",
        "на участь у дослідженні Sugar‑Sugar з прогнозування глюкози",
        "та пов’язану з цим обробку даних",
        "ID дослідження учасника: _________________________________ (надається під час реєстрації)",
        "Наведені нижче згоди фіксуються електронно під час реєстрації шляхом позначення відповідних прапорців.",
        "Я беру участь у дослідженні добровільно. Я уважно прочитав/ла інформацію для учасників і наведені там повідомлення про захист даних та пов’язані права.",
        "Мене поінформовано, що в межах дослідження будуть збиратися, оброблятися та використовуватися мої персональні дані, зокрема дані про здоров’я (статус діабету, історія використання CGM, показники прогнозування).",
        "Як зазначено в інформації для учасників, обробка відбувається у псевдонімізованому вигляді командою дослідження в IBIMA Університетської медицини Ростока. HEALES виступає обробником за договором AVV/DPA.",
        "Я усвідомлюю, що у разі анонімізованого зберігання моїх даних їх видалення на моє прохання більше неможливе.",
        "Право на відкликання",
        "Мене поінформовано, що я можу припинити участь будь‑коли без пояснення причин і без негативних наслідків для себе. Також мені відомо, що я можу відкликати згоду на обробку даних у будь‑який момент з дією на майбутнє, не впливаючи на правомірність обробки, здійсненої до відкликання. Інформацію щодо відкликання, наведені у відповідній інформації для учасників, я прийняв/ла до відома.",
        "Я цим заявляю про добровільну згоду на участь у дослідженні Sugar‑Sugar. Я розумію, що можу припинити участь у будь‑який момент без пояснення причин і без негативних наслідків для себе.",
        "Я даю згоду (з урахуванням відповідної інформації для учасників) на обробку моїх персональних даних та даних про здоров’я з такими цілями:",
        "☐  Збір, обробка та використання моїх даних, перелічених в інформації для учасників, зокрема моїх даних про здоров’я (статус діабету, історія використання CGM, показники прогнозу), у псевдонімізованому вигляді для проведення дослідження командою IBIMA Університетської медицини Ростока, а також HEALES як обробником даних. Зберігання відбувається на серверах у межах ЄС. Наукові дані зберігаються до 10 років після анонімізації файлу відповідності. Відкликання можливе до 12 місяців після завершення; після цього ідентифікація неможлива. Правова підстава: ст. 6(1)(a) GDPR та ст. 9(2)(a) GDPR.",
        "☐  Учасник підтверджує, що йому/їй 18 років або більше.",
        "Опційні згоди (відмова не впливає на участь у дослідженні):",
        "☐  Я даю згоду завантажити власні CGM‑дані для опційного завдання прогнозування на власних даних. Сирі дані видаляються протягом 30 днів; зберігаються лише псевдонімізовані виділені сегменти.",
        "☐  Я погоджуюся, що команда дослідження повідомить мене через зареєстровану електронну пошту про результати дослідження після їх публікації.",
        "Повідомлення від HEALES або пов’язаних проєктів будуть представлені на окремій сторінці після завершення дослідження і є повністю незалежними від участі та згод у межах дослідження.",
        "Це дослідження отримало схвалення Комітету з етики Університетської медицини Ростока. Реєстраційний номер: Ref. number A 2026-0064. Отримано Комітетом з етики: 27 лютого 2026.",
        "Питання або запити на відкликання: anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
    "ro": [
        "Informații pentru participanți",
        "Predicția evoluției glucozei pe baza tiparelor anterioare ale valorilor: un studiu online de benchmarking",
        "Titlu scurt: studiul Sugar‑Sugar de predicție a glucozei",
        "Te rugăm să citești cu atenție următorul text înainte de a lua o decizie. Pentru întrebări, contactează conducerea studiului.",
        "Stimată participantă, stimate participant,",
        "dorim să te întrebăm dacă dorești să participi la un studiu științific. Prin acest studiu se urmărește investigarea sistematică, pentru prima dată, a cât de precis pot oamenii să prezică evoluțiile pe termen scurt ale glucozei pe baza datelor CGM (date de la un dispozitiv de monitorizare continuă a glucozei). Până acum nu există un astfel de reper de comparație. Participarea ta ajută la stabilirea unei valori de referință pentru acuratețea predicției umane, care pe termen lung poate fi folosită și pentru evaluarea modelelor de predicție asistate de calculator. Astfel de modele sunt utilizate tot mai des în cercetarea diabetului și în îngrijirea persoanelor cu diabet și pot fi validate și comparate pe baza datelor colectate aici.",
        "Studiul este realizat la IBIMA (Institutul de Biostatistică și Informatică în Medicină și Cercetarea Îmbătrânirii) din cadrul Universitätsmedizin Rostock, în colaborare cu HEALES (Healthy Life Extension Society, Bruxelles), care acționează ca finanțator al studiului. Conducerea studiului: Anton Kulaga (Universitätsmedizin Rostock) și Livia Zaharia (HEALES). Sunt planificați cel puțin 200 de participanți.",
        "Participarea la studiu nu are niciun impact asupra unui eventual tratament medical. Este un studiu exclusiv online, fără întâlniri în persoană sau măsuri medicale.",
        "Cum se desfășoară studiul?",
        "Participarea are loc complet online prin aplicația web Sugar‑Sugar și durează aproximativ 15–20 de minute.",
        "Mai întâi vei completa un chestionar scurt de înregistrare (vârstă, gen, țara de reședință, statutul diabetului, istoricul utilizării CGM; opțional: greutate și înălțime). Apoi vei rezolva 6 până la 12 sarcini de predicție. Fiecare sarcină prezintă trei ore de date CGM istorice dintr-un set de date public, anonim. Vei primi momentele timpurii ale seriilor temporale colectate în alte studii; momentele târzii trebuie ghicite. Vei desena evoluția așteptată a glucozei pentru următoarele 60 de minute, plasând 12 puncte într-un grafic. Înaintea sarcinilor evaluate ți se vor oferi două runde opționale de exersare.",
        "După fiecare sarcină se afișează valorile reale ale glucozei și se calculează metrici de acuratețe. La final primești un rezumat al acurateții predicției tale (determinată din datele istorice), precum și un rang percentil.",
        "Acuratețea predicției este comparată în patru grupuri: persoane cu diabet și utilizare CGM, persoane cu diabet fără utilizare CGM, persoane fără diabet cu utilizare CGM și persoane fără diabet fără experiență CGM; pe baza datelor colectate la început vei fi repartizat într-unul dintre aceste grupuri.",
        "Încărcare opțională a propriilor date CGM",
        "Dacă folosești un dispozitiv CGM, poți încărca opțional un fișier de export (CSV sau JSON). Din acesta se extrag segmente scurte și se folosesc pentru sarcini suplimentare de predicție. Astfel se poate compara dacă îți prezici propriile date mai precis decât datele istorice. Data și ora nu sunt afișate în interfața utilizatorului. Fișierul brut încărcat este șters în 30 de zile; se păstrează doar segmentele extrase în formă pseudonimizată.",
        "Participare voluntară",
        "Participarea la acest studiu este voluntară. Vei fi inclus doar dacă îți exprimi consimțământul. Dacă nu participi sau dacă ulterior dorești să te retragi, nu vei avea niciun dezavantaj. Poți încheia participarea oricând fără a oferi motive. Datele parțiale din runde finalizate sunt incluse doar dacă au fost finalizate cel puțin 6 runde valide.",
        "Cine poate participa?",
        "Pot participa persoane majore (18+) cu acces la internet. Sunt eligibile atât persoane cu diabet, cât și fără diabet.",
        "Ce riscuri, disconfort sau efecte secundare posibile sunt asociate participării?",
        "Deoarece în cadrul studiului nostru sunt colectate doar date, participarea nu implică riscuri medicale. Valorile de acuratețe sunt relevante exclusiv pentru cercetare și nu au nicio semnificație diagnostică sau clinică.",
        "Ce beneficii posibile rezultă din participarea la studiu?",
        "Nu există un beneficiu personal pentru sănătate prin participare. Participarea contribuie la un benchmark sistematic al capacității umane de a prezice valorile glucozei. Vei primi un rezumat personal al acurateții și un card al rezultatelor. Nu există compensație financiară.",
        "Ce drepturi și condiții sunt asociate participării?",
        "Poți încheia participarea la studiu oricând, chiar și fără a oferi motive. Dacă nu dorești să participi sau dorești să te retragi, nu vei avea niciun dezavantaj.",
        "Te informăm că pentru acest proiect și studiul însoțitor nu a fost încheiată o asigurare separată, deoarece este un studiu exclusiv online fără măsuri medicale și, prin urmare, cu risc minim.",
        "Studiul a primit avizul favorabil al comisiei de etică competente.",
        "Protecția datelor",
        "În cele ce urmează dorim să te informăm pe larg ce se întâmplă cu datele tale personale în cadrul studiului și cum le gestionăm.",
        "1. Cine este responsabil pentru prelucrarea și stocarea datelor?",
        "Universitätsmedizin Rostock este responsabilă pentru prelucrarea și stocarea datelor în cadrul studiului. HEALES operează aplicația Sugar‑Sugar ca împuternicit (processor) în baza unui contract de împuternicire (AVV/DPA) cu Universitätsmedizin Rostock. Poți contacta instituția responsabilă la următoarele coordonate:",
        "Instituție responsabilă",
        "Universitätsmedizin Rostock – subentitate cu personalitate juridică a Universității din Rostock",
        "Schillingallee 35",
        "18057 Rostock",
        "www.med.uni-rostock.de",
        "2. Cui te poți adresa pentru întrebări despre studiu sau despre protecția datelor?",
        "Pentru a-ți proteja cât mai bine interesele, te poți adresa oricând, pentru orice întrebări privind participarea ta în cadrul studiului sau desfășurarea lui, următorilor responsabili:",
        "Contacte pentru studiu",
        "Anton Kulaga: anton.kulaga@uni-rostock.de",
        "Livia Zaharia: livia.zaharia@uni-rostock.de",
        "Pentru întrebări legate de protecția datelor, îți stă la dispoziție și responsabilul cu protecția datelor al Universitätsmedizin Rostock:",
        "Responsabil cu protecția datelor al Universitätsmedizin Rostock",
        "Responsabilul cu protecția datelor",
        "Doberaner Str. 142",
        "18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "Dacă dorești să îți exerciți dreptul de a depune o plângere privind prelucrări ilegale de date, te rugăm să te adresezi autorității de supraveghere competente:",
        "Autoritate competentă de supraveghere în domeniul protecției datelor",
        "Der Landesbeauftragte für Datenschutz und Informationsfreiheit M‑V",
        "Schloss Schwerin, Lennéstr. 1",
        "19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. Ce date avem nevoie de la tine pentru desfășurarea studiului?",
        "În cadrul studiului sunt colectate următoarele date:",
        "Date de contact: adresă de email (pentru înregistrare, gestionarea retragerii consimțământului și re-notificare opțională).",
        "Date demografice: vârstă, gen, țara de reședință; opțional: greutate și înălțime.",
        "Date legate de sănătate: statutul diabetului (da/nu, tip, durată), istoricul utilizării CGM.",
        "Date de studiu: valorile glucozei prezise și, dacă este cazul, reale, metrici de acuratețe (MAE, MSE, RMSE, MAPE).",
        "Preferințe de consimțământ: documentarea deciziilor tale de consimțământ.",
        "Opțional – propriile date CGM: dacă sunt încărcate, se păstrează doar segmentele extrase în formă pseudonimizată.",
        "4. De unde obținem aceste date?",
        "Toate datele sunt colectate exclusiv direct de la tine ca participant, prin introducerile tale în aplicația web Sugar‑Sugar (chestionar de înregistrare, sarcini de predicție, încărcare CGM opțională). Nu sunt obținute date din alte surse. Temeiul juridic al prelucrării este consimțământul tău explicit conform art. 6 alin. 1 lit. a GDPR și art. 9 alin. 2 lit. a GDPR.",
        "5. În ce scopuri sunt necesare datele tale și cum sunt protejate?",
        "Datele menționate mai sus sunt necesare exclusiv pentru investigarea științifică a capacității umane de predicție a glucozei și sunt folosite doar în acest scop.",
        "Fiecărei persoane participante i se atribuie la înregistrare un ID de studiu aleator. Toate datele de cercetare sunt asociate exclusiv cu acest ID. Un fișier separat, criptat, de corespondență care asociază ID‑ul de studiu cu adresa de email este stocat separat și este accesibil doar conducerii studiului (Anton Kulaga și Livia Zaharia). Acest fișier este folosit exclusiv pentru gestionarea cererilor de retragere și notificarea opțională a rezultatelor. Pentru ceilalți membri ai studiului nu este posibilă identificarea ta.",
        "Datele sunt criptate în timpul transferului și în repaus. Stocarea are loc pe servere din Uniunea Europeană (hosting conform GDPR). Accesul este reglementat pe bază de roluri.",
        "6. Unde și pentru cât timp sunt stocate datele?",
        "Datele colectate în cadrul studiului sunt mai întâi pseudonimizate. Fișierul de corespondență (email – ID de studiu) este de așteptat să fie distrus la 12 luni după finalizarea studiului. După aceea, datele de studiu sunt anonimizate, adică nu mai permit nicio asociere cu persoana ta.",
        "Datele de cercetare pseudonimizate sunt păstrate 10 ani conform standardelor de cercetare din Germania. Fișierele brute CGM încărcate sunt șterse în 30 de zile.",
        "7. Cine are acces la datele tale? Sunt datele transmise sau publicate?",
        "Datele de cercetare pseudonimizate sunt accesibile doar echipei de studiu de la IBIMA. Fișierul de corespondență este accesibil doar conducerii studiului (Anton Kulaga și Livia Zaharia). Rezultatele studiului sunt publicate doar în formă agregată și anonimizată. Nu este posibilă identificarea persoanelor din publicații.",
        "8. Ce drepturi privind protecția datelor ai în cadrul participării la studiu?",
        "Conform reglementărilor aplicabile (GDPR), ai dreptul de acces la datele stocate, rectificarea datelor inexacte, ștergerea datelor, restricționarea prelucrării, opoziția față de prelucrări nejustificate și dreptul la portabilitatea datelor stocate despre tine. De asemenea, ai dreptul de a-ți retrage consimțământul. Aceste drepturi pot fi exercitate oricând față de instituția responsabilă.",
        "Dreptul de retragere",
        "În principiu, îți poți retrage consimțământul oricând fără a oferi motive. Te rugăm să contactezi conducerea studiului (anton.kulaga@uni-rostock.de). Retragerea produce efecte din momentul în care o transmiți. Prelucrarea datelor până la acel moment rămâne legală. În caz de retragere, se va iniția ștergerea datelor cu caracter personal. Datele de studiu deja anonimizate și care nu mai pot fi asociate cu tine vor continua să fie folosite în cadrul studiului. Fișierul de corespondență este de așteptat să fie distrus la 12 luni după finalizarea studiului; după aceea nu mai este posibilă asocierea individuală și, implicit, ștergerea.",
        "Temei juridic: art. 6 alin. 1 lit. a GDPR și art. 9 alin. 2 lit. a GDPR.",
        "Declarație de consimțământ",
        "pentru participarea la studiul Sugar‑Sugar de predicție a glucozei",
        "și prelucrarea de date asociată",
        "ID‑ul de studiu al persoanei participante: _________________________________ (atribuit la înregistrare)",
        "Consimțămintele de mai jos sunt înregistrate electronic la momentul înregistrării prin bifarea căsuțelor corespunzătoare.",
        "Particip voluntar la studiu. Am citit cu atenție informațiile pentru participanți și notele privind protecția datelor și drepturile asociate.",
        "Am fost informat(ă) că în cadrul studiului vor fi colectate, prelucrate și utilizate date cu caracter personal despre mine, în special date de sănătate (statutul diabetului, istoricul utilizării CGM, performanța predicției).",
        "Conform informațiilor pentru participanți, prelucrarea are loc în formă pseudonimizată de către echipa de studiu de la IBIMA, Universitätsmedizin Rostock. HEALES acționează ca împuternicit (processor) în cadrul unui AVV/DPA.",
        "Sunt conștient(ă) că, în cazul în care datele mele sunt stocate anonimizat, ștergerea lor la cererea mea nu mai este posibilă.",
        "Dreptul de retragere",
        "Am fost informat(ă) că îmi pot încheia participarea oricând fără a oferi motive, fără dezavantaje pentru mine. De asemenea, știu că îmi pot retrage consimțământul pentru prelucrarea datelor oricând, cu efect pentru viitor, fără a afecta legalitatea prelucrării efectuate până la retragere. Am luat la cunoștință informațiile privind retragerea consimțământului din informațiile pentru participanți aferente.",
        "Declar prin prezenta acordul meu voluntar de a participa la studiul Sugar‑Sugar de predicție a glucozei. Sunt conștient(ă) că îmi pot încheia participarea oricând fără a oferi motive, fără dezavantaje pentru mine.",
        "Îmi dau consimțământul (incluzând informațiile pentru participanți aferente) pentru prelucrarea datelor mele personale și de sănătate în următoarele scopuri:",
        "☐  Colectarea, prelucrarea și utilizarea datelor mele enumerate în informațiile pentru participanți, în special a datelor de sănătate (statutul diabetului, istoricul utilizării CGM, performanța predicției), în formă pseudonimizată pentru desfășurarea studiului de către echipa de la IBIMA, Universitätsmedizin Rostock, precum și de către HEALES ca împuternicit. Stocarea are loc pe servere din Uniunea Europeană. Datele de cercetare sunt păstrate până la 10 ani după anonimizarea fișierului de corespondență. Retragerea este posibilă până la 12 luni după finalizarea studiului; după aceea identificarea nu mai este posibilă. Temei juridic: art. 6 alin. 1 lit. a GDPR și art. 9 alin. 2 lit. a GDPR.",
        "☐  Persoana participantă confirmă că are 18 ani sau mai mult.",
        "Consimțăminte opționale (refuzul nu influențează participarea la studiu):",
        "☐  Sunt de acord să încarc propriile date CGM pentru sarcina opțională de predicție cu datele mele. Datele brute sunt șterse în 30 de zile; se păstrează doar segmentele extrase, pseudonimizate.",
        "☐  Sunt de acord ca echipa de studiu să mă informeze, prin adresa de email înregistrată, despre rezultatele studiului după publicare.",
        "Mesajele de la HEALES sau proiecte conexe vor fi prezentate pe o pagină separată după finalizarea studiului și sunt complet independente de participarea la studiu și de consimțământ.",
        "Acest studiu a primit aprobarea Comisiei de Etică a Universitätsmedizin Rostock. Număr de înregistrare: Ref. number A 2026-0064. Primit de Comisia de Etică: 27 februarie 2026.",
        "Întrebări sau cereri de retragere: anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
    "fr": [
        "Informations pour les participants",
        "Prédiction des trajectoires de glucose basée sur les schémas de valeurs précédents : une étude d'analyse comparative en ligne",
        "Titre court : Étude de prédiction du glucose Sugar-Sugar",
        "Veuillez lire attentivement le texte suivant avant de prendre une décision. Si vous avez des questions, veuillez contacter la direction de l'étude.",
        "Cher participant,",
        "nous aimerions vous demander si vous souhaitez participer à une étude scientifique. Cette étude vise à étudier systématiquement, pour la première fois, la précision avec laquelle les humains peuvent prédire les trajectoires de glucose à court terme sur la base des données CGM (données provenant d'un appareil de surveillance continue du glucose). Jusqu'à présent, il n'existe pas de telle référence. Votre participation aide à établir une valeur de référence pour la précision de la prédiction humaine, qui pourra également être utilisée à long terme pour évaluer les modèles de prédiction informatiques. De tels modèles de prédiction sont de plus en plus utilisés dans la recherche sur le diabète et les soins du diabète et peuvent être validés et comparés à l'aide des données collectées ici.",
        "L'étude est menée à l'IBIMA (Institut de biostatistique et d'informatique en médecine et recherche sur le vieillissement) de l'Université de médecine de Rostock, en collaboration avec HEALES (Healthy Life Extension Society, Bruxelles), qui agit en tant que financeur de l'étude. La direction de l'étude est assurée par Anton Kulaga (Université de médecine de Rostock) et Livia Zaharia (HEALES). Au moins 200 participants sont prévus.",
        "La participation à l'étude n'a aucune influence sur un quelconque traitement médical. Il s'agit d'une étude purement en ligne sans rendez-vous en personne ni procédures médicales.",
        "Comment l'étude fonctionne-t-elle ?",
        "La participation se déroule entièrement en ligne via l'application Web Sugar-Sugar et prend environ 15 à 20 minutes.",
        "Tout d'abord, vous remplirez un court questionnaire d'inscription (âge, sexe, pays de résidence, statut diabétique, historique d'utilisation du CGM ; facultatif : poids et taille). Vous effectuerez ensuite 6 à 12 tâches de prédiction. Chaque tâche montre trois heures de données CGM historiques provenant d'un ensemble de données anonymisé et publiquement connu. On vous montrera les premiers points temporels de séries chronologiques collectées dans d'autres études ; vous devrez deviner les points temporels ultérieurs. Vous tracerez la trajectoire de glucose attendue pour les 60 minutes suivantes en plaçant 12 points dans un graphique. Avant les tâches notées, deux manches d'entraînement facultatives vous seront proposées.",
        "Après chaque tâche, les valeurs de glucose réelles sont affichées et les mesures de précision sont calculées. À la fin, vous recevrez un résumé de votre précision de prédiction (déterminée à partir des données historiques) ainsi qu'un rang centile.",
        "La précision de la prédiction est comparée entre quatre groupes : personnes diabétiques utilisant un CGM, personnes diabétiques n'utilisant pas de CGM, personnes non diabétiques utilisant un CGM et personnes non diabétiques n'ayant pas d'expérience du CGM ; sur la base des données collectées au début, vous serez affecté à l'un de ces groupes.",
        "Téléchargement facultatif de vos propres données CGM",
        "Si vous utilisez vous-même un appareil CGM, vous pouvez facultativement télécharger un fichier d'exportation (CSV ou JSON). Des segments courts en sont extraits et utilisés pour des tâches de prédiction supplémentaires. Cela permet de comparer si vous prédisez vos propres données avec plus de précision que les données historiques. La date et l'heure ne sont pas affichées dans l'interface utilisateur. Le fichier brut téléchargé est supprimé dans les 30 jours ; seuls les segments extraits sont conservés sous forme pseudonymisée.",
        "Participation volontaire",
        "La participation à cette étude est volontaire. Vous ne serez inclus que si vous déclarez votre consentement. Si vous ne participez pas à l'étude ou si vous souhaitez vous en retirer plus tard, vous ne subirez aucun préjudice. Vous pouvez mettre fin à votre participation à tout moment sans donner de raisons. Les données partielles des manches terminées ne sont incluses que si au moins 6 manches valides ont été effectuées.",
        "Qui peut participer ?",
        "Les adultes âgés de 18 ans et plus ayant accès à Internet peuvent participer. Les personnes avec et sans diabète sont éligibles.",
        "Quels risques, inconvénients ou effets secondaires possibles sont associés à votre participation ?",
        "Puisque notre étude ne recueille que des données, il n'y a aucun risque médical associé à la participation. Les valeurs de précision sont pertinentes exclusivement pour la recherche et n'ont aucune signification diagnostique ou clinique.",
        "Quels avantages possibles résultent de votre participation à l'étude ?",
        "Il n'y a aucun avantage personnel pour votre santé à participer. La participation contribue à établir une référence systématique de la capacité humaine à prédire les valeurs de glucose. Vous recevrez un résumé personnel de votre précision ainsi qu'une fiche de résultats. Il n'y a pas de compensation monétaire.",
        "Quels droits et conditions sont associés à la participation ?",
        "Vous pouvez mettre fin à votre participation à l'étude à tout moment, même sans donner de raisons. Si vous ne souhaitez pas participer ou si vous souhaitez vous retirer, vous ne subirez aucun préjudice.",
        "Veuillez noter qu'aucune assurance séparée n'a été souscrite pour ce projet et l'étude qui l'accompagne, car il s'agit d'une étude purement en ligne sans procédures médicales et elle comporte donc un risque minimal.",
        "L'étude a reçu un avis favorable du comité d'éthique responsable.",
        "Protection des données",
        "Ci-dessous, nous aimerions vous informer de manière complète sur ce qu'il advient de vos données personnelles dans le cadre de l'étude et sur la manière dont nous les traitons.",
        "1. Qui est responsable du traitement et du stockage des données ?",
        "L'Université de médecine de Rostock est responsable du traitement et du stockage des données dans le cadre de l'étude. HEALES exploite l'application Sugar-Sugar en tant que sous-traitant dans le cadre d'un contrat de traitement des données (DPA) avec l'Université de médecine de Rostock. Vous pouvez contacter l'institution responsable aux coordonnées suivantes :",
        "Institution responsable : Université de médecine de Rostock – sous-entité juridiquement capable de l'Université de Rostock",
        "Schillingallee 35, 18057 Rostock",
        "www.med.uni-rostock.de",
        "2. Qui pouvez-vous contacter pour toute question concernant l'étude ou la protection des données ?",
        "Pour protéger au mieux vos intérêts, vous pouvez également contacter à tout moment les personnes suivantes pour toute question concernant votre participation à l'étude ou la procédure générale :",
        "Contacts de l'étude :",
        "• Anton Kulaga : anton.kulaga@uni-rostock.de",
        "• Livia Zaharia : livia.zaharia@uni-rostock.de",
        "Pour les questions relatives à la protection des données, le délégué à la protection des données de l'Université de médecine de Rostock est également disponible :",
        "Doberaner Str. 142, 18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "Si vous souhaitez exercer votre droit de réclamation concernant un traitement illégal de données, veuillez contacter l'autorité de contrôle compétente :",
        "Commissaire d'État à la protection des données et à la liberté d'information M-V",
        "Schloss Schwerin, Lennéstr. 1, 19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. De quelles données avons-nous besoin de votre part pour mener l'étude ?",
        "Les données suivantes sont collectées dans le cadre de l'étude :",
        "Coordonnées : adresse e-mail (pour l'inscription, la gestion du retrait et la re-notification facultative).",
        "Données démographiques : âge, sexe, pays de résidence ; facultatif : poids et taille.",
        "Données relatives à la santé : statut diabétique (oui/non, type, durée), historique d'utilisation du CGM.",
        "Données d'étude : valeurs de glucose prédites et, le cas échéant, réelles, mesures de précision (MAE, MSE, RMSE, MAPE).",
        "Préférences de consentement : documentation de vos décisions de consentement.",
        "Facultatif – propres données CGM : si elles sont téléchargées, seuls les segments extraits sont conservés sous forme pseudonymisée.",
        "4. Où obtenons-nous ces données ?",
        "Toutes les données sont collectées exclusivement directement auprès de vous en tant que participant, via vos saisies dans l'application Web Sugar-Sugar (questionnaire d'inscription, tâches de prédiction, téléchargement CGM facultatif). Aucune donnée n'est obtenue d'autres sources. La base juridique du traitement est votre consentement explicite conformément à l'Art. 6(1)(a) du RGPD et à l'Art. 9(2)(a) du RGPD.",
        "5. À quelles fins avons-nous besoin de vos données et comment sont-elles protégées ?",
        "Les données énumérées ci-dessus sont nécessaires exclusivement pour l'étude scientifique de la capacité humaine de prédiction du glucose et ne sont utilisées qu'à cette fin.",
        "Chaque participant reçoit un identifiant d'étude aléatoire lors de l'inscription. Toutes les données de recherche sont liées exclusivement à cet identifiant. Un fichier de correspondance crypté séparé liant l'identifiant de l'étude à l'adresse e-mail est stocké séparément et n'est accessible qu'à la direction de l'étude (Anton Kulaga et Livia Zaharia). Ce fichier est utilisé exclusivement pour gérer les demandes de retrait et les notifications de résultats facultatives. Pour les autres membres de l'étude, il n'est donc pas possible de vous identifier.",
        "Les données sont cryptées pendant la transmission et au repos. Le stockage a lieu sur des serveurs au sein de l'Union européenne (hébergement conforme au RGPD). L'accès est réglementé en fonction des rôles.",
        "6. Où et pendant combien de temps les données sont-elles stockées ?",
        "Les données collectées dans le cadre de l'étude sont d'abord pseudonymisées. Le fichier de correspondance (e-mail – ID d'étude) devrait être détruit 12 mois après la fin de l'étude. Après cela, les données de l'étude sont anonymisées, c'est-à-dire qu'elles ne permettent plus aucune déduction sur votre identité.",
        "Les données de recherche pseudonymisées sont conservées pendant 10 ans conformément aux normes de recherche allemandes. Les fichiers CGM bruts téléchargés sont supprimés dans les 30 jours.",
        "7. Qui connaîtra vos données ? Vos données seront-elles partagées ou publiées ?",
        "Les données de recherche pseudonymisées ne sont accessibles qu'à l'équipe d'étude de l'IBIMA. Le fichier de correspondance n'est accessible qu'à la direction de l'étude (Anton Kulaga et Livia Zaharia). Les résultats de l'étude sont publiés uniquement sous une forme agrégée et anonymisée. Il n'est pas possible de déduire des personnes individuelles à partir des publications.",
        "8. Quels sont vos droits en matière de protection des données lors de votre participation à l'étude ?",
        "Conformément aux réglementations applicables en matière de protection des données (RGPD), vous disposez d'un droit d'accès aux données stockées, de rectification des données inexactes, d'effacement des données, de limitation du traitement, d'opposition au traitement déraisonnable et du droit à la portabilité des données stockées à votre sujet. Vous avez également le droit de retirer votre consentement. Vous pouvez exercer ces droits à tout moment auprès de l'institution responsable.",
        "Droit de retrait",
        "En principe, vous pouvez retirer votre consentement à tout moment sans donner de raisons. Veuillez contacter la direction de l'étude (anton.kulaga@uni-rostock.de). Votre retrait prend effet à partir du moment où vous le soumettez. Le traitement de vos données jusqu'à ce point reste licite. En cas de retrait, la suppression des données personnelles sera initiée. Les données de l'étude qui ont déjà été anonymisées et ne peuvent donc plus être liées à vous continueront d'être utilisées dans le cadre de l'étude. Le fichier de correspondance devrait être détruit 12 mois après la fin de l'étude ; après cela, une attribution individuelle et donc une suppression ne sont plus possibles.",
        "Base juridique : Art. 6(1)(a) du RGPD et Art. 9(2)(a) du RGPD.",
        "Déclaration de consentement pour la participation à l'étude de prédiction du glucose Sugar-Sugar et le traitement des données associé",
        "Identifiant d'étude de la personnée participante : _________________________________ (attribué lors de l'inscription)",
        "Les consentements suivants sont enregistrés par voie électronique au moment de l'inscription en cochant les cases correspondantes.",
        "Je participe à l'étude volontairement. J'ai lu attentivement les informations pour les participants à l'étude et l'avis de protection des données inclus ainsi que les droits associés.",
        "J'ai été informé que des données personnelles seront collectées, traitées et utilisées dans le cadre de l'étude, en particulier mes données de santé (statut diabétique, historique d'utilisation du CGM, performances de prédiction).",
        "Comme indiqué dans les informations pour les participants, le traitement a lieu sous forme pseudonymisée par l'équipe d'étude de l'IBIMA, Université de médecine de Rostock. HEALES agit en tant que sous-traitant dans le cadre d'un DPA.",
        "Je suis conscient que si mes données sont stockées de manière anonyme, leur suppression à ma demande n'est plus possible.",
        "Droit de retrait",
        "J'ai été informé que je peux mettre fin à ma participation à tout moment sans donner de raisons, sans aucun préjudice pour moi. Je suis également conscient que je peux retirer mon consentement au traitement des données à tout moment avec effet pour l'avenir, sans affecter la licéité du traitement effectué sur la base du consentement jusqu'au retrait. J'ai pris note des informations sur le retrait du consentement dans les informations pour les participants correspondantes.",
        "Je déclare par la présente mon accord volontaire pour participer à l'étude de prédiction du glucose Sugar-Sugar. Je comprends que je peux mettre fin à ma participation à tout moment sans donner de raisons, sans aucun préjudice pour moi.",
        "Je consens par la présente, y compris aux informations pour les participants correspondantes, au traitement de mes données personnelles et de santé aux fins suivantes :",
        "☐ Collecte, traitement et utilisation de mes données énumérées dans les informations pour les participants, en particulier mes données de santé (statut diabétique, historique d'utilisation du CGM, performances de prédiction), sous forme pseudonymisée pour la conduite de l'étude par l'équipe d'étude de l'IBIMA, Université de médecine de Rostock, ainsi que HEALES en tant que sous-traitant. Le stockage a lieu sur des serveurs au sein de l'Union européenne. Les données de recherche sont conservées jusqu'à 10 ans après l'anonymisation du fichier de correspondance. Le retrait est possible jusqu'à 12 mois après la fin de l'étude ; après cela, l'identification n'est plus possible. Base juridique : Art. 6(1)(a) du RGPD et Art. 9(2)(a) du RGPD.",
        "☐ La personne participante confirme être âgée de 18 ans ou plus.",
        "Consentements facultatifs (le refus n'a aucune influence sur la participation à l'étude) :",
        "☐ Je consens au téléchargement de mes propres données CGM pour la tâche de prédiction facultative avec mes propres données. Les données brutes seront supprimées dans les 30 jours ; seuls les segments extraits pseudonymisés seront conservés.",
        "☐ J'accepte d'être informé par l'équipe d'étude via l'adresse e-mail enregistrée des résultats de l'étude après publication.",
        "Les notifications de HEALES ou de projets connexes seront présentées sur une page séparée après la fin de l'étude et sont complètement indépendantes de la participation à l'étude et du consentement.",
        "Cette étude a reçu l'approbation du comité d'éthique de l'University Medicine Rostock. Numéro d'enregistrement : Réf. number A 2026-0064. Reçu par le comité d'éthique : 27 février 2026.",
        "Questions ou demandes de retrait : anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
    "es": [
        "Información para participantes",
        "Predicción de trayectorias de glucosa basada en patrones de valores previos: un estudio de benchmarking en línea",
        "Título corto: Estudio de predicción de glucosa Sugar-Sugar",
        "Lea atentamente el siguiente texto antes de tomar una decisión. Si tiene preguntas, póngase en contacto con la dirección del estudio.",
        "Estimado participante,",
        "nos gustaría preguntarle si desea participar en un estudio científico. Este estudio tiene como objetivo investigar sistemáticamente, por primera vez, con qué precisión pueden los humanos predecir las trayectorias de glucosa a corto plazo basándose en los datos de CGM (datos de un dispositivo de monitorización continua de glucosa). Hasta ahora, no existe tal punto de referencia. Su participación ayuda a establecer un valor de referencia para la precisión de la predicción humana, que también puede utilizarse a largo plazo para evaluar los modelos de predicción basados en computadora. Tales modelos de predicción se utilizan cada vez más en la investigación y el cuidado de la diabetes y pueden validarse y compararse utilizando los datos recopilados aquí.",
        "El estudio se lleva a cabo en el IBIMA (Instituto de Bioestadística e Informática en Medicina e Investigación sobre el Envejecimiento) del Centro Médico Universitario de Rostock, en colaboración con HEALES (Healthy Life Extension Society, Bruselas), que actúa como financiador del estudio. La dirección del estudio está a cargo de Anton Kulaga (Centro Médico Universitario de Rostock) y Livia Zaharia (HEALES). Se han previsto al menos 200 participantes.",
        "La participación en el estudio no tiene ninguna influencia en ningún tratamiento médico. Se trata de un estudio puramente en línea sin citas presenciales ni procedimientos médicos.",
        "¿Cómo funciona el estudio?",
        "La participación se realiza íntegramente en línea a través de la aplicación web Sugar-Sugar y dura unos 15-20 minutos.",
        "Primero, completará un breve cuestionario de registro (edad, género, país de residencia, estado de diabetes, historial de uso de CGM; opcional: peso y altura). Luego completará entre 6 y 12 tareas de predicción. Cada tarea muestra tres horas de datos históricos de CGM de un conjunto de datos anónimo y públicamente conocido. Se le mostrarán los primeros puntos de series temporales recopiladas en otros estudios; se le pedirá que adivine los puntos de tiempo posteriores. Dibujará la trayectoria de glucosa esperada para los siguientes 60 minutos colocando 12 puntos en un gráfico. Antes de las tareas calificadas, se le ofrecerán dos rondas de práctica opcionales.",
        "Después de cada tarea, se muestran los valores reales de glucosa y se calculan las métricas de precisión. Al final, recibirá un resumen de su precisión de predicción (determinada a partir de los datos históricos), así como un rango percentil.",
        "La precisión de la predicción se compara en cuatro grupos: personas con diabetes y uso de CGM, personas con diabetes sin uso de CGM, personas sin diabetes con uso de CGM y personas sin diabetes sin experiencia con CGM; según los datos recopilados al principio, se le asignará a uno de estos grupos.",
        "Carga opcional de sus propios datos de CGM",
        "Si usted mismo utiliza un dispositivo CGM, opcionalmente puede cargar un archivo de exportación (CSV o JSON). Se extraen segmentos cortos del mismo y se utilizan para tareas de predicción adicionales. Esto permite comparar si predice sus propios datos con mayor precisión que los datos históricos. La fecha y la hora no se muestran en la interfaz de usuario. El archivo original cargado se elimina en un plazo de 30 días; solo se conservan los segmentos extraídos de forma seudonimizada.",
        "Participación voluntaria",
        "La participación en este estudio es voluntaria. Solo se le incluirá si declara su consentimiento. Si no participa en el estudio o desea retirarse más tarde, no sufrirá ninguna desventaja. Puede finalizar la participación en cualquier momento sin dar razones. Los datos parciales de las rondas completadas solo se incluyen si se han completado al menos 6 rondas válidas.",
        "¿Quién puede participar?",
        "Pueden participar adultos de 18 años o más con acceso a Internet. Son elegibles tanto las personas con diabetes como las que no la padecen.",
        "¿Qué posibles riesgos, molestias o efectos secundarios están asociados con su participación?",
        "Dado que nuestro estudio solo recopila datos, no existen riesgos médicos asociados con la participación. Los valores de precisión son relevantes exclusivamente para la investigación y no tienen ningún significado diagnóstico o clínico.",
        "¿Qué posibles beneficios resultan de su participación en el estudio?",
        "No hay ningún beneficio personal para su salud por participar. La participación contribuye a un punto de referencia sistemático de la capacidad humana de predicción de los valores de glucosa. Recibirá un resumen de precisión personal, así como una tarjeta de resultados. No hay compensación monetaria.",
        "¿Qué derechos y condiciones están asociados con la participación?",
        "Puede finalizar su participación en el estudio en cualquier momento, incluso sin dar razones. Si no desea participar o desea retirarse, no sufrirá ninguna desventaja.",
        "Tenga en cuenta que no se ha contratado ningún seguro por separado para este proyecto y el estudio que lo acompaña, ya que se trata de un estudio puramente en línea sin procedimientos médicos y, por lo tanto, conlleva un riesgo mínimo.",
        "El estudio ha recibido la aprobación del comité de ética responsable.",
        "Protección de datos",
        "A continuación, nos gustaría informarle exhaustivamente sobre lo que sucede con sus datos personales dentro del estudio y cómo los manejamos.",
        "1. ¿Quién es responsable del procesamiento y almacenamiento de datos?",
        "El Centro Médico Universitario de Rostock es responsable del procesamiento y almacenamiento de datos dentro del estudio. HEALES opera la aplicación Sugar-Sugar como procesador en el marco de un acuerdo de procesamiento de datos (DPA) con el Centro Médico Universitario de Rostock. Puede ponerse en contacto con la institución responsable en los siguientes datos de contacto:",
        "Institución responsable: Centro Médico Universitario de Rostock – subentidad legalmente capaz de la Universidad de Rostock",
        "Schillingallee 35, 18057 Rostock",
        "www.med.uni-rostock.de",
        "2. ¿Con quién puede contactar con preguntas sobre el estudio o la protección de datos?",
        "Para salvaguardar mejor sus intereses, también puede ponerse en contacto con las siguientes personas en cualquier momento con cualquier pregunta sobre su participación en el estudio o el procedimiento general:",
        "Contactos del estudio:",
        "• Anton Kulaga: anton.kulaga@uni-rostock.de",
        "• Livia Zaharia: livia.zaharia@uni-rostock.de",
        "Para preguntas sobre protección de datos, también está disponible el oficial de protección de datos del Centro Médico Universitario de Rostock:",
        "Doberaner Str. 142, 18057 Rostock",
        "0381 494 5155 | datenschutz@med.uni-rostock.de",
        "Si desea ejercer su derecho a reclamar sobre un procesamiento de datos ilegal, póngase en contacto con la autoridad de supervisión competente:",
        "Comisionado Estatal de Protección de Datos y Libertad de Información M-V",
        "Schloss Schwerin, Lennéstr. 1, 19053 Schwerin",
        "info@datenschutz-mv.de",
        "3. ¿Qué datos necesitamos de usted para realizar el estudio?",
        "Los siguientes datos se recopilan dentro del estudio:",
        "Datos de contacto: dirección de correo electrónico (para el registro, el manejo de la retirada y la renotificación opcional).",
        "Datos demográficos: edad, género, país de residencia; opcional: peso y altura.",
        "Datos relacionados con la salud: estado de diabetes (sí/no, tipo, duración), historial de uso de CGM.",
        "Datos del estudio: valores de glucosa predichos y, si corresponde, reales, métricas de precisión (MAE, MSE, RMSE, MAPE).",
        "Preferencias de consentimiento: documentación de sus decisiones de consentimiento.",
        "Opcional – datos propios de CGM: si se cargan, solo se conservan los segmentos extraídos de forma seudonimizada.",
        "4. ¿De dónde obtenemos estos datos?",
        "Todos los datos se recopilan exclusivamente de forma directa de usted como participante, a través de sus entradas en la aplicación web Sugar-Sugar (cuestionario de registro, tareas de predicción, carga opcional de CGM). No se obtienen datos de otras fuentes. La base legal para el procesamiento es su consentimiento explícito de conformidad con el Art. 6(1)(a) del GDPR y el Art. 9(2)(a) del GDPR.",
        "5. ¿Para qué fines necesitamos sus datos y cómo se protegen?",
        "Los datos enumerados anteriormente se necesitan exclusivamente para la investigación científica de la capacidad humana de predicción de glucosa y se utilizan solo para este propósito.",
        "Cada persona participante recibe una ID de estudio aleatoria durante el registro. Todos los datos de investigación están vinculados exclusivamente a esta ID. Un archivo de mapeo encriptado separado que vincula la ID del estudio con la dirección de correo electrónico se almacena por separado y es accesible solo para la dirección del estudio (Anton Kulaga y Livia Zaharia). Este archivo se utiliza exclusivamente para manejar las solicitudes de retirada y las notificaciones de resultados opcionales. Para el resto del personal del estudio, por lo tanto, no es posible identificarle.",
        "Los datos se encriptan durante la transmisión y en reposo. El almacenamiento se realiza en servidores dentro de la Unión Europea (alojamiento que cumple con el GDPR). El acceso está regulado en función de los roles.",
        "6. ¿Dónde y durante cuánto tiempo se almacenan los datos?",
        "Los datos recopilados dentro del estudio se seudonimizan primero. Se espera que el archivo de mapeo (correo electrónico – ID del estudio) se destruya 12 meses después de la finalización del estudio. Después de eso, los datos del estudio se anonimizan, es decir, ya no permiten ninguna inferencia sobre su identidad.",
        "Los datos de investigación seudonimizados se conservan durante 10 años de acuerdo con los estándares de investigación alemanes. Los archivos de CGM originales cargados se eliminan en un plazo de 30 días.",
        "7. ¿Quién conocerá sus datos? ¿Se compartirán o publicarán sus datos?",
        "Los datos de investigación seudonimizados son accesibles solo para el equipo del estudio en IBIMA. El archivo de mapeo es accesible solo para la dirección del estudio (Anton Kulaga y Livia Zaharia). Los resultados del estudio se publican solo de forma agregada y anónima. No es posible inferir personas individuales a partir de las publicaciones.",
        "8. ¿Qué derechos de protección de datos tiene al participar en el estudio?",
        "De acuerdo con las regulaciones de protección de datos aplicables (GDPR), tiene derecho a acceder a los datos almacenados, rectificar los datos inexactos, borrar los datos, restringir el procesamiento, oponerse al procesamiento irrazonable y el derecho a la portabilidad de los datos almacenados sobre usted. También tiene derecho a retirar su consentimiento. Puede ejercer estos derechos en cualquier momento ante la institución responsable.",
        "Derecho de retirada",
        "En principio, puede retirar su consentimiento en cualquier momento sin dar razones. Póngase en contacto con la dirección del estudio (anton.kulaga@uni-rostock.de). Su retirada surte efecto desde el momento en que la envía. El procesamiento de sus datos hasta ese momento sigue siendo legal. En caso de retirada, se iniciará la eliminación de los datos personales. Los datos del estudio que ya hayan sido anonimizados y, por lo tanto, ya no puedan vincularse a usted, se seguirán utilizando dentro del estudio. Se espera que el archivo de mapeo se destruya 12 meses después de la finalización del estudio; después de eso, ya no es posible una asignación individual y, por lo tanto, la eliminación.",
        "Base legal: Art. 6(1)(a) del GDPR y Art. 9(2)(a) del GDPR.",
        "Declaración de consentimiento para la participación en el estudio de predicción de glucosa Sugar-Sugar y el procesamiento de datos asociado",
        "ID del estudio de la persona participante: _________________________________ (asignado al registrarse)",
        "Los siguientes consentimientos se registran electrónicamente en el momento del registro marcando las casillas correspondientes.",
        "Participo en el estudio voluntariamente. He leído atentamente la información para los participantes del estudio y el aviso de protección de datos incluido y los derechos asociados.",
        "Se me ha informado que se recopilarán, procesarán y utilizarán datos personales dentro del estudio, en particular mis datos de salud (estado de diabetes, historial de uso de CGM, rendimiento de predicción).",
        "Como se indica en la información para los participantes, el procesamiento se realiza de forma seudonimizada por el equipo del estudio en IBIMA, Centro Médico Universitario de Rostock. HEALES actúa como procesador bajo un DPA.",
        "Soy consciente de que si mis datos se almacenan de forma anónima, ya no es posible eliminarlos a mi petición.",
        "Derecho de retirada",
        "Se me ha informado que puedo finalizar mi participación en cualquier momento sin dar razones, sin ninguna desventaja para mí. También soy consciente de que puedo retirar mi consentimiento para el procesamiento de datos en cualquier momento con efecto para el futuro, sin afectar la legalidad del procesamiento llevado a cabo sobre la base del consentimiento hasta la retirada. He tomado nota de la información sobre la retirada del consentimiento en la información para los participantes correspondiente.",
        "Por la presente declaro mi acuerdo voluntario para participar en el estudio de predicción de glucosa Sugar-Sugar. Entiendo que puedo finalizar mi participación en cualquier momento sin dar razones, sin ninguna desventaja para mí.",
        "Por la presente doy mi consentimiento, incluida la información para los participantes correspondiente, para el procesamiento de mi datos personales y de salud para los siguientes fines:",
        "☐ Recopilación, procesamiento y uso de mis datos enumerados en la información para los participantes, en particular mis datos de salud (estado de diabetes, historial de uso de CGM, rendimiento de predicción), en forma seudonimizada para la realización del estudio por parte del equipo del estudio en IBIMA, Centro Médico Universitario de Rostock, así como HEALES como procesador. El almacenamiento se realiza en servidores dentro de la Unión Europea. Los datos de investigación se conservan hasta 10 años después de la anonimización del archivo de mapeo. La retirada es posible hasta 12 meses después de la finalización del estudio; después ya no es posible la identificación. Base legal: Art. 6(1)(a) del GDPR y Art. 9(2)(a) del GDPR.",
        "☐ La persona participante confirma tener 18 años de edad o más.",
        "Consentimientos opcionales (la negativa no influye en la participación en el estudio):",
        "☐ Doy mi consentimiento para cargar mis propios datos de CGM para la tarea de predicción opcional con mis propios datos. Los datos originales se eliminarán en un plazo de 30 días; solo se conservarán los segmentos extraídos seudonimizados.",
        "☐ Acepto que el equipo del estudio me informe a través de la dirección de correo electrónico registrada sobre los resultados del estudio después de su publicación.",
        "Las notificaciones de HEALES o proyectos relacionados se presentarán en una página separada después de que finalice el estudio y son completamente independientes de la participación en el estudio y el consentimiento.",
        "Este estudio ha recibido la aprobación del Comité de Ética del Centro Médico Universitario de Rostock. Número de registro: Ref. number A 2026-0064. Recibido por el Comité de Ética: 27 de febrero de 2026.",
        "Preguntas o solicitudes de retirada: anton.kulaga@uni-rostock.de | livia.zaharia@uni-rostock.de",
    ],
}


def consent_notice_paragraphs(locale: Optional[str]) -> list[str]:
    # Backwards-compatible API used by older callers.
    # We keep this so existing code can still render non-German locales,
    # even though German now comes from markdown.
    loc = normalize_locale(locale)
    if loc == "de":
        md = consent_notice_markdown_de()
        return [x.strip() for x in md.splitlines() if x.strip()]
    translated = _CONSENT_NOTICE_TRANSLATIONS.get(loc)
    if translated is None:
        md = consent_notice_markdown_de()
        return [x.strip() for x in md.splitlines() if x.strip()]
    return translated


def consent_notice_children(locale: Optional[str], theme: str = "light") -> list[object]:
    loc = normalize_locale(locale)

    # Prefer locale-specific markdown if it exists.
    md_candidates = []
    base = _consent_notice_md_path()
    if len(base.suffixes) >= 2 and base.suffixes[-2:] == [".docx", ".md"]:
        md_candidates.append(base.with_suffix(f".{loc}.md"))
    md_candidates.append(base.with_name(f"patient_consent_form.{loc}.md"))
    md_candidates.append(base.with_name(f"patient_consent_form_{loc}.md"))

    md_text = ""
    for p in md_candidates:
        md_text = _read_markdown_file(p)
        if md_text:
            break

    if not md_text:
        if loc == "de":
            md_text = consent_notice_markdown_de()
        else:
            # Fallback: join translated paragraphs into a readable markdown block.
            translated = _CONSENT_NOTICE_TRANSLATIONS.get(loc)
            if translated:
                md_text = _sanitize_markdown("\n\n".join(translated))
            else:
                md_text = consent_notice_markdown_de()

    # Style subtitles by exact-line matching (avoid partial replacements like "Data protection officer...").
    replacements = {
        "en": {
            "Participant information": "### Participant information",
            "Data protection": "### Data protection",
            "Right of withdrawal": "### Right of withdrawal",
        },
        "uk": {
            "Інформація для учасників": "### Інформація для учасників",
            "Захист даних": "### Захист даних",
            "Право на відкликання": "### Право на відкликання",
        },
        "ro": {
            "Informații pentru participanți": "### Informații pentru participanți",
            "Protecția datelor": "### Protecția datelor",
            "Dreptul de retragere": "### Dreptul de retragere",
        },
        "ru": {
            "Информация для участников": "### Информация для участников",
            "Защита данных": "### Защита данных",
            "Право на отказ": "### Право на отказ",
        },
        "zh": {
            "参与者信息": "### 参与者信息",
            "数据保护": "### 数据保护",
            "撤回权": "### 撤回权",
        },
        "fr": {
            "Informations pour les participants": "### Informations pour les participants",
            "Protection des données": "### Protection des données",
            "Droit de retrait": "### Droit de retrait",
        },
        "es": {
            "Información para participantes": "### Información para participantes",
            "Protección de datos": "### Protección de datos",
            "Derecho de retirada": "### Derecho de retirada",
        },
        "de": {
            "**Teilnehmerinformation**": "### Teilnehmerinformation",
            "## Datenschutz": "### Datenschutz",
            "## Widerrufsrecht": "### Widerrufsrecht",
        },
    }
    if loc in replacements:
        for old, new in replacements[loc].items():
            escaped_old = re.escape(old)
            md_text = re.sub(rf"(?m)^{escaped_old}\s*$", new, md_text)

    # Remove accidental heading styling from data protection officer line(s).
    officer_lines = [
        "Data protection officer of University Medicine Rostock",
        "Datenschutzbeauftragter der Universitätsmedizin Rostock",
        "Уповноважений із захисту даних Університетської медицини Ростока",
    ]
    for line in officer_lines:
        md_text = re.sub(rf"(?m)^###\s*{re.escape(line)}\s*$", line, md_text)

    # Make numbered lines bold
    md_text = re.sub(r"(\d+)\. ", r"**\1.** ", md_text)

    # Indent text after numbered lines until the next numbered line or heading.
    paragraphs = md_text.split("\n\n")
    out_text = ""
    first = True
    indent = False
    after_numbered = False

    for p in paragraphs:
        stripped = p.strip()
        is_heading = stripped.startswith("###")
        is_numbered = re.match(r"\*\*\d+\.\*\*", stripped)

        if is_heading:
            if not first:
                out_text += "\n\n"
            out_text += p
            first = False
            indent = False
            after_numbered = False
            continue

        if is_numbered:
            if not first:
                out_text += "\n\n"
            out_text += p
            first = False
            indent = True
            after_numbered = True
            continue

        if indent:
            # No blank line under numbered bullets; daughter paragraphs are indented directly.
            out_text += "\n" + f'<div style="margin-left: 20px;">{p}</div>'
            after_numbered = False
            continue

        # Standard paragraph spacing
        if not first:
            out_text += "\n\n"
        out_text += p
        first = False
        after_numbered = False

    md_text = out_text

    return [
        static_markdown_iframe(
            md_text,
            title=t("ui.landing.patient_consent_form_title", locale=loc),
            iframe_style={"height": "min(55vh, 480px)"},
            theme=theme,
        )
    ]

