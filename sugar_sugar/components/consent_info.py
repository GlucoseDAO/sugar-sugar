from pathlib import Path
from typing import List

from dash import html

TITLES = {
    # English
    "Participant Information",
    "Predicting Glucose Trends Based on Prior Value Patterns: An Online Benchmarking Study",
    "Short title: Sugar-Sugar Glucose Forecasting Study",
    "How does the study work?",
    "Optional upload of your own CGM data",
    "Voluntary participation",
    "Who can participate?",
    "What possible risks, discomforts, or side effects are associated with participation?",
    "What potential benefits does participation offer?",
    "What rights and conditions are associated with participation?",
    "Data Protection",
    "1. Who is responsible for data processing and storage?",
    "Responsible Institution",
    "2. Who can you contact with questions about the study or data protection?",
    "Study contacts",
    "Data Protection Officer, Rostock University Medical Center",
    "Competent data protection supervisory authority",
    "3. What data do we need from you for the study?",
    "4. Where does this data come from?",
    "5. For what purposes is your data needed, and how is it protected?",
    "6. Where and for how long is data stored?",
    "7. Who has access to your data? Will your data be shared or published?",
    "8. What data protection rights do you have when participating in the study?",
    "Right to withdraw",
    "Declaration of Consent",
    "for participation in the Sugar-Sugar Glucose Forecasting Study",
    "and the associated data processing",
    # German
    "Teilnehmerinformation",
    "Vorhersage des Glukoseverlaufs anhand vorheriger Wertemuster: Eine Online-Benchmarking-Studie",
    "Kurztitel: Sugar-Sugar Glukoseprognose-Studie",
    "Wie läuft die Studie ab?",
    "Optionaler Upload eigener CGM-Daten",
    "Freiwilligkeit der Teilnahme",
    "Wer kann teilnehmen?",
    "Welche möglichen Risiken, Beschwerden oder Begleiterscheinungen sind mit Ihrer Teilnahme verbunden?",
    "Welcher mögliche Nutzen ergibt sich aus Ihrer Teilnahme an der Studie?",
    "Welche Rechte und Bedingungen sind mit der Teilnahme verbunden?",
    "Datenschutz",
    "1. Wer ist verantwortlich für die Datenverarbeitung und -speicherung?",
    "Verantwortliche Einrichtung",
    "2. An wen können Sie sich bei Fragen zur Studie oder zum Datenschutz wenden?",
    "Ansprechpartner zur Studie",
    "Datenschutzbeauftragter der Universitätsmedizin Rostock",
    "Zuständige Datenschutzaufsichtsbehörde",
    "3. Welche Daten von Ihnen benötigen wir für die Durchführung der Studie?",
    "4. Woher erhalten wir diese Daten?",
    "5. Zu welchen Zwecken werden Ihre Daten benötigt und wie werden sie geschützt?",
    "6. Wo und wie lange werden die Daten gespeichert?",
    "7. Wer erlangt Kenntnis von Ihren Daten? Werden Ihre Daten weitergegeben und veröffentlicht?",
    "8. Welche Datenschutzrechte haben Sie bei einer Teilnahme an der Studie?",
    "Widerrufsrecht",
    "Einwilligungserklärung",
    "zur Teilnahme an der Studie Sugar-Sugar Glukoseprognose-Studie",
    "und die damit verbundene Datenverarbeitung",
}

def get_consent_info_components(locale: str) -> List[html.Component]:
    project_root = Path(__file__).resolve().parents[2]
    
    # Fallback to English for unsupported locales
    if locale == "de":
        filepath = project_root / "partinfodeutsch.txt"
    else:
        filepath = project_root / "partinfoenglish.txt"
        
    if not filepath.exists():
        return [html.Div(f"Missing {filepath.name}")]
        
    with open(filepath, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    components = []
    
    for l in lines:
        is_data_protection = l in ["Data Protection", "Datenschutz"]
        is_title = l in TITLES
        
        if is_data_protection:
            components.append(
                html.H3(
                    l,
                    style={"fontSize": "26px", "fontWeight": "900", "color": "#0f172a", "marginTop": "24px", "marginBottom": "12px"},
                )
            )
        elif is_title:
            components.append(
                html.H4(
                    l,
                    style={"fontSize": "20px", "fontWeight": "800", "color": "#0f172a", "marginTop": "18px", "marginBottom": "8px"},
                )
            )
        else:
            components.append(
                html.Div(
                    l,
                    style={"color": "#334155", "lineHeight": "1.6", "marginBottom": "8px"},
                )
            )
            
    return components
