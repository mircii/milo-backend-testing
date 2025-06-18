# 05_backend/fallbackHandler.py

# Fallback logică structurată pe categorii
fallback_config = {
    "volunteering": {
        "tags": [
            "volunteering_activities", "volunteering_enrollment", "volunteering_contact",
            "volunteering_eligibility", "volunteering_certificate", "volunteering_credits"
        ],
        "message": "Vrei informații despre voluntariat? Alege una dintre opțiuni:",
        "options": [
            {"label": "Activități disponibile", "query": "Ce activități de voluntariat există?"},
            {"label": "Cum te înscrii", "query": "Cum mă pot înscrie ca voluntar?"},
            {"label": "Eligibilitate", "query": "Cine poate deveni voluntar?"},
            {"label": "Certificat", "query": "Pot primi certificat pentru voluntariat?"},
            {"label": "Contact", "query": "Cu cine pot lua legătura pentru voluntariat?"},
            {"label": "Credite", "query": "Voluntariatul oferă credite?"}
        ]
    },
    "psychological": {
        "tags": [
            "psychological_counseling_topics", "psychological_counseling_eligibility",
            "psychological_counseling_definition", "psychological_counseling_appointment",
            "psychological_counseling_cost", "psychological_counseling_benefits",
            "psychological_counseling_session_duration", "psychological_counseling_session_count",
            "psychological_counseling_location"
        ],
        "message": "Ai nevoie de informații despre consiliere psihologică? Alege o temă:",
        "options": [
            {"label": "Definiție", "query": "Ce este consilierea psihologica?"},
            {"label": "Eligibilitate", "query": "Cine poate beneficia de consiliere psihologică?"},
            {"label": "Durată", "query": "Cât durează o ședință de consiliere psihologică?"},
            {"label": "Număr ședințe", "query": "De câte ședințe am nevoie?"},
            {"label": "Cost", "query": "Cât costă consilierea psihologică?"},
            {"label": "Programare", "query": "Cum pot face o programare?"}
        ]
    },
    "career": {
        "tags": ["career_counseling_cost", "career_counseling_general_info", "career_counseling_scheduling"],
        "message": "Cauți informații despre consiliere în carieră?",
        "options": [
            {"label": "Informații generale", "query": "Ce presupune consilierea în carieră?"},
            {"label": "Costuri", "query": "Cât costă consilierea în carieră?"},
            {"label": "Programare", "query": "Cum fac o programare pentru consiliere în carieră?"}
        ]
    }
}

def getFallbackForTag(tag):
    for category, config in fallback_config.items():
        if tag in config["tags"]:
            return {
                "type": "fallback_options",
                "tag": category,
                "message": config["message"],
                "options": config["options"]
            }
    return None