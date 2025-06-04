"""
Constants used across the application
"""

EQ5D_DIMENSIONS = ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]

COCHLEAR_IMPLANT_CATEGORIES = {
    "social": [
        "social_activities",
        "social_confidence",
        "social_relationships",
        "social_communication"
    ],
    "academic": [
        "academic_performance",
        "academic_confidence",
        "academic_engagement",
        "academic_communication",
        "academic_focus"
    ],
    "fatigue": [
        "listening_fatigue",
        "mental_fatigue"
    ]
} 

# Références possibles par type de questionnaire
REFERENCE_PROFESSIONALS = {
    "OKS": ["Physiothérapeute", "Kinésithérapeute", "Ergothérapeute"],
    "EQ-5D": ["Infirmier spécialisé", "Psychologue", "Travailleur social"],
    "Cochlear Implant": ["Audiologiste", "Orthophoniste", "Psychologue"]
}

# Texte des questions EQ-5D pour enrichir le prompt détaillé
EQ5D_QUESTIONS = {
    "mobility": "1. MOBILITÉ : Veuillez cocher la case qui décrit le mieux votre niveau de mobilité aujourd'hui.",
    "self_care": "2. AUTONOMIE DE LA PERSONNE : Veuillez cocher la case qui décrit le mieux vos problèmes en matière de soins auto-administrés aujourd'hui.",
    "usual_activities": "3. ACTIVITÉS COURANTES : Veuillez cocher la case qui décrit le mieux vos problèmes à accomplir vos activités courantes aujourd'hui.",
    "pain_discomfort": "4. DOULEURS/GÊNE : Veuillez cocher la case qui décrit le mieux votre niveau de douleurs/gêne aujourd'hui.",
    "anxiety_depression": "5. ANXIÉTÉ/DÉPRESSION : Veuillez cocher la case qui décrit le mieux votre niveau d'anxiété/dépression aujourd'hui."
}

# Texte des questions pour le questionnaire sur les implants cochléaires
COCHLEAR_IMPLANT_QUESTIONS = {
    "practice_1": "À quelle fréquence vas-tu sur la lune?",
    "practice_2": "À quel point est-il difficile pour toi de soulever ces livres?",
    "noisy_1": "À quel point est-il difficile d'entendre les gens à la cafétéria ou dans le local où tu manges le midi?",
    "noisy_2": "À quel point est-il difficile pour toi de comprendre les paroles d'une chanson?",
    "noisy_3": "À quel point est-il difficile pour toi de comprendre lorsque tu écoutes la télévision ou des films?",
    "noisy_4": "À quel point est-il difficile pour toi d'entendre les autres quand tu vas manger au restaurant?",
    "academic_1": "Utilises-tu un micro à l'école?",
    "academic_2": "À quel point est-il difficile pour toi d'entendre en classe?",
    "academic_3": "À quel point la lecture est-elle difficile pour toi à l'école?",
    "academic_4": "À quel point est-il difficile pour toi de rester attentif à l'école?",
    "academic_5": "À quel point est-il difficile pour toi de terminer tes devoirs seul(e)?",
    "oral_1": "À quel point est-il difficile pour les gens de comprendre ce que tu dis?",
    "oral_2": "À quel point est-il difficile de comprendre ce qui se passe dans un groupe?",
    "oral_3": "À quel point est-il difficile pour toi de comprendre quelqu'un qui te parle?",
    "fatigue_1": "À quelle fréquence enlèves-tu ton implant cochléaire pour prendre une pause?",
    "fatigue_2": "À quel point te sens-tu fatigué(e) après avoir écouté longtemps?",
    "social_1": "Combien d'amis as-tu?",
    "social_2": "À quel point es-tu à l'aise de te faire de nouveaux amis?",
    "social_3": "À quelle fréquence les gens se moquent-ils de ton implant cochléaire?",
    "social_4": "À quelle fréquence te retires-tu lorsque tu es avec d'autres enfants?",
    "emotional_1": "À quelle fréquence t'es-tu senti(e) heureux(se) la semaine dernière?",
    "emotional_2": "À quelle fréquence t'es-tu senti(e) triste la semaine dernière?",
    "emotional_3": "À quelle fréquence t'es-tu senti(e) de mauvaise humeur la semaine dernière?",
    "emotional_4": "À quelle fréquence t'es-tu senti(e) en colère la semaine dernière?"
}

# Signification des scores pour le questionnaire sur les implants cochléaires
COCHLEAR_IMPLANT_SCORES_MEANING = """
Échelle de scores pour les questions de difficulté (pas difficile, un peu difficile, difficile, très difficile):
- 1 = Pas difficile
- 2 = Un peu difficile
- 3 = Difficile
- 4 = Très difficile

Échelle de scores pour les questions de fréquence (toujours, souvent, parfois, jamais):
- 1 = Toujours
- 2 = Souvent
- 3 = Parfois
- 4 = Jamais

Échelle de scores pour le nombre d'amis:
- 1 = Aucun ami
- 2 = Quelques amis
- 3 = Des amis
- 4 = Beaucoup d'amis

Échelle de scores pour le confort social:
- 1 = Très à l'aise
- 2 = À l'aise
- 3 = Un peu à l'aise
- 4 = Pas à l'aise

Échelle de scores pour la fatigue:
- 1 = Pas fatigué(e)
- 2 = Un peu fatigué(e)
- 3 = Fatigué(e)
- 4 = Très fatigué(e)
"""

# Signification des scores pour chaque dimension EQ-5D (échelle de 1 à 5)
EQ5D_SCORES_MEANING = """
Échelle de scores pour chaque dimension (plus le score est bas, meilleur est l'état de santé):
- 1 = Aucun problème
- 2 = Problèmes légers
- 3 = Problèmes modérés
- 4 = Problèmes sévères
- 5 = Problèmes extrêmes/Incapacité complète

Pour l'échelle visuelle analogique (EVA):
- Score de 0 à 100, où 0 représente le pire état de santé imaginable et 100 le meilleur.
"""