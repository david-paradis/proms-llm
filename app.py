import streamlit as st
import pandas as pd
import os
from datetime import datetime
import base64

from google import genai
from google.genai import types

from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Titre de l'application
st.set_page_config(page_title="Assistant d'Interprétation de PROMs via LLM", layout="wide")
st.title("Assistant d'Interprétation de PROMs via LLM")

# Textes des questions OHS pour enrichir le prompt détaillé
OHS_QUESTIONS = {
    "q1": "1. À quel point avez-vous eu mal à la hanche durant les 4 dernières semaines?",
    "q2": "2. Avez-vous eu des difficultés à vous laver et à vous sécher (tout le corps) à cause de votre hanche durant les 4 dernières semaines?",
    "q3": "3. Avez-vous eu des difficultés à monter ou descendre d'une voiture ou à utiliser les transports en commun à cause de votre hanche durant les 4 dernières semaines?",
    "q4": "4. Avez-vous été capable de mettre vos chaussettes, collants ou bas?",
    "q5": "5. Avez-vous pu faire vos courses ménagères tout(e) seul(e)?",
    "q6": "6. Pendant combien de temps avez-vous pu marcher avant que la douleur de hanche ne devienne trop intense (avec ou sans canne)?",
    "q7": "7. Avez-vous pu monter un étage d'escalier?",
    "q8": "8. Pendant les 4 dernières semaines, après une activité en position assise (repas, voyage dans un train ou une voiture...), est-ce que vous avez souffert d'une douleur soudaine et forte en vous mettant debout?",
    "q9": "9. Avez-vous boité en marchant, à cause de votre hanche?",
    "q10": "10. Avez-vous ressenti des douleurs brutales ou lancinantes au niveau de la hanche affectée?",
    "q11": "11. Est-ce que vos douleurs de hanche vous ont gêné dans votre sommeil la nuit dernière?",
    "q12": "12. Est-ce que la douleur à la hanche vous a gêné dans votre travail habituel (y compris les tâches ménagères)?"
}

# Signification des scores pour chaque question OHS (échelle de 0 à 4)
OHS_SCORES_MEANING = """
Échelle de scores pour chaque question:
- 0 = Douleur/Difficulté sévère ou incapacité
- 1 = Douleur/Difficulté importante 
- 2 = Douleur/Difficulté modérée
- 3 = Peu de douleur/difficulté
- 4 = Aucune douleur/difficulté
"""

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

# Fonctions utilitaires
def load_data():
    """Charge les données depuis le CSV ou génère des données si le fichier n'existe pas"""
    csv_path = "patients_proms.csv"
    if not os.path.exists(csv_path):
        from data_generator import generate_synthetic_data
        df = generate_synthetic_data(file_path=csv_path)
    else:
        df = pd.read_csv(csv_path)
    return df

def get_patient_data(df, patient_id, questionnaire_type=None):
    """Récupère les données d'un patient spécifique, filtré par type de questionnaire si spécifié"""
    patient_data = df[df['patient_id'] == patient_id].copy()
    
    if questionnaire_type:
        patient_data = patient_data[patient_data['questionnaire_type'] == questionnaire_type]
    
    patient_data['date_collecte'] = pd.to_datetime(patient_data['date_collecte'])
    patient_data = patient_data.sort_values('date_collecte')
    return patient_data

def construct_prompt(patient_data, interpretation_guide, questionnaire_type):
    """Construit le prompt pour le LLM basé sur les données du patient et les directives d'interprétation"""
    patient_id = patient_data['patient_id'].iloc[0]
    date_operation = patient_data['date_operation'].iloc[0]
    
    # Récupérer les références pour ce type de questionnaire
    references = patient_data[patient_data['reference_professional'].notna()].copy()
    references_text = ""
    if not references.empty:
        references_text = "\nRéférences à d'autres professionnels de la santé:\n"
        for _, ref in references.iterrows():
            references_text += f"- {ref['reference_professional']} le {ref['reference_date']}\n"
    
    # Formatage des données pour le prompt
    if questionnaire_type == "OHS":
        scores_data = [(row['date_collecte'].strftime('%Y-%m-%d'), row['score_total'], row['periode']) 
                       for _, row in patient_data.iterrows()]
        
        prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir un résumé clair et concis pour un clinicien.

Contexte Clinique:
{interpretation_guide}

Date d'opération: {date_operation}
{references_text}

Données du Patient:
Voici les scores OHS pour le patient {patient_id} :
{scores_data}

Tâche:
Analyse ces scores en te basant sur les directives fournies. Rédige un résumé (max 3-4 phrases) mettant en lumière : 
1. Le statut actuel du patient selon le dernier score
2. L'évolution globale depuis la première mesure
3. La comparaison des scores pré et post-opératoires
4. L'impact potentiel des références à d'autres professionnels sur l'évolution des scores
5. Tout changement notable entre les deux dernières mesures
6. Toute recommandation pertinente basée sur ces données

Utilise un langage simple et direct adapté à un professionnel de santé.
"""
    
    elif questionnaire_type == "EQ-5D":
        # Pour EQ-5D, on extrait les scores de dimensions et VAS
        eq5d_data = []
        for _, row in patient_data.iterrows():
            date = row['date_collecte'].strftime('%Y-%m-%d')
            periode = row['periode']
            dimensions = {}
            for dim in ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]:
                col_name = f"score_eq_{dim}"
                if col_name in row and not pd.isna(row[col_name]):
                    dimensions[dim] = row[col_name]
            
            vas_score = row.get('score_eq_vas')
            if not pd.isna(vas_score):
                eq5d_data.append((date, dimensions, vas_score, periode))
        
        prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir un résumé clair et concis pour un clinicien.

Contexte Clinique:
{interpretation_guide}

Date d'opération: {date_operation}
{references_text}

Données du Patient:
Voici les scores EQ-5D pour le patient {patient_id} :
{eq5d_data}

Tâche:
Analyse ces scores en te basant sur les directives fournies. Rédige un résumé (max 3-4 phrases) mettant en lumière : 
1. Le statut actuel du patient selon les derniers scores des dimensions et l'EVA
2. L'évolution globale depuis la première mesure
3. La comparaison des scores pré et post-opératoires
4. L'impact potentiel des références à d'autres professionnels sur l'évolution des scores
5. Tout changement notable entre les deux dernières mesures
6. Toute recommandation pertinente basée sur ces données

Utilise un langage simple et direct adapté à un professionnel de santé.
"""
    
    elif questionnaire_type == "Cochlear Implant":
        # Pour le questionnaire sur les implants cochléaires
        cochlear_data = []
        for _, row in patient_data.iterrows():
            date = row['date_collecte'].strftime('%Y-%m-%d')
            periode = row['periode']
            scores = {}
            for category, questions in COCHLEAR_IMPLANT_CATEGORIES.items():
                category_scores = {}
                for q in questions:
                    col_name = f"score_{q}"
                    if col_name in row and not pd.isna(row[col_name]):
                        category_scores[q] = row[col_name]
                if category_scores:
                    scores[category] = category_scores
            if scores:
                cochlear_data.append((date, scores, periode))
        
        prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir un résumé clair et concis pour un clinicien.

Contexte Clinique:
{interpretation_guide}

Date d'opération: {date_operation}
{references_text}

Données du Patient:
Voici les scores du questionnaire sur les implants cochléaires pour le patient {patient_id} :
{cochlear_data}

Tâche:
Analyse ces scores en te basant sur les directives fournies. Rédige un résumé (max 3-4 phrases) mettant en lumière : 
1. Le statut actuel du patient selon les derniers scores
2. L'évolution globale depuis la première mesure
3. La comparaison des scores pré et post-opératoires
4. L'impact potentiel des références à d'autres professionnels sur l'évolution des scores
5. Tout changement notable entre les deux dernières mesures
6. Toute recommandation pertinente basée sur ces données

Utilise un langage simple et direct adapté à un professionnel de santé.
"""
    else:
        prompt = "Type de questionnaire non pris en charge."
    
    return prompt

def construct_detailed_prompt(patient_data, interpretation_guide, questionnaire_type):
    """Construit un prompt détaillé incluant les scores de chaque question et leur texte"""
    patient_id = patient_data['patient_id'].iloc[0]
    date_operation = patient_data['date_operation'].iloc[0]
    
    # Récupérer les références pour ce type de questionnaire
    references = patient_data[patient_data['reference_professional'].notna()].copy()
    references_text = ""
    if not references.empty:
        references_text = "\nRéférences à d'autres professionnels de la santé:\n"
        for _, ref in references.iterrows():
            references_text += f"- {ref['reference_professional']} le {ref['reference_date']}\n"
    
    if questionnaire_type == "OHS":
        # Informations sur l'échelle de mesure OHS
        scale_info = OHS_SCORES_MEANING
        
        # Formatage des questions et de leur signification
        questions_text = "\n".join([f"{key}: {text}" for key, text in OHS_QUESTIONS.items()])
        
        # Formatage des données par date de collecte avec scores détaillés
        detailed_scores = []
        for _, row in patient_data.iterrows():
            date = row['date_collecte'].strftime('%Y-%m-%d')
            periode = "pré-opératoire" if row['periode'] == "pre" else "post-opératoire"
            score_total = row['score_total']
            reference_info = ""
            if pd.notna(row.get('reference_professional')):
                reference_info = f" (Référence à {row['reference_professional']} le {row['reference_date']})"
            
            # Récupération des scores par question
            question_scores = []
            for q in range(1, 13):
                q_col = f'score_q{q}'
                if q_col in row and not pd.isna(row[q_col]):
                    q_score = row[q_col]
                    question_scores.append(f"Q{q}: {q_score}/4")
            
            question_details = ", ".join(question_scores)
            detailed_scores.append(f"Date: {date} ({periode}){reference_info}, Score total: {score_total}/48\nScores détaillés: {question_details}")
        
        detailed_scores_text = "\n\n".join(detailed_scores)
        
        # Création du prompt complet
        prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir une analyse détaillée mais concise pour un clinicien.

Contexte Clinique:
{interpretation_guide}

Date d'opération: {date_operation}
{references_text}

{scale_info}

Voici les questions du questionnaire Oxford Hip Score (OHS):
{questions_text}

Données du Patient {patient_id}:
{detailed_scores_text}

Tâche:
Analyse ces scores en te basant sur les directives fournies et les réponses détaillées aux questions. Fournis:

1. Un résumé général (2-3 phrases) sur l'état actuel du patient et son évolution globale
2. Une analyse des domaines spécifiques les plus problématiques pour le patient (en te référant aux questions spécifiques)
3. Une analyse des domaines qui se sont le plus améliorés ou détériorés entre les collectes
4. Une analyse détaillée des changements pré et post-opératoires
5. Une analyse de l'impact des références à d'autres professionnels sur l'évolution des scores, en particulier:
   - L'impact des physiothérapeutes/kinésithérapeutes sur la mobilité et la douleur
   - L'impact des ergothérapeutes sur les activités quotidiennes
   - La temporalité des améliorations par rapport aux dates de référence
6. Des recommandations cliniques potentielles basées sur les réponses aux questions spécifiques

Utilise un langage clinique précis mais accessible. Organise ta réponse avec des sous-titres pour faciliter la lecture rapide.
"""
    
    elif questionnaire_type == "EQ-5D":
        # Informations sur l'échelle de mesure EQ-5D
        scale_info = EQ5D_SCORES_MEANING
        
        # Formatage des questions et de leur signification
        questions_text = "\n".join([f"{key}: {text}" for key, text in EQ5D_QUESTIONS.items()])
        
        # Formatage des données par date de collecte avec scores détaillés
        detailed_scores = []
        for _, row in patient_data.iterrows():
            date = row['date_collecte'].strftime('%Y-%m-%d')
            periode = "pré-opératoire" if row['periode'] == "pre" else "post-opératoire"
            reference_info = ""
            if pd.notna(row.get('reference_professional')):
                reference_info = f" (Référence à {row['reference_professional']} le {row['reference_date']})"
            
            # Récupération des scores des dimensions
            dimension_scores = []
            for dim in ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]:
                dim_col = f'score_eq_{dim}'
                if dim_col in row and not pd.isna(row[dim_col]):
                    dim_score = row[dim_col]
                    dimension_scores.append(f"{dim}: {dim_score}/5")
            
            vas_score = row.get('score_eq_vas')
            if pd.isna(vas_score):
                vas_text = "Non disponible"
            else:
                vas_text = f"{vas_score}/100"
            
            dimension_details = ", ".join(dimension_scores)
            detailed_scores.append(f"Date: {date} ({periode}){reference_info}\nDimensions: {dimension_details}\nÉchelle visuelle analogique (EVA): {vas_text}")
        
        detailed_scores_text = "\n\n".join(detailed_scores)
        
        # Création du prompt complet
        prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir une analyse détaillée mais concise pour un clinicien.

Contexte Clinique:
{interpretation_guide}

Date d'opération: {date_operation}
{references_text}

{scale_info}

Voici les dimensions du questionnaire EQ-5D:
{questions_text}

Données du Patient {patient_id}:
{detailed_scores_text}

Tâche:
Analyse ces scores en te basant sur les directives fournies et les réponses détaillées aux dimensions. Fournis:

1. Un résumé général (2-3 phrases) sur l'état actuel du patient et son évolution globale
2. Une analyse des dimensions spécifiques les plus problématiques pour le patient
3. Une analyse des dimensions qui se sont le plus améliorées ou détériorées entre les collectes
4. Une analyse de l'évolution de l'EVA et sa corrélation avec les dimensions
5. Une analyse détaillée des changements pré et post-opératoires
6. Une analyse de l'impact des références à d'autres professionnels sur l'évolution des scores, en particulier:
   - L'impact des psychologues sur l'anxiété/dépression et le bien-être général
   - L'impact des travailleurs sociaux sur les activités courantes et l'autonomie
   - L'impact des infirmiers spécialisés sur la douleur et les soins auto-administrés
   - La temporalité des améliorations par rapport aux dates de référence
7. Des recommandations cliniques potentielles basées sur les réponses aux dimensions spécifiques

Utilise un langage clinique précis mais accessible. Organise ta réponse avec des sous-titres pour faciliter la lecture rapide.
"""
    
    elif questionnaire_type == "Cochlear Implant":
        # Informations sur l'échelle de mesure
        scale_info = COCHLEAR_IMPLANT_SCORES_MEANING
        
        # Formatage des questions et de leur signification
        questions_text = "\n".join([f"{key}: {text}" for key, text in COCHLEAR_IMPLANT_QUESTIONS.items()])
        
        # Formatage des données par date de collecte avec scores détaillés
        detailed_scores = []
        for _, row in patient_data.iterrows():
            date = row['date_collecte'].strftime('%Y-%m-%d')
            periode = "pré-opératoire" if row['periode'] == "pre" else "post-opératoire"
            reference_info = ""
            if pd.notna(row.get('reference_professional')):
                reference_info = f" (Référence à {row['reference_professional']} le {row['reference_date']})"
            
            # Récupération des scores par catégorie
            categories = {
                "Pratique": ["practice_1", "practice_2"],
                "Environnements bruyants": ["noisy_1", "noisy_2", "noisy_3", "noisy_4"],
                "Fonctionnement académique": ["academic_1", "academic_2", "academic_3", "academic_4", "academic_5"],
                "Communication orale": ["oral_1", "oral_2", "oral_3"],
                "Fatigue": ["fatigue_1", "fatigue_2"],
                "Fonctionnement social": ["social_1", "social_2", "social_3", "social_4"],
                "Fonctionnement émotionnel": ["emotional_1", "emotional_2", "emotional_3", "emotional_4"]
            }
            
            category_scores = []
            for category, questions in categories.items():
                scores = []
                for q in questions:
                    q_col = f'score_{q}'
                    if q_col in row and not pd.isna(row[q_col]):
                        scores.append(f"{q}: {row[q_col]}/4")
                if scores:
                    category_scores.append(f"{category}: {', '.join(scores)}")
            
            detailed_scores.append(f"Date: {date} ({periode}){reference_info}\n" + "\n".join(category_scores))
        
        detailed_scores_text = "\n\n".join(detailed_scores)
        
        # Création du prompt complet
        prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir une analyse détaillée mais concise pour un clinicien.

Contexte Clinique:
{interpretation_guide}

Date d'opération: {date_operation}
{references_text}

{scale_info}

Voici les questions du questionnaire sur les implants cochléaires:
{questions_text}

Données du Patient {patient_id}:
{detailed_scores_text}

Tâche:
Analyse ces scores en te basant sur les directives fournies et les réponses détaillées aux questions. Fournis:

1. Un résumé général (2-3 phrases) sur l'état actuel du patient et son évolution globale
2. Une analyse des domaines spécifiques les plus problématiques pour le patient
3. Une analyse des domaines qui se sont le plus améliorés ou détériorés entre les collectes
4. Une évaluation de l'impact sur le fonctionnement académique et social
5. Une analyse du bien-être émotionnel et de la fatigue
6. Une analyse détaillée des changements pré et post-opératoires
7. Une analyse de l'impact des références à d'autres professionnels sur l'évolution des scores, en particulier:
   - L'impact des audiologistes sur la perception auditive et la communication
   - L'impact des orthophonistes sur la communication orale et le langage
   - L'impact des psychologues spécialisés sur le bien-être émotionnel et social
   - La temporalité des améliorations par rapport aux dates de référence
8. Des recommandations cliniques potentielles basées sur les réponses aux questions spécifiques

Utilise un langage clinique précis mais accessible. Organise ta réponse avec des sous-titres pour faciliter la lecture rapide.
"""
    else:
        prompt = "Type de questionnaire non pris en charge."
    
    return prompt

def get_llm_response(prompt, provider="google", model=None):
    """Obtient une réponse du LLM en fonction du fournisseur choisi"""
    response = ""
    
    try:
        if provider == "google":
            # Configuration Google
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                return "Erreur: Clé API Google non configurée"
            
            # Utilisation de l'approche client pour Gemini
            client = genai.Client(api_key=google_api_key)
            
            # Création du contenu
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            # Configuration de la génération
            model_name = model or "gemini-2.5-pro-exp-03-25"
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )
            
            # Génération de la réponse
            response_obj = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generate_content_config
            )
            
            response = response_obj.text
            
        else:
            response = "Fournisseur LLM non supporté"
            
    except Exception as e:
        response = f"Erreur lors de l'appel au LLM: {str(e)}"
        
    return response

# Chargement des données
data = load_data()

# Barre latérale pour les options et paramètres
st.sidebar.header("Paramètres")

# Sélection du provider LLM
llm_provider = st.sidebar.selectbox(
    "Choisissez un fournisseur de LLM",
    options=["google"],
    index=0  # Google par défaut
)

# Modèles pour chaque fournisseur
model_options = {
    "google": ["gemini-1.0-pro", "gemini-1.5-pro", "gemini-2.5-pro-exp-03-25"]
}

llm_model = st.sidebar.selectbox(
    "Choisissez un modèle",
    options=model_options[llm_provider],
    index=2  # gemini-2.5-pro-exp-03-25 par défaut pour Google
)

# Sélection du type de questionnaire
questionnaire_type = st.sidebar.radio(
    "Sélectionnez le type de questionnaire à analyser",
    options=["OHS", "EQ-5D", "Cochlear Implant"],
    index=0  # OHS par défaut
)

# Configuration API (clés stockées dans .env)
if llm_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.text_input("Clé API Google", type="password", key="google_api_key")
    if st.session_state.get("google_api_key"):
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key

# Information sur la génération de données synthétiques
st.sidebar.markdown("---")
if st.sidebar.button("Régénérer les données synthétiques"):
    from data_generator import generate_synthetic_data
    data = generate_synthetic_data(file_path="patients_proms.csv")
    st.rerun()

# Titre dynamique selon le questionnaire sélectionné
questionnaire_full_names = {
    "OHS": "Oxford Hip Score",
    "EQ-5D": "EuroQol 5-Dimension",
    "Cochlear Implant": "Questionnaire sur les implants cochléaires pour enfants d'âge scolaire"
}
st.header(f"Analyse du questionnaire {questionnaire_full_names.get(questionnaire_type, questionnaire_type)}")

# Valeur par défaut pour les directives d'interprétation en fonction du questionnaire
default_interpretation_guides = {
    "OHS": """Interprétation clinique de l'Oxford Hip Score (OHS) - Score de 0 à 48:
- 0-19 : Symptômes sévères de l'arthrose de la hanche.
- 20-29 : Symptômes modérés.
- 30-39 : Symptômes légers.
- 40-48 : Fonction satisfaisante de la hanche.
- Une amélioration de 5 points est généralement considérée comme cliniquement significative (Minimal Clinically Important Difference - MCID).
- Comparer le score actuel au score précédent pour évaluer l'évolution (amélioration, détérioration, stabilité).
- Noter toute baisse significative ou score restant très bas malgré le traitement.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien.""",
    
    "EQ-5D": """Interprétation clinique du questionnaire EQ-5D-5L:
- Les scores des dimensions vont de 1 (aucun problème) à 5 (problèmes extrêmes).
- Une diminution de score dans une dimension indique une amélioration.
- L'EVA va de 0 (pire état de santé imaginable) à 100 (meilleur état de santé imaginable).
- Une augmentation de l'EVA indique une amélioration de l'état de santé global.
- Une variation de l'EVA de 7-10 points est généralement considérée comme cliniquement significative.
- Porter attention aux dimensions avec les scores les plus élevés (≥3) qui nécessitent une attention particulière.
- Analyser si les améliorations/détériorations dans des dimensions spécifiques se reflètent dans le score EVA.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien.""",
    
    "Cochlear Implant": """Interprétation clinique du questionnaire sur les implants cochléaires pour enfants d'âge scolaire:
- Les scores vont de 1 (meilleur) à 4 (pire) pour la plupart des questions.
- Pour les questions de difficulté, un score plus élevé indique plus de difficultés.
- Pour les questions de fréquence, un score plus élevé indique une fréquence plus faible.
- Analyser les différentes catégories (environnements bruyants, fonctionnement académique, communication orale, fatigue, fonctionnement social, fonctionnement émotionnel) séparément.
- Porter une attention particulière aux scores élevés (≥3) qui indiquent des difficultés importantes.
- Noter toute détérioration dans les domaines académiques ou sociaux qui pourrait nécessiter des interventions.
- Analyser l'impact de la fatigue sur le fonctionnement quotidien.
- Évaluer le bien-être émotionnel et social de l'enfant.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien."""
}

# Liste des patients avec filtre pour le type de questionnaire sélectionné
patients_with_selected_questionnaire = data[data['questionnaire_type'] == questionnaire_type]['patient_id'].unique()
if len(patients_with_selected_questionnaire) == 0:
    st.warning(f"Aucun patient n'a de données pour le questionnaire {questionnaire_type}. Veuillez régénérer les données synthétiques ou choisir un autre type de questionnaire.")
    selected_patient = None
else:
    unique_patients = sorted(patients_with_selected_questionnaire)
    selected_patient = st.selectbox("Sélectionnez un patient", unique_patients)

if selected_patient:
    # Récupérer les données du patient sélectionné pour le questionnaire choisi
    patient_data = get_patient_data(data, selected_patient, questionnaire_type)
    
    if patient_data.empty:
        st.warning(f"Aucune donnée {questionnaire_type} disponible pour ce patient.")
    else:
        # Afficher les informations du patient dans la barre latérale
        st.sidebar.markdown("---")
        st.sidebar.subheader("Informations du patient")
        
        # Afficher la date d'opération
        date_operation = patient_data['date_operation'].iloc[0]
        st.sidebar.write(f"Date d'opération: {date_operation}")
        
        # Afficher un résumé des collectes
        pre_op_count = len(patient_data[patient_data['periode'] == 'pre'])
        post_op_count = len(patient_data[patient_data['periode'] == 'post'])
        st.sidebar.write(f"Nombre de collectes pré-opératoires: {pre_op_count}")
        st.sidebar.write(f"Nombre de collectes post-opératoires: {post_op_count}")
        
        # Afficher les références à d'autres professionnels
        references = patient_data[patient_data['reference_professional'].notna()]
        if not references.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Références à d'autres professionnels")
            for _, ref in references.iterrows():
                st.sidebar.write(f"- {ref['reference_professional']} le {ref['reference_date']}")
        
        # Afficher les données brutes selon le type de questionnaire
        if questionnaire_type == "OHS":
            st.subheader(f"Scores {questionnaire_type} du patient")
            
            # Extraire et formater les colonnes pertinentes pour OHS
            ohs_columns = ['date_collecte', 'periode', 'reference_professional', 'reference_date', 'score_total']
            ohs_columns.extend([f'score_q{i}' for i in range(1, 13)])
            
            # Vérifier quelles colonnes sont présentes
            available_ohs_columns = [col for col in ohs_columns if col in patient_data.columns]
            
            if len(available_ohs_columns) > 1:  # Au moins 'date_collecte' et une autre colonne
                display_df = patient_data[available_ohs_columns].copy()
                
                # Renommer les colonnes pour meilleure lisibilité
                rename_map = {
                    'date_collecte': 'Date de collecte',
                    'periode': 'Période',
                    'reference_professional': 'Professionnel référent',
                    'reference_date': 'Date de référence',
                    'score_total': 'Score total'
                }
                rename_map.update({f'score_q{i}': f'Q{i}' for i in range(1, 13)})
                display_df = display_df.rename(columns=rename_map)
                
                # Colorer les lignes selon la période
                def color_period(val):
                    if val == 'pre':
                        return 'background-color: #ffcccc'  # Rouge clair pour pré-opératoire
                    else:
                        return 'background-color: #ccffcc'  # Vert clair pour post-opératoire
                
                styled_df = display_df.style.applymap(color_period, subset=['Période'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("Aucune donnée détaillée OHS n'est disponible pour ce patient.")
            
            # Affichage des questions OHS (collapsible)
            with st.expander("Voir les questions du questionnaire OHS"):
                st.subheader("Questions du questionnaire OHS:")
                for q_num, question in OHS_QUESTIONS.items():
                    st.write(f"**Q{q_num}:** {question}")
            
        elif questionnaire_type == "EQ-5D":
            st.subheader(f"Scores {questionnaire_type} du patient")
            
            # Extraire et formater les colonnes pertinentes pour EQ-5D
            eq_columns = ['date_collecte', 'periode', 'reference_professional', 'reference_date']
            eq_columns.extend([f'score_eq_{dim}' for dim in ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]])
            eq_columns.append('score_eq_vas')
            
            # Vérifier quelles colonnes sont présentes
            available_eq_columns = [col for col in eq_columns if col in patient_data.columns]
            
            if len(available_eq_columns) > 1:  # Au moins 'date_collecte' et une autre colonne
                display_df = patient_data[available_eq_columns].copy()
                
                # Renommer les colonnes pour meilleure lisibilité
                rename_map = {
                    'date_collecte': 'Date de collecte',
                    'periode': 'Période',
                    'reference_professional': 'Professionnel référent',
                    'reference_date': 'Date de référence',
                    'score_eq_mobility': 'Mobilité',
                    'score_eq_self_care': 'Autonomie',
                    'score_eq_usual_activities': 'Activités courantes',
                    'score_eq_pain_discomfort': 'Douleur/Gêne',
                    'score_eq_anxiety_depression': 'Anxiété/Dépression',
                    'score_eq_vas': 'EVA (0-100)'
                }
                
                # Appliquer uniquement les renommages pour les colonnes disponibles
                rename_cols = {k: v for k, v in rename_map.items() if k in available_eq_columns}
                display_df = display_df.rename(columns=rename_cols)
                
                # Colorer les lignes selon la période
                def color_period(val):
                    if val == 'pre':
                        return 'background-color: #ffcccc'  # Rouge clair pour pré-opératoire
                    else:
                        return 'background-color: #ccffcc'  # Vert clair pour post-opératoire
                
                styled_df = display_df.style.applymap(color_period, subset=['Période'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("Aucune donnée détaillée EQ-5D n'est disponible pour ce patient.")
            
            # Affichage des questions EQ-5D (collapsible)
            with st.expander("Voir les dimensions du questionnaire EQ-5D"):
                st.subheader("Dimensions du questionnaire EQ-5D:")
                for dim, question in EQ5D_QUESTIONS.items():
                    st.write(f"**{dim.replace('_', ' ').title()}:** {question.split(': ', 1)[1] if ': ' in question else question}")
                
                st.markdown("""
                **Échelle visuelle analogique (EVA):** Sur une échelle de 0 à 100, où 0 représente le pire état de santé imaginable et 100 le meilleur état de santé imaginable, comment évaluez-vous votre état de santé aujourd'hui?
                """)
            
        elif questionnaire_type == "Cochlear Implant":
            st.subheader(f"Scores {questionnaire_type} du patient")
            
            # Extraire et formater les colonnes pertinentes pour le questionnaire sur les implants cochléaires
            cochlear_implant_columns = ['date_collecte', 'periode', 'reference_professional', 'reference_date']
            cochlear_implant_columns.extend([f'score_{q}' for q in COCHLEAR_IMPLANT_QUESTIONS.keys()])
            
            # Vérifier quelles colonnes sont présentes
            available_cochlear_implant_columns = [col for col in cochlear_implant_columns if col in patient_data.columns]
            
            if len(available_cochlear_implant_columns) > 1:  # Au moins 'date_collecte' et une autre colonne
                display_df = patient_data[available_cochlear_implant_columns].copy()
                
                # Renommer les colonnes pour meilleure lisibilité
                rename_map = {
                    'date_collecte': 'Date de collecte',
                    'periode': 'Période',
                    'reference_professional': 'Professionnel référent',
                    'reference_date': 'Date de référence'
                }
                rename_map.update({f'score_{q}': f'Q{q}' for q in COCHLEAR_IMPLANT_QUESTIONS.keys()})
                display_df = display_df.rename(columns=rename_map)
                
                # Colorer les lignes selon la période
                def color_period(val):
                    if val == 'pre':
                        return 'background-color: #ffcccc'  # Rouge clair pour pré-opératoire
                    else:
                        return 'background-color: #ccffcc'  # Vert clair pour post-opératoire
                
                styled_df = display_df.style.applymap(color_period, subset=['Période'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("Aucune donnée détaillée pour le questionnaire sur les implants cochléaires n'est disponible pour ce patient.")
            
            # Affichage des questions pour le questionnaire sur les implants cochléaires (collapsible)
            with st.expander("Voir les questions du questionnaire sur les implants cochléaires"):
                st.subheader("Questions du questionnaire sur les implants cochléaires:")
                for q, question in COCHLEAR_IMPLANT_QUESTIONS.items():
                    st.write(f"**{q}:** {question}")
        
        # Zone pour les directives d'interprétation avec valeur par défaut selon le questionnaire
        interpretation_guide = st.text_area(
            "Directives d'interprétation clinique",
            value=default_interpretation_guides.get(questionnaire_type, ""),
            height=200
        )
        
        # Création de colonnes pour les boutons
        col1, col2 = st.columns(2)
        
        # Bouton pour analyse standard (scores totaux uniquement)
        with col1:
            if questionnaire_type == "OHS":
                analysis_button_text = "Analyse simplifiée (scores totaux)"
            elif questionnaire_type == "EQ-5D":
                analysis_button_text = "Analyse simplifiée (dimensions et EVA)"
            else:
                analysis_button_text = "Analyse simplifiée (scores totaux)"
                
            if st.button(analysis_button_text, type="primary"):
                with st.spinner("Analyse en cours..."):
                    # Construire le prompt standard
                    prompt = construct_prompt(patient_data, interpretation_guide, questionnaire_type)
                    
                    # Obtenir la réponse du LLM
                    response = get_llm_response(prompt, provider=llm_provider, model=llm_model)
                    
                    # Afficher la réponse
                    st.header("Résumé généré par le LLM")
                    st.markdown(response)
                    
                    # Afficher le prompt utilisé (collapsible)
                    with st.expander("Voir le prompt utilisé"):
                        st.code(prompt)
        
        # Bouton pour analyse détaillée (avec scores de chaque question/dimension)
        with col2:
            if questionnaire_type == "OHS":
                detailed_button_text = "Analyse détaillée (scores par question)"
            elif questionnaire_type == "EQ-5D":
                detailed_button_text = "Analyse détaillée (par dimension et EVA)"
            else:
                detailed_button_text = "Analyse détaillée (scores totaux)"
                
            if st.button(detailed_button_text, type="primary"):
                with st.spinner("Analyse détaillée en cours..."):
                    # Construire le prompt détaillé
                    detailed_prompt = construct_detailed_prompt(patient_data, interpretation_guide, questionnaire_type)
                    
                    # Obtenir la réponse du LLM
                    detailed_response = get_llm_response(detailed_prompt, provider=llm_provider, model=llm_model)
                    
                    # Afficher la réponse
                    st.header("Analyse détaillée générée par le LLM")
                    st.markdown(detailed_response)
                    
                    # Afficher le prompt utilisé (collapsible)
                    with st.expander("Voir le prompt détaillé utilisé"):
                        st.code(detailed_prompt)
        
        # Afficher plus d'informations sur le patient pour les cliniciens intéressés
        with st.expander("Statistiques supplémentaires"):
            # Affichage d'un résumé statistique adapté au type de questionnaire
            if questionnaire_type == "OHS" and 'score_total' in patient_data.columns:
                st.subheader("Statistiques OHS")
                
                # Calcul des statistiques clés
                first_score = patient_data['score_total'].iloc[0]
                last_score = patient_data['score_total'].iloc[-1]
                min_score = patient_data['score_total'].min()
                max_score = patient_data['score_total'].max()
                mean_score = patient_data['score_total'].mean()
                
                # Création de colonnes pour les statistiques
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Premier score", f"{first_score}")
                col2.metric("Dernier score", f"{last_score}", f"{last_score - first_score:+g}")
                col3.metric("Score minimum", f"{min_score}")
                col4.metric("Score maximum", f"{max_score}")
                col5.metric("Score moyen", f"{mean_score:.1f}")
                
            elif questionnaire_type == "EQ-5D" and 'score_eq_vas' in patient_data.columns:
                st.subheader("Statistiques EQ-5D")
                
                # Calcul des statistiques pour l'EVA
                vas_data = patient_data[~pd.isna(patient_data['score_eq_vas'])]['score_eq_vas']
                
                if not vas_data.empty:
                    first_vas = vas_data.iloc[0]
                    last_vas = vas_data.iloc[-1]
                    min_vas = vas_data.min()
                    max_vas = vas_data.max()
                    mean_vas = vas_data.mean()
                    
                    # Création de colonnes pour les statistiques EVA
                    st.write("**Évolution de l'Échelle Visuelle Analogique (EVA):**")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Premier EVA", f"{first_vas:.0f}")
                    col2.metric("Dernier EVA", f"{last_vas:.0f}", f"{last_vas - first_vas:+.0f}")
                    col3.metric("EVA minimum", f"{min_vas:.0f}")
                    col4.metric("EVA maximum", f"{max_vas:.0f}")
                    col5.metric("EVA moyen", f"{mean_vas:.1f}")
                
                # Statistiques pour les dimensions
                st.write("**Évolution des dimensions (1=meilleur, 5=pire):**")
                
                # Afficher une table avec l'évolution de chaque dimension
                dim_cols = [col for col in patient_data.columns if col.startswith('score_eq_') and col != 'score_eq_vas']
                
                if dim_cols:
                    dim_data = patient_data[['date_collecte'] + dim_cols].copy()
                    dim_data['date_collecte'] = dim_data['date_collecte'].dt.strftime('%Y-%m-%d')
                    
                    # Renommer les colonnes pour meilleure lisibilité
                    rename_map = {
                        'date_collecte': 'Date',
                        'score_eq_mobility': 'Mobilité',
                        'score_eq_self_care': 'Autonomie',
                        'score_eq_usual_activities': 'Activités',
                        'score_eq_pain_discomfort': 'Douleur',
                        'score_eq_anxiety_depression': 'Anxiété'
                    }
                    
                    rename_cols = {k: v for k, v in rename_map.items() if k in dim_data.columns}
                    dim_data = dim_data.rename(columns=rename_cols)
                    
                    st.dataframe(dim_data, use_container_width=True) 