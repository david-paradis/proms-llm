import streamlit as st
import pandas as pd
import os
from datetime import datetime
import base64
import numpy as np
import json
from constants import (
    EQ5D_DIMENSIONS, 
    COCHLEAR_IMPLANT_CATEGORIES,
    REFERENCE_PROFESSIONALS,
    EQ5D_QUESTIONS,
    COCHLEAR_IMPLANT_QUESTIONS,
    COCHLEAR_IMPLANT_SCORES_MEANING,
    EQ5D_SCORES_MEANING
)

from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

# Charger les variables d'environnement depuis .env
load_dotenv()

# Charger la configuration du questionnaire OKS
with open('oks_questionnaire.json', 'r', encoding='utf-8') as f:
    OKS_CONFIG = json.load(f)

# Charger la configuration du questionnaire CIQOL-35
with open('ciqol35_questionnaire.json', 'r', encoding='utf-8') as f:
    CIQOL35_CONFIG = json.load(f)

# Configuration des questionnaires
QUESTIONNAIRE_CONFIGS = {
    "OKS": {
        "scale": "0-48 points",
        "categories": {
            "severe": "0-19",
            "moderate": "20-29",
            "mild": "30-39",
            "satisfactory": "40-48"
        },
        "mcid": "5 points",
        "score_format": "X/48",
        "question_format": "X/4",
        "domains": {
            "douleur": ["Q1", "Q8", "Q9", "Q10", "Q11", "Q12"],
            "fonction": ["Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
        },
        "interpretation_guide": """Interprétation clinique du Score de genou Oxford (OKS) - Score de 0 à 48:
- 0-19 : Symptômes sévères de l'arthrose du genou.
- 20-29 : Symptômes modérés.
- 30-39 : Symptômes légers.
- 40-48 : Fonction satisfaisante du genou.
- Une différence de 5 points est généralement considérée comme cliniquement significative (Minimal Clinically Important Difference - MCID).
- Comparer le score actuel au score précédent pour évaluer l'évolution (amélioration, détérioration, stabilité).
- Noter toute baisse significative ou score restant très bas malgré le traitement.
- Analyser séparément les domaines de douleur et de fonction.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien.""",
        "questions": OKS_CONFIG["questions"],
        "formulas": OKS_CONFIG["formulas"]
    },
    "EQ-5D": {
        "scale": "1-5 pour chaque dimension, 0-100 pour l'EVA",
        "categories": {
            "dimensions": "1 (aucun problème) à 5 (problèmes extrêmes)",
            "vas": "0 (pire état) à 100 (meilleur état)"
        },
        "mcid": "7-10 points pour l'EVA",
        "score_format": "X/5 pour les dimensions, X/100 pour l'EVA",
        "domains": {
            "mobility": "Mobilité",
            "self_care": "Autonomie",
            "usual_activities": "Activités courantes",
            "pain_discomfort": "Douleur/Gêne",
            "anxiety_depression": "Anxiété/Dépression",
            "vas": "Échelle visuelle analogique"
        },
        "interpretation_guide": """Interprétation clinique du questionnaire EQ-5D-5L:
- Les scores des dimensions vont de 1 (aucun problème) à 5 (problèmes extrêmes).
- Une diminution de score dans une dimension indique une amélioration.
- L'EVA va de 0 (pire état de santé imaginable) à 100 (meilleur état de santé imaginable).
- Une augmentation de l'EVA indique une amélioration de l'état de santé global.
- Une variation de l'EVA de 7-10 points est généralement considérée comme cliniquement significative.
- Porter attention aux dimensions avec les scores les plus élevés (≥3) qui nécessitent une attention particulière.
- Analyser si les améliorations/détériorations dans des dimensions spécifiques se reflètent dans le score EVA.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien."""
    },
    "Cochlear Implant": {
        "scale": "1-4 points",
        "categories": {
            "best": "1",
            "worst": "4"
        },
        "mcid": "1 point",
        "score_format": "X/4",
        "domains": {
            "practice": "Pratique",
            "noisy": "Environnements bruyants",
            "academic": "Fonctionnement académique",
            "oral": "Communication orale",
            "fatigue": "Fatigue",
            "social": "Fonctionnement social",
            "emotional": "Fonctionnement émotionnel"
        },
        "interpretation_guide": """Interprétation clinique du questionnaire sur les implants cochléaires pour enfants d'âge scolaire:
- Les scores vont de 1 (meilleur) à 4 (pire) pour la plupart des questions.
- Pour les questions de difficulté, un score plus élevé indique plus de difficultés.
- Pour les questions de fréquence, un score plus élevé indique une fréquence plus faible.
- Analyser les différentes catégories (environnements bruyants, fonctionnement académique, communication orale, fatigue, fonctionnement social, fonctionnement émotionnel) séparément.
- Porter une attention particulière aux scores élevés (≥3) qui indiquent des difficultés importantes.
- Noter toute détérioration dans les domaines académiques ou sociaux qui pourrait nécessiter des interventions.
- Analyser l'impact de la fatigue sur le fonctionnement quotidien.
- Évaluer le bien-être émotionnel et social de l'enfant.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien."""
    },
    "CIQOL-35": {
        "scale": "1-5 points par question (35 questions au total)",
        "categories": {
            "best": "35-70 points",
            "good": "71-105 points", 
            "moderate": "106-140 points",
            "poor": "141-175 points"
        },
        "mcid": "10 points",
        "score_format": "X/175",
        "question_format": "X/5",
        "domains": {
            "communication": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
            "social": ["Q9", "Q10", "Q11", "Q12", "Q13", "Q14"],
            "emotionnel": ["Q15", "Q16", "Q17", "Q18", "Q19", "Q20"],
            "environnemental": ["Q21", "Q22", "Q23", "Q24", "Q25"],
            "fatigue": ["Q26", "Q27", "Q28"],
            "usage": ["Q29", "Q30", "Q31", "Q32", "Q33"],
            "qualite_vie": ["Q34", "Q35"]
        },
        "interpretation_guide": """Interprétation clinique du questionnaire CIQOL-35 (Cochlear Implant Quality of Life):
- Les scores vont de 1 (jamais/pas du tout) à 5 (toujours/énormément) pour chaque question sur les 2 dernières semaines.
- Score total possible: 35-175 points (35 = meilleure qualité de vie, 175 = qualité de vie la plus affectée).
- Domaines évalués: Communication (8Q), Social (6Q), Émotionnel (6Q), Environnemental (5Q), Fatigue (3Q), Usage (5Q), Qualité de vie (2Q).
- Attention particulière aux scores élevés (≥4) qui indiquent des difficultés fréquentes ou importantes.
- Analyser séparément chaque domaine pour identifier les zones spécifiques d'impact.
- Noter les variations entre pré et post-implantation pour évaluer les bénéfices.
- Une augmentation du score indique une détérioration de la qualité de vie.
- Une diminution du score indique une amélioration de la qualité de vie.
- Une différence de 10 points est généralement considérée comme cliniquement significative (MCID).
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien.""",
        "questions": CIQOL35_CONFIG["questions"],
        "formulas": CIQOL35_CONFIG["formulas"]
    }
}

# Titre de l'application
APP_TITLE = "Analyse des PROMs"
st.set_page_config(page_title="Assistant d'Interprétation de PROMs via LLM", layout="wide")
st.title("Assistant d'Interprétation de PROMs via LLM")

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

def calculate_oks_subscores(row, config):
    """Calcule les sous-scores OKS (douleur et fonction) à partir des scores des questions individuelles"""
    subscores = {}
    
    for formula in config["formulas"]:
        if formula["nom"] in ["Douleur", "Fonction"]:
            # Extraire les questions utilisées pour ce domaine
            equation = formula["equation"]
            
            # Pour Douleur: questions 1,8,9,10,11,12
            # Pour Fonction: questions 2,3,4,5,6,7
            if formula["nom"] == "Douleur":
                question_nums = ["1", "8", "9", "10", "11", "12"]
            elif formula["nom"] == "Fonction":
                question_nums = ["2", "3", "4", "5", "6", "7"]
            
            # Calculer la somme des scores des questions
            total_score = 0
            valid_questions = 0
            for q_num in question_nums:
                q_col = f'score_q{q_num}'
                if q_col in row and not pd.isna(row[q_col]):
                    total_score += float(row[q_col])
                    valid_questions += 1
            
            # Calculer le score selon la formule (100/24)*somme
            if valid_questions > 0:
                subscores[f'score_{formula["nom"].lower()}'] = (100/24) * total_score
    
    return subscores

def calculate_ciqol35_subscores(row, config):
    """Calcule les sous-scores CIQOL-35 à partir des scores des questions individuelles"""
    subscores = {}
    
    for formula in config["formulas"]:
        # Extraire l'équation et les questions utilisées
        equation = formula["equation"]
        
        # Parser l'équation pour extraire les questions
        import re
        question_pattern = r'question(\d+)'
        question_nums = re.findall(question_pattern, equation)
        
        if question_nums:
            # Calculer la somme des scores des questions
            total_score = 0
            valid_questions = 0
            for q_num in question_nums:
                q_col = f'score_q{q_num}'
                if q_col in row and not pd.isna(row[q_col]):
                    total_score += float(row[q_col])
                    valid_questions += 1
            
            # Calculer le score selon la formule
            if valid_questions > 0:
                # Extraire le facteur multiplicateur de l'équation (ex: (100/40)* -> 100/40)
                factor_match = re.search(r'\((\d+)/(\d+)\)\*', equation)
                if factor_match:
                    numerator = int(factor_match.group(1))
                    denominator = int(factor_match.group(2))
                    score = (numerator / denominator) * total_score
                else:
                    # Pour le score total, pas de multiplication
                    score = total_score
                
                subscores[f'score_{formula["nom"].lower()}'] = score
    
    return subscores

def construct_structured_prompt(patient_data, interpretation_guide, questionnaire_type):
    """Construit un prompt pour générer une analyse structurée en JSON"""
    # Extraire les informations du patient
    patient_id = patient_data['patient_id'].iloc[0]
    date_operation = patient_data['date_operation'].iloc[0]
    
    # Récupérer les références pour ce type de questionnaire
    references = patient_data[patient_data['reference_professional'].notna()].copy()
    references_text = ""
    if not references.empty:
        references_text = "\nRéférences à d'autres professionnels de la santé:\n"
        for _, ref in references.iterrows():
            references_text += f"- {ref['reference_professional']} le {ref['reference_date']}\n"

    config = QUESTIONNAIRE_CONFIGS.get(questionnaire_type, {})
    
    # Préparer les informations du questionnaire
    questionnaire_info = ""
    if questionnaire_type in ["OKS", "CIQOL-35"]:
        questionnaire_name = "OKS" if questionnaire_type == "OKS" else "CIQOL-35"
        questionnaire_info = f"""## Structure du Questionnaire {questionnaire_name}

Chaque question est identifiée par un ID unique et a des choix de réponse spécifiques :

"""
        for q in config["questions"]:
            questionnaire_info += f"""Question {q["calculation_variable_name"].replace("question", "")} (ID: {q["id"]}):
{q["text"]}
Choix possibles:
{chr(10).join(f"- {choice['value']}: {choice['text']}" for choice in q["choices"])}

"""
    
    # Formatage des données selon le type de questionnaire
    detailed_scores = []
    for _, row in patient_data.iterrows():
        date = row['date_collecte'].strftime('%Y-%m-%d')
        periode = "pré-opératoire" if row['periode'] == "pre" else "post-opératoire"
        reference_info = ""
        if pd.notna(row.get('reference_professional')):
            reference_info = f" (Référence à {row['reference_professional']} le {row['reference_date']})"
        
        if questionnaire_type in ["OKS", "CIQOL-35"]:
            # Créer une entrée de résultat au format JSON
            result_entry = {
                "date": date,
                "type": periode,
                "calculated_scores": [],
                "results": []
            }
            
            # Calculer les sous-scores selon le type de questionnaire
            if questionnaire_type == "OKS":
                subscores = calculate_oks_subscores(row, config)
            else:  # CIQOL-35
                subscores = calculate_ciqol35_subscores(row, config)
            
            # Ajouter les scores calculés
            for formula in config["formulas"]:
                formula_col = f'score_{formula["nom"].lower()}'
                score_value = None
                
                # Utiliser le score total existant ou le sous-score calculé
                if formula["nom"] == "Total" and formula_col in row and not pd.isna(row[formula_col]):
                    score_value = float(row[formula_col])
                elif formula_col in subscores:
                    score_value = subscores[formula_col]
                
                if score_value is not None:
                    # Déterminer les limites selon le type de questionnaire
                    if questionnaire_type == "OKS":
                        maximum = 100 if formula["nom"] in ["Douleur", "Fonction"] else 48
                    else:  # CIQOL-35
                        maximum = 100 if formula["nom"] != "Total" else 175
                    
                    result_entry["calculated_scores"].append({
                        "name": formula["nom"].lower(),
                        "score": score_value,
                        "minimum": 0 if questionnaire_type == "OKS" else (35 if formula["nom"] == "Total" else 0),
                        "maximum": maximum
                    })
            
            # Ajouter les scores détaillés pour chaque question
            for q in config["questions"]:
                q_num = q["calculation_variable_name"].replace("question", "")
                q_col = f'score_q{q_num}'
                if q_col in row and not pd.isna(row[q_col]):
                    score_value = float(row[q_col])
                    # Trouver le choix correspondant au score
                    choice_text = next((choice["text"] for choice in q["choices"] 
                                     if choice["value"] == score_value), "Non spécifié")
                    
                    result_entry["results"].append({
                        "question_id": q["id"],
                        "question_number": int(q_num),
                        "question_text": q["text"],
                        "choice": choice_text,
                        "value": score_value
                    })
            
            detailed_scores.append(result_entry)
            
        elif questionnaire_type == "EQ-5D":
            # Format JSON pour EQ-5D
            result_entry = {
                "date": date,
                "type": periode,
                "dimensions": {},
                "vas": None
            }
            
            for dim in ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]:
                dim_col = f'score_eq_{dim}'
                if dim_col in row and not pd.isna(row[dim_col]):
                    result_entry["dimensions"][dim] = float(row[dim_col])
            
            if 'score_eq_vas' in row and not pd.isna(row['score_eq_vas']):
                result_entry["vas"] = float(row['score_eq_vas'])
            
            detailed_scores.append(result_entry)
            
        elif questionnaire_type == "Cochlear Implant":
            # Format JSON pour Cochlear Implant
            result_entry = {
                "date": date,
                "type": periode,
                "categories": {}
            }
            
            for category, questions in config["domains"].items():
                if isinstance(questions, list):
                    category_scores = {}
                    for q in questions:
                        q_col = f'score_{q}'
                        if q_col in row and not pd.isna(row[q_col]):
                            category_scores[q] = float(row[q_col])
                    if category_scores:
                        result_entry["categories"][category] = category_scores
            
            detailed_scores.append(result_entry)
    
    # Construction du prompt avec la structure JSON
    prompt = f"""## Instructions Système

Tu es un assistant médical spécialisé dans l'analyse comparative de questionnaires PROMs. Tu reçois des données structurées et dois retourner une analyse clinique concise sous format JSON standardisé.

## Contexte Clinique {questionnaire_type}

- **Échelle :** {config.get('scale', '')}
- **MCID :** Différence cliniquement significative = {config.get('mcid', '')}
- **Domaines fonctionnels :** {', '.join(config.get('domains', {}).keys())}

{questionnaire_info}

## Directives d'Interprétation Clinique

{config.get('interpretation_guide', '')}

## Logique d'Analyse

### Critères de Classification

1. **Détérioration Notable :** Baisse ≥2 points entre collectes
2. **Amélioration Notable :** Hausse ≥2 points entre collectes OU amélioration soutenue ≥1 point
3. **Stable Problématique :** Score ≤1/4 maintenu sur ≥2 collectes consécutives
4. **Fluctuation :** Variations importantes (≥2 points) sans tendance claire

### Contexte Opératoire et Temporel

- **Identifier automatiquement** si collecte est pré-opératoire ou post-opératoire basé sur :
    - Date d'opération fournie vs date de collecte
    - Indicateurs "preop"/"postop" dans les métadonnées de collecte
- **Terminologie temporelle** à utiliser dans les descriptions :
    - "pré-op" pour les collectes avant l'opération
    - "post-opératoire" pour le contexte général après l'opération

### Règles de Priorisation

- Prioriser les détériorations post-opératoires (particulièrement préoccupantes)
- Distinguer les améliorations post-opératoires attendues vs les rechutes
- Analyser les patterns : amélioration initiale puis rechute, détérioration progressive, etc.
- Considérer l'impact des références externes sur la chronologie
- Utiliser les scores calculés pour identifier les domaines fonctionnels et leur évolution

## Format de Sortie Requis

```json
{{
  "patient_analysis": {{
    "executive_summary": {{
      "overall_trend": "improvement|deterioration|stable|fluctuating",
      "key_concern": "Description concise du principal enjeu clinique",
      "current_score": "{config.get('score_format', '')}",
      "score_category": "severe|moderate|mild|satisfactory"
    }},
    "notable_deteriorations": {{
      "items": [
        {{
          "id": "questionX",
          "name": "Résumé de la question",
          "description": "Évolution détaillée avec scores explicites et contexte temporel"
        }}
      ]
    }},
    "stable_problematic_areas": {{
      "items": [
        {{
          "id": "questionX",
          "name": "Résumé de la question",
          "description": "Score persistant avec valeurs explicites et impact fonctionnel"
        }}
      ]
    }},
    "notable_improvements": {{
      "items": [
        {{
          "id": "douleur",
          "name": "Résumé du domaine de la douleur",
          "description": "Progression positive de la douleur avec scores calculés explicites"
        }}
      ]
    }}
  }}
}}
```

### Instructions pour les Descriptions

- **Toujours inclure les scores numériques** dans chaque description
- **Pour les détériorations** : Mentionner score initial → score final avec contexte temporel
- **Pour les améliorations** : Indiquer la progression
- **Pour les stables problématiques** : Préciser le score maintenu et sa persistance
- **Contextualiser temporellement** quand pertinent (dates, références externes)
- **Utiliser le texte des questions** pour donner du contexte aux analyses

## Instructions de Traitement

1. **Analyser chronologiquement** toutes les collectes pour identifier les tendances
2. **Utiliser les scores calculés** fournis dans les données d'entrée pour identifier les domaines fonctionnels
3. **Déterminer le contexte opératoire** de chaque collecte
4. **Calculer les évolutions** par question et identifier les changements significatifs
5. **Contextualiser temporellement** en utilisant la terminologie médicale standardisée
6. **Prioriser** les éléments par importance clinique
7. **Utiliser une terminologie médicale précise** mais accessible
8. **Maintenir la neutralité clinique** - observer et décrire les patterns, ne pas diagnostiquer
9. **Référencer le texte des questions** dans les analyses pour donner du contexte

## Données du Patient {patient_id}

{json.dumps({"results": detailed_scores}, indent=2, ensure_ascii=False)}

{references_text}

IMPORTANT: Ta réponse doit être un JSON valide et uniquement du JSON, sans autre texte.
"""
    
    return prompt

def get_llm_response(prompt, provider="google", model=None):
    """Obtient une réponse du LLM en fonction du fournisseur choisi"""
    response = ""
    start_time = time.time()
    token_info = None
    
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
                    role="system",
                    parts=[types.Part.from_text(text="Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Tu dois toujours répondre en JSON valide, sans autre texte.")],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            # Configuration de la génération
            model_name = model or llm_config["google"]["model"]
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )
            
            # Génération de la réponse
            response_obj = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generate_content_config
            )
            
            response = response_obj.text
            # Pour Google, nous n'avons pas accès aux tokens directement
            
        elif provider == "azure":
            # Configuration Azure OpenAI
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not azure_api_key:
                return "Erreur: Clé API Azure OpenAI non configurée"
            
            import openai
            
            # Configuration du client Azure OpenAI
            client = openai.AzureOpenAI(
                api_key=azure_api_key,
                api_version=llm_config["azure"]["api_version"],
                azure_endpoint=llm_config["azure"]["endpoint"]
            )
            
            # Génération de la réponse
            response_obj = client.chat.completions.create(
                model=llm_config["azure"]["model"],
                messages=[
                    {"role": "system", "content": "Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Tu dois toujours répondre en JSON valide, sans autre texte."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=100000
            )
            
            # Extraction de la réponse et des informations sur les tokens
            if hasattr(response_obj, 'choices') and len(response_obj.choices) > 0:
                if hasattr(response_obj.choices[0], 'message'):
                    response = response_obj.choices[0].message.content
                else:
                    response = str(response_obj.choices[0])
                
                # Extraire les informations sur les tokens
                if hasattr(response_obj, 'usage'):
                    token_info = {
                        'prompt_tokens': response_obj.usage.prompt_tokens,
                        'completion_tokens': response_obj.usage.completion_tokens,
                        'total_tokens': response_obj.usage.total_tokens
                    }
            else:
                response = str(response_obj)
            
        else:
            response = "Fournisseur LLM non supporté"
            
    except Exception as e:
        response = f"Erreur lors de l'appel au LLM: {str(e)}"
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Créer un expander pour afficher les logs dans le navigateur
    with st.expander("Logs de la réponse LLM"):
        st.write(f"Temps d'exécution: {execution_time:.2f} secondes")
        st.write(f"Provider: {provider}")
        st.write(f"Modèle: {model or llm_config[provider]['model']}")
        if token_info:
            st.write("Utilisation des tokens:")
            st.write(f"- Tokens du prompt: {token_info['prompt_tokens']}")
            st.write(f"- Tokens de la réponse: {token_info['completion_tokens']}")
            st.write(f"- Total des tokens: {token_info['total_tokens']}")
        st.write("Réponse complète:")
        st.code(response, language="json")
        
    return response

# Chargement des données
data = load_data()

# Barre latérale pour les options et paramètres
st.sidebar.header("Paramètres")

# Sélection du provider LLM
llm_provider = st.sidebar.radio(
    "Choisissez un fournisseur de LLM",
    options=["google", "azure"],
    index=0  # Google par défaut
)

# Configuration des modèles et endpoints pour chaque fournisseur
llm_config = {
    "google": {
        "model": "gemini-2.5-pro-exp-03-25",
        "endpoint": None  # Utilise l'endpoint par défaut de Google
    },
    "azure": {
        "model": "o3-mini",  # Nom du déploiement Azure
        "endpoint": "https://david-m1s9gx3t-eastus2.openai.azure.com/",
        "api_version": "2024-12-01-preview"
    }
}

# Configuration API (clés stockées dans .env)
if llm_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.text_input("Clé API Google", type="password", key="google_api_key")
    if st.session_state.get("google_api_key"):
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
elif llm_provider == "azure" and not os.getenv("AZURE_OPENAI_API_KEY"):
    st.sidebar.text_input("Clé API Azure OpenAI", type="password", key="azure_api_key")
    if st.session_state.get("azure_api_key"):
        os.environ["AZURE_OPENAI_API_KEY"] = st.session_state.azure_api_key

# Sélection du type de questionnaire
questionnaire_type = st.sidebar.radio(
    "Sélectionnez le type de questionnaire à analyser",
    options=["OKS", "EQ-5D", "Cochlear Implant", "CIQOL-35"],
    index=0  # OKS par défaut
)

# Information sur la génération de données synthétiques
st.sidebar.markdown("---")
if st.sidebar.button("Régénérer les données synthétiques"):
    from data_generator import generate_synthetic_data
    data = generate_synthetic_data(file_path="patients_proms.csv")
    st.rerun()

# Titre dynamique selon le questionnaire sélectionné
questionnaire_full_names = {
    "OKS": "Score de genou Oxford",
    "EQ-5D": "EuroQol 5-Dimension",
    "Cochlear Implant": "Questionnaire sur les implants cochléaires pour enfants d'âge scolaire",
    "CIQOL-35": "Cochlear Implant Quality of Life - 35 questions"
}
st.header(f"Analyse du questionnaire {questionnaire_full_names.get(questionnaire_type, questionnaire_type)}")

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
        
        # Afficher les directives d'interprétation dans un expander
        with st.expander("Directives d'interprétation clinique"):
            st.markdown(QUESTIONNAIRE_CONFIGS[questionnaire_type]["interpretation_guide"])
        
        # Afficher les données brutes selon le type de questionnaire
        if questionnaire_type == "OKS":
            st.subheader(f"Scores {questionnaire_type} du patient")
            
            # Extraire et formater les colonnes pertinentes pour OKS
            oks_columns = ['date_collecte', 'periode', 'reference_professional', 'reference_date']
            # Ajouter les colonnes de scores des questions
            oks_columns.extend([f'score_q{q["calculation_variable_name"].replace("question", "")}' for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]])
            # Ajouter les colonnes de scores calculés
            oks_columns.extend([f'score_{formula["nom"].lower()}' for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]])
            
            # Vérifier quelles colonnes sont présentes
            available_oks_columns = [col for col in oks_columns if col in patient_data.columns]
            
            if len(available_oks_columns) > 1:  # Au moins 'date_collecte' et une autre colonne
                display_df = patient_data[available_oks_columns].copy()
                
                # Renommer les colonnes pour meilleure lisibilité
                rename_map = {
                    'date_collecte': 'Date de collecte',
                    'periode': 'Période',
                    'reference_professional': 'Professionnel référent',
                    'reference_date': 'Date de référence'
                }
                # Renommer les colonnes des questions
                for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                    q_num = q["calculation_variable_name"].replace("question", "")
                    rename_map[f'score_q{q_num}'] = f'Q{q_num}'
                # Renommer les colonnes des scores calculés
                for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]:
                    rename_map[f'score_{formula["nom"].lower()}'] = formula["nom"]
                
                display_df = display_df.rename(columns=rename_map)
                
                # Colorer les lignes selon la période
                def color_period(val):
                    if val == 'pre':
                        return 'background-color: #ffcccc'  # Rouge clair pour pré-opératoire
                    else:
                        return 'background-color: #ccffcc'  # Vert clair pour post-opératoire
                
                styled_df = display_df.style.applymap(color_period, subset=['Période'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Afficher les scores détaillés pour chaque collecte
                for idx, original_row in patient_data.iterrows():
                    date_str = original_row['date_collecte'].strftime('%Y-%m-%d %H:%M:%S')
                    periode_str = "pré-opératoire" if original_row['periode'] == "pre" else "post-opératoire"
                    
                    with st.expander(f"Scores détaillés du {date_str} ({periode_str})"):
                        # Calculer les sous-scores pour cette ligne
                        subscores = calculate_oks_subscores(original_row, QUESTIONNAIRE_CONFIGS[questionnaire_type])
                        
                        # Afficher les scores calculés
                        st.write("**Scores calculés:**")
                        for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]:
                            formula_col = f'score_{formula["nom"].lower()}'
                            score_value = None
                            
                            # Utiliser le score total existant ou le sous-score calculé
                            if formula["nom"] == "Total" and formula_col in original_row and not pd.isna(original_row[formula_col]):
                                score_value = float(original_row[formula_col])
                            elif formula["nom"] in ["Douleur", "Fonction"] and formula_col in subscores:
                                score_value = subscores[formula_col]
                            
                            if score_value is not None:
                                max_score = 100 if formula["nom"] in ["Douleur", "Fonction"] else 48
                                st.write(f"- {formula['nom']}: {score_value:.1f}/{max_score}")
                        
                        # Afficher les scores détaillés par question
                        st.write("\n**Scores détaillés par question:**")
                        for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                            q_num = q["calculation_variable_name"].replace("question", "")
                            q_col = f'score_q{q_num}'
                            if q_col in original_row and not pd.isna(original_row[q_col]):
                                # Trouver le choix correspondant au score
                                score_value = float(original_row[q_col])
                                choice_text = next((choice["text"] for choice in q["choices"] 
                                                 if choice["value"] == score_value), "Non spécifié")
                                st.write(f"- Question {q_num} (ID: {q['id']}): {score_value:.1f}/4 - {choice_text}")
                                st.write(f"  {q['text']}")
            
            # Affichage des questions OKS (collapsible)
            with st.expander("Voir les questions du questionnaire OKS"):
                st.subheader("Questions du questionnaire OKS:")
                for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                    st.write(f"**Q{q['id']}:** {q['text']}")
            
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
        
        elif questionnaire_type == "CIQOL-35":
            st.subheader(f"Scores {questionnaire_type} du patient")
            
            # Extraire et formater les colonnes pertinentes pour CIQOL-35
            ciqol_columns = ['date_collecte', 'periode', 'reference_professional', 'reference_date']
            # Ajouter les colonnes de scores des questions
            ciqol_columns.extend([f'score_q{q["calculation_variable_name"].replace("question", "")}' for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]])
            # Ajouter les colonnes de scores calculés
            ciqol_columns.extend([f'score_{formula["nom"].lower()}' for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]])
            
            # Vérifier quelles colonnes sont présentes
            available_ciqol_columns = [col for col in ciqol_columns if col in patient_data.columns]
            
            if len(available_ciqol_columns) > 1:  # Au moins 'date_collecte' et une autre colonne
                display_df = patient_data[available_ciqol_columns].copy()
                
                # Renommer les colonnes pour meilleure lisibilité
                rename_map = {
                    'date_collecte': 'Date de collecte',
                    'periode': 'Période',
                    'reference_professional': 'Professionnel référent',
                    'reference_date': 'Date de référence'
                }
                # Renommer les colonnes des questions
                for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                    q_num = q["calculation_variable_name"].replace("question", "")
                    rename_map[f'score_q{q_num}'] = f'Q{q_num}'
                # Renommer les colonnes des scores calculés
                for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]:
                    rename_map[f'score_{formula["nom"].lower()}'] = formula["nom"]
                
                display_df = display_df.rename(columns=rename_map)
                
                # Colorer les lignes selon la période
                def color_period(val):
                    if val == 'pre':
                        return 'background-color: #ffcccc'  # Rouge clair pour pré-opératoire
                    else:
                        return 'background-color: #ccffcc'  # Vert clair pour post-opératoire
                
                styled_df = display_df.style.applymap(color_period, subset=['Période'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Afficher les scores détaillés pour chaque collecte
                for idx, original_row in patient_data.iterrows():
                    date_str = original_row['date_collecte'].strftime('%Y-%m-%d %H:%M:%S')
                    periode_str = "pré-opératoire" if original_row['periode'] == "pre" else "post-opératoire"
                    
                    with st.expander(f"Scores détaillés du {date_str} ({periode_str})"):
                        # Calculer les sous-scores pour cette ligne
                        subscores = calculate_ciqol35_subscores(original_row, QUESTIONNAIRE_CONFIGS[questionnaire_type])
                        
                        # Afficher les scores calculés
                        st.write("**Scores calculés:**")
                        for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]:
                            formula_col = f'score_{formula["nom"].lower()}'
                            score_value = None
                            
                            # Utiliser le score total existant ou le sous-score calculé
                            if formula["nom"] == "Total" and formula_col in original_row and not pd.isna(original_row[formula_col]):
                                score_value = float(original_row[formula_col])
                            elif formula_col in subscores:
                                score_value = subscores[formula_col]
                            
                            if score_value is not None:
                                max_score = 100 if formula["nom"] != "Total" else 175
                                min_score = 0 if formula["nom"] != "Total" else 35
                                st.write(f"- {formula['nom']}: {score_value:.1f} (échelle: {min_score}-{max_score})")
                        
                        # Afficher les scores détaillés par question
                        st.write("\n**Scores détaillés par question:**")
                        for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                            q_num = q["calculation_variable_name"].replace("question", "")
                            q_col = f'score_q{q_num}'
                            if q_col in original_row and not pd.isna(original_row[q_col]):
                                # Trouver le choix correspondant au score
                                score_value = float(original_row[q_col])
                                choice_text = next((choice["text"] for choice in q["choices"] 
                                                 if choice["value"] == score_value), "Non spécifié")
                                st.write(f"- Question {q_num} (ID: {q['id']}): {score_value:.1f}/5 - {choice_text}")
                                st.write(f"  {q['text']}")
            
            # Affichage des questions CIQOL-35 (collapsible)
            with st.expander("Voir les questions du questionnaire CIQOL-35"):
                st.subheader("Questions du questionnaire CIQOL-35:")
                for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                    st.write(f"**Q{q['calculation_variable_name'].replace('question', '')} (ID: {q['id']}):** {q['text']}")
                    choices_text = ", ".join([f"{choice['value']}: {choice['text']}" for choice in q['choices']])
                    st.write(f"  Choix: {choices_text}")
        
        # Bouton pour l'analyse détaillée
        if questionnaire_type in ["OKS", "CIQOL-35"]:
            analysis_button_text = "Analyse structurée (scores par question et domaines)"
        elif questionnaire_type == "EQ-5D":
            analysis_button_text = "Analyse structurée (par dimension et EVA)"
        else:
            analysis_button_text = "Analyse structurée (scores totaux)"

        if st.button(analysis_button_text, type="primary"):
            with st.spinner("Analyse en cours..."):
                # Construire le prompt structuré
                structured_prompt = construct_structured_prompt(patient_data, "", questionnaire_type)
                
                # Obtenir la réponse du LLM
                structured_response = get_llm_response(structured_prompt, provider=llm_provider, model=llm_config[llm_provider]["model"])
                
                try:
                    # Parser la réponse JSON
                    analysis_json = json.loads(structured_response)
                    
                    # Afficher l'analyse de manière structurée
                    st.header("Analyse structurée générée par le LLM")
                    
                    # Résumé exécutif
                    st.subheader("Résumé exécutif")
                    summary = analysis_json["patient_analysis"]["executive_summary"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tendance globale", summary["overall_trend"])
                        st.metric("Score actuel", summary["current_score"])
                    with col2:
                        st.metric("Catégorie", summary["score_category"])
                        st.write("**Enjeu principal:**", summary["key_concern"])
                    
                    # Détériorations notables
                    if analysis_json["patient_analysis"]["notable_deteriorations"]["items"]:
                        st.subheader("Détériorations notables")
                        for item in analysis_json["patient_analysis"]["notable_deteriorations"]["items"]:
                            st.write(f"**{item['name']} ({item['id']}):** {item['description']}")
                    
                    # Zones stables problématiques
                    if analysis_json["patient_analysis"]["stable_problematic_areas"]["items"]:
                        st.subheader("Zones stables problématiques")
                        for item in analysis_json["patient_analysis"]["stable_problematic_areas"]["items"]:
                            st.write(f"**{item['name']} ({item['id']}):** {item['description']}")
                    
                    # Améliorations notables
                    if analysis_json["patient_analysis"]["notable_improvements"]["items"]:
                        st.subheader("Améliorations notables")
                        for item in analysis_json["patient_analysis"]["notable_improvements"]["items"]:
                            st.write(f"**{item['name']} ({item['id']}):** {item['description']}")
                    
                    # Afficher le prompt utilisé (collapsible)
                    with st.expander("Voir le prompt utilisé"):
                        st.code(structured_prompt)
                    
                except json.JSONDecodeError:
                    st.error("Erreur: La réponse du LLM n'est pas un JSON valide")
                    st.code(structured_response)
                except Exception as e:
                    st.error(f"Erreur lors de l'affichage de l'analyse: {str(e)}")
                    st.code(structured_response)
        
        # Afficher plus d'informations sur le patient pour les cliniciens intéressés
        with st.expander("Statistiques supplémentaires"):
            # Affichage d'un résumé statistique adapté au type de questionnaire
            if questionnaire_type == "OKS" and 'score_total' in patient_data.columns:
                st.subheader("Statistiques OKS")
                
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