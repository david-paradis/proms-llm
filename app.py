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

# Charger les variables d'environnement depuis .env
load_dotenv()

# Charger la configuration du questionnaire OKS
with open('oks_questionnaire.json', 'r', encoding='utf-8') as f:
    OKS_CONFIG = json.load(f)

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
- Une amélioration de 5 points est généralement considérée comme cliniquement significative (Minimal Clinically Important Difference - MCID).
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
    
    # Formatage des données selon le type de questionnaire
    detailed_scores_text = []
    for _, row in patient_data.iterrows():
        date = row['date_collecte'].strftime('%Y-%m-%d')
        periode = "pré-opératoire" if row['periode'] == "pre" else "post-opératoire"
        reference_info = ""
        if pd.notna(row.get('reference_professional')):
            reference_info = f" (Référence à {row['reference_professional']} le {row['reference_date']})"
        
        if questionnaire_type == "OKS":
            # Collecter les scores calculés
            calculated_scores = []
            for formula in config["formulas"]:
                formula_col = f'score_{formula["nom"].lower()}'
                if formula_col in row and not pd.isna(row[formula_col]):
                    score_value = float(row[formula_col])
                    if formula["nom"] == "Total":
                        calculated_scores.append(f"Score Total: {score_value:.1f}/48")
                    elif formula["nom"] == "Douleur":
                        calculated_scores.append(f"Score Douleur: {score_value:.1f}/100")
                    elif formula["nom"] == "Fonction":
                        calculated_scores.append(f"Score Fonction: {score_value:.1f}/100")
            
            # Collecter les scores détaillés
            question_scores = []
            for q in config["questions"]:
                q_num = q["calculation_variable_name"].replace("question", "")  # Extraire le numéro de la variable de calcul
                q_col = f'score_q{q_num}'
                if q_col in row and not pd.isna(row[q_col]):
                    score_value = float(row[q_col])
                    # Trouver le choix correspondant au score
                    choice_text = next((choice["text"] for choice in q["choices"] 
                                     if choice["value"] == score_value), "Non spécifié")
                    question_scores.append(f"Question {q_num}: {score_value:.1f}/4 - {choice_text}")
            
            # Créer le texte formaté pour cette collecte
            collecte_text = f"""Date: {date} ({periode}){reference_info}
{', '.join(calculated_scores)}

Scores détaillés par question:
{chr(10).join(f"- {score}" for score in question_scores)}"""
            
            detailed_scores_text.append(collecte_text)
            
        elif questionnaire_type == "EQ-5D":
            # Format existant pour EQ-5D
            scores = []
            for dim in ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]:
                dim_col = f'score_eq_{dim}'
                if dim_col in row and not pd.isna(row[dim_col]):
                    scores.append(f"{dim}: {row[dim_col]}/5")
            if 'score_eq_vas' in row and not pd.isna(row['score_eq_vas']):
                scores.append(f"EVA: {row['score_eq_vas']}/100")
            detailed_scores_text.append(f"Date: {date} ({periode}){reference_info}\n" + ", ".join(scores))
            
        elif questionnaire_type == "Cochlear Implant":
            # Format existant pour Cochlear Implant
            scores = []
            for category, questions in config["domains"].items():
                if isinstance(questions, list):
                    category_scores = []
                    for q in questions:
                        q_col = f'score_{q}'
                        if q_col in row and not pd.isna(row[q_col]):
                            category_scores.append(f"{q}: {row[q_col]}/4")
                    if category_scores:
                        scores.append(f"{category}: {', '.join(category_scores)}")
            detailed_scores_text.append(f"Date: {date} ({periode}){reference_info}\n" + "\n".join(scores))

    # Construction du prompt avec la nouvelle structure
    prompt = f"""## Instructions Système

Tu es un assistant médical spécialisé dans l'analyse comparative de questionnaires PROMs. Tu reçois des données structurées et dois retourner une analyse clinique concise sous format JSON standardisé.

## Contexte Clinique {questionnaire_type}

- **Échelle :** {config.get('scale', '')}
- **MCID :** Différence cliniquement significative = {config.get('mcid', '')}
- **Domaines fonctionnels :** {', '.join(config.get('domains', {}).keys())}

## Directives d'Interprétation Clinique

{config.get('interpretation_guide', '')}

## Logique d'Analyse

### Critères de Classification

1. **Détérioration Notable :** Baisse ≥2 points entre collectes OU score final < score initial avec baisse ≥1 point
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
          "id": "questionX",
          "name": "Résumé de la question",
          "description": "Progression positive avec scores explicites"
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

## Instructions de Traitement

1. **Analyser chronologiquement** toutes les collectes pour identifier les tendances
2. **Utiliser les scores calculés** fournis dans les données d'entrée pour identifier les domaines fonctionnels
3. **Déterminer le contexte opératoire** de chaque collecte
4. **Calculer les évolutions** par question et identifier les changements significatifs
5. **Contextualiser temporellement** en utilisant la terminologie médicale standardisée
6. **Prioriser** les éléments par importance clinique
7. **Utiliser une terminologie médicale précise** mais accessible
8. **Maintenir la neutralité clinique** - observer et décrire les patterns, ne pas diagnostiquer

## Données du Patient {patient_id}

{chr(10).join(detailed_scores_text)}

{references_text}

IMPORTANT: Ta réponse doit être un JSON valide et uniquement du JSON, sans autre texte.
"""
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
            
            # Extraction de la réponse
            if hasattr(response_obj, 'choices') and len(response_obj.choices) > 0:
                if hasattr(response_obj.choices[0], 'message'):
                    response = response_obj.choices[0].message.content
                else:
                    response = str(response_obj.choices[0])
            else:
                response = str(response_obj)
            
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
    options=["OKS", "EQ-5D", "Cochlear Implant"],
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
    "Cochlear Implant": "Questionnaire sur les implants cochléaires pour enfants d'âge scolaire"
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
            oks_columns.extend([f'score_q{q["id"][-2:]}' for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]])  # Utiliser le bon format de nom de colonne
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
                rename_map.update({f'score_q{q["id"][-2:]}': f'Q{q["id"][-2:]}' for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]})  # Utiliser le bon format de nom de colonne
                rename_map.update({f'score_{formula["nom"].lower()}': f'{formula["nom"]}' for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]})
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
                for _, row in display_df.iterrows():
                    with st.expander(f"Scores détaillés du {row['Date de collecte']} ({row['Période']})"):
                        # Afficher les scores calculés
                        st.write("**Scores calculés:**")
                        for formula in QUESTIONNAIRE_CONFIGS[questionnaire_type]["formulas"]:
                            score_col = formula["nom"]
                            if score_col in row and not pd.isna(row[score_col]):
                                st.write(f"- {score_col}: {row[score_col]:.1f}/{100 if formula['nom'] in ['Douleur', 'Fonction'] else 48}")
                        
                        # Afficher les scores détaillés par question
                        st.write("\n**Scores détaillés par question:**")
                        for q in QUESTIONNAIRE_CONFIGS[questionnaire_type]["questions"]:
                            q_num = q["id"][-2:]
                            q_col = f'Q{q_num}'
                            if q_col in row and not pd.isna(row[q_col]):
                                st.write(f"- {q_col}: {row[q_col]:.1f}/4 - {q['text']}")
            else:
                st.info("Aucune donnée détaillée OKS n'est disponible pour ce patient.")
            
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
        
        # Bouton pour l'analyse détaillée
        if questionnaire_type == "OKS":
            analysis_button_text = "Analyse structurée (scores par question)"
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
                    
                    # Afficher le JSON brut (collapsible)
                    with st.expander("Voir l'analyse JSON brute"):
                        st.json(analysis_json)
                    
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