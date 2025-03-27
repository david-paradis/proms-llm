import streamlit as st
import pandas as pd
import os
from datetime import datetime
import openai
import anthropic
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

# Signification des scores pour chaque question (échelle de 0 à 4)
OHS_SCORES_MEANING = """
Échelle de scores pour chaque question:
- 0 = Douleur/Difficulté sévère ou incapacité
- 1 = Douleur/Difficulté importante 
- 2 = Douleur/Difficulté modérée
- 3 = Peu de douleur/difficulté
- 4 = Aucune douleur/difficulté
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

def get_patient_data(df, patient_id):
    """Récupère les données d'un patient spécifique"""
    patient_data = df[df['patient_id'] == patient_id].copy()
    patient_data['date_collecte'] = pd.to_datetime(patient_data['date_collecte'])
    patient_data = patient_data.sort_values('date_collecte')
    return patient_data

def construct_prompt(patient_data, interpretation_guide):
    """Construit le prompt pour le LLM basé sur les données du patient et les directives d'interprétation"""
    # Formatage des données pour le prompt
    scores_data = [(row['date_collecte'].strftime('%Y-%m-%d'), row['score_total']) 
                   for _, row in patient_data.iterrows()]
    
    prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir un résumé clair et concis pour un clinicien occupé.

Contexte Clinique:
{interpretation_guide}

Données du Patient:
Voici les scores OHS pour le patient {patient_data['patient_id'].iloc[0]} :
{scores_data}

Tâche:
Analyse ces scores en te basant sur les directives fournies. Rédige un résumé (max 3-4 phrases) mettant en lumière : 
1. Le statut actuel du patient selon le dernier score
2. L'évolution globale depuis la première mesure
3. Tout changement notable entre les deux dernières mesures
4. Toute recommandation pertinente basée sur ces données

Utilise un langage simple et direct adapté à un professionnel de santé.
"""
    return prompt

def construct_detailed_prompt(patient_data, interpretation_guide):
    """Construit un prompt détaillé incluant les scores de chaque question et leur texte"""
    patient_id = patient_data['patient_id'].iloc[0]
    
    # Informations sur l'échelle de mesure
    scale_info = OHS_SCORES_MEANING
    
    # Formatage des questions et de leur signification
    questions_text = "\n".join([f"{key}: {text}" for key, text in OHS_QUESTIONS.items()])
    
    # Formatage des données par date de collecte avec scores détaillés
    detailed_scores = []
    for _, row in patient_data.iterrows():
        date = row['date_collecte'].strftime('%Y-%m-%d')
        score_total = row['score_total']
        
        # Récupération des scores par question
        question_scores = []
        for q in range(1, 13):
            q_col = f'score_q{q}'
            if q_col in row:
                q_score = row[q_col]
                question_scores.append(f"Q{q}: {q_score}/4")
        
        question_details = ", ".join(question_scores)
        detailed_scores.append(f"Date: {date}, Score total: {score_total}/48\nScores détaillés: {question_details}")
    
    detailed_scores_text = "\n\n".join(detailed_scores)
    
    # Création du prompt complet
    prompt = f"""Tu es un assistant médical spécialisé dans l'analyse de questionnaires PROMs. Ton but est de fournir une analyse détaillée mais concise pour un clinicien.

Contexte Clinique:
{interpretation_guide}

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
4. Des recommandations cliniques potentielles basées sur les réponses aux questions spécifiques

Utilise un langage clinique précis mais accessible. Organise ta réponse avec des sous-titres pour faciliter la lecture rapide.
"""
    return prompt

def get_llm_response(prompt, provider="google", model=None):
    """Obtient une réponse du LLM en fonction du fournisseur choisi"""
    response = ""
    
    try:
        if provider == "openai":
            # Configuration OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return "Erreur: Clé API OpenAI non configurée"
                
            client = openai.OpenAI(api_key=openai_api_key)
            completion = client.chat.completions.create(
                model=model or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Tu es un assistant médical spécialisé dans l'analyse de PROMs."},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion.choices[0].message.content
            
        elif provider == "anthropic":
            # Configuration Anthropic
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                return "Erreur: Clé API Anthropic non configurée"
                
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            message = client.messages.create(
                model=model or "claude-3-sonnet-20240229",
                max_tokens=1000,
                system="Tu es un assistant médical spécialisé dans l'analyse de PROMs.",
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text
            
        elif provider == "google":
            # Configuration Google
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                return "Erreur: Clé API Google non configurée"
            
            # Utilisation de l'approche client pour Gemini
            client = genai.Client(api_key=google_api_key)
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            model_obj = model or "gemini-2.5-pro-exp-03-25"
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )
            
            response_obj = client.models.generate_content(
                model=model_obj,
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
    options=["openai", "anthropic", "google"],
    index=2  # Google par défaut
)

# Modèles pour chaque fournisseur
model_options = {
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
    "google": ["gemini-1.0-pro", "gemini-1.5-pro", "gemini-2.5-pro-exp-03-25"]
}

llm_model = st.sidebar.selectbox(
    "Choisissez un modèle",
    options=model_options[llm_provider],
    index=2 if llm_provider == "google" else 0  # gemini-2.5-pro-exp-03-25 par défaut pour Google
)

# Configuration API (clés stockées dans .env)
if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.sidebar.text_input("Clé API OpenAI", type="password", key="openai_api_key")
    if st.session_state.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        
elif llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
    st.sidebar.text_input("Clé API Anthropic", type="password", key="anthropic_api_key")
    if st.session_state.get("anthropic_api_key"):
        os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_api_key
        
elif llm_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.text_input("Clé API Google", type="password", key="google_api_key")
    if st.session_state.get("google_api_key"):
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key

# Information sur la génération de données synthétiques
st.sidebar.markdown("---")
if st.sidebar.button("Régénérer les données synthétiques"):
    from data_generator import generate_synthetic_data
    data = generate_synthetic_data(file_path="patients_proms.csv")
    st.experimental_rerun()

# Liste des patients disponibles
unique_patients = sorted(data['patient_id'].unique())
selected_patient = st.selectbox("Sélectionnez un patient", unique_patients)

# Récupérer les données du patient sélectionné
patient_data = get_patient_data(data, selected_patient)

# Affichage des données brutes
st.header("Scores OHS du patient")
st.dataframe(
    patient_data[['date_collecte', 'questionnaire_type', 'score_total']]
    .rename(columns={
        'date_collecte': 'Date de collecte', 
        'questionnaire_type': 'Type de questionnaire', 
        'score_total': 'Score total'
    }),
    use_container_width=True
)

# Affichage des scores détaillés par question (collapsible)
with st.expander("Voir les scores détaillés par question"):
    # Créer un tableau avec les scores de chaque question pour chaque date
    if not patient_data.empty:
        # Préparer les colonnes: Date + Q1-Q12
        question_cols = [f'score_q{i}' for i in range(1, 13)]
        
        if all(col in patient_data.columns for col in question_cols):
            detail_df = patient_data[['date_collecte'] + question_cols].copy()
            
            # Renommer les colonnes pour meilleure lisibilité
            rename_cols = {'date_collecte': 'Date de collecte'}
            rename_cols.update({f'score_q{i}': f'Q{i}' for i in range(1, 13)})
            detail_df = detail_df.rename(columns=rename_cols)
            
            st.dataframe(detail_df, use_container_width=True)
            
            # Afficher le texte de chaque question pour référence
            st.subheader("Questions du questionnaire OHS:")
            for q_num, question in enumerate(OHS_QUESTIONS.values(), 1):
                st.write(f"**Q{q_num}:** {question.split('. ', 1)[1] if '. ' in question else question}")
        else:
            st.info("Les données détaillées par question ne sont pas disponibles pour ce patient.")

# Zone pour les directives d'interprétation
default_guidelines = """Interprétation clinique de l'Oxford Hip Score (OHS) - Score de 0 à 48:
- 0-19 : Symptômes sévères de l'arthrose de la hanche.
- 20-29 : Symptômes modérés.
- 30-39 : Symptômes légers.
- 40-48 : Fonction satisfaisante de la hanche.
- Une amélioration de 5 points est généralement considérée comme cliniquement significative (Minimal Clinically Important Difference - MCID).
- Comparer le score actuel au score précédent pour évaluer l'évolution (amélioration, détérioration, stabilité).
- Noter toute baisse significative ou score restant très bas malgré le traitement.
- Demeurer nuancé et faire des suggestions extrêmement délicatement, veillant à ne pas assumer le rôle d'un clinicien."""

interpretation_guide = st.text_area(
    "Directives d'interprétation clinique",
    value=default_guidelines,
    height=200
)

# Création de colonnes pour les boutons
col1, col2 = st.columns(2)

# Bouton pour analyse standard (scores totaux uniquement)
with col1:
    if st.button("Analyse simplifiée (scores totaux)", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Construire le prompt standard
            prompt = construct_prompt(patient_data, interpretation_guide)
            
            # Obtenir la réponse du LLM
            response = get_llm_response(prompt, provider=llm_provider, model=llm_model)
            
            # Afficher la réponse
            st.header("Résumé généré par le LLM")
            st.markdown(response)
            
            # Afficher le prompt utilisé (collapsible)
            with st.expander("Voir le prompt utilisé"):
                st.code(prompt)

# Bouton pour analyse détaillée (avec scores de chaque question)
with col2:
    if st.button("Analyse détaillée (scores par question)", type="primary"):
        with st.spinner("Analyse détaillée en cours..."):
            # Construire le prompt détaillé
            detailed_prompt = construct_detailed_prompt(patient_data, interpretation_guide)
            
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
    # Affichage d'un résumé statistique
    if not patient_data.empty:
        st.subheader("Statistiques")
        
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