import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv
from constants import EQ5D_DIMENSIONS, COCHLEAR_IMPLANT_CATEGORIES, REFERENCE_PROFESSIONALS

def generate_synthetic_data(num_patients=10, entries_per_patient=(3, 8), file_path="patients_proms.csv"):
    """
    Génère des données synthétiques pour les scores OKS, EQ-5D et le questionnaire sur les implants cochléaires.
    Chaque entrée correspond à un type de questionnaire.
    
    Parameters:
    - num_patients: Nombre de patients à générer
    - entries_per_patient: Tuple contenant (min, max) nombre d'entrées par patient
    - file_path: Chemin où sauvegarder le fichier CSV
    """
    # Initialisation des données
    data = []
    patients_info = {}  # Pour stocker les informations des patients
    

    
    # Création d'IDs de patients (ex: P001, P002, etc.)
    patient_ids = [f"P{str(i+1).zfill(3)}" for i in range(num_patients)]
    
    # Date de début (environ 2 ans dans le passé)
    start_date = datetime.now() - timedelta(days=730)
    
    for patient_id in patient_ids:
        # Générer une date d'opération aléatoire dans la période de suivi
        operation_date = start_date + timedelta(days=random.randint(180, 540))  # Entre 6 et 18 mois après le début
        
        # Décider du modèle d'évolution pour ce patient
        evolution_pattern = random.choice(["improvement", "deterioration", "fluctuating", "stable"])
        
        # Déterminer le nombre d'entrées pour ce patient
        num_entries = random.randint(entries_per_patient[0], entries_per_patient[1])
        
        # Premier score (baseline)
        if evolution_pattern == "improvement":
            initial_oks_score = random.randint(10, 25)  # Score bas (symptômes modérés à sévères)
            initial_cochlear_scores = {q: random.randint(3, 4) for category in COCHLEAR_IMPLANT_CATEGORIES.values() for q in category}
        elif evolution_pattern == "deterioration":
            initial_oks_score = random.randint(30, 45)  # Score élevé (bon état initial)
            initial_cochlear_scores = {q: random.randint(1, 2) for category in COCHLEAR_IMPLANT_CATEGORIES.values() for q in category}
        else:
            initial_oks_score = random.randint(15, 35)  # Score variable
            initial_cochlear_scores = {q: random.randint(2, 3) for category in COCHLEAR_IMPLANT_CATEGORIES.values() for q in category}
        
        # Génération des entrées pour ce patient
        # Assurer qu'il y a des entrées pré et post-opératoires
        pre_op_dates = sorted([start_date + timedelta(days=random.randint(0, 180)) for _ in range(random.randint(1, 3))])
        post_op_dates = sorted([operation_date + timedelta(days=random.randint(30, 540)) for _ in range(random.randint(2, 5))])
        dates = pre_op_dates + post_op_dates
        
        # Générer des dates de référence pour chaque type de questionnaire
        reference_dates = {}
        for q_type in ["OKS", "EQ-5D", "Cochlear Implant"]:
            if random.random() < 0.7:  # 70% de chance d'avoir une référence
                # Choisir une date aléatoire entre la première et la dernière collecte
                ref_date = random.choice(dates) + timedelta(days=random.randint(1, 30))
                reference_dates[q_type] = {
                    "date": ref_date,
                    "professional": random.choice(REFERENCE_PROFESSIONALS[q_type])
                }
        
        current_oks_score = initial_oks_score
        current_cochlear_scores = initial_cochlear_scores.copy()
        
        # Initial state for EQ-5D (dimension score 1 = good, vas 100 = good)
        initial_eq_dims = {dim: random.randint(2, 4) for dim in EQ5D_DIMENSIONS}
        initial_eq_vas = random.randint(30, 70)
        if evolution_pattern == "improvement": # Start worse
            initial_eq_dims = {dim: random.randint(3, 5) for dim in EQ5D_DIMENSIONS}
            initial_eq_vas = random.randint(10, 40)
        elif evolution_pattern == "deterioration": # Start better
            initial_eq_dims = {dim: random.randint(1, 2) for dim in EQ5D_DIMENSIONS}
            initial_eq_vas = random.randint(60, 90)
        current_eq_dims = initial_eq_dims.copy()
        current_eq_vas = initial_eq_vas

        for i, date in enumerate(dates):
            q_type = random.choice(["OKS", "EQ-5D", "Cochlear Implant"])
            entry = {
                "patient_id": patient_id,
                "date_collecte": date.strftime("%Y-%m-%d"),
                "questionnaire_type": q_type,
                "date_operation": operation_date.strftime("%Y-%m-%d"),
                "periode": "pre" if date < operation_date else "post",
                "reference_professional": None,
                "reference_date": None
            }
            
            # Vérifier si cette entrée correspond à une date de référence
            if q_type in reference_dates:
                ref_info = reference_dates[q_type]
                if abs((date - ref_info["date"]).days) < 7:  # Si la collecte est proche de la date de référence
                    entry["reference_professional"] = ref_info["professional"]
                    entry["reference_date"] = ref_info["date"].strftime("%Y-%m-%d")

            if q_type == "OKS":
                # --- OKS Score Generation ---
                is_post_op = date >= operation_date
                
                # Si c'est une référence à un physiothérapeute, augmenter la probabilité d'amélioration
                if entry["reference_professional"] in ["Physiothérapeute", "Kinésithérapeute"]:
                    change = random.randint(3, 8) if random.random() > 0.1 else random.randint(-1, 2)
                else:
                    if evolution_pattern == "improvement":
                        if is_post_op:
                            change = random.randint(2, 8) if random.random() > 0.1 else random.randint(-2, 2)
                        else:
                            change = random.randint(0, 4) if random.random() > 0.3 else random.randint(-2, 2)
                    elif evolution_pattern == "deterioration":
                        if is_post_op:
                            change = random.randint(-1, 4) if random.random() > 0.3 else random.randint(-3, 1)
                        else:
                            change = -random.randint(2, 6) if random.random() > 0.3 else random.randint(-2, 2)
                    elif evolution_pattern == "fluctuating":
                        if is_post_op:
                            change = random.randint(-3, 6) if random.random() > 0.2 else random.randint(-2, 2)
                        else:
                            change = random.randint(-4, 4)
                    else: # stable
                        if is_post_op:
                            change = random.randint(0, 4) if random.random() > 0.2 else random.randint(-2, 2)
                        else:
                            change = random.randint(-2, 2)
                
                current_oks_score = max(0, min(48, current_oks_score + change))
                entry["score_total"] = current_oks_score

                # Distribute total score among 12 questions (0-4)
                question_scores = []
                remaining_score = current_oks_score
                for q in range(1, 12):
                    if remaining_score <= 0:
                        q_score = 0
                    else:
                        likely_max = min(4, int(remaining_score / (12.0 - q)) + random.choice([-1, 0, 1, 1, 2]))
                        q_score = random.randint(0, max(0, min(4, remaining_score, likely_max)))
                    question_scores.append(q_score)
                    remaining_score -= q_score
                
                last_q_score = max(0, min(4, remaining_score))
                question_scores.append(last_q_score)
                
                entry["score_total"] = sum(question_scores)

                for q_idx, q_score in enumerate(question_scores, 1):
                    entry[f"score_q{q_idx}"] = q_score
                
                # Also update EQ-5D state based on OKS change
                eq_dim_change_tendency = -1 if change > 2 else (1 if change < -2 else 0)
                eq_vas_change_tendency = change 
                for dim in EQ5D_DIMENSIONS:
                    dim_change = eq_dim_change_tendency + random.choice([-1, 0, 0, 1])
                    current_eq_dims[dim] = max(1, min(5, current_eq_dims[dim] + dim_change))
                vas_change = eq_vas_change_tendency + random.randint(-10, 10)
                current_eq_vas = max(0, min(100, current_eq_vas + vas_change))

            elif q_type == "EQ-5D":
                # --- EQ-5D Score Generation ---
                is_post_op = date >= operation_date
                
                # Si c'est une référence à un professionnel, ajuster les changements
                if entry["reference_professional"]:
                    if entry["reference_professional"] == "Psychologue":
                        # Amélioration plus marquée pour l'anxiété/dépression
                        dim_change_tendency = -1
                        vas_change_tendency = random.randint(5, 15)
                    else:
                        dim_change_tendency = -0.5
                        vas_change_tendency = random.randint(0, 10)
                else:
                    if evolution_pattern == "improvement":
                        if is_post_op:
                            dim_change_tendency = -1
                            vas_change_tendency = random.randint(5, 15)
                        else:
                            dim_change_tendency = -0.5
                            vas_change_tendency = random.randint(0, 10)
                    elif evolution_pattern == "deterioration":
                        if is_post_op:
                            dim_change_tendency = -0.5
                            vas_change_tendency = random.randint(0, 10)
                        else:
                            dim_change_tendency = 1
                            vas_change_tendency = -random.randint(5, 15)
                    elif evolution_pattern == "fluctuating":
                        if is_post_op:
                            dim_change_tendency = random.choice([-1, 0, 0.5])
                            vas_change_tendency = random.randint(-5, 15)
                        else:
                            dim_change_tendency = random.choice([-1, 0, 1])
                            vas_change_tendency = random.randint(-15, 15)
                    else: # stable
                        if is_post_op:
                            dim_change_tendency = -0.5
                            vas_change_tendency = random.randint(0, 10)
                        else:
                            dim_change_tendency = 0
                            vas_change_tendency = random.randint(-5, 5)

                # Apply changes to dimensions
                for dim in EQ5D_DIMENSIONS:
                    dim_change = int(dim_change_tendency) + random.choice([-1, 0, 0, 1])
                    current_eq_dims[dim] = max(1, min(5, current_eq_dims[dim] + dim_change))
                    entry[f"score_eq_{dim}"] = current_eq_dims[dim]

                vas_change = vas_change_tendency + random.randint(-10, 10)
                current_eq_vas = max(0, min(100, current_eq_vas + vas_change))
                entry["score_eq_vas"] = current_eq_vas
                
                # Update OKS state based on EQ-5D change
                avg_dim_change = sum(current_eq_dims[d] - initial_eq_dims[d] for d in EQ5D_DIMENSIONS) / 5.0
                oks_change_from_dims = -int(avg_dim_change * 3)
                oks_change_from_vas = int((current_eq_vas - initial_eq_vas) * 0.2)
                
                oks_change = random.choice([oks_change_from_dims, oks_change_from_vas]) + random.randint(-5, 5)
                current_oks_score = max(0, min(48, current_oks_score + oks_change))
                initial_eq_dims = current_eq_dims.copy()
                initial_eq_vas = current_eq_vas

            elif q_type == "Cochlear Implant":
                # Génération des scores pour le questionnaire sur les implants cochléaires
                is_post_op = date >= operation_date
                
                # Si c'est une référence à un audiologiste ou orthophoniste, augmenter la probabilité d'amélioration
                if entry["reference_professional"] in ["Audiologiste", "Orthophoniste"]:
                    change_tendency = -1.5
                else:
                    if evolution_pattern == "improvement":
                        if is_post_op:
                            change_tendency = -1.5
                        else:
                            change_tendency = -0.5
                    elif evolution_pattern == "deterioration":
                        if is_post_op:
                            change_tendency = -0.5
                        else:
                            change_tendency = 1
                    elif evolution_pattern == "fluctuating":
                        if is_post_op:
                            change_tendency = random.choice([-1, -0.5, 0])
                        else:
                            change_tendency = random.choice([-1, 0, 1])
                    else:  # stable
                        if is_post_op:
                            change_tendency = -0.5
                        else:
                            change_tendency = 0

                # Mettre à jour les scores pour chaque question
                for category, questions in COCHLEAR_IMPLANT_CATEGORIES.items():
                    for q in questions:
                        change = int(change_tendency) + random.choice([-1, 0, 0, 1])
                        current_cochlear_scores[q] = max(1, min(4, current_cochlear_scores[q] + change))
                        entry[f"score_{q}"] = current_cochlear_scores[q]

                # Ajuster les scores en fonction de la cohérence entre les catégories
                if "academic_2" in current_cochlear_scores and "fatigue_2" in current_cochlear_scores:
                    if current_cochlear_scores["academic_2"] < 3:
                        current_cochlear_scores["fatigue_2"] = min(4, current_cochlear_scores["fatigue_2"] + 1)
                    else:
                        current_cochlear_scores["fatigue_2"] = max(1, current_cochlear_scores["fatigue_2"] - 1)
                    entry["score_fatigue_2"] = current_cochlear_scores["fatigue_2"]

                if "emotional_1" in current_cochlear_scores:
                    social_score = sum(current_cochlear_scores.get(q, 2) for q in COCHLEAR_IMPLANT_CATEGORIES["social"]) / 4
                    academic_score = sum(current_cochlear_scores.get(q, 2) for q in COCHLEAR_IMPLANT_CATEGORIES["academic"]) / 5
                    fatigue_score = sum(current_cochlear_scores.get(q, 2) for q in COCHLEAR_IMPLANT_CATEGORIES["fatigue"]) / 2
                    
                    avg_score = (social_score + academic_score + fatigue_score) / 3
                    emotional_change = 1 if avg_score < 2.5 else -1
                    current_cochlear_scores["emotional_1"] = max(1, min(4, current_cochlear_scores["emotional_1"] + emotional_change))
                    entry["score_emotional_1"] = current_cochlear_scores["emotional_1"]

            data.append(entry)
        
        # Stocker les informations du patient
        patients_info[patient_id] = {
            "date_operation": operation_date.strftime("%Y-%m-%d"),
            "evolution_pattern": evolution_pattern,
            "references": reference_dates
        }
    
    # Création du DataFrame et sauvegarde en CSV
    df = pd.DataFrame(data)
    
    # Define columns order - ensure all possible score columns are included
    oks_q_cols = [f"score_q{i}" for i in range(1, 13)]
    eq_dim_cols = [f"score_eq_{dim}" for dim in EQ5D_DIMENSIONS]
    cochlear_cols = [f"score_{q}" for category in COCHLEAR_IMPLANT_CATEGORIES.values() for q in category]
    cols = ["patient_id", "date_collecte", "date_operation", "periode", "questionnaire_type", 
            "reference_professional", "reference_date", "score_total"] + oks_q_cols + eq_dim_cols + ["score_eq_vas"] + cochlear_cols
    
    # Add missing columns with NaN
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[cols] # Reorder
    df.sort_values(by=["patient_id", "date_collecte"], inplace=True)
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    # Sauvegarder les informations des patients dans un fichier séparé
    patients_df = pd.DataFrame.from_dict(patients_info, orient='index')
    patients_df.to_csv("patients_info.csv", index=True, index_label="patient_id")
    
    print(f"Données synthétiques (OKS, EQ-5D & Cochlear Implant) générées et sauvegardées dans {file_path}")
    print(f"Informations des patients sauvegardées dans patients_info.csv")
    return df

if __name__ == "__main__":
    generate_synthetic_data() 