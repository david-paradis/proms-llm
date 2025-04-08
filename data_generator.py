import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv

# EQ-5D Dimensions
EQ5D_DIMENSIONS = ["mobility", "self_care", "usual_activities", "pain_discomfort", "anxiety_depression"]

def generate_synthetic_data(num_patients=10, entries_per_patient=(3, 8), file_path="patients_proms.csv"):
    """
    Génère des données synthétiques pour les scores OHS et EQ-5D et les enregistre dans un fichier CSV.
    Chaque entrée correspond soit à un OHS, soit à un EQ-5D.
    
    Parameters:
    - num_patients: Nombre de patients à générer
    - entries_per_patient: Tuple contenant (min, max) nombre d'entrées par patient
    - file_path: Chemin où sauvegarder le fichier CSV
    """
    # Initialisation des données
    data = []
    
    # Création d'IDs de patients (ex: P001, P002, etc.)
    patient_ids = [f"P{str(i+1).zfill(3)}" for i in range(num_patients)]
    
    # Date de début (environ 2 ans dans le passé)
    start_date = datetime.now() - timedelta(days=730)
    
    for patient_id in patient_ids:
        # Décider du modèle d'évolution pour ce patient
        evolution_pattern = random.choice(["improvement", "deterioration", "fluctuating", "stable"])
        
        # Déterminer le nombre d'entrées pour ce patient
        num_entries = random.randint(entries_per_patient[0], entries_per_patient[1])
        
        # Premier score (baseline)
        if evolution_pattern == "improvement":
            initial_ohs_score = random.randint(10, 25)  # Score bas (symptômes modérés à sévères)
        elif evolution_pattern == "deterioration":
            initial_ohs_score = random.randint(30, 45)  # Score élevé (bon état initial)
        else:
            initial_ohs_score = random.randint(15, 35)  # Score variable
        
        # Génération des entrées pour ce patient
        dates = sorted([start_date + timedelta(days=random.randint(0, 700)) for _ in range(num_entries)])
        
        current_ohs_score = initial_ohs_score
        
        # Initial state for EQ-5D (dimension score 1 = good, vas 100 = good)
        # Start with moderate problems (scores 2-4) and moderate VAS (30-70)
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
            q_type = random.choice(["OHS", "EQ-5D"])
            entry = {
                "patient_id": patient_id,
                "date_collecte": date.strftime("%Y-%m-%d"),
                "questionnaire_type": q_type,
            }

            if q_type == "OHS":
                # --- OHS Score Generation ---
                if evolution_pattern == "improvement":
                    change = random.randint(2, 8) if random.random() > 0.2 else random.randint(-2, 2)
                elif evolution_pattern == "deterioration":
                    change = -random.randint(2, 8) if random.random() > 0.2 else random.randint(-2, 2)
                elif evolution_pattern == "fluctuating":
                    change = random.randint(-8, 8)
                else: # stable
                    change = random.randint(-3, 3)
                
                current_ohs_score = max(0, min(48, current_ohs_score + change))
                entry["score_total"] = current_ohs_score

                # Distribute total score among 12 questions (0-4)
                question_scores = []
                remaining_score = current_ohs_score
                for q in range(1, 12):
                    if remaining_score <= 0:
                        q_score = 0
                    else:
                        # Heuristic: tend towards scores reflecting the total score range
                        likely_max = min(4, int(remaining_score / (12.0 - q)) + random.choice([-1, 0, 1, 1, 2]))
                        q_score = random.randint(0, max(0, min(4, remaining_score, likely_max)))
                    question_scores.append(q_score)
                    remaining_score -= q_score
                
                # Last question takes remainder, capped at 4
                last_q_score = max(0, min(4, remaining_score))
                question_scores.append(last_q_score)
                
                # Recalculate total score based on actual question scores
                entry["score_total"] = sum(question_scores)

                for q_idx, q_score in enumerate(question_scores, 1):
                    entry[f"score_q{q_idx}"] = q_score
                
                # Also update EQ-5D state based on OHS change for next potential EQ-5D entry
                # Inverse relationship: higher OHS change -> lower EQ dim change & higher VAS change
                eq_dim_change_tendency = -1 if change > 2 else (1 if change < -2 else 0)
                eq_vas_change_tendency = change 
                for dim in EQ5D_DIMENSIONS:
                    dim_change = eq_dim_change_tendency + random.choice([-1, 0, 0, 1])
                    current_eq_dims[dim] = max(1, min(5, current_eq_dims[dim] + dim_change))
                vas_change = eq_vas_change_tendency + random.randint(-10, 10)
                current_eq_vas = max(0, min(100, current_eq_vas + vas_change))

            elif q_type == "EQ-5D":
                # --- EQ-5D Score Generation ---
                # Determine change tendency based on pattern
                dim_change_tendency = 0 # 1:worse, -1:better
                vas_change_tendency = 0 # positive:better, negative:worse
                if evolution_pattern == "improvement":
                    dim_change_tendency = -1
                    vas_change_tendency = random.randint(5, 15)
                elif evolution_pattern == "deterioration":
                    dim_change_tendency = 1
                    vas_change_tendency = -random.randint(5, 15)
                elif evolution_pattern == "fluctuating":
                    dim_change_tendency = random.choice([-1, 0, 1])
                    vas_change_tendency = random.randint(-15, 15)
                else: # stable
                    dim_change_tendency = 0
                    vas_change_tendency = random.randint(-5, 5)

                # Apply changes to dimensions (1-5, 1=best)
                for dim in EQ5D_DIMENSIONS:
                    # Apply tendency with some randomness
                    dim_change = dim_change_tendency + random.choice([-1, 0, 0, 1])
                    current_eq_dims[dim] = max(1, min(5, current_eq_dims[dim] + dim_change))
                    entry[f"score_eq_{dim}"] = current_eq_dims[dim]

                # Apply change to VAS (0-100, 100=best)
                vas_change = vas_change_tendency + random.randint(-10, 10)
                current_eq_vas = max(0, min(100, current_eq_vas + vas_change))
                entry["score_eq_vas"] = current_eq_vas
                
                # Also update OHS state based on EQ-5D change for next potential OHS entry
                # Higher EQ dim change -> lower OHS change
                # Higher VAS change -> higher OHS change
                avg_dim_change = sum(current_eq_dims[d] - initial_eq_dims[d] for d in EQ5D_DIMENSIONS) / 5.0
                ohs_change_from_dims = -int(avg_dim_change * 3) # Heuristic scaling
                ohs_change_from_vas = int((current_eq_vas - initial_eq_vas) * 0.2) # Heuristic scaling
                
                ohs_change = random.choice([ohs_change_from_dims, ohs_change_from_vas]) + random.randint(-5, 5)
                current_ohs_score = max(0, min(48, current_ohs_score + ohs_change))
                initial_eq_dims = current_eq_dims.copy() # Update baseline for next calc
                initial_eq_vas = current_eq_vas

            data.append(entry)
    
    # Création du DataFrame et sauvegarde en CSV
    df = pd.DataFrame(data)
    
    # Define columns order - ensure all possible score columns are included
    ohs_q_cols = [f"score_q{i}" for i in range(1, 13)]
    eq_dim_cols = [f"score_eq_{dim}" for dim in EQ5D_DIMENSIONS]
    cols = ["patient_id", "date_collecte", "questionnaire_type", "score_total"] + ohs_q_cols + eq_dim_cols + ["score_eq_vas"]
    
    # Add missing columns with NaN
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[cols] # Reorder
    df.sort_values(by=["patient_id", "date_collecte"], inplace=True)
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC) # Quote non-numeric to handle NaNs properly
    
    print(f"Données synthétiques (OHS & EQ-5D) générées et sauvegardées dans {file_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data() 