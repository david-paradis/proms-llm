import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv

def generate_synthetic_data(num_patients=10, entries_per_patient=(3, 8), file_path="patients_proms.csv"):
    """
    Génère des données synthétiques pour les scores OHS (Oxford Hip Score) et les enregistre dans un fichier CSV.
    
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
            initial_score = random.randint(10, 25)  # Score bas (symptômes modérés à sévères)
        elif evolution_pattern == "deterioration":
            initial_score = random.randint(30, 45)  # Score élevé (bon état initial)
        else:
            initial_score = random.randint(15, 35)  # Score variable
        
        # Génération des entrées pour ce patient
        dates = sorted([start_date + timedelta(days=random.randint(0, 700)) for _ in range(num_entries)])
        
        current_score = initial_score
        
        for i, date in enumerate(dates):
            # Ajustement du score selon le modèle d'évolution
            if evolution_pattern == "improvement":
                # Tendance à l'amélioration
                change = random.randint(2, 8) if random.random() > 0.2 else random.randint(-2, 2)
            elif evolution_pattern == "deterioration":
                # Tendance à la détérioration
                change = -random.randint(2, 8) if random.random() > 0.2 else random.randint(-2, 2)
            elif evolution_pattern == "fluctuating":
                # Fluctuations importantes
                change = random.randint(-8, 8)
            else:  # stable
                # Peu de changement
                change = random.randint(-3, 3)
            
            current_score = max(0, min(48, current_score + change))  # Limiter entre 0 et 48
            
            # Calcul des scores individuels des questions
            # Pour l'OHS, exactement 12 questions avec des scores de 0 à 4 chacune
            question_scores = []
            
            # Distribuer le score total parmi les 12 questions
            # Chaque question doit avoir un score entre 0 et 4
            remaining_score = current_score
            for q in range(1, 12):  # Les 11 premières questions
                if remaining_score <= 0:
                    q_score = 0
                else:
                    max_possible = min(4, remaining_score)
                    q_score = random.randint(0, max_possible)
                    remaining_score -= q_score
                
                question_scores.append(q_score)
            
            # La dernière question prend le reste du score, plafonné à 4
            if remaining_score > 4:
                # Si le reste est > 4, on plafonne à 4 et on ajuste le score total
                question_scores.append(4)
                current_score = sum(question_scores)
            else:
                question_scores.append(remaining_score)
            
            # Ajouter l'entrée aux données
            entry = {
                "patient_id": patient_id,
                "date_collecte": date.strftime("%Y-%m-%d"),
                "questionnaire_type": "OHS",
                "score_total": current_score
            }
            
            # Ajouter les scores individuels des 12 questions
            for q_idx, q_score in enumerate(question_scores, 1):
                entry[f"score_q{q_idx}"] = q_score
            
            data.append(entry)
    
    # Création du DataFrame et sauvegarde en CSV
    df = pd.DataFrame(data)
    df.sort_values(by=["patient_id", "date_collecte"], inplace=True)
    df.to_csv(file_path, index=False)
    
    print(f"Données synthétiques générées et sauvegardées dans {file_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data() 