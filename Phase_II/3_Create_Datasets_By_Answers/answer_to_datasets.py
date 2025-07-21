"""
answer_to_datasets.py

This script processes a dataset of LLM-generated responses evaluated through Big Five personality vectors.
It selects the most and least personality-aligned responses per question and personality type.

The final output is:
- A JSON file for each personality type (Analyst, Diplomat, Sentinel, Explorer), containing:
    - The prompt (question)
    - The most aligned response ("chosen")
    - The least aligned response ("rejected")
    - Cosine-like similarity scores
    - Vectors used in evaluation
- A Hugging Face-compatible CSV format for preference training.

Usage:
    Place the `personality_answers_big5.json` file in the root directory before running this script.

Author: José Miguel Nicolás García

"""


import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Load the generated responses and personality scores
with open("personality_answers_big5.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Big Five traits (OCEAN model)
traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


# Datasetby type of vector
datasets = {
    "analyst": [],
    "diplomat": [],
    "sentinel": [],
    "explorer": [],
}

vector_map = {
    "vector_90_60_35_25_35": "analyst",
    "vector_90_50_45_90_60": "diplomat",
    "vector_25_90_50_65_40": "sentinel",
    "vector_40_25_80_65_25": "explorer",
}

#Common use Metrics
# Cosine similarity
# def similarity(vec1, vec2):
#     v1 = np.array([vec1[t] for t in traits]).reshape(1, -1)
#     v2 = np.array([vec2[t] for t in traits]).reshape(1, -1)
#     return float(cosine_similarity(v1, v2)[0][0])

#Exponential similarity+ mean
# def similarity(vec1, vec2, alpha=7.0):
#     penalties = [np.exp(-alpha * (vec1[t] - vec2[t]) ** 2) for t in traits]
#     return float(np.mean(penalties))



#-------------Designed similarity metric-------------
#Exponential similarity+ geometric mean
def similarity(vec1, vec2, alpha=5.0):
    """
    Calculates exponential similarity between two Big Five personality vectors.

    Args:
        vec1 (dict): First Big Five trait vector.
        vec2 (dict): Second Big Five trait vector.
        alpha (float): Smoothing factor for penalty (default is 5.0).

    Returns:
        float: Geometric mean of trait-wise similarity (0.0 to 1.0).
    """
    penalties = [np.exp(-alpha * (vec1[t] - vec2[t]) ** 2) for t in traits]
    product = np.prod(penalties)
    return float(product ** (1 / len(penalties)))


# Dataset preprocessing
for entry in data:
    # Get the 4 base responses from the vanilla LLM (no vector prompt)
    base_answers = [entry["base_LLM"][f"answer_{i+1}"] for i in range(4)]

    for key, label in vector_map.items():
        if key not in entry:
            continue

        vector_answers = [entry[key][f"answer_{i+1}"] for i in range(4)]
        target_vector = entry[key]["target_vector"]

        # Select the answer most similar to the target personality vector
        vector_scores = [similarity(target_vector, ans["big5"]["Score"]) for ans in vector_answers]
        idx_best = int(np.argmax(vector_scores))
        chosen = vector_answers[idx_best]["final_response"]["Answer"]
        score_chosen = vector_scores[idx_best]


        # Choose the worst of ALL answers (base + other vectors)
 
        all_candidates = []
        for ans in base_answers:
            all_candidates.append({
                "vector": ans["big5"]["Score"],
                "score": similarity(target_vector, ans["big5"]["Score"]),
                "text": ans["final_response"] if isinstance(ans["final_response"], str) else ans["final_response"]["Answer"]
            })

        for other_key in vector_map:
            if other_key != key and other_key in entry:
                for i in range(4):
                    ans = entry[other_key][f"answer_{i+1}"]
                    text = ans["final_response"]["Answer"] if isinstance(ans["final_response"], dict) else ans["final_response"]
                    all_candidates.append({
                        "vector": ans["big5"]["Score"],
                        "score": similarity(target_vector, ans["big5"]["Score"]),
                        "text": text,
                    })

        # Choose worsts and best scores
        worst = min(all_candidates, key=lambda x: x["score"])
        rejected = worst["text"]
        score_rejected = worst["score"]
        vector_rejected = worst["vector"]


        #Appending one dataset entry
        datasets[label].append({
            "prompt": entry["question"],
            "target_vector": target_vector,
            "chosen": chosen,
            "chosen_vector": vector_answers[idx_best]["big5"]["Score"],
            "score_chosen": score_chosen,
            "rejected": rejected,
            "rejected_vector": vector_rejected,
            "score_rejected": score_rejected
        })

# Saving JSON
for label, rows in datasets.items():
    with open(f"./answers_to_datasets/{label}_preference_dataset.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

        

# --- Hugging Face format conversion ---

def prompt_formating(prompt):
    """
    Formats a prompt into Hugging Face conversation format.

    Args:
        prompt (str): The input user prompt.

    Returns:
        list: List with a single dict representing user message.
    """ 
    return [{"role": "user", "content": prompt}]

def answer_formating(answer):
    """
    Formats an answer into Hugging Face assistant message format.

    Args:
        answer (str): The generated assistant answer.

    Returns:
        list: List with a single dict representing assistant message.
    """
    return [{"role": "assistant", "content": answer}]

# Convert JSON datasets to Hugging Face CSV format
for label, rows in datasets.items():
    df = pd.read_json(f"./answers_to_datasets/{label}_preference_dataset.json")
    df_output=pd.DataFrame()
    df_output['prompt']=df['prompt'].apply(prompt_formating)
    df_output['chosen']=df['chosen'].apply(answer_formating)
    df_output['rejected']=df['rejected'].apply(answer_formating)
    df_output.to_csv(f"./answers_to_datasets/hf_format_{label}_preference_dataset.csv")
