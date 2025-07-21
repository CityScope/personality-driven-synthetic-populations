"""
dataset_analysis.py

This script analyzes the output preference datasets (chosen vs rejected responses) generated from
LLM simulations of personality profiles. It performs the following for each dataset:

1. Loads JSON data and computes basic statistics.
2. Plots histograms of alignment scores for chosen and rejected responses.
3. Calculates how many chosen responses surpass a defined alignment threshold.
4. Computes delta score (chosen - rejected) distributions.
5. Evaluates trait-level alignment between target and chosen vectors.

Outputs:
- Histogram plots (saved as PNG)
- Terminal statistics summaries
- Count of perfectly aligned chosen responses per dataset

Make sure the `answers_to_datasets` folder exists and contains the *_preference_dataset.json files.

Author: José Miguel Nicolás García
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd

# Dataset names (without extension)
dataset_names = [
    "diplomat_preference_dataset",
    "analyst_preference_dataset",
    "explorer_preference_dataset",
    "sentinel_preference_dataset"
]


umbral_chosen = 0.95 # Score threshold to consider a chosen response "strongly aligned"
trait_threshold = 0.25 # Trait-level similarity threshold for evaluating well alignment

# Folder containing input JSONs and where plots will be saved
input_dir = "./answers_to_datasets"
os.makedirs(input_dir, exist_ok=True)

# Big Five Traits
traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

for dataset_name in dataset_names:
    print(f"\n=== Processing dataset:  {dataset_name} ===\n")

    # 1. Load dataset from JSON
    input_path = os.path.join(input_dir, f"{dataset_name}.json")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Convert to DataFrame for analysis
    df = pd.DataFrame(data)

    # 3. Compute mean alignment scores
    mean_chosen = df["score_chosen"].mean()
    mean_rejected = df["score_rejected"].mean()

    # 4. Plot histogram of alignment scores (chosen vs rejected)
    plt.figure(figsize=(10, 5))
    plt.hist(df["score_chosen"], bins=10, alpha=0.7, label=f"Chosen (mean={mean_chosen:.2f})")
    plt.hist(df["score_rejected"], bins=10, alpha=0.7, label=f"Rejected (mean={mean_rejected:.2f})")
    plt.xlabel("Alignment Score")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Alignment Scores - {dataset_name}")
    plt.legend()
    plt.grid(True)
    output_path_scores = os.path.join(input_dir, f"{dataset_name}_histogram_scores.png")
    plt.savefig(output_path_scores)
    plt.close()

    # 5. Count how many chosen responses exceed alignment threshold
    count_above = (df["score_chosen"] > umbral_chosen).sum()
    total = len(df)
    print(f"→ {count_above} of {total} 'chosen' responses ({count_above/total:.1%}) have score > {umbral_chosen}")

    # 6. Compute delta score (difference between chosen and rejected)
    df["delta_score"] = df["score_chosen"] - df["score_rejected"]

    # 7. Plot histogram of delta scores
    plt.figure(figsize=(10, 5))
    plt.hist(df["delta_score"], bins=20, color='orange', alpha=0.85)
    plt.xlabel("Delta Score (chosen - rejected)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Delta Scores - {dataset_name}")
    plt.grid(True)
    output_path_delta = os.path.join(input_dir, f"{dataset_name}_histogram_delta_score.png")
    plt.savefig(output_path_delta)
    plt.close()

     # 8. Count responses where chosen vector is aligned with target on all traits
    count_perfect_alignment = sum(
        all(abs(example["chosen_vector"][trait] - example["target_vector"][trait]) <= trait_threshold
            for trait in traits)
        for example in data
    )
    print(f"→ {count_perfect_alignment} of {total} 'chosen' responses "
          f"({count_perfect_alignment/total:.1%}) are within ±{trait_threshold} on all 5 traits.")

    # 9. Display summary statistics in console
    print("\nBasic statistics for alignment scores:")
    print(df[["score_chosen", "score_rejected", "delta_score"]].describe())
    print(f"\nHistograms saved at:\n- {output_path_scores}\n- {output_path_delta}\n")
