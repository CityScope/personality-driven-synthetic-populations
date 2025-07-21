"""
sentiments.py
This script analyzes emotional tone in generated responses from various MBTI-style
language model personas (Analyst, Diplomat, Explorer, Sentinel) using the
GoEmotions classifier (`SamLowe/roberta-base-go_emotions` from HuggingFace).

Steps performed:
1. Load model responses from JSON files.
2. Run GoEmotions classification on the responses.
3. Compute average emotion scores per label for each model.
4. Visualize and save a bar chart comparing mean emotion scores across models.

Outputs:
- PNG plot comparing mean emotional scores.
Author: José Miguel Nicolás García
"""


import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import defaultdict
import os

# === MODEL CONFIGURATION ===
model_files = {
    "Analyst": "./Finetuning/generated_responses_Analyst.json",
    "Diplomat": "./Finetuning/generated_responses_Diplomat_001-b.json",
    "Explorer": "./Finetuning/generated_responses_Explorer.json",
    "Sentinel": "./Finetuning/generated_responses_Sentinel.json"
}

model_colors = {
    "Analyst": "#d1a1b7",    
    "Diplomat": "#9ac172",   
    "Explorer": "#e4c640",  
    "Sentinel": "#75cbcc"    
}

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})


goemotions_scores = defaultdict(dict)


def load_answers(json_path):
    """Loads answer text data from a given JSON file.

    Args:
        json_path (str): Path to the JSON file containing model responses.

    Returns:
        List[str]: A list of non-empty 'answer' fields.
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    return [entry.get("answer", "") for entry in data if entry.get("answer")]


def analyze_goemotions(descriptions, model_name):
    """Runs the GoEmotions classifier on a list of descriptions and stores mean emotion scores.

    Args:
        descriptions (List[str]): A list of textual responses to analyze.
        model_name (str): Identifier for the model being analyzed.
    """
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    results = classifier(descriptions)

    sentiment_scores = defaultdict(list)
    for entry in results:
        for sentiment in entry:
            sentiment_scores[sentiment['label']].append(sentiment['score'])

    mean_scores = {k: np.mean(v) for k, v in sentiment_scores.items()}
    goemotions_scores[model_name] = mean_scores


def plot_emotion_means(data_dict, title, ylabel, filename):
    """Plots a bar chart comparing mean emotion scores across models.

    Args:
        data_dict (dict): Dictionary mapping model names to label:mean_score dictionaries.
        title (str): Title for the plot.
        ylabel (str): Y-axis label.
        filename (str): Output filename (PNG) without extension.
    """
    labels = sorted(set(k for model in data_dict for k in data_dict[model]))
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (model_name, values) in enumerate(data_dict.items()):
        heights = [values.get(label, 0) for label in labels]
        offset = (i - len(data_dict)/2) * width + width/2
        ax.bar(x + offset, heights, width=width, label=model_name, color=model_colors[model_name])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f"{filename}.png")
    plt.close()

if __name__ == "__main__":

    for model_name, json_path in model_files.items():
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue

        print(f"Processing: {model_name}")
        answers = load_answers(json_path)
        if answers:
            analyze_goemotions(answers, model_name)

    # Plot mean scores for all emotions
    plot_emotion_means(goemotions_scores, "Emotions Mean Scores", "Mean Score", "GoEmotions_Comparison_Personalities")



