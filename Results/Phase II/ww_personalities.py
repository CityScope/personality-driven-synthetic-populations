"""
ww_personalities.py
This script analyzes generated responses from different MBTI-style personality models 
(Analyst, Diplomat, Explorer, Sentinel). It performs the following:

1. Loads the responses from JSON files.
2. Cleans and deduplicates the text data.
3. Generates word clouds using filtered text (excluding stop words).
4. Plots and saves the word clouds as a PNG image.
5. Prints the top N most frequent (non-stopword) words per model.

The main goal is to visualize and compare linguistic patterns across model-generated answers.
Author: José Miguel Nicolás García
"""
import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# === FILE AND COLOR CONFIGURATION ===
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


# Set of stop words to exclude from analysis
stop_words = {
        'a', 'an', 'the', 'and', 'is', 'of', 'to', 'with', 'in', 'on', 'for', 'from',
        'at', 'by', 'as', 'this', 'that', 'it', 'he', 'she', 'they', 'we', 'who',
        'but', 'or', 'his', 'her', 'its', 'our', 'their', 'be', 'been', 'being', 
        'financial'
    }

TOP_N = 40  # Number of top words to show per model

# === TEXT CLEANING AND PROCESSING ===
def clean_and_dedup(text):
    """Removes punctuation, converts to lowercase, and removes repeated words.

    Args:
        text (str): Raw input string.

    Returns:
        str: Cleaned string with duplicates removed, preserving word order.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    return ' '.join(dict.fromkeys(words))


def generate_wordcloud_and_counts(texts, color):
    """Generates a word cloud and word frequency counter from a list of texts.

    Args:
        texts (List[str]): List of strings (answers) to process.
        color (str): Hex color code used for all words in the cloud.

    Returns:
        tuple: WordCloud object and Counter of word frequencies (excluding stop words).
    """
    cleaned_texts = [clean_and_dedup(t) for t in texts]
    all_text = " ".join(cleaned_texts)
    
    
    # Remove stopwords from the combined text
    filtered_words = [word for word in all_text.split() if word not in stop_words]
    filtered_text = " ".join(filtered_words)

    # Generate word cloud with custom color
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=lambda *args, **kwargs: color
    ).generate(filtered_text)

    word_counts = Counter(filtered_words)
    return wordcloud, word_counts



# === LOAD AND PROCESS ALL MODELS ===
clouds = {}
counts = {}

for model_name, file_path in model_files.items():
    with open(file_path, "r") as f:
        data = json.load(f)

    answers = [entry.get("answer", "") for entry in data if entry.get("answer")]
    cloud, word_count = generate_wordcloud_and_counts(answers, model_colors[model_name])
    clouds[model_name] = cloud
    counts[model_name] = word_count

# === PLOTTING FUNCTION ===
def plot_wordclouds(clouds_dict, output_filename):
    """Plots and saves one word cloud per model as a single vertical image.

    Args:
        clouds_dict (dict): Dictionary mapping model names to WordCloud objects.
        output_filename (str): File path to save the final image (PNG format).
    """
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 20))

    for idx, model_name in enumerate(model_files.keys()):
        ax = axes[idx]
        ax.imshow(clouds_dict[model_name], interpolation='bilinear')
        ax.axis('off')

        # Add model name as overlay label
        ax.text(
            0.98, -0.08, model_name, transform=ax.transAxes,
            fontsize=20,
            color=model_colors[model_name],
            ha='right', va='bottom',
            weight='bold',
            alpha=0.9,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.7)
        )

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()


plot_wordclouds(clouds, "WordClouds_MBTI_Answers.png")


# === PRINT TOP-N WORDS ===
def print_top_words(counts_dict, title_prefix):
    """Prints the top N most frequent words (excluding stop words) per model.

    Args:
        counts_dict (dict): Dictionary mapping model names to word Counter objects.
        title_prefix (str): Title label to print above the output section.
    """

    print(f"\n\n===== Top {TOP_N} Words for {title_prefix} =====")
    for model_name in model_files.keys():
        print(f"\n--- {model_name} ---")
        filtered_items = [(word, count) for word, count in counts_dict[model_name].items() if word not in stop_words]
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        for word, count in filtered_items[:TOP_N]:
            print(f"{word} ({count})")


print_top_words(counts, "Answer Field")
