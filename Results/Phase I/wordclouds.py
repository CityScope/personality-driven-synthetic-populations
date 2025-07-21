
"""
wordcloouds.py
This script processes JSON profile data from several language models
(LLaMA, Dolphin, Mistral, Qwen) to generate and visualize word clouds 
based on two types of textual descriptions:

1. General Descriptions
2. Personality Descriptions (Big Five Traits)

It also prints the top N most frequent words (excluding common stop words) 
used by each model in these descriptions.

Visual outputs:
- Word clouds saved as PNG images for both description types.

Textual outputs:
- Top-N word frequencies printed to console for each model and description type.

Author: José Miguel Nicolás García
"""
import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Mapping of model names to corresponding JSON files containing profile data.
model_files = {
    "LLaMA": "profiles_cambridge-llama3.1_8b.json",
    "Dolphin": "profiles_cambridge-dolphin-llama3_8b.json",
    "Mistral": "profiles_cambridge-mistral_7b.json",
    "Qwen": "profiles_cambridge-qwen3_8b.json"
}

model_colors = {
    "LLaMA": "#4E79A7",   # blue
    "Dolphin": "#59A14F",   # green
    "Mistral":  "#F28E2B",   # orange
    "Qwen": "#E15759"       # red
}

# Number of top words to display in output.
TOP_N = 40  

def clean_and_dedup(text):
    """Cleans a string by removing punctuation and converting to lowercase.
    Also removes duplicate words while preserving order.

    Args:
        text (str): The input string to clean.

    Returns:
        str: A cleaned string with duplicates removed.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    return ' '.join(dict.fromkeys(words))  # Elimina repeticiones en la misma frase

def generate_wordcloud_and_counts(texts):
    """Generates a word cloud and word frequency counter from a list of texts.

    Args:
        texts (List[str]): A list of textual descriptions.

    Returns:
        tuple: A WordCloud object and a Counter dictionary of word frequencies.
    """
    cleaned_texts = [clean_and_dedup(t) for t in texts]
    all_text = " ".join(cleaned_texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    word_counts = Counter(word for text in cleaned_texts for word in text.split())
    return wordcloud, word_counts

# Initialize dictionaries to hold word clouds and word counts
general_clouds = {}
bigfive_clouds = {}
general_counts = {}
bigfive_counts = {}


# Process each model's profile file
for model_name, file_path in model_files.items():
    with open(file_path, "r") as f:
        profiles = json.load(f)

    general_descriptions = [
        p.get("General", {}).get("General Description", "") for p in profiles
    ]
    bigfive_descriptions = [
        p.get("Psychological and Cognitive", {}).get("Personality/Big Five Traits", {}).get("General Big Five Description", "")
        for p in profiles
    ]

    general_descriptions = list(filter(None, general_descriptions))
    bigfive_descriptions = list(filter(None, bigfive_descriptions))

    # Generate word clouds and frequency counts
    general_clouds[model_name], general_counts[model_name] = generate_wordcloud_and_counts(general_descriptions)
    bigfive_clouds[model_name], bigfive_counts[model_name] = generate_wordcloud_and_counts(bigfive_descriptions)


def plot_wordclouds(clouds_dict, output_filename):
    """Plots word clouds for all models and saves the result as a PNG image.

    Args:
        clouds_dict (dict): Dictionary mapping model names to WordCloud objects.
        output_filename (str): Name of the output PNG file.
    """
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 20))

    for idx, model_name in enumerate(model_files.keys()):
        ax = axes[idx]
        ax.imshow(clouds_dict[model_name], interpolation='bilinear')
        ax.axis('off')

        # Add model name as a label in the bottom-right corner
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


# Generate and save word cloud plots
plot_wordclouds(general_clouds,  "WordClouds_GeneralDescriptionsAllModels.png")
plot_wordclouds(bigfive_clouds,  "WordClouds_BigFiveDescriptionsAllModels.png")


def print_top_words(counts_dict, title_prefix):
    """Prints the top N most frequent words for each model,
    excluding common English stop words.

    Args:
        counts_dict (dict): Dictionary mapping model names to Counter objects.
        title_prefix (str): A label to identify the output group (e.g., "General Descriptions").
    """
    stop_words = {
        'a', 'an', 'the', 'and', 'is', 'of', 'to', 'with', 'in', 'on', 'for', 'from',
        'at', 'by', 'as', 'this', 'that', 'it', 'he', 'she', 'they', 'we', 'who',
        'but', 'or', 'his', 'her', 'its', 'our', 'their', 'be', 'been', 'being'
    }

    print(f"\n\n===== Top {TOP_N} Words for {title_prefix} =====")
    for model_name in model_files.keys():
        print(f"\n--- {model_name} ---")
        filtered_items = [(word, count) for word, count in counts_dict[model_name].items() if word not in stop_words]
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        for word, count in filtered_items[:TOP_N]:
            print(f"{word}({count}),")


# Print top-N word frequency tables to console
print_top_words(general_counts, "General Descriptions")
print_top_words(bigfive_counts, "Big Five Descriptions")