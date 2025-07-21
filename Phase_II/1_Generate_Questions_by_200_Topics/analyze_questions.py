"""
analyze_questions.py
Utility script for analyzing and cleaning a dataset of personality-related questions.

This script performs the following steps:
1. Loads and inspects the original dataset (`personality_questions_dataset.json`).
2. Removes duplicate entries based on question text and saves a cleaned version.
3. Compares against a file of answered questions to extract unanswered items.

All outputs are saved in JSON format for downstream use.

Usage:
    $ python analyze_questions.py

Author: José Miguel Nicolás García
"""

import json
import random
from pathlib import Path

# === Configuration Variables ===

INPUT_QUESTIONS = "personality_questions_dataset.json"          # Original dataset
OUTPUT_UNIQUE = "unique_personality_questions_dataset.json"     # Cleaned dataset
ANSWERS_FILE = "personality_answers_big5.json"                  # File containing answered questions
UNANSWERED_OUTPUT = "unanswered_questions.json"                 # Output with remaining questions


# === Utility Functions ===

def load_json(path):
    """Load a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    """Save a dictionary or list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def analyze_dataset(data):
    """Print high-level statistics about the dataset."""
    total_questions = len(data)
    unique_texts = {entry["question"].strip() for entry in data}
    unique_topics = {entry["topic"] for entry in data}

    print(f"Total questions: {total_questions}")
    print(f"Unique question texts: {len(unique_texts)}")
    print(f"Distinct topics: {len(unique_topics)}")
    print("Topics:")
    for topic in sorted(unique_topics):
        print(f"  - {topic}")

def deduplicate_questions(data):
    """Remove duplicate questions based on cleaned text."""
    seen = set()
    unique_entries = []
    for entry in data:
        question_text = entry["question"].strip()
        if question_text not in seen:
            seen.add(question_text)
            unique_entries.append(entry)
    return unique_entries

def get_unanswered_questions(all_questions, answered_questions):
    """Return questions that have not yet been answered."""
    answered_set = {entry["question"].strip() for entry in answered_questions}
    return [entry for entry in all_questions if entry["question"].strip() not in answered_set]


# === Main Execution ===

def main():
    if not Path(INPUT_QUESTIONS).exists():
        print(f"File not found: {INPUT_QUESTIONS}")
        return

    # Step 1 — Load and analyze full dataset
    raw_data = load_json(INPUT_QUESTIONS)
    analyze_dataset(raw_data)

    # Step 2 — Remove duplicate questions
    unique_data = deduplicate_questions(raw_data)
    save_json(unique_data, OUTPUT_UNIQUE)
    print(f"\nRemoved {len(raw_data) - len(unique_data)} duplicate questions.")
    print(f"Saved {len(unique_data)} unique questions to '{OUTPUT_UNIQUE}'.")

    # Step 3 — Generate unanswered question subset
    print("\nGenerating unanswered test dataset...")
    answered_data = load_json(ANSWERS_FILE)
    unanswered_data = get_unanswered_questions(unique_data, answered_data)
    random.shuffle(unanswered_data)
    save_json(unanswered_data, UNANSWERED_OUTPUT)

    print(f"Unanswered questions: {len(unanswered_data)}")
    print(f"Saved to: {UNANSWERED_OUTPUT}")


if __name__ == "__main__":
    main()
