"""
csv_to_json.py
Script to convert a CSV file of personality questions into structured JSON format.

This utility parses each line of the CSV file containing questions and topics, and
generates a JSON array of objects with the following format:

{
  "topic": "...",
  "question": "..."
}

Author: José Miguel Nicolás García
"""

import json

# === File Paths ===

INPUT_PATH = "personality_questions_dataset.csv"
OUTPUT_PATH = "personality_questions_dataset.json"


# === Conversion Logic ===

questions_json = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "," not in line:
            continue  # Skip empty or malformed lines
        topic, question = line.split(",", 1)
        questions_json.append({
            "topic": topic.strip(),
            "question": question.strip()
        })


# === Save Output ===

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(questions_json, f, ensure_ascii=False, indent=2)

print(f"File saved in {OUTPUT_PATH}")
