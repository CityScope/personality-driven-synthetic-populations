"""
generate_questions.py
Script for generating a dataset of open-ended personality questions using an LLM API.

This script iterates over a predefined list of psychological and behavioral topics,
generates 100 diverse open-ended questions per topic, and stores the output in CSV format.

It ensures:
- Unique questions per topic
- Structured and JSON-compliant responses
- Avoids regeneration of already-processed topics (if file exists)


Author: José Miguel Nicolás García
"""

import pandas as pd
import requests
import json
import os
from tqdm import tqdm
import time

# === Configuration Variables ===

OUTPUT_PATH = "personality_questions_dataset.csv"   # Output file
MODEL = "llama3:latest"                              # LLM model to query
URL = "http://localhost:11436/api/generate"          # Ollama API endpoint
QUESTIONS_PER_TOPIC = 100                            # Questions to generate per topic


# === Dataset Initialization ===

if os.path.exists(OUTPUT_PATH):
    df_existing = pd.read_csv(OUTPUT_PATH)
    processed_topics = set(df_existing["topic"].unique())
    first_write = False
else:
    processed_topics = set()
    first_write = True


# === Topics to Process (trimmed here for brevity; full list loaded in actual script) ===

topics = [  # Add full list of ~160+ topics here...
    "Attitude towards money",
    "Reaction to criticism",
    "Sense of humor",
    "Level of ambition",
    "Relationship with family",
    # ... (truncated for brevity)
]

topics_to_process = [t for t in topics if t not in processed_topics]
print(f"Total topics to process: {len(topics_to_process)}")


# === Prompt Template ===

def build_prompt(topic):
    """Generate prompt to send to the LLM for a specific topic."""
    return f"""
You are an expert in personality psychology and in designing varied and high-quality open-ended questions.

Your task is to generate exactly {QUESTIONS_PER_TOPIC} unique and in-depth questions that explore the following personality topic:

Topic: {topic}

Rules:
- Each question must be deeply relevant to the topic.
- Questions must not repeat or rephrase each other.
- Cover emotional, cognitive, behavioral, moral, and situational aspects.
- Use different styles: reflection, hypothetical scenarios, personal experiences, etc.
- All questions must be open-ended (no yes/no, no multiple-choice).
- The output must be a list of JSON Lines format (one JSON object per line).
- Each object must follow this structure:
{{
  "list": [
    {{"1": "How do you usually react when your routine is disrupted?"}},
    {{"2": "Describe a situation where you had to adapt quickly."}},
    ...
    {{"100": "What kind of life changes do you find hardest to accept?"}}
  ]
}}
"""


# === LLM Interaction ===

def generate_questions_for_topic(topic):
    """Generate a list of questions for a given topic via LLM API."""
    prompt = build_prompt(topic)
    valid_format = False
    correct_number = False

    while not (valid_format and correct_number):
        try:
            response = requests.post(URL, json={
                "model": MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False
            })

            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.text}")
                time.sleep(10)
                continue

            response_data = json.loads(response.text)
            raw_response = response_data["response"].strip()

            # Fix edge cases (missing brackets, malformed endings)
            if raw_response.endswith("]"):
                raw_response += "\n}"
            elif raw_response.endswith("}") and "]" not in raw_response:
                raw_response = raw_response.replace("}", "]\n}")

            parsed = json.loads(raw_response)
            items = parsed["list"]

            valid_format = all(
                isinstance(d, dict) and
                len(d) == 1 and
                list(d.keys())[0].isdigit() and
                isinstance(list(d.values())[0], str)
                for d in items
            )

            if valid_format:
                sorted_items = sorted(
                    ((int(k), v) for d in items for k, v in d.items()),
                    key=lambda x: x[0]
                )[:QUESTIONS_PER_TOPIC]
                correct_number = [k for k, _ in sorted_items] == list(range(1, QUESTIONS_PER_TOPIC + 1))
                if correct_number:
                    return [{str(k): v} for k, v in sorted_items]

        except Exception as e:
            print(f"Error parsing response: {e}")
            valid_format = False
            correct_number = False
            time.sleep(5)


# === Generation Loop ===

for topic in tqdm(topics_to_process, desc="Generating questions", unit="topic"):
    print(f"\n Generating questions for: {topic}")
    questions = generate_questions_for_topic(topic)
    
    batch = []
    for q in questions:
        idx, text = list(q.items())[0]
        batch.append({"topic": topic, "question": text})

    df_batch = pd.DataFrame(batch)
    df_batch.to_csv(OUTPUT_PATH, mode="a", index=False, header=first_write)
    first_write = False









