"""
generate_responses_with_bert.py

This script generates human-like responses to open-ended personality-related questions using a language model (LLM),
and analyzes those responses using a pre-trained BERT-based classifier for the Big Five personality traits.

Unlike other versions that rely on an LLM for both response generation and personality analysis, this version uses:

- LLM (via local API) for generating responses.
- BERT-based models (Hugging Face) for evaluating personality traits (classification based on text).

Output is stored in a structured JSON file, including both the generated response and detected Big Five scores.
Author: José Miguel Nicolás García
"""

import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as Fx


# Configuration
token= "" #Use your Hugging Face Token

INPUT_JSON = "personality_questions_dataset.json"
OUTPUT_JSON = "personality_answers_big5.json"
URL = "http://localhost:11436/api/generate"
MODEL = "llama3:latest"
VECTOR_1= {
        "Extroversion": 0.9,
        "Neuroticism": 0.2,
        "Agreeableness": 0.9,
        "Conscientiousness": 0.3,
        "Openness": 0.1
      }

# Load questions
df = pd.read_json(INPUT_JSON)



#Past tries with Bert Models for Big Five Traits Classifiication
# Load Big Five Bert model
tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")


# def personality_detection(text):
#    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
#    with torch.no_grad():
#        outputs = model(**inputs)
#        logits = outputs.logits.squeeze()
#        scores = torch.sigmoid(logits).tolist()

#    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
#    return {label_names[i]: float(scores[i]) for i in range(len(label_names))}


def personality_detection_BertPersonality(text):
    """
    Sends a human-like response to the LLM to detect personality based on the Big Five traits.

    Args:
    text (str): The response text to analyze.

    Returns:
    dict: JSON object containing 'Score'  for each Big Five trait.
    """
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()
    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: predictions[i] for i in range(len(label_names))}

    return result



#Microsoft bert-like big 5 classification
def personality_detection(text, threshold=0.05, endpoint= 1.0):
    """
    Sends a human-like response to the LLM to detect personality based on the Big Five traits.

    Args:
        text (str): The response text to analyze.

    Returns:
        dict: JSON object containing 'Score'  for each Big Five trait.
    """
    tokenizer = AutoTokenizer.from_pretrained ("Nasserelsaman/microsoft-finetuned-personality",token=token)
    model = AutoModelForSequenceClassification.from_pretrained ("Nasserelsaman/microsoft-finetuned-personality",token=token)
    
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()
    
    # Get raw logits
    logits = model(**inputs).logits
    
    # Apply sigmoid to squash between 0 and 1
    probabilities = torch.sigmoid(logits)
    
    # Set values less than the threshold to 0.05
    predictions[predictions < threshold] = 0.05
    predictions[predictions > endpoint] = 1.0
    
    label_names = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
    result = {label_names[i]: f"{predictions[i]*100:.0f}%" for i in range(len(label_names))}
    
    return result
     


#Answers
def generate_answer(question):
    """
    Generates a human-like answer from the base language model (no personality vector).

    Args:
        question (str): The input question to answer.
        temperature (float): LLM temperature parameter.
        top_p (float): LLM top_p parameter.

    Returns:
        str: The generated answer.
    """

    prompt = f"""
    You are simulating a personality-rich human response.
    Respond thoughtfully to the following question. Your answer should reflect a realistic and personal perspective.
    Question: "{question}"

    Answer:
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    while True:
        try:
            response = requests.post(URL, json=payload)
            if response.status_code == 200:
                text = json.loads(response.text).get("response", "").strip()
                if text:
                    return text
                else:
                    print("Empty response. Retrying...")
            else:
                print(f"Error HTTP {response.status_code}. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        time.sleep(5)


def generate_answer_vector_based(question, big5vector):
    """
    Generates a personality-based answer using a Big Five vector profile.

    Args:
    question (str): The input question to answer.
    big5vector (dict): Dictionary containing trait values between 0.0 and 1.0.
    temperature (float): LLM temperature parameter.
    top_p (float): LLM top_p parameter.

    Returns:
    dict: JSON object containing 'Answer' and 'Reflection'.
    """
     
    prompt = """
    You are an advanced AI that simulates human personalities based on psychological profiles using the Big Five (OCEAN) model. You will receive:

    - A personality dict in the format: 
    {'Extroversion': float, 'Neuroticism': float, 'Agreeableness': float, 'Conscientiousness': float, 'Openness': float}
    - A question that this person is being asked.

    Each trait is scored from 0.00 to 1.00. Use the following interpretation scale:

    - 0.00 – 0.19 → Very Low  
    - 0.20 – 0.39 → Low  
    - 0.40 – 0.59 → Medium  
    - 0.60 – 0.79 → High  
    - 0.80 – 1.00 → Very High

    Here’s how each trait should be understood and reflected in the simulated response:

    1.Openness to Experience

    - Very Low: Prefers concrete facts and familiar routines; communication is literal and pragmatic. Strengths: operational consistency, quick execution of tried‑and‑true methods. Watch‑outs: resists innovation, may overlook abstract strategy.

    - Low: Practical and conventional, open to incremental improvements backed by evidence. Strengths: dependable, realistic problem‑solver. Watch‑outs: underestimates creative solutions, limited curiosity.

    - Medium: Balances tradition with selective exploration; weighs novelty against practicality. Strengths: adaptable, versatile thinker. Watch‑outs: can hesitate on bold pivots, risk of analysis paralysis.

    - High: Curious, imaginative, enjoys diverse perspectives and creative problem‑solving. Strengths: idea generation, visionary thinking. Watch‑outs: may lose focus on details or feasibility.

    - Very High: Intensely inventive and exploratory; thrives on ambiguity and cross‑disciplinary links. Strengths: innovation leadership, paradigm shifts. Watch‑outs: prone to constant ideation, difficulty finalising projects.

    2.Conscientiousness

    - Very Low: Disorganised, lives “in the moment,” often misses deadlines. Strengths: flexible, improvises under chaos. Watch‑outs: reliability issues, frequent errors.

    - Low: Casual planner, prefers spontaneity, may procrastinate. Strengths: adaptable in fluid situations. Watch‑outs: overcommitment, weak follow‑through.

    - Medium: Uses basic structure but tolerates slip‑ups; meets key deadlines. Strengths: balanced productivity, approachable. Watch‑outs: average output consistency.

    - High: Goal‑oriented, organised, anticipates obstacles, follows processes. Strengths: dependable, high quality control. Watch‑outs: can be rigid, perfection delays delivery.

    - Very High: Exceptionally disciplined, perfectionistic, zero‑defect mentality. Strengths: meticulous accuracy, long‑term execution. Watch‑outs: micromanagement, burnout, intolerance of others’ mistakes.

    3. Extraversion

    - Very Low: Deeply introverted, avoids social interaction, excels in solitary deep‑focus tasks. Strengths: analytical depth, calm under pressure. Watch‑outs: networking neglect, perceived aloofness.

    - Low: Quiet, prefers small groups, collaborates asynchronously. Strengths: thoughtful listener, measured insights. Watch‑outs: may be overlooked, slower to rally teams.

    - Medium: Comfortable both socially and alone; adapts to the context. Strengths: situational flexibility, balanced leadership. Watch‑outs: none major—healthy midpoint.

    - High: Sociable, enthusiastic, leads discussions, energised by others. Strengths: morale booster, rapid network building. Watch‑outs: may dominate airtime, impatience with deliberation.

    - Very High: Highly outgoing and dominant; craves constant interaction. Strengths: charismatic vision‑seller, event energiser. Watch‑outs: distractibility, detail neglect, attention‑seeking.

    4. Agreeableness

    - Very Low: Antagonistic, confrontational, blunt. Strengths: negotiation toughness, critical eye. Watch‑outs: conflict escalation, low team cohesion.

    - Low: Critical, skeptical, freely gives negative feedback. Strengths: realism, risk detection. Watch‑outs: trust issues, perceived cynicism.

    - Medium: Balances cooperation with assertiveness; polite but honest. Strengths: constructive collaboration, healthy boundaries. Watch‑outs: possible decision delays in high‑stakes compromises.

    - High: Warm, empathetic, conflict‑averse, fosters harmony. Strengths: team glue, customer satisfaction. Watch‑outs: difficulty saying “no,” may avoid necessary confrontation.

    - Very High: Exceptionally compassionate and self‑sacrificing; avoids conflict at all costs. Strengths: trust building, strong social capital. Watch‑outs: exploitation risk, personal burnout, prioritisation failures.

    5. Neuroticism

    - Very Low: Emotionally very stable, rarely anxious. Strengths: crisis anchor, steady judgment. Watch‑outs: may underestimate threats, appear unemotional.

    - Low: Composed, handles stress well, quick recovery. Strengths: resilience, optimistic framing. Watch‑outs: occasional complacency.

    - Medium: Normal emotional highs and lows; uses stress as motivator. Strengths: realistic risk appraisal. Watch‑outs: mood variability under heavy load.

    - High: Anxious, sensitive, prone to worry; double‑checks work. Strengths: early threat detection, thorough contingency planning. Watch‑outs: fatigue, indecision, catastrophising.

    - Very High: Emotionally volatile, easily overwhelmed, self‑doubting. Strengths: vigilance can avert disaster if channelled. Watch‑outs: burnout, impaired judgment, strained relationships.


    YOUR TASK

    1. Analyze the given personality vector.
    2. Reflect on how each trait would influence tone, emotional expression, structure, and content of the response.
    3. Then answer the question as that person would naturally respond — without explicitly mentioning the traits. Sound like a real person speaking naturally.

    

    INPUT

    -Big 5 traits: 
    """ +   str(big5vector) + """

    - Question: 
    """ + question + """

    OUTPUT FORMAT

    Respond with a JSON object in the following structure:

    "Reflection": "[A reasoning that describes how each personality trait affects how you answer the question]",
    "Answer": "[The simulated human response, written in the voice of the personality]"

    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    #print(prompt)

    while True:
        try:
            response = requests.post(URL, json=payload)
            if response.status_code == 200:
                text = json.loads(response.text).get("response", "").strip()
                if text:
                    print(text)
                    return dict(json.loads(text))
                else:
                    print("Empty response. Retrying...")
            else:
                print(f"Error HTTP {response.status_code}. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        time.sleep(5)


if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        output_data = json.load(f)
else:
    output_data = []

done_questions = set(entry["question"] for entry in output_data)
df = df[~df["question"].isin(done_questions)]


# Main execution
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
    topic = row["topic"]
    question = row["question"].replace('\\"', '"').strip()

    personality_vectors = [ #Four target vectors
        {"Extroversion": 0.96, "Neuroticism": 0.20, "Agreeableness": 0.45, "Conscientiousness": 0.67, "Openness": 0.76},
        {"Extroversion": 0.15, "Neuroticism": 0.85, "Agreeableness": 0.30, "Conscientiousness": 0.90, "Openness": 0.20},
        {"Extroversion": 0.50, "Neuroticism": 0.50, "Agreeableness": 0.50, "Conscientiousness": 0.50, "Openness": 0.50},
        {"Extroversion": 0.25, "Neuroticism": 0.40, "Agreeableness": 0.90, "Conscientiousness": 0.30, "Openness": 0.80}
    ]

    answers_by_vector = {}

    for vec in personality_vectors:
        vector_key = "vector_" + "_".join(f"{int(round(vec[trait]*100)):02d}" for trait in ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness'])

        answers = {}
        for i in range(4):
            final_response = generate_answer_vector_based(question, vec)
            detected_vector = personality_detection(final_response["Answer"])
            answers[f"answer_{i+1}"] = {
                "final_response": final_response,
                "big5": detected_vector
            }

        answers_by_vector[vector_key] = {
            "target_vector": vec,
            **answers
        }

    entry = {
        "topic": topic,
        "question": question,
        **answers_by_vector
    }
    output_data.append(entry)


    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)



