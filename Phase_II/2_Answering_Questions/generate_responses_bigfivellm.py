"""
personality_simulation.py

This script generates personality-based answers to questions using a language model and evaluates
the responses using the Big Five (OCEAN) psychological model.

It works by:
1. Loading a set of open-ended questions.
2. Generating simulated responses using various Big Five personality vectors.
3. Analyzing each answer with a second LLM to infer the personality expressed.
4. Storing the results in a structured JSON file.

Author: José Miguel Nicolás García
"""


import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os



INPUT_JSON = "./personality_questions_dataset.json"
OUTPUT_JSON = "./personality_answers_big5.json"
URL = "http://localhost:11434/api/generate" #Ollama API
MODEL = "llama3:latest"
MODEL2 = "llama3:latest"


# Load questions and mixed the df order
df = pd.read_json(INPUT_JSON)
df = (
    df.groupby("topic", group_keys=False)
      .apply(lambda x: x.sample(frac=1))  # Mix by topic
      .sample(frac=1)                     # Global mix
      .reset_index(drop=True)
)

#Avoiding answer done questions again

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        output_data = json.load(f)
else:
    output_data = []
    

done_questions = set(entry["question"] for entry in output_data)
df = df[~df["question"].isin(done_questions)]


def is_valid_personality_output(d):
    """
    Validates that the response dictionary from the LLM matches the expected Big Five personality output format.

    Args:
        d (dict): The response dictionary to validate.

    Returns:
        bool: True if the dictionary includes the required structure and trait values; False otherwise.
    """
    if not isinstance(d, dict):
        return False
    if "Score" not in d or "Reflection" not in d:
        return False
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    for t in traits:
        if t not in d["Score"] or not isinstance(d["Score"][t], float):
            return False
        if not (0.0 <= d["Score"][t] <= 1.0):
            return False
        if t not in d["Reflection"] or not isinstance(d["Reflection"][t], str):
            return False
    return True


def personality_detection(text):
    """
    Sends a text response to the LLM and requests a Big Five analysis, returning the scores and trait rationales.

    Args:
        text (str): The user-authored response to be analyzed.

    Returns:
        dict: A dictionary containing two sub-keys:
              - "Score": float values from 0.0 to 1.0 for each Big Five trait.
              - "Reflection": explanatory notes per trait.
    """
    prompt = """
            You are an expert in psychological profiling. Your task is to analyze a given human-authored text and infer the author's personality based on the Big Five personality traits (OCEAN model). 

            The five traits are:
            - Openness 
            - Conscientiousness
            - Extraversion
            - Agreeableness
            - Neuroticism

            Each trait must be scored on a scale from 0.0 (Very Low) to 1.0 (Very High). You must also explain, with a short rationale based on the tone, language, and content of the text, how you derived each score.
            Use the following interpretation scale:

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




            Return a JSON object with the following structure:

            {
            "Reflection": {
                "Openness": "How this trait influences the score",
                "Conscientiousness": "How this trait influences the score",
                "Extraversion": "How this trait influences the score",
                "Agreeableness": "How this trait influences the score",
                "Neuroticism": "How this trait influences the score"
            },
            "Score": {
                "Openness": float (0.0 to 1.0),
                "Conscientiousness": float (0.0 to 1.0),
                "Extraversion": float (0.0 to 1.0),
                "Agreeableness": float (0.0 to 1.0),
                "Neuroticism": float (0.0 to 1.0)
            },
            
            }

            Text to analyze:
            """
    prompt+=text
    payload = {
        "model": MODEL2,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    while True:
        try:
            response = requests.post(URL, json=payload)
            if response.status_code == 200:
                res = json.loads(response.text).get("response", "").strip()
                try:
                    d = json.loads(res)
                    if is_valid_personality_output(d):
                        #print(d)
                        return d
                    else:
                        print("Invalid personality format. Retrying...")
                except json.JSONDecodeError:
                    print("Response not parseable as JSON. Retrying...")
            else:
                print(f"Error HTTP {response.status_code}. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        time.sleep(5)


     

def generate_answer(question, temperature, top_p):
    """
    Generates a base personality-style answer to a given question using the LLM without any specific personality vector.

    Args:
        question (str): The question to answer.
        temperature (float): Sampling temperature for the LLM.
        top_p (float): Top-p (nucleus) sampling value.

    Returns:
        str: The generated answer text.
    """
    prompt = f"""
    You are simulating a personality-rich human response.
    Respond thoughtfully to the following question. Your answer should reflect a realistic and personal perspective.
    Question: "{question}"
    """


    prompt = f"""
    You are a person with a distinct personality. Respond naturally and briefly to the following question. Your answer should reflect a realistic and personal perspective.
    Question: "{question}"
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {    
            "temperature": temperature,
            "top_p": top_p,
        }

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

def is_valid_response(d):
    """
    Validates the structure of a JSON response generated using a Big Five vector prompt.

    Args:
        d (dict): The JSON-decoded dictionary.

    Returns:
        bool: True if the dictionary contains both a valid "Answer" and structured "Reflection" fields.
    """

    return (
        isinstance(d, dict)
        and "Reflection" in d and "Answer" in d
        and isinstance(d["Reflection"], dict)
        and all(
            k in d["Reflection"] and isinstance(d["Reflection"][k], str)
            for k in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        )
        and isinstance(d["Answer"], str)
    )


def generate_answer_vector_based(question, big5vector, temperature, top_p):
    """
    Generates an LLM response that simulates a person with a specific Big Five personality vector.

    Args:
        question (str): The question to respond to.
        big5vector (dict): Dictionary of Big Five traits with float values between 0.0 and 1.0.
        temperature (float): Sampling temperature for the LLM.
        top_p (float): Top-p sampling value.

    Returns:
        dict: A JSON-compatible dictionary containing:
              - "Answer": The generated text response.
              - "Reflection": Trait-based rationale for the response.
    """

    prompt = """
    You are an advanced AI that simulates human personalities based on psychological profiles using the Big Five (OCEAN) model. You will receive:

    - A personality dict in the format: 
    {'Extraversion': float, 'Neuroticism': float, 'Agreeableness': float, 'Conscientiousness': float, 'Openness': float}
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

    "Reflection": {
        "Openness": " How this trait influences the response",
        "Conscientiousness": "How this trait influences the response",
        "Extraversion": "How this trait influences the response",
        "Agreeableness": "How this trait influences the response",
        "Neuroticism": "How this trait influences the response"
    },
    "Answer": "The simulated human response, written in the voice of the personality"
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {    
            "temperature": temperature,
            "top_p": top_p,
        }
    }



    while True:
        try:
            response = requests.post(URL, json=payload)
            if response.status_code == 200:
                text = json.loads(response.text).get("response", "").strip()
                if text:
                    d=dict(json.loads(text))
                    if is_valid_response(d):
                        #print(text)
                        return d
                else:
                    print("Empty response. Retrying...")
            else:
                print(f"Error HTTP {response.status_code}. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        time.sleep(5)



# Define multiple LLM sampling profiles for variation in response style
GENERATION_PROFILES = [
    {"temperature": 0.5, "top_p": 0.95},  #  very secure and less creativity
    {"temperature": 0.7, "top_p": 0.90},  # balanced
    {"temperature": 0.95, "top_p": 0.85},  # more creative
    {"temperature": 0.8, "top_p": 0.95},  # control more neutral
]


#Main part of the script
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
    topic = row["topic"]
    question = row["question"].replace('\\"', '"').strip()

    personality_vectors = [
    {"Openness": 0.9, "Conscientiousness": 0.6, "Extraversion": 0.35, "Agreeableness": 0.25, "Neuroticism": 0.35},  # Analyst
    {"Openness": 0.9, "Conscientiousness": 0.5, "Extraversion": 0.45, "Agreeableness": 0.9, "Neuroticism": 0.6},  # Diplomatic
    {"Openness": 0.25, "Conscientiousness": 0.9, "Extraversion": 0.50, "Agreeableness": 0.65, "Neuroticism": 0.40},  # Sentinel
    {"Openness": 0.4, "Conscientiousness": 0.25, "Extraversion": 0.8, "Agreeableness": 0.65, "Neuroticism": 0.25}   # Explorer
    ]


    answers_by_vector = {}

    for vec in personality_vectors:
        vector_key = "vector_" + "_".join(f"{int(round(vec[trait]*100)):02d}" for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'])

        answers = {}
        for i in range(4):
            final_response = generate_answer_vector_based(question, vec, GENERATION_PROFILES[i]["temperature"], GENERATION_PROFILES[i]["top_p"])
            detected_vector = personality_detection(final_response["Answer"])
            print(i)
            answers[f"answer_{i+1}"] = {
                "final_response": final_response,
                "big5": detected_vector
            }

        answers_by_vector[vector_key] = {
            "target_vector": vec,
            **answers
        }

    #Without vector 
    answers_base = {}
    vector_key="base_LLM"
    for i in range(4):
        final_response = generate_answer(question, GENERATION_PROFILES[i]["temperature"], GENERATION_PROFILES[i]["top_p"])
        detected_vector = personality_detection(final_response)
        print(i)
        answers_base[f"answer_{i+1}"] = {
            "final_response": final_response,
            "big5": detected_vector
        }

    
    answers_by_vector[vector_key] = {
            "target_vector": "base_LLM",
            **answers_base
        }
    entry = {
        "topic": topic,
        "question": question,
        **answers_by_vector,
     
    }
    output_data.append(entry)


    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)



