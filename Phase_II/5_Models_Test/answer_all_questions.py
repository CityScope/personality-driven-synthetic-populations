"""
answer_all_questions.py

This script loads a LoRA-fine-tuned LLaMA 3 model simulating a specific personality type (e.g., Diplomat),
generates personality-rich responses to a dataset of questions, evaluates each response's Big Five personality traits,
and stores the final results in a structured JSON file.

Workflow:
1. Loads a pre-trained base LLaMA 3 model and merges it with a personality-specific LoRA checkpoint.
2. Generates responses using the merged model for a subset of prompts.
3. Sends each response to a local LLM API to infer Big Five personality traits.
4. Saves the full result (question, response, scores) to disk.

Author: José Miguel Nicolás García
"""


import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import requests
import time

# Model ID for generation (can be used to select local model endpoint)
MODEL2 = "llama3.1:8b"
URL = "http://localhost:11434/api/generate"


# Dictionary linking personality types to their respective LoRA checkpoint paths
model_paths = {
    "Analyst": "/mnt/storage/josemi/checkpoints/analystLlama-3-8B-beta001/checkpoint-20" ,
    "Diplomat": "/mnt/storage/josemi/checkpoints/diplomatLlama-3-8B-beta001/checkpoint-50",
    "Explorer": "/mnt/storage/josemi/checkpoints/explorerLlama-3-8B-beta001/checkpoint-40", 
    "Sentinel": "/mnt/storage/josemi/checkpoints/sentinelLlama-3-8B-beta001/checkpoint-30", 
    "LLaMA3": "meta-llama/Meta-Llama-3-8B-Instruct"
}

# === Configuration Section ===
selected_model_key = "Explorer"  # ← Change to select a different LoRA personality model
model_path = model_paths[selected_model_key]  
input_path = "./test.json"
output_path = f"generated_responses_{selected_model_key}_001-b.json"
num_samples = 100


# === Load tokenizer and base model ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load base LLaMA 3 model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter checkpoint and merge with base model
adapter_ckpt = "/mnt/storage/josemi/checkpoints/diplomatLlama-3-8B-beta001/checkpoint-50"
model = PeftModel.from_pretrained(base_model, adapter_ckpt)
model = model.merge_and_unload()


# === Load and sample questions ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)
sampled_data = data[:num_samples] 

# === Create text-generation pipeline ===
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Utilities ===
def is_valid_personality_output(d):
    """
    Checks whether the response from the personality LLM is valid and complete.
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
    Sends the generated response to a local LLM API to infer Big Five personality scores.

    Args:
        text (str): The response text to analyze.

    Returns:
        dict: A JSON object with "Score" (floats 0–1) and "Reflection" (explanations).
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


# === Generate responses and analyze ===
results = []
for i, example in enumerate(sampled_data):
    topic= example["topic"]
    prompt = example["question"]
    # Format prompt using chat template for conversational LLaMA 3
    chat = [ 
        {"role": "system", "content": "You are simulating a personality-rich human response. Respond thoughtfully to the following question. Your answer should reflect a realistic and personal perspective."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    # Generate response
    full_output = gen(formatted_prompt, do_sample=True, temperature=0.7, top_p=0.9, max_length=200)[0]['generated_text']

    if "<|start_header_id|>assistant<|end_header_id|>" in full_output:
        response = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        raise ValueError("Unexpected format in model output")

    # Analyze generated response for personality traits
    big5 = personality_detection(response)

    # Store result in memory and on disk
    results.append({
        "topic": prompt,
        "question": prompt,
        "answer": response,
        "big_5": big5
    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[{i+1}/{num_samples}]  Prompt processed and saved to {output_path}")

print(f"\n{num_samples} responses saved to {output_path}")


