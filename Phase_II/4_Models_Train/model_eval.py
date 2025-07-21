"""
evaluate_dpo_checkpoints.py

This script evaluates a series of DPO-trained LLaMA 3 model checkpoints using a preference dataset.
It uses Hugging Face's TRL `DPOTrainer` to compute the evaluation loss (`eval_loss`) for each checkpoint.

The dataset is structured in a format containing:
- `prompt`: system + user input
- `chosen`: preferred response
- `rejected`: less preferred response

The script selects the checkpoint with the lowest eval loss and prints its path.

Requirements:
- Transformers
- TRL (HuggingFace's Reinforcement Learning library)
- Datasets
- PEFT (for LoRA-based models)
- HuggingFace Hub login token

Input:
- Checkpoints inside a directory (named `checkpoint-XX`)
- JSON file with the preference dataset (3-answer format)

Output:
- Console output indicating evaluation loss per checkpoint
- Selected best checkpoint based on eval_loss

Author: José Miguel Nicolás García
"""

import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from datasets import load_dataset
import json

from huggingface_hub import login
login("")#Use your Hugging Face token

# === Paths and Checkpoint Discovery ===
base_dir = "/mnt/storage/josemi/checkpoints/diplomatLlama-3-8B-beta001-2"
checkpoints = sorted(        # List all checkpoint paths sorted numerically (e.g. checkpoint-10, checkpoint-20...)
    [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)

# === Tokenizer Setup ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === Load and split dataset ===
dataset = load_dataset("json", data_files="./3answers_to_datasets/diplomat_preference_dataset.json", split="train")
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset= split_dataset["train"]
eval_dataset = split_dataset["test"]


# === Format prompts into chat-compatible input ===
chat_template = tokenizer.chat_template

def dataset_format(example):
    """
    Formats a dataset entry with chat prompt + chosen/rejected responses.

    Args:
        example (dict): A sample from the preference dataset.

    Returns:
        dict: Formatted sample with full chat prompt, chosen, and rejected.
    """
    # Add system message
    message = {"role": "system", "content":"You are simulating a personality-rich human response. Respond thoughtfully to the following question. Your answer should reflect a realistic and personal perspective."}
    system = tokenizer.apply_chat_template([message], tokenize=False)

    # Add user question
    message = {"role": "user", "content": example['prompt']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    chosen = example['chosen'] 
    rejected = example['rejected'] 

    # Return full sample
    return {"prompt": system + prompt, "chosen": chosen, "rejected": rejected}

# Apply formatting to the evaluation dataset
eval_dataset = eval_dataset.map(dataset_format, remove_columns=eval_dataset.column_names)

# === DPO Evaluation Configuration ===
eval_config = DPOConfig(
    per_device_eval_batch_size=8,
    max_prompt_length=1024,
    max_length=1024,
    bf16=True,
    remove_unused_columns=False
)

# === Evaluation Loop Over All Checkpoints ===E
best_ckpt = None
best_loss = float("inf")
results = {}

for ckpt_path in checkpoints:
    print(f"Evaluating {ckpt_path}...")

    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

    # Initialize DPOTrainer for evaluation only
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        args=eval_config,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    # Run evaluation and capture loss
    metrics = trainer.evaluate()
    loss = metrics["eval_loss"]
    print(f"{ckpt_path} => eval_loss: {loss:.4f}")
    results[ckpt_path] = loss

    # Track best model
    if loss < best_loss:
        best_loss = loss
        best_ckpt = ckpt_path
    
    # Clean memory
    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

# === Final Best Result ===
print(f"\nBest checkpoint: {best_ckpt} with eval_loss = {best_loss:.4f}")


