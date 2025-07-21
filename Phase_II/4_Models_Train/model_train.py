"""
model_train.py

This script fine-tunes a LLaMA 3 model using Direct Preference Optimization (DPO) with LoRA adapters,
based on a preference-formatted dataset. It includes:
- 8-bit quantization using BitsAndBytes for memory efficiency
- Chat-format prompt construction using HuggingFace templates
- Training and evaluation splits
- Weights & Biases logging for training monitoring
- Final model merging and saving

Stages:
1. Load and quantize base model
2. Prepare training and evaluation dataset
3. Run DPO training with LoRA
4. Save adapter and final merged model

Requirements:
- transformers
- datasets
- peft
- trl
- bitsandbytes
- wandb
- huggingface_hub

Author: José Miguel Nicolás García
"""
import os
import wandb
import gc
import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    PeftModel, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import DPOTrainer, setup_chat_format, DPOConfig
import bitsandbytes as bnb
from datasets import load_dataset
from huggingface_hub import login
login("") #Use your hugging face token



###########################1-Load model and tokenizer###########################
# Base and output paths
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
new_model = "/mnt/storage/josemi/checkpoints/diplomatLlama-3-8B-beta001-2" # Directorio donde se guardan los checkpoints durante el entrenamiento (DPOTrainer)
adapter_ckpt = "/mnt/storage/josemi/final_ckpt" # Donde se guarda el adapter LoRA antes de hacer merge
final_model_path = "/mnt/storage/josemi/models/diplomatLlama-3-8B-beta001-2" # Donde se guarda el modelo final fusionado


# Quantization config (8-bit inference with BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None  
)

# Load tokenizer and quantized base model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)
# Load reference model (used for KL regularization in DPO)
ref_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)


# LoRA configuration: inject adapters into key attention/projection layers
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)



###########################2-Load dataset###########################

# Load JSON dataset and split into train/eval
split_dataset = load_dataset("json", data_files="./3answers_to_datasets/diplomat_preference_dataset.json", split="train").train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# Format dataset to use chat-style prompts with system + user input
def dataset_format(example):
    message = {"role": "system", "content":"You are simulating a personality-rich human response. Respond thoughtfully to the following question. Your answer should reflect a realistic and personal perspective."}
    system = tokenizer.apply_chat_template([message], tokenize=False)
    # Formatear el prompt
    message = {"role": "user", "content": example['prompt']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    # Elegidos y rechazados
    chosen = example['chosen'] 
    rejected = example['rejected'] 
    return {
        "prompt":  system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


# Set padding behavior for left-aligned training
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Apply formatting in parallel
train_dataset = train_dataset.map(dataset_format, remove_columns=train_dataset.column_names, num_proc=os.cpu_count())
eval_dataset = eval_dataset.map(dataset_format, remove_columns=eval_dataset.column_names, num_proc=os.cpu_count())



    
###########################3-Training Part###########################


# Login to Weights & Biases for training logging
wandb.login(key="") #Use your wandb face token


# DPO training configuration and trainer    
training_args = DPOConfig(
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    max_steps=50,
    logging_steps=1,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=7,
    bf16=True,
    report_to="wandb",
    beta=0.01,
    max_prompt_length=1024,
    max_length=1024,
    force_use_ref_model=True,
    save_strategy="steps",
    save_steps=5

)


dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer, 
    peft_config=peft_config,
)


# Fine-tuning model with DPO
dpo_trainer.train()

print("Training completed successfully.")

# Save adapter (LoRA) weights
dpo_trainer.model.save_pretrained(adapter_ckpt)
tokenizer.save_pretrained(adapter_ckpt)

# Flush memory
del dpo_trainer, model, ref_model
gc.collect()
torch.cuda.empty_cache()



########################### 4. Merge and Save Final Model ###########################

# Reload model in FP16 (instead of NF4)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    return_dict=True,
    torch_dtype=torch.float16,
)


# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, adapter_ckpt)
model = model.merge_and_unload()

# Save model and tokenizer
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Create pipeline
#pipeline = transformers.pipeline(
#    "text-generation",
#    model=new_model,
#    tokenizer=tokenizer
#)


     

# # Format prompt
# message = [
#     {"role": "system", "content": "You are a helpful assistant chatbot that provides concise answers."},
#     {"role": "user", "content": "What are GPUs and why would I use them for machine learning tasks?"}
# ]
# prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

# # Generate text
# sequences = pipeline(
#     prompt,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
#     num_return_sequences=1,
#     max_length=200,
# )
# print(sequences[0]['generated_text'])
print("Final model successfully saved.")