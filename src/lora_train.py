# LoRA fine-tuning skeleton (PEFT) for a causal LLM (Phi-2 in this example).
import os
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "microsoft/phi-2"
DATA_FILE = "../yoda_dataset.json"       
OUTPUT_DIR = "../models/phi2-yoda-lora"  
MAX_LENGTH = 512                         
MAX_STEPS = 800                          
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 1         
GRAD_ACCUM_STEPS = 8
FP16 = True

# LoRA hyperparams
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ----------------------------
#Safety checks / create dirs
# ----------------------------
os.makedirs(Path(OUTPUT_DIR).parent, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
#Load tokenizer
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

#Phi-2 does not have a pad token by default; we set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
#Quantization config + model load
# ----------------------------
#Using BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model in 4-bit (this may take a minute)...")
# Model load with quantization config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for k-bit training
print("Preparing model for k-bit (PEFT) training...")
model = prepare_model_for_kbit_training(model)

# ----------------------------
#Create and attach LoRA adapter
# ----------------------------
print("Attaching LoRA adapter...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ----------------------------
# Load dataset
# ----------------------------
# We'll transform each example into a single training
print(f"Loading dataset from {DATA_FILE} ...")
if not Path(DATA_FILE).exists():
    raise FileNotFoundError(f"{DATA_FILE} not found. Create a small yoda_dataset.json first.")

raw_ds = load_dataset("json", data_files=DATA_FILE)

# Convert examples into single text prompt
def make_training_text(example):
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()

    # The model will learn to map the "prompt" section -> "response"
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{out}\n"
    return text

def preprocess_examples(batch):
    #Batch is a dict with keys "instruction", "input", "output"
    texts = [make_training_text(x) for x in batch["data"]]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Map function to create "text" field
def map_to_text(example):
    return {"text": make_training_text(example)}

# Create a dataset where each item contains the "text"
ds = raw_ds["train"].map(map_to_text)

#Tokenize the dataset
def tokenize_fn(ex):
    out = tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_ds = ds.map(tokenize_fn, batched=False, remove_columns=["text"])

# ----------------------------
#Data collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------
#Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    warmup_steps=50,
    fp16=FP16,
    logging_steps=20,
    save_steps=200,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none"
)

# ----------------------------
#Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
)

# ----------------------------
#Training
# ----------------------------
print("Starting training... (this may run overnight based on max_steps)")
trainer.train()

# ----------------------------
#Save the LoRA adapter and push
# ----------------------------
print(f"Saving LoRA adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)

print("Done. You can now load the adapter with `AutoModelForCausalLM.from_pretrained(base_model).merge_and_unload()` or use PEFT to load the adapter at inference time.")
