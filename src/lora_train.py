# lora_train.py
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import torch

# -------------------------------
#Helper function to list model modules
# -------------------------------
def list_model_module_names(model, max_lines=200):
    """
    Prints the first N module names (module path strings) from the model.
    Use these strings (or substrings) to choose `target_modules` for LoRA.
    """
    print("\n[DEBUG] Listing model modules (first {} lines):".format(max_lines))
    count = 0
    for name, _ in model.named_modules():
        print(name)
        count += 1
        if count >= max_lines:
            break
    print("[DEBUG] end module list\n")


# -------------------------------
#Load JSON dataset
# -------------------------------
with open("yoda_dataset.json", "r") as f:
    data_list = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data_list)

# -------------------------------
#Format prompts
# -------------------------------
def format_prompt(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()

    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    return {"prompt": prompt, "response": output_text}

dataset = dataset.map(format_prompt)

# -------------------------------
#Load tokenizer and model
# -------------------------------
MODEL_NAME = "microsoft/phi-2"
print(f"[INFO] Using base model: {MODEL_NAME}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("[INFO] Loading model with 4-bit quantization (may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

# -------------------------------
#Tokenize dataset
# -------------------------------
def tokenize(example):
    full_prompt = example["prompt"] + " " + example["response"]
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokenized["labels"] = [
        t if t != tokenizer.pad_token_id else -100 for t in tokenized["input_ids"]
    ]
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=False)

# -------------------------------
#Set up LoRA
# -------------------------------
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "Wq", "Wqkv", "q", "kv", "o", "gate_proj",
        "down_proj", "up_proj", "fc1", "fc2", "dense"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print("[INFO] Applying LoRA adapters (this will inspect model module names)...")
model = get_peft_model(model, lora_config)

# -------------------------------
#Data collator
# -------------------------------
data_collator = default_data_collator

# -------------------------------
#Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./lora_starwars",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=6,
    fp16=True,
    optim="paged_adamw_32bit",
)

# -------------------------------
#Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
#Start training
# -------------------------------
trainer.train()

# -------------------------------
#Save LoRA weights
# -------------------------------
model.save_pretrained("./lora_starwars")
