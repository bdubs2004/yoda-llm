#Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------
# Model selection
# -------------------------
MODEL = "microsoft/phi-2"  # 2.7B version

# -------------------------
# 4-bit quantization config
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# -------------------------
# Load tokenizer
# -------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

# -----------------------------
# Load model with quantization
# -----------------------------
print("Loading model (4-bit, device_map=auto)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto"  # Let HF assign layers to GPU/CPU
)

print("Model loaded successfully. Device of first parameter:", next(model.parameters()).device)

# -------------------------
# Quick Test
# -------------------------
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(input_ids, max_new_tokens=20)
print("Sample output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
