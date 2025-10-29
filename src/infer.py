import os
import json
import torch
import faiss
import numpy as np

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Sentence embedding model for FAISS
from sentence_transformers import SentenceTransformer

# ==========================
# Configurations / Paths
# ==========================
SRC_DIR = os.path.dirname(__file__)
MODEL_NAME = "microsoft/phi-2"
LORA_PATH = os.path.join(SRC_DIR, "lora_starwars")
FAISS_INDEX_PATH = os.path.join(SRC_DIR, "yoda_faiss_index")
PROCESSED_DOCS_PATH = os.path.join(SRC_DIR, "processed")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding model (for FAISS retrieval)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# Load tokenizer and base model
# ==========================
print("\n🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("🔹 Loading base model (Phi-2)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,   # save VRAM
    device_map="auto"
)

# ==========================
# Load LoRA adapter (fine-tuned weights)
# ==========================
if not os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
    raise FileNotFoundError(f"❌ 'adapter_config.json' not found in {LORA_PATH}. Make sure your LoRA weights are there!")

print("🔹 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(device)
model.eval()
print("✅ LoRA adapter loaded successfully!")

# ==========================
# Load FAISS index + metadata
# ==========================
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"❌ FAISS index not found at {FAISS_INDEX_PATH}. Build it first using build_faiss.py")

print("\n🔹 Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

metadata_file = os.path.join(PROCESSED_DOCS_PATH, "chunk_metadata.json")
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"❌ Chunk metadata not found at {metadata_file}. Run preprocessing first!")

print("🔹 Loading processed chunk metadata...")
with open(metadata_file, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# ==========================
# Helper: FAISS Search
# ==========================
def search_faiss(query, k=3):
    """Retrieve top-k relevant chunks using FAISS"""
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)

    valid_results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunk_metadata):
            valid_results.append(chunk_metadata[idx]["text"])
    return valid_results

# ==========================
# Helper: Generate Response
# ==========================
def generate_response(prompt, max_tokens=100):
    """Generate Yoda-style answer using LoRA-finetuned Phi-2 model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Trim everything before the final "Answer (Yoda-style):"
    if "Answer (Yoda-style):" in response:
        response = response.split("Answer (Yoda-style):")[-1].strip()

    return response

# ==========================
# Exported: get_response()
# ==========================
def get_response(user_query: str) -> str:
    """
    Combines FAISS retrieval and Phi-2 LoRA generation
    into one clean function callable from app.py
    """
    # Retrieve relevant text chunks
    retrieved_chunks = search_faiss(user_query, k=3)
    context = " ".join(retrieved_chunks)

    # Build structured prompt
    prompt = f"""Answer the question based on the context below:

Context: {context}

Question: {user_query}

Answer (Yoda-style):"""

    # Generate and return final model output
    answer = generate_response(prompt)
    return answer

# ==========================
# CLI Mode (for testing)
# ==========================
if __name__ == "__main__":
    print("\n🌌 Yoda LLM (LoRA-Finetuned) Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye, young Padawan.")
            break

        print(f"\nYoda: {get_response(user_query)}\n")
