import os
import json
import torch
# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM
#FAISS imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================
# Configurations / Paths
# ==========================
#Models and paths
MODEL_NAME = "microsoft/phi-2"
FAISS_INDEX_PATH = "yoda_faiss_index"
PROCESSED_DOCS_PATH = "processed"

#Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
#Load the tokenizer and model
# ==========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    #Quantization to save VRAM
    load_in_4bit=True,
    #Accelerate handles placement       
    device_map="auto"        
)



# ==========================
#Load FAISS index
# ==========================
print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# ==========================
#Load processed chunks metadata
# ==========================
#JSON mapping chunks to their text
with open(os.path.join(PROCESSED_DOCS_PATH, "chunk_metadata.json"), "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# ==========================
#Helper function: search FAISS
# ==========================
def search_faiss(query, k=3):
    # Embed query
    query_embedding = embedder.encode([query])
    
    # Search in FAISS index
    distances, indices = faiss_index.search(query_embedding, k)

    #print(f"[DEBUG] Retrieved indices: {indices}")  # Debugging

    # Filter out bad indices (like -1 or out of range)
    valid_results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunk_metadata):
            valid_results.append(chunk_metadata[idx])

    return valid_results

# ==========================
#Helper function: Generate text
# ==========================
def generate_response(prompt, max_tokens=100):
    """
    Given a prompt, generates a response from the model.
    """
    #Tokenizes the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    #Generate tokens
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    #Decodes the output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ==========================
#Main interaction loop
# ==========================
if __name__ == "__main__":
    print("Yoda LLM RAG Inference Ready! Type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        
        #Retrieve relevant document chunks from FAISS
        retrieved_chunks = search_faiss(user_query, k=3)
        context = " ".join(retrieved_chunks)
        
        #Construct prompt with context
        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {user_query}\nAnswer (Yoda-style):"
        
        #Generate model response
        response = generate_response(prompt)
        print(f"Yoda: {response}")
