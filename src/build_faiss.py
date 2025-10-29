# This script reads the processed Star Wars/Yoda text chunks
# and builds a FAISS vector store for retrieval-augmented generation (RAG).

# ==========================
# Imports
# ==========================
import os
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# ==========================
# Configuration
# ==========================
# Folder where processed JSON chunks are
PROCESSED_DIR = Path(__file__).parent / "processed"

# Path to save the FAISS index
FAISS_INDEX_PATH = Path(__file__).parent / "yoda_faiss_index"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================
# Load embedding model
# ==========================
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ==========================
# Gather all chunks
# ==========================
print("Reading processed files...")
all_texts = []

# Loop through all JSON files in processed folder
for file_path in PROCESSED_DIR.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        # Checks each JSON file for text chunks
        chunks = json.load(f)
        all_texts.extend(chunks)

print(f"Total chunks found: {len(all_texts)}")

if not all_texts:
    raise ValueError("❌ No text chunks found in processed folder. Make sure your preprocessing step ran correctly!")

# ==========================
# Generate embeddings
# ==========================
print("Generating embeddings...")
embeddings = embed_model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

# ==========================
# Create FAISS index
# ==========================
# Dimension of embeddings
embedding_dim = embeddings.shape[1]

# Using IndexFlatL2
print("Creating FAISS index...")
index = faiss.IndexFlatL2(embedding_dim)

# Add embeddings to index
print("Adding embeddings to FAISS index...")
index.add(embeddings)

# ==========================
# Metadata saving
# ==========================
metadata = []
for i, chunk in enumerate(all_texts):
    metadata.append({
        "id": i,
        "text": chunk
    })

# Save metadata to processed folder
os.makedirs(PROCESSED_DIR, exist_ok=True)
metadata_path = PROCESSED_DIR / "chunk_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"[INFO] Saved metadata for {len(metadata)} chunks to {metadata_path}")

# ==========================
# Save index to disk
# ==========================
print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
faiss.write_index(index, str(FAISS_INDEX_PATH))

print("✅ FAISS index built and saved successfully!")
