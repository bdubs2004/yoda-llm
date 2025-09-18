# Imports
from pathlib import Path
import fitz  # pymupdf
import docx
import json
from tqdm import tqdm

# Paths
DATA_DIR = Path("../data")          
OUTPUT_DIR = Path("../processed")   
OUTPUT_DIR.mkdir(exist_ok=True)

# Chunk size
CHUNK_SIZE = 512  # characters per chunk
# Supported file types
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#Main processing function
def main():
    files = list(DATA_DIR.glob("*.*"))
    if not files:
        print(f"No files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} files in {DATA_DIR}")
# Process each file
    for file_path in tqdm(files, desc="Processing files"):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif suffix in [".docx", ".doc"]:
            text = extract_text_from_docx(file_path)
        else:
            print(f"Skipping unsupported file type: {file_path.name}")
            continue

        # Skip empty files
        if not text.strip():
            continue

        chunks = chunk_text(text)
        output_file = OUTPUT_DIR / f"{file_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Processing complete! Chunks saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
