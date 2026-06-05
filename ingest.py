"""
ingest.py — Load Feynman's texts into ChromaDB for RAG retrieval.

HOW TO USE:
1. Put your source files in the ./sources/ directory
   Supported: .txt, .pdf, .md
   
   SUGGESTED SOURCES (all freely available):
   - "Feynman Lectures on Physics" excerpts (feynmanlectures.caltech.edu)
   - "Surely You're Joking, Mr. Feynman!" excerpts
   - "The Pleasure of Finding Things Out" (interviews)
   - "There's Plenty of Room at the Bottom" (paper, free)
   - "QED: The Strange Theory of Light and Matter" excerpts
   - Feynman's 1959 Caltech talk transcripts
   - Nobel Prize lecture (nobelprize.org, free)
   
2. Run: python ingest.py

This script:
- Reads all documents from ./sources/
- Splits them into overlapping chunks
- Embeds each chunk using sentence-transformers
- Stores in ChromaDB (./chroma_db/)
"""

import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ─── Config ──────────────────────────────────────────────────────────────────
SOURCES_DIR = Path("sources")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "feynman_knowledge"
CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, good quality, free

# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        return ""


def load_document(path: Path) -> str:
    """Load a document based on its extension."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return read_txt(path)
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        print(f"Skipping unsupported file type: {path}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to split at sentence/paragraph boundaries.
    """
    # Clean up whitespace
    text = " ".join(text.split())

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunks.append(text[start:])
            break

        # Try to find a good break point (sentence end, paragraph)
        break_chars = [". ", "! ", "? ", "\n", ", "]
        best_break = end
        for bc in break_chars:
            idx = text.rfind(bc, start + chunk_size // 2, end)
            if idx != -1:
                best_break = idx + len(bc)
                break

        chunks.append(text[start:best_break].strip())
        start = best_break - overlap

    return [c for c in chunks if len(c) > 50]  # drop tiny fragments


# ─── Main ingestion ───────────────────────────────────────────────────────────

def ingest():
    if not SOURCES_DIR.exists():
        SOURCES_DIR.mkdir()
        print(f"Created {SOURCES_DIR}/ — add your Feynman source files there, then run again.")
        return

    source_files = list(SOURCES_DIR.glob("**/*"))
    source_files = [f for f in source_files if f.suffix.lower() in (".txt", ".pdf", ".md")]

    if not source_files:
        print(f"No .txt, .pdf, or .md files found in {SOURCES_DIR}/")
        print("Add Feynman source files and run again.")
        return

    print(f"Found {len(source_files)} source file(s). Ingesting...\n")

    # Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection to rebuild fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared existing collection.")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    doc_ids = []
    doc_texts = []
    doc_metas = []

    for file_path in source_files:
        print(f"Processing: {file_path.name}")
        text = load_document(file_path)

        if not text.strip():
            print(f"  → Empty or unreadable, skipping.")
            continue

        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks created")

        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.stem}_{i:04d}"
            doc_ids.append(doc_id)
            doc_texts.append(chunk)
            doc_metas.append({
                "source": file_path.name,
                "chunk_index": i,
                "char_count": len(chunk)
            })

        total_chunks += len(chunks)

    if not doc_texts:
        print("No content to ingest.")
        return

    # Batch insert (ChromaDB handles large batches well)
    BATCH_SIZE = 100
    for i in range(0, len(doc_ids), BATCH_SIZE):
        collection.add(
            ids=doc_ids[i:i+BATCH_SIZE],
            documents=doc_texts[i:i+BATCH_SIZE],
            metadatas=doc_metas[i:i+BATCH_SIZE]
        )
        print(f"  Inserted batch {i//BATCH_SIZE + 1}/{(len(doc_ids)-1)//BATCH_SIZE + 1}")

    print(f"\n✓ Ingestion complete! {total_chunks} total chunks from {len(source_files)} files.")
    print(f"  Stored in: {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()