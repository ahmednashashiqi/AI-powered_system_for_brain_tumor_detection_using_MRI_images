# -*- coding: utf-8 -*-
"""
Build a local RAG index using FastEmbed (no transformers / no TF).
Outputs: rag/index.npz + rag/meta.json
"""
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    from fastembed import TextEmbedding
except Exception as e:
    raise RuntimeError("fastembed is required. Install: pip install fastembed") from e

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
INDEX_NPZ  = ROOT / "index.npz"
META_JSON  = ROOT / "meta.json"

# تقسيم شظايا أنسب
CHUNK_SIZE = 350
OVERLAP    = 100

# IMPORTANT: يجب أن يطابق retriever.py
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# لدعم عربي/إنجليزي: MODEL_NAME = "BAAI/bge-m3"

def read_corpus_files() -> List[Dict]:
    items = []
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Corpus folder not found: {CORPUS_DIR}")
    for p in CORPUS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if txt.strip():
                items.append({"path": str(p), "text": txt})
    return items

def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        ch = text[i:i+size].strip()
        if ch:
            chunks.append(ch)
        i += step
    return chunks

def main():
    files = read_corpus_files()
    if not files:
        raise RuntimeError(f"No .txt/.md files found in {CORPUS_DIR}")
    meta, all_chunks = [], []
    for f in files:
        chunks = split_into_chunks(f["text"])
        for j, ch in enumerate(chunks):
            meta.append({"text": ch, "source": f["path"], "chunk_id": j})
            all_chunks.append(ch)

    print(f"[RAG] Files: {len(files)} | Total chunks: {len(all_chunks)}")
    if not all_chunks:
        raise RuntimeError("Corpus files exist but no chunks were produced; check CHUNK_SIZE/OVERLAP.")

    print(f"[RAG] Loading FastEmbed: {MODEL_NAME}")
    embedder = TextEmbedding(model_name=MODEL_NAME)

    print("[RAG] Encoding chunks...")
    vecs = []
    for emb in embedder.embed(all_chunks, batch_size=128):
        vecs.append(emb)
    embeddings = np.vstack(vecs).astype("float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms

    np.savez_compressed(INDEX_NPZ, embeddings=embeddings)
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[RAG] Saved index: {INDEX_NPZ}")
    print(f"[RAG] Saved meta : {META_JSON}")
    print("[RAG] Done.")

if __name__ == "__main__":
    main()
