# -*- coding: utf-8 -*-
"""
Simple RAG retriever that loads index.npz + meta.json built by build_index.py
Performs cosine similarity search and returns:
- sim  : true cosine in [0,1] (for filtering & display)
- score: sim + keyword bonus (for internal ranking only)
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from fastembed import TextEmbedding
except Exception as e:
    raise RuntimeError("fastembed is required for RAG. pip install fastembed") from e

ROOT = Path(__file__).resolve().parent
INDEX_NPZ = ROOT / "index.npz"
META_JSON = ROOT / "meta.json"

# IMPORTANT: يجب أن يطابق build_index.py
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# لدعم عربي/إنجليزي: MODEL_NAME = "BAAI/bge-m3"

class RAGRetriever:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 64):
        if not INDEX_NPZ.exists() or not META_JSON.exists():
            raise FileNotFoundError(f"RAG index not found: {INDEX_NPZ}, {META_JSON}")
        data = np.load(INDEX_NPZ)
        self.embeddings = data["embeddings"].astype("float32")  # [N, d], L2-normalized
        self.meta: List[Dict[str, Any]] = json.loads(META_JSON.read_text(encoding="utf-8"))
        if len(self.meta) != self.embeddings.shape[0]:
            raise RuntimeError("meta length != embeddings rows; rebuild the index.")
        self.model = TextEmbedding(model_name=model_name)
        self.batch_size = batch_size

    def _embed_query(self, q: str) -> np.ndarray:
        vecs = []
        for v in self.model.embed([q], batch_size=1):
            vecs.append(v)
        qv = np.asarray(vecs[0], dtype="float32")
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        return qn

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        n = int(self.embeddings.shape[0])
        if n == 0:
            return []
        k = max(1, min(int(top_k), n))

        # 1) cosine sim (vectors already normalized)
        qn = self._embed_query(query.strip())
        sims = self.embeddings @ qn  # [-1..1], غالبًا موجبة هنا
        if np.isnan(sims).any():
            sims = np.where(np.isnan(sims), -np.inf, sims)

        part = np.argpartition(sims, -k)[-k:]
        top_order = part[np.argsort(sims[part])][::-1]

        results: List[Dict[str, Any]] = []
        for i in top_order:
            m = self.meta[int(i)]
            sim = float(sims[int(i)])
            sim_disp = max(0.0, min(sim, 1.0))  # للعرض/الفلترة فقط
            results.append({
                "text": m.get("text", ""),
                "source": m.get("source", ""),
                "chunk_id": int(m.get("chunk_id", i)),
                "sim": sim_disp,         # نعرض ونفلتر بهذا
                "score": sim_disp        # سنضيف البونص لاحقًا للترتيب
            })

        # 2) Hybrid keyword bonus (للترتيب فقط)
        q_terms = [t.lower() for t in query.split() if len(t) >= 3]
        def kw_bonus(txt: str) -> float:
            t = txt.lower()
            cnt = sum(t.count(w) for w in q_terms)
            return 0.02 * cnt

        for r in results:
            r["score"] = r["score"] + kw_bonus(r["text"])

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
