from typing import List, Tuple
import numpy as np
from app.embedding import embed_texts

def rerank(query: str, passages: List[Tuple[str, dict]], top_k: int = 5):
    if not passages:
        return []
    texts = [p[0] for p in passages]
    # Embed query and passages using Gemini embeddings
    vecs = embed_texts([query] + texts)
    q = vecs[0]
    docs = vecs[1:]
    # Cosine similarity (vectors already L2-normalized in embed_texts)
    sims = (docs @ q).tolist()  # type: ignore
    ranked_idx = sorted(range(len(texts)), key=lambda i: sims[i], reverse=True)
    out = []
    for i in ranked_idx[:top_k]:
        text, payload = passages[i]
        out.append(((text, payload), float(sims[i])))
    return out
