import os
from typing import List
import numpy as np
import google.generativeai as genai

MODEL_EMBED = "text-embedding-004"

def embed_texts(texts: List[str]):
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not configured for embeddings")
    # Use official batch API when possible
    if len(texts) == 1:
        result = genai.embed_content(model=MODEL_EMBED, content=texts[0])
    else:
        result = genai.batch_embed_contents(model=MODEL_EMBED, contents=texts)
    # Handle various SDK shapes: dict with 'embeddings', dict with 'embedding', list of vectors, list of dicts/objects
    vectors: List[np.ndarray] = []
    if isinstance(result, dict):
        if "embeddings" in result:
            emb_list = result["embeddings"]
            for item in emb_list:
                if isinstance(item, dict) and "values" in item:
                    vectors.append(np.array(item["values"], dtype=np.float32))
                elif isinstance(item, (list, tuple)):
                    vectors.append(np.array(item, dtype=np.float32))
        elif "embedding" in result:
            item = result["embedding"]
            if isinstance(item, dict) and "values" in item:
                vectors.append(np.array(item["values"], dtype=np.float32))
            elif isinstance(item, (list, tuple)):
                vectors.append(np.array(item, dtype=np.float32))
    elif isinstance(result, list):
        # Could be list of vectors or list of dicts with 'values'
        for item in result:
            if isinstance(item, dict) and "values" in item:
                vectors.append(np.array(item["values"], dtype=np.float32))
            elif isinstance(item, (list, tuple)):
                vectors.append(np.array(item, dtype=np.float32))
    else:
        # As a last resort, try attribute access
        if hasattr(result, "embeddings"):
            for item in getattr(result, "embeddings"):
                vals = getattr(item, "values", None)
                if vals is not None:
                    vectors.append(np.array(vals, dtype=np.float32))
    if not vectors:
        raise RuntimeError("Failed to parse embeddings from Gemini response")
    # L2 normalize
    arrs = []
    for v in vectors:
        n = np.linalg.norm(v)
        arrs.append(v / (n + 1e-10))
    return np.stack(arrs)
