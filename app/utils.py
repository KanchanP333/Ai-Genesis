from typing import List

def simple_chunk(text: str, max_len: int = 800) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_len:
            chunks.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks
