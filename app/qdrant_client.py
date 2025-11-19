import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import uuid

VECTOR_SIZE = 768  # Gemini text-embedding-004
COLLECTION = "documents"

_qclient: Optional[QdrantClient] = None

def qclient() -> QdrantClient:
    global _qclient
    if _qclient is None:
        url = os.getenv("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL must be set to a Qdrant Cloud endpoint (e.g., https://<cluster>.cloud.qdrant.io)")
        api_key = os.getenv("QDRANT_API_KEY")
        _qclient = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=30.0)
    return _qclient

def ensure_collection():
    client = qclient()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
            on_disk_payload=True,
        )
        # Create payload indexes used by filters
        _ensure_payload_indexes(client)
        return
    # Ensure payload indexes even if collection already exists
    _ensure_payload_indexes(client)
    # Validate vector size; recreate if mismatched
    try:
        info = client.get_collection(COLLECTION)
        cfg = info.config.params
        current_size = None
        vectors = getattr(cfg, 'vectors', None)
        if isinstance(vectors, rest.VectorParams):
            current_size = vectors.size
        elif isinstance(vectors, dict):
            current_size = vectors.get('size')
        if current_size and current_size != VECTOR_SIZE:
            client.delete_collection(COLLECTION)
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
                on_disk_payload=True,
            )
            _ensure_payload_indexes(client)
    except Exception:
        # Fallback: attempt to (re)create
        try:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
                on_disk_payload=True,
            )
            _ensure_payload_indexes(client)
        except Exception:
            pass

def _ensure_payload_indexes(client: QdrantClient):
    # Index fields used by filtered_search
    fields = [
        ("uploader_level", rest.PayloadSchemaType.INTEGER),
        ("dept", rest.PayloadSchemaType.KEYWORD),
        ("project", rest.PayloadSchemaType.KEYWORD),
        ("allow_roles", rest.PayloadSchemaType.KEYWORD),
    ]
    for name, schema in fields:
        try:
            client.create_payload_index(collection_name=COLLECTION, field_name=name, field_schema=schema)
        except Exception:
            # ignore if already exists or not supported
            pass

def upsert_text_chunks(chunks: List[str], embeddings, payloads: List[Dict[str, Any]]):
    client = qclient()
    points = []
    for i, (text, vec, payload) in enumerate(zip(chunks, embeddings, payloads)):
        payload = dict(payload)
        payload.update({"text": text})
        points.append(rest.PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload))
    client.upsert(collection_name=COLLECTION, points=points)


def filtered_search(query_vec, user_level: int, user_role: str, user_dept: str, user_project: str, top_k: int = 20):
    client = qclient()
    must = [
        rest.FieldCondition(
            key="uploader_level",
            range=rest.Range(lte=user_level)
        ),
        rest.FieldCondition(key="dept", match=rest.MatchValue(value=user_dept)),
        rest.FieldCondition(key="project", match=rest.MatchValue(value=user_project)),
        rest.FieldCondition(key="allow_roles", match=rest.MatchValue(value=user_role)),
    ]
    query_filter = rest.Filter(must=must)
    result = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec.tolist(),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    )
    passages = []
    for r in result:
        payload = r.payload or {}
        text = payload.get("text", "")
        passages.append((text, payload))
    return passages
