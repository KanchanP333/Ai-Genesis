from __future__ import annotations
from fastapi import APIRouter, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import os
import json
import numpy as np

from app import auth
from app.embedding import embed_texts

# Qdrant local helper (separate collection)
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import uuid

SMART_COLLECTION = "smart_access"
VECTOR_SIZE = 768
_SMART_AVAILABLE = True

_router_client: Optional[QdrantClient] = None

def qclient() -> QdrantClient:
    global _router_client
    if _router_client is None:
        url = os.getenv("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL must be set to a Qdrant Cloud endpoint (e.g., https://<cluster>.cloud.qdrant.io)")
        api_key = os.getenv("QDRANT_API_KEY")
        _router_client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=30.0)
    return _router_client

def ensure_collection() -> bool:
    global _SMART_AVAILABLE
    try:
        c = qclient()
        existing = [col.name for col in c.get_collections().collections]
        if SMART_COLLECTION not in existing:
            c.create_collection(
                collection_name=SMART_COLLECTION,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
                on_disk_payload=True,
            )
            _ensure_payload_indexes(c)
            _SMART_AVAILABLE = True
            return _SMART_AVAILABLE
        # Ensure indexes even if collection exists
        _ensure_payload_indexes(c)
        _SMART_AVAILABLE = True
    except Exception:
        _SMART_AVAILABLE = False
    return _SMART_AVAILABLE

def _ensure_payload_indexes(c: QdrantClient):
    # Index fields used in filters/scroll
    fields = [
        ("employee_id", rest.PayloadSchemaType.KEYWORD),
        ("device_id", rest.PayloadSchemaType.KEYWORD),
        ("ts_epoch", rest.PayloadSchemaType.FLOAT),
        ("flagged", rest.PayloadSchemaType.BOOL),
    ]
    for name, schema in fields:
        try:
            c.create_payload_index(collection_name=SMART_COLLECTION, field_name=name, field_schema=schema)
        except Exception:
            # ignore if already exists
            pass

# Admin dependency (accept header or ?token=)
def admin_required(request: Request, authorization: str | None = Header(default=None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    else:
        token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    if payload.get("role") not in {"manager", "admin"}:
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"payload": payload, "token": token}

router = APIRouter()

# Admin-tunable settings (in-memory; can be wired to DB later)
THRESHOLD = 0.85
BASELINE_DAYS = 30


def to_summary_text(payload: Dict[str, Any]) -> str:
    # Build a compact behavior summary for embedding
    parts = []
    eid = payload.get("employee_id", "")
    parts.append(f"employee_id: {eid}")
    parts.append(f"page: {payload.get('page', '')}")
    parts.append(f"mouse_moves: {payload.get('mouse_moves', 0)}")
    parts.append(f"typing_speed_cpm: {payload.get('typing_cpm', 0)}")
    parts.append(f"typing_burstiness: {payload.get('typing_burstiness', 0)}")
    parts.append(f"ip: {payload.get('ip', '')}")
    parts.append(f"device_id: {payload.get('device_id', '')}")
    parts.append(f"seen_device_before: {payload.get('seen_device_before', False)}")
    parts.append(f"ua: {payload.get('user_agent', '')[:200]}")
    parts.append(f"ts: {payload.get('timestamp', '')}")
    return " | ".join(str(x) for x in parts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_employee_vectors(employee_id: str, days: int = 60) -> List[rest.ScoredPoint]:
    if not ensure_collection():
        return []
    c = qclient()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    flt = rest.Filter(
        must=[
            rest.FieldCondition(key="employee_id", match=rest.MatchValue(value=employee_id)),
            rest.FieldCondition(key="ts_epoch", range=rest.Range(gte=float(cutoff.timestamp()))),
        ]
    )
    # HACK: we don't know a query vector; use empty vector and filter-only search is not supported here.
    # Instead, use scroll API.
    out: List[rest.ScoredPoint] = []
    next_offset = None
    while True:
        page = c.scroll(
            collection_name=SMART_COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=True,
            limit=256,
            offset=next_offset,
        )
        out.extend(page[0])
        next_offset = page[1]
        if not next_offset:
            break
    return out


def upsert_event(vector: np.ndarray, payload: Dict[str, Any]):
    if not ensure_collection():
        return
    c = qclient()
    # Avoid conflicting with Qdrant point ID field
    payload = dict(payload)
    if 'id' in payload:
        payload.pop('id', None)
    pt = rest.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector.tolist(),
        payload=payload,
    )
    c.upsert(collection_name=SMART_COLLECTION, points=[pt])


def seen_device_before(employee_id: str, device_id: str) -> bool:
    if not device_id:
        return False
    if not ensure_collection():
        return False
    try:
        c = qclient()
        flt = rest.Filter(must=[
            rest.FieldCondition(key="employee_id", match=rest.MatchValue(value=employee_id)),
            rest.FieldCondition(key="device_id", match=rest.MatchValue(value=device_id)),
        ])
        page = c.scroll(collection_name=SMART_COLLECTION, scroll_filter=flt, with_payload=False, with_vectors=False, limit=1)
        return len(page[0]) > 0
    except Exception:
        return False


@router.post("/smart-access/collect")
async def collect_event(event: Dict[str, Any], request: Request):
    """
    Accepts a JSON body with fields like:
    {
      "employee_id": "E123",
      "page": "/dashboard",
      "mouse_moves": 1234,
      "typing_cpm": 240,
      "typing_burstiness": 0.42,
      "ip": "1.2.3.4",
      "device_id": "abc123",
      "seen_device_before": true,
      "user_agent": "...",
      "timestamp": "2025-11-18T08:00:00Z"
    }
    """
    ok = ensure_collection()
    # Normalize basics
    eid = (event.get("employee_id") or "").strip()
    if not eid:
        raise HTTPException(status_code=400, detail="employee_id required")
    ts = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
    event["timestamp"] = ts
    event["ts_iso"] = ts
    # compute epoch seconds for numeric filtering
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        event["ts_epoch"] = dt.timestamp()
    except Exception:
        event["ts_epoch"] = datetime.now(timezone.utc).timestamp()
    # Infer UA/IP if missing
    if not event.get("user_agent"):
        event["user_agent"] = request.headers.get("user-agent", "")
    if not event.get("ip"):
        # starlette's client host
        client = request.client
        event["ip"] = getattr(client, 'host', '') if client else ''
    # Device seen before check
    dev_id = (event.get("device_id") or '').strip()
    if event.get("seen_device_before") is None:
        event["seen_device_before"] = seen_device_before(eid, dev_id)
    # Build summary and embed
    summary = to_summary_text(event)
    vec = embed_texts([summary])[0]

    # Anomaly decision: need baseline from last 30 days
    pts = get_employee_vectors(eid, days=BASELINE_DAYS * 2)
    now = datetime.now(timezone.utc)
    day_ago_30 = now - timedelta(days=BASELINE_DAYS)
    base_vecs: List[np.ndarray] = []
    base_days = set()
    for p in pts:
        try:
            te = p.payload.get("ts_epoch")
            if te and float(te) >= day_ago_30.timestamp():
                v = np.array(p.vector, dtype=float)
                base_vecs.append(v)
                dkey = datetime.fromtimestamp(float(te), tz=timezone.utc).strftime('%Y-%m-%d')
                base_days.add(dkey)
        except Exception:
            pass
    flagged = False
    score = None
    if len(base_days) >= BASELINE_DAYS and base_vecs:
        centroid = np.mean(np.stack(base_vecs, axis=0), axis=0)
        score = cosine(np.array(vec, dtype=float), centroid)
        if score < THRESHOLD:
            flagged = True
    # Store record
    payload = dict(event)
    payload.update({
        "employee_id": eid,
        "summary": summary,
        "flagged": flagged,
        "score": score if score is not None else None,
        "ts_iso": ts,
    })
    if ok:
        upsert_event(vec, payload)
        return {"ok": True, "flagged": flagged, "score": score, "stored": True}
    else:
        return {"ok": True, "flagged": False, "score": None, "stored": False}


@router.get("/smart-access/check")
async def check_access(employee_id: str, request: Request):
    """Return allow/deny based on most recent event score within 24h."""
    ensure_collection()
    pts = get_employee_vectors(employee_id, days=2)
    pts_sorted = sorted(pts, key=lambda p: (p.payload or {}).get("ts_iso", ""), reverse=True)
    for p in pts_sorted:
        pl = p.payload or {}
        score = pl.get("score")
        if score is None:
            continue
        allow = score >= THRESHOLD
        return {"employee_id": employee_id, "allow": allow, "score": score, "threshold": THRESHOLD}
    return {"employee_id": employee_id, "allow": True, "reason": "no recent score"}


@router.get("/smart-access/admin/settings", response_class=HTMLResponse)
async def settings_page(admin: dict = Depends(admin_required)):
    themed = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Smart Access • Settings</title>
  <style>
    :root{{--bg:#000;--card:#0b0b0c;--fg:#fff;--muted:#9ca3af;--border:#1f2937;--accent:#b91c1c}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}}
    .container{{max-width:900px;margin:40px auto;padding:0 16px}}
    .header{{display:flex;align-items:center;justify-content:space-between;padding:8px 0}}
    .brand{{display:flex;gap:10px;align-items:center}}
    .logo{{width:32px;height:32px;border-radius:8px;background:var(--accent)}}
    .title{{font-weight:700;letter-spacing:.08em}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px}}
    label{{display:block;color:var(--muted);font-size:13px;margin:8px 0 6px}}
    input{{width:100%;background:#0f0f10;color:var(--fg);border:1px solid var(--border);border-radius:10px;padding:10px}}
    .btn{{display:inline-block;background:var(--accent);color:#fff;padding:10px 14px;border-radius:10px;text-decoration:none;border:none}}
    a.btn.gray, .btn.gray{{background:#111827}}
  </style>
</head>
  <body>
  <div class='container'>
    <div class='header'>
      <div class='brand'><div class='logo'></div><div class='title'>J.A.R.V.I.S</div></div>
      <nav class='actions' style='display:flex;gap:8px'>
        <a class='btn gray' href='/'>Home</a>
        <a class='btn gray' href='/smart-access/admin?token={admin['token']}'>Smart Access</a>
        <a class='btn gray' href='/autoteam'>Auto Team</a>
        <a class='btn gray' href='/screening/jobs?token={admin['token']}'>Screening Admin</a>
      </nav>
    </div>
    <div class='card'>
      <form method='post'>
        <input type='hidden' name='token' value='{admin['token']}'/>
        <label>Threshold (cosine, 0-1)</label>
        <input name='threshold' value='{THRESHOLD}'/>
        <label>Baseline days</label>
        <input name='baseline_days' value='{BASELINE_DAYS}'/>
        <div style='margin-top:12px'>
          <button class='btn' type='submit'>Save</button>
          <a class='btn gray' href='/smart-access/admin?token={admin['token']}'>Back</a>
        </div>
      </form>
    </div>
  </div>
</body>
</html>
    """
    return HTMLResponse(themed)


@router.post("/smart-access/admin/settings", response_class=HTMLResponse)
async def settings_save(request: Request, admin: dict = Depends(admin_required)):
    global THRESHOLD, BASELINE_DAYS
    form = await request.form()
    try:
        THRESHOLD = max(0.0, min(1.0, float(form.get('threshold', THRESHOLD))))
    except Exception:
        pass
    try:
        BASELINE_DAYS = max(1, int(form.get('baseline_days', BASELINE_DAYS)))
    except Exception:
        pass
    return await settings_page(admin)


@router.post("/smart-access/admin/recompute-centroid")
async def recompute_centroid(employee_id: str, admin: dict = Depends(admin_required)):
    ensure_collection()
    pts = get_employee_vectors(employee_id, days=BASELINE_DAYS)
    vecs = [np.array(p.vector, dtype=float) for p in pts if p.vector is not None]
    if not vecs:
        return {"ok": False, "error": "no vectors"}
    centroid = np.mean(np.stack(vecs, axis=0), axis=0)
    payload = {"employee_id": employee_id, "type": "centroid", "ts_iso": datetime.now(timezone.utc).isoformat()}
    upsert_event(centroid, payload)
    return {"ok": True}


@router.get("/smart-access/admin", response_class=HTMLResponse)
async def admin_dashboard(admin: dict = Depends(admin_required), employee_id: Optional[str] = None):
    ensure_collection()
    c = qclient()
    # Show recent flagged or filter by employee
    must = [rest.FieldCondition(key="flagged", match=rest.MatchValue(value=True))]
    if employee_id:
        must.append(rest.FieldCondition(key="employee_id", match=rest.MatchValue(value=employee_id)))
    flt = rest.Filter(must=must)
    # Use scroll to fetch up to 500
    items = []
    next_offset = None
    total = 0
    while True and len(items) < 500:
        page = c.scroll(
            collection_name=SMART_COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=128,
            offset=next_offset,
        )
        points = page[0]
        total += len(points)
        for p in points:
            pl = p.payload or {}
            items.append({
                "employee_id": pl.get("employee_id"),
                "score": pl.get("score"),
                "ts": pl.get("ts_iso"),
                "page": pl.get("page"),
                "device_id": pl.get("device_id"),
            })
        next_offset = page[1]
        if not next_offset:
            break
    # Themed output
    rows = []
    for it in items:
        rows.append(
            f"<tr>"
            f"<td>{it['employee_id']}</td>"
            f"<td>{it['ts']}</td>"
            f"<td>{(it['score'] if it['score'] is not None else 'N/A')}</td>"
            f"<td>{it['page']}</td>"
            f"<td>{it['device_id']}</td>"
            f"<td><a class='btn gray' href='/smart-access/admin/user/{it['employee_id']}?token={admin['token']}'>View</a></td>"
            f"</tr>"
        )
    themed = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Smart Access • Admin</title>
  <style>
    :root{{--bg:#000;--card:#0b0b0c;--fg:#fff;--muted:#9ca3af;--border:#1f2937;--accent:#b91c1c}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}}
    .container{{max-width:1100px;margin:40px auto;padding:0 16px}}
    .header{{display:flex;align-items:center;justify-content:space-between;padding:8px 0}}
    .brand{{display:flex;gap:10px;align-items:center}}
    .logo{{width:32px;height:32px;border-radius:8px;background:var(--accent)}}
    .title{{font-weight:700;letter-spacing:.08em}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-top:10px}}
    input{{width:100%;background:#0f0f10;color:var(--fg);border:1px solid var(--border);border-radius:10px;padding:10px}}
    .btn{{display:inline-block;background:var(--accent);color:#fff;padding:8px 12px;border-radius:10px;text-decoration:none;border:none;font-size:13px}}
    .btn.gray{{background:#111827;color:#fff}}
    table{{width:100%;border-collapse:collapse}}
    th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--border)}}
    thead th{{color:var(--muted);font-weight:600}}
  </style>
</head>
  <body>
  <div class='container'>
    <div class='header'>
      <div class='brand'><div class='logo'></div><div class='title'>J.A.R.V.I.S</div></div>
      <nav class='actions' style='display:flex;gap:8px'>
        <a class='btn gray' href='/'>Home</a>
        <a class='btn gray' href='/smart-access/admin?token={admin['token']}'>Smart Access</a>
        <a class='btn gray' href='/autoteam'>Auto Team</a>
        <a class='btn gray' href='/screening/jobs?token={admin['token']}'>Screening Admin</a>
      </nav>
    </div>
    <div class='card'>
      <form method='get' style='display:flex;gap:10px;align-items:center'>
        <input type='hidden' name='token' value='{admin['token']}'/>
        <input name='employee_id' placeholder='Filter by employee id' value='{employee_id or ''}'/>
        <button class='btn' type='submit'>Filter</button>
        <a class='btn gray' href='/smart-access/admin/settings?token={admin['token']}'>Settings</a>
      </form>
    </div>
    <div class='card'>
      <table>
        <thead><tr><th>Employee</th><th>Time</th><th>Score</th><th>Page</th><th>Device</th><th>Actions</th></tr></thead>
        <tbody>{''.join(rows) if rows else "<tr><td colspan='6' style='color:#9ca3af'>No flagged events</td></tr>"}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
    """
    return HTMLResponse(themed)


@router.get("/smart-access/admin/user/{employee_id}", response_class=HTMLResponse)
async def admin_user_detail(employee_id: str, admin: dict = Depends(admin_required)):
    ensure_collection()
    pts = get_employee_vectors(employee_id, days=60)
    pts_sorted = sorted(pts, key=lambda p: (p.payload or {}).get("ts_iso", ""))
    rows = []
    for p in pts_sorted:
        pl = p.payload or {}
        rows.append(
            f"<tr>"
            f"<td>{pl.get('ts_iso')}</td>"
            f"<td>{(pl.get('score') if pl.get('score') is not None else 'N/A')}</td>"
            f"<td>{('yes' if pl.get('flagged') else 'no')}</td>"
            f"<td>{pl.get('page')}</td>"
            f"<td><details><summary>summary</summary><pre>{(pl.get('summary') or '')}</pre></details></td>"
            f"</tr>"
        )
    themed = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Smart Access • {employee_id}</title>
  <style>
    :root{{--bg:#000;--card:#0b0b0c;--fg:#fff;--muted:#9ca3af;--border:#1f2937;--accent:#b91c1c}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}}
    .container{{max-width:1100px;margin:40px auto;padding:0 16px}}
    .header{{display:flex;align-items:center;justify-content:space-between;padding:8px 0}}
    .brand{{display:flex;gap:10px;align-items:center}}
    .logo{{width:32px;height:32px;border-radius:8px;background:var(--accent)}}
    .title{{font-weight:700;letter-spacing:.08em}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-top:10px}}
    table{{width:100%;border-collapse:collapse}}
    th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--border)}}
    thead th{{color:var(--muted);font-weight:600}}
    pre{{white-space:pre-wrap;background:#0f0f10;border:1px solid var(--border);border-radius:10px;padding:12px;color:var(--fg)}}
    .btn{{display:inline-block;background:var(--accent);color:#fff;padding:8px 12px;border-radius:10px;text-decoration:none;border:none;font-size:13px}}
    .btn.gray{{background:#111827;color:#fff}}
  </style>
</head>
  <body>
  <div class='container'>
    <div class='header'>
      <div class='brand'><div class='logo'></div><div class='title'>J.A.R.V.I.S</div></div>
      <nav class='actions' style='display:flex;gap:8px'>
        <a class='btn gray' href='/'>Home</a>
        <a class='btn gray' href='/smart-access/admin?token={admin['token']}'>Smart Access</a>
        <a class='btn gray' href='/autoteam'>Auto Team</a>
        <a class='btn gray' href='/screening/jobs?token={admin['token']}'>Screening Admin</a>
      </nav>
    </div>
    <div class='card'>
      <h2 style='margin:0 0 8px'>User Detail: {employee_id}</h2>
      <table>
        <thead><tr><th>Time</th><th>Score</th><th>Flagged</th><th>Page</th><th>Summary</th></tr></thead>
        <tbody>{''.join(rows) if rows else "<tr><td colspan='5' style='color:#9ca3af'>No events</td></tr>"}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
    """
    return HTMLResponse(themed)
