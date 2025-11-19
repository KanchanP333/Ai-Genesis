import os
import io
import threading
import subprocess
import sys
from pathlib import Path
from typing import List
import requests
import json
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel

from app import auth
from app.models import LoginRequest, TokenResponse, User, UploadResponse, QueryRequest, QueryResponse
from app.embedding import embed_texts
from app.reranker import rerank
from app.qdrant_client import ensure_collection, upsert_text_chunks, filtered_search
from app.utils import simple_chunk
from app.screening.db import init_db as screening_init_db
from app.screening import routes as screening
from app.smart_access import routes as smart_access
try:
    from app.accounts.db import init_db as accounts_init_db, get_session as accounts_session
    from app.accounts.models import Project, ProjectMembership
    from sqlmodel import select
    _HAS_ACCOUNTS = True
except Exception:
    _HAS_ACCOUNTS = False

import google.generativeai as genai
from dotenv import load_dotenv

# Always load .env from project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash"

app = FastAPI(title="Enterprise RAG with Roles/Levels")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(screening.router, prefix="/screening")
app.include_router(smart_access.router)

_PERF_MOUNTED = False
# Mount Performance Meter (Flask) at /perf via WSGI
# Add its src path to sys.path, then import the Flask app and mount it.
try:
    perf_src = Path(__file__).resolve().parent / "Performance_Meter" / "src"
    if str(perf_src) not in sys.path:
        sys.path.append(str(perf_src))
    from perfmeter.api import APP as PERF_FLASK_APP, _db_init  # type: ignore
    _db_init()
    app.mount("/perf", WSGIMiddleware(PERF_FLASK_APP))
    _PERF_MOUNTED = True
except Exception:
    # If unavailable (e.g., missing Windows deps), skip mounting gracefully
    _PERF_MOUNTED = False

if not _PERF_MOUNTED:
    @app.get("/perf/ui", response_class=HTMLResponse)
    async def perf_ui_fallback():
        return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Performance Meter — J.A.R.V.I.S</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
    .container{max-width:860px;margin:0 auto;padding:24px}
    .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
  </style>
</head>
<body>
  <div class="header">
    <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 24px;">
      <div class="brand"><div class="logo">JR</div><div class="title">Upload Documents</div></div>
      <nav class="actions" style="display:flex; gap:10px">
        <a id="nav-home" href="/" style="color:#9ca3af">Home</a>
        <a id="nav-sa" href="/smart-access/admin" style="color:#9ca3af">Smart Access</a>
        <a id="nav-autoteam" href="/autoteam" style="color:#9ca3af">Auto Team</a>
        <a id="nav-screening" href="/screening/jobs" style="color:#9ca3af">Screening Admin</a>
      </nav>
    </div>
  </div>
  <div class="container">
    <div class="card">
      <h1 style="margin-top:0">Performance Meter</h1>
      <p class="muted">The embedded Flask app could not be mounted on this host (likely due to OS-specific dependencies). You can still use the rest of J.A.R.V.I.S.</p>
      <p>To enable this module, run on Windows or install the required desktop hooks, then restart the backend.</p>
    </div>
  </div>
</body>
</html>
"""

# Try to import Whisper/Gemini summarizer and recorder for Advanced AI Interviewer
_HAS_ADV_SUMMARY = False
_HAS_ADV_RECORD = False
try:
    whisper_path = Path(__file__).resolve().parent / "whisper-large_v3"
    if str(whisper_path) not in sys.path:
        sys.path.append(str(whisper_path))
    from segment_transcribe import summarize_with_gemini, build_ffmpeg_audio_cmd, transcribe_loop  # type: ignore
    _HAS_ADV_SUMMARY = True
    _HAS_ADV_RECORD = True
except Exception:
    _HAS_ADV_SUMMARY = False
    _HAS_ADV_RECORD = False

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

class MeResponse(BaseModel):
    username: str
    role: str
    level: int
    dept: str
    project: str

class ProjectsResponse(BaseModel):
    projects: List[str]

@app.on_event("startup")
async def startup():
    ensure_collection()
    screening_init_db()
    # init accounts DB if available
    try:
        if _HAS_ACCOUNTS:
            accounts_init_db()
    except Exception:
        pass
    # prepare smart access collection
    try:
        smart_access.ensure_collection()
    except Exception:
        pass
    # Try to start J.A.R.V.I.S vite dev server in background
    def _start_jarvis():
        try:
            subprocess.Popen(["npm", "run", "dev"], cwd=os.path.join(os.getcwd(), "J.A.R.V.I.S"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    threading.Thread(target=_start_jarvis, daemon=True).start()

@app.post("/auth/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth.create_access_token(
        sub=user["username"], role=user["role"], level=user["level"], dept=user.get("dept", ""), project=user.get("project", "")
    )
    return TokenResponse(access_token=token)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return User(username=payload["sub"], role=payload["role"], level=payload["level"], dept=payload.get("dept", ""), project=payload.get("project", ""))

@app.get("/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)):
    return MeResponse(username=user.username, role=user.role, level=user.level, dept=user.dept, project=user.project)

@app.get("/projects", response_model=ProjectsResponse)
async def get_projects(user: User = Depends(get_current_user)):
    allowed = _allowed_projects_for_user(user)
    return ProjectsResponse(projects=allowed)

def _allowed_projects_for_user(user: User) -> List[str]:
    """Derive list of projects user may access.
    Prefer accounts DB memberships if available; otherwise fallback to env + user's project.
    """
    if _HAS_ACCOUNTS:
        try:
            with accounts_session() as s:
                # Admin/manager: all projects
                if user.role in {"manager", "admin"}:
                    names = [p.name for p in s.exec(select(Project)).all()]
                    if user.project and user.project not in names:
                        names.append(user.project)
                    return sorted(list(dict.fromkeys(names)))
                # Staff: memberships only
                mems = s.exec(select(ProjectMembership).where(ProjectMembership.username == user.username)).all()
                names = [m.project_name for m in mems]
                if not names and user.project:
                    names = [user.project]
                return sorted(list(dict.fromkeys(names)))
        except Exception:
            pass
    # Fallback to env var
    env_projects = os.getenv("PROJECTS", "")
    available = [p.strip() for p in env_projects.split(",") if p.strip()]
    if user.project and user.project not in available:
        available.append(user.project)
    if user.role in {"manager", "admin"}:
        return sorted(list(dict.fromkeys(available)))
    allowed = [p for p in available if p == user.project]
    if not allowed and user.project:
        allowed = [user.project]
    return allowed

@app.post("/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    audience: str = Form(default="all"),  # one of: all, managers, employees, custom
    allow_roles_custom: str = Form(default=""),  # used when audience=custom, comma-separated roles
    project_override: str | None = Form(default=None),  # optional: override user's project for tagging
    user: User = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported in demo")
    raw = (await file.read()).decode("utf-8", errors="ignore")
    chunks = simple_chunk(raw)
    vecs = embed_texts(chunks)
    audience = (audience or "all").strip().lower()
    if audience not in {"all", "managers", "employees", "custom"}:
        raise HTTPException(status_code=400, detail="Invalid audience")
    roles_final = []
    if audience == "all":
        roles_final = ["staff", "manager", "admin"]
    elif audience == "managers":
        roles_final = ["manager", "admin"]
    elif audience == "employees":
        roles_final = ["staff"]
    else:
        roles_final = [r.strip() for r in allow_roles_custom.split(",") if r.strip()]
        if not roles_final:
            raise HTTPException(status_code=400, detail="Custom audience requires allow_roles_custom")
    # Enforce project permission: only allow override if within allowed projects
    allowed_projects = _allowed_projects_for_user(user)
    desired_project = (project_override or user.project or "").strip() or user.project
    if desired_project and desired_project not in allowed_projects:
        raise HTTPException(status_code=403, detail="Not allowed to upload to this project")
    project_final = desired_project
    payloads = [
        {
            "uploader": user.username,
            "uploader_role": user.role,
            "uploader_level": user.level,
            "filename": file.filename,
            "dept": user.dept,
            "project": project_final,
            "audience": audience,
            "allow_roles": roles_final,
        }
        for _ in chunks
    ]
    upsert_text_chunks(chunks, vecs, payloads)
    return UploadResponse(ids=[file.filename])

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, user: User = Depends(get_current_user)):
    # Embed query
    qvec = embed_texts([req.query])[0]
    # Filtered vector search
    passages = filtered_search(
        qvec,
        user_level=user.level,
        user_role=user.role,
        user_dept=user.dept,
        user_project=user.project,
        top_k=max(10, req.top_k)
    )
    # Rerank locally; reranked items are ((text, payload), score)
    reranked = rerank(req.query, passages, top_k=req.top_k)
    contexts = [pair[0] for pair, score in reranked]

    # Call Gemini for answer
    if not os.getenv("GEMINI_API_KEY"):
        answer = "[Gemini API key not configured]\n\nTop contexts:\n" + "\n---\n".join(contexts)
        sources = [pair[1] for pair, score in reranked]
        return QueryResponse(answer=answer, sources=sources)

    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "You are a corporate assistant. Use the provided context chunks to answer.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Question: {req.query}\n\n"
        "Context:\n" + "\n\n".join(f"- {c}" for c in contexts) + "\n\nAnswer:"
    )
    resp = model.generate_content(prompt)
    answer = resp.text if hasattr(resp, "text") else str(resp)
    sources = [pair[1] for pair, score in reranked]
    return QueryResponse(answer=answer, sources=sources)

from fastapi.responses import RedirectResponse

@app.get("/", response_class=RedirectResponse)
async def root():
    # Redirect to J.A.R.V.I.S dev server home
    return RedirectResponse(url="http://localhost:5173/")


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>J.A.R.V.I.S — Enterprise RAG</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    *{box-sizing:border-box}
    body{font-family:system-ui, sans-serif; margin:0; background:var(--bg); color:var(--fg);}
    a{color:var(--accent)}
    .container{max-width:1100px; margin:0 auto; padding:24px}
    .row{display:flex; gap:12px}
    .col{flex:1}
    .card{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px}
    .muted{color:var(--muted)}
    input,button,textarea,select{padding:10px 12px; margin:8px 0; width:100%; background:#0f0f10; color:var(--fg); border:1px solid var(--border); border-radius:10px}
    button{background:var(--accent); border:none; cursor:pointer; transition:transform .08s ease, background .2s}
    button:hover{background:var(--accent2)}
    button:active{transform:translateY(1px)}
    .btn-secondary{background:#111827}
    .header{position:sticky; top:0; z-index:5; background:linear-gradient(180deg, rgba(0,0,0,.9), rgba(0,0,0,.6)); border-bottom:1px solid var(--border)}
    .brand{display:flex; align-items:center; gap:12px}
    .logo{width:40px; height:40px; border:2px solid var(--accent); border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:700}
    .title{font-weight:900; letter-spacing:6px}
    .grid{display:grid; grid-template-columns: 1fr; gap:16px}
    .section-title{margin:0 0 8px 0}
    /* Chat styles */
    .chat{display:flex; flex-direction:column; gap:10px; height:420px; overflow:auto; padding:8px; border:1px solid var(--border); border-radius:12px; background:#0a0a0a}
    .msg{max-width:80%; padding:10px 12px; border-radius:12px; line-height:1.4; white-space:pre-wrap}
    .msg.user{background:#101010; border:1px solid var(--border); align-self:flex-end}
    .msg.ai{background:#120606; border:1px solid #2a0a0a}
    .msg .tag{display:block; font-size:12px; color:var(--muted); margin-bottom:4px}
    .typing{display:inline-flex; gap:6px; align-items:center}
    .dot{width:6px; height:6px; border-radius:50%; background:var(--muted); opacity:.4; animation:blink 1s infinite}
    .dot:nth-child(2){animation-delay:.15s}
    .dot:nth-child(3){animation-delay:.3s}
    @keyframes blink{0%{opacity:.2} 50%{opacity:1} 100%{opacity:.2}}
    pre{background:#0f0f10; padding:12px; border:1px solid var(--border); border-radius:10px;}
    .pill{display:inline-block; padding:6px 10px; background:#120606; border:1px solid #2a0a0a; color:#fda4a4; border-radius:999px; font-size:12px}
    .actions{display:flex; gap:8px}
    .divider{height:1px; background:var(--border); margin:20px 0}
  </style>
</head>
<body>
  <div class="header">
    <div class="container" style="display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 24px;">
      <div class="brand">
        <div class="logo">JR</div>
        <div class="title">J.A.R.V.I.S</div>
      </div>
      <nav class="actions" style="display:flex; gap:10px">
        <a id="nav-home" href="/" style="color:#9ca3af">Home</a>
        <a id="nav-sa" href="/smart-access/admin" style="color:#9ca3af">Smart Access</a>
        <a id="nav-perf" href="/perf/ui" style="color:#9ca3af">Performance Meter</a>
        <a id="nav-autoteam" href="/autoteam" style="color:#9ca3af">Auto Team</a>
        <a id="nav-adv" href="/interviewer-advanced" style="color:#9ca3af">Advanced Interviewer</a>
        <a id="nav-screening" href="/screening/jobs" style="color:#9ca3af">Screening Admin</a>
      </nav>
    </div>
  </div>
  <div class="container" style="padding-top:24px">
  <div class="grid">
    <section id="query" class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:8px">
        <h2 class="section-title">Chat</h2>
        <div style="display:flex; gap:8px; align-items:center">
          <select id="projectSelect" title="Project" style="width:220px">
            <option value="">Default Project</option>
          </select>
          <a id="uploadBtn" class="pill" href="#" onclick="openUpload(event)">Upload Documents</a>
        </div>
      </div>
      <div id="chat" class="chat">
        <div class="msg ai"><span class="tag">J.A.R.V.I.S</span>Ask me anything about your documents. I follow your role/level/dept/project.</div>
      </div>
      <div class="row">
        <div class="col">
          <textarea id="q" rows="2" placeholder="Type your question..."></textarea>
        </div>
        <div style="width:140px; display:flex; align-items:flex-end">
          <button style="width:100%" onclick="doQuery()">Send</button>
        </div>
      </div>
      <div class="divider"></div>
      <h3 class="section-title">Sources</h3>
      <pre id="sources"></pre>
    </section>
  </div>
  </div>
<script>
// Accept token from URL and persist
try{
  const urlToken = new URLSearchParams(location.search).get('token');
  if(urlToken){ localStorage.setItem('token', urlToken); }
}catch(e){}
let token = localStorage.getItem('token')||'';
let deviceId = localStorage.getItem('device_id')||'';
if(!deviceId){
  deviceId = 'dev-' + Math.random().toString(36).slice(2) + '-' + Date.now();
  localStorage.setItem('device_id', deviceId);
}
let employeeId = '';
// Smart Access collectors
let saMouseMoves = 0;
let saKeyTimes = [];
let saTimer = null;

async function bootstrapUser(){
  if(!token){ return; }
  const me = await fetch('/me', {headers:{'Authorization':'Bearer '+token}});
  const meTxt = await me.text();
  try { const m = JSON.parse(meTxt); employeeId = m.username || ''; await setupProjects(); } catch(e) { /* ignore */ }
  startSmartAccessCollector();
}

function setupNav(){
  try{
    const t = localStorage.getItem('token')||'';
    const set = (id, href) => { const el = document.getElementById(id); if(el){ el.href = href; } };
    set('nav-home','/');
    set('nav-sa','/smart-access/admin' + (t ? ('?token='+encodeURIComponent(t)) : ''));
    set('nav-perf','/perf/ui');
    set('nav-autoteam','/autoteam');
    set('nav-adv','/interviewer-advanced');
    set('nav-screening','/screening/jobs' + (t ? ('?token='+encodeURIComponent(t)) : ''));
  }catch(e){}
}

function openScreening(){
  if(!token){ alert('Please login first'); return; }
  const url = '/screening/jobs?token='+encodeURIComponent(token);
  window.open(url, '_blank');
}

function openSmartAccess(){
  if(!token){ alert('Please login first'); return; }
  const url = '/smart-access/admin?token='+encodeURIComponent(token);
  window.open(url, '_blank');
}

// ---------- Smart Access Collector ----------
function startSmartAccessCollector(){
  // mouse
  window.addEventListener('mousemove', () => { saMouseMoves++; });
  // typing
  window.addEventListener('keydown', () => { saKeyTimes.push(Date.now()); });
  if(saTimer) clearInterval(saTimer);
  saTimer = setInterval(sendSmartAccessEvent, 15000); // every 15s
  document.addEventListener('visibilitychange', () => { if(document.visibilityState==='hidden'){ sendSmartAccessEvent(true); } });
  window.addEventListener('beforeunload', () => { sendSmartAccessEvent(true); });
}

async function sendSmartAccessEvent(sync=false){
  if(!employeeId) return;
  // compute typing CPM and burstiness
  const now = Date.now();
  const horizonMs = 15000; // last 15s
  saKeyTimes = saKeyTimes.filter(t => now - t <= horizonMs);
  const strokes = saKeyTimes.length;
  const typingCpm = Math.round((strokes / (horizonMs/60000)));
  let typingBurst = 0;
  if(saKeyTimes.length > 2){
    const intervals = [];
    for(let i=1;i<saKeyTimes.length;i++){ intervals.push(saKeyTimes[i]-saKeyTimes[i-1]); }
    const avg = intervals.reduce((a,b)=>a+b,0)/intervals.length;
    const variance = intervals.reduce((a,b)=>a+Math.pow(b-avg,2),0)/intervals.length;
    typingBurst = Math.min(1, Math.max(0, Math.sqrt(variance)/1000));
  }
  const payload = {
    employee_id: employeeId,
    page: location.pathname,
    mouse_moves: saMouseMoves,
    typing_cpm: typingCpm,
    typing_burstiness: typingBurst,
    device_id: deviceId,
    seen_device_before: null,
    user_agent: navigator.userAgent,
    timestamp: new Date().toISOString(),
  };
  // reset mouse counter after sampling
  saMouseMoves = 0;
  try{
    await fetch('/smart-access/collect', {
      method:'POST',
      headers:{'Content-Type':'application/json', 'Authorization': token ? ('Bearer '+token) : ''},
      body: JSON.stringify(payload)
    });
  }catch(e){ /* ignore */ }
}

function openUpload(e){
  e.preventDefault();
  const proj = document.getElementById('projectSelect').value||'';
  const url = '/demo/upload?token='+encodeURIComponent(token)+'&project='+encodeURIComponent(proj);
  window.open(url, '_blank');
}

async function doQuery(){
  const chat = document.getElementById('chat');
  const q = document.getElementById('q').value;
  if(!q.trim()) return;
  chat.insertAdjacentHTML('beforeend', `<div class="msg user"><span class="tag">You</span>${q.replace(/</g,'&lt;')}</div>`);
  document.getElementById('q').value='';
  const typing = document.createElement('div');
  typing.className='msg ai';
  typing.innerHTML = '<span class="tag">J.A.R.V.I.S</span><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
  chat.appendChild(typing); chat.scrollTop = chat.scrollHeight;

  const r = await fetch('/query', {method:'POST', headers:{'Authorization':'Bearer '+token, 'Content-Type':'application/json'}, body: JSON.stringify({query: q, top_k: 5})});
  const data = await r.json();
  typing.remove();
  const ans = (data.answer||'').replace(/</g,'&lt;');
  chat.insertAdjacentHTML('beforeend', `<div class="msg ai"><span class="tag">J.A.R.V.I.S</span>${ans}</div>`);
  document.getElementById('sources').innerText = JSON.stringify(data.sources, null, 2);
  chat.scrollTop = chat.scrollHeight;
}
bootstrapUser();
setupNav();
</script>
</body>
</html>
"""
@app.get("/interviewer-advanced", response_class=HTMLResponse)
async def interviewer_advanced_ui():
    warn_msg = 'Enabled' if _HAS_ADV_SUMMARY else 'Note: Gemini summarizer available. Upload/paste transcript to summarize.'
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Advanced AI Interviewer — J.A.R.V.I.S</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
    .container{max-width:900px;margin:0 auto;padding:24px}
    .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
    input,button,textarea{padding:10px 12px;margin:8px 0;width:100%;background:#0f0f10;color:var(--fg);border:1px solid var(--border);border-radius:10px}
    button{background:var(--accent);border:none;cursor:pointer}
    button:hover{background:var(--accent2)}
    .header{position:sticky;top:0;z-index:5;background:linear-gradient(180deg,rgba(0,0,0,.9),rgba(0,0,0,.6));border-bottom:1px solid var(--border)}
    .brand{display:flex;align-items:center;gap:12px}
    .logo{width:40px;height:40px;border:2px solid var(--accent);border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:700}
    pre{background:#0f0f10;padding:12px;border:1px solid var(--border);border-radius:10px;white-space:pre-wrap}
  </style>
</head>
<body>
  <div class="header">
    <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 24px;">
      <div class="brand"><div class="logo">JR</div><div class="title">Advanced AI Interviewer</div></div>
      <nav class="actions" style="display:flex; gap:10px">
        <a id="nav-home" href="/" style="color:#9ca3af">Home</a>
        <a id="nav-sa" href="/smart-access/admin" style="color:#9ca3af">Smart Access</a>
        <a id="nav-perf" href="/perf/ui" style="color:#9ca3af">Performance Meter</a>
        <a id="nav-autoteam" href="/autoteam" style="color:#9ca3af">Auto Team</a>
        <a id="nav-adv" href="/interviewer-advanced" style="color:#9ca3af">Advanced Interviewer</a>
        <a id="nav-screening" href="/screening/jobs" style="color:#9ca3af">Screening Admin</a>
      </nav>
    </div>
  </div>
  <div class="container" style="padding-top:24px">
    <div class="card">
      <h2 style="margin:0 0 8px 0">Transcript</h2>
      <textarea id="txt" rows="8" placeholder="Paste transcript here..."></textarea>
      <div>Or upload .txt transcript: <input id="file" type="file" accept=".txt,text/plain" style="width:auto"/></div>
      <button onclick="summarize()">Summarize</button>
      <div id="warn" style="color:#fbbf24;font-size:12px;margin-top:6px;">""" + warn_msg + """</div>
     </div>
     <div class="card" style="margin-top:16px;">
       <h2 style="margin:0 0 8px 0">Summary</h2>
       <pre id="out"></pre>
     </div>
     <div class="card" style="margin-top:16px;">
      <h2 style="margin:0 0 8px 0">Live Record (Windows dshow/wasapi)</h2>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px">
        <div><label>Mic name<input id="mic" placeholder="@device_cm_...\\wave_... or device name"/></label></div>
        <div><label>Speaker name<input id="spk" placeholder="Stereo Mix ... or @device_cm_..."/></label></div>
        <div><label>Speaker API<select id="api"><option value="wasapi">wasapi</option><option value="dshow">dshow</option></select></label></div>
        <div><label>Model<input id="model" value="small.en"/></label></div>
        <div><label>Segment (s)<input id="seg" value="5"/></label></div>
        <div><label>Keep last<input id="keep" value="10"/></label></div>
      </div>
      <div style="display:flex; gap:8px; margin-top:8px">
        <button onclick="recStart()" {'disabled' if not _HAS_ADV_RECORD else ''}>Start Recording</button>
        <button onclick="recStop()">Stop & Summarize</button>
        <button onclick="recStatus()">Status</button>
      </div>
      <pre id="status" style="margin-top:8px"></pre>
    </div>
   </div>
  <script>
  // token-aware nav
  (function(){
    try{
      const t = localStorage.getItem('token')||'';
      const set=(id,href)=>{ const el=document.getElementById(id); if(el) el.href=href; };
      set('nav-sa','/smart-access/admin' + (t? ('?token='+encodeURIComponent(t)) : ''));
      set('nav-screening','/screening/jobs' + (t? ('?token='+encodeURIComponent(t)) : ''));
    }catch(e){}
  })();
  async function summarize(){
    const out=document.getElementById('out'); out.textContent='';
    const ta=document.getElementById('txt'); const f=document.getElementById('file');
    const fd=new FormData(); if(ta.value.trim()) fd.append('transcript_text', ta.value.trim()); if(f.files && f.files[0]) fd.append('file', f.files[0]);
    const r = await fetch('/interviewer-advanced/summary', { method:'POST', body: fd }); const data = await r.json().catch(()=>({ok:false,error:'Bad JSON'}));
    out.textContent = (data.summary || data.error || JSON.stringify(data));
  }
  async function recStart(){
    const status = document.getElementById('status');
    const body = {
      mic_name: document.getElementById('mic').value||null,
      speaker_name: document.getElementById('spk').value||null,
      speaker_api: document.getElementById('api').value||'wasapi',
      segment_time: parseInt(document.getElementById('seg').value||'5'),
      model: document.getElementById('model').value||'small.en',
      keep_last: parseInt(document.getElementById('keep').value||'10')
    };
    const r = await fetch('/interviewer-advanced/record/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    status.textContent = await r.text();
  }
  async function recStop(){
    const status = document.getElementById('status');
    const r = await fetch('/interviewer-advanced/record/stop', { method:'POST' });
    const data = await r.json().catch(()=>({ok:false}));
    status.textContent = JSON.stringify(data, null, 2);
    if(data.summary){ document.getElementById('out').textContent = data.summary; }
  }
  async function recStatus(){
    const status = document.getElementById('status');
    const r = await fetch('/interviewer-advanced/record/status');
    status.textContent = await r.text();
  }
  </script>
</body>
</html>
"""

@app.post("/interviewer-advanced/summary")
async def interviewer_advanced_summary(
    transcript_text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    # Prefer uploaded file, else use pasted text
    raw = (transcript_text or "").strip()
    if file is not None:
        try:
            raw = (await file.read()).decode("utf-8", errors="ignore")
        except Exception:
            pass
    if not raw:
        return {"ok": False, "error": "Empty transcript"}
    # Write to temp path under whisper-large_v3/output and call summarizer
    base = Path(__file__).resolve().parent / "whisper-large_v3"
    out_dir = base / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "web_transcript.txt"
    summary_path = out_dir / "web_summary.txt"
    transcript_path.write_text(raw, encoding="utf-8")
    summary_text = None
    if _HAS_ADV_SUMMARY:
        try:
            summary_text = summarize_with_gemini(transcript_path, summary_path)  # type: ignore
        except Exception:
            summary_text = None
    if not summary_text:
        # Fallback: use our configured Gemini client directly
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            "You are an interview analyst. Summarize this transcript into sections: Communication style; Reasoning and coherence; Topics and interests; Sentiment and attitude; Risks; Verdict (2-3 sentences).\n\n" + raw
        )
        resp = model.generate_content(prompt)
        summary_text = getattr(resp, "text", "") or ""
    return {"ok": True, "summary": summary_text}

# ---- Advanced recorder endpoints ----
_ADV_STATE = { 'proc': None, 'thread': None, 'stop_event': None, 'running': False }

@app.post("/interviewer-advanced/record/start")
async def adv_record_start(payload: dict):
    if not _HAS_ADV_RECORD:
        return HTMLResponse("Recorder not available on this host", status_code=501)
    if _ADV_STATE['running']:
        return HTMLResponse("Already running", status_code=400)
    mic_name = payload.get('mic_name') or None
    speaker_name = payload.get('speaker_name') or None
    speaker_api = str(payload.get('speaker_api') or 'wasapi')
    segment_time = int(payload.get('segment_time') or 5)
    model = str(payload.get('model') or 'small.en')
    keep_last = int(payload.get('keep_last') or 10)
    base = Path(__file__).resolve().parent / "whisper-large_v3"
    chunks = base / "chunks"
    transcript = base / "output" / "transcript.txt"
    chunks.mkdir(parents=True, exist_ok=True)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    out_pattern = str(chunks / "seg_%06d.wav")
    # build ffmpeg command
    try:
        cmd = build_ffmpeg_audio_cmd(speaker_api=speaker_api, mic_name=mic_name, speaker_name=speaker_name, out_pattern=out_pattern, segment_time=segment_time)  # type: ignore
    except Exception as e:
        return HTMLResponse(f"Failed to build capture cmd: {e}", status_code=400)
    # start ffmpeg and transcriber
    stop_event = threading.Event()
    proc = subprocess.Popen(cmd)
    thread = threading.Thread(target=transcribe_loop, args=(model, chunks, transcript, keep_last, stop_event), daemon=True)
    thread.start()
    _ADV_STATE.update({'proc': proc, 'thread': thread, 'stop_event': stop_event, 'running': True})
    return HTMLResponse("Started", status_code=200)

@app.post("/interviewer-advanced/record/stop")
async def adv_record_stop():
    if not _HAS_ADV_RECORD:
        return { 'ok': False, 'error': 'Recorder not available' }
    if not _ADV_STATE['running']:
        return { 'ok': False, 'error': 'Not running' }
    try:
        _ADV_STATE['stop_event'].set()  # type: ignore
        proc = _ADV_STATE['proc']
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        th = _ADV_STATE['thread']
        if th:
            th.join(timeout=3)
    finally:
        _ADV_STATE.update({'proc': None, 'thread': None, 'stop_event': None, 'running': False})
    # summarize the captured transcript
    base = Path(__file__).resolve().parent / "whisper-large_v3"
    transcript_path = base / "output" / "transcript.txt"
    summary_path = base / "output" / "summary.txt"
    summary_text = None
    if _HAS_ADV_SUMMARY:
        try:
            summary_text = summarize_with_gemini(transcript_path, summary_path)  # type: ignore
        except Exception:
            summary_text = None
    if not summary_text:
        try:
            raw = transcript_path.read_text('utf-8') if transcript_path.exists() else ''
        except Exception:
            raw = ''
        model_g = genai.GenerativeModel(MODEL_NAME)
        resp = model_g.generate_content("Summarize interview:\n\n" + (raw or ''))
        summary_text = getattr(resp, 'text', '') or ''
    return { 'ok': True, 'summary': summary_text }

@app.get("/interviewer-advanced/record/status")
async def adv_record_status():
    running = bool(_ADV_STATE.get('running'))
    return HTMLResponse(f"running={running}")


class AutoTeamRequest(BaseModel):
    prompt: str
    include_employees: bool = False

@app.get("/autoteam", response_class=HTMLResponse)
async def autoteam_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Auto Team Assembler — J.A.R.V.I.S</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
    .container{max-width:900px;margin:0 auto;padding:24px}
    .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
    input,button,textarea{padding:10px 12px;margin:8px 0;width:100%;background:#0f0f10;color:var(--fg);border:1px solid var(--border);border-radius:10px}
    button{background:var(--accent);border:none;cursor:pointer}
    button:hover{background:var(--accent2)}
    .header{position:sticky;top:0;z-index:5;background:linear-gradient(180deg,rgba(0,0,0,.9),rgba(0,0,0,.6));border-bottom:1px solid var(--border)}
    .brand{display:flex;align-items:center;gap:12px}
    .logo{width:40px;height:40px;border:2px solid var(--accent);border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:700}
    pre{background:#0f0f10;padding:12px;border:1px solid var(--border);border-radius:10px;white-space:pre-wrap}
    label{display:flex;align-items:center;gap:8px}
  </style>
</head>
<body>
  <div class="header">
    <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 24px;">
      <div class="brand"><div class="logo">JR</div><div class="title">Auto Team Assembler</div></div>
      <nav class="actions" style="display:flex; gap:10px">
        <a id="nav-home" href="/" style="color:#9ca3af">Home</a>
        <a id="nav-sa" href="/smart-access/admin" style="color:#9ca3af">Smart Access</a>
        <a id="nav-perf" href="/perf/ui" style="color:#9ca3af">Performance Meter</a>
        <a id="nav-autoteam" href="/autoteam" style="color:#9ca3af">Auto Team</a>
        <a id="nav-adv" href="/interviewer-advanced" style="color:#9ca3af">Advanced Interviewer</a>
        <a id="nav-screening" href="/screening/jobs" style="color:#9ca3af">Screening Admin</a>
      </nav>
    </div>
  </div>
  <div class="container" style="padding-top:24px">
    <div class="card">
      <h2 style="margin:0 0 8px 0">Prompt</h2>
      <textarea id="prompt" rows="6" placeholder="Describe the team you need: roles, skills, constraints..."></textarea>
      <label><input id="include" type="checkbox"/> Include employees.json if available</label>
      <button onclick="send()">Send</button>
    </div>
    <div class="card" style="margin-top:16px;">
      <h2 style="margin:0 0 8px 0">Result</h2>
      <pre id="out"></pre>
    </div>
  </div>
  <script>
  async function send(){
    const out = document.getElementById('out'); out.textContent='';
    const r = await fetch('/autoteam/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({prompt: document.getElementById('prompt').value||'', include_employees: document.getElementById('include').checked})});
    const data = await r.json().catch(()=>({ok:false,error:'Bad JSON'}));
    out.textContent = data.text || data.error || JSON.stringify(data);
  }
  </script>
</body>
</html>
"""

@app.post("/autoteam/chat")
async def autoteam_chat(req: AutoTeamRequest):
    prompt = (req.prompt or '').strip()
    if not prompt:
        return {"ok": False, "error": "Empty prompt"}
    ctx = ''
    if req.include_employees:
        try:
            epath = Path(__file__).resolve().parent / 'bitshackathoncode' / 'employees.json'
            if epath.exists():
                ctx = "\n\nEmployees JSON (for context):\n" + epath.read_text(encoding='utf-8')
        except Exception:
            ctx = ''
    model = genai.GenerativeModel(MODEL_NAME)
    full = "You are an expert team assembler. Given the request, propose a team with roles, names/emails (if provided), and justification. Use Markdown tables for team lists.\n\nRequest:\n" + prompt + ctx
    try:
        resp = model.generate_content(full)
        text = getattr(resp, 'text', '') or ''
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "text": text}

@app.get("/demo/upload", response_class=HTMLResponse)
async def demo_upload():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Upload Documents — J.A.R.V.I.S</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    body{font-family:system-ui, sans-serif; margin:0; background:var(--bg); color:var(--fg)}
    .container{max-width:800px; margin:40px auto; padding:24px}
    .card{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px}
    input,button,textarea,select{padding:10px 12px; margin:8px 0; width:100%; background:#0f0f10; color:var(--fg); border:1px solid var(--border); border-radius:10px}
    button{background:var(--accent); border:none; cursor:pointer}
    .muted{color:var(--muted)}
  </style>
  <script>
  let token='';
  function getParam(n){ try{ return new URLSearchParams(location.search).get(n)||'' }catch(e){ return '' } }
  async function init(){
    const t = getParam('token'); if(t){ localStorage.setItem('token', t); }
    token = localStorage.getItem('token')||'';
    await populateProjects();
    const proj = getParam('project'); if(proj){ const sel=document.getElementById('project'); for(const o of sel.options){ if(o.value===proj){ sel.value=proj; break; } } }
    // nav token-aware
    try{
      const set=(id,href)=>{ const el=document.getElementById(id); if(el) el.href=href; };
      set('nav-sa','/smart-access/admin' + (token? ('?token='+encodeURIComponent(token)) : ''));
      set('nav-screening','/screening/jobs' + (token? ('?token='+encodeURIComponent(token)) : ''));
    }catch(e){}
  }
  async function populateProjects(){
    try{
      const r = await fetch('/projects', {headers:{'Authorization':'Bearer '+token}});
      if(!r.ok) return;
      const data = await r.json();
      const sel = document.getElementById('project');
      sel.innerHTML='';
      (data.projects||[]).forEach(p=>{
        const opt = document.createElement('option'); opt.value=p; opt.textContent=p; sel.appendChild(opt);
      });
    }catch(e){}
  }
  async function upload(){
    const f = document.getElementById('file').files[0]; if(!f){ alert('Choose a .txt file'); return; }
    const fd = new FormData();
    fd.append('file', f);
    fd.append('audience', document.getElementById('audience').value);
    fd.append('allow_roles_custom', document.getElementById('roles_custom').value);
    fd.append('project_override', document.getElementById('project').value);
    const r = await fetch('/documents', {method:'POST', headers:{'Authorization':'Bearer '+token}, body: fd});
    const txt = await r.text();
    document.getElementById('result').innerText = txt;
  }
  </script>
</head>
<body onload="init()">
  <div class="container">
    <h1>Upload Documents</h1>
    <div class="card">
      <label>Project</label>
      <select id="project"></select>
      <label>Audience</label>
      <select id="audience">
        <option value="all">All</option>
        <option value="managers">Managers only</option>
        <option value="employees">Employees only</option>
        <option value="custom">Custom roles</option>
      </select>
      <label>Custom roles (comma separated)</label>
      <input id="roles_custom" placeholder="e.g. manager,admin"/>
      <label>File (.txt)</label>
      <input type="file" id="file"/>
      <button onclick="upload()">Upload</button>
      <pre class="muted" id="result"></pre>
    </div>
  </div>
</body>
</html>
"""
