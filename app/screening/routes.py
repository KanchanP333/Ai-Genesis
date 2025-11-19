from __future__ import annotations
from fastapi import APIRouter, Request, Form, HTTPException, UploadFile, File, Header, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from sqlmodel import select
from typing import Optional
import json
import re
import google.generativeai as genai

from .db import get_session
from .models import Job, Candidate
from app import auth


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
    role = payload.get("role")
    if role not in {"manager", "admin"}:
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"payload": payload, "token": token}

router = APIRouter()


def page(title: str, body: str) -> str:
    head = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>""" + str(title) + """</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    *{box-sizing:border-box}
    body{font-family:system-ui, sans-serif; margin:0; background:var(--bg); color:var(--fg);}
    a{color:var(--accent)}
    .container{max-width:1000px; margin:0 auto; padding:24px}
    .header{position:sticky; top:0; z-index:5; background:linear-gradient(180deg, rgba(0,0,0,.9), rgba(0,0,0,.6)); border-bottom:1px solid var(--border)}
    .brand{display:flex; align-items:center; gap:12px}
    .logo{width:40px; height:40px; border:2px solid var(--accent); border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:700}
    nav.actions a{color:#9ca3af; text-decoration:none}
    input,textarea,button,select{padding:10px 12px; margin:8px 0; width:100%; background:#0f0f10; color:var(--fg); border:1px solid var(--border); border-radius:10px}
    button{background:var(--accent); border:none; cursor:pointer}
    button:hover{background:var(--accent2)}
    .row{display:flex; gap:12px; flex-wrap:wrap}
    .col{flex:1; min-width:220px}
    .card{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; margin:12px 0;}
    .muted{color:var(--muted); font-size:14px}
    pre{background:#0f0f10; padding:12px; border:1px solid var(--border); border-radius:10px; white-space:pre-wrap}
    a.button{display:inline-block; padding:8px 12px; background:#111827; color:white; border-radius:6px; text-decoration:none;}
  </style>
</head>
<body>
  <div class="header">
    <div class="container" style="display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 24px;">
      <div class="brand"><div class="logo">JR</div><div class="title">J.A.R.V.I.S</div></div>
      <nav class="actions" style="display:flex; gap:10px">
        <a id="nav-home" href="/">Home</a>
        <a id="nav-sa" href="/smart-access/admin">Smart Access</a>
        <a id="nav-perf" href="/perf/ui">Performance Meter</a>
        <a id="nav-autoteam" href="/autoteam">Auto Team</a>
        <a id="nav-adv" href="/interviewer-advanced">Advanced Interviewer</a>
        <a id="nav-screening" href="/screening/jobs">Screening Admin</a>
      </nav>
    </div>
  </div>
  <div class="container">
"""
    tail = """
  </div>
  <script>
  (function(){
    try{
      const t = localStorage.getItem('token')||'';
      const set=(id,href)=>{ const el=document.getElementById(id); if(el) el.href=href; };
      set('nav-sa','/smart-access/admin' + (t? ('?token='+encodeURIComponent(t)) : ''));
      set('nav-screening','/screening/jobs' + (t? ('?token='+encodeURIComponent(t)) : ''));
    }catch(e){}
  })();
  </script>
</body>
</html>
"""
    return head + str(body) + tail


@router.get("/jobs", response_class=HTMLResponse)
async def list_jobs(admin: dict = Depends(admin_required)):
    with get_session() as s:
        jobs = s.exec(select(Job).order_by(Job.created_at.desc())).all()
    items = []
    for j in jobs:
        items.append(
            """
            <div class='card'>
              <h3>{title}</h3>
              <div class='muted'>Job Public ID: {pid}</div>
              <p>{desc}...</p>
              <div class='row'>
                <a class='button' href='/screening/apply/{pid}' target='_blank'>Public apply link</a>
                <a class='button' href='/screening/jobs/{pid}/candidates?token={token}'>View candidates</a>
              </div>
            </div>
            """.format(title=j.title, pid=j.public_id, desc=j.description[:280], token=admin["token"]) 
        )
    body = f"""
    <h1>Jobs</h1>
    <div><a class='button' href='/screening/jobs/new?token={admin["token"]}'>Create Job</a></div>
    <div>{{items}}</div>
    """.replace("{items}", "\n".join(items) if items else "<p class='muted'>No jobs yet.</p>")
    return page("Jobs", body)


@router.get("/jobs/new", response_class=HTMLResponse)
async def new_job_form(admin: dict = Depends(admin_required)):
    body = """
    <h1>New Job</h1>
    <form method='post' action='/screening/jobs'>
      <label>Title</label>
      <input name='title' required placeholder='Senior Backend Engineer'/>
      <label>Description</label>
      <textarea name='description' rows='8' required placeholder='Role overview, responsibilities, qualifications...'></textarea>
      <label>Constraints (optional)</label>
      <textarea name='constraints' rows='4' placeholder='e.g., Must be in Dubai, Python + FastAPI, 5+ years'></textarea>
      <button type='submit'>Create</button>
    </form>
    """
    return page("New Job", body)


@router.post("/jobs", response_class=HTMLResponse)
async def create_job(title: str = Form(...), description: str = Form(...), constraints: Optional[str] = Form(None), admin: dict = Depends(admin_required)):
    job = Job(title=title.strip(), description=description.strip(), constraints=(constraints or "").strip() or None)
    with get_session() as s:
        s.add(job)
        s.commit()
        s.refresh(job)
    body = f"""
    <h1>Job Created</h1>
    <div class='card'>
      <h3>{job.title}</h3>
      <p>{job.description[:400]}...</p>
      <p class='muted'>Public ID: {job.public_id}</p>
      <p><a class='button' href='/screening/apply/{job.public_id}' target='_blank'>Public apply link</a></p>
      <p><a href='/screening/jobs?token={admin_token}'>Back to Jobs</a></p>
    </div>
    """
    return page("Job Created", body.format(admin_token=admin["token"]))


@router.get("/apply/{public_id}", response_class=HTMLResponse)
async def apply_form(public_id: str):
    with get_session() as s:
        job = s.exec(select(Job).where(Job.public_id == public_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    themed = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Apply • {job.title}</title>
  <style>
    :root{{--bg:#000;--card:#0b0b0c;--fg:#fff;--muted:#9ca3af;--border:#1f2937;--accent:#b91c1c;--accent-2:#7A0000}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial,Apple Color Emoji,Segoe UI Emoji}}
    .container{{max-width:920px;margin:40px auto;padding:0 16px}}
    .header{{display:flex;align-items:center;justify-content:space-between;padding:12px 0}}
    .brand{{display:flex;gap:10px;align-items:center}}
    .logo{{width:36px;height:36px;border-radius:8px;background:var(--accent)}}
    .title{{font-weight:700;letter-spacing:.08em}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px}}
    label{{display:block;font-size:13px;color:var(--muted);margin:10px 0 6px}}
    input,textarea{{width:100%;background:#0f0f10;color:var(--fg);border:1px solid var(--border);border-radius:10px;padding:12px}}
    input[type=file]{{padding:10px;background:#0f0f10}}
    .hint{{color:var(--muted);font-size:13px}}
    .btn{{display:inline-flex;align-items:center;gap:8px;background:var(--accent);color:#fff;border:none;border-radius:10px;padding:10px 14px;cursor:pointer;text-decoration:none}}
    .btn.gray{{background:#111827}}
    .row{{display:flex;gap:14px;flex-wrap:wrap}}
  </style>
  <script>
    function onFileChange(el){{
      const txt = document.getElementById('resume_text');
      if(el && el.files && el.files.length){{ txt.removeAttribute('required'); }} else {{ txt.setAttribute('required','required'); }}
    }}
  </script>
  </head>
  <body>
    <div class='container'>
      <div class='header'>
        <div class='brand'>
          <div class='logo'></div>
          <div class='title'>J.A.R.V.I.S</div>
        </div>
        <div class='hint'>Secure AI Interview • Public Application</div>
      </div>

      <div class='card' style='margin-top:8px'>
        <h2 style='margin:0 0 6px'>{job.title}</h2>
        <div class='hint'>Job ID: {job.public_id}</div>
        <p class='hint' style='margin-top:10px'>Provide your details below. You can upload a file or paste your resume text.</p>
        <form method='post' enctype='multipart/form-data' style='margin-top:12px'>
          <label>Name</label>
          <input name='name' required placeholder='Your full name'/>
          <label>Email</label>
          <input name='email' type='email' required placeholder='you@example.com'/>
          <div class='row'>
            <div style='flex:1;min-width:240px'>
              <label>Resume file (PDF/DOCX/TXT)</label>
              <input type='file' name='resume_file' accept='.pdf,.docx,.txt' onchange='onFileChange(this)'/>
            </div>
            <div style='flex:1;min-width:240px'>
              <label>Resume (text)</label>
              <textarea id='resume_text' name='resume_text' rows='10' required placeholder='Paste your resume here...'></textarea>
            </div>
          </div>
          <label>Additional info (optional)</label>
          <textarea name='extra_inputs' rows='4' placeholder='Anything else we should know...'></textarea>
          <div style='margin-top:14px'>
            <button class='btn' type='submit'>Submit Application</button>
          </div>
        </form>
      </div>
    </div>
  </body>
 </html>
    """
    return HTMLResponse(content=themed)


@router.get("/jobs/{public_id}/candidates", response_class=HTMLResponse)
async def job_candidates(public_id: str, status: Optional[str] = None, q: Optional[str] = None, admin: dict = Depends(admin_required)):
    token = admin["token"]
    with get_session() as s:
        job = s.exec(select(Job).where(Job.public_id == public_id)).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        stmt = select(Candidate).where(Candidate.job_id == job.id)
        if status:
            stmt = stmt.where(Candidate.status == status)
        cands = s.exec(stmt.order_by(Candidate.created_at.desc())).all()
    # Simple client-side filter for q (name/email) to keep SQL simple
    if q:
        ql = q.lower()
        cands = [c for c in cands if ql in c.name.lower() or ql in c.email.lower()]

    rows = []
    for c in cands:
        rows.append(
            f"""
            <tr>
              <td>{c.name}</td>
              <td class='muted'>{c.email}</td>
              <td>{(c.score if c.score is not None else 'N/A')}</td>
              <td>{('Yes' if c.fits else ('No' if c.fits is not None else 'N/A'))}</td>
              <td>{c.status}</td>
              <td>
                <a href='/screening/candidates/{c.candidate_public_id}?token={token}'>View</a>
                | <a href='/screening/candidates/{c.candidate_public_id}/status?to=under_review&token={token}'>Under review</a>
                | <a href='/screening/candidates/{c.candidate_public_id}/status?to=accepted&token={token}'>Accept</a>
                | <a href='/screening/candidates/{c.candidate_public_id}/status?to=rejected&token={token}'>Reject</a>
              </td>
            </tr>
            """
        )
    body = f"""
    <h1>Candidates: {job.title}</h1>
    <div class='card'>
      <form method='get'>
        <input type='hidden' name='token' value='{token}'/>
        <div class='row'>
          <div class='col'>
            <label>Status</label>
            <select name='status'>
              <option value=''>All</option>
              <option value='received' {'selected' if (status=='received') else ''}>Received</option>
              <option value='under_review' {'selected' if (status=='under_review') else ''}>Under review</option>
              <option value='accepted' {'selected' if (status=='accepted') else ''}>Accepted</option>
              <option value='rejected' {'selected' if (status=='rejected') else ''}>Rejected</option>
            </select>
          </div>
          <div class='col'>
            <label>Search</label>
            <input name='q' value='{q or ''}' placeholder='name or email'/>
          </div>
        </div>
        <button type='submit'>Filter</button>
      </form>
    </div>
    <table style='width:100%; border-collapse:collapse'>
      <thead>
        <tr>
          <th align='left'>Name</th>
          <th align='left'>Email</th>
          <th align='left'>Score</th>
          <th align='left'>Fits</th>
          <th align='left'>Status</th>
          <th align='left'>Actions</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    <p class='muted'><a href='/screening/jobs?token={token}'>&larr; Back to jobs</a></p>
    """.replace("{rows}", "\n".join(rows) if rows else "<tr><td colspan='6' class='muted'>No candidates</td></tr>")
    return page("Candidates", body)


@router.get("/candidates/{cand_public_id}", response_class=HTMLResponse)
async def view_candidate(cand_public_id: str, request: Request, authorization: str | None = Header(default=None)):
    # Public view is allowed. If a valid admin token is provided (header or query), show full details.
    admin_payload = None
    admin_token = None
    try:
        tok = None
        if authorization and authorization.lower().startswith("bearer "):
            tok = authorization.split(" ", 1)[1]
        else:
            tok = request.query_params.get("token")
        if tok:
            p = auth.decode_token(tok)
            if p and p.get("role") in {"manager", "admin"}:
                admin_payload = p
                admin_token = tok
    except Exception:
        pass

    with get_session() as s:
        cand = s.exec(select(Candidate).where(Candidate.candidate_public_id == cand_public_id)).first()
        job_public_id = None
        if cand:
            job_row = s.exec(select(Job).where(Job.id == cand.job_id)).first()
            job_public_id = job_row.public_id if job_row else None
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    extra_admin = ""
    back = ""
    if admin_payload:
        extra_admin = f"""
        <details open><summary>Resume (text)</summary><pre>{(cand.resume_text or 'N/A')}</pre></details>
        <details open><summary>Additional answers/info</summary><pre>{(cand.extra_inputs or 'N/A')}</pre></details>
        """
        if job_public_id:
            back = f"<p class='muted'><a href='/screening/jobs/{job_public_id}/candidates?token={admin_token}'>&larr; Back to candidates</a></p>"

    body = f"""
    <h1>Application</h1>
    <div class='card'>
      <p><strong>Name:</strong> {cand.name}</p>
      <p><strong>Email:</strong> {cand.email}</p>
      <p><strong>Status:</strong> {cand.status}</p>
      <p><strong>Score:</strong> {cand.score if cand.score is not None else 'N/A'}</p>
      <p><strong>Fits:</strong> {('Yes' if cand.fits else ('No' if cand.fits is not None else 'N/A'))}</p>
      <details><summary>Summary</summary><pre>{(cand.summary or 'N/A')}</pre></details>
      {extra_admin}
    </div>
    {back}
    """
    return page("Application", body)


@router.get("/candidates/{cand_public_id}/status", response_class=HTMLResponse)
async def update_status(cand_public_id: str, to: str, admin: dict = Depends(admin_required)):
    if to not in {"received", "under_review", "accepted", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    with get_session() as s:
        cand = s.exec(select(Candidate).where(Candidate.candidate_public_id == cand_public_id)).first()
        if not cand:
            raise HTTPException(status_code=404, detail="Candidate not found")
        cand.status = to
        s.add(cand)
        s.commit()
    # Bounce back to candidate view for simplicity, preserving token
    return RedirectResponse(url=f"/screening/candidates/{cand_public_id}?token={admin['token']}", status_code=303)


@router.get("/jobs/{public_id}/candidates.csv")
async def export_candidates_csv(public_id: str, admin: dict = Depends(admin_required)):
    import csv
    from io import StringIO
    with get_session() as s:
        job = s.exec(select(Job).where(Job.public_id == public_id)).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        cands = s.exec(select(Candidate).where(Candidate.job_id == job.id).order_by(Candidate.created_at.desc())).all()

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "candidate_public_id", "job_public_id", "name", "email", "status", "score", "fits", "created_at", "summary", "extra_inputs", "resume_text"
    ])
    for c in cands:
        writer.writerow([
            c.id,
            c.candidate_public_id,
            job.public_id,
            c.name,
            c.email,
            c.status,
            c.score if c.score is not None else "",
            ("yes" if c.fits else ("no" if c.fits is not None else "")),
            c.created_at.isoformat(),
            (c.summary or "").replace("\n", " ").strip(),
            (c.extra_inputs or "").replace("\n", " ").strip(),
            (c.resume_text or "").replace("\n", " ").strip(),
        ])
    buf.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=candidates_{public_id}.csv"}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)


def parse_gemini_eval(text: str) -> tuple[Optional[float], Optional[str], Optional[bool]]:
    # Expect JSON with fields: score (0-100), summary (string), fits (boolean)
    try:
        data = json.loads(text)
        # score may be string/number
        score_val = data.get("score")
        try:
            score = float(score_val) if score_val is not None else None
        except Exception:
            score = None
        # accept different keys for summary
        summary = data.get("summary") or data.get("reason") or data.get("rationale")
        # accept different keys for fit/fits
        fits = data.get("fits") if data.get("fits") is not None else data.get("fit")
        if isinstance(fits, str):
            fits = fits.strip().lower() in {"true", "yes", "y", "1"}
        return score, summary, fits
    except Exception:
        pass
    # Fallback: regex extract
    m_score = re.search(r"score\D(\d{1,3})", text, re.I)
    score = float(m_score.group(1)) if m_score else None
    m_fit = re.search(r"(fits|fit)\D(yes|no|true|false)", text, re.I)
    fits = None
    if m_fit:
        fits = m_fit.group(2).lower() in {"yes", "true"}
    # crude summary as whole text
    # try to capture a reason/summary segment if present
    m_reason = re.search(r"(reason|rationale|summary)\s*[:\-]\s*(.+)", text, re.I | re.S)
    if m_reason:
        summary = m_reason.group(2).strip()
    else:
        summary = text.strip()
    return score, summary, fits


def _read_resume_file(f: UploadFile) -> str:
    import os
    from pypdf import PdfReader
    from docx import Document
    name = (f.filename or '').lower()
    raw = f.file.read()
    # Reset pointer if needed
    try:
        f.file.seek(0)
    except Exception:
        pass
    if name.endswith('.pdf'):
        import io
        reader = PdfReader(io.BytesIO(raw))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or '')
            except Exception:
                pass
        return "\n".join(texts).strip()
    if name.endswith('.docx'):
        import io
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    # Fallback .txt or unknown
    try:
        return raw.decode('utf-8', errors='ignore')
    except Exception:
        return ''


@router.post("/apply/{public_id}", response_class=HTMLResponse)
async def apply_submit(
    public_id: str,
    name: str = Form(...),
    email: str = Form(...),
    resume_text: str = Form(''),
    extra_inputs: Optional[str] = Form(None),
    resume_file: UploadFile | None = File(None),
):
    with get_session() as s:
        job = s.exec(select(Job).where(Job.public_id == public_id)).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    # Parse resume
    parsed_text = (resume_text or '').strip()
    file_meta = ''
    if resume_file and (resume_file.filename or '').strip():
        try:
            parsed = _read_resume_file(resume_file)
            if parsed:
                parsed_text = parsed
            file_meta = f"\n[File:{resume_file.filename} | type:{resume_file.content_type}]"
        except Exception:
            pass
    if not parsed_text:
        raise HTTPException(status_code=400, detail="Resume is required (file or text)")

    # Compose evaluation prompt
    constraints = job.constraints or ""
    prompt = (
        "You are an expert technical recruiter. Evaluate the candidate against the job description and constraints.\n"
        "Return ONLY a compact JSON with keys: score (0-100), summary (concise 2-4 sentences), fits (true/false).\n\n"
        f"Job Title: {job.title}\n"
        f"Job Description:\n{job.description}\n\n"
        f"Constraints:\n{constraints}\n\n"
        f"Candidate Name: {name}\n"
        f"Candidate Email: {email}\n"
        f"Candidate Resume:\n{parsed_text}\n\n"
        f"Candidate Additional Info:\n{(extra_inputs or '').strip()}\n\n"
        "JSON:"
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    text = resp.text if hasattr(resp, "text") else str(resp)
    score, summary, fits = parse_gemini_eval(text)

    # Store filename metadata alongside extra inputs to avoid schema changes
    extra_combined = ((extra_inputs or '').strip() + file_meta).strip() or None
    cand = Candidate(
        job_id=job.id,
        name=name.strip(),
        email=email.strip(),
        resume_text=parsed_text,
        extra_inputs=extra_combined,
        score=score,
        summary=summary,
        fits=fits,
    )
    with get_session() as s:
        s.add(cand)
        s.commit()
        s.refresh(cand)

    themed = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Application submitted</title>
  <style>
    :root{{--bg:#000;--card:#0b0b0c;--fg:#fff;--muted:#9ca3af;--border:#1f2937;--accent:#b91c1c}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}}
    .container{{max-width:820px;margin:40px auto;padding:0 16px}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px}}
    .btn{{display:inline-block;background:var(--accent);color:#fff;padding:10px 14px;border-radius:10px;text-decoration:none;margin-right:8px}}
    .hint{{color:var(--muted);font-size:14px}}
    pre{{white-space:pre-wrap;background:#0f0f10;border:1px solid var(--border);border-radius:10px;padding:12px;color:var(--fg)}}
  </style>
  </head>
  <body>
    <div class='container'>
      <h1 style='margin:0 0 10px'>Thank you, {name}!</h1>
      <p class='hint'>We have recorded your application.</p>
      <div class='card' style='margin-top:14px'>
        <h3 style='margin:0 0 8px'>Automated screening</h3>
        <p><strong>Score:</strong> {score if score is not None else 'N/A'}</p>
        <p><strong>Fits:</strong> {('Yes' if fits else ('No' if fits is not None else 'N/A'))}</p>
        <details open><summary>Summary</summary><pre>{(summary or 'N/A')}</pre></details>
      </div>
      <div style='margin-top:16px'>
        <div class='hint'>You can check your application status any time here:</div>
        <div style='margin-top:8px'>
          <a class='btn' href='/screening/candidates/{cand.candidate_public_id}' target='_blank'>View my application</a>
          <a class='btn' style='background:#111827' href='/screening/apply/{public_id}'>Back</a>
        </div>
      </div>
    </div>
  </body>
 </html>
    """
    return HTMLResponse(content=themed)


# === JSON APIs for React-native Interviewer UI ===

@router.get("/api/jobs")
async def api_list_jobs(admin: dict = Depends(admin_required)):
    with get_session() as s:
        jobs = s.exec(select(Job).order_by(Job.created_at.desc())).all()
        # Preload candidate counts
        counts: dict[str, int] = {}
        for j in jobs:
            cands_cnt = s.exec(select(Candidate).where(Candidate.job_id == j.id)).all()
            counts[j.public_id] = len(cands_cnt)
    out = []
    for j in jobs:
        out.append({
            "title": j.title,
            "public_id": j.public_id,
            "description": j.description,
            "constraints": j.constraints,
            "created_at": j.created_at.isoformat(),
            "candidate_count": counts.get(j.public_id, 0),
        })
    return {"jobs": out}


@router.post("/api/jobs")
async def api_create_job(payload: dict, admin: dict = Depends(admin_required)):
    title = (payload.get("title") or "").strip()
    description = (payload.get("description") or "").strip()
    constraints = (payload.get("constraints") or None)
    if not title or not description:
        raise HTTPException(status_code=400, detail="title and description required")
    job = Job(title=title, description=description, constraints=constraints)
    with get_session() as s:
        s.add(job)
        s.commit()
        s.refresh(job)
    return {"ok": True, "public_id": job.public_id}


@router.get("/api/jobs/{public_id}/candidates")
async def api_list_candidates(public_id: str, admin: dict = Depends(admin_required)):
    with get_session() as s:
        job = s.exec(select(Job).where(Job.public_id == public_id)).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        cands = s.exec(select(Candidate).where(Candidate.job_id == job.id).order_by(Candidate.created_at.desc())).all()
    out = []
    for c in cands:
        out.append({
            "candidate_public_id": c.candidate_public_id,
            "name": c.name,
            "email": c.email,
            "status": c.status,
            "score": c.score,
            "fits": c.fits,
            "created_at": c.created_at.isoformat(),
            "summary": c.summary,
        })
    return {"job": {"title": job.title, "public_id": job.public_id}, "candidates": out}


@router.post("/api/candidates/{cand_public_id}/status")
async def api_update_status(cand_public_id: str, payload: dict, admin: dict = Depends(admin_required)):
    to = (payload.get("to") or "").strip()
    if to not in {"received", "under_review", "accepted", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    with get_session() as s:
        cand = s.exec(select(Candidate).where(Candidate.candidate_public_id == cand_public_id)).first()
        if not cand:
            raise HTTPException(status_code=404, detail="Candidate not found")
        cand.status = to
        s.add(cand)
        s.commit()
    return {"ok": True}
