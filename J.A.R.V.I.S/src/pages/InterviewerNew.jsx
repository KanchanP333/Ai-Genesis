import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

export default function InterviewerNew(){
  const navigate = useNavigate();
  const API = 'http://localhost:8000';
  const [token, setToken] = useState('');
  const [me, setMe] = useState(null);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({ title: '', description: '', constraints: '' });
  const [createdId, setCreatedId] = useState('');

  useEffect(()=>{
    const t = localStorage.getItem('token')||'';
    if(!t){ navigate('/login'); return; }
    setToken(t);
    (async ()=>{
      try{
        const meRes = await fetch(`${API}/me`, { headers:{Authorization:'Bearer '+t} });
        if(!meRes.ok) throw new Error('auth');
        const meJson = await meRes.json();
        setMe(meJson);
      }catch(e){}
      setLoading(false);
    })();
  },[]);

  async function submit(e){
    e.preventDefault();
    if(!['manager','admin'].includes(me?.role)) return;
    const r = await fetch(`${API}/screening/api/jobs`, { method:'POST', headers:{'Content-Type':'application/json', Authorization:'Bearer '+token}, body: JSON.stringify(form) });
    if(r.ok){
      const j = await r.json();
      setCreatedId(j.public_id || '');
    }
  }

  if(loading){ return <div className="min-h-screen bg-black text-white flex items-center justify-center">Loading...</div>; }
  if(!me || !['manager','admin'].includes(me.role)){
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-xl font-semibold">Not authorized</div>
          <div className="text-neutral-400 mt-2">This page is for managers/admins only.</div>
          <div className="mt-4"><Link className="text-red-400" to="/main">Go to RAG Chat</Link></div>
        </div>
      </div>
    );
  }

  const BASE = 'http://localhost:8000';
  const adminCandidatesUrl = createdId ? `${BASE}/screening/jobs/${encodeURIComponent(createdId)}/candidates?token=${encodeURIComponent(token)}` : '';
  const publicApplyUrl = createdId ? `${BASE}/screening/apply/${encodeURIComponent(createdId)}` : '';

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-neutral-800 bg-black/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-3xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-red-600" />
            <h1 className="text-lg font-semibold tracking-tight">{createdId ? 'Job Created' : 'New Job'}</h1>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <Link className="hover:text-red-400" to="/interviewer">Back to Interviewer</Link>
          </nav>
        </div>
      </header>
      <main className="max-w-3xl mx-auto px-4 py-8">
        {!createdId ? (
          <form onSubmit={submit} className="rounded-xl border border-neutral-800 bg-neutral-950 p-4 space-y-3">
            <label className="text-sm text-neutral-400">Title</label>
            <input className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={form.title} onChange={e=>setForm(v=>({...v,title:e.target.value}))} required />
            <label className="text-sm text-neutral-400">Description</label>
            <textarea className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" rows={8} value={form.description} onChange={e=>setForm(v=>({...v,description:e.target.value}))} required />
            <label className="text-sm text-neutral-400">Constraints (optional)</label>
            <textarea className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" rows={4} value={form.constraints} onChange={e=>setForm(v=>({...v,constraints:e.target.value}))} />
            <div className="flex gap-2">
              <button className="px-4 py-2 rounded-md bg-red-600 hover:bg-red-500" type="submit">Create Job</button>
              <Link to="/interviewer" className="px-4 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700">Cancel</Link>
            </div>
          </form>
        ) : (
          <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-5 space-y-4">
            <div className="text-lg font-semibold">Your job is live</div>
            <div className="text-neutral-300">Share this public apply link with candidates:</div>
            <div className="flex items-center gap-2">
              <input readOnly className="flex-1 px-3 py-2 rounded-md bg-black border border-neutral-800" value={publicApplyUrl} />
              <button className="px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700" onClick={()=>navigator.clipboard.writeText(publicApplyUrl)}>Copy</button>
              <a className="px-3 py-2 rounded-md bg-red-600 hover:bg-red-500" href={publicApplyUrl} target="_blank" rel="noreferrer">Open</a>
            </div>
            <div className="text-neutral-300">Admin candidates page:</div>
            <div className="flex items-center gap-2">
              <input readOnly className="flex-1 px-3 py-2 rounded-md bg-black border border-neutral-800" value={adminCandidatesUrl} />
              <button className="px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700" onClick={()=>navigator.clipboard.writeText(adminCandidatesUrl)}>Copy</button>
              <a className="px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700" href={adminCandidatesUrl} target="_blank" rel="noreferrer">Open</a>
            </div>
            <div className="pt-2 flex gap-2">
              <Link to="/interviewer" className="px-4 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700">Back to Interviewer</Link>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
