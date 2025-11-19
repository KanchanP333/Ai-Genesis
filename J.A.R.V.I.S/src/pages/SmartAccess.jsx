import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

export default function SmartAccess(){
  const API = 'http://localhost:8000';
  const navigate = useNavigate();
  const [token, setToken] = useState('');
  const [me, setMe] = useState(null);
  const [loading, setLoading] = useState(true);
  const [event, setEvent] = useState({
    employee_id: 'E123',
    page: '/dashboard',
    mouse_moves: 230,
    typing_cpm: 260,
    typing_burstiness: 0.38,
    device_id: 'dev-001',
  });
  const [collectResult, setCollectResult] = useState(null);
  const [checkEmp, setCheckEmp] = useState('E123');
  const [checkResult, setCheckResult] = useState(null);

  useEffect(()=>{
    const t = localStorage.getItem('token')||'';
    setToken(t);
    (async()=>{
      if(!t){ setLoading(false); return; }
      try{
        const r = await fetch(`${API}/me`, { headers:{ Authorization: 'Bearer '+t }});
        if(r.ok){ setMe(await r.json()); }
      }catch(e){}
      setLoading(false);
    })();
  },[]);

  async function sendCollect(){
    setCollectResult(null);
    try{
      const r = await fetch(`${API}/smart-access/collect`, { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(event) });
      const j = await r.json();
      setCollectResult(j);
    }catch(e){ setCollectResult({ ok:false, error: String(e) }); }
  }

  async function doCheck(){
    setCheckResult(null);
    try{
      const r = await fetch(`${API}/smart-access/check?employee_id=${encodeURIComponent(checkEmp)}`);
      const j = await r.json();
      setCheckResult(j);
    }catch(e){ setCheckResult({ ok:false, error: String(e) }); }
  }

  const adminLink = (path) => `${API}${path}?token=${encodeURIComponent(token)}`;

  if(loading){ return <div className="min-h-screen bg-black text-white flex items-center justify-center">Loading...</div>; }

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-neutral-800 bg-black/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-red-600" />
            <h1 className="text-lg font-semibold tracking-tight">Smart Access</h1>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <Link className="hover:text-red-400" to="/main">RAG Chat</Link>
            <Link className="hover:text-red-400" to="/interviewer">AI Interviewer</Link>
            <span className="text-red-500">Smart Access</span>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid md:grid-cols-12 gap-6">
          <section className="md:col-span-7 space-y-6">
            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <h2 className="font-medium mb-3">Behavior Event Tester</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Employee ID</label>
                  <input className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={event.employee_id} onChange={e=>setEvent(v=>({...v,employee_id:e.target.value}))} />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Page</label>
                  <input className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={event.page} onChange={e=>setEvent(v=>({...v,page:e.target.value}))} />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Mouse moves</label>
                  <input type="number" className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={event.mouse_moves} onChange={e=>setEvent(v=>({...v,mouse_moves:Number(e.target.value) || 0}))} />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Typing (CPM)</label>
                  <input type="number" className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={event.typing_cpm} onChange={e=>setEvent(v=>({...v,typing_cpm:Number(e.target.value) || 0}))} />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Burstiness</label>
                  <input type="number" step="0.01" className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={event.typing_burstiness} onChange={e=>setEvent(v=>({...v,typing_burstiness:Number(e.target.value) || 0}))} />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Device ID</label>
                  <input className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800" value={event.device_id} onChange={e=>setEvent(v=>({...v,device_id:e.target.value}))} />
                </div>
              </div>
              <div className="mt-3">
                <button onClick={sendCollect} className="px-3 py-2 rounded-md bg-red-600 hover:bg-red-500 text-sm">Send Event</button>
              </div>
              {collectResult && (
                <div className="mt-3 rounded-md border border-neutral-800 bg-black p-3 text-neutral-300 text-sm">
                  <div>Flagged: {String(collectResult.flagged)}</div>
                  <div>Score: {collectResult.score==null? 'N/A' : collectResult.score.toFixed ? collectResult.score.toFixed(3) : collectResult.score}</div>
                </div>
              )}
            </div>

            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <h2 className="font-medium mb-3">Check Access</h2>
              <div className="flex gap-2">
                <input className="flex-1 px-3 py-2 rounded-md bg-black border border-neutral-800" value={checkEmp} onChange={e=>setCheckEmp(e.target.value)} placeholder="Employee ID" />
                <button onClick={doCheck} className="px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700 text-sm">Check</button>
              </div>
              {checkResult && (
                <div className="mt-3 rounded-md border border-neutral-800 bg-black p-3 text-neutral-300 text-sm">
                  <div>Employee: {checkResult.employee_id}</div>
                  {'allow' in checkResult && <div>Allow: {String(checkResult.allow)}</div>}
                  {'score' in checkResult && <div>Score: {checkResult.score==null? 'N/A' : checkResult.score.toFixed ? checkResult.score.toFixed(3) : checkResult.score}</div>}
                  {'threshold' in checkResult && <div>Threshold: {checkResult.threshold}</div>}
                  {'reason' in checkResult && <div>Reason: {checkResult.reason}</div>}
                </div>
              )}
            </div>
          </section>

          <aside className="md:col-span-5 space-y-6">
            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <h3 className="font-medium mb-3">Admin Console</h3>
              {me && ['manager','admin'].includes(me.role) ? (
                <div className="space-y-2">
                  <a className="block px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700 text-sm" href={adminLink('/smart-access/admin')} target="_blank" rel="noreferrer">Open Dashboard</a>
                  <a className="block px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700 text-sm" href={adminLink('/smart-access/admin/settings')} target="_blank" rel="noreferrer">Settings</a>
                </div>
              ) : (
                <div className="text-neutral-500 text-sm">Login as manager/admin to view admin links.</div>
              )}
            </div>

            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <h3 className="font-medium mb-3">About</h3>
              <p className="text-neutral-400 text-sm">Smart Access models behavioral patterns. Events are embedded and compared to a baseline centroid. Below-threshold similarity is flagged.</p>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
