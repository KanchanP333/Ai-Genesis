import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

export default function Interviewer() {
  const navigate = useNavigate();
  const API = 'http://localhost:8000';
  const [token, setToken] = useState('');
  const [me, setMe] = useState(null);
  const [loading, setLoading] = useState(true);
  const [jobs, setJobs] = useState([]);
  const [selJob, setSelJob] = useState(null);
  const [cands, setCands] = useState([]);
  const [newJob, setNewJob] = useState({ title: '', description: '', constraints: '' });
  const [expanded, setExpanded] = useState({});
  const [jobId, setJobId] = useState('');

  useEffect(() => {
    const t = localStorage.getItem('token') || '';
    if (!t) { navigate('/login'); return; }
    setToken(t);
    (async () => {
      try {
        const meRes = await fetch(`${API}/me`, { headers: { Authorization: 'Bearer ' + t } });
        if (!meRes.ok) throw new Error('auth');
        const meJson = await meRes.json();
        setMe(meJson);
        if (!['manager','admin'].includes(meJson.role)) { setLoading(false); return; }
        await loadJobs(t);
      } catch (e) {}
      setLoading(false);
    })();
  }, []);

  async function loadJobs(tk = token) {
    const r = await fetch(`${API}/screening/api/jobs`, { headers: { Authorization: 'Bearer ' + tk } });
    if (!r.ok) return;
    const j = await r.json();
    setJobs(j.jobs || []);
    if ((j.jobs || []).length && !selJob) {
      selectJob(j.jobs[0]);
    }
  }

  function toggleExpand(candId){
    setExpanded(prev => ({ ...prev, [candId]: !prev[candId] }));
  }

  async function selectJob(job) {
    setSelJob(job);
    setCands([]);
    if (!job) return;
    const r = await fetch(`${API}/screening/api/jobs/${encodeURIComponent(job.public_id)}/candidates`, { headers: { Authorization: 'Bearer ' + token } });
    if (!r.ok) return;
    const j = await r.json();
    setCands(j.candidates || []);
  }

  async function createJob(e) {
    e.preventDefault();
    const payload = { ...newJob };
    const r = await fetch(`${API}/screening/api/jobs`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + token }, body: JSON.stringify(payload) });
    if (r.ok) {
      setNewJob({ title: '', description: '', constraints: '' });
      await loadJobs();
    }
  }

  async function updateStatus(candId, to) {
    const r = await fetch(`${API}/screening/api/candidates/${encodeURIComponent(candId)}/status`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + token }, body: JSON.stringify({ to }) });
    if (r.ok) {
      await selectJob(selJob);
    }
  }

  if (loading) {
    return <div className="min-h-screen bg-black text-white flex items-center justify-center">Loading...</div>;
  }
  if (!me || !['manager','admin'].includes(me.role)) {
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

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-neutral-800 bg-black/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-red-600" />
            <h1 className="text-lg font-semibold tracking-tight">J.A.R.V.I.S</h1>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <Link className="hover:text-red-400" to="/main">RAG Chat</Link>
            <span className="text-red-500">AI Interviewer</span>
            <a className="hover:text-red-400" href="/demo/upload" target="_blank" rel="noreferrer">Upload Docs</a>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid md:grid-cols-12 gap-6">
          <section className="md:col-span-5 space-y-6">
            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="font-medium">Jobs</h2>
                <div className="flex items-center gap-2">
                  <button onClick={loadJobs} className="px-3 py-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 text-sm">Refresh</button>
                  <Link to="/interviewer/new" className="px-3 py-1.5 rounded-md bg-red-600 hover:bg-red-500 text-sm">New Job</Link>
                </div>
              </div>
              <ul className="divide-y divide-neutral-800">
                {jobs.map(j => (
                  <li key={j.public_id} className={`py-3 cursor-pointer ${selJob && selJob.public_id===j.public_id ? 'bg-black' : ''}`} onClick={() => selectJob(j)}>
                    <div className="font-medium">{j.title}</div>
                    <div className="text-neutral-400 text-xs flex items-center gap-3">
                      <span>{j.public_id}</span>
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-neutral-900 border border-neutral-800">{j.candidate_count ?? 0} candidates</span>
                    </div>
                  </li>
                ))}
                {!jobs.length && <li className="py-6 text-neutral-500 text-sm text-center">No jobs yet</li>}
              </ul>
            </div>

            {/* Creation moved to /interviewer/new */}
          </section>

          <section className="md:col-span-7 space-y-6">
            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="font-medium">Candidates {selJob ? `for ${selJob.title}` : ''}</h2>
                {selJob && <a className="px-3 py-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 text-sm" href={`${API}/screening/jobs/${encodeURIComponent(selJob.public_id)}/candidates?token=${encodeURIComponent(token)}`} target="_blank" rel="noreferrer">Open admin</a>}
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-neutral-400">
                    <th className="py-2">Name</th>
                    <th className="py-2">Email</th>
                    <th className="py-2">Score</th>
                    <th className="py-2">Fits</th>
                    <th className="py-2">Status</th>
                    <th className="py-2">Summary</th>
                    <th className="py-2">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800">
                  {cands.map(c => (
                    <React.Fragment key={c.candidate_public_id}>
                      <tr>
                        <td className="py-2">{c.name}</td>
                        <td className="py-2 text-neutral-400">{c.email}</td>
                        <td className="py-2">{c.score ?? 'N/A'}</td>
                        <td className="py-2">{c.fits==null? 'N/A' : (c.fits ? 'Yes' : 'No')}</td>
                        <td className="py-2">{c.status}</td>
                        <td className="py-2">
                          {c.summary ? (
                            <button className="text-xs px-2 py-1 bg-neutral-800 rounded" onClick={()=>toggleExpand(c.candidate_public_id)}>
                              {expanded[c.candidate_public_id] ? 'Hide' : 'View'}
                            </button>
                          ) : <span className="text-neutral-500">N/A</span>}
                        </td>
                        <td className="py-2 space-x-2">
                          <button className="text-xs px-2 py-1 bg-neutral-800 rounded" onClick={()=>updateStatus(c.candidate_public_id,'under_review')}>Under review</button>
                          <button className="text-xs px-2 py-1 bg-green-700 rounded" onClick={()=>updateStatus(c.candidate_public_id,'accepted')}>Accept</button>
                          <button className="text-xs px-2 py-1 bg-red-700 rounded" onClick={()=>updateStatus(c.candidate_public_id,'rejected')}>Reject</button>
                        </td>
                      </tr>
                      {expanded[c.candidate_public_id] && (
                        <tr>
                          <td colSpan={7} className="py-2">
                            <div className="rounded border border-neutral-800 bg-black p-3 text-neutral-300 whitespace-pre-wrap">{c.summary}</div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                  {!cands.length && <tr><td className="py-10 text-center text-neutral-500" colSpan={7}>Select a job</td></tr>}
                </tbody>
              </table>
            </div>

            <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
              <h3 className="font-medium mb-3">Public Apply</h3>
              <p className="text-neutral-400 text-sm mb-3">Open the public application form by Job Public ID.</p>
              <input value={jobId} onChange={e=>setJobId(e.target.value)} placeholder="e.g. abc123" className="w-full px-3 py-2 rounded-md bg-black border border-neutral-800 focus:outline-none focus:ring-1 focus:ring-red-600" />
              <div className="mt-3 flex gap-2">
                <a className="px-3 py-2 rounded-md bg-neutral-800 hover:bg-neutral-700 text-sm" href={jobId ? `${API}/screening/apply/${encodeURIComponent(jobId)}` : '#'} target="_blank" rel="noreferrer">Open Apply</a>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
