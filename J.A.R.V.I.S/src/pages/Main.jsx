import React, { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

export default function Main(){
  const [token, setToken] = useState('')
  const navigate = useNavigate()
  const API = 'http://localhost:8000'
  const deviceIdRef = useRef('')
  const mmCount = useRef(0)
  const keyTimes = useRef([])
  const collecting = useRef(false)
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Ask me anything about your documents. I follow your role/level/dept/project.' }
  ])
  const [q, setQ] = useState('')
  const [sending, setSending] = useState(false)
  const chatEndRef = useRef(null)
  const [openSources, setOpenSources] = useState({})

  useEffect(()=>{
    const t = localStorage.getItem('token')||''
    if(!t){
      navigate('/login')
      return
    }
    setToken(t)
    // Initialize device id
    let did = localStorage.getItem('device_id')
    if(!did){ did = 'dev-' + Math.random().toString(36).slice(2,10); localStorage.setItem('device_id', did) }
    deviceIdRef.current = did

    // Simple listeners to track behavior
    function onMM(){ mmCount.current += 1 }
    function onKey(){ keyTimes.current.push(Date.now()) }
    window.addEventListener('mousemove', onMM)
    window.addEventListener('keydown', onKey)

    // Start auto-collection for 30s, sending every 5s
    if(!collecting.current){
      collecting.current = true
      const started = Date.now()
      const iv = setInterval(async ()=>{
        const elapsed = (Date.now() - started) / 1000
        const mm = mmCount.current
        mmCount.current = 0
        const times = keyTimes.current
        keyTimes.current = []
        // compute typing cpm and burstiness
        const keyCount = times.length
        const cpm = Math.round((keyCount / 5) * 60) // approx per 5s to per minute
        let burst = 0
        if(times.length > 2){
          const intervals = []
          for(let i=1;i<times.length;i++){ intervals.push((times[i]-times[i-1])/1000) }
          const mean = intervals.reduce((a,b)=>a+b,0)/intervals.length
          const variance = intervals.reduce((a,b)=>a+(b-mean)*(b-mean),0)/intervals.length
          burst = Math.min(2, Math.max(0, Math.sqrt(variance)))
        }
        const payload = {
          employee_id: localStorage.getItem('username') || 'unknown',
          page: '/main',
          mouse_moves: mm,
          typing_cpm: cpm,
          typing_burstiness: Number(burst.toFixed(2)),
          device_id: deviceIdRef.current,
          timestamp: new Date().toISOString(),
        }
        try{ await fetch(`${API}/smart-access/collect`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) }) }catch(e){}
        if(elapsed >= 30){ clearInterval(iv); collecting.current = false }
      }, 5000)
    }

    return () => {
      window.removeEventListener('mousemove', onMM)
      window.removeEventListener('keydown', onKey)
    }
  },[])

  useEffect(()=>{ chatEndRef.current && chatEndRef.current.scrollIntoView({behavior:'smooth'}) }, [messages])

  async function handleSend(){
    const text = (q||'').trim()
    if(!text || sending) return
    setSending(true)
    setMessages(prev => [...prev, {role:'user', text}])
    setQ('')
    try{
      const r = await fetch(`${API}/query`, { method:'POST', headers:{ 'Authorization': 'Bearer '+token, 'Content-Type':'application/json' }, body: JSON.stringify({ query: text, top_k: 5 }) })
      const data = await r.json()
      const ans = (data && data.answer) ? data.answer : ''
      setMessages(prev => [...prev, {role:'ai', text: ans, sources: Array.isArray(data?.sources) ? data.sources : [] }])
    }catch(e){
      setMessages(prev => [...prev, {role:'ai', text: 'Error getting answer.'}])
    }finally{
      setSending(false)
    }
  }

  function openUpload(){
    const url = `${API}/demo/upload?token=${encodeURIComponent(token)}`
    window.open(url, '_blank')
  }

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <header className="sticky top-0 z-10 border-b border-gray-800 bg-black/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 border-2 border-[#7A0000] rounded-lg flex items-center justify-center font-bold">JR</div>
            <div className="tracking-widest font-extrabold">J.A.R.V.I.S</div>
            <span className="text-gray-500 text-sm ml-3">Enterprise RAG</span>
          </div>
          <nav className="flex items-center gap-3">
            <Link to="/" className="text-gray-300 hover:text-white">Home</Link>
            <Link to="/interviewer" className="text-gray-300 hover:text-white">AI Interviewer</Link>
            <a href="http://localhost:8000/interviewer-advanced" target="_blank" rel="noreferrer" className="text-gray-300 hover:text-white">Advanced AI Interviewer</a>
            <Link to="/smart-access" className="text-gray-300 hover:text-white">Smart Access</Link>
            <a href="http://localhost:8000/perf/ui" target="_blank" rel="noreferrer" className="text-gray-300 hover:text-white">Performance Meter</a>
            <a href="http://localhost:8000/autoteam" target="_blank" rel="noreferrer" className="text-gray-300 hover:text-white">Auto Team</a>
            <Link to="/login" className="text-gray-300 hover:text-white">Switch User</Link>
          </nav>
        </div>
      </header>
      <main className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="w-full px-6 py-6 space-y-3">
            {messages.map((m, i) => (
              <div key={i} className="space-y-2">
                <div className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                  <div className={
                    'max-w-[80%] whitespace-pre-wrap leading-relaxed ' +
                    (m.role === 'user' ? 'bg-[#101010] rounded-2xl px-4 py-2' : 'bg-[#120606] rounded-2xl px-4 py-2')
                  }>
                    {m.text}
                  </div>
                </div>
                {m.role === 'ai' && Array.isArray(m.sources) && m.sources.length > 0 && (
                  <div className="pl-2">
                    <button
                      onClick={()=>setOpenSources(s=>({...s, [i]: !s[i]}))}
                      className="text-xs text-gray-300 hover:text-white"
                    >
                      {openSources[i] ? 'Hide sources' : 'Show sources'}
                    </button>
                    {openSources[i] && (
                      <div className="mt-2 text-sm space-y-2">
                        {m.sources.map((s, si) => {
                          const fname = s?.filename || s?.payload?.filename
                          const proj = s?.project || s?.payload?.project
                          const text = s?.text || s?.payload?.text
                          return (
                            <div key={si} className="bg-[#0b0b0b] rounded-xl p-3">
                              {fname ? (<div className="font-medium">{fname}</div>) : null}
                              {proj ? (<div className="text-gray-400">Project: {proj}</div>) : null}
                              {text ? (
                                <pre className="mt-2 whitespace-pre-wrap text-gray-300 text-xs">{text}</pre>
                              ) : (
                                <pre className="whitespace-pre-wrap text-gray-300 text-xs">{JSON.stringify(s, null, 2)}</pre>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        </div>
        <div className="bg-black/80">
          <div className="w-full px-6 py-4 flex items-end gap-3">
            <button onClick={openUpload} className="px-3 py-2 rounded-lg bg-[#111827] text-white">Upload Documents</button>
            <textarea value={q} onChange={e=>setQ(e.target.value)} rows={2} placeholder="Type your question..." className="flex-1 bg-[#0f0f10] rounded-xl px-3 py-2 text-sm" />
            <button onClick={handleSend} disabled={sending} className="px-4 py-2 rounded-lg bg-[#7A0000] hover:bg-[#520000] disabled:opacity-50">Send</button>
          </div>
        </div>
      </main>
    </div>
  )
}
