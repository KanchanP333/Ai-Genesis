import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function Login() {
  const [username, setUsername] = useState('carol')
  const [password, setPassword] = useState('carol123')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function submit(e){
    e.preventDefault()
    setError(''); setLoading(true)
    try{
      const form = new URLSearchParams()
      form.append('username', username)
      form.append('password', password)
      form.append('grant_type', 'password')
      const r = await fetch('http://localhost:8000/auth/login', { method:'POST', body: form })
      if(!r.ok){ throw new Error('Invalid credentials') }
      const data = await r.json()
      localStorage.setItem('token', data.access_token)
      navigate('/main')
    }catch(err){ setError(err.message||'Login failed') }
    finally{ setLoading(false) }
  }

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center px-6">
      <div className="w-full max-w-md bg-gradient-to-br from-gray-950 to-black border border-gray-800 rounded-2xl p-8 shadow-2xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 border-2 border-[#7A0000] rounded-lg flex items-center justify-center font-bold">JR</div>
          <div className="tracking-widest font-extrabold">J.A.R.V.I.S</div>
        </div>
        <h1 className="text-2xl font-bold mb-2">Welcome back</h1>
        <p className="text-gray-400 mb-6">Sign in to continue to your workspace</p>
        {error && <div className="mb-4 text-sm text-red-400">{error}</div>}
        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="text-sm text-gray-300">Username</label>
            <input value={username} onChange={e=>setUsername(e.target.value)} className="w-full mt-1 px-3 py-2 bg-[#0f0f10] border border-gray-800 rounded-lg focus:outline-none focus:border-[#7A0000]"/>
          </div>
          <div>
            <label className="text-sm text-gray-300">Password</label>
            <input type="password" value={password} onChange={e=>setPassword(e.target.value)} className="w-full mt-1 px-3 py-2 bg-[#0f0f10] border border-gray-800 rounded-lg focus:outline-none focus:border-[#7A0000]"/>
          </div>
          <button disabled={loading} className="w-full py-2 rounded-lg bg-[#7A0000] hover:bg-[#520000] transition-colors disabled:opacity-60">
            {loading? 'Signing in...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  )
}
