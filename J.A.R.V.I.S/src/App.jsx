import * as React from "react";
import { useState, useEffect } from "react";
import { Routes, Route, Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Menu, X, ArrowRight, Zap, Shield, Brain } from "lucide-react";

// Minimal classnames utility
function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}

// MENU ITEMS (hide Performance Meter and Advanced Interviewer)
const menuItems = [
  { name: "Main", href: "/main" },
  { name: "Auto Team Assembler", href: "#" },
  { name: "Smart Access", href: "/smart-access" },
];

// CLIENT LOGOS
const clients = [
  { name: "Opus", src: "src/assets/qdrant.png" },
  { name: "AI Studio", src: "src/assets/aistudio.png" },
  { name: "Gemini", src: "src/assets/gemini.png" },
  { name: "Opus2", src: "src/assets/opus.png" },
];

// FEATURES
const features = [
  {
    icon: Brain,
    title: "AI-Powered Intelligence",
    description: "Advanced algorithms that learn and adapt to your testing needs"
  },
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Execute thousands of tests in seconds with optimized performance"
  },
  {
    icon: Shield,
    title: "Enterprise Security",
    description: "Bank-level encryption and compliance with industry standards"
  }
];

// Custom Font
const CustomFont = () => (
  <style>
    {`
      @font-face {
        font-family: 'JarvisFont';
        src: url('/fonts/60798222a30d083d-s.p.woff2') format('woff2');
        font-weight: normal;
        font-style: normal;
        font-display: swap;
      }
      body {
        font-family: 'JarvisFont', sans-serif;
      }

      /* Logo styles */
      .logo-container {
        display: flex;
        align-items: center;
        gap: 14px;
      }
      .logo-card {
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .logo-placeholder {
        width: 48px;
        height: 48px;
        background: #111;
        border: 2px solid #7A0000;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        border-radius: 10px;
      }
      .logo-title {
        font-size: 10px;
        margin-top: 4px;
        color: #7A0000;
      }
    `}
  </style>
);

export const HeroSection = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="relative bg-black text-white overflow-x-hidden">
      <CustomFont />

      {/* Background Grid */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-br from-black via-gray-900 to-black opacity-50 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(122,0,0,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(122,0,0,0.1)_1px,transparent_1px)] bg-[size:50px_50px]"></div>
      </div>

      {/* NAVBAR */}
      <header
        className={cn(
          "fixed w-full z-30 transition-all duration-300 backdrop-blur-xl",
          scrolled
            ? "bg-black/90 shadow-[0_4px_20px_rgba(0,0,0,0.6)] translate-y-0 border-b border-[#7A0000]/20"
            : "bg-transparent"
        )}
      >
        <div className="mx-auto max-w-7xl px-6 flex justify-between items-center py-4">

          {/* Logo */}
          <div className="flex items-center gap-4">

            <div className="logo-container">
              <div className="logo-card">
                <img src="/src/assets/onlylogo.png" width="48" height="48" alt="Logo" />
              </div>
            </div>

            {/* J.A.R.V.I.S Text */}
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="text-xl md:text-2xl font-bold tracking-widest hover:text-[#7A0000] transition-colors duration-300"
            >
              J.<span className="tracking-wide">A.</span>R.<span className="tracking-wide">V.</span>I.<span className="tracking-wide">S</span>
            </button>
          </div>

          {/* Desktop Menu */}
          <ul className="hidden lg:flex gap-8 items-center">
            {menuItems.map((item, i) => (
              <li key={i} className="relative group">
                <a
                  href={item.href}
                  target={item.target || undefined}
                  rel={item.target === "_blank" ? "noopener noreferrer" : undefined}
                  className="text-sm text-gray-300 hover:text-white transition-colors duration-300"
                >
                  {item.name}
                </a>

                {/* Sliding Underline */}
                <span className="absolute left-0 -bottom-1 w-0 h-[2px] bg-[#7A0000] transition-all duration-300 group-hover:w-full"></span>
              </li>
            ))}
            <li>
              <Button size="sm" className="bg-[#7A0000] hover:bg-[#520000] text-white">
                Demo
              </Button>
            </li>
          </ul>

          {/* Mobile Hamburger */}
          <div className="lg:hidden">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="p-2 rounded-md hover:bg-gray-800 transition-colors"
            >
              {menuOpen ? <X className="text-white w-6 h-6" /> : <Menu className="text-white w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div
          className={cn(
            "lg:hidden absolute left-0 w-full bg-black/95 backdrop-blur-xl border-b border-[#7A0000]/20 overflow-hidden transition-all duration-500 ease-in-out",
            menuOpen ? "max-h-96 opacity-100 py-6" : "max-h-0 opacity-0"
          )}
        >
          <ul className="flex flex-col gap-4 items-center">
            {menuItems.map((item, i) => (
              <li key={i}>
                <a
                  href={item.href}
                  target={item.target || undefined}
                  rel={item.target === "_blank" ? "noopener noreferrer" : undefined}
                  className="text-gray-200 hover:text-[#7A0000] text-lg transition-colors"
                  onClick={() => setMenuOpen(false)}
                >
                  {item.name}
                </a>
              </li>
            ))}
            <li className="mt-4">
              <Button size="sm" className="bg-[#7A0000] hover:bg-[#520000] text-white">
                Get Started
              </Button>
            </li>
          </ul>
        </div>
      </header>

      {/* HERO */}
      <section className="relative flex flex-col items-center justify-center text-center min-h-screen px-6 pt-32 pb-16">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-[#7A0000] rounded-full opacity-10 blur-3xl"></div>

        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 mb-8 bg-[#7A0000]/10 border border-[#7A0000]/30 rounded-full">
            <Zap className="w-4 h-4 text-[#7A0000]" />
            <span className="text-sm text-gray-300">AI-Powered Enterprise Platform</span>
          </div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-extrabold mb-6 text-white leading-tight">
            Modern Software Testing
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#ff000034] to-[#7A0000]">
              Reimagined
            </span>
          </h1>

          <p className="max-w-2xl mx-auto text-lg md:text-xl mb-8 text-gray-400">
            Harness the power of artificial intelligence to automate, optimize, and elevate your software testing workflow.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Link to="/login" className="bg-[#7A0000] hover:bg-[#520000] text-white group px-5 py-3 rounded-md inline-flex items-center gap-2">
              Start Building
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>

          <div className="mt-16 lg:mt-24 w-full max-w-5xl mx-auto">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent z-10"></div>
              <img
                src="/src/assets/jarvis.png"
                alt="Dashboard"
                className="rounded-xl shadow-2xl border border-gray-800 w-full"
              />
            </div>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="relative py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, idx) => (
              <div
                key={idx}
                className="group p-8 bg-gradient-to-br from-gray-900 to-black border border-gray-800 rounded-xl hover:border-[#7A0000]/50 transition-all duration-300"
              >
                <div className="w-14 h-14 bg-[#7A0000]/10 rounded-lg flex items-center justify-center mb-6 group-hover:bg-[#7A0000]/20">
                  <feature.icon className="w-7 h-7 text-[#7A0000]" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-white">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* TRUSTED BY */}
      <section className="relative py-24 text-center px-6">
        <div className="max-w-7xl mx-auto">
          <h3 className="text-3xl font-bold mb-4 text-white">Powered By</h3>
          <p className="text-gray-400 mb-12 max-w-2xl mx-auto">
            Built with cutting-edge AI technology from industry leaders
          </p>

          <div className="flex flex-wrap justify-center items-center gap-12">
            {clients.map((client, idx) => (
              <div key={idx} className="group">
                <img
                  src={client.src}
                  alt={client.name}
                  className="h-16 object-contain grayscale opacity-60 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-300"
                />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="bg-gradient-to-br from-[#7A0000]/20 to-transparent border border-[#7A0000]/30 rounded-2xl p-12">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">
              Ready to Transform Your Testing?
            </h2>
            <p className="text-xl text-gray-400 mb-8">
              Join thousands of teams already using J.A.R.V.I.S to ship better software faster.
            </p>
            <Link to="/login" className="bg-[#7A0000] hover:bg-[#520000] text-white group px-5 py-3 rounded-md inline-flex items-center gap-2">
              Try A Demo
            </Link>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-gray-800 py-8 px-6">
        <div className="max-w-7xl mx-auto text-center text-gray-500 text-sm">
          2025 J.A.R.V.I.S. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

import Login from "./pages/Login.jsx";
import Main from "./pages/Main.jsx";
import Interviewer from "./pages/Interviewer.jsx";
import InterviewerNew from "./pages/InterviewerNew.jsx";
import SmartAccess from "./pages/SmartAccess.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HeroSection />} />
      <Route path="/login" element={<Login />} />
      <Route path="/main" element={<Main />} />
      <Route path="/interviewer" element={<Interviewer />} />
      <Route path="/interviewer/new" element={<InterviewerNew />} />
      <Route path="/smart-access" element={<SmartAccess />} />
    </Routes>
  );
}