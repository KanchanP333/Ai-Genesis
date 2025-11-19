import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from segment_transcribe import build_ffmpeg_audio_cmd, transcribe_loop, summarize_with_gemini
import re


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Whisper Segmented Transcriber")
        self.proc: Optional[subprocess.Popen] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self._last_transcript_size = 0

        self.var_mic = tk.StringVar()
        self.var_spk = tk.StringVar()
        self.var_api = tk.StringVar(value="wasapi")
        self.var_segment = tk.IntVar(value=5)
        self.var_model = tk.StringVar(value="small.en")
        self.var_keep = tk.IntVar(value=10)
        self.var_workdir = tk.StringVar(value=str(Path("chunks").resolve()))
        self.var_transcript = tk.StringVar(value=str(Path("output/transcript.txt").resolve()))

        self._build_ui()
        self._refresh_devices()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        ttk.Label(frm, text="Microphone").grid(row=0, column=0, sticky="w")
        self.cmb_mic = ttk.Combobox(frm, textvariable=self.var_mic, width=50)
        self.cmb_mic.grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(frm, text="Speaker").grid(row=1, column=0, sticky="w")
        self.cmb_spk = ttk.Combobox(frm, textvariable=self.var_spk, width=50)
        self.cmb_spk.grid(row=1, column=1, sticky="ew", padx=6)

        ttk.Label(frm, text="Speaker API").grid(row=2, column=0, sticky="w")
        self.cmb_api = ttk.Combobox(frm, textvariable=self.var_api, values=["wasapi", "dshow"], state="readonly")
        self.cmb_api.grid(row=2, column=1, sticky="w", padx=6)

        btn_refresh = ttk.Button(frm, text="Refresh Devices", command=self._refresh_devices)
        btn_refresh.grid(row=0, column=2, rowspan=3, sticky="nsw", padx=6)

        ttk.Label(frm, text="Model").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_model, width=24).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="Segment (s)").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(frm, from_=1, to=60, textvariable=self.var_segment, width=10).grid(row=4, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="Keep last").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(frm, from_=0, to=1000, textvariable=self.var_keep, width=10).grid(row=5, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="Workdir").grid(row=6, column=0, sticky="w")
        ent_work = ttk.Entry(frm, textvariable=self.var_workdir, width=60)
        ent_work.grid(row=6, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="...", command=self._choose_workdir).grid(row=6, column=2, sticky="w")

        ttk.Label(frm, text="Transcript").grid(row=7, column=0, sticky="w")
        ent_out = ttk.Entry(frm, textvariable=self.var_transcript, width=60)
        ent_out.grid(row=7, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="...", command=self._choose_transcript).grid(row=7, column=2, sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=3, pady=8, sticky="w")
        self.btn_start = ttk.Button(btns, text="Start", command=self.start)
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.btn_start.grid(row=0, column=0, padx=(0, 8))
        self.btn_stop.grid(row=0, column=1)

        txt_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        txt_frame.grid(row=1, column=0, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.txt = tk.Text(txt_frame, height=20, wrap="word")
        self.txt.grid(row=0, column=0, sticky="nsew")
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)
        sb = ttk.Scrollbar(txt_frame, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set)

    def _choose_workdir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_workdir.set(d)

    def _choose_transcript(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if f:
            self.var_transcript.set(f)

    def _refresh_devices(self):
        mics = self._list_dshow_devices()
        spk = self._list_wasapi_devices()
        if not spk:
            spk = self._list_dshow_devices()
        self.cmb_mic["values"] = mics
        self.cmb_spk["values"] = spk
        if mics and not self.var_mic.get():
            self.var_mic.set(mics[0])
        if spk and not self.var_spk.get():
            self.var_spk.set(spk[0])

    def _list_dshow_devices(self) -> List[str]:
        try:
            proc = subprocess.run([
                "ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=8)
            return self._parse_device_names(proc.stdout)
        except Exception:
            return []

    def _list_wasapi_devices(self) -> List[str]:
        try:
            proc = subprocess.run([
                "ffmpeg", "-hide_banner", "-f", "wasapi", "-list_devices", "true", "-i", "dummy"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=8)
            return self._parse_device_names(proc.stdout)
        except Exception:
            return []

    def _parse_device_names(self, text: str) -> List[str]:
        names: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("\"") and line.endswith("\"") and len(line) > 2:
                names.append(line.strip('"'))
            else:
                i = line.find('"')
                j = line.rfind('"')
                if 0 <= i < j:
                    names.append(line[i+1:j])
        uniq = []
        seen = set()
        for n in names:
            if n and n not in seen:
                uniq.append(n)
                seen.add(n)
        return uniq

    def start(self):
        if self.worker or self.proc:
            return
        mic = self.var_mic.get().strip() or None
        spk = self.var_spk.get().strip() or None
        api = self.var_api.get().strip()
        seg = int(self.var_segment.get())
        model = self.var_model.get().strip()
        keep = int(self.var_keep.get())
        workdir = Path(self.var_workdir.get()).expanduser()
        transcript = Path(self.var_transcript.get()).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        transcript.parent.mkdir(parents=True, exist_ok=True)

        if not mic and not spk:
            messagebox.showerror("Error", "Select at least a microphone or speaker device")
            return

        out_pattern = str((workdir / "seg_%06d.wav").resolve())
        try:
            cmd = build_ffmpeg_audio_cmd(
                speaker_api=api,
                mic_name=mic,
                speaker_name=spk,
                out_pattern=out_pattern,
                segment_time=seg,
            )
        except SystemExit as e:
            messagebox.showerror("Error", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build ffmpeg command: {e}")
            return

        try:
            self.proc = subprocess.Popen(cmd)
        except FileNotFoundError:
            messagebox.showerror("Error", "ffmpeg not found. Please install ffmpeg and ensure it is in PATH.")
            self.proc = None
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start ffmpeg: {e}")
            self.proc = None
            return

        self.stop_event = threading.Event()
        self.worker = threading.Thread(
            target=transcribe_loop,
            args=(model, workdir, transcript, keep, self.stop_event),
            daemon=True,
        )
        self.worker.start()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._last_transcript_size = 0
        self._schedule_tail()

    def _schedule_tail(self):
        self._update_transcript_view()
        if self.worker is not None:
            self.root.after(500, self._schedule_tail)

    def _update_transcript_view(self):
        try:
            p = Path(self.var_transcript.get())
            if p.exists():
                s = p.stat().st_size
                if s != self._last_transcript_size:
                    self._last_transcript_size = s
                    with p.open("r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    lines = content.splitlines()[-200:]
                    self.txt.delete("1.0", tk.END)
                    self.txt.insert(tk.END, "\n".join(lines))
                    self.txt.see(tk.END)
        except Exception:
            pass

    def stop(self):
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        if self.worker:
            self.worker = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        # Post-processing: clean transcript prefixes and run Gemini summarization
        try:
            transcript = Path(self.var_transcript.get())
            if transcript.exists():
                try:
                    text = transcript.read_text(encoding="utf-8")
                    pattern = re.compile(r"^\s*\[[^\]]*?\.wav\]\s*", re.MULTILINE)
                    cleaned = pattern.sub("", text)
                    if cleaned != text:
                        transcript.write_text(cleaned, encoding="utf-8")
                except Exception:
                    pass
                summary_path = transcript.parent / "summary.txt"
                out = summarize_with_gemini(transcript, summary_path)
                if out:
                    messagebox.showinfo("Gemini Summary", f"Summary written to: {summary_path}")
        except Exception:
            pass

    def on_close(self):
        try:
            self.stop()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.geometry("900x600")
    root.mainloop()


if __name__ == "__main__":
    main()
