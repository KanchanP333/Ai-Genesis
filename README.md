# J.A.R.V.I.S
### Just. A. Reliable. Virtual. Intelligence. System

## Overview
J.A.R.V.I.S reimagines how organizations build and manage their teams.  
It’s an AI-powered workforce intelligence platform that authenticates candidates, evaluates talent, assigns roles, tracks performance, and assembles high-performing teams to maximize organizational productivity.

## 1. Auto Team Assembler

An AI-assisted web app to analyze `.txt` documents, chat with an LLM (via OpenRouter), and extract an Employees table that you can download or use to email individuals directly from a modal UI.

## Features
- **Chat with AI** using OpenRouter
- **.txt uploads** (client + server enforced)
- **Markdown reply rendering** with safe HTML escaping
- **Employees table detection** from LLM reply
  - Robust parsing of Markdown tables (handles code fences, alignment colons, leading/trailing pipes, light markdown in cells)
  - Flexible header matching for `EID/ID`, `Name`, `Email`
- **Download table** as a text file
- **Add Employees to Project** modal
  - Shows parsed `EID | Name | Email`
  - Project name input
  - Per-employee Gmail compose buttons

## Tech Stack
- Frontend: Vanilla HTML/CSS/JS
- Backend: Node.js + Express + Multer
- LLM: OpenRouter API (model configurable via `.env`)

## Getting Started

### Prerequisites
- Node.js 18+
- An OpenRouter API key: https://openrouter.ai/

### Install
```bash
npm install
```

### Environment
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=sk-or-v1-...your-key...
PORT=3000
OPENROUTER_MODEL=openai/gpt-oss-120b
SYSTEM_PROMPT="You are SyncLab Assistant. Be helpful, concise, and neutral. Whenever asked to return any employees table, present it as a clean Markdown table and always include an 'EID' column as the first column. Use readable column headers and include no raw JSON or extra explanatory text inside the table. Outside the table, you may provide a one-line summary."
```

Notes:
- Do NOT commit real API keys to version control.
- You can freely modify `SYSTEM_PROMPT` to guide the model’s output style (e.g., always produce a Markdown table with `EID | Name | Email`).

### Run
```bash
npm start
# Server running at http://localhost:3000
```
Open http://localhost:3000 in your browser.

## Usage
1. Type a message asking for an employee list (e.g., “List potential team members with EID, Name, Email”).
2. Optionally upload `.txt` files to provide context. `.txt` is accepted.
3. Submit. The reply is rendered with simple Markdown formatting.
4. If the reply contains a Markdown table, you’ll see:
   - **Download Table**: saves only the table as `employees-table.txt`.
   - **Add Employees to Project**: opens a modal displaying the parsed table.
5. In the modal:
   - Enter a project name (optional).
   - Click **Email** next to a person to open Gmail compose with prefilled to/subject/body.

## API Endpoints
- `GET /health` → `{ ok: true }` for health checks
- `POST /upload` → In-memory `.txt` file acceptance check (used internally)
- `POST /chat` → Core endpoint: forwards text + files to LLM and returns structured reply

## Implementation Notes
- File uploads are processed in-memory via Multer; nothing is persisted.
- Reply rendering uses `escapeHtml` and `formatReply` to prevent HTML injection.
- Table detection is resilient to:
  - Code fences around tables (``` blocks)
  - Alignment patterns like `| :--- | ---: |`
  - Leading/trailing pipes per row
  - Mild markdown within cells (bold, code)
- Column alignment is normalized so EID/Name/Email line up with headers.

## Customizing the System Prompt
Edit `.env` → `SYSTEM_PROMPT`. Suggested guidance:
- Ask the model to always include a Markdown table with the first column named `EID`.
- Keep one-line summaries outside the table.

## Project Structure
```
AutoTeamAssembler/
├─ index.html        # Frontend UI + chat client + table parsing + modal
├─ server.js         # Express server, OpenRouter integration, endpoints
├─ package.json      # Scripts and dependencies
├─ .env              # Local environment variables (not committed)
└─ README.md         # This file
```

## Scripts
- `npm start` → start the Express server

## Troubleshooting
- Button not visible:
  - Ensure the LLM replied with a proper Markdown table (header row, separator row, then data rows).
  - Refresh the browser after making code changes.
- “.txt files are accepted”:
  - The backend enforces `.txt`. Convert other files to plain text before uploading.
- Emails not opening:
  - The app uses a Gmail compose URL. Ensure you’re logged into Gmail in the browser.

## Security
- Keep your `OPENROUTER_API_KEY` private and out of version control.

## 2. Performance Meter (Windows)

Edge-device activity meter with rules-based privacy, role profiles, local metrics aggregation, and optional Gemini scoring.

## Features
- Rules-driven privacy: exclude specific apps from keystroke/mouse metrics.
- Tracks per-app active time, words typed, backspaces, and cursor movement.
- Role profiles: engineer, hr, coder (editable in `profiles.yaml`).
- Local JSONL logs under `data/`.
- Optional Gemini API scoring (off by default).

## Quick start
1. Install Python 3.10+ (Windows).
2. Create venv and install deps:
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
3. Configure rules and profiles:
   - Edit `rules.txt` to list personal apps to exclude from input metrics.
   - Edit `profiles.yaml` to adjust role expectations.
4. (Optional) Create `.env` for Gemini (2.5 Flash):
   ```
   GEMINI_API_KEY=your_key
   GEMINI_MODEL=gemini-2.5-flash
   ```
5. Run:
   ```bash
   python -m perfmeter --role coder --rules rules.txt
   ```

## Rules file format
```
# lines starting with # are comments
[exclude_apps]
# exe names, case-insensitive
chrome.exe
spotify.exe
whatsapp.exe

[include_apps]
# optional; if provided, only these apps are tracked for metrics/time
# not set by default
```

## Privacy
- No content captured (no key text, only counts and word boundaries).
- When an app is excluded, keystroke/mouse metrics are paused. Time-in-app is still tracked.
- Data is stored locally as JSONL. You own it.

## Notes
- Requires Windows with desktop access.
- Some features may require normal user privileges; admin not required.
- If you use corporate lockdowns, hooks may be blocked.

### Gemini API
- Default model: `gemini-2.5-flash`.
- REST endpoint: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`.
- Auth: `x-goog-api-key` header is used by the client.
## 3. Smart Interviewer (Windows)

Real‑time Windows audio capture (mic and/or system audio), transcription via Faster‑Whisper, and end‑of‑session analysis with Google Gemini.

- Capture: FFmpeg (DirectShow; optional WASAPI if your FFmpeg supports it)
- Transcription: Faster‑Whisper (`small.en` by default)
- Summary: Gemini `google-generativeai` written to `output/summary.txt`
- Interfaces: GUI (`gui.py`) and CLI (`segment_transcribe.py`)

## Features

- **Mic + Speakers mix** using FFmpeg `amix` when both are selected.
- **Chunked capture** into 5s 16kHz mono WAV files under `chunks/`.
- **Continuous transcription**; plain text appended to `output/transcript.txt`.
- **End summary** generated by Gemini after you stop in the GUI (or via a CLI one‑liner).

## Requirements

- Windows 10/11
- Python 3.9+
- FFmpeg on PATH
- Python packages (see `requirements.txt`):
  - `faster-whisper`
  - `google-generativeai`
  - `python-dotenv`

Install:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment (.env)

Create a `.env` file in project root:

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
# Optional: select model supported for your account/region
# GEMINI_MODEL=gemini-1.5-flash
```

The code auto-loads `.env`. You can also export variables in your shell.

## Discover devices

List exact device names for FFmpeg:

```powershell
# DirectShow devices (audio/video)
ffmpeg -hide_banner -list_devices true -f dshow -i dummy

# WASAPI devices (only if your ffmpeg supports wasapi)
ffmpeg -hide_banner -list_devices true -f wasapi -i dummy
```

If names have Unicode or spaces, prefer the "Alternative name" GUID line FFmpeg prints.

## Run (GUI)

```powershell
python gui.py
```

Steps:

- Select Mic and Speaker devices (choose `Speaker API` dshow or wasapi).
- Click Start, then Stop. On Stop, transcript is cleaned and Gemini summary is written to `output/summary.txt`.

## Run (CLI)

Replace names with your exact devices from the FFmpeg list.

DirectShow Stereo Mix only:

```powershell
python segment_transcribe.py `
  --speaker-name "Stereo Mix (Realtek(R) Audio)" `
  --speaker-api dshow `
  --segment-time 5 `
  --workdir chunks `
  --model small.en `
  --keep-last 10 `
  --transcript output/transcript.txt
```

DirectShow Mic + Stereo Mix (mixed):

```powershell
python segment_transcribe.py `
  --mic-name "@device_cm_{...}\\wave_{...}" `
  --speaker-name "@device_cm_{...}\\wave_{...}" `
  --speaker-api dshow `
  --segment-time 5 `
  --workdir chunks `
  --model small.en `
  --keep-last 10 `
  --transcript output/transcript.txt
```

Generate summary after CLI run:

```powershell
python -c "from segment_transcribe import summarize_with_gemini; import pathlib; summarize_with_gemini(pathlib.Path('output/transcript.txt'), pathlib.Path('output/summary.txt'))"
```

## Project structure

- `gui.py` — Tk GUI launcher
- `segment_transcribe.py` — chunked capture/transcribe + Gemini summary
- `audio-capture/recorder.py` — example FFmpeg screen/audio recorder
- `chunks/` — generated WAV segments
- `output/transcript.txt` — rolling transcript
- `output/summary.txt` — Gemini analysis
- `.gitignore` — excludes large weights and outputs
- `requirements.txt` — Python dependencies

## Troubleshooting

- If WASAPI is unavailable: use DirectShow devices and alternative GUID names.
- First run is slow: model downloads for Faster‑Whisper.
- Gemini 404: upgrade `google-generativeai` and select a supported `GEMINI_MODEL`.

## 4. Enterprise RAG Backend (FastAPI)

Robust, role- and project-aware Retrieval-Augmented Generation (RAG) API that powers the chat.

### Why
- Enforce authorization by role/level/dept/project when retrieving documents.
- Keep the chat fast and auditable with explicit search/rerank/generate stages.

### How it works
- Auth: JWT via `POST /auth/login` returns `access_token`. Set it in the SPA `Authorization: Bearer <token>`.
- Upload: `POST /documents` accepts `.txt`, chunks, embeds with Gemini embeddings, and upserts into Qdrant with payload
  `{uploader, dept, project, role allowlist, uploader_level, filename, text}`.
- Search: `POST /query` embeds the question, performs filtered vector search in Qdrant restricted by
  `uploader_level <= user.level`, `allow_roles` contains user.role, and matching `dept` and `project`.
- Rerank: Lightweight local reranker scores top passages and keeps the best K.
- Generate: Gemini model synthesizes the answer from reranked contexts. Returns `answer` and `sources` (payloads).

### Core endpoints
- `POST /auth/login` -> `{ access_token }`
- `GET /me` -> `{ username, role, level, dept, project }`
- `POST /documents` -> `{ ids }` (txt-only uploads)
- `POST /query` -> `{ answer: str, sources: List[payload] }`
- `GET /` -> redirects to the SPA (Vite dev server)

### Environment
- `JWT_SECRET` secret used by the backend for signing tokens
- `GEMINI_API_KEY` Google Generative AI key
- `QDRANT_URL` e.g. `http://localhost:6333`
- Optional: `PROJECTS` comma-delimited default projects when Accounts DB is absent

### Storage
- Qdrant collection `documents`, 768-dim vectors, COSINE distance, payload carries text and metadata.

## 5. J.A.R.V.I.S Frontend (React + Vite)

Modern SPA that hosts the main chat experience and navigation across all modules.

### UX highlights
- Single sticky top navigation (black/red) linking to Advanced Interviewer, Auto Team, Performance Meter, Smart Access, Screening Admin.
- Full-screen chat: messages fill the page; input and Send are anchored at the bottom.
- Upload button opens backend `/demo/upload?token=...` to ingest `.txt` quickly.
- Sources toggle per AI message: expand to view each supporting chunk’s filename, project, and text.

### Behavior collection (Smart Access)
- On login, the SPA collects lightweight behavior metrics for ~30 seconds (mouse movement counts, typing CPM, burstiness)
  and posts to `/smart-access/collect` every 5s. Collection is resilient if Qdrant is down.

### Dev
- Runs automatically from backend startup (spawns `npm run dev` in `J.A.R.V.I.S/`).

## 6. Smart Access (Behavior Analytics)

Collects client-side behavior signals and performs anomaly scoring to flag unusual sessions.

### Why
- Establish a personal baseline per employee to detect sudden deviations (possible account sharing, unusual usage patterns).

### How it works
- Client posts JSON events to `POST /smart-access/collect` including `employee_id`, `page`, `mouse_moves`, `typing_cpm`, `typing_burstiness`, `device_id`, `timestamp`.
- Server computes an embedding for the event vector and, if Qdrant is available, stores it in collection `smart_access`.
- Baseline policy: requires ~30 unique days of historical events before enabling scoring to reduce false positives.
- Scoring: cosine similarity vs. centroid of baseline vectors; if below `THRESHOLD`, marks `flagged=true`.
- Device familiarity: `seen_device_before(employee_id, device_id)` returns False gracefully when Qdrant is unavailable.

### Resilience
- If Qdrant is down or unreachable, API still returns `200` with `{ ok: true, stored: false, score: null, flagged: false }`.

### Admin UI
- `GET /smart-access/admin?token=...` to view and adjust `THRESHOLD` and `BASELINE_DAYS`.

## 7. Screening Admin (FastAPI)

Administrative UI for managing jobs and candidates for structured screening.

### Why
- Centralize job postings and candidate tracking with a simple admin console.

### Features
- Jobs listing, candidate lists per job, job details.
- Uniform top navigation and dark theme.
- Token-aware links: if `?token=...` is present, they propagate to protected pages.

### API
- `GET /screening/api/jobs`
- `GET /screening/api/jobs/{job_id}/candidates`

### UI
- `GET /screening/jobs?token=...`

## 8. Advanced AI Interviewer (Web UI)

In-browser transcript summarization UI and optional Windows recorder endpoints mounted under FastAPI.

### Why
- Quickly paste/upload transcripts and generate structured interview summaries with Gemini.

### Web endpoints
- `GET /interviewer-advanced` — Web page with nav, paste/upload text, summarize button.
- `POST /interviewer-advanced/summary` — Accepts pasted text or `.txt` file, returns `{ ok, summary }`.
- Recorder (Windows only): `/interviewer-advanced/record/{start,stop,status}` invoked by the UI;
  backed by FFmpeg + Whisper loop when available.

### Notes
- If Windows-specific deps are unavailable, recording endpoints are disabled; the summarize flow still works.
- Nav links are token-aware for protected areas like Smart Access and Screening.

## 9. Running the full stack

### Prerequisites
- Python 3.12+, Node 18+, Docker (for Qdrant)

### Start services
```bash
docker compose up -d             # starts Qdrant at http://localhost:6333
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
# Backend will spawn the J.A.R.V.I.S Vite dev server automatically
```

### Upload + chat
- Open http://localhost:5173, login, click Upload Documents (opens `/demo/upload?token=...`), then chat.

### Troubleshooting Qdrant
- Connection refused: ensure the container is up; verify `QDRANT_URL` env.
- If down: RAG queries and uploads will error; Smart Access will keep working but skip store/score.

---
## Legacy content (Whisper model card)

---
language:
- en
- zh
- de
- es
- ru
- ko
- fr
- ja
- pt
- tr
- pl
- ca
- nl
- ar
- sv
- it
- id
- hi
- fi
- vi
- he
- uk
- el
- ms
- cs
- ro
- da
- hu
- ta
- 'no'
- th
- ur
- hr
- bg
- lt
- la
- mi
- ml
- cy
- sk
- te
- fa
- lv
- bn
- sr
- az
- sl
- kn
- et
- mk
- br
- eu
- is
- hy
- ne
- mn
- bs
- kk
- sq
- sw
- gl
- mr
- pa
- si
- km
- sn
- yo
- so
- af
- oc
- ka
- be
- tg
- sd
- gu
- am
- yi
- lo
- uz
- fo
- ht
- ps
- tk
- nn
- mt
- sa
- lb
- my
- bo
- tl
- mg
- as
- tt
- haw
- ln
- ha
- ba
- jw
- su
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
- unsloth
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
pipeline_tag: automatic-speech-recognition
license: apache-2.0
base_model:
- openai/whisper-large-v3
---
<div>
  <p style="margin-bottom: 0; margin-top: 0;">
    <strong>See <a href="https://huggingface.co/collections/unsloth/text-to-speech-tts-models-68007ab12522e96be1e02155">our collection</a> for all our TTS model uploads.</strong>
  </p>
  <p style="margin-bottom: 0;">
    <em>Learn to fine-tune TTS models - <a href="https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning">Read our Guide</a>.</em>
  </p>
<p style="margin-top: 0;margin-bottom: 0;">
    <em><a href="https://docs.unsloth.ai/basics/unsloth-dynamic-v2.0-gguf">Unsloth Dynamic 2.0</a> achieves superior accuracy & outperforms other leading quants.</em>
  </p>
  <div style="display: flex; gap: 5px; align-items: center; ">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
    <a href="https://discord.gg/unsloth">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png" width="173">
    </a>
    <a href="https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning">
      <img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="143">
    </a>
  </div>
<h1 style="margin-top: 0rem;">✨ Run & Fine-tune TTS models with Unsloth!</h1>
</div>

- Fine-tune TTS models for free using our Google [Colab notebooks here](https://docs.unsloth.ai/get-started/unsloth-notebooks#text-to-speech-tts-notebooks)!
- Read our Blog about TTS support: [unsloth.ai/blog/tts](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)

| Unsloth supports          |    Free Notebooks                                                                                           | Performance | Memory use |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **Orpheus-TTS**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5x faster | 58% less |
| **Whisper Large V3**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Whisper.ipynb)               | 1.5x faster | 50% less |
| **Qwen3 (14B)**      | [▶️ Start on Colab](https://docs.unsloth.ai/get-started/unsloth-notebooks)               | 2x faster | 70% less |
| **Llama 3.2 Vision (11B)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)               | 1.8x faster | 50% less |

# Whisper

Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation, proposed in the paper 
[Robust Speech Recognition via Large-Scale Weak Supervision](https://huggingface.co/papers/2212.04356) by Alec Radford 
et al. from OpenAI. Trained on >5M hours of labeled data, Whisper demonstrates a strong ability to generalise to many 
datasets and domains in a zero-shot setting.

Whisper large-v3 has the same architecture as the previous [large](https://huggingface.co/openai/whisper-large) and [large-v2](https://huggingface.co/openai/whisper-large-v2) 
models, except for the following minor differences:

1. The spectrogram input uses 128 Mel frequency bins instead of 80
2. A new language token for Cantonese

The Whisper large-v3 model was trained on 1 million hours of weakly labeled audio and 4 million hours of pseudo-labeled 
audio collected using Whisper [large-v2](https://huggingface.co/openai/whisper-large-v2) . The model was trained for 2.0 epochs over this mixture dataset.

The large-v3 model shows improved performance over a wide variety of languages, showing 10% to 20% reduction of errors 
compared to Whisper [large-v2](https://huggingface.co/openai/whisper-large-v2) . For more details on the different checkpoints available, refer to the section [Model details](#model-details).

**Disclaimer**: Content for this model card has partly been written by the 🤗 Hugging Face team, and partly copied and 
pasted from the original model card.

## Usage

Whisper large-v3 is supported in Hugging Face 🤗 Transformers. To run the model, first install the Transformers 
library. For this example, we'll also install 🤗 Datasets to load toy audio dataset from the Hugging Face Hub, and 
🤗 Accelerate to reduce the model loading time:

```bash
pip install --upgrade pip
pip install --upgrade transformers datasets[audio] accelerate
```

The model can be used with the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
class to transcribe audios of arbitrary length:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:

```python
result = pipe("audio.mp3")
```

Multiple audio files can be transcribed in parallel by specifying them as a list and setting the `batch_size` parameter:

```python
result = pipe(["audio_1.mp3", "audio_2.mp3"], batch_size=2)
```

Transformers is compatible with all Whisper decoding strategies, such as temperature fallback and condition on previous 
tokens. The following example demonstrates how to enable these heuristics:

```python
generate_kwargs = {
    "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

result = pipe(sample, generate_kwargs=generate_kwargs)
```

Whisper predicts the language of the source audio automatically. If the source audio language is known *a-priori*, it 
can be passed as an argument to the pipeline:

```python
result = pipe(sample, generate_kwargs={"language": "english"})
```

By default, Whisper performs the task of *speech transcription*, where the source audio language is the same as the target
text language. To perform *speech translation*, where the target text is in English, set the task to `"translate"`:

```python
result = pipe(sample, generate_kwargs={"task": "translate"})
```

Finally, the model can be made to predict timestamps. For sentence-level timestamps, pass the `return_timestamps` argument:

```python
result = pipe(sample, return_timestamps=True)
print(result["chunks"])
```

And for word-level timestamps:

```python
result = pipe(sample, return_timestamps="word")
print(result["chunks"])
```

The above arguments can be used in isolation or in combination. For example, to perform the task of speech transcription 
where the source audio is in French, and we want to return sentence-level timestamps, the following can be used:

```python
result = pipe(sample, return_timestamps=True, generate_kwargs={"language": "french", "task": "translate"})
print(result["chunks"])
```

<details>

<summary> For more control over the generation parameters, use the model + processor API directly: </summary>

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Audio, load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

inputs = processor(
    sample["array"],
    sampling_rate=sample["sampling_rate"],
    return_tensors="pt",
    truncation=False,
    padding="longest",
    return_attention_mask=True,
)
inputs = inputs.to(device, dtype=torch_dtype)

gen_kwargs = {
    "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

pred_ids = model.generate(**inputs, **gen_kwargs)
pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)

print(pred_text)
```

</details>

## Additional Speed & Memory Improvements

You can apply additional speed and memory improvements to Whisper to further reduce the inference speed and VRAM 
requirements.

### Chunked Long-Form

Whisper has a receptive field of 30-seconds. To transcribe audios longer than this, one of two long-form algorithms are
required:
1. **Sequential:** uses a "sliding window" for buffered inference, transcribing 30-second slices one after the other
2. **Chunked:** splits long audio files into shorter ones (with a small overlap between segments), transcribes each segment independently, and stitches the resulting transcriptions at the boundaries

The sequential long-form algorithm should be used in either of the following scenarios:
1. Transcription accuracy is the most important factor, and speed is less of a consideration
2. You are transcribing **batches** of long audio files, in which case the latency of sequential is comparable to chunked, while being up to 0.5% WER more accurate

Conversely, the chunked algorithm should be used when:
1. Transcription speed is the most important factor
2. You are transcribing a **single** long audio file

By default, Transformers uses the sequential algorithm. To enable the chunked algorithm, pass the `chunk_length_s` 
parameter to the `pipeline`. For large-v3, a chunk length of 30-seconds is optimal. To activate batching over long 
audio files, pass the argument `batch_size`:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

#### Torch compile

The Whisper forward pass is compatible with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html)
for 4.5x speed-ups.

**Note:** `torch.compile` is currently not compatible with the Chunked long-form algorithm or Flash Attention 2 ⚠️

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

# Enable static cache and compile the forward pass
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# 2 warmup steps
for _ in tqdm(range(2), desc="Warm-up step"):
    with sdpa_kernel(SDPBackend.MATH):
        result = pipe(sample.copy(), generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})

# fast run
with sdpa_kernel(SDPBackend.MATH):
    result = pipe(sample.copy())

print(result["text"])
```

#### Flash Attention 2

We recommend using [Flash-Attention 2](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2) if your GPU supports it and you are not using [torch.compile](#torch-compile). 
To do so, first install [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```
pip install flash-attn --no-build-isolation
```

Then pass `attn_implementation="flash_attention_2"` to `from_pretrained`:

```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
```

#### Torch Scale-Product-Attention (SDPA)

If your GPU does not support Flash Attention, we recommend making use of PyTorch [scaled dot-product attention (SDPA)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). 
This attention implementation is activated **by default** for PyTorch versions 2.1.1 or greater. To check 
whether you have a compatible PyTorch version, run the following Python code snippet:

```python
from transformers.utils import is_torch_sdpa_available

print(is_torch_sdpa_available())
```

If the above returns `True`, you have a valid version of PyTorch installed and SDPA is activated by default. If it 
returns `False`, you need to upgrade your PyTorch version according to the [official instructions](https://pytorch.org/get-started/locally/)

Once a valid PyTorch version is installed, SDPA is activated by default. It can also be set explicitly by specifying 
`attn_implementation="sdpa"` as follows:

```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
```

For more information about how to use the SDPA refer to the [Transformers SDPA documentation](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention).


## Model details

Whisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model. There are two
flavours of Whisper model: English-only and multilingual. The English-only models were trained on the task of English 
speech recognition. The multilingual models were trained simultaneously on multilingual speech recognition and speech 
translation. For speech recognition, the model predicts transcriptions in the *same* language as the audio. For speech 
translation, the model predicts transcriptions to a *different* language to the audio.

Whisper checkpoints come in five configurations of varying model sizes. The smallest four are available as English-only 
and multilingual. The largest checkpoints are multilingual only. All ten of the pre-trained checkpoints 
are available on the [Hugging Face Hub](https://huggingface.co/models?search=openai/whisper). The 
checkpoints are summarised in the following table with links to the models on the Hub:

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |
| large-v3 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v3) |


## Fine-Tuning

The pre-trained Whisper model demonstrates a strong ability to generalise to different datasets and domains. However, 
its predictive capabilities can be improved further for certain languages and tasks through *fine-tuning*. The blog 
post [Fine-Tune Whisper with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) provides a step-by-step 
guide to fine-tuning the Whisper model with as little as 5 hours of labelled data.

### Evaluated Use

The primary intended users of these models are AI researchers studying robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research.

The models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them.

In particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes.


## Training Data

The large-v3 checkpoint is trained on 1 million hours of weakly labeled audio and 4 million hours of pseudo-labeled audio collected using Whisper large-v2. 

As discussed in [the accompanying paper](https://cdn.openai.com/papers/whisper.pdf), we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language.


## Performance and Limitations

Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level. 

However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.

Our models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in [the paper accompanying this release](https://cdn.openai.com/papers/whisper.pdf). 

In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis on these limitations are provided in [the paper](https://cdn.openai.com/papers/whisper.pdf). It is likely that this behavior and hallucinations may be worse on lower-resource and/or lower-discoverability languages.


## Broader Implications

We anticipate that Whisper models’ transcription capabilities may be used for improving accessibility tools. While Whisper models cannot be used for real-time transcription out of the box – their speed and size suggest that others may be able to build applications on top of them that allow for near-real-time speech recognition and translation. The real value of beneficial applications built on top of Whisper models suggests that the disparate performance of these models may have real economic implications.

There are also potential dual use concerns that come with releasing Whisper. While we hope the technology will be used primarily for beneficial purposes, making ASR technology more accessible could enable more actors to build capable surveillance technologies or scale up existing surveillance efforts, as the speed and accuracy allow for affordable automatic transcription and translation of large volumes of audio communication. Moreover, these models may have some capabilities to recognize specific individuals out of the box, which in turn presents safety concerns related both to dual use and disparate performance. In practice, we expect that the cost of transcription is not the limiting factor of scaling up surveillance projects.


### BibTeX entry and citation info
```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```