import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, List

from faster_whisper import WhisperModel
import threading
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore
import re
try:
    # Load environment variables from a .env file if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def log(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def has_input_format(fmt: str) -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", fmt, "-i", "dummy"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return "Unknown input format" not in proc.stdout
    except Exception:
        return False


def build_ffmpeg_audio_cmd(
    speaker_api: str,
    mic_name: Optional[str],
    speaker_name: Optional[str],
    out_pattern: str,
    segment_time: int,
) -> List[str]:
    if not mic_name and not speaker_name:
        raise SystemExit("Provide at least --mic-name or --speaker-name")

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
    ]

    input_index = 0
    mic_index = None
    spk_index = None

    if mic_name:
        cmd += [
            "-thread_queue_size", "1024",
            "-f", "dshow",
            "-i", f"audio={mic_name}",
        ]
        mic_index = input_index
        input_index += 1

    if speaker_name:
        if speaker_api == "wasapi" and has_input_format("wasapi"):
            cmd += [
                "-thread_queue_size", "1024",
                "-f", "wasapi",
                "-i", f"{speaker_name}:loopback",
            ]
        else:
            cmd += [
                "-thread_queue_size", "1024",
                "-f", "dshow",
                "-i", f"audio={speaker_name}",
            ]
        spk_index = input_index
        input_index += 1

    filter_complex: List[str] = []
    maps: List[str] = []

    if mic_index is not None and spk_index is not None:
        filter_complex = [
            "-filter_complex",
            f"[{mic_index}:a][{spk_index}:a]amix=inputs=2:normalize=0[aout]",
            "-map", "[aout]",
        ]
    elif mic_index is not None:
        maps = ["-map", f"{mic_index}:a:0"]
    elif spk_index is not None:
        maps = ["-map", f"{spk_index}:a:0"]

    # Audio formatting and segmentation into 5s wav files
    audio_fmt = [
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-segment_format", "wav",
        "-reset_timestamps", "1",
        out_pattern,
    ]

    return cmd + (filter_complex if filter_complex else []) + maps + audio_fmt


def transcribe_loop(model_name: str, input_dir: Path, transcript_path: Path, keep_last: int, stop_event: Optional[threading.Event] = None):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    seen = set()
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    log("Transcriber: ready. Watching for new segments...")

    def wait_until_complete(p: Path, timeout_s: float = 5.0, settle_ms: int = 300) -> bool:
        """Wait until file size stabilizes (ffmpeg finished writing). Returns True if stable, False on timeout."""
        end = time.time() + timeout_s
        last_size = -1
        while time.time() < end:
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                time.sleep(settle_ms / 1000.0)
                continue
            if size > 0 and size == last_size:
                return True
            last_size = size
            time.sleep(settle_ms / 1000.0)
        return False

    def prune_old_segments(dir_path: Path, keep: int):
        if keep <= 0:
            return
        files = sorted(dir_path.glob("seg_*.wav"))
        if len(files) <= keep:
            return
        to_delete = files[: len(files) - keep]
        for f in to_delete:
            try:
                f.unlink()
                log(f"Deleted old segment {f.name}")
            except Exception as e:
                log(f"Could not delete {f.name}: {e}")

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            # Prune early to avoid buildup if producer outpaces consumer
            prune_old_segments(input_dir, keep_last)
            wavs = sorted(input_dir.glob("seg_*.wav"))
            new_files = [p for p in wavs if p not in seen]
            for p in new_files:
                seen.add(p)
                log(f"Transcribing {p.name} ...")
                # Wait for file to be fully written
                if not wait_until_complete(p):
                    log(f"Skipping {p.name}: file did not stabilize in time")
                    continue

                # Skip tiny files (e.g., < 1 KB) which are likely incomplete
                try:
                    if p.stat().st_size < 1024:
                        log(f"Skipping {p.name}: too small ({p.stat().st_size} bytes)")
                        continue
                except FileNotFoundError:
                    continue

                try:
                    segments, info = model.transcribe(str(p), language="en", vad_filter=True, beam_size=5)
                    text = "".join(s.text for s in segments).strip()
                except Exception as e:
                    log(f"Failed to transcribe {p.name}: {e}")
                    text = ""

                # Append plain text (no filename prefix) to transcript file
                if text:
                    with transcript_path.open("a", encoding="utf-8") as f:
                        f.write(f"{text}\n")
                    # Also print to console
                    print(text, flush=True)
            # Prune old segments to avoid size creep
            prune_old_segments(input_dir, keep_last)
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("Transcriber: stopping.")


def summarize_with_gemini(transcript_path: Path, summary_path: Path) -> Optional[str]:
    """Summarize the transcript with Gemini. Returns the summary text or None on failure."""
    try:
        content = transcript_path.read_text(encoding="utf-8")
    except Exception as e:
        log(f"Could not read transcript: {e}")
        return None
    if not content.strip():
        log("Transcript is empty; skipping Gemini summarization")
        return None
    # Clean any legacy prefixes like: [seg_000123.wav] ...
    try:
        cleaned_lines = []
        pattern = re.compile(r"^\s*\[[^\]]*?\.wav\]\s*")
        for line in content.splitlines():
            cleaned_lines.append(pattern.sub("", line))
        content = "\n".join(cleaned_lines)
    except Exception:
        pass
    if genai is None:
        log("google-generativeai not installed; skipping Gemini summarization")
        return None
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        log("GOOGLE_API_KEY not set; skipping Gemini summarization")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "You are an interview analyst. Read the following raw transcript (may contain fillers and glitches) "
            "and produce a concise, objective profile of the speaker.\n\n"
            "Required sections:\n"
            "1) Communication style (tone, clarity, confidence, politeness, aggression, filler usage).\n"
            "2) Reasoning and coherence (logical flow, contradictions, specificity).\n"
            "3) Topics and interests (key themes mentioned).\n"
            "4) Sentiment and attitude (overall mood and stance).\n"
            "5) Potential risks/red flags (e.g., hostility, vagueness, overclaims).\n"
            "6) Verdict: a short 2–3 sentence summary of what kind of person this appears to be, strictly based on speech patterns and word choice.\n\n"
            "Output in plain text. Avoid quoting long spans.\n\n"
            "Transcript follows:\n\n" + content
        )
        resp = model.generate_content(prompt)
        summary = getattr(resp, "text", "").strip()
        if not summary:
            log("Gemini returned empty summary")
            return None
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")
        log(f"Wrote Gemini summary to {summary_path}")
        return summary
    except Exception as e:
        log(f"Gemini summarization failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Record 5s audio chunks and transcribe each with faster-whisper")
    ap.add_argument("--mic-name", type=str, default=None, help="DirectShow microphone device name")
    ap.add_argument("--speaker-name", type=str, default=None, help="Speaker device name (WASAPI loopback or dshow)")
    ap.add_argument("--speaker-api", type=str, default="wasapi", choices=["wasapi", "dshow"], help="API for speaker capture")
    ap.add_argument("--segment-time", type=int, default=5)
    ap.add_argument("--workdir", type=str, default="chunks", help="Directory to store 5s wav chunks")
    ap.add_argument("--model", type=str, default="small.en")
    ap.add_argument("--keep-last", type=int, default=10, help="Keep only the most recent N segment files (default: 10)")
    ap.add_argument("--transcript", type=str, default="output/transcript.txt")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(workdir / "seg_%06d.wav")

    # Build ffmpeg command to write 5s wav files
    cmd = build_ffmpeg_audio_cmd(
        speaker_api=args.speaker_api,
        mic_name=args.mic_name,
        speaker_name=args.speaker_name,
        out_pattern=out_pattern,
        segment_time=args.segment_time,
    )

    log("\nRecording 5s segments with ffmpeg:")
    log(" ", " ".join(cmd))
    log("Output directory:", str(workdir.resolve()))
    log("Transcript file:", str(Path(args.transcript).resolve()))
    log("Press Ctrl+C to stop.")

    # Start ffmpeg producer
    proc = subprocess.Popen(cmd)

    try:
        # Start transcriber loop
        transcribe_loop(args.model, workdir, Path(args.transcript), args.keep_last)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
