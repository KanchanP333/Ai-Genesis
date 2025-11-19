import argparse
import shutil
import subprocess
import sys
import datetime
from pathlib import Path
import os
try:
    import msvcrt  # Windows-specific
except ImportError:  # non-Windows
    msvcrt = None

def log(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        log("Error: ffmpeg is not installed or not on PATH.")
        sys.exit(1)


def run_and_capture(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def has_input_format(fmt: str) -> bool:
    """Return True if ffmpeg recognizes the given input format, else False.
    We try initializing with a dummy input and parse the diagnostic.
    """
    out = run_and_capture(["ffmpeg", "-hide_banner", "-f", fmt, "-i", "dummy"])  # type: ignore[arg-type]
    # If unknown, ffmpeg prints: Unknown input format: 'fmt'
    return "Unknown input format" not in out


def detect_screen_grabber(force_gdi: bool = False) -> str:
    """Choose the best available screen capture format on Windows.
    Preference: ddagrab (if available), else gdigrab. If force_gdi is True, try gdigrab first.
    Raises SystemExit if neither is available.
    """
    candidates = ["gdigrab", "ddagrab"] if force_gdi else ["ddagrab", "gdigrab"]
    for fmt in candidates:
        if has_input_format(fmt):
            return fmt
    log("Error: neither 'ddagrab' nor 'gdigrab' is supported by your ffmpeg build.")
    log("Try updating ffmpeg or install a build with Windows screen capture support.")
    sys.exit(2)


def list_dshow_devices():
    out = run_and_capture(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])  # noqa: E501
    video, audio = [], []
    current = None
    for line in out.splitlines():
        line = line.strip()
        if "DirectShow video devices" in line:
            current = "video"
        elif "DirectShow audio devices" in line:
            current = "audio"
        elif line.startswith("\"") and line.endswith("\""):
            name = line.strip('"')
            if current == "video":
                video.append(name)
            elif current == "audio":
                audio.append(name)
    return video, audio


def list_wasapi_devices():
    out = run_and_capture(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "wasapi", "-i", "dummy"])  # noqa: E501
    devices = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("\"") and line.endswith("\""):
            devices.append(line.strip('"'))
    return devices


def choose_from_list(title, items):
    if not items:
        return None
    log(f"\n{title}")
    for i, it in enumerate(items):
        log(f"  [{i}] {it}")
    while True:
        sel = input("Select index (or blank to skip): ").strip()
        if sel == "":
            return None
        if sel.isdigit() and 0 <= int(sel) < len(items):
            return items[int(sel)]
        log("Invalid selection. Try again.")


def build_ffmpeg_cmd(args, mic_name, speaker_name):
    output = args.output
    if not output:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = f"recording_{ts}.mkv"

    cmd = [
        "ffmpeg",
        "-y",
        # Auto-detect screen grabber: prefer ddagrab, fallback to gdigrab; --use-gdigrab forces preference.
        "-f", detect_screen_grabber(force_gdi=args.use_gdigrab),
        "-framerate", str(args.fps),
        "-i", "desktop",
    ]

    input_count = 1

    if mic_name:
        cmd += [
            "-thread_queue_size", "1024",
            "-f", "dshow",
            "-i", f"audio={mic_name}",
        ]
        input_count += 1

    if speaker_name:
        # Prefer WASAPI loopback. If --speaker_api dshow, just use dshow.
        if args.speaker_api == "wasapi":
            if has_input_format("wasapi"):
                cmd += [
                    "-thread_queue_size", "1024",
                    "-f", "wasapi",
                    "-i", f"{speaker_name}:loopback",
                ]
                input_count += 1
            else:
                log("WASAPI not supported by this ffmpeg build. Falling back to DirectShow for speaker if available.")
                cmd += [
                    "-thread_queue_size", "1024",
                    "-f", "dshow",
                    "-i", f"audio={speaker_name}",
                ]
                input_count += 1
        else:
            cmd += [
                "-thread_queue_size", "1024",
                "-f", "dshow",
                "-i", f"audio={speaker_name}",
            ]
            input_count += 1

    # Mapping: video + up to two audio streams
    maps = ["-map", "0:v:0"]
    if mic_name:
        maps += ["-map", "1:a:0"]
    if speaker_name:
        idx = 2 if mic_name else 1
        maps += ["-map", f"{idx}:a:0"]

    # Codecs
    vcodec = ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p"]
    acodec = ["-c:a", "aac", "-b:a", f"{args.audio_bitrate}k"]

    extra = []
    if args.size:
        extra += ["-vf", f"scale={args.size}"]
    if args.duration:
        extra += ["-t", str(args.duration)]

    cmd += maps + vcodec + acodec + extra + [output]
    return cmd, output


def build_ffmpeg_live_audio_cmd(args, mic_name, speaker_name):
    """Build an ffmpeg command that captures mic and/or speaker and writes 16kHz mono s16le PCM to stdout.
    Returns the command list. No video is captured in this mode.
    """
    if not mic_name and not speaker_name:
        log("Error: --live-stdout requires at least one of --mic-name or --speaker-name.")
        sys.exit(3)

    cmd = [
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
        if args.speaker_api == "wasapi" and has_input_format("wasapi"):
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

    filter_complex = []
    maps = []

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

    pcm_params = [
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-sample_fmt", "s16",
        "-f", "s16le",
        "pipe:1",
    ]

    cmd += (filter_complex if filter_complex else []) + maps + pcm_params
    return cmd


def main():
    check_ffmpeg()

    parser = argparse.ArgumentParser(description="Screen + mic + speaker recorder (Windows, ffmpeg)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=str, default=None, help="Ex: 1920:1080; omit for native")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", type=str, default="veryfast", choices=[
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"
    ])
    parser.add_argument("--audio-bitrate", type=int, default=192)
    parser.add_argument("--duration", type=int, default=None, help="Seconds; omit to record until Ctrl+C")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use-gdigrab", action="store_true", help="Use gdigrab instead of ddagrab for screen")
    parser.add_argument("--speaker-api", type=str, default="wasapi", choices=["wasapi", "dshow"], help="How to capture speaker")
    parser.add_argument("--noninteractive", action="store_true", help="Do not prompt; rely on --mic-name/--speaker-name")
    parser.add_argument("--mic-name", type=str, default=None)
    parser.add_argument("--speaker-name", type=str, default=None)
    parser.add_argument("--live-stdout", action="store_true", help="Stream mixed audio (16kHz mono s16le) to stdout for live transcription; no file output and no video capture")

    args = parser.parse_args()

    mic_name = args.mic_name
    speaker_name = args.speaker_name

    if not args.noninteractive:
        # Enumerate devices if user didn't supply names
        if mic_name is None:
            _, dshow_audio = list_dshow_devices()
            mic_name = choose_from_list("Select Microphone (DirectShow):", dshow_audio)
        if speaker_name is None:
            if args.speaker_api == "wasapi":
                wasapi_devices = list_wasapi_devices()
                speaker_name = choose_from_list("Select Speaker (WASAPI, will use loopback):", wasapi_devices)
            else:
                _, dshow_audio = list_dshow_devices()
                speaker_name = choose_from_list("Select Speaker (DirectShow):", dshow_audio)

    if args.live_stdout:
        cmd = build_ffmpeg_live_audio_cmd(args, mic_name, speaker_name)
        log("\nRunning (live audio -> stdout, s16le 16kHz mono):")
        log(" ", " ".join(cmd))
        log("\nPress Ctrl+C to stop. Streaming raw audio to stdout.")
        try:
            # Ensure stdout is binary on Windows
            if msvcrt is not None:
                try:
                    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
                except Exception:
                    pass
            # Let ffmpeg write directly to our stdout (the pipe), avoiding Python writes
            subprocess.run(cmd)
        except KeyboardInterrupt:
            log("\nStopping...")
    else:
        cmd, out = build_ffmpeg_cmd(args, mic_name, speaker_name)

        log("\nRunning:")
        log(" ", " ".join(cmd))
        log("\nPress Ctrl+C to stop. Saving to:", out)
        try:
            # Stream ffmpeg output to console for progress
            subprocess.run(cmd)
        except KeyboardInterrupt:
            log("\nStopping...")


if __name__ == "__main__":
    main()
