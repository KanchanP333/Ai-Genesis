# Windows Screen + Mic + Speaker Recorder (Python + ffmpeg)

This small CLI wraps ffmpeg to record your screen, microphone, and speaker (system audio loopback) locally.

- Script: `recorder.py`
- Requires: `ffmpeg` on PATH (confirmed in this project)
- Output: `.mkv` by default (H.264 video + AAC audio)

## Quick start

1) Open a terminal in the project directory.
2) Run interactive device selection and record:

```bash
python recorder.py --fps 30 --preset veryfast
```

- You'll be prompted to pick a microphone and a speaker device.
- Press Ctrl+C to stop recording.

## Non-interactive usage

If you already know exact device names, pass them directly. You can list devices first (see below).

```bash
python recorder.py --fps 30 --preset veryfast \
  --mic-name "Microphone (YourMicDevice)" \
  --speaker-name "Speakers (YourOutputDevice)" \
  --speaker-api wasapi \
  --noninteractive \
  --output demo.mkv
```

Notes:
- `--speaker-api wasapi` records system audio using WASAPI loopback (recommended).
- If WASAPI names are tricky, you can try `--speaker-api dshow` and pass a DirectShow audio device name.

## Listing devices

- DirectShow devices (video + audio):

```bash
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

- WASAPI devices (for loopback names):

```bash
ffmpeg -hide_banner -list_devices true -f wasapi -i dummy
```

Copy names exactly as shown (including punctuation/casing).

## Common options

- `--fps 30` set capture framerate.
- `--size 1920:1080` scale output; omit for native resolution.
- `--crf 23` quality (lower is higher quality, larger file).
- `--preset veryfast` encoding speed/efficiency tradeoff.
- `--duration 10` limit to N seconds (auto-stop).
- `--use-gdigrab` use GDI capture instead of `ddagrab` (fallback for older systems/drivers).

## Examples

- Record native screen, pick devices interactively, stop manually:

```bash
python recorder.py
```

- Record 60 fps, scale to 1920x1080, stop after 30 seconds:

```bash
python recorder.py --fps 60 --size 1920:1080 --duration 30 --output short_test.mkv
```

- Non-interactive with known device names:

```bash
python recorder.py --noninteractive \
  --mic-name "Microphone (USB Audio Device)" \
  --speaker-name "Speakers (Realtek(R) Audio)" \
  --speaker-api wasapi \
  --output session.mkv
```

## Troubleshooting

- If capture fails immediately, try `--use-gdigrab`.
- If you get no system audio, re-choose `--speaker-api wasapi` and ensure you selected an output device, not a mic.
- Check Windows Privacy settings: allow microphone access to desktop apps.
- If encoding is too heavy, try `--preset ultrafast` or reduce `--fps`.

## Output format

- Video: H.264 (`libx264`) yuv420p
- Audio: AAC at `--audio-bitrate` kbps (default 192)
- Container: MKV (easy multi-track). You can change extension to `.mp4`, but MKV is typically safer for multi-audio streams.

## Where files go

- Saved to the current working directory unless you pass an absolute `--output` path.
