# CLIPFORGE AI

**Local AI clipping studio for long-form video.** Paste any video link — YouTube, a direct `.mp4`
file URL, Vimeo, X, Dailymotion or any other hosted stream — (or drop a file), and CLIPFORGE
downloads it, transcribes it locally, finds *every* moment worth publishing, writes vertical clips with
accurate word-level captions, and renders MP4s you can post.

Everything runs on your own machine. No uploads, no accounts, no cloud, no per-minute billing.

```
Video link ─► download ─► 16 kHz audio ─► transcription (faster-whisper) ─► speaker diarization
      ─► sentence & topic segmentation ─► candidate discovery ─► scoring ─► dedupe
      ─► boundary optimisation ─► reframing + captions + audio mix ─► FFmpeg render ─► 1080×1920 MP4
```

---

## Quick start (Windows)

```bat
:: 1. Python 3.10+ and Node 20+ are required. Then:
cd Clip-forge
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
pip install -r backend\requirements-ai.txt      :: faster-whisper + OpenCV (optional but recommended)

:: 2. Start the app (API + worker + web UI on http://127.0.0.1:8317)
python -m clipforge

:: Optional: build the UI so it is served from the same port as the API
cd frontend && npm install && npm run build && cd ..
python -m clipforge
```

The first run creates a `data/` folder next to the repo with your database, projects, caches, logs and
exports. Whisper models are downloaded once (about 40 MB for `tiny`, 480 MB for `small`) and then reused
offline.

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-ai.txt
python -m clipforge            # or: python -m clipforge serve --no-browser
```

### Developer mode (hot reload)

```bash
# terminal 1 - API + worker
python -m clipforge serve --no-browser
# terminal 2 - UI with hot reload
cd frontend && npm install && npm run dev     # http://localhost:3000
```

---

## Command line

| Command | What it does |
| --- | --- |
| `python -m clipforge` | Start the API, the render worker and the web UI; opens your browser. |
| `python -m clipforge serve --no-browser` | Same, without launching a browser. |
| `python -m clipforge worker` | Run only the job worker (e.g. on a GPU machine, several copies are fine). |
| `python -m clipforge analyze <youtube-url>` | Headless: analyse a video and print the discovered clips. |
| `python -m clipforge doctor` | Print the hardware, FFmpeg and AI-stack report. |

---

## How clip discovery works (no fake moments, ever)

1. **Transcript first.** Every word is stored with `start`, `end`, `confidence` and `speaker`
   (`TranscriptSegment.words_json`), so nothing downstream has to guess.
2. **Sentence + topic segmentation.** Words are grouped by punctuation, pauses and speaker turns; a
   TextTiling-style pass finds topic blocks.
3. **Candidate sweep.** Every start sentence × end sentence combination that fits the length window
   (default 35–75 s, target 60 s) is evaluated — a two-hour podcast produces thousands of evaluated
   windows, not a fixed "top 10".
4. **Fourteen quality signals** are computed per window (hook strength, emotional intensity, curiosity
   gap, standalone-ness, story arc, payoff, novelty, shareability, pace, filler ratio, confidence,
   repetition, abrupt edges, context dependency) and combined with the weights below.
5. **Penalties** demote clips that cannot stand alone: context dependency 30 %, abrupt start 14 %,
   low ASR confidence 18 %, abrupt end 12 %, repetition 14 %, filler 10 %, weak delivery 12 %.
6. **Boundaries are optimised** so each clip opens on a hook and closes on a payoff — edges never cut
   through a word.
7. **Deduplication** removes overlaps and near-duplicates (temporal IoU + TF-IDF cosine), then a final
   overlap guard makes sure two selected clips never sit on the same moment.
8. **Selection by mode**: `best` (only ≥ threshold), `balanced` (threshold plus a few near-misses),
   `max` (everything that clears the hard gate: ≥ 0.85 × threshold, context dependency < 0.75,
   confidence penalty < 0.6). There is **no artificial cap** — a rich 2-hour episode can yield 8, 17,
   31 or more clips; a thin one yields none, and the UI says why.

Every clip stores **WHY THIS CLIP**: the factors, the penalty breakdown, the quotable line, the hook,
the category and human-readable reasons derived from the actual transcript.

### Local LLM (optional)

If [Ollama](https://ollama.com) is running (`http://127.0.0.1:11434`) with a model such as `qwen3:4b`,
CLIPFORGE asks it for a second opinion on which passages matter. Those suggestions are capped at 45 %
influence and can never invent a timestamp that is not in the transcript. Without Ollama the pipeline
runs in *analytical mode* and produces the same kind of output — the LLM is a refinement, not a crutch.

### Vision AI (optional)

With OpenCV installed (`pip install opencv-python-headless`), CLIPFORGE samples frames to find faces and
the visual centre of interest, which drives crop tracking and smart reframing. Without it, the same
layouts are produced from centre-priority framing. Vision never blocks a render.

---

## What lands on disk

```
data/
  clipforge.db                     # SQLite: projects, transcript words, candidates, clips, jobs, settings
  projects/<slug>_<id>/
    source/                        # the downloaded/imported video
    audio/speech_16k.wav           # transcription input (cached)
    transcript/                    # sentences.json, transcript.txt, provided.srt (if you supplied one)
    analysis/                      # language.json, segmentation.json, candidates.json, clips.json
    clips/clip_001/                # plan.json, words.json, captions.ass, captions.srt
    renders/                       # final MP4s + previews/ + thumbnails/
    metadata/source.json
  exports/                         # copies of finished clips, named from your template
  logs/{app,worker,render,ai}.log  # structured logs, rotated
```

Deleting a project from the UI removes its folder (with confirmation). `Settings → Storage` can also
clean renders older than N days.

---

## Features

**Input** — any video link: YouTube (watch/short/embed/youtu.be), direct media file URLs (`.mp4`,
`.mov`, `.webm`, … served over HTTP) and other hosted pages (Vimeo, X, Dailymotion, …) via yt-dlp;
servers that refuse `HEAD` probes are handled with a ranged `GET` fallback; local uploads (`.mp4`,
`.mov`, `.mkv`, `.webm`, `.avi`, `.m4a`, `.mp3`, `.wav`); transcript files (`.json3`, `.srt`, `.vtt`,
timestamped `.txt`) attached to a project are used instead of ASR.

**Transcription** — faster-whisper with word-level timestamps (CUDA if available, otherwise CPU int8),
optional WhisperX alignment + diarization, and a numpy-based acoustic diarizer (pitch, MFCC-style
features, k-means) that always works without extra downloads. Language detection understands
Hindi / English / Hinglish / mixed-script audio and **never translates unless you switch translation
on** (*Translate to English* uses Whisper's translation task) — you can also force a language in Settings.

**Editing** — silence removal with natural pauses preserved (default: drop > 0.35 s), smart reframing
and speaker tracking, split-screen layouts (50/50, 60/40, 65/35, 70/30), speaker-full-frame,
B-roll and gameplay layouts, blurred/cinematic backgrounds, punch-in zooms on emphasis, and a batch
render queue (render all / selected / cancel / retry).

**Captions** — six presets (Minimal, Cinematic, Bold Creator, Karaoke, Highlight, Documentary) with
full customisation (font, size, colours, outline, shadow, background, animation, position, safe
margins), semantic line breaking, word-by-word highlighting, emphasis words, Devanagari-capable font
selection, ASS burn-in plus SRT export.

**Audio** — voice-first chain (high-pass, de-esser-ish EQ, compressor, gain), EBU R128 loudness
normalisation to −14 LUFS (−1.5 dBTP), ducking that drops music to 8–15 % and gameplay to 5–12 % of
voice level, optional music bed and normalisation of imported assets.

**Assets** — `assets/gameplay/{subway_surfer,temple_run,minecraft,parkour,racing,satisfying,simulation}`,
`assets/broll/*`, `assets/music/*`. Import from a folder path or upload from the UI. CLIPFORGE never
scrapes or downloads copyrighted material; you supply whatever you are licensed to use.

**Runtime** — hardware detection (CPU, RAM, disk, NVIDIA/AMD/Intel GPU, CUDA, VRAM) with automatic
CPU fallback and Whisper-model recommendations; SQLite-backed job queue that survives crashes;
SSE stream for live progress; structured logs; friendly errors with actionable hints and never a raw
stack trace in the UI.

### Export

H.264 (CRF 19) + AAC 192 kbps, 1080×1920 (or 1080×1080 / 1920×1080), 30 fps (60 fps optional),
`+faststart`, named `<project>_<index>_<title>.mp4`. Hardware encoders (`h264_nvenc`, `h264_qsv`,
`h264_amf`, `h264_videotoolbox`) are used when detected and configured.

---

## API

The UI talks to the same REST API you can script against (`/api/docs` for the live schema):

```
GET    /api/system/status | hardware | diagnostics | logs | errors | ffmpeg | capabilities
GET    /api/settings              PUT /api/settings        POST /api/settings/reset   GET /api/settings/schema
GET    /api/templates             POST /api/templates       POST /api/templates/{id}/apply
POST   /api/projects              POST /api/projects/upload
GET    /api/projects              GET  /api/projects/{id}[/status|/transcript|/candidates|/clips|/queue|/source|/disk]
POST   /api/analyze               POST /api/projects/{id}/analyze | /cancel | /render-all | /open-folder
POST   /api/projects/{id}/transcript          (attach .srt/.vtt/.json3/.txt)
DELETE /api/projects/{id}
GET    /api/clips/{id}            PATCH /api/clips/{id}      DELETE /api/clips/{id}
POST   /api/clips/{id}/render | /preview | /cancel | /regenerate | /duplicate
GET    /api/clips/{id}/captions | /captions/srt | /command | /preview | /file | /thumbnail | /subtitles | /assets
POST   /api/clips/render-all      POST /api/captions/preview   (caption style preview, no clip needed)
GET    /api/assets[/gameplay|/broll|/music|/folders]      POST /api/assets/upload | /import-path | /scan
GET    /api/jobs | /jobs/{id} | /jobs/queue | /workers    POST /api/jobs/{id}/cancel | /retry | /retry-failed | /purge
GET    /api/system/storage        POST /api/system/cleanup   ({"days", "renders", "dry_run"})
GET    /api/events                (SSE: job, clip and project events; heartbeat every 15 s)
```

`POST /api/projects/{id}/render-all` queues only the clips that still need a render unless you pass
`clip_ids` or `"force": true`. Creation options are stored per project only when you send them; anything
you leave out keeps following the global Settings.

Errors are always structured and safe to display:

```json
{ "error": { "code": "transcription_failed",
             "message": "CLIPFORGE could not transcribe this video.",
             "hint": "Install the AI extras: pip install -r requirements-ai.txt",
             "context": {} } }
```

---

## Configuration

`Settings` has nine sections (General, AI & analysis, Transcription, Video & clips, Captions, Gameplay
& layouts, Audio, Export, Storage) and persists to SQLite — no config files to edit. Environment
variables (`CLIPFORGE_DATA_DIR`, `CLIPFORGE_HOST`, `CLIPFORGE_PORT`, `CLIPFORGE_FFMPEG`,
`CLIPFORGE_LOG_LEVEL`, `CLIPFORGE_ALLOW_LOCAL_PATHS`, …) override the defaults; see `backend/.env.example`.

Locally hosted means locally safe: the server binds to `127.0.0.1` by default, media streaming is
restricted to the data directory, uploads are size-limited and sanitised, and FFmpeg always runs from
an argument vector (never a shell string).

**Hosting it on a server.** When `CLIPFORGE_HOST` is not a loopback address (e.g. `0.0.0.0` on Render or
Docker), features that touch the server's own filesystem are switched off: importing a file by path,
opening folders, and changing path settings (export folder, ffmpeg/ffprobe, cookies, model cache).
Uploads work as usual. Set `CLIPFORGE_ALLOW_LOCAL_PATHS=true` only on a machine you control. Whisper
models that would not fit in the container's memory limit are swapped for the largest one that does,
and the job log says so. Running the UI with `next start` proxies every request under `/api` to
`CLIPFORGE_API`.

**Storage.** *Settings → Storage → Clean up now* (or `POST /api/system/cleanup`) removes cached
downloads and unfinished projects older than `cleanup_days`, trims the download cache to `max_cache_gb`,
optionally deletes old renders (the clips can be re-rendered), and — with *Keep the source video* off —
deletes the sources of fully rendered projects. Expired caches are also trimmed on every start.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| “The speech recognition model could not be loaded” | First transcription needs to download the model. If huggingface.co is blocked, set `HF_ENDPOINT=https://hf-mirror.com` (or any mirror) and retry. If you already have captions, attach an SRT/VTT instead. |
| `ffprobe was not found next to ffmpeg` | Install a full FFmpeg build and point `Settings → Video` at it, or set `CLIPFORGE_FFMPEG`. The bundled binary works without ffprobe but probing is slower. |
| Renders are slow | Use a smaller Whisper model for analysis and a faster preset (`veryfast`) for rendering; enable `h264_nvenc` if you have an NVIDIA GPU. |
| A project produced no clips | That is a real result: nothing cleared the score threshold. Lower `min_score`, switch the mode to “Everything publishable”, or raise `clip length` limits. The candidate tab shows the scores that missed. |
| Ollama warnings | Optional. Install Ollama and pull a model (`ollama pull qwen3:4b`), or disable the LLM in Settings. |
| Something looks broken | `python -m clipforge doctor`, then `Settings → Diagnostics` (recent errors) and `logs/`. |

---

## Tests

```bash
cd backend
python -m pytest -q                          # unit + API tests (fast, offline)
python -m pytest -q -m "not e2e"             # skip the full pipeline test
```

The end-to-end test needs a real video. Generate one locally (about 5½ minutes, including a scripted
sponsor block the analyser must reject). Narration is synthesised with espeak/mespeak when one is
installed, and falls back to syllable-shaped tones when not - the audio still has real speech energy
and real pauses, and the script text drives the analysis through its caption file):

```bash
python tools/make_test_media.py --out /tmp/fix/source.mp4      # writes manifest.json next to it
cd backend && python -m pytest -q -m e2e
```

Set `CLIPFORGE_TEST_MEDIA=/path/to/video.mp4` to use your own footage instead.

---

## Architecture

```
frontend/  Next.js 15 · React 19 · Tailwind 4      (typed client, SSE-driven UI)
backend/
  clipforge/
    api/         FastAPI routes, schemas, range-request media streaming
    services/    project + clip business logic shared by API, worker and CLI
    jobs/        SQLite job queue, worker threads, job manager
    pipeline/    analyze (11 stages) · edit (plans) · render (FFmpeg orchestration)
    media/       process runner, ffmpeg, framing, timeline, captions, assets, yt-dlp
    ai/          transcribe, diarize, language, segment, features, candidates, scoring, llm, vision
    db.py        SQLAlchemy models + session helpers
```

Every heavy step reports real progress through the job’s stage log, and the pipeline is cancellable at
any point (`cancel_requested` on the job row, checked between stages and inside long loops).
