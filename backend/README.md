# ClipForge backend

The ClipForge backend turns long videos into short, captioned vertical clips. It is local-first: transcription, scoring, framing and rendering all run on your own machine, and no media leaves it unless you explicitly enable remote providers (`ALLOW_REMOTE_PROVIDERS=true`).

This directory holds the **API**, the **processing worker** and the **AI pipeline** described in [`ClipForge_TRD.md`](../ClipForge_TRD.md) and [`ClipForge_PRD.md`](../ClipForge_PRD.md).

```
upload ─▶ API (FastAPI) ─▶ jobs table (DB queue + leases) ─▶ worker ─▶ pipeline ─▶ clips + renders
              ▲  SSE progress ◀── job_events ◀───────────────────────────┘
```

## Quick start

### Docker (recommended)

```bash
cp backend/.env.example backend/.env       # set JWT_SECRET
docker compose up --build                  # API on http://localhost:8000, plus a worker
docker compose --profile ollama up --build # optional: local LLM for titles/hooks (LLM_PROVIDER=ollama)
```

The interactive API docs are served at <http://localhost:8000/api/v1/docs>.

### Bare metal

You need Python 3.11+ (3.12 recommended), plus `ffmpeg` and `ffprobe` built with libass.

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ai,dev]"           # "ai" installs faster-whisper
cp .env.example .env
python -m clipforge.api              # terminal 1: applies migrations, then serves on :8000
python -m clipforge.worker           # terminal 2: processes jobs
```

To try the pipeline without downloading Whisper weights, set `TRANSCRIPTION_PROVIDER=fake`. It produces a deterministic test transcript.

## Layout

| Path | What lives there |
|---|---|
| `clipforge/core` | Settings (env vars, TRD §37), error catalogue, job state machine, versioning, structured JSON logging |
| `clipforge/db` | SQLAlchemy models for all TRD §26 tables, plus idempotency keys, worker heartbeats and analytics; session handling (SQLite WAL or Postgres) |
| `clipforge/api` | FastAPI app factory, dependencies (auth/CSRF/rate limits), schemas, routes, SSE |
| `clipforge/services` | Business logic: auth, uploads, jobs, clips, quotas/entitlements, billing stub, cleanup, audit, analytics, system metrics |
| `clipforge/queue` | DB-backed queue with lease locking and crash recovery; optional Redis wake-ups |
| `clipforge/storage` | `StorageProvider`: local filesystem and S3, with UUID keys and signed URLs |
| `clipforge/security` | argon2 hashing, opaque session tokens, magic-byte checks, path safety, rate limiter |
| `clipforge/media` | FFmpeg/ffprobe wrappers (argument arrays, timeouts, cancellation), framing, ASS captions, render args |
| `clipforge/scoring` | Pure functions: sentence segmentation, topic boundaries, candidate windows, features, weighted scoring, boundary optimisation |
| `clipforge/worker` | Worker loop, pipeline stages, model manager (CUDA/VRAM detection, fallback chain), resource guard, providers (Whisper, LLM, vision, embeddings) |
| `config/*.yaml` | Scoring weights, plans/quotas, render profiles and caption presets. All config, no code. |
| `migrations/` | Alembic migrations, applied automatically on start-up (`AUTO_MIGRATE`) |

## Pipeline

Each stage implements `validate → execute → persist → validate_output → cleanup` and records versioned, checksummed artifacts in `work/{job_id}/manifest.json`, which is mirrored to `jobs.manifest`. A job that is retried or recovered after a crash skips every stage whose artifacts are still valid.

| Stage | Job status | Output |
|---|---|---|
| `validate` | VALIDATING | ffprobe metadata, limit checks |
| `ingest` | INGESTING | checksum-verified source artifact |
| `audio_extract` | EXTRACTING_AUDIO | 16 kHz mono WAV, per-second loudness |
| `transcribe` | TRANSCRIBING | word-timestamped transcript (faster-whisper). Falls back `large-v3 → medium → small → base` on GPU, then on CPU, before failing with `MODEL_UNAVAILABLE` |
| `segment` | SEGMENTING | sentences and topic boundaries |
| `candidate_generation`, `candidate_scoring` | CANDIDATE_SCORING | 10–90 s windows on sentence boundaries, scored with weights from `config/scoring.yaml`. Raw features are stored. Overlap suppression; optional LLM re-ranking and metadata (falls back to heuristics) |
| `boundary_optimization` | BOUNDARY_OPTIMIZATION | natural start/end points (pauses, no cut words); clip rows created |
| `visual_analysis` | FRAME_ANALYSIS | sampled-frame face detection (OpenCV); `auto` picks face tracking, centre crop or fit with blurred background |
| `caption_layout` | CAPTION_GENERATION | caption timelines in the chosen preset (Clean, Bold, Creator, Minimal, High contrast) |
| `render`, `qa` | RENDERING | H.264/AAC MP4 in 9:16, 16:9 or 1:1 with burned-in captions, plus a thumbnail and `.ass` subtitles. ffprobe QA checks codecs, dimensions, duration and decodability; a failed QA is retried once |
| `finalize` | FINALIZING → COMPLETED | usage recorded, retention timers started, intermediates deleted |

### Failure handling

- Errors use the TRD §30 catalogue: `{code, message, retryable, diagnostic_id}`. Stack traces never reach users.
- Retryable failures are retried up to 3 times: immediately after the first failure, then with exponential backoff.
- If a worker loses its lease, it stops without touching job state, and another worker resumes the job from the manifest.
- The resource guard leaves jobs `QUEUED` while disk space, RAM or the work-dir quota is insufficient.

## API overview (`/api/v1`)

| Area | Endpoints |
|---|---|
| Auth | `POST auth/register`, `auth/login`, `auth/logout`, `auth/refresh`, `auth/password`, `GET auth/me` |
| Uploads | `POST uploads` (multipart or raw stream), `POST uploads/init` → `PUT uploads/{id}/chunks?offset=` → `POST uploads/{id}/complete`; `GET/DELETE uploads/{id}`, `GET uploads` |
| Jobs | `POST jobs`, `GET jobs`, `GET/DELETE jobs/{id}`, `POST jobs/{id}/cancel`, `POST jobs/{id}/retry`, `GET jobs/{id}/clips`, `GET jobs/{id}/events`, `GET jobs/{id}/transcript`, `GET jobs/{id}/stream` (SSE) |
| Clips | `GET/PATCH/DELETE clips/{id}`, `POST clips/{id}/render`, `GET clips/{id}/renders`, `GET clips/{id}/download`, `GET clips/{id}/thumbnail`, `GET clips/{id}/subtitles` |
| Me | `GET me`, `GET me/usage`, `GET/PATCH me/settings` |
| Admin | `GET admin/jobs`, `GET admin/jobs/{id}`, `POST admin/jobs/{id}/retry`, `POST admin/jobs/{id}/cancel`, `GET admin/users`, `PATCH admin/users/{id}`, `GET admin/system`, `POST admin/cleanup` |
| Ops | `GET health`, `GET health/ready`, `GET health/worker`, `GET config`, `GET /metrics` (Prometheus) |

Implementation notes:

- **Auth.** Clients authenticate with an HttpOnly session cookie, which requires the double-submit CSRF header `X-CSRF-Token` on mutations, or with an `Authorization: Bearer <token>` header. Tokens are opaque and stored as SHA-256 hashes; passwords are hashed with argon2id.
- **Idempotency.** `POST uploads`, `uploads/init` and `jobs` accept an `Idempotency-Key` header.
- **Downloads.** Download links are HMAC-signed and expiring, and support `Range` requests. With S3 storage they redirect to a pre-signed URL.
- **SSE.** The stream emits `job.progress`, `job.event` and `job.finished` events and resumes from `Last-Event-ID`.
- **Clip edits.** Editing `start`, `end`, `aspect_ratio`, `caption_preset`, `captions_enabled` or `framing_mode` queues a re-render. Only framing, captions, render and QA re-run; the transcript is reused.
- **Local mode.** In local mode the first registered user becomes admin. With `AUTH_ENABLED=false`, a single implicit user is used; cloud mode always requires auth.

## Configuration

All settings are environment variables; see [`.env.example`](.env.example) for the full annotated list. The main ones:

| Setting | Effect |
|---|---|
| `DEPLOYMENT_MODE` | `local`: private, unlimited plan by default, auth optional. `cloud`: auth enforced, quotas from `config/plans.yaml` |
| `DATABASE_URL` | Empty means SQLite in `STORAGE_ROOT`; use Postgres (`postgresql+psycopg://…`) in the cloud |
| `STORAGE_BACKEND` | `local` or `s3` |
| `WHISPER_MODEL` | `auto` chooses a model based on available hardware |
| `LLM_PROVIDER` | `heuristic` (default, no model needed), `ollama` or `transformers`. `openai_compatible` is only used when `ALLOW_REMOTE_PROVIDERS=true` |
| `WORKER_CONCURRENCY`, `MAX_CONCURRENT_FFMPEG`, `MAX_CONCURRENT_TRANSCRIPTIONS` | Throughput limits |
| `RETENTION_HOURS`, `OUTPUT_RETENTION_HOURS` | Privacy-first retention: by default sources are deleted 24 h after completion and exports after 7 days |

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q                          # unit and integration tests (integration needs ffmpeg)
pytest tests/unit -q               # fast, no ffmpeg needed
alembic revision --autogenerate -m "describe change"   # after model changes
alembic check                      # CI verifies that migrations match the models
```

The integration tests generate a synthetic video with lavfi and run the complete flow with the `fake` transcription provider: upload → job → worker → clips → signed download → edit → re-render, plus crash-resume, cancel/retry and chunked uploads.
