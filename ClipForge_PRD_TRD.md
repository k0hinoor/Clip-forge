# ClipForge AI --- PRD + TRD

# ClipForge AI --- Product Requirements Document (PRD)

**Document version:** 1.0\
**Status:** Build-ready baseline\
**Product type:** AI-assisted video clipping web application\
**Primary objective:** Automatically turn long-form video into
high-quality short-form clips with captions, vertical reframing,
metadata, and downloadable exports.\
**Cost objective:** Zero mandatory infrastructure/API spend for the
initial deployment and friend-only usage; local-first processing is the
default.\
**Business objective:** Start free, validate demand, then monetize
without rebuilding the core system.

------------------------------------------------------------------------

## 1. Product Vision

ClipForge AI is a local-first, automation-heavy clipping platform for
creators who have long videos but do not want to manually watch hours of
footage to find short-form content.

A user uploads a video. The system automatically:

1.  validates and ingests the media;
2.  extracts technical metadata;
3.  transcribes speech;
4.  segments the transcript;
5.  detects candidate moments;
6.  scores candidates for short-form potential;
7.  expands each candidate to a coherent clip boundary;
8.  optionally detects the active speaker/subject;
9.  reframes the video to a selected aspect ratio;
10. generates captions;
11. renders the final clip;
12. generates a title/hook/description;
13. presents previews;
14. allows download/export;
15. deletes temporary source and render data according to retention
    policy.

The product should feel like: **upload → wait → receive usable clips**,
not a video editor that requires manual work.

------------------------------------------------------------------------

## 2. Problem Statement

Creators, students, streamers, podcasters, educators, and small media
teams often have long recordings containing multiple potential
short-form moments. Manual clipping is slow because it requires:

-   watching the source;
-   finding strong moments;
-   identifying exact start/end timestamps;
-   cutting footage;
-   reframing for vertical platforms;
-   creating captions;
-   choosing hooks/titles;
-   exporting;
-   repeating the process for multiple clips.

Existing AI clipping products may be expensive, quota-limited,
privacy-sensitive, or dependent on paid APIs.

ClipForge should solve the workflow while keeping processing local
whenever possible.

------------------------------------------------------------------------

## 3. Target Users

### Primary

-   Individual creators
-   Streamers
-   Podcasters
-   Students making project/demo content
-   Friends sharing gaming/commentary content
-   Small social-media teams

### Secondary

-   Coaches/educators
-   Agencies
-   Small businesses
-   Event organizers
-   Community managers

### Not a target for v1

-   Professional Hollywood/post-production workflows
-   Enterprise-scale media asset management
-   Full nonlinear video editing
-   Real-time livestream clipping

------------------------------------------------------------------------

## 4. Core Product Promise

**Input:** One long-form video.

**Output:** A ranked collection of short-form-ready clips with:

-   start/end timestamps;
-   reason for selection;
-   transcript;
-   captions;
-   vertical/horizontal output;
-   title;
-   hook;
-   optional description;
-   downloadable MP4.

The system must never claim that an AI-generated clip is objectively
"viral." It should describe it as a candidate scored against measurable
content signals.

------------------------------------------------------------------------

# 5. Product Principles

1.  **Local-first:** expensive compute should run on the owner's machine
    whenever practical.
2.  **Zero mandatory API dependency:** the core clipping pipeline must
    function without paid AI APIs.
3.  **Asynchronous by default:** uploads create jobs; processing must
    never block the web request.
4.  **Resumable:** a failed stage must restart from the failed stage
    rather than from zero.
5.  **Observable:** every job has a state, progress, logs, timestamps,
    and error code.
6.  **Privacy-first:** source files are temporary by default.
7.  **Provider abstraction:** AI models must be replaceable without
    rewriting business logic.
8.  **Graceful degradation:** if one AI capability fails, useful output
    should still be produced where possible.
9.  **Monetization without core lock-in:** free users must receive
    useful output.
10. **No silent resource abuse:** CPU/GPU/disk usage must be bounded.

------------------------------------------------------------------------

# 6. MVP Scope

## 6.1 Upload

Supported initial inputs:

-   MP4
-   MOV
-   MKV
-   WebM
-   M4V

Recommended v1 constraints:

-   maximum file size: configurable, default 2 GB;
-   maximum duration: configurable, default 3 hours;
-   one active processing job per free user;
-   configurable concurrency at server level.

Upload requirements:

-   drag and drop;
-   file picker;
-   upload progress;
-   cancel upload;
-   checksum/hash;
-   duplicate detection;
-   resumable/chunked upload where supported.

------------------------------------------------------------------------

## 6.2 Automatic Transcription

The system should use a local speech-to-text model such as
faster-whisper.

Required transcript data:

-   full text;
-   segment start/end;
-   word timestamps when available;
-   language;
-   confidence metadata when available.

Transcript must be persisted separately from rendered video so
downstream stages can reuse it.

------------------------------------------------------------------------

## 6.3 Candidate Detection

The system must identify candidate moments using multiple signals.

### Content signals

-   emotional intensity;
-   novelty;
-   surprising statement;
-   strong opinion;
-   useful information;
-   question/answer completeness;
-   story arc;
-   punchline;
-   conflict/debate;
-   educational value;
-   explicit conclusion;
-   call-to-action;
-   conversational energy.

### Structural signals

-   sentence boundaries;
-   speaker turns;
-   topic boundaries;
-   silence;
-   unusually long pauses;
-   abrupt topic changes;
-   transcript density.

### Audio signals

-   loudness;
-   speech activity;
-   silence ratio;
-   energy variation.

### Visual signals

Optional in v1, stronger in v2:

-   scene changes;
-   face presence;
-   active speaker;
-   visual motion;
-   screen/gameplay changes.

------------------------------------------------------------------------

# 7. Clip Scoring

Each candidate receives a score from 0--100.

The score is not a prediction of virality.

Example feature groups:

  Feature                          Example Weight
  ------------------------------ ----------------
  Hook strength                                20
  Self-contained meaning                       20
  Emotional/interesting signal                 15
  Information value                            15
  Narrative completeness                       10
  Audio quality                                 5
  Visual quality                                5
  Caption suitability                           5
  Length suitability                            5

Weights must be configurable.

The UI should expose **why** a clip was selected.

Example:

> Strong opening hook · complete thought · high information density ·
> clear ending

------------------------------------------------------------------------

# 8. Clip Boundary Generation

The system must not blindly cut at arbitrary seconds.

Boundary logic should prefer:

-   sentence boundaries;
-   speaker-turn boundaries;
-   pauses;
-   semantic completion;
-   natural openings;
-   natural endings.

Default target duration:

-   20--60 seconds.

Configurable maximum:

-   90 seconds.

Configurable minimum:

-   10 seconds.

The algorithm may add padding before/after a detected moment to preserve
context.

------------------------------------------------------------------------

# 9. Caption Generation

Captions are generated from the same transcript used for clip detection.

Requirements:

-   word/phrase timing;
-   line wrapping;
-   safe-area placement;
-   configurable font;
-   configurable size;
-   configurable maximum characters per line;
-   optional emphasized words;
-   optional speaker colors;
-   subtitle burn-in for final render.

Caption presets:

1.  Clean
2.  Bold
3.  Creator
4.  Minimal
5.  High contrast

Caption rendering must be deterministic from a saved configuration.

------------------------------------------------------------------------

# 10. Smart Reframing

Initial aspect ratios:

-   9:16
-   16:9
-   1:1

Default short-form output: **9:16**.

For talking-head content:

-   detect face;
-   maintain face inside safe area;
-   smoothly move crop when subject changes.

For screen/gameplay content:

-   avoid destructive cropping;
-   optionally use letterboxing or smart scaling.

For multiple faces:

-   select active speaker using audio + visual signals when available;
-   otherwise use center crop.

A user must be able to override auto framing.

------------------------------------------------------------------------

# 11. Output Generation

Each approved candidate should produce:

-   MP4;
-   H.264 video;
-   AAC audio;
-   selected aspect ratio;
-   burned-in captions if enabled.

Output filenames should be deterministic:

`clip_<job_id>_<rank>_<short_slug>.mp4`

The system should also produce:

-   thumbnail;
-   transcript excerpt;
-   title;
-   hook;
-   description;
-   hashtags/keywords as optional metadata.

------------------------------------------------------------------------

# 12. User Flow

## Flow A --- Automatic clipping

1.  Open website.
2.  Upload video.
3.  Select:
    -   target platform;
    -   aspect ratio;
    -   desired number of clips;
    -   caption preset.
4.  Start processing.
5.  Show job progress.
6.  Display candidate clips.
7.  User previews clips.
8.  User downloads selected clips.
9.  System cleans temporary files.

## Flow B --- Manual correction

User opens a candidate and changes:

-   start time;
-   end time;
-   caption style;
-   aspect ratio;
-   title.

System rerenders only the affected stage.

## Flow C --- Failure

If a job fails:

-   show human-readable error;
-   preserve completed intermediate stages;
-   allow retry;
-   record technical error internally;
-   never expose secrets or stack traces.

------------------------------------------------------------------------

# 13. Job States

Required states:

`QUEUED`

`VALIDATING`

`INGESTING`

`EXTRACTING_AUDIO`

`TRANSCRIBING`

`SEGMENTING`

`CANDIDATE_SCORING`

`BOUNDARY_OPTIMIZATION`

`FRAME_ANALYSIS`

`CAPTION_GENERATION`

`RENDERING`

`FINALIZING`

`COMPLETED`

`FAILED`

`CANCELLED`

`EXPIRED`

State transitions must be validated by the backend.

------------------------------------------------------------------------

# 14. Automation Requirements

The product should be fully automated after upload.

### Automated pipeline

Upload → validate → metadata → audio extraction → transcription →
semantic segmentation → candidate generation → candidate scoring →
boundary optimization → visual analysis → caption generation → rendering
→ QA validation → thumbnail generation → result publication → retention
timer → cleanup

### Automated QA

Before marking a clip complete:

-   verify file exists;
-   verify duration;
-   verify video stream;
-   verify audio stream;
-   verify codec;
-   verify captions when enabled;
-   verify output dimensions;
-   verify file is playable;
-   verify no zero-byte/corrupt output.

Failed QA automatically triggers one retry.

------------------------------------------------------------------------

# 15. Free Tier

The initial product can be completely free to the creator when
self-hosted.

Suggested public free tier:

-   limited processing minutes/day;
-   limited concurrent jobs;
-   max upload duration;
-   lower priority queue;
-   local processing pool.

These limits must be configuration-driven, not hard-coded.

------------------------------------------------------------------------

# 16. Monetization

The monetization system must be designed but not required for
local/private deployment.

## Revenue options

### A. Freemium

Free: - limited monthly processing minutes; - watermark optional; -
limited export quality; - queue priority: low.

Paid: - higher limits; - faster queue; - batch processing; - premium
caption presets; - cloud processing credits.

### B. Credit model

Charge by processing minute.

Example internal unit:

`1 processing_credit = 1 minute of source video`

Pricing should be configurable.

### C. Affiliate revenue

Potential affiliate placements:

-   creator tools;
-   microphones;
-   cameras;
-   editing hardware;
-   hosting;
-   creator software.

Affiliate modules must be clearly labeled.

### D. Sponsorship

Optional sponsor placement on the dashboard.

### E. Ads

Ads may be added to free-tier UI later.

Do not place ads over video export controls or interfere with creator
workflow.

------------------------------------------------------------------------

# 17. Zero-Cost Business Architecture

The product must support three deployment modes.

### Mode 1 --- Personal

Everything runs on one PC.

`Browser → Local API → Local Worker → Local Models`

Expected mandatory cost: \$0.

### Mode 2 --- Small private service

Frontend can be hosted on a free-tier static/web host while processing
remains on the owner's PC.

`Browser → Free Web Host → Secure Tunnel → Local Backend`

### Mode 3 --- Public SaaS

Cloud workers can be added later.

The architecture must not require a rewrite.

The same job abstraction must support:

-   local worker;
-   remote worker;
-   GPU worker;
-   CPU worker.

------------------------------------------------------------------------

# 18. Admin Automation

Admin dashboard must display:

-   active jobs;
-   queued jobs;
-   failed jobs;
-   processing time;
-   GPU/CPU usage;
-   disk usage;
-   daily minutes processed;
-   users;
-   retention/cleanup status;
-   model failures;
-   worker health;
-   revenue metrics when monetization is enabled.

Automated alerts:

-   disk above threshold;
-   worker offline;
-   repeated model failures;
-   queue too large;
-   repeated failed renders;
-   abnormal processing time.

------------------------------------------------------------------------

# 19. Privacy

Default policy:

-   source videos are temporary;
-   processing artifacts have retention TTL;
-   completed exports expire after configurable period;
-   users can delete jobs immediately;
-   logs must not contain source media;
-   logs must not contain authentication secrets;
-   analytics should use aggregated metadata.

Default recommended retention:

-   source video: delete after successful processing + 24 hours;
-   intermediate files: delete after pipeline completion;
-   final export: retain for 7 days;
-   job metadata: retain longer for analytics unless user deletes it.

------------------------------------------------------------------------

# 20. Abuse Protection

Required before public deployment:

-   rate limiting;
-   upload size limits;
-   duration limits;
-   per-user job quotas;
-   MIME validation;
-   extension validation;
-   file signature validation;
-   path traversal protection;
-   authentication;
-   authorization;
-   cleanup of abandoned uploads;
-   resource quotas;
-   queue backpressure.

Do not execute arbitrary user-supplied shell commands.

FFmpeg arguments must be generated from validated internal parameters.

------------------------------------------------------------------------

# 21. Success Metrics

### Product metrics

-   upload completion rate;
-   processing completion rate;
-   median processing time per source minute;
-   clips generated per video;
-   clip download rate;
-   rerender rate;
-   manual boundary edit rate;
-   percentage of clips accepted/downloaded;
-   job failure rate.

### Business metrics

-   free-to-paid conversion;
-   processing minutes per user;
-   paid processing minutes;
-   revenue per active user;
-   infrastructure cost per processed hour;
-   gross margin.

### Quality metrics

-   transcript accuracy;
-   clip acceptance rate;
-   caption correction rate;
-   false-positive candidate rate;
-   render failure rate.

------------------------------------------------------------------------

# 22. Non-Functional Requirements

-   responsive UI;
-   keyboard accessible;
-   clear progress;
-   no blocking uploads;
-   resumable processing;
-   deterministic rendering;
-   structured logs;
-   API versioning;
-   configuration through environment variables;
-   no secrets committed to Git;
-   database migrations;
-   automated cleanup;
-   testable pipeline stages.

------------------------------------------------------------------------

# 23. Future Features

-   URL import where legally/technically permitted;
-   YouTube/Twitch upload integrations;
-   batch processing;
-   automatic posting;
-   multilingual transcription;
-   translation;
-   voice isolation;
-   filler-word removal;
-   silence removal;
-   AI B-roll suggestions;
-   custom brand kits;
-   team workspaces;
-   scheduled publishing;
-   clip performance feedback loop;
-   personalized scoring models.

------------------------------------------------------------------------

# 24. Explicit Non-Goals for v1

Do not build:

-   full video editor;
-   livestream infrastructure;
-   proprietary foundation model;
-   custom GPU server;
-   complex social network;
-   automatic posting to every platform;
-   training a custom ML model before collecting evaluation data.

------------------------------------------------------------------------

# 25. Acceptance Criteria

The MVP is complete when a user can:

1.  upload a supported video;
2.  see upload progress;
3.  start an asynchronous job;
4.  see real-time job status;
5.  receive a transcript;
6.  receive multiple ranked clip candidates;
7.  preview candidates;
8.  generate vertical clips;
9.  burn captions;
10. download completed clips;
11. retry failed jobs;
12. delete a job;
13. have temporary files automatically cleaned;
14. use the system without a paid AI API;
15. run the core pipeline locally.

------------------------------------------------------------------------

# ClipForge AI --- Technical Requirements & Design Document (TRD)

**Document version:** 1.0\
**Status:** Build-ready baseline\
**Architecture:** Local-first, modular, asynchronous, provider-agnostic\
**Primary implementation target:** Windows/Linux single-machine
deployment first; public SaaS later\
**Core principle:** Build once so the worker can move from local CPU/GPU
to cloud workers without changing product logic.

------------------------------------------------------------------------

# 1. Architecture Overview

``` text
                         ┌───────────────────────────┐
                         │        Web Browser        │
                         │ Next.js / React Frontend  │
                         └─────────────┬─────────────┘
                                       │ HTTPS
                                       ▼
                         ┌───────────────────────────┐
                         │        API Gateway        │
                         │       FastAPI Backend     │
                         └─────────────┬─────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
                 ▼                     ▼                     ▼
        ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
        │ Authentication │    │ PostgreSQL/    │    │ Object/File    │
        │ & Authorization│    │ SQLite         │    │ Storage        │
        └────────────────┘    └────────────────┘    └────────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Job Queue / State│
                              └────────┬─────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                ┌─────────────────┐         ┌─────────────────┐
                │ Local Worker    │         │ Future Remote   │
                │ CPU/GPU         │         │ Worker          │
                └────────┬────────┘         └────────┬────────┘
                         │                           │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │ Processing Pipeline       │
                         │ FFmpeg / Whisper / AI     │
                         └───────────────────────────┘
```

------------------------------------------------------------------------

# 2. Technology Stack

## Frontend

Recommended:

-   Next.js
-   React
-   TypeScript
-   Tailwind CSS
-   native fetch or lightweight API client

Frontend responsibilities:

-   authentication UI;
-   upload;
-   job status;
-   clip gallery;
-   preview;
-   settings;
-   admin UI;
-   billing UI when enabled.

The frontend must never contain secret API keys.

------------------------------------------------------------------------

# 3. Backend

Recommended:

-   Python 3.12+
-   FastAPI
-   Pydantic
-   SQLAlchemy
-   Alembic
-   Uvicorn
-   structured logging

Backend responsibilities:

-   authentication;
-   authorization;
-   upload orchestration;
-   job creation;
-   job state management;
-   queue management;
-   worker coordination;
-   model provider selection;
-   metadata;
-   clip APIs;
-   cleanup;
-   quotas;
-   analytics;
-   admin operations.

------------------------------------------------------------------------

# 4. Database Strategy

## Local

SQLite is acceptable for single-machine/private use.

Use WAL mode.

## Public

PostgreSQL should be the production database.

The application must use SQLAlchemy so the database layer is
replaceable.

------------------------------------------------------------------------

# 5. Redis / Queue Strategy

For local MVP, avoid unnecessary infrastructure.

Recommended local strategy:

-   database-backed job queue;
-   one worker process;
-   OS/process locking.

Optional:

-   Redis + Celery/RQ/Arq.

Production architecture should support Redis without requiring changes
to pipeline business logic.

------------------------------------------------------------------------

# 6. Storage Strategy

Use an abstraction:

``` text
StorageProvider
├── LocalFilesystemStorage
├── S3CompatibleStorage
└── FutureCloudStorage
```

Methods:

-   put();
-   get();
-   delete();
-   exists();
-   stat();
-   signed_url() where supported.

Local directory structure:

``` text
data/
  uploads/
  work/
  outputs/
  thumbnails/
  transcripts/
  logs/
```

Never use user-controlled filenames directly as filesystem paths.

Generate UUID-based internal paths.

------------------------------------------------------------------------

# 7. Job Architecture

Every processing request becomes a Job.

Example:

``` json
{
  "job_id": "uuid",
  "user_id": "uuid",
  "source_asset_id": "uuid",
  "status": "TRANSCRIBING",
  "progress": 42,
  "current_stage": "transcription",
  "created_at": "...",
  "started_at": "...",
  "completed_at": null,
  "attempt": 1,
  "error_code": null
}
```

The worker must be idempotent.

If the same stage is retried, it should reuse valid artifacts instead of
recreating everything.

------------------------------------------------------------------------

# 8. Pipeline Contract

Every stage must implement the same conceptual interface:

``` text
Stage
  validate(input)
  execute(input)
  persist(output)
  validate_output(output)
  cleanup()
```

Pipeline:

``` text
INGEST
↓
MEDIA_PROBE
↓
AUDIO_EXTRACT
↓
TRANSCRIBE
↓
SEMANTIC_SEGMENT
↓
CANDIDATE_GENERATION
↓
CANDIDATE_SCORING
↓
BOUNDARY_OPTIMIZATION
↓
VISUAL_ANALYSIS
↓
CAPTION_LAYOUT
↓
RENDER
↓
QA
↓
FINALIZE
```

------------------------------------------------------------------------

# 9. Stage 1 --- Media Ingestion

Use FFmpeg/ffprobe.

Validate:

-   MIME;
-   file signature;
-   codec;
-   duration;
-   dimensions;
-   audio streams;
-   video streams;
-   file size.

Reject:

-   corrupted media;
-   unsupported streams;
-   excessive duration;
-   excessive resolution;
-   malicious paths;
-   malformed containers.

Store immutable media metadata.

------------------------------------------------------------------------

# 10. Stage 2 --- Audio Extraction

Extract a normalized audio representation.

Recommended:

-   mono/stereo as appropriate;
-   16 kHz or model-compatible sampling;
-   WAV/PCM for transcription where required.

Do not overwrite original source.

Artifact:

``` text
audio.wav
```

------------------------------------------------------------------------

# 11. Stage 3 --- Transcription

Default provider:

**faster-whisper**

Model selection must be configurable:

``` text
tiny
base
small
medium
large-v3
```

Default local model depends on available VRAM/RAM.

The user should not need to understand model configuration.

Output schema:

``` json
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 12.42,
      "end": 18.90,
      "text": "...",
      "words": [
        {
          "start": 12.42,
          "end": 12.81,
          "word": "..."
        }
      ]
    }
  ]
}
```

------------------------------------------------------------------------

# 12. Stage 4 --- Semantic Segmentation

Group transcript segments into coherent topics.

Use:

-   sentence boundaries;
-   pauses;
-   embeddings where available;
-   topic transitions;
-   speaker changes.

A segment should have:

``` text
segment_id
start
end
text
topic_id
speaker_id
```

Embeddings are optional and must not become a mandatory cloud
dependency.

------------------------------------------------------------------------

# 13. Stage 5 --- Candidate Generation

Generate overlapping candidate windows.

Constraints:

``` text
MIN_CLIP_SECONDS = 10
TARGET_MIN_SECONDS = 20
TARGET_MAX_SECONDS = 60
MAX_CLIP_SECONDS = 90
```

Candidate generation must consider semantic boundaries.

Avoid:

-   starting mid-sentence;
-   ending mid-sentence;
-   isolated contextless statements.

------------------------------------------------------------------------

# 14. Stage 6 --- Candidate Scoring

Implement a deterministic scoring service.

Example:

``` text
score =
  hook * 0.20 +
  completeness * 0.20 +
  emotional_interest * 0.15 +
  information_value * 0.15 +
  narrative_completeness * 0.10 +
  audio_quality * 0.05 +
  visual_quality * 0.05 +
  caption_suitability * 0.05 +
  length_fit * 0.05
```

All weights belong in configuration.

Store raw features and final score.

This enables future model training without redesigning the database.

------------------------------------------------------------------------

# 15. LLM Integration

The LLM must be behind an abstraction:

``` text
LLMProvider
├── LocalOllamaProvider
├── LocalTransformersProvider
└── OptionalRemoteProvider
```

The application must work when all remote providers are disabled.

Use the local LLM for:

-   candidate reasoning;
-   title generation;
-   hook generation;
-   descriptions;
-   semantic classification.

Do not send full source video to an LLM.

Send transcript excerpts and structured metadata.

------------------------------------------------------------------------

# 16. Local Model Runtime

Recommended architecture:

``` text
ModelManager
  ├── model registry
  ├── health check
  ├── warm-up
  ├── memory check
  ├── provider selection
  └── fallback
```

For a local NVIDIA GPU:

-   detect CUDA;
-   detect VRAM;
-   select appropriate model;
-   use GPU when available;
-   fall back to CPU when necessary.

The application must not assume a specific GPU.

------------------------------------------------------------------------

# 17. Visual Analysis

Optional MVP stage.

Possible components:

-   OpenCV;
-   FFmpeg frame extraction;
-   face detection;
-   scene detection.

Store:

``` text
frame_timestamp
face_boxes
scene_id
motion_score
```

Do not continuously process every frame at maximum resolution.

Use sampled frames.

------------------------------------------------------------------------

# 18. Smart Cropping

For 9:16:

1.  determine source dimensions;
2.  determine target crop;
3.  detect faces/subjects;
4.  generate crop trajectory;
5.  smooth trajectory;
6.  render using FFmpeg.

Crop parameters must be stored so a render can be reproduced.

------------------------------------------------------------------------

# 19. Caption Engine

Input:

-   transcript words;
-   clip start/end;
-   style configuration.

Output:

-   caption timeline;
-   ASS subtitle file or equivalent;
-   final burned-in captions.

Caption schema:

``` json
{
  "start": 1.2,
  "end": 2.8,
  "text": "This is the important part.",
  "emphasis": ["important"]
}
```

------------------------------------------------------------------------

# 20. Rendering Engine

Use FFmpeg.

Never construct shell commands using raw user input.

Prefer subprocess argument arrays.

Example conceptual command:

``` text
ffmpeg
-i INPUT
-filter_complex FILTER_GRAPH
-map ...
-c:v libx264
-c:a aac
OUTPUT
```

Security requirements:

-   no `shell=True`;
-   validated paths;
-   validated numeric parameters;
-   whitelist codecs;
-   whitelist filters;
-   timeout;
-   process kill on cancellation.

------------------------------------------------------------------------

# 21. Render Presets

Presets should be database/config driven.

Example:

``` yaml
vertical:
  width: 1080
  height: 1920
  fps: source
  video_codec: h264
  audio_codec: aac
```

Do not hard-code platform-specific settings into frontend code.

------------------------------------------------------------------------

# 22. QA Stage

Use ffprobe after rendering.

Validate:

-   output exists;
-   file size \> minimum;
-   duration \> minimum;
-   video stream exists;
-   audio stream exists;
-   expected dimensions;
-   supported codec;
-   readable container.

Optional:

-   generate a low-resolution preview;
-   decode first/middle/last frames.

------------------------------------------------------------------------

# 23. REST API

Base:

`/api/v1`

Endpoints:

``` text
POST   /auth/register
POST   /auth/login
POST   /auth/logout

POST   /uploads
GET    /uploads/{id}
DELETE /uploads/{id}

POST   /jobs
GET    /jobs/{id}
POST   /jobs/{id}/cancel
POST   /jobs/{id}/retry

GET    /jobs/{id}/clips
GET    /clips/{id}
PATCH  /clips/{id}
POST   /clips/{id}/render
DELETE /clips/{id}

GET    /health
GET    /health/worker

GET    /me/usage
GET    /me/settings

GET    /admin/jobs
GET    /admin/users
GET    /admin/system
```

------------------------------------------------------------------------

# 24. Real-Time Progress

Preferred:

-   Server-Sent Events for simple deployments.

Alternative:

-   WebSocket.

Events:

``` json
{
  "type": "job.progress",
  "job_id": "...",
  "stage": "TRANSCRIBING",
  "progress": 42
}
```

Frontend must reconnect automatically.

The backend must expose the latest persisted state so reconnecting does
not lose progress.

------------------------------------------------------------------------

# 25. Authentication

For private MVP:

-   optional local authentication.

For public deployment:

-   email/password or OAuth;
-   hashed passwords using a modern password hashing algorithm;
-   secure sessions;
-   CSRF protection where applicable;
-   refresh/session rotation;
-   account deletion.

Never store plaintext passwords.

------------------------------------------------------------------------

# 26. Authorization

Every resource query must be scoped to the authenticated user.

Example:

``` text
SELECT clip
WHERE clip.id = requested_id
AND clip.user_id = authenticated_user_id
```

Admin APIs require an explicit admin role.

Never trust `user_id` supplied by the frontend.

------------------------------------------------------------------------

# 27. Database Entities

Core tables:

``` text
users
sessions
projects
assets
jobs
job_events
transcripts
transcript_segments
candidates
candidate_features
clips
render_profiles
renders
usage_counters
subscriptions
payments
affiliate_events
audit_logs
system_settings
```

------------------------------------------------------------------------

# 28. Key Relationships

``` text
User
 ├── Projects
 │    ├── Assets
 │    │    └── Jobs
 │    │         ├── Transcript
 │    │         ├── Candidates
 │    │         └── Clips
 │    │              └── Renders
 │    └── Settings
 └── Usage
```

------------------------------------------------------------------------

# 29. Job Events

Every state transition should create an event:

``` text
job_id
event_type
stage
timestamp
duration_ms
message
error_code
metadata_json
```

Do not store raw stack traces in user-visible events.

------------------------------------------------------------------------

# 30. Error Model

Stable error codes:

``` text
MEDIA_TOO_LARGE
MEDIA_TOO_LONG
UNSUPPORTED_FORMAT
MEDIA_CORRUPTED
TRANSCRIPTION_FAILED
MODEL_UNAVAILABLE
INSUFFICIENT_MEMORY
RENDER_FAILED
OUTPUT_VALIDATION_FAILED
QUOTA_EXCEEDED
STORAGE_FULL
WORKER_OFFLINE
JOB_CANCELLED
UNKNOWN_ERROR
```

Errors should include:

-   code;
-   retryable;
-   user message;
-   internal diagnostic ID.

------------------------------------------------------------------------

# 31. Retry Strategy

Retry only retryable failures.

Recommended:

``` text
attempt 1 → immediate retry for transient process failure
attempt 2 → exponential backoff
attempt 3 → mark failed
```

Do not endlessly retry:

-   invalid media;
-   quota exceeded;
-   unsupported format;
-   invalid configuration.

------------------------------------------------------------------------

# 32. Resource Management

The worker must enforce:

-   maximum concurrent FFmpeg processes;
-   maximum concurrent transcription jobs;
-   maximum GPU memory use;
-   maximum temporary disk usage;
-   maximum job duration.

Before starting a job:

``` text
check disk
check RAM
check GPU availability
check worker capacity
```

If resources are insufficient:

`QUEUED` instead of crashing.

------------------------------------------------------------------------

# 33. Cleanup Service

A scheduled cleanup task must:

1.  find expired source files;
2.  find abandoned uploads;
3.  delete temporary artifacts;
4.  remove expired exports;
5.  release orphan database records;
6.  record cleanup metrics.

Cleanup must be idempotent.

------------------------------------------------------------------------

# 34. Quota Engine

All quotas must be configurable.

Example:

``` yaml
free:
  daily_source_minutes: 60
  concurrent_jobs: 1
  max_duration_minutes: 60

pro:
  monthly_source_minutes: 1000
  concurrent_jobs: 3
  max_duration_minutes: 180
```

The local private deployment can set these values to effectively
unlimited.

------------------------------------------------------------------------

# 35. Monetization Architecture

Never couple payment logic directly to processing logic.

Use:

``` text
EntitlementService
UsageService
BillingProvider
```

Pipeline asks:

``` text
can_process(user, duration)
```

It does not care whether the answer comes from:

-   free plan;
-   subscription;
-   credits;
-   admin override.

Future billing provider can be added without changing worker code.

------------------------------------------------------------------------

# 36. Zero-Cost Mode

Environment variable:

``` text
DEPLOYMENT_MODE=local
```

Local mode:

-   SQLite;
-   local filesystem;
-   local models;
-   local queue;
-   no payment provider;
-   no paid AI API;
-   optional local authentication.

Cloud mode:

``` text
DEPLOYMENT_MODE=cloud
```

uses external infrastructure providers.

------------------------------------------------------------------------

# 37. Configuration

All deployment-specific values must use environment variables or config
files.

Examples:

``` text
APP_ENV
DATABASE_URL
STORAGE_ROOT
MAX_UPLOAD_BYTES
MAX_VIDEO_SECONDS
WORKER_CONCURRENCY
WHISPER_MODEL
LLM_PROVIDER
OLLAMA_BASE_URL
LLM_MODEL
FFMPEG_PATH
RETENTION_HOURS
FREE_DAILY_MINUTES
JWT_SECRET
```

Secrets must never be committed.

------------------------------------------------------------------------

# 38. Observability

Structured JSON logs:

``` json
{
  "timestamp": "...",
  "level": "INFO",
  "service": "worker",
  "job_id": "...",
  "stage": "TRANSCRIBING",
  "duration_ms": 12345
}
```

Metrics:

-   jobs started;
-   jobs completed;
-   jobs failed;
-   processing seconds;
-   queue length;
-   average stage duration;
-   disk usage;
-   GPU usage;
-   model errors.

------------------------------------------------------------------------

# 39. Health Endpoints

`/health`

Checks API process.

`/health/ready`

Checks:

-   database;
-   storage;
-   worker availability.

`/health/worker`

Checks:

-   model runtime;
-   FFmpeg;
-   disk;
-   GPU if configured.

------------------------------------------------------------------------

# 40. Security Requirements

Mandatory:

-   input validation;
-   output encoding;
-   secure headers;
-   rate limits;
-   authentication;
-   authorization;
-   path isolation;
-   upload scanning/validation;
-   no arbitrary command execution;
-   secret isolation;
-   audit logs for privileged actions.

Never expose:

-   filesystem paths;
-   internal stack traces;
-   environment variables;
-   model server credentials.

------------------------------------------------------------------------

# 41. Privacy/Security of Media

Source files should be stored outside the public web root.

Downloads should use:

-   authenticated streaming;
-   signed URLs in cloud mode;
-   access-controlled endpoints in local mode.

No public directory listing.

------------------------------------------------------------------------

# 42. API Idempotency

Upload and job creation should support idempotency keys.

Example:

``` text
Idempotency-Key: <uuid>
```

If a request is repeated after a network timeout, the server returns the
existing resource instead of creating duplicate jobs.

------------------------------------------------------------------------

# 43. Database Transactions

Use transactions for:

-   creating job;
-   changing job state;
-   recording usage;
-   creating render record;
-   marking completion.

Never deduct usage permanently before confirming the job was accepted.

------------------------------------------------------------------------

# 44. Worker Locking

A job may only have one active worker lease.

Concept:

``` text
job.lock_owner
job.lock_expires_at
```

Workers periodically renew leases.

If a worker dies, an expired lease allows recovery.

------------------------------------------------------------------------

# 45. Crash Recovery

On startup:

1.  find jobs in active states;
2.  check worker lease;
3.  mark abandoned jobs recoverable;
4.  validate existing artifacts;
5.  resume from last valid stage.

Never assume process memory is durable.

------------------------------------------------------------------------

# 46. Artifact Manifest

Every job should have a manifest:

``` json
{
  "job_id": "...",
  "artifacts": {
    "source": "...",
    "audio": "...",
    "transcript": "...",
    "candidates": "...",
    "subtitles": "...",
    "rendered_clips": []
  }
}
```

Each artifact should include:

-   path/key;
-   checksum;
-   size;
-   created_at;
-   stage;
-   version.

This makes the pipeline reproducible and resumable.

------------------------------------------------------------------------

# 47. Versioning

Record:

-   pipeline version;
-   transcription model;
-   LLM model;
-   scoring configuration version;
-   render profile version.

Example:

``` text
pipeline_version = 1.0.0
scoring_version = 1.0
whisper_model = small
render_profile = vertical_v1
```

This is critical for debugging quality changes.

------------------------------------------------------------------------

# 48. Testing Strategy

## Unit tests

Test:

-   scoring;
-   boundaries;
-   quota calculations;
-   state transitions;
-   filename generation;
-   security validation.

## Integration tests

Test:

-   upload → job;
-   job → transcript;
-   transcript → candidates;
-   candidate → render;
-   cleanup.

## Media tests

Maintain a small test corpus:

-   talking head;
-   podcast;
-   gaming;
-   silent video;
-   multilingual video;
-   corrupted file;
-   long video.

## Failure tests

Simulate:

-   FFmpeg crash;
-   transcription crash;
-   disk full;
-   worker death;
-   database interruption;
-   duplicate request;
-   cancellation.

------------------------------------------------------------------------

# 49. Performance Targets

Initial targets on reasonable consumer hardware:

-   API requests should return quickly;
-   upload should stream rather than load entire file into RAM;
-   pipeline stages should report progress;
-   transcription should use GPU when available;
-   render should avoid unnecessary intermediate encoding;
-   repeated renders should reuse transcript and analysis.

Exact real-time factor is hardware-dependent and must be measured rather
than promised.

------------------------------------------------------------------------

# 50. Deployment Layout

Recommended repository:

``` text
clipforge/
├── apps/
│   ├── web/
│   └── api/
├── worker/
├── packages/
│   ├── contracts/
│   ├── scoring/
│   ├── media/
│   └── shared/
├── models/
├── migrations/
├── scripts/
├── tests/
├── data/
├── docker/
├── .env.example
├── docker-compose.yml
├── README.md
└── LICENSE
```

------------------------------------------------------------------------

# 51. Backend Module Layout

``` text
api/
├── main.py
├── config.py
├── dependencies.py
├── routes/
├── schemas/
├── services/
│   ├── auth.py
│   ├── jobs.py
│   ├── uploads.py
│   ├── clips.py
│   ├── quotas.py
│   └── billing.py
├── db/
├── storage/
├── queue/
└── security/
```

Worker:

``` text
worker/
├── main.py
├── runner.py
├── pipeline/
│   ├── base.py
│   ├── ingest.py
│   ├── audio.py
│   ├── transcription.py
│   ├── segmentation.py
│   ├── candidates.py
│   ├── scoring.py
│   ├── framing.py
│   ├── captions.py
│   ├── rendering.py
│   └── qa.py
├── models/
├── providers/
└── resources/
```

------------------------------------------------------------------------

# 52. Provider Interfaces

Use interfaces/protocols rather than direct imports throughout business
logic.

``` text
TranscriptionProvider
LLMProvider
VisionProvider
StorageProvider
QueueProvider
BillingProvider
```

This is one of the most important architectural decisions.

------------------------------------------------------------------------

# 53. Local AI Provider Priority

Default:

``` text
1. Local GPU model
2. Local CPU model
3. Optional configured remote provider
```

Never automatically send user media/transcripts to a remote provider
unless the user/deployment configuration explicitly permits it.

------------------------------------------------------------------------

# 54. Model Fallback

Example:

``` text
large model unavailable
        ↓
medium available?
        ↓ no
small available?
        ↓ no
base available?
        ↓ no
return MODEL_UNAVAILABLE
```

The user should see:

> The local AI model is unavailable. Retry or choose a smaller model.

------------------------------------------------------------------------

# 55. Monetization Without Infrastructure Lock-In

The public SaaS can eventually use:

``` text
Free → local/cloud shared queue
Pro → higher quota
Creator → batch + priority
Team → shared workspace
```

But the processing engine only sees an entitlement:

``` json
{
  "processing_minutes_remaining": 500,
  "priority": 2,
  "batch_enabled": true
}
```

This keeps billing completely separate from AI processing.

------------------------------------------------------------------------

# 56. Analytics Event Schema

Example:

``` json
{
  "event": "clip_downloaded",
  "user_id": "...",
  "clip_id": "...",
  "timestamp": "...",
  "properties": {
    "duration": 42,
    "aspect_ratio": "9:16"
  }
}
```

Never collect unnecessary sensitive content.

------------------------------------------------------------------------

# 57. Cost-Control Rules

The public deployment must have hard limits.

Never allow:

-   unlimited anonymous uploads;
-   unlimited concurrent FFmpeg processes;
-   unlimited storage;
-   unlimited transcription;
-   unlimited retry loops.

This prevents a free service from becoming an accidental crypto-mining
farm for video processing.

------------------------------------------------------------------------

# 58. Recommended Build Order

### Phase 1 --- Foundation

-   repository;
-   configuration;
-   database;
-   API;
-   upload;
-   FFmpeg;
-   local storage.

### Phase 2 --- AI

-   faster-whisper;
-   transcript schema;
-   segmentation;
-   local LLM;
-   candidate scoring.

### Phase 3 --- Video

-   clip boundaries;
-   captions;
-   9:16 crop;
-   render;
-   QA.

### Phase 4 --- UX

-   dashboard;
-   progress;
-   preview;
-   downloads;
-   settings.

### Phase 5 --- Reliability

-   retries;
-   crash recovery;
-   cleanup;
-   resource limits;
-   tests;
-   observability.

### Phase 6 --- Monetization

-   entitlements;
-   usage metering;
-   billing provider;
-   plans;
-   payment webhooks.

### Phase 7 --- Scale

-   PostgreSQL;
-   Redis;
-   remote workers;
-   object storage;
-   GPU workers.

------------------------------------------------------------------------

# 59. Definition of Done --- Backend

The backend is considered production-ready for the initial release when:

-   all pipeline stages are independently testable;
-   jobs are asynchronous;
-   jobs survive worker restarts;
-   failed jobs can resume;
-   source files are isolated;
-   users cannot access other users' assets;
-   uploads are validated;
-   FFmpeg cannot execute arbitrary input;
-   quotas are enforced;
-   cleanup is automatic;
-   model providers are replaceable;
-   local-only mode works without paid APIs;
-   metrics and structured logs exist;
-   database migrations work from a fresh installation.

------------------------------------------------------------------------

# 60. Critical Architectural Decisions

These decisions should not be casually changed during implementation:

1.  **The API never performs long video processing synchronously.**
2.  **The worker owns media processing.**
3.  **The database owns job state.**
4.  **Artifacts are persisted independently from process memory.**
5.  **AI providers are abstracted.**
6.  **Billing is abstracted from processing.**
7.  **All limits are configuration-driven.**
8.  **Local mode works without paid services.**
9.  **Cloud mode can reuse the same pipeline.**
10. **Every pipeline version/model/config is recorded.**

These ten rules are the backbone of the system.
