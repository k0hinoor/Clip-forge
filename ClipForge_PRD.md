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
