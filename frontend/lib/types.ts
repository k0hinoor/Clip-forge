// Shared types mirroring the FastAPI payloads in backend/clipforge/api.

export type ClipMode = "best" | "balanced" | "max";
export type AspectRatio = "9:16" | "1:1" | "16:9";
export type LayoutMode = "split" | "podcast" | "broll" | "gameplay" | "cinematic" | "blur";

export interface ApiError {
  code: string;
  message: string;
  hint?: string;
  detail?: string;
  context?: Record<string, unknown>;
}

/** Project lifecycle as reported by the API. */
export type ProjectState = "draft" | "queued" | "running" | "ready" | "failed" | "cancelled";
/** Clip lifecycle as reported by the API. */
export type ClipState = "pending" | "queued" | "rendering" | "rendered" | "failed";
export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface ProjectSummary {
  id: string;
  title: string;
  slug: string;
  source_url: string;
  source_type: string;
  status: string;
  stage: string;
  status_message: string;
  progress: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
  language: string;
  language_mode: string;
  language_secondary: string;
  clip_count: number;
  candidate_count: number;
  word_count: number;
  segment_count: number;
  transcript_source: string;
  transcript_preference: string;
  transcript_filename: string;
  transcript_timing: "word" | "cue" | "";
  speakers: number;
  created_at: string;
  updated_at: string;
  error: ApiError | null;
  stats: Record<string, any>;
  settings: Record<string, any>;
  rendered_clips?: number;
  pending_clips?: number;
  active_job?: Job | null;
  /** Only on the detail payload. */
  has_source?: boolean;
  language_name?: string;
}

export interface ProjectDetail extends ProjectSummary {
  has_source: boolean;
  language_name: string;
  source_file: string;
  clips: ClipSummary[];
  jobs: Job[];
}

export interface Job {
  id: string;
  kind: "analyze" | "render_clip" | "render_preview" | "scan_assets" | string;
  status: JobState;
  progress: number;
  stage: string;
  message: string;
  project_id: string;
  clip_id: string;
  priority: number;
  error: ApiError | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  attempts: number;
  cancel_requested?: boolean;
  duration_seconds?: number | null;
  /** Per-stage log (GET /api/jobs/{id}, or with_log=true). */
  stages?: LogLine[];
  result?: Record<string, any>;
  /** Added by the queue endpoint. */
  clip_title?: string;
  clip_index?: number;
  project_title?: string;
}

export interface LogLine {
  stage: string;
  message: string;
  progress?: number;
}

export interface StageState {
  key: string;
  label: string;
  /** The pipeline reports `state`; older payloads used `status`. */
  state?: "pending" | "running" | "done" | "failed";
  status?: "pending" | "active" | "done" | "failed" | "skipped";
  detail?: string;
  progress: number;
  message?: string;
  weight?: number;
}

export interface ProjectStatus {
  project: ProjectSummary & {
    word_count?: number;
    segment_count?: number;
    speakers?: number;
    clip_count?: number;
    transcript_source?: string;
    transcript_preference?: string;
    transcript_filename?: string;
    transcript_timing?: string;
  };
  job: Job | null;
  stages: StageState[];
  candidates: {
    discovered: number;
    stored: number;
    scored: number;
    scoring_errors: number;
    threshold_pass: number;
    context_rejections: number;
    overlap_rejections: number;
    final_accepted: number;
    average_score: number;
    highest_score: number;
    threshold: number;
    queued: number;
    rendered: number;
    top_rejection_reason: string;
    statuses: Record<string, number>;
  };
  diagnostics: Record<string, any>;
  clips: { total: number; rendered: number; failed: number; pending: number; queued: number; rendering: number };
}

export interface QueueState {
  running: Job[];
  queued: Job[];
  failed: Job[];
  cancelled: Job[];
  recent: Job[];
  counts: { running: number; queued: number; failed: number; total: number };
}

export interface ClipSummary {
  id: string;
  project_id: string;
  index: number;
  title: string;
  hook: string;
  summary: string;
  category: string;
  category_label: string;
  score: number;
  start: number;
  end: number;
  duration: number;
  status: ClipState;
  progress: number;
  stage: string;
  why: string[];
  factors: Record<string, number>;
  has_render: boolean;
  has_preview: boolean;
  has_thumbnail: boolean;
  file_size: number;
  width: number;
  height: number;
  fps: number;
  render_seconds: number;
  error: ApiError | null;
  created_at: string;
  rendered_at: string | null;
  render_url: string | null;
  preview_url: string | null;
  thumbnail_url: string | null;
  subtitle_url: string | null;
  words?: TranscriptWord[];
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  speaker: string;
}

export interface TranscriptSegment {
  index: number;
  start: number;
  end: number;
  text: string;
  speaker: string;
  confidence: number;
  word_count: number;
  has_word_timings?: boolean;
  words?: TranscriptWord[];
}

export interface Transcript {
  project_id: string;
  language: string;
  language_name: string;
  language_mode: string;
  language_secondary: string;
  engine: string;
  model: string;
  source: string;
  preference: string;
  filename: string;
  timing_granularity: "word" | "cue" | "";
  word_count: number;
  segment_count: number;
  speakers: number;
  segments: TranscriptSegment[];
  count: number;
}

export interface Candidate {
  id: string;
  start: number;
  end: number;
  duration: number;
  title: string;
  hook: string;
  summary: string;
  category: string;
  category_label: string;
  score: number;
  confidence: number;
  reason: string;
  rejection_code: string;
  status: string;
  factors: Record<string, number>;
  penalties: Record<string, number>;
  highlights: string[];
  transcript_text: string;
  duplicate_of: string;
}

export interface TimelineSegment {
  src_start: number;
  src_end: number;
  out_start: number;
  out_end: number;
}

export interface ClipDetail extends ClipSummary {
  captions_enabled: boolean;
  plan: {
    layout: LayoutMode;
    split_ratio: number;
    gameplay: AssetRef | null;
    broll: AssetRef | null;
    music: AssetRef | null;
    notes: string[];
    zoom_points: { time: number; strength: number; reason: string }[];
    crop: Record<string, any> | null;
    timeline: {
      segments: TimelineSegment[];
      removed_seconds?: number;
      output_duration?: number;
      notes?: string[];
      source_start: number;
      source_end: number;
    };
  };
  /** Effective per-clip switches (project settings + this clip's edits). */
  settings: { aspect_ratio: AspectRatio; remove_silence: boolean; auto_zoom: boolean; gameplay_enabled: boolean; music_enabled: boolean };
  captions: CaptionPlan | null;
  caption_presets: string[];
  needs_render: boolean;
  project: { id: string; title: string; language: string; language_mode: string; duration: number };
}

export interface ClipAssetOptions {
  gameplay: Asset[];
  broll: Asset[];
  music: Asset[];
}

export interface CaptionPlan {
  lines: { index: number; text: string; start: number; end: number; has_word_timings?: boolean; words: { text: string; start: number; end: number }[] }[];
  timing_granularity?: "word" | "cue";
  /** The full style behind the preset, so the editor can show what is active. */
  theme?: { preset?: string; [key: string]: unknown };
  preset?: string;
  animation?: string;
  word_count?: number;
  line_count?: number;
  language: string;
  font: string;
  notes: string[];
}

export interface AssetRef {
  id: string;
  name: string;
  category?: string;
  path?: string;
  reason?: string;
}

export type AssetKind = "gameplay" | "broll" | "music";

export interface Asset {
  id: string;
  name: string;
  kind: AssetKind;
  category: string;
  filename: string;
  path: string;
  size_bytes: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
  enabled: boolean;
  favorite: boolean;
  tags: string[];
  stream_url: string;
  created_at: string;
}

export interface AssetLibrary {
  counts: Record<AssetKind, number>;
  by_category: Record<AssetKind, Record<string, number>>;
  total_seconds: number;
}

export interface SettingsSchemaField {
  name: string;
  label: string;
  type: "integer" | "number" | "boolean" | "string" | "enum" | "object";
  options: (string | Record<string, unknown>)[];
  default: unknown;
  value: unknown;
  section: string;
  help: string;
  /** Server paths cannot be changed on a public deployment. */
  readonly?: boolean;
  // pydantic constraint metadata, e.g. {"ge": 1, "le": 64}
  ge?: number;
  le?: number;
  gt?: number;
  lt?: number;
}

export interface FfmpegInfo {
  available: boolean;
  ffmpeg: string;
  ffprobe: string;
  version: string;
  source: string;
  libass: boolean;
  drawtext: boolean;
  overlay: boolean;
  libx264: boolean;
  encoders: string[];
  hwaccels: string[];
  problems: string[];
}

export interface AiStack {
  faster_whisper: boolean;
  whisperx: boolean;
  opencv: boolean;
  torch: boolean;
  torch_cuda: boolean;
  faster_whisper_version: string;
  transcription_available: boolean;
}

export interface GpuDevice {
  name: string;
  vendor?: string;
  vram_total_mb?: number;
  vram_free_mb?: number;
  driver?: string;
  compute?: string;
}

export interface HardwareInfo {
  os: string;
  os_version: string;
  arch: string;
  python: string;
  cpu: { name: string; logical_cores: number; physical_cores: number; usage_percent: number };
  memory: { total_gb: number; available_gb: number; used_percent: number; container_limited?: boolean };
  gpu: { available: boolean; vendor: string; cuda: boolean; devices: GpuDevice[] };
  disk: { total_gb?: number; free_gb?: number; used_percent?: number };
  ffmpeg: FfmpegInfo;
  ai: AiStack;
  recommended: { whisper_device: string; whisper_model: string; hw_accel: string; concurrency: number };
}

export interface HardwareReport {
  hardware: HardwareInfo;
  ai: AiStack;
  ffmpeg: FfmpegInfo;
}

export interface SystemNote {
  level: "info" | "warning" | "error" | string;
  title: string;
  detail: string;
}

export interface Diagnostics {
  paths: { data_dir: string; database: string; logs: string; exports: string };
  logs: { app: string; worker: string; render: string; ai: string };
  ffmpeg: FfmpegInfo;
  ai: AiStack;
  errors: { at?: string; file?: string; logger?: string; message?: string }[];
  python: string;
  platform: string;
}

export interface SystemStatus {
  app: string;
  version: string;
  ready: boolean;
  data_dir: string;
  ffmpeg: FfmpegInfo;
  ai: AiStack;
  hardware: HardwareInfo;
  workers: { running: boolean; workers: number; worker_names: string[]; busy: string[]; uptime_seconds: number };
  queue: { queued: number; running: number };
  usage: { cpu_percent: number; memory_percent: number; process_memory_mb: number; process_cpu_percent: number };
  /** What this deployment allows (local paths / folders only on the desktop app). */
  features: { local_paths: boolean; open_folder: boolean };
  notes: SystemNote[];
}

export interface CleanupReport {
  days: number;
  dry_run: boolean;
  cache: { removed_files: number; freed_bytes: number; remaining_bytes: number };
  projects_removed: { id: string; title: string; status: string; bytes: number }[];
  renders_removed: number;
  source_files_removed: number;
  freed_bytes: number;
  usage: Record<string, number>;
}

export interface CaptionPresetInfo {
  label: string;
  description: string;
  theme: Record<string, unknown>;
}

