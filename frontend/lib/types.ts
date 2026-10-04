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
  speakers: number;
  created_at: string;
  updated_at: string;
  error_code: string;
  error_message: string;
  stats: Record<string, any>;
  settings: Record<string, any>;
  rendered_clips?: number;
  pending_clips?: number;
  active_job?: Job | null;
}

export interface Job {
  id: string;
  kind: string;
  label: string;
  status: string;
  progress: number;
  stage: string;
  message: string;
  project_id: string;
  clip_id: string;
  error_code: string;
  error_message: string;
  error_hint: string;
  created_at: string;
  started_at: string;
  finished_at: string;
  attempts: number;
  duration_seconds?: number;
  log?: LogLine[];
  result?: Record<string, any>;
}

export interface LogLine {
  at: number;
  time?: string;
  message: string;
  stage?: string;
}

export interface StageState {
  key: string;
  label: string;
  status: "pending" | "active" | "done" | "failed" | "skipped";
  progress: number;
  message: string;
}

export interface ProjectStatus {
  project: ProjectSummary;
  stages: StageState[];
  queue: QueueState;
  message?: string;
}

export interface QueueState {
  running: Job[];
  queued: Job[];
  failed: Job[];
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
  confidence: number;
  start: number;
  end: number;
  duration: number;
  status: string;
  progress: number;
  stage: string;
  why: string[];
  factors: Record<string, number>;
  highlights: string[];
  output_path: string;
  export_path: string;
  thumbnail: string;
  rendered_at: string;
  error_code: string;
  error_message: string;
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
  status: string;
  factors: Record<string, number>;
  penalties: Record<string, number>;
  highlights: string[];
  transcript_text: string;
  duplicate_of: string;
}

export interface ClipDetail extends ClipSummary {
  plan: {
    layout: LayoutMode;
    split_ratio: number;
    gameplay: AssetRef | null;
    broll: AssetRef | null;
    music: AssetRef | null;
    notes: string[];
    zoom_points: { time: number; strength: number; reason: string }[];
    crop: Record<string, any> | null;
    timeline: { segments: any[]; removed_seconds?: number; notes?: string[]; source_start: number; source_end: number };
  };
  captions: CaptionPlan | null;
  caption_presets: string[];
  needs_render: boolean;
  transcript_text?: string;
  project: { id: string; title: string; language: string; language_mode: string; duration: number };
}

export interface CaptionPlan {
  lines: { index: number; text: string; start: number; end: number; words: TranscriptWord[] }[];
  theme: Record<string, any>;
  language: string;
  font: string;
  notes: string[];
}

export interface AssetRef {
  id: string;
  name: string;
  kind: string;
  category: string;
  path: string;
  duration: number;
  reason?: string;
}

export interface Asset {
  id: string;
  name: string;
  kind: string;
  category: string;
  path: string;
  size: number;
  size_label: string;
  duration: number;
  width: number;
  height: number;
  fps: number;
  enabled: boolean;
  favorite: boolean;
  licence: string;
  source: string;
  created_at: string;
  has_audio: boolean;
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
  memory: { total_gb: number; available_gb: number; used_percent: number };
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
  errors: { at?: string; logger?: string; message?: string }[];
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
  notes: SystemNote[];
}

