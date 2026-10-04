// Thin, typed client for the CLIPFORGE API. All calls go through the Next.js
// rewrite to the local backend, so there is a single origin and no CORS games.

import type {
  Asset,
  AssetKind,
  AssetLibrary,
  Candidate,
  CaptionPlan,
  CaptionPresetInfo,
  CleanupReport,
  ClipAssetOptions,
  ClipDetail,
  ClipSummary,
  Diagnostics,
  HardwareReport,
  Job,
  ProjectDetail,
  ProjectStatus,
  ProjectSummary,
  QueueState,
  SettingsSchemaField,
  SystemStatus,
  Transcript,
} from "./types";

export class ApiFailure extends Error {
  code: string;
  hint: string;
  status: number;

  constructor(message: string, hint = "", code = "error", status = 500) {
    super(message);
    this.name = "ApiFailure";
    this.code = code;
    this.hint = hint;
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...(init?.headers ?? {}),
      },
      cache: "no-store",
    });
  } catch {
    throw offline();
  }
  return parseResponse<T>(response.status, response.ok, await response.text());
}

function offline(): ApiFailure {
  return new ApiFailure(
    "CLIPFORGE could not reach its backend.",
    "Make sure the API is running (python -m clipforge serve) and try again.",
    "offline",
    0,
  );
}

function parseResponse<T>(status: number, ok: boolean, text: string): T {
  let payload: any = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }
  if (!ok) {
    const error = payload?.error ?? payload?.detail ?? {};
    const message = typeof error === "string" ? error : error.message || failureMessage(status);
    throw new ApiFailure(message, error.hint ?? "", error.code ?? String(status), status);
  }
  return (payload as T) ?? ({} as T);
}

function failureMessage(status: number): string {
  if (status === 502 || status === 503 || status === 504) return "The CLIPFORGE backend is not responding.";
  if (status === 413) return "That file is too large for the server.";
  return `Request failed (${status})`;
}

/** Multipart upload with progress (fetch cannot report upload progress). */
function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.responseType = "text";
    if (onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(event.loaded / event.total);
      };
    }
    xhr.onload = () => {
      try {
        resolve(parseResponse<T>(xhr.status, xhr.status >= 200 && xhr.status < 300, xhr.responseText));
      } catch (error) {
        reject(error);
      }
    };
    xhr.onerror = () => reject(offline());
    xhr.send(form);
  });
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
const query = (params: Record<string, string | number | boolean | undefined>) =>
  new URLSearchParams(
    Object.entries(params)
      .filter(([, value]) => value !== undefined && value !== "")
      .map(([key, value]) => [key, String(value)]),
  ).toString();

export type SettingsPayload = {
  settings: Record<string, any>;
  schema: Record<string, SettingsSchemaField>;
  sections: Record<string, any>;
  paths: Record<string, string>;
  caption_presets: Record<string, CaptionPresetInfo>;
};

export type RenderAllResult = { queued: number; jobs: string[]; skipped: number; queue: QueueState };

export const api = {
  // ---------------------------------------------------------------- system
  status: () => request<SystemStatus>("/api/system/status"),
  hardware: (refresh = false) => request<HardwareReport>(`/api/system/hardware?refresh=${refresh}`),
  diagnostics: () => request<Diagnostics>("/api/system/diagnostics"),
  logs: (name: string, lines = 300) => request<{ file: string; lines: string[] }>(`/api/system/logs?${query({ name, lines })}`),
  errors: (limit = 30) => request<{ errors: any[] }>(`/api/system/errors?limit=${limit}`),
  capabilities: () => request<any>("/api/system/capabilities"),
  probe: (url: string) => request<any>("/api/system/probe", json({ url })),
  testOllama: (base_url = "", model = "") =>
    request<{ status: { available: boolean; base_url: string; model: string; models: string[]; error: string; version: string }; models: string[] }>(
      "/api/system/ollama/test",
      json({ base_url, model }),
    ),
  ollamaModels: (base_url = "") =>
    request<{ models: string[]; base_url: string; current: string }>(`/api/system/ollama/models?${query({ base_url })}`),
  cleanup: (options: { days?: number; renders?: boolean; dry_run?: boolean } = {}) =>
    request<CleanupReport>("/api/system/cleanup", json(options)),
  storage: () => request<{ usage: Record<string, number>; disk: { free_gb: number; total_gb: number } }>("/api/system/storage"),

  // -------------------------------------------------------------- settings
  settings: () => request<SettingsPayload>("/api/settings"),
  settingsSchema: () => request<{ schema: Record<string, SettingsSchemaField>; sections: string[] }>("/api/settings/schema"),
  updateSettings: (patch: Record<string, unknown>) =>
    request<{ settings: Record<string, any>; updated: string[] }>("/api/settings", { ...json(patch), method: "PUT" }),
  resetSettings: () => request<{ settings: Record<string, any> }>("/api/settings/reset", { method: "POST" }),
  templates: () => request<{ templates: any[] }>("/api/templates"),
  createTemplate: (payload: any) => request<any>("/api/templates", json(payload)),
  deleteTemplate: (id: string) => request<any>(`/api/templates/${id}`, { method: "DELETE" }),
  applyTemplate: (id: string, projectId = "") =>
    request<any>(`/api/templates/${id}/apply?${query({ project_id: projectId })}`, { method: "POST" }),
  captionStylePreview: (preset: string, theme: Record<string, unknown> = {}) =>
    request<CaptionPlan>("/api/captions/preview", json({ preset, theme })),

  // -------------------------------------------------------------- projects
  projects: (search = "") => request<{ projects: ProjectSummary[]; total: number }>(`/api/projects?${query({ search })}`),
  project: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),
  projectStatus: (id: string) => request<ProjectStatus>(`/api/projects/${id}/status`),
  projectQueue: (id: string) => request<{ queue: QueueState; renders: Job[]; summary: any }>(`/api/projects/${id}/queue`),
  transcript: (id: string, withWords = true) => request<Transcript>(`/api/projects/${id}/transcript?with_words=${withWords}`),
  candidates: (id: string, limit = 300) => request<{ candidates: Candidate[]; stats: any }>(`/api/projects/${id}/candidates?limit=${limit}`),
  projectClips: (id: string, sort = "score", category = "") =>
    request<{ clips: ClipSummary[]; categories: any[]; stats: any }>(`/api/projects/${id}/clips?${query({ sort, category })}`),
  createProject: (payload: { url: string; title?: string; analyze?: boolean; options?: Record<string, unknown> }) =>
    request<{ project: ProjectSummary; job: Job | null }>("/api/projects", json({ analyze: true, ...payload })),
  uploadProject: (file: File, options: Record<string, unknown>, title = "", onProgress?: (fraction: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    form.append("title", title);
    form.append("options", JSON.stringify(options));
    form.append("analyze", "true");
    return upload<{ project: ProjectSummary; job: Job | null }>("/api/projects/upload", form, onProgress);
  },
  uploadTranscript: (projectId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return upload<{ file: string; words: number; message: string }>(`/api/projects/${projectId}/transcript`, form);
  },
  analyze: (projectId: string, options?: Record<string, unknown>, force = false) =>
    request<{ project_id: string; job: Job }>(`/api/projects/${projectId}/analyze`, json({ options, force })),
  cancelProject: (id: string) => request<{ cancelled_jobs: number }>(`/api/projects/${id}/cancel`, { method: "POST" }),
  renderAll: (id: string, clipIds?: string[], exportFiles = false) =>
    request<RenderAllResult>(`/api/projects/${id}/render-all`, json({ clip_ids: clipIds ?? [], export: exportFiles })),
  openFolder: (id: string) => request<{ opened: string }>(`/api/projects/${id}/open-folder`, { method: "POST" }),
  disk: (id: string) => request<any>(`/api/projects/${id}/disk`),
  deleteProject: (id: string) => request<any>(`/api/projects/${id}?remove_files=true`, { method: "DELETE" }),

  // ----------------------------------------------------------------- clips
  clip: (id: string) => request<ClipDetail>(`/api/clips/${id}`),
  updateClip: (id: string, patch: Record<string, unknown>) =>
    request<ClipDetail>(`/api/clips/${id}`, { ...json(patch), method: "PATCH" }),
  deleteClip: (id: string) => request<{ deleted: string; project_id: string }>(`/api/clips/${id}?remove_files=true`, { method: "DELETE" }),
  renderClip: (id: string, body: { export?: boolean; overrides?: Record<string, unknown> } = {}) =>
    request<{ job: Job; clip_id: string }>(`/api/clips/${id}/render`, json(body)),
  previewClip: (id: string) => request<{ job: Job; clip_id: string }>(`/api/clips/${id}/preview`, json({})),
  cancelClip: (id: string) => request<{ cancelled: number }>(`/api/clips/${id}/cancel`, { method: "POST" }),
  regenerateClip: (id: string, body: { window?: number; use_llm?: boolean } = {}) =>
    request<ClipSummary & { changed: boolean; explanation: string; llm_note: string }>(`/api/clips/${id}/regenerate`, json(body)),
  duplicateClip: (id: string) => request<ClipDetail>(`/api/clips/${id}/duplicate`, { method: "POST" }),
  captionPreview: (id: string, preset: string, theme: Record<string, unknown> = {}) =>
    request<CaptionPlan>(`/api/clips/${id}/captions/preview`, json({ preset, theme })),
  clipCommand: (id: string, quality = "final") => request<{ command: string }>(`/api/clips/${id}/command?quality=${quality}`),
  clipAssets: (id: string) => request<ClipAssetOptions>(`/api/clips/${id}/assets`),
  renderAllClips: (clipIds: string[], exportFiles = false) =>
    request<RenderAllResult>("/api/clips/render-all", json({ clip_ids: clipIds, export: exportFiles })),

  // ---------------------------------------------------------------- assets
  assets: (kind = "", category = "") =>
    request<{ assets: Asset[]; library: AssetLibrary; categories: Record<AssetKind, string[]> }>(`/api/assets?${query({ kind, category })}`),
  uploadAsset: (file: File, kind: string, category: string, name = "", onProgress?: (fraction: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    form.append("category", category);
    form.append("name", name);
    return upload<{ asset: Asset }>("/api/assets/upload", form, onProgress);
  },
  importAssetPath: (payload: { path: string; kind: string; category: string; name?: string; copy?: boolean }) =>
    request<{ asset: Asset }>("/api/assets/import-path", json(payload)),
  scanAssets: () => request<{ imported: unknown[] }>("/api/assets/scan", { method: "POST" }),
  updateAsset: (id: string, patch: { enabled?: boolean; favorite?: boolean; category?: string }) =>
    request<Asset>(`/api/assets/${id}`, { ...json(patch), method: "PATCH" }),
  deleteAsset: (id: string, removeFile = true) => request<any>(`/api/assets/${id}?remove_file=${removeFile}`, { method: "DELETE" }),

  // ------------------------------------------------------------------ jobs
  jobs: (limit = 100) => request<{ jobs: Job[] }>(`/api/jobs?limit=${limit}`),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  queue: () => request<{ queue: QueueState; workers: any }>("/api/jobs/queue"),
  workers: () => request<any>("/api/workers"),
  startWorkers: (workers = 0) => request<any>(`/api/workers/start?workers=${workers}`, { method: "POST" }),
  stopWorkers: () => request<any>("/api/workers/stop", { method: "POST" }),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) => request<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),
  retryFailed: (projectId = "") => request<{ retried: number }>(`/api/jobs/retry-failed?${query({ project_id: projectId })}`, { method: "POST" }),
  purgeJobs: (projectId = "") => request<{ removed: number }>(`/api/jobs/purge?${query({ project_id: projectId })}`, { method: "POST" }),
};

// ------------------------------------------------------------------ events

/** One Server-Sent Event: ``type`` plus the event's own fields. */
export type StreamEvent = { type: string; at?: number; project_id?: string; job_id?: string; clip_id?: string; [key: string]: any };

/**
 * Subscribe to the live event stream (optionally for one project).
 * EventSource reconnects by itself; the returned function closes the stream.
 */
export function subscribeEvents(onEvent: (event: StreamEvent) => void, projectId = ""): () => void {
  if (typeof window === "undefined" || typeof EventSource === "undefined") return () => undefined;
  const source = new EventSource(`/api/events?${query({ project_id: projectId, include_hardware: false })}`);
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch {
      /* ignore malformed frames */
    }
  };
  return () => source.close();
}
