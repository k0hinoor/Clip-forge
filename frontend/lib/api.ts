// Thin, typed client for the CLIPFORGE API. All calls go through the Next.js
// rewrite to the local backend, so there is a single origin and no CORS games.

import type {
  Asset,
  Candidate,
  ClipDetail,
  ClipSummary,
  HardwareReport,
  Job,
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
  } catch (error) {
    throw new ApiFailure(
      "CLIPFORGE could not reach its local backend.",
      "Make sure the app is running (python -m clipforge) and try again.",
      "offline",
      0,
    );
  }

  const text = await response.text();
  let payload: any = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const error = payload?.error ?? payload?.detail ?? {};
    const message =
      typeof error === "string"
        ? error
        : error.message || `Request failed (${response.status})`;
    throw new ApiFailure(message, error.hint ?? "", error.code ?? String(response.status), response.status);
  }
  return (payload as T) ?? ({} as T);
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  // ---------------------------------------------------------------- system
  status: () => request<SystemStatus>("/api/system/status"),
  hardware: (refresh = false) => request<HardwareReport>(`/api/system/hardware?refresh=${refresh}`),
  diagnostics: () => request<any>("/api/system/diagnostics"),
  logs: (name: string, lines = 300) => request<{ name: string; lines: string[] }>(`/api/system/logs?name=${name}&lines=${lines}`),
  errors: (limit = 30) => request<{ errors: any[] }>(`/api/system/errors?limit=${limit}`),
  capabilities: () => request<any>("/api/system/capabilities"),
  probe: (url: string) => request<any>("/api/system/probe", json({ url })),
  testOllama: (base_url: string, model: string) => request<any>("/api/system/ollama/test", json({ base_url, model })),
  ollamaModels: (base_url = "") => request<{ models: string[]; error?: string }>(`/api/system/ollama/models?base_url=${encodeURIComponent(base_url)}`),

  // -------------------------------------------------------------- settings
  settings: () =>
    request<{
      settings: Record<string, any>;
      schema: Record<string, SettingsSchemaField>;
      sections: Record<string, any>;
      paths: Record<string, string>;
      caption_presets: Record<string, any>;
    }>("/api/settings"),
  settingsSchema: () =>
    request<{ schema: Record<string, SettingsSchemaField>; sections: Record<string, string> }>("/api/settings/schema"),
  updateSettings: (patch: Record<string, unknown>) => request<{ settings: Record<string, any> }>("/api/settings", { ...json(patch), method: "PUT" }),
  resetSettings: () => request<{ settings: Record<string, any> }>("/api/settings/reset", { method: "POST" }),
  templates: () => request<{ templates: any[] }>("/api/templates"),
  createTemplate: (payload: any) => request<any>("/api/templates", json(payload)),
  deleteTemplate: (id: string) => request<any>(`/api/templates/${id}`, { method: "DELETE" }),
  applyTemplate: (id: string, projectId = "") => request<any>(`/api/templates/${id}/apply?project_id=${projectId}`, { method: "POST" }),

  // -------------------------------------------------------------- projects
  projects: (search = "") => request<{ projects: ProjectSummary[]; total: number }>(`/api/projects?search=${encodeURIComponent(search)}`),
  project: (id: string) => request<ProjectSummary>(`/api/projects/${id}`),
  projectStatus: (id: string) => request<ProjectStatus>(`/api/projects/${id}/status`),
  projectQueue: (id: string) => request<{ queue: QueueState; renders: any[]; summary: any }>(`/api/projects/${id}/queue`),
  transcript: (id: string, withWords = true) => request<Transcript>(`/api/projects/${id}/transcript?with_words=${withWords}`),
  candidates: (id: string, limit = 300) => request<{ candidates: Candidate[]; stats: any }>(`/api/projects/${id}/candidates?limit=${limit}`),
  projectClips: (id: string, sort = "score", category = "") =>
    request<{ clips: ClipSummary[]; categories: any[]; stats: any }>(`/api/projects/${id}/clips?sort=${sort}&category=${category}`),
  createProject: (payload: { url: string; title?: string; analyze?: boolean; options?: Record<string, unknown> }) =>
    request<{ project: ProjectSummary; job: Job | null }>("/api/projects", json({ analyze: true, ...payload })),
  uploadProject: async (file: File, options: Record<string, unknown>, title = "") => {
    const form = new FormData();
    form.append("file", file);
    form.append("title", title);
    form.append("options", JSON.stringify(options));
    form.append("analyze", "true");
    return request<{ project: ProjectSummary; job: Job | null }>("/api/projects/upload", { method: "POST", body: form });
  },
  uploadTranscript: async (projectId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ file: string; words: number; message: string }>(`/api/projects/${projectId}/transcript`, {
      method: "POST",
      body: form,
    });
  },
  analyze: (projectId: string, options?: Record<string, unknown>) =>
    request<{ project_id: string; job: Job }>(`/api/projects/${projectId}/analyze`, json({ options })),
  cancelProject: (id: string) => request<{ cancelled_jobs: number }>(`/api/projects/${id}/cancel`, { method: "POST" }),
  renderAll: (id: string, clipIds?: string[], exportFiles = false) =>
    request<{ queued: any[]; skipped: any[]; queue: QueueState }>(`/api/projects/${id}/render-all`, json({ clip_ids: clipIds, export: exportFiles })),
  openFolder: (id: string) => request<{ opened: string }>(`/api/projects/${id}/open-folder`, { method: "POST" }),
  disk: (id: string) => request<any>(`/api/projects/${id}/disk`),
  deleteProject: (id: string) => request<any>(`/api/projects/${id}?remove_files=true`, { method: "DELETE" }),

  // ----------------------------------------------------------------- clips
  clip: (id: string) => request<ClipDetail>(`/api/clips/${id}`),
  updateClip: (id: string, patch: Record<string, unknown>) => request<ClipDetail>(`/api/clips/${id}`, { ...json(patch), method: "PATCH" }),
  deleteClip: (id: string) => request<any>(`/api/clips/${id}?remove_files=true`, { method: "DELETE" }),
  renderClip: (id: string, body: Record<string, unknown> = {}) => request<any>(`/api/clips/${id}/render`, json(body)),
  previewClip: (id: string, body: Record<string, unknown> = {}) => request<any>(`/api/clips/${id}/preview`, json(body)),
  cancelClip: (id: string) => request<any>(`/api/clips/${id}/cancel`, { method: "POST" }),
  regenerateClip: (id: string, body: Record<string, unknown> = {}) => request<any>(`/api/clips/${id}/regenerate`, json(body)),
  duplicateClip: (id: string) => request<any>(`/api/clips/${id}/duplicate`, { method: "POST" }),
  captionPreview: (id: string, preset: string, theme: Record<string, unknown> = {}) =>
    request<any>(`/api/clips/${id}/captions/preview`, json({ preset, theme })),
  clipCommand: (id: string, quality = "final") => request<{ command: string; plan: string[] }>(`/api/clips/${id}/command?quality=${quality}`),
  clipAssets: (id: string) => request<any>(`/api/clips/${id}/assets`),
  renderAllClips: (clipIds: string[], exportFiles = false) => request<any>("/api/clips/render-all", json({ clip_ids: clipIds, export: exportFiles })),

  // ---------------------------------------------------------------- assets
  assets: (kind = "", category = "") => request<{ assets: Asset[]; library: any }>(`/api/assets?kind=${kind}&category=${category}`),
  gameplayAssets: (category = "") => request<{ assets: Asset[]; categories: string[]; folders: any }>(`/api/assets/gameplay?category=${category}`),
  musicAssets: () => request<{ assets: Asset[] }>("/api/assets/music"),
  uploadAsset: async (file: File, kind: string, category: string, name = "") => {
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    form.append("category", category);
    form.append("name", name);
    return request<{ asset: Asset }>("/api/assets/upload", { method: "POST", body: form });
  },
  importAssetPath: (payload: { path: string; kind: string; category: string; name?: string; copy?: boolean }) =>
    request<{ asset: Asset }>("/api/assets/import-path", json(payload)),
  scanAssets: () => request<any>("/api/assets/scan", { method: "POST" }),
  updateAsset: (id: string, patch: Record<string, unknown>) => request<any>(`/api/assets/${id}`, { ...json(patch), method: "PATCH" }),
  deleteAsset: (id: string, removeFile = true) => request<any>(`/api/assets/${id}?remove_file=${removeFile}`, { method: "DELETE" }),

  // ------------------------------------------------------------------ jobs
  jobs: (limit = 100) => request<{ jobs: Job[] }>(`/api/jobs?limit=${limit}`),
  queue: () => request<{ queue: QueueState; workers: any }>("/api/jobs/queue"),
  workers: () => request<any>("/api/workers"),
  startWorkers: (workers = 0) => request<any>(`/api/workers/start?workers=${workers}`, { method: "POST" }),
  stopWorkers: () => request<any>("/api/workers/stop", { method: "POST" }),
  cancelJob: (id: string) => request<any>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) => request<any>(`/api/jobs/${id}/retry`, { method: "POST" }),
  retryFailed: (projectId = "") => request<any>(`/api/jobs/retry-failed?project_id=${projectId}`, { method: "POST" }),
  purgeJobs: (projectId = "") => request<any>(`/api/jobs/purge?project_id=${projectId}`, { method: "POST" }),
};

// ------------------------------------------------------------------ events

export type StreamEvent = { type: string; data: any; project_id?: string; job_id?: string; at?: number };

export function subscribeEvents(onEvent: (event: StreamEvent) => void, projectId = ""): () => void {
  const url = projectId ? `/api/events?project_id=${encodeURIComponent(projectId)}` : "/api/events";
  const source = new EventSource(url);
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch {
      /* ignore malformed frames */
    }
  };
  source.onerror = () => {
    // EventSource reconnects on its own; nothing to do here.
  };
  return () => source.close();
}
