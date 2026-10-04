"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Link2, Loader2, Plus, Trash2, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { useLiveRefresh } from "@/lib/live";
import { CaptionStyleSelect, LAYOUT_OPTIONS, Toggle } from "@/components/Controls";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { clock, isAnalysing, statusChip, when } from "@/lib/format";
import type { CaptionPresetInfo, ProjectSummary } from "@/lib/types";

const LENGTHS: Record<string, { min: number; target: number; max: number; label: string }> = {
  short: { min: 15, target: 25, max: 30, label: "Short · 15–30 s" },
  standard: { min: 35, target: 60, max: 75, label: "Standard · 35–75 s" },
  long: { min: 60, target: 90, max: 120, label: "Long · 60–120 s" },
  custom: { min: 35, target: 60, max: 75, label: "Custom" },
};

const RATIOS = [
  ["9:16", "9:16 vertical"],
  ["1:1", "1:1 square"],
  ["16:9", "16:9 landscape"],
];

const ACCEPTED_VIDEO = "video/*,.mp4,.mov,.mkv,.webm,.m4v,.avi";

type Options = {
  aspect_ratio: string;
  caption_preset: string;
  layout: string;
  remove_silence: boolean;
  auto_zoom: boolean;
  captions_enabled: boolean;
  min_score: number;
  clip_mode: string;
  translate_captions: boolean;
  llm_enabled: boolean;
  min_clip_seconds: number;
  target_clip_seconds: number;
  max_clip_seconds: number;
};

const FALLBACK_OPTIONS: Options = {
  aspect_ratio: "9:16",
  caption_preset: "bold_creator",
  layout: "split",
  remove_silence: true,
  auto_zoom: true,
  captions_enabled: true,
  min_score: 70,
  clip_mode: "balanced",
  translate_captions: false,
  llm_enabled: true,
  min_clip_seconds: 35,
  target_clip_seconds: 60,
  max_clip_seconds: 75,
};

function optionsFromSettings(settings: Record<string, any>): Options {
  const pick = <K extends keyof Options>(key: K): Options[K] => (settings[key] ?? FALLBACK_OPTIONS[key]) as Options[K];
  return {
    aspect_ratio: pick("aspect_ratio"),
    caption_preset: settings.caption?.preset ?? FALLBACK_OPTIONS.caption_preset,
    layout: pick("layout"),
    remove_silence: pick("remove_silence"),
    auto_zoom: pick("auto_zoom"),
    captions_enabled: pick("captions_enabled"),
    min_score: pick("min_score"),
    clip_mode: pick("clip_mode"),
    translate_captions: pick("translate_captions"),
    llm_enabled: pick("llm_enabled"),
    min_clip_seconds: pick("min_clip_seconds"),
    target_clip_seconds: pick("target_clip_seconds"),
    max_clip_seconds: pick("max_clip_seconds"),
  };
}

function lengthKeyFor(options: Options): string {
  const match = Object.entries(LENGTHS).find(
    ([key, preset]) =>
      key !== "custom" &&
      preset.min === options.min_clip_seconds &&
      preset.target === options.target_clip_seconds &&
      preset.max === options.max_clip_seconds,
  );
  return match ? match[0] : "custom";
}

export default function StudioPage() {
  const toast = useToast();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [lengthKey, setLengthKey] = useState("standard");
  const [uploadsEnabled, setUploadsEnabled] = useState(true);
  const [presets, setPresets] = useState<Record<string, CaptionPresetInfo>>({});
  const fileInput = useRef<HTMLInputElement>(null);

  // The form starts from the saved Settings; only what the user changes here is
  // sent with the project, everything else keeps following Settings.
  const [options, setOptions] = useState<Options>(FALLBACK_OPTIONS);
  const [touched, setTouched] = useState<Set<keyof Options>>(new Set());

  const change = (patch: Partial<Options>) => {
    setOptions((current) => ({ ...current, ...patch }));
    setTouched((current) => new Set([...current, ...(Object.keys(patch) as (keyof Options)[])]));
  };

  const chosenOptions = (): Record<string, unknown> =>
    Object.fromEntries([...touched].map((key) => [key, options[key]]));

  const load = useCallback(
    async (silent = false) => {
      try {
        const list = await api.projects();
        setProjects(list.projects);
        setLoaded(true);
      } catch (error) {
        if (!silent) toast.fail(error, "Could not load your projects.");
      }
    },
    [toast],
  );

  useEffect(() => {
    load();
    api
      .settings()
      .then((payload) => {
        const initial = optionsFromSettings(payload.settings);
        setOptions(initial);
        setLengthKey(lengthKeyFor(initial));
        setUploadsEnabled(payload.settings.uploads_enabled !== false);
        setPresets(payload.caption_presets ?? {});
      })
      .catch(() => undefined); // the form still works with the defaults
  }, [load]);

  useLiveRefresh(() => load(true), {
    match: (event) => /^(project|job)\./.test(event.type),
    throttleMs: 1500,
    intervalMs: 10000,
  });

  const applyLength = (key: string) => {
    setLengthKey(key);
    const preset = LENGTHS[key];
    change({ min_clip_seconds: preset.min, target_clip_seconds: preset.target, max_clip_seconds: preset.max });
  };

  const submit = async () => {
    const link = url.trim();
    if (!link) {
      toast.push({ kind: "error", title: "Paste a video link first." });
      return;
    }
    setBusy(true);
    try {
      const result = await api.createProject({ url: link, options: chosenOptions() });
      toast.ok("Analysis started", result.project.title);
      setUrl("");
      await load(true);
    } catch (error) {
      toast.fail(error, "Could not start the analysis.");
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    setUploadProgress(0);
    try {
      await api.uploadProject(file, chosenOptions(), file.name.replace(/\.[^.]+$/, ""), setUploadProgress);
      toast.ok("Upload received — analysis queued", file.name);
      await load(true);
    } catch (error) {
      toast.fail(error, "The upload could not be processed.");
    } finally {
      setBusy(false);
      setUploadProgress(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const remove = async (project: ProjectSummary) => {
    if (!window.confirm(`Delete “${project.title}” and its renders?`)) return;
    try {
      await api.deleteProject(project.id);
      toast.ok("Project deleted");
      await load(true);
    } catch (error) {
      toast.fail(error, "Could not delete the project.");
    }
  };

  return (
    <div className="space-y-6">
      <section className="card p-4">
        <div className="flex flex-col gap-2 sm:flex-row">
          <div className="relative flex-1">
            <Link2 size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-3" />
            <input
              className="input pl-9"
              placeholder="Paste a video link — YouTube, Vimeo, X, TikTok or a direct .mp4"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && !busy && submit()}
              spellCheck={false}
              aria-label="Video link"
            />
          </div>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy && uploadProgress === null ? <Loader2 size={15} className="animate-spin" /> : null}
            Find clips
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => fileInput.current?.click()}
            disabled={busy || !uploadsEnabled}
            title={uploadsEnabled ? "Analyse a video file from your computer" : "Uploads are disabled in Settings"}
          >
            <Upload size={15} />
            Upload
          </button>
          <input
            ref={fileInput}
            type="file"
            accept={ACCEPTED_VIDEO}
            hidden
            onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])}
          />
        </div>

        {uploadProgress !== null ? (
          <div className="mt-3 flex items-center gap-2">
            <ProgressBar value={uploadProgress} className="max-w-sm" />
            <span className="text-[11px] text-text-3">
              {uploadProgress < 1 ? `Uploading · ${Math.round(uploadProgress * 100)}%` : "Upload complete — preparing the project…"}
            </span>
          </div>
        ) : null}

        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="block">
            <span className="label">Aspect ratio</span>
            <select className="select" value={options.aspect_ratio} onChange={(event) => change({ aspect_ratio: event.target.value })}>
              {RATIOS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="label">Caption style</span>
            <CaptionStyleSelect value={options.caption_preset} presets={presets} onChange={(value) => change({ caption_preset: value })} />
          </label>
          <label className="block">
            <span className="label">Clip length</span>
            <select className="select" value={lengthKey} onChange={(event) => applyLength(event.target.value)}>
              {Object.entries(LENGTHS).map(([key, preset]) => (
                <option key={key} value={key}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="label">Layout</span>
            <select className="select" value={options.layout} onChange={(event) => change({ layout: event.target.value })}>
              {LAYOUT_OPTIONS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          type="button"
          className="mt-3 inline-flex items-center gap-1 text-xs text-text-3 hover:text-text-2"
          onClick={() => setAdvanced(!advanced)}
          aria-expanded={advanced}
        >
          <ChevronDown size={13} className={`transition-transform ${advanced ? "" : "-rotate-90"}`} />
          Advanced
        </button>

        {advanced ? (
          <div className="mt-3 grid gap-3 border-t border-line pt-3 sm:grid-cols-2 lg:grid-cols-3">
            <Toggle
              label="Remove silences"
              hint="Tighten long pauses"
              value={options.remove_silence}
              onChange={(value) => change({ remove_silence: value })}
            />
            <Toggle
              label="Burn in captions"
              hint="Word-level subtitles on the video"
              value={options.captions_enabled}
              onChange={(value) => change({ captions_enabled: value })}
            />
            <Toggle
              label="Punch-ins"
              hint="Small zoom on the strongest lines"
              value={options.auto_zoom}
              onChange={(value) => change({ auto_zoom: value })}
            />
            <Toggle
              label="Translate to English"
              hint="Whisper translates speech in any language"
              value={options.translate_captions}
              onChange={(value) => change({ translate_captions: value })}
            />
            <Toggle
              label="Local LLM review"
              hint="Ollama, if it is reachable"
              value={options.llm_enabled}
              onChange={(value) => change({ llm_enabled: value })}
            />
            <label className="block">
              <span className="label">Minimum score · {options.min_score}</span>
              <input
                type="range"
                min={40}
                max={95}
                value={options.min_score}
                onChange={(event) => change({ min_score: Number(event.target.value) })}
                className="w-full accent-accent"
              />
            </label>
            <label className="block">
              <span className="label">How many clips</span>
              <select className="select" value={options.clip_mode} onChange={(event) => change({ clip_mode: event.target.value })}>
                <option value="best">Only the strongest</option>
                <option value="balanced">Balanced</option>
                <option value="max">Everything publishable</option>
              </select>
            </label>
            {lengthKey === "custom" ? (
              <div className="sm:col-span-2">
                <span className="label">Length range — minimum / target / maximum (seconds)</span>
                <div className="flex gap-2">
                  <input
                    className="input"
                    type="number"
                    min={10}
                    max={120}
                    aria-label="Minimum seconds"
                    value={options.min_clip_seconds}
                    onChange={(event) => change({ min_clip_seconds: Number(event.target.value) })}
                  />
                  <input
                    className="input"
                    type="number"
                    min={15}
                    max={180}
                    aria-label="Target seconds"
                    value={options.target_clip_seconds}
                    onChange={(event) => change({ target_clip_seconds: Number(event.target.value) })}
                  />
                  <input
                    className="input"
                    type="number"
                    min={20}
                    max={300}
                    aria-label="Maximum seconds"
                    value={options.max_clip_seconds}
                    onChange={(event) => change({ max_clip_seconds: Number(event.target.value) })}
                  />
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="space-y-2">
        <div className="flex items-baseline justify-between">
          <h2 className="panel-title">Projects</h2>
          <span className="text-xs text-text-3">{projects.length}</span>
        </div>

        {projects.length === 0 ? (
          <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
            <Plus size={20} className="text-text-3" />
            <p className="mt-1 text-sm font-medium text-text-2">{loaded ? "No projects yet" : "Loading projects…"}</p>
            <p className="max-w-sm text-xs text-text-3">
              Paste a link or upload a video above. A one-hour episode takes a few minutes to transcribe and analyse.
            </p>
          </div>
        ) : (
          <ul className="card divide-y divide-line overflow-hidden">
            {projects.map((project) => {
              const chip = statusChip(project.status);
              const working = isAnalysing(project.status);
              return (
                <li key={project.id} className="flex items-center gap-3 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <Link href={`/projects/${project.id}`} className="block truncate text-sm font-medium text-text hover:text-accent">
                      {project.title}
                    </Link>
                    <p className="mt-0.5 flex flex-wrap items-center gap-x-2.5 text-[11px] text-text-3">
                      <span>{when(project.created_at)}</span>
                      {project.duration ? <span>{clock(project.duration, true)}</span> : null}
                      {project.clip_count ? <span>{project.clip_count} clips</span> : null}
                      {project.rendered_clips ? <span>{project.rendered_clips} rendered</span> : null}
                      {project.language ? <span className="uppercase">{project.language}</span> : null}
                    </p>
                    {working ? (
                      <div className="mt-2 flex items-center gap-2">
                        <ProgressBar value={project.progress} className="max-w-xs" />
                        <span className="truncate text-[11px] text-text-3">
                          {Math.round((project.progress || 0) * 100)}% · {project.status_message || project.stage}
                        </span>
                      </div>
                    ) : null}
                    {project.status === "failed" && project.error?.message ? (
                      <p className="mt-1 truncate text-[11px] text-bad" title={project.error.hint || project.error.message}>
                        {project.error.message}
                      </p>
                    ) : null}
                  </div>
                  <span className={chip.className}>{chip.label}</span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon"
                    onClick={() => remove(project)}
                    aria-label={`Delete ${project.title}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}
