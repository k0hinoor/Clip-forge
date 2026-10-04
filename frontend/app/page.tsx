"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Link2, Loader2, Plus, Trash2, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { clock, statusChip, when } from "@/lib/format";
import type { ProjectSummary } from "@/lib/types";

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

const LAYOUTS = [
  ["podcast", "Full frame"],
  ["blur", "Blurred background"],
  ["cinematic", "Cinematic crop"],
  ["split", "Split screen"],
  ["gameplay", "Gameplay background"],
  ["broll", "B-roll split"],
];

const CAPTION_STYLES = [
  ["bold_creator", "Bold (Shorts classic)"],
  ["minimal", "Minimal"],
  ["karaoke", "Karaoke"],
  ["cinematic", "Cinematic"],
  ["highlight", "Highlight"],
  ["documentary", "Documentary"],
];

export default function StudioPage() {
  const toast = useToast();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [lengthKey, setLengthKey] = useState("standard");
  const fileInput = useRef<HTMLInputElement>(null);

  const [options, setOptions] = useState({
    aspect_ratio: "9:16",
    caption_preset: "bold_creator",
    layout: "podcast",
    split_ratio: 65,
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
  });

  const load = useCallback(
    async (silent = false) => {
      try {
        const list = await api.projects();
        setProjects(list.projects);
      } catch (error) {
        if (!silent) toast.fail(error, "Could not load your projects.");
      }
    },
    [toast],
  );

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 6000);
    return () => window.clearInterval(timer);
  }, [load]);

  const applyLength = (key: string) => {
    setLengthKey(key);
    const preset = LENGTHS[key];
    setOptions((current) => ({
      ...current,
      min_clip_seconds: preset.min,
      target_clip_seconds: preset.target,
      max_clip_seconds: preset.max,
    }));
  };

  const submit = async () => {
    const link = url.trim();
    if (!link) {
      toast.push({ kind: "error", title: "Paste a video link first." });
      return;
    }
    setBusy(true);
    try {
      const result = await api.createProject({ url: link, options });
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
    try {
      await api.uploadProject(file, options, file.name.replace(/\.[^.]+$/, ""));
      toast.ok("Upload received", file.name);
      await load(true);
    } catch (error) {
      toast.fail(error, "The upload could not be processed.");
    } finally {
      setBusy(false);
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
              onKeyDown={(event) => event.key === "Enter" && submit()}
              spellCheck={false}
            />
          </div>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? <Loader2 size={15} className="animate-spin" /> : null}
            Find clips
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => fileInput.current?.click()} disabled={busy}>
            <Upload size={15} />
            Upload
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="video/*,audio/*"
            hidden
            onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])}
          />
        </div>

        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <span className="label">Aspect ratio</span>
            <select
              className="select"
              value={options.aspect_ratio}
              onChange={(event) => setOptions({ ...options, aspect_ratio: event.target.value })}
            >
              {RATIOS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <span className="label">Caption style</span>
            <select
              className="select"
              value={options.caption_preset}
              onChange={(event) => setOptions({ ...options, caption_preset: event.target.value })}
            >
              {CAPTION_STYLES.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <span className="label">Clip length</span>
            <select className="select" value={lengthKey} onChange={(event) => applyLength(event.target.value)}>
              {Object.entries(LENGTHS).map(([key, preset]) => (
                <option key={key} value={key}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <span className="label">Layout</span>
            <select
              className="select"
              value={options.layout}
              onChange={(event) => setOptions({ ...options, layout: event.target.value })}
            >
              {LAYOUTS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="button"
          className="mt-3 inline-flex items-center gap-1 text-xs text-text-3 hover:text-text-2"
          onClick={() => setAdvanced(!advanced)}
        >
          <ChevronDown size={13} className={`transition-transform ${advanced ? "" : "-rotate-90"}`} />
          Advanced
        </button>

        {advanced ? (
          <div className="mt-3 grid gap-3 border-t border-line pt-3 sm:grid-cols-2 lg:grid-cols-3">
            <Toggle
              label="Remove silences"
              hint="Tighten pauses over 0.35 s"
              value={options.remove_silence}
              onChange={(value) => setOptions({ ...options, remove_silence: value })}
            />
            <Toggle
              label="Burn in captions"
              hint="Word-level subtitles on the video"
              value={options.captions_enabled}
              onChange={(value) => setOptions({ ...options, captions_enabled: value })}
            />
            <Toggle
              label="Punch-ins"
              hint="Small zoom on the strongest lines"
              value={options.auto_zoom}
              onChange={(value) => setOptions({ ...options, auto_zoom: value })}
            />
            <Toggle
              label="Translate captions"
              hint="Off: keep the spoken language"
              value={options.translate_captions}
              onChange={(value) => setOptions({ ...options, translate_captions: value })}
            />
            <Toggle
              label="Local LLM review"
              hint="Ollama, if it is running"
              value={options.llm_enabled}
              onChange={(value) => setOptions({ ...options, llm_enabled: value })}
            />
            <div>
              <span className="label">Minimum score · {options.min_score}</span>
              <input
                type="range"
                min={50}
                max={95}
                value={options.min_score}
                onChange={(event) => setOptions({ ...options, min_score: Number(event.target.value) })}
                className="w-full accent-accent"
              />
            </div>
            <div>
              <span className="label">How many clips</span>
              <select
                className="select"
                value={options.clip_mode}
                onChange={(event) => setOptions({ ...options, clip_mode: event.target.value })}
              >
                <option value="best">Only the strongest</option>
                <option value="balanced">Balanced</option>
                <option value="max">Everything publishable</option>
              </select>
            </div>
            {lengthKey === "custom" ? (
              <div className="sm:col-span-2">
                <span className="label">Length range (seconds)</span>
                <div className="flex gap-2">
                  <input
                    className="input"
                    type="number"
                    min={10}
                    max={120}
                    value={options.min_clip_seconds}
                    onChange={(event) => setOptions({ ...options, min_clip_seconds: Number(event.target.value) })}
                  />
                  <input
                    className="input"
                    type="number"
                    min={15}
                    max={180}
                    value={options.target_clip_seconds}
                    onChange={(event) => setOptions({ ...options, target_clip_seconds: Number(event.target.value) })}
                  />
                  <input
                    className="input"
                    type="number"
                    min={20}
                    max={240}
                    value={options.max_clip_seconds}
                    onChange={(event) => setOptions({ ...options, max_clip_seconds: Number(event.target.value) })}
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
            <p className="mt-1 text-sm font-medium text-text-2">No projects yet</p>
            <p className="max-w-sm text-xs text-text-3">
              Paste a link above. Analysis runs on this machine; a one-hour episode takes a few minutes to transcribe.
            </p>
          </div>
        ) : (
          <ul className="card divide-y divide-line overflow-hidden">
            {projects.map((project) => {
              const chip = statusChip(project.status);
              const working = project.status === "analyzing" || project.status === "queued";
              return (
                <li key={project.id} className="flex items-center gap-3 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <Link
                      href={`/projects/${project.id}`}
                      className="block truncate text-sm font-medium text-text hover:text-accent"
                    >
                      {project.title}
                    </Link>
                    <p className="mt-0.5 flex flex-wrap items-center gap-x-2.5 text-[11px] text-text-3">
                      <span>{when(project.created_at)}</span>
                      {project.duration ? <span>{clock(project.duration, true)}</span> : null}
                      {project.clip_count ? <span>{project.clip_count} clips</span> : null}
                      {project.rendered_clips ? <span>{project.rendered_clips} rendered</span> : null}
                      {project.language ? <span>{project.language}</span> : null}
                    </p>
                    {working ? (
                      <div className="mt-2 flex items-center gap-2">
                        <ProgressBar value={project.progress} className="max-w-xs" />
                        <span className="truncate text-[11px] text-text-3">
                          {Math.round((project.progress || 0) * 100)}% · {project.status_message || project.stage}
                        </span>
                      </div>
                    ) : null}
                    {project.error_message ? (
                      <p className="mt-1 truncate text-[11px] text-bad">{project.error_message}</p>
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

function Toggle({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className="flex items-center justify-between gap-3 rounded-md border border-line bg-surface-2 px-3 py-2 text-left"
      aria-pressed={value}
    >
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-text">{label}</span>
        {hint ? <span className="block truncate text-[11px] text-text-3">{hint}</span> : null}
      </span>
      <span
        className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${value ? "bg-accent" : "bg-line-strong"}`}
      >
        <span
          className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${value ? "left-3.5" : "left-0.5"}`}
        />
      </span>
    </button>
  );
}
