"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Clapperboard,
  FileAudio,
  Gauge,
  Link2,
  Loader2,
  RefreshCw,
  Scissors,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
} from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { clock, duration, statusChip, when } from "@/lib/format";
import type { ProjectSummary, SystemStatus } from "@/lib/types";

const DEFAULTS = {
  clip_mode: "balanced",
  min_clip_seconds: 35,
  target_clip_seconds: 60,
  max_clip_seconds: 75,
  min_score: 70,
  aspect_ratio: "9:16",
  layout: "split",
  split_ratio: 65,
  gameplay_enabled: true,
  remove_silence: true,
  captions_enabled: true,
  llm_enabled: true,
  translate_captions: false,
};

export default function StudioPage() {
  const toast = useToast();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [url, setUrl] = useState("");
  const [options, setOptions] = useState(DEFAULTS);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [search, setSearch] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(
    async (silent = false) => {
      try {
        const [list, system] = await Promise.all([api.projects(search), api.status()]);
        setProjects(list.projects);
        setStatus(system);
      } catch (error) {
        if (!silent) toast.fail(error, "Could not load your projects.");
      }
    },
    [search, toast],
  );

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 8000);
    return () => window.clearInterval(timer);
  }, [load]);

  const startFromUrl = async () => {
    if (!url.trim()) {
      toast.push({ kind: "error", title: "Paste a YouTube link first." });
      return;
    }
    setBusy(true);
    try {
      const result = await api.createProject({ url: url.trim(), options });
      toast.ok("Analysis queued", `Project ${result.project.title.slice(0, 48)} is being processed locally.`);
      setUrl("");
      await load(true);
    } catch (error) {
      toast.fail(error, "Could not start the analysis.");
    } finally {
      setBusy(false);
    }
  };

  const startFromFile = async (file: File) => {
    setBusy(true);
    try {
      const result = await api.uploadProject(file, options, file.name.replace(/\.[^.]+$/, ""));
      toast.ok("Upload received", "Analysis is running on your machine.");
      await load(true);
    } catch (error) {
      toast.fail(error, "The upload could not be processed.");
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const remove = async (project: ProjectSummary) => {
    if (!window.confirm(`Delete “${project.title}” and all of its renders?`)) return;
    try {
      await api.deleteProject(project.id);
      toast.ok("Project deleted");
      await load(true);
    } catch (error) {
      toast.fail(error, "Could not delete the project.");
    }
  };

  const hardware = status?.hardware;
  const recommendation = (status as any)?.recommendation;

  return (
    <div className="space-y-8">
      <section className="card p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-black tracking-tight text-mist-200">Turn long videos into shorts</h1>
            <p className="mt-1 max-w-2xl text-sm leading-relaxed text-mist-400">
              Paste a YouTube link or drop a file. CLIPFORGE downloads it, transcribes it locally, finds every moment
              worth publishing, and renders vertical clips with captions - entirely on this machine.
            </p>
          </div>
          {hardware ? (
            <div className="chip">
              <Gauge size={13} />
              {hardware.cpu?.split(" ").slice(0, 3).join(" ") || "CPU"} · {hardware.ram_gb} GB RAM
              {hardware.gpu?.available ? ` · ${hardware.gpu.devices?.[0]?.name ?? hardware.gpu.vendor}` : " · CPU only"}
            </div>
          ) : null}
        </div>

        <div className="mt-5 grid gap-3 lg:grid-cols-[1fr_auto_auto]">
          <div className="relative">
            <Link2 size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-mist-400" />
            <input
              className="input pl-9"
              placeholder="https://www.youtube.com/watch?v=…"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && startFromUrl()}
            />
          </div>
          <button type="button" className="btn btn-primary" onClick={startFromUrl} disabled={busy}>
            {busy ? <Loader2 size={15} className="animate-spin" /> : <Wand2 size={15} />}
            Analyse &amp; cut
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => fileInput.current?.click()} disabled={busy}>
            <Upload size={15} /> Upload file
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="video/*,audio/*"
            hidden
            onChange={(event) => event.target.files?.[0] && startFromFile(event.target.files[0])}
          />
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Field label="Clip length (s)">
            <div className="flex gap-2">
              <input
                className="input"
                type="number"
                value={options.min_clip_seconds}
                min={10}
                max={120}
                onChange={(event) => setOptions({ ...options, min_clip_seconds: Number(event.target.value) })}
              />
              <input
                className="input"
                type="number"
                value={options.target_clip_seconds}
                min={15}
                max={180}
                onChange={(event) => setOptions({ ...options, target_clip_seconds: Number(event.target.value) })}
              />
              <input
                className="input"
                type="number"
                value={options.max_clip_seconds}
                min={20}
                max={240}
                onChange={(event) => setOptions({ ...options, max_clip_seconds: Number(event.target.value) })}
              />
            </div>
          </Field>
          <Field label={`Minimum score · ${options.min_score}`}>
            <input
              type="range"
              min={50}
              max={95}
              step={1}
              value={options.min_score}
              onChange={(event) => setOptions({ ...options, min_score: Number(event.target.value) })}
              className="w-full accent-flare-500"
            />
          </Field>
          <Field label="How many clips">
            <select
              className="select"
              value={options.clip_mode}
              onChange={(event) => setOptions({ ...options, clip_mode: event.target.value })}
            >
              <option value="best">Only the strongest</option>
              <option value="balanced">Balanced</option>
              <option value="max">Everything publishable</option>
            </select>
          </Field>
          <Field label="Aspect ratio">
            <select
              className="select"
              value={options.aspect_ratio}
              onChange={(event) => setOptions({ ...options, aspect_ratio: event.target.value })}
            >
              <option value="9:16">9:16 vertical</option>
              <option value="1:1">1:1 square</option>
              <option value="16:9">16:9 landscape</option>
            </select>
          </Field>
          <Field label="Layout">
            <select
              className="select"
              value={options.layout}
              onChange={(event) => setOptions({ ...options, layout: event.target.value })}
            >
              <option value="split">Split screen</option>
              <option value="podcast">Speaker full frame</option>
              <option value="broll">With B-roll</option>
              <option value="gameplay">Gameplay background</option>
              <option value="blur">Blurred background</option>
              <option value="cinematic">Cinematic</option>
            </select>
          </Field>
          <Field label="Split ratio">
            <select
              className="select"
              value={options.split_ratio}
              onChange={(event) => setOptions({ ...options, split_ratio: Number(event.target.value) })}
              disabled={!["split", "broll", "gameplay"].includes(options.layout)}
            >
              {[50, 60, 65, 70].map((ratio) => (
                <option key={ratio} value={ratio}>
                  {ratio} / {100 - ratio}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <button type="button" className="btn btn-quiet mt-2 px-0 text-xs" onClick={() => setAdvanced(!advanced)}>
          {advanced ? "Hide" : "Show"} more options
        </button>

        {advanced ? (
          <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div className="card-tight flex items-center justify-between p-3">
              <span className="text-xs text-mist-300">Remove silences</span>
              <Toggle value={options.remove_silence} onChange={(value) => setOptions({ ...options, remove_silence: value })} />
            </div>
            <div className="card-tight flex items-center justify-between p-3">
              <span className="text-xs text-mist-300">Burn in captions</span>
              <Toggle value={options.captions_enabled} onChange={(value) => setOptions({ ...options, captions_enabled: value })} />
            </div>
            <div className="card-tight flex items-center justify-between p-3">
              <span className="text-xs text-mist-300">Use local LLM</span>
              <Toggle value={options.llm_enabled} onChange={(value) => setOptions({ ...options, llm_enabled: value })} />
            </div>
            <div className="card-tight flex items-center justify-between p-3">
              <span className="text-xs text-mist-300">Add gameplay</span>
              <Toggle value={options.gameplay_enabled} onChange={(value) => setOptions({ ...options, gameplay_enabled: value })} />
            </div>
            <div className="card-tight flex items-center justify-between p-3">
              <span className="text-xs text-mist-300">Translate captions</span>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-mist-400">off by default</span>
                <Toggle value={options.translate_captions} onChange={(value) => setOptions({ ...options, translate_captions: value })} />
              </div>
            </div>
          </div>
        ) : null}
      </section>

      {recommendation?.note ? (
        <div className="card-tight flex flex-wrap items-center gap-2 p-3 text-xs text-mist-300">
          <Sparkles size={14} className="text-amber-glow" />
          {recommendation.note}
        </div>
      ) : null}

      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-lg font-bold text-mist-200">Your projects</h2>
          <input
            className="input ml-auto max-w-xs"
            placeholder="Search…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <button type="button" className="btn btn-quiet" onClick={() => load()} aria-label="Refresh">
            <RefreshCw size={15} />
          </button>
        </div>

        {projects.length === 0 ? (
          <div className="card flex flex-col items-center gap-2 p-10 text-center">
            <Clapperboard size={26} className="text-mist-400" />
            <p className="text-sm font-semibold text-mist-300">No projects yet</p>
            <p className="max-w-md text-xs text-mist-400">
              Paste a link above to create your first one. A 1-hour episode usually takes a few minutes to transcribe on
              CPU and a couple of minutes per clip to render.
            </p>
          </div>
        ) : (
          <ul className="grid gap-3 lg:grid-cols-2">
            {projects.map((project) => {
              const chip = statusChip(project.status);
              const active = project.active_job;
              return (
                <li key={project.id} className="card p-4">
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <Link href={`/projects/${project.id}`} className="block truncate font-semibold text-mist-200 hover:text-flare-400">
                        {project.title}
                      </Link>
                      <p className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-mist-400">
                        <span>{when(project.created_at)}</span>
                        {project.duration ? <span>{clock(project.duration, true)} long</span> : null}
                        {project.clip_count ? <span>{project.clip_count} clips</span> : null}
                        {project.rendered_clips ? <span>{project.rendered_clips} rendered</span> : null}
                        {project.language ? (
                          <span>
                            {project.language}
                            {project.language_mode !== "monolingual" ? ` · ${project.language_mode}` : ""}
                          </span>
                        ) : null}
                      </p>
                    </div>
                    <span className={chip.className}>{chip.label}</span>
                    <button type="button" className="btn btn-quiet p-1.5" onClick={() => remove(project)} aria-label="Delete">
                      <Trash2 size={14} />
                    </button>
                  </div>
                  {project.status === "analyzing" || project.status === "queued" ? (
                    <div className="mt-3 space-y-1.5">
                      <ProgressBar value={project.progress} />
                      <p className="text-[11px] text-mist-400">
                        {Math.round((project.progress || 0) * 100)}% · {project.status_message || project.stage}
                      </p>
                    </div>
                  ) : null}
                  {project.error_message ? (
                    <p className="mt-2 text-[11px] text-flare-400">{project.error_message}</p>
                  ) : null}
                  {project.word_count ? (
                    <p className="mt-2 flex items-center gap-3 text-[11px] text-mist-400">
                      <span className="inline-flex items-center gap-1">
                        <FileAudio size={12} /> {project.word_count.toLocaleString()} words
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Scissors size={12} /> {project.candidate_count} moments considered
                      </span>
                    </p>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="label">{label}</span>
      {children}
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (value: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${value ? "bg-signal-500/80" : "bg-ink-600"}`}
      aria-pressed={value}
    >
      <span
        className={`absolute top-0.5 h-4 w-4 rounded-full bg-mist-200 transition-all ${value ? "left-[1.15rem]" : "left-0.5"}`}
      />
    </button>
  );
}
