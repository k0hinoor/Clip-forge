"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  FileText,
  FolderOpen,
  ListVideo,
  Play,
  RefreshCw,
  Scissors,
  Sparkles,
  Square,
  XCircle,
} from "lucide-react";
import { api, subscribeEvents } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar, Spinner, StageList } from "@/components/Progress";
import { clock, duration, scoreTone, statusChip } from "@/lib/format";
import type { Candidate, ClipSummary, ProjectStatus, Transcript } from "@/lib/types";

type Tab = "clips" | "transcript" | "candidates";

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;
  const toast = useToast();

  const [status, setStatus] = useState<ProjectStatus | null>(null);
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [tab, setTab] = useState<Tab>("clips");
  const [sort, setSort] = useState("score");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(
    async (silent = false) => {
      try {
        const statusPayload = await api.projectStatus(projectId);
        setStatus(statusPayload);
        if (statusPayload.project.clip_count > 0 || ["ready", "failed", "cancelled"].includes(statusPayload.project.status)) {
          const [clipPayload, candidatePayload] = await Promise.all([
            api.projectClips(projectId, sort),
            api.candidates(projectId, 200).catch(() => ({ candidates: [], stats: null })),
          ]);
          setClips(clipPayload.clips);
          setCandidates(candidatePayload.candidates);
        }
      } catch (error) {
        if (!silent) toast.fail(error, "Could not load this project.");
      }
    },
    [projectId, sort, toast],
  );

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 5000);
    const unsubscribe = subscribeEvents((event) => {
      if (event.type.startsWith("analysis") || event.type.startsWith("clip") || event.type.startsWith("job")) {
        load(true);
      }
    }, projectId);
    return () => {
      window.clearInterval(timer);
      unsubscribe();
    };
  }, [load, projectId]);

  const loadTranscript = async () => {
    if (transcript) return;
    try {
      setTranscript(await api.transcript(projectId, true));
    } catch (error) {
      toast.fail(error, "The transcript is not ready yet.");
    }
  };

  const project = status?.project;
  const analysing = project?.status === "analyzing" || project?.status === "queued";
  const clipIds = useMemo(() => clips.map((clip) => clip.id), [clips]);
  const chosen = useMemo(() => clipIds.filter((id) => selected[id]), [clipIds, selected]);

  const actions = {
    async analyse() {
      setBusy(true);
      try {
        await api.analyze(projectId);
        toast.ok("Analysis queued");
        await load(true);
      } catch (error) {
        toast.fail(error);
      } finally {
        setBusy(false);
      }
    },
    async cancel() {
      try {
        await api.cancelProject(projectId);
        toast.ok("Cancelling…", "Running jobs stop at the next safe point.");
      } catch (error) {
        toast.fail(error);
      }
    },
    async renderAll() {
      setBusy(true);
      try {
        const result = await api.renderAll(projectId, chosen.length ? chosen : undefined);
        toast.ok(`${result.queued.length} renders queued`, result.skipped.length ? `${result.skipped.length} were already queued.` : undefined);
        await load(true);
      } catch (error) {
        toast.fail(error);
      } finally {
        setBusy(false);
      }
    },
    async openFolder() {
      try {
        await api.openFolder(projectId);
        toast.ok("Opened the render folder");
      } catch (error) {
        toast.fail(error);
      }
    },
    async attachTranscript(file: File) {
      try {
        const result = await api.uploadTranscript(projectId, file);
        toast.ok("Transcript attached", result.message);
        await load(true);
      } catch (error) {
        toast.fail(error);
      } finally {
        if (fileInput.current) fileInput.current.value = "";
      }
    },
  };

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <Link href="/" className="text-[11px] font-semibold uppercase tracking-widest text-mist-400 hover:text-flare-400">
            ← Studio
          </Link>
          <h1 className="mt-1 truncate text-xl font-black text-mist-200">{project?.title ?? "Loading…"}</h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-mist-400">
            {project?.duration ? <span>{clock(project.duration, true)} source</span> : null}
            {project?.word_count ? <span>{project.word_count.toLocaleString()} words</span> : null}
            {project?.speakers ? <span>{project.speakers} speaker{project.speakers > 1 ? "s" : ""}</span> : null}
            {project?.language ? (
              <span>
                {project.language}
                {project.language_mode && project.language_mode !== "monolingual" ? ` (${project.language_mode})` : ""}
              </span>
            ) : null}
            {project?.clip_count ? <span>{project.clip_count} clips</span> : null}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {analysing ? (
            <button type="button" className="btn btn-ghost" onClick={actions.cancel}>
              <Square size={14} /> Cancel
            </button>
          ) : (
            <button type="button" className="btn btn-ghost" onClick={actions.analyse} disabled={busy}>
              <RefreshCw size={14} /> {project?.clip_count ? "Re-analyse" : "Analyse"}
            </button>
          )}
          <button type="button" className="btn btn-ghost" onClick={actions.openFolder}>
            <FolderOpen size={14} /> Folder
          </button>
          <button type="button" className="btn btn-primary" onClick={actions.renderAll} disabled={busy || clips.length === 0}>
            {busy ? <Spinner /> : <ListVideo size={14} />}
            Render {chosen.length ? `${chosen.length} selected` : "all"}
          </button>
        </div>
      </header>

      {project?.error_message ? (
        <div className="card border-flare-500/40 bg-flare-500/5 p-4">
          <p className="flex items-center gap-2 text-sm font-semibold text-flare-400">
            <XCircle size={15} /> {project.error_message}
          </p>
          {project.error_code ? <p className="mt-1 text-[11px] text-mist-400">Error code: {project.error_code}</p> : null}
        </div>
      ) : null}

      <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="card p-4">
          <h2 className="text-sm font-bold text-mist-200">Progress</h2>
          <div className="mt-3 space-y-3">
            <ProgressBar value={project?.progress ?? 0} />
            <p className="text-xs text-mist-400">
              {Math.round((project?.progress ?? 0) * 100)}% · {project?.status_message || project?.stage || "waiting"}
            </p>
            {status ? <StageList stages={status.stages} /> : null}
          </div>
        </div>

        <div className="card space-y-3 p-4">
          <h2 className="text-sm font-bold text-mist-200">Source</h2>
          {project && ["ready", "analyzing", "rendering"].includes(project.status) ? (
            <video
              className="w-full rounded-lg border border-ink-700 bg-black"
              controls
              preload="metadata"
              src={`/api/projects/${projectId}/source`}
            />
          ) : (
            <p className="text-xs text-mist-400">
              The source video appears here once the download completes.
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <button type="button" className="btn btn-ghost text-xs" onClick={() => fileInput.current?.click()}>
              <FileText size={13} /> Attach transcript file
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".srt,.vtt,.json3,.txt"
              hidden
              onChange={(event) => event.target.files?.[0] && actions.attachTranscript(event.target.files[0])}
            />
            <span className="chip">Cached: transcript · audio</span>
          </div>
          <p className="text-[11px] leading-relaxed text-mist-400">
            Already have captions? Attach an <span className="mono">.srt</span>, <span className="mono">.vtt</span>,{" "}
            <span className="mono">.json3</span> or timestamped <span className="mono">.txt</span> and CLIPFORGE will use
            it instead of speech recognition.
          </p>
        </div>
      </section>

      <nav className="flex gap-1 border-b border-ink-700/70 pb-2">
        {(
          [
            ["clips", `Clips (${clips.length})`],
            ["transcript", "Transcript"],
            ["candidates", `Moments considered (${candidates.length})`],
          ] as [Tab, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={`btn ${tab === key ? "btn-ghost" : "btn-quiet"}`}
            onClick={() => {
              setTab(key);
              if (key === "transcript") loadTranscript();
            }}
          >
            {label}
          </button>
        ))}
        {tab === "clips" ? (
          <select className="select ml-auto max-w-[11rem]" value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="score">Best score</option>
            <option value="chronological">Chronological</option>
            <option value="duration">Longest</option>
            <option value="duration_asc">Shortest</option>
            <option value="category">Category</option>
          </select>
        ) : null}
      </nav>

      {tab === "clips" ? (
        clips.length === 0 ? (
          <Empty
            icon={<Scissors size={22} />}
            title="No clips yet"
            body="Run the analysis - CLIPFORGE only creates clips it can justify from the transcript."
          />
        ) : (
          <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {clips.map((clip) => {
              const chip = statusChip(clip.status);
              return (
                <li key={clip.id} className="card overflow-hidden">
                  <div className="relative aspect-[9/16] bg-ink-900">
                    {clip.status === "rendered" ? (
                      <video className="h-full w-full object-cover" controls preload="metadata" src={`/api/clips/${clip.id}/file`} />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center">
                        <Play size={20} className="text-mist-400" />
                        <p className="text-[11px] text-mist-400">{clip.status === "rendering" ? "Rendering…" : "Preview not rendered"}</p>
                        <Link href={`/projects/${projectId}/clips/${clip.id}`} className="btn btn-ghost text-xs">
                          Open editor
                        </Link>
                      </div>
                    )}
                    <label className="absolute left-2 top-2 flex items-center gap-1.5 rounded-md bg-ink-950/80 px-2 py-1 text-[11px]">
                      <input
                        type="checkbox"
                        checked={Boolean(selected[clip.id])}
                        onChange={(event) => setSelected({ ...selected, [clip.id]: event.target.checked })}
                        className="accent-flare-500"
                      />
                      select
                    </label>
                    <span className="absolute right-2 top-2 chip">{chip.label}</span>
                  </div>
                  <div className="space-y-2 p-3">
                    <div className="flex items-start gap-2">
                      <span className="mono text-[11px] text-mist-400">
                        #{String(clip.index).padStart(2, "0")} · {clock(clip.start)}–{clock(clip.end)}
                      </span>
                      <span className={`mono ml-auto text-sm font-bold ${scoreTone(clip.score)}`}>{clip.score.toFixed(1)}</span>
                    </div>
                    <p className="line-clamp-2 text-sm font-semibold leading-snug text-mist-200">{clip.title}</p>
                    <p className="line-clamp-2 text-[11px] italic leading-relaxed text-mist-400">“{clip.hook}”</p>
                    <div className="flex items-center gap-2 text-[10px] text-mist-400">
                      <span className="chip">{clip.category_label || clip.category}</span>
                      <span>{duration(clip.duration)}</span>
                    </div>
                    {clip.why?.length ? (
                      <ul className="space-y-0.5 text-[11px] text-mist-300">
                        {clip.why.slice(0, 2).map((reason, index) => (
                          <li key={index} className="flex gap-1.5">
                            <Sparkles size={11} className="mt-0.5 shrink-0 text-amber-glow" />
                            {reason}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                    <div className="flex gap-2 pt-1">
                      <Link href={`/projects/${projectId}/clips/${clip.id}`} className="btn btn-ghost flex-1 justify-center text-xs">
                        Edit reasons &amp; frame
                      </Link>
                      {clip.status === "rendered" ? (
                        <a className="btn btn-quiet p-2" href={`/api/clips/${clip.id}/file?download=true`} aria-label="Download">
                          <Download size={14} />
                        </a>
                      ) : null}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )
      ) : null}

      {tab === "transcript" ? (
        transcript ? (
          <div className="card max-h-[65vh] overflow-y-auto p-4 scroll-thin">
            <p className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-mist-400">
              <span className="chip">{transcript.engine || "unknown engine"}</span>
              <span>{transcript.model}</span>
              <span>{transcript.word_count.toLocaleString()} words</span>
              <span>{transcript.language_name}</span>
              {transcript.language_mode !== "monolingual" ? <span className="chip chip-warn">{transcript.language_mode}</span> : null}
            </p>
            <ol className="space-y-2">
              {transcript.segments.map((segment) => (
                <li key={segment.index} className="grid grid-cols-[5.5rem_1fr] gap-3 text-sm">
                  <span className="mono pt-0.5 text-[11px] text-mist-400">{clock(segment.start)}</span>
                  <span>
                    <span className="mr-2 align-top text-[10px] font-semibold uppercase tracking-wide text-flare-400">
                      {segment.speaker}
                    </span>
                    <span className="leading-relaxed text-mist-200">{segment.text}</span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ) : (
          <Spinner label="Loading transcript…" />
        )
      ) : null}

      {tab === "candidates" ? (
        candidates.length === 0 ? (
          <Empty icon={<Sparkles size={22} />} title="Nothing analysed yet" body="Candidate moments appear here with their scores and the reasons behind them." />
        ) : (
          <ul className="space-y-2">
            {candidates.map((candidate) => (
              <li key={candidate.id} className="card-tight p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="mono text-[11px] text-mist-400">
                    {clock(candidate.start)}–{clock(candidate.end)} · {duration(candidate.duration)}
                  </span>
                  <span className={`mono text-sm font-bold ${scoreTone(candidate.score)}`}>{candidate.score.toFixed(1)}</span>
                  <span className="chip">{candidate.category_label || candidate.category}</span>
                  <span className={`chip ${candidate.status === "selected" ? "chip-good" : candidate.status === "duplicate" ? "" : "chip-warn"}`}>
                    {candidate.status}
                  </span>
                </div>
                <p className="mt-1.5 text-sm text-mist-200">{candidate.title}</p>
                <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-mist-400">{candidate.transcript_text || candidate.summary}</p>
                {candidate.highlights?.length ? (
                  <p className="mt-1 text-[11px] text-mist-300">· {candidate.highlights.slice(0, 3).join(" · ")}</p>
                ) : null}
                {candidate.reason ? <p className="mt-1 text-[11px] italic text-mist-400">{candidate.reason}</p> : null}
              </li>
            ))}
          </ul>
        )
      ) : null}
    </div>
  );
}

function Empty({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="card flex flex-col items-center gap-2 p-10 text-center">
      <span className="text-mist-400">{icon}</span>
      <p className="text-sm font-semibold text-mist-300">{title}</p>
      <p className="max-w-md text-xs text-mist-400">{body}</p>
    </div>
  );
}
