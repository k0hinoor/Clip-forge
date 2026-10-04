"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, FileText, FolderOpen, Loader2, Play, Square } from "lucide-react";
import { api, subscribeEvents } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar, Spinner, StageList } from "@/components/Progress";
import { clock, duration, scoreTone, statusChip } from "@/lib/format";
import type { Candidate, ClipSummary, ProjectStatus, Transcript } from "@/lib/types";

type Tab = "clips" | "transcript" | "rejected";

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
        const hasClips = statusPayload.project.clip_count > 0 || ["ready", "failed", "cancelled"].includes(statusPayload.project.status);
        if (hasClips) {
          const [clipPayload, candidatePayload] = await Promise.all([
            api.projectClips(projectId, sort),
            api.candidates(projectId, 200).catch(() => ({ candidates: [], counts: {} })),
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
    const unsubscribe = subscribeEvents(
      (event) => {
        if (event.type.startsWith("analysis") || event.type.startsWith("clip") || event.type.startsWith("job")) {
          load(true);
        }
      },
      projectId,
    );
    return () => {
      window.clearInterval(timer);
      unsubscribe();
    };
  }, [load, projectId]);

  const project = status?.project;
  const analysing = project?.status === "analyzing" || project?.status === "queued";
  const chosen = useMemo(() => clips.filter((clip) => selected[clip.id]).map((clip) => clip.id), [clips, selected]);

  const analyse = async () => {
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
  };

  const cancel = async () => {
    try {
      await api.cancelProject(projectId);
      toast.ok("Cancelling");
    } catch (error) {
      toast.fail(error);
    }
  };

  const renderAll = async () => {
    setBusy(true);
    try {
      const result = await api.renderAll(projectId, chosen.length ? chosen : undefined);
      toast.ok(`${result.queued.length} renders queued`);
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const openFolder = async () => {
    try {
      await api.openFolder(projectId);
    } catch (error) {
      toast.fail(error);
    }
  };

  const attachTranscript = async (file: File) => {
    try {
      const result = await api.uploadTranscript(projectId, file);
      toast.ok("Transcript attached", result.message);
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const openTranscript = () => {
    setTab("transcript");
    if (!transcript) {
      api
        .transcript(projectId, true)
        .then(setTranscript)
        .catch((error) => toast.fail(error, "The transcript is not ready yet."));
    }
  };

  const stages = (status?.stages ?? []).map((stage) => ({
    key: stage.key,
    label: stage.label,
    detail: stage.detail,
    state: stage.state ?? (stage.status === "active" ? "running" : stage.status === "skipped" ? "done" : stage.status ?? "pending"),
    progress: stage.progress,
  }));

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <Link href="/" className="inline-flex items-center gap-1 text-xs text-text-3 hover:text-text-2">
            <ArrowLeft size={12} /> Studio
          </Link>
          <h1 className="mt-1 truncate text-lg font-semibold tracking-tight text-text">{project?.title ?? "Loading…"}</h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2.5 text-xs text-text-3">
            {project?.duration ? <span>{clock(project.duration, true)} source</span> : null}
            {project?.word_count ? <span>{project.word_count.toLocaleString()} words</span> : null}
            {project?.speakers ? <span>{project.speakers} speakers</span> : null}
            {project?.language ? <span>{project.language}</span> : null}
            {project?.clip_count ? <span>{project.clip_count} clips</span> : null}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {analysing ? (
            <button type="button" className="btn btn-secondary" onClick={cancel}>
              <Square size={13} /> Cancel
            </button>
          ) : (
            <button type="button" className="btn btn-secondary" onClick={analyse} disabled={busy}>
              {project?.clip_count ? "Re-analyse" : "Analyse"}
            </button>
          )}
          <button type="button" className="btn btn-ghost" onClick={openFolder}>
            <FolderOpen size={14} /> Folder
          </button>
          <button type="button" className="btn btn-primary" onClick={renderAll} disabled={busy || clips.length === 0}>
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={13} />}
            Render {chosen.length ? `${chosen.length} selected` : "all"}
          </button>
        </div>
      </header>

      {project?.error_message ? (
        <div className="card border-bad/40 px-4 py-3">
          <p className="text-[13px] font-medium text-bad">{project.error_message}</p>
          {project.error_code ? <p className="mt-0.5 text-[11px] text-text-3">{project.error_code}</p> : null}
        </div>
      ) : null}

      <section className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <h2 className="panel-title">Progress</h2>
            <span className="text-xs text-text-3">{Math.round((project?.progress ?? 0) * 100)}%</span>
          </div>
          <div className="mt-3">
            <ProgressBar value={project?.progress ?? 0} />
            <p className="mt-1.5 text-xs text-text-3">{project?.status_message || project?.stage || "waiting"}</p>
          </div>
          {analysing || project?.status === "ready" ? (
            <div className="mt-4 border-t border-line pt-3">
              <StageList stages={stages} />
            </div>
          ) : null}
        </div>

        <div className="card space-y-3 p-4">
          <h2 className="panel-title">Source</h2>
          {project && ["ready", "analyzing", "rendering"].includes(project.status) ? (
            <video className="w-full rounded-md border border-line" controls preload="metadata" src={`/api/projects/${projectId}/source`} />
          ) : (
            <p className="text-xs text-text-3">The source appears here once the download completes.</p>
          )}
          <div>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => fileInput.current?.click()}>
              <FileText size={13} /> Attach transcript
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".srt,.vtt,.json3,.txt"
              hidden
              onChange={(event) => event.target.files?.[0] && attachTranscript(event.target.files[0])}
            />
            <p className="mt-2 text-[11px] leading-relaxed text-text-3">
              Have accurate captions already? Attach an .srt, .vtt, .json3 or timestamped .txt and it is used instead of
              speech recognition.
            </p>
          </div>
        </div>
      </section>

      <nav className="flex items-center gap-1 border-b border-line">
        {(
          [
            ["clips", `Clips ${clips.length ? `(${clips.length})` : ""}`],
            ["transcript", "Transcript"],
            ["rejected", `Considered ${candidates.length ? `(${candidates.length})` : ""}`],
          ] as [Tab, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => (key === "transcript" ? openTranscript() : setTab(key))}
            className={`-mb-px border-b-2 px-3 py-2 text-[13px] font-medium transition-colors ${
              tab === key ? "border-accent text-text" : "border-transparent text-text-3 hover:text-text-2"
            }`}
          >
            {label}
          </button>
        ))}
        {tab === "clips" ? (
          <select className="select ml-auto my-1 max-w-[10rem]" value={sort} onChange={(event) => setSort(event.target.value)}>
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
            title="No clips yet"
            body="Run the analysis. Clipforge only keeps moments it can justify from the transcript — the Considered tab shows what missed the threshold."
          />
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {clips.map((clip) => {
              const chip = statusChip(clip.status);
              return (
                <li key={clip.id} className="card overflow-hidden">
                  <div className="relative aspect-[9/16]">
                    {clip.status === "rendered" ? (
                      <video className="h-full w-full object-cover" controls preload="metadata" src={`/api/clips/${clip.id}/file`} />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center gap-2 bg-surface-2 text-center">
                        <Play size={18} className="text-text-3" />
                        <p className="text-[11px] text-text-3">
                          {clip.status === "rendering" ? `Rendering · ${Math.round((clip.progress ?? 0) * 100)}%` : "Not rendered"}
                        </p>
                        {clip.status === "rendering" ? (
                          <div className="w-2/3">
                            <ProgressBar value={clip.progress ?? 0} />
                          </div>
                        ) : null}
                      </div>
                    )}
                    <label className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-1.5 py-1 text-[11px] text-white">
                      <input
                        type="checkbox"
                        checked={Boolean(selected[clip.id])}
                        onChange={(event) => setSelected({ ...selected, [clip.id]: event.target.checked })}
                        className="accent-accent"
                      />
                      select
                    </label>
                    <span className="absolute right-2 top-2 chip">{chip.label}</span>
                  </div>

                  <div className="space-y-2 p-3">
                    <div className="flex items-baseline gap-2">
                      <span className="mono text-[11px] text-text-3">
                        {clock(clip.start)}–{clock(clip.end)}
                      </span>
                      <span className="mono ml-auto text-[13px] font-semibold tabular-nums text-text">
                        {clip.score.toFixed(0)}
                      </span>
                    </div>
                    <p className="line-clamp-2 text-[13px] font-medium leading-snug text-text">{clip.title}</p>
                    <p className="line-clamp-2 text-[11px] leading-relaxed text-text-3">“{clip.hook}”</p>
                    <div className="flex items-center gap-2 text-[11px] text-text-3">
                      <span className="chip">{clip.category_label || clip.category}</span>
                      <span>{duration(clip.duration)}</span>
                    </div>
                    <div className="flex gap-2 pt-1">
                      <Link href={`/projects/${projectId}/clips/${clip.id}`} className="btn btn-secondary btn-sm flex-1">
                        Edit
                      </Link>
                      {clip.status === "rendered" ? (
                        <a className="btn btn-ghost btn-sm" href={`/api/clips/${clip.id}/file?download=true`}>
                          MP4
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
            <p className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-text-3">
              <span className="chip">{transcript.engine || "unknown engine"}</span>
              {transcript.model ? <span>{transcript.model}</span> : null}
              <span>{transcript.word_count.toLocaleString()} words</span>
              <span>{transcript.language_name}</span>
              {transcript.language_mode && transcript.language_mode !== "monolingual" ? (
                <span className="chip chip-warn">{transcript.language_mode}</span>
              ) : null}
            </p>
            <ol className="space-y-2">
              {transcript.segments.map((segment) => (
                <li key={segment.index} className="grid grid-cols-[4.5rem_1fr] gap-3 text-[13px]">
                  <span className="mono pt-0.5 text-[11px] text-text-3">{clock(segment.start)}</span>
                  <span>
                    {segment.speaker ? (
                      <span className="mr-2 text-[11px] font-medium text-accent">{segment.speaker}</span>
                    ) : null}
                    <span className="leading-relaxed text-text-2">{segment.text}</span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ) : (
          <Spinner label="Loading transcript…" />
        )
      ) : null}

      {tab === "rejected" ? (
        candidates.length === 0 ? (
          <Empty title="Nothing analysed yet" body="Every window the analyser scored appears here, including the ones that did not make the cut." />
        ) : (
          <ul className="card divide-y divide-line overflow-hidden">
            {candidates.map((candidate) => (
              <li key={candidate.id} className="px-4 py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="mono text-[11px] text-text-3">
                    {clock(candidate.start)}–{clock(candidate.end)} · {duration(candidate.duration)}
                  </span>
                  <span className={`mono text-[13px] font-semibold ${scoreTone(candidate.score)}`}>{candidate.score.toFixed(0)}</span>
                  <span className="chip">{candidate.category_label || candidate.category}</span>
                  <span className={`chip ${candidate.status === "selected" ? "chip-good" : ""}`}>{candidate.status}</span>
                </div>
                <p className="mt-1 text-[13px] text-text-2">{candidate.title}</p>
                <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-text-3">
                  {candidate.transcript_text || candidate.summary}
                </p>
                {candidate.reason ? <p className="mt-1 text-[11px] italic text-text-3">{candidate.reason}</p> : null}
              </li>
            ))}
          </ul>
        )
      ) : null}
    </div>
  );
}

function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
      <p className="text-sm font-medium text-text-2">{title}</p>
      <p className="max-w-md text-xs text-text-3">{body}</p>
    </div>
  );
}
