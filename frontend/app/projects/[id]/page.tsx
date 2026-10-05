"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ArrowLeft, Download, FileText, FolderOpen, Loader2, Mic, Play, RefreshCw, Square } from "lucide-react";
import { api } from "@/lib/api";
import { isProjectEvent, useLiveRefresh } from "@/lib/live";
import { useSystem } from "@/components/System";
import { useToast } from "@/components/Toast";
import { ProgressBar, Spinner, StageList } from "@/components/Progress";
import { clock, duration, isAnalysing, mediaUrl, scoreTone, statusChip } from "@/lib/format";
import type { Candidate, ClipSummary, ProjectStatus, Transcript } from "@/lib/types";

type Tab = "clips" | "transcript" | "rejected";

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;
  const toast = useToast();
  const { features } = useSystem();

  const [status, setStatus] = useState<ProjectStatus | null>(null);
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [tab, setTab] = useState<Tab>("clips");
  const [sort, setSort] = useState("score");
  const [busy, setBusy] = useState(false);
  const [missing, setMissing] = useState(false);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [debugMode, setDebugMode] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const loadStatus = useCallback(async () => {
    const payload = await api.projectStatus(projectId);
    setStatus(payload);
    return payload;
  }, [projectId]);

  const loadClips = useCallback(async () => {
    const [clipPayload, candidatePayload] = await Promise.all([
      api.projectClips(projectId, sort),
      api.candidates(projectId, 200).catch(() => ({ candidates: [] as Candidate[], stats: {} })),
    ]);
    setClips(clipPayload.clips);
    setCandidates(candidatePayload.candidates);
  }, [projectId, sort]);

  const load = useCallback(
    async (silent = false) => {
      try {
        await loadStatus();
        await loadClips();
      } catch (error: any) {
        if (error?.status === 404) setMissing(true);
        else if (!silent) toast.fail(error, "Could not load this project.");
      }
    },
    [loadStatus, loadClips, toast],
  );

  useEffect(() => {
    load();
  }, [load]);

  // Progress ticks only need the (cheap) status; anything else refreshes clips too.
  useLiveRefresh(
    (event) => {
      if (event?.type === "job.progress" && !event.clip_id) loadStatus().catch(() => undefined);
      else load(true);
    },
    { projectId, match: isProjectEvent, throttleMs: 1000, intervalMs: 15000 },
  );

  const project = status?.project;
  const analysing = isAnalysing(project?.status);
  const diagnostics = status?.candidates;
  const configuredThreshold = Number(project?.settings?.min_score ?? diagnostics?.threshold ?? 70);
  const chosen = useMemo(() => clips.filter((clip) => selected[clip.id]).map((clip) => clip.id), [clips, selected]);
  const unrendered = clips.filter((clip) => !["rendered", "queued", "rendering"].includes(clip.status)).length;

  const analyse = async (options?: Record<string, unknown>, force = false) => {
    if (clips.length && !window.confirm("Re-analysing replaces the current accepted clips and their edits. Continue?")) return;
    setBusy(true);
    try {
      await api.analyze(projectId, options, force);
      toast.ok("Analysis queued");
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const lowerThresholdAndReanalyse = async () => {
    const next = Math.max(0, configuredThreshold - 10);
    await analyse({ min_score: next, debug_mode: false });
  };

  const runDebugAnalysis = async () => {
    await analyse({ debug_mode: debugMode, ...(debugMode ? { min_score: 0 } : {}) });
  };

  const useWhisper = async () => {
    setBusy(true);
    try {
      const result = await api.useWhisper(projectId);
      toast.ok("Whisper selected", result.message);
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const retranscribeAnyway = async () => analyse(undefined, true);

  const cancel = async () => {
    try {
      const result = await api.cancelProject(projectId);
      toast.ok(result.cancelled_jobs ? "Cancelling…" : "Nothing is running for this project");
      await load(true);
    } catch (error) {
      toast.fail(error);
    }
  };

  const renderAll = async () => {
    setBusy(true);
    try {
      const result = await api.renderAll(projectId, chosen.length ? chosen : undefined, features.local_paths);
      if (result.queued) {
        toast.ok(`${result.queued} render${result.queued === 1 ? "" : "s"} queued`, result.skipped ? `${result.skipped} already rendered or queued` : undefined);
      } else {
        toast.ok("Everything is already rendered or queued");
      }
      setSelected({});
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
      setSelected({});
      setTab("transcript");
      toast.ok("Transcript attached", result.message);
      await Promise.all([
        load(true),
        api.transcript(projectId, false).then(setTranscript).catch(() => undefined),
      ]);
    } catch (error) {
      toast.fail(error);
    } finally {
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const openTranscript = () => {
    setTab("transcript");
    api
      .transcript(projectId, false)
      .then(setTranscript)
      .catch((error) => toast.fail(error, "The transcript is not ready yet."));
  };

  if (missing) {
    return (
      <div className="card flex flex-col items-center gap-2 px-6 py-12 text-center">
        <p className="text-sm font-medium text-text-2">This project does not exist</p>
        <p className="text-xs text-text-3">It may have been deleted.</p>
        <Link href="/" className="btn btn-secondary btn-sm mt-2">
          <ArrowLeft size={12} /> Back to the Studio
        </Link>
      </div>
    );
  }

  const stages = (status?.stages ?? []).map((stage) => ({
    key: stage.key,
    label: stage.label,
    detail: stage.detail ?? stage.message,
    state: stage.state ?? (stage.status === "active" ? "running" : stage.status === "skipped" ? "done" : stage.status ?? "pending"),
    progress: stage.progress,
  })) as { key: string; label: string; detail?: string; state: "pending" | "running" | "done" | "failed"; progress: number }[];

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <Link href="/" className="inline-flex items-center gap-1 text-xs text-text-3 hover:text-text-2">
            <ArrowLeft size={12} /> Studio
          </Link>
          <h1 className="mt-1 truncate text-lg font-semibold tracking-tight text-text">{project?.title ?? "Loading…"}</h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2.5 text-xs text-text-3">
            {project ? <span className={statusChip(project.status).className}>{statusChip(project.status).label}</span> : null}
            {project?.duration ? <span>{clock(project.duration, true)} source</span> : null}
            {project?.word_count ? <span>{project.word_count.toLocaleString()} words</span> : null}
            {project?.speakers ? <span>{project.speakers} speakers</span> : null}
            {project?.language_name || project?.language ? <span>{project.language_name || project.language}</span> : null}
            {project?.clip_count ? <span>{project.clip_count} clips</span> : null}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {analysing ? (
            <button type="button" className="btn btn-secondary" onClick={cancel}>
              <Square size={13} /> Cancel
            </button>
          ) : (
            <button type="button" className="btn btn-secondary" onClick={() => analyse()} disabled={busy || !project}>
              {project?.clip_count || project?.status === "ready" ? "Re-analyse" : "Analyse"}
            </button>
          )}
          {features.open_folder ? (
            <button type="button" className="btn btn-ghost" onClick={openFolder}>
              <FolderOpen size={14} /> Folder
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-primary"
            onClick={renderAll}
            disabled={busy || clips.length === 0 || (!chosen.length && unrendered === 0)}
            title={!chosen.length && unrendered === 0 ? "Every clip is rendered or queued" : undefined}
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={13} />}
            Render {chosen.length ? `${chosen.length} selected` : unrendered ? `${unrendered} remaining` : "all"}
          </button>
        </div>
      </header>

      {project?.status === "failed" && project.error?.message ? (
        <div className="card border-bad/40 px-4 py-3">
          <p className="text-[13px] font-medium text-bad">{project.error.message}</p>
          {project.error.hint ? <p className="mt-0.5 text-[11px] text-text-2">{project.error.hint}</p> : null}
          {project.error.code ? <p className="mono mt-0.5 text-[10px] text-text-3">{project.error.code}</p> : null}
        </div>
      ) : null}

      <section className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <h2 className="panel-title">Analysis</h2>
            <span className="text-xs text-text-3">{Math.round((project?.progress ?? 0) * 100)}%</span>
          </div>
          <div className="mt-3">
            <ProgressBar value={project?.progress ?? 0} />
            <p className="mt-1.5 text-xs text-text-3">{project?.status_message || project?.stage || "waiting"}</p>
          </div>
          {stages.length && project?.status !== "draft" ? (
            <div className="mt-4 border-t border-line pt-3">
              <StageList stages={stages} />
            </div>
          ) : null}
          {diagnostics && (diagnostics.discovered > 0 || analysing) ? (
            <div className="mt-4 grid grid-cols-2 gap-2 border-t border-line pt-3 text-[11px] sm:grid-cols-4">
              <Count label="Discovered" value={diagnostics.discovered} />
              <Count label="Scored" value={diagnostics.scored} />
              <Count label="Threshold pass" value={diagnostics.threshold_pass} />
              <Count label="Accepted" value={diagnostics.final_accepted} />
              <Count label="Scoring errors" value={diagnostics.scoring_errors} warn={diagnostics.scoring_errors > 0} />
              <Count label="Context rejected" value={diagnostics.context_rejections} />
              <Count label="Overlap rejected" value={diagnostics.overlap_rejections} />
              <Count label="Render queue / done" value={`${diagnostics.queued} / ${diagnostics.rendered}`} />
            </div>
          ) : null}
        </div>

        <div className="card space-y-3 p-4">
          <h2 className="panel-title">Source</h2>
          {project?.has_source ? (
            <video className="w-full rounded-md border border-line" controls preload="metadata" src={`/api/projects/${projectId}/source`} />
          ) : (
            <p className="text-xs text-text-3">The source appears here once it has been downloaded.</p>
          )}
          {project?.transcript_source ? (
            <div className="rounded-md border border-line bg-surface-2 px-3 py-2.5">
              <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <span className="chip chip-good">{transcriptSourceLabel(project.transcript_source)}</span>
                {project.transcript_filename ? <span className="text-text-2">{project.transcript_filename}</span> : null}
              </div>
              <p className="mt-1.5 text-[11px] text-text-3">
                {(project.segment_count ?? 0).toLocaleString()} segments · {(project.word_count ?? 0).toLocaleString()} words · {project.transcript_timing === "cue" ? "cue timing; no word timings" : "word timing"}
              </p>
              {project.transcript_preference === "whisper" && project.transcript_source.startsWith("uploaded_") ? (
                <p className="mt-1 text-[11px] text-warn">Whisper is selected for the next run; the saved transcript remains this upload until Whisper succeeds.</p>
              ) : null}
            </div>
          ) : null}
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => fileInput.current?.click()} disabled={analysing || busy}>
              <FileText size={13} /> {project?.transcript_source.startsWith("uploaded_") ? "Replace transcript" : "Attach transcript"}
            </button>
            {(project?.transcript_source.startsWith("uploaded_") || project?.transcript_source === "source_captions") ? (
              <button type="button" className="btn btn-ghost btn-sm" onClick={useWhisper} disabled={analysing || busy || project.transcript_preference === "whisper"}>
                <Mic size={13} /> Use Whisper instead
              </button>
            ) : project?.transcript_source === "whisper" ? (
              <button type="button" className="btn btn-ghost btn-sm" onClick={retranscribeAnyway} disabled={analysing || busy}>
                <RefreshCw size={13} /> Re-transcribe anyway
              </button>
            ) : null}
            <input
              ref={fileInput}
              type="file"
              accept=".srt,.vtt,.json3,.txt"
              hidden
              onChange={(event) => event.target.files?.[0] && attachTranscript(event.target.files[0])}
            />
          </div>
          <p className="text-[11px] leading-relaxed text-text-3">
            An uploaded .srt, .vtt, .json3 or timestamped .txt is authoritative: its cue text and timestamps are used for analysis and captions. Whisper is skipped unless you explicitly choose it.
          </p>
        </div>
      </section>

      {project?.status === "ready" && clips.length === 0 ? (
        <section className="card border border-warn/40 bg-warn/5 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle size={17} className="mt-0.5 shrink-0 text-warn" />
            <div className="min-w-0 flex-1">
              <h2 className="text-sm font-semibold text-text">Analysis finished with no accepted clips</h2>
              <p className="mt-1 text-xs leading-relaxed text-text-2">
                {diagnostics?.discovered
                  ? `${diagnostics.discovered} candidates were found; ${diagnostics.scored} scored successfully. Average score ${diagnostics.average_score.toFixed(1)}/100, highest ${diagnostics.highest_score.toFixed(1)}/100, threshold ${diagnostics.threshold.toFixed(1)}. `
                  : "No valid candidate windows were found in the transcript. "}
                {diagnostics?.top_rejection_reason
                  ? `Most common rejection: ${rejectionLabel(diagnostics.top_rejection_reason)}.`
                  : diagnostics?.scoring_errors
                    ? `${diagnostics.scoring_errors} scoring/parsing errors were recorded; see Considered for candidates and the analysis log for details.`
                    : "Timestamp, overlap, threshold and context diagnostics are available below."}
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button type="button" className="btn btn-secondary btn-sm" onClick={() => setTab("rejected")}>
                  View {diagnostics?.stored ?? candidates.length} considered
                </button>
                <button type="button" className="btn btn-secondary btn-sm" onClick={lowerThresholdAndReanalyse} disabled={busy || configuredThreshold <= 0}>
                  Lower threshold by 10 and re-analyse
                </button>
                <label className="flex cursor-pointer items-center gap-2 rounded border border-line px-2.5 py-1.5 text-[11px] text-text-2">
                  <input type="checkbox" checked={debugMode} onChange={(event) => setDebugMode(event.target.checked)} className="accent-accent" />
                  Debug mode (threshold 0; context gates off)
                </label>
                <button type="button" className="btn btn-primary btn-sm" onClick={runDebugAnalysis} disabled={busy}>
                  {busy ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                  Re-analyse{debugMode ? " in debug mode" : ""}
                </button>
              </div>
            </div>
          </div>
        </section>
      ) : null}

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
          <select className="select my-1 ml-auto max-w-[10rem]" value={sort} onChange={(event) => setSort(event.target.value)} aria-label="Sort clips">
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
            title={analysing ? "Finding clips…" : "No clips yet"}
            body="Clipforge only keeps moments it can justify from the transcript — the Considered tab shows what missed the threshold."
          />
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {clips.map((clip) => (
              <ClipCard
                key={clip.id}
                clip={clip}
                projectId={projectId}
                selected={Boolean(selected[clip.id])}
                onSelect={(value) => setSelected({ ...selected, [clip.id]: value })}
              />
            ))}
          </ul>
        )
      ) : null}

      {tab === "transcript" ? (
        transcript ? (
          <div className="card max-h-[65vh] overflow-y-auto p-4 scroll-thin">
            <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-text-3">
              <span className="chip chip-good">{transcriptSourceLabel(transcript.source)}</span>
              {transcript.filename ? <span className="font-medium text-text-2">{transcript.filename}</span> : null}
              <span>{transcript.segment_count.toLocaleString()} segments</span>
              <span>{transcript.word_count.toLocaleString()} words</span>
              <span>{transcript.language_name}</span>
              <span className="chip">{transcript.timing_granularity === "cue" ? "cue timing · no word timings" : "word timing"}</span>
              {transcript.language_mode && transcript.language_mode !== "monolingual" ? (
                <span className="chip chip-warn">{transcript.language_mode}</span>
              ) : null}
              {transcript.engine && transcript.source === "whisper" ? <span className="text-text-3">{transcript.engine}</span> : null}
            </div>
            <div className="mb-3 flex flex-wrap gap-2">
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => fileInput.current?.click()} disabled={analysing || busy}>
                <FileText size={12} /> Replace transcript
              </button>
              {transcript.source.startsWith("uploaded_") || transcript.source === "source_captions" ? (
                <button type="button" className="btn btn-ghost btn-sm" onClick={useWhisper} disabled={analysing || busy || transcript.preference === "whisper"}>
                  <Mic size={12} /> Use Whisper instead
                </button>
              ) : transcript.source === "whisper" ? (
                <button type="button" className="btn btn-ghost btn-sm" onClick={retranscribeAnyway} disabled={analysing || busy}>
                  <RefreshCw size={12} /> Re-transcribe anyway
                </button>
              ) : null}
            </div>
            <ol className="space-y-2">
              {transcript.segments.map((segment) => (
                <li key={segment.index} className="grid grid-cols-[4.5rem_1fr] gap-3 text-[13px]">
                  <span className="mono pt-0.5 text-[11px] text-text-3">{clock(segment.start)}</span>
                  <span>
                    {segment.speaker ? <span className="mr-2 text-[11px] font-medium text-accent">{segment.speaker}</span> : null}
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
                  {candidate.rejection_code ? <span className="chip chip-warn">{rejectionLabel(candidate.rejection_code)}</span> : null}
                </div>
                <p className="mt-1 text-[13px] text-text-2">{candidate.title}</p>
                <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-text-3">{candidate.transcript_text || candidate.summary}</p>
                {candidate.reason ? <p className="mt-1 text-[11px] italic text-text-3">{candidate.reason}</p> : null}
              </li>
            ))}
          </ul>
        )
      ) : null}
    </div>
  );
}

function ClipCard({
  clip,
  projectId,
  selected,
  onSelect,
}: {
  clip: ClipSummary;
  projectId: string;
  selected: boolean;
  onSelect: (value: boolean) => void;
}) {
  const chip = clip.status === "pending" && clip.has_render ? { label: "Edited · re-render", className: "chip chip-warn" } : statusChip(clip.status);
  const busy = clip.status === "rendering" || clip.status === "queued";
  const version = clip.rendered_at ?? clip.file_size;
  return (
    <li className="card overflow-hidden">
      <div className="relative aspect-[9/16] bg-black">
        {clip.render_url && !busy ? (
          <video
            className="h-full w-full object-contain"
            controls
            preload="none"
            poster={mediaUrl(clip.thumbnail_url, version) || undefined}
            src={mediaUrl(clip.render_url, version)}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 bg-surface-2 px-4 text-center">
            {busy ? <Loader2 size={18} className="animate-spin text-text-3" /> : <Play size={18} className="text-text-3" />}
            <p className="text-[11px] text-text-3">
              {clip.status === "rendering"
                ? `Rendering · ${Math.round((clip.progress ?? 0) * 100)}%`
                : clip.status === "queued"
                  ? "Waiting for a worker"
                  : clip.status === "failed"
                    ? "Render failed"
                    : "Not rendered"}
            </p>
            {clip.status === "rendering" ? (
              <div className="w-2/3">
                <ProgressBar value={clip.progress ?? 0} />
              </div>
            ) : null}
            {clip.status === "failed" && clip.error?.message ? <p className="line-clamp-3 text-[11px] text-bad">{clip.error.message}</p> : null}
          </div>
        )}
        <label className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-1.5 py-1 text-[11px] text-white">
          <input type="checkbox" checked={selected} onChange={(event) => onSelect(event.target.checked)} className="accent-accent" />
          select
        </label>
        <span className={`absolute right-2 top-2 ${chip.className}`}>{chip.label}</span>
      </div>

      <div className="space-y-2 p-3">
        <div className="flex items-baseline gap-2">
          <span className="mono text-[11px] text-text-3">
            {clock(clip.start)}–{clock(clip.end)}
          </span>
          <span className={`mono ml-auto text-[13px] font-semibold tabular-nums ${scoreTone(clip.score)}`}>{clip.score.toFixed(0)}</span>
        </div>
        <p className="line-clamp-2 text-[13px] font-medium leading-snug text-text">{clip.title}</p>
        {clip.hook ? <p className="line-clamp-2 text-[11px] leading-relaxed text-text-3">“{clip.hook}”</p> : null}
        <div className="flex items-center gap-2 text-[11px] text-text-3">
          <span className="chip">{clip.category_label || clip.category}</span>
          <span>{duration(clip.duration)}</span>
        </div>
        <div className="flex gap-2 pt-1">
          <Link href={`/projects/${projectId}/clips/${clip.id}`} className="btn btn-secondary btn-sm flex-1">
            Edit
          </Link>
          {clip.render_url ? (
            <a className="btn btn-ghost btn-sm" href={`${clip.render_url}?download=true`} download>
              <Download size={13} /> MP4
            </a>
          ) : null}
        </div>
      </div>
    </li>
  );
}

function Count({ label, value, warn = false }: { label: string; value: number | string; warn?: boolean }) {
  return (
    <div className="rounded border border-line bg-surface-2 px-2.5 py-2">
      <div className={`mono text-sm font-semibold ${warn ? "text-warn" : "text-text"}`}>{value}</div>
      <div className="mt-0.5 text-[10px] text-text-3">{label}</div>
    </div>
  );
}

function transcriptSourceLabel(source: string) {
  if (source.startsWith("uploaded_")) return `Uploaded ${source.slice("uploaded_".length).toUpperCase()}`;
  if (source === "whisper") return "Whisper";
  if (source === "source_captions") return "Source captions";
  return source || "Transcript not set";
}

function rejectionLabel(code: string) {
  const labels: Record<string, string> = {
    below_threshold: "below threshold",
    context_dependency: "needs earlier context",
    not_standalone: "not standalone",
    low_confidence: "low transcript confidence",
    overlap_duplicate: "overlaps another moment",
    semantic_duplicate: "semantic duplicate",
    overlap_conflict: "overlap conflict",
    invalid_timestamp: "invalid timestamp",
    scoring_error: "scoring error",
    max_clips: "clip limit reached",
  };
  return labels[code] || code.replaceAll("_", " ");
}

function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
      <p className="text-sm font-medium text-text-2">{title}</p>
      <p className="max-w-md text-xs text-text-3">{body}</p>
    </div>
  );
}
