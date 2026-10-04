"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Download, Film, Loader2, Play } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar, Spinner } from "@/components/Progress";
import { clock, duration, statusChip } from "@/lib/format";
import type { ClipDetail } from "@/lib/types";

const CAPTION_STYLES = [
  ["bold_creator", "Bold (Shorts classic)"],
  ["minimal", "Minimal"],
  ["karaoke", "Karaoke"],
  ["cinematic", "Cinematic"],
  ["highlight", "Highlight"],
  ["documentary", "Documentary"],
];

const LAYOUTS = [
  ["podcast", "Full frame"],
  ["blur", "Blurred background"],
  ["cinematic", "Cinematic crop"],
  ["split", "Split screen"],
  ["gameplay", "Gameplay background"],
  ["broll", "B-roll split"],
];

export default function ClipEditorPage() {
  const params = useParams<{ id: string; clipId: string }>();
  const { id: projectId, clipId } = params;
  const toast = useToast();
  const video = useRef<HTMLVideoElement>(null);

  const [clip, setClip] = useState<ClipDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState({
    title: "",
    start: 0,
    end: 0,
    layout: "podcast",
    split_ratio: 65,
    caption_preset: "",
    captions_enabled: true,
  });

  const load = useCallback(
    async (silent = false) => {
      try {
        const detail = await api.clip(clipId);
        setClip(detail);
        setDraft({
          title: detail.title,
          start: detail.start,
          end: detail.end,
          layout: detail.plan?.layout ?? "podcast",
          split_ratio: detail.plan?.split_ratio ?? 65,
          caption_preset: detail.captions?.theme?.preset ?? "",
          captions_enabled: detail.captions ? detail.captions_enabled !== false : true,
        });
      } catch (error) {
        if (!silent) toast.fail(error, "Could not open this clip.");
      }
    },
    [clipId, toast],
  );

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 5000);
    return () => window.clearInterval(timer);
  }, [load]);

  const save = async () => {
    setBusy(true);
    try {
      await api.updateClip(clipId, {
        title: draft.title,
        start: draft.start,
        end: draft.end,
        layout: draft.layout,
        split_ratio: draft.split_ratio,
        captions_enabled: draft.captions_enabled,
        ...(draft.caption_preset ? { caption_preset: draft.caption_preset } : {}),
      });
      toast.ok("Saved", "Captions, framing and pacing were rebuilt for this clip.");
      await load(true);
    } catch (error) {
      toast.fail(error, "Those edits could not be applied.");
    } finally {
      setBusy(false);
    }
  };

  const render = async () => {
    setBusy(true);
    try {
      await api.renderClip(clipId, { export: true });
      toast.ok("Render queued", "It appears in the Queue as soon as a worker is free.");
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const markTime = (which: "start" | "end") => {
    const current = video.current?.currentTime;
    if (current === undefined) return;
    // Player time is relative to the rendered clip; the stored trim is in source time.
    const offset = clip?.start ?? 0;
    const value = Number((offset + current).toFixed(2));
    setDraft({ ...draft, [which]: value });
  };

  if (!clip) return <Spinner label="Loading clip…" />;

  const chip = statusChip(clip.status);
  const captionLines = clip.captions?.lines ?? [];

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <Link href={`/projects/${projectId}`} className="inline-flex items-center gap-1 text-xs text-text-3 hover:text-text-2">
            <ArrowLeft size={12} /> Project
          </Link>
          <h1 className="mt-1 truncate text-lg font-semibold tracking-tight text-text">
            Clip {String(clip.index).padStart(2, "0")}
          </h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-text-3">
            <span className="mono">
              {clock(clip.start)}–{clock(clip.end)}
            </span>
            <span>{duration(clip.duration)}</span>
            <span className="chip">{clip.category_label || clip.category}</span>
            <span className={chip.className}>{chip.label}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn btn-primary" onClick={render} disabled={busy}>
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Film size={14} />} Render
          </button>
        </div>
      </header>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,20rem)_1fr]">
        <div className="card overflow-hidden">
          <div className="aspect-[9/16]">
            {clip.status === "rendered" || clip.has_preview ? (
              <video ref={video} className="h-full w-full" controls preload="metadata" src={`/api/clips/${clipId}/preview`} />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-2 bg-surface-2 p-6 text-center">
                <Play size={20} className="text-text-3" />
                <p className="text-xs text-text-3">
                  {clip.status === "rendering" ? `Rendering · ${Math.round((clip.progress ?? 0) * 100)}%` : "Not rendered yet"}
                </p>
                {clip.status === "rendering" ? (
                  <div className="w-2/3">
                    <ProgressBar value={clip.progress ?? 0} />
                  </div>
                ) : null}
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-2 border-t border-line p-3">
            {clip.status === "rendered" ? (
              <>
                <a className="btn btn-secondary btn-sm" href={`/api/clips/${clipId}/file?download=true`}>
                  <Download size={13} /> MP4
                </a>
                <a className="btn btn-ghost btn-sm" href={`/api/clips/${clipId}/captions/srt`}>
                  <Download size={13} /> SRT
                </a>
              </>
            ) : (
              <span className="text-[11px] text-text-3">Render to get the MP4 and its SRT.</span>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="card p-4">
            <h2 className="panel-title">Trim</h2>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <div>
                <span className="label">Start (s)</span>
                <div className="flex gap-2">
                  <input
                    className="input"
                    type="number"
                    step="0.1"
                    min={0}
                    value={draft.start}
                    onChange={(event) => setDraft({ ...draft, start: Number(event.target.value) })}
                  />
                  <button type="button" className="btn btn-secondary" onClick={() => markTime("start")} title="Use the player position">
                    Here
                  </button>
                </div>
              </div>
              <div>
                <span className="label">End (s)</span>
                <div className="flex gap-2">
                  <input
                    className="input"
                    type="number"
                    step="0.1"
                    min={0}
                    value={draft.end}
                    onChange={(event) => setDraft({ ...draft, end: Number(event.target.value) })}
                  />
                  <button type="button" className="btn btn-secondary" onClick={() => markTime("end")} title="Use the player position">
                    Here
                  </button>
                </div>
              </div>
            </div>
            <p className="mt-2 text-[11px] text-text-3">
              Length {duration(Math.max(0, draft.end - draft.start))} · “Here” reads the player position.
            </p>
          </div>

          <div className="card p-4">
            <h2 className="panel-title">Look</h2>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <div>
                <span className="label">Title</span>
                <input className="input" value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} />
              </div>
              <div>
                <span className="label">Layout</span>
                <select className="select" value={draft.layout} onChange={(event) => setDraft({ ...draft, layout: event.target.value })}>
                  {LAYOUTS.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
              {["split", "broll", "gameplay"].includes(draft.layout) ? (
                <div>
                  <span className="label">Speaker share</span>
                  <select
                    className="select"
                    value={draft.split_ratio}
                    onChange={(event) => setDraft({ ...draft, split_ratio: Number(event.target.value) })}
                  >
                    {[50, 60, 65, 70].map((ratio) => (
                      <option key={ratio} value={ratio}>
                        {ratio} / {100 - ratio}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
              <div>
                <span className="label">Caption style</span>
                <select
                  className="select"
                  value={draft.caption_preset}
                  disabled={!draft.captions_enabled}
                  onChange={(event) => setDraft({ ...draft, caption_preset: event.target.value })}
                >
                  <option value="">Project default</option>
                  {CAPTION_STYLES.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <label className="mt-3 flex items-center gap-2 text-[13px] text-text-2">
              <input
                type="checkbox"
                className="accent-accent"
                checked={draft.captions_enabled}
                onChange={(event) => setDraft({ ...draft, captions_enabled: event.target.checked })}
              />
              Burn captions into the video
            </label>

            <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-3">
              <button type="button" className="btn btn-primary" onClick={save} disabled={busy}>
                Save changes
              </button>
              <span className="text-[11px] text-text-3">Saving rebuilds pacing, framing and captions for this clip.</span>
            </div>
          </div>

          <div className="card p-4">
            <h2 className="panel-title">Captions · {captionLines.length} lines</h2>
            {captionLines.length === 0 ? (
              <p className="mt-2 text-xs text-text-3">
                {draft.captions_enabled
                  ? "No words in this range yet — save the trim and the captions are rebuilt from the transcript."
                  : "Captions are switched off for this clip."}
              </p>
            ) : (
              <ol className="mt-2 max-h-56 space-y-1 overflow-y-auto scroll-thin">
                {captionLines.map((line: any) => (
                  <li key={line.index} className="flex gap-3 text-xs">
                    <span className="mono shrink-0 text-text-3">{clock(line.start)}</span>
                    <span className="text-text-2">{line.text}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div className="card p-4">
            <h2 className="panel-title">Why this moment</h2>
            <p className="mt-2 text-[13px] italic leading-relaxed text-text-2">“{clip.hook}”</p>
            {clip.why?.length ? (
              <ul className="mt-2 space-y-1">
                {clip.why.map((reason, index) => (
                  <li key={index} className="flex gap-2 text-xs leading-relaxed text-text-3">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-text-3" />
                    {reason}
                  </li>
                ))}
              </ul>
            ) : null}
            {clip.plan?.notes?.length ? (
              <ul className="mt-3 space-y-1 border-t border-line pt-2">
                {clip.plan.notes.map((note: string, index: number) => (
                  <li key={index} className="text-[11px] text-text-3">
                    · {note}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </div>
      </section>
    </div>
  );
}
