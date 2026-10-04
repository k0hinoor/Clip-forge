"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import {
  Copy,
  Download,
  Eye,
  Film,
  Info,
  Layers,
  RefreshCw,
  Rows3,
  Save,
  Sparkles,
  Terminal,
  Wand2,
} from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { Spinner } from "@/components/Progress";
import { clock, duration, scoreTone, statusChip } from "@/lib/format";
import type { ClipDetail } from "@/lib/types";

const PRESETS = ["minimal", "cinematic", "bold_creator", "karaoke", "highlight", "documentary"];
const LAYOUTS = ["split", "podcast", "broll", "gameplay", "cinematic", "blur"];

export default function ClipEditorPage() {
  const params = useParams<{ id: string; clipId: string }>();
  const { id: projectId, clipId } = params;
  const toast = useToast();

  const [clip, setClip] = useState<ClipDetail | null>(null);
  const [preview, setPreview] = useState<any>(null);
  const [command, setCommand] = useState<string>("");
  const [showCommand, setShowCommand] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState({
    title: "",
    start: 0,
    end: 0,
    layout: "split",
    split_ratio: 65,
    caption_preset: "",
    captions_enabled: true,
    zoom: true,
  });

  const load = useCallback(
    async (silent = false) => {
      try {
        const detail = await api.clip(clipId);
        setClip(detail);
        setDraft((current) => ({
          ...current,
          title: detail.title,
          start: detail.start,
          end: detail.end,
          layout: detail.plan?.layout ?? "split",
          split_ratio: detail.plan?.split_ratio ?? 65,
          caption_preset: detail.captions?.theme?.preset ?? "",
        }));
        api.clipCommand(clipId).then((payload) => setCommand(payload.command)).catch(() => undefined);
      } catch (error) {
        if (!silent) toast.fail(error, "Could not open this clip.");
      }
    },
    [clipId, toast],
  );

  useEffect(() => {
    load();
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
        caption_preset: draft.caption_preset || undefined,
      });
      toast.ok("Clip updated", "Captions, framing and the timeline were rebuilt from the transcript.");
      await load(true);
    } catch (error) {
      toast.fail(error, "Those edits could not be applied.");
    } finally {
      setBusy(false);
    }
  };

  const act = async (kind: "render" | "preview" | "regenerate" | "duplicate") => {
    setBusy(true);
    try {
      if (kind === "render") {
        await api.renderClip(clipId, { export: true });
        toast.ok("Render queued");
      } else if (kind === "preview") {
        const result = await api.previewClip(clipId);
        toast.ok("Preview queued", "A fast low-resolution render will appear in the queue.");
        void result;
      } else if (kind === "regenerate") {
        const result = await api.regenerateClip(clipId);
        toast.ok("Re-scored this moment", result.message ?? "The best boundaries were kept.");
      } else {
        await api.duplicateClip(clipId);
        toast.ok("Clip duplicated");
      }
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const loadPreview = async (preset: string) => {
    try {
      const payload = await api.captionPreview(clipId, preset);
      setPreview(payload);
    } catch (error) {
      toast.fail(error, "Caption preview failed.");
    }
  };

  if (!clip) return <Spinner label="Loading clip…" />;

  const chip = statusChip(clip.status);
  const timeline = clip.plan?.timeline;
  const removedSeconds = timeline?.segments
    ? Math.max(0, clip.duration - (timeline.segments[timeline.segments.length - 1]?.out_end ?? clip.duration))
    : 0;

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <Link href={`/projects/${projectId}`} className="text-[11px] font-semibold uppercase tracking-widest text-mist-400 hover:text-flare-400">
            ← Project
          </Link>
          <h1 className="mt-1 text-xl font-black text-mist-200">
            Clip {String(clip.index).padStart(2, "0")} · <span className={scoreTone(clip.score)}>{clip.score.toFixed(1)}</span>
          </h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-mist-400">
            <span className="mono">
              {clock(clip.start)}–{clock(clip.end)}
            </span>
            <span>{duration(clip.duration)}</span>
            <span className="chip">{clip.category_label || clip.category}</span>
            <span className={chip.className}>{chip.label}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn btn-ghost" onClick={() => act("preview")} disabled={busy}>
            <Eye size={14} /> Preview
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => act("regenerate")} disabled={busy}>
            <RefreshCw size={14} /> Re-score
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => act("duplicate")} disabled={busy}>
            <Copy size={14} /> Duplicate
          </button>
          <button type="button" className="btn btn-primary" onClick={() => act("render")} disabled={busy}>
            {busy ? <Spinner /> : <Film size={14} />} Render clip
          </button>
        </div>
      </header>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,22rem)_1fr]">
        <div className="card overflow-hidden">
          <div className="aspect-[9/16] bg-ink-900">
            {clip.status === "rendered" ? (
              <video className="h-full w-full" controls src={`/api/clips/${clipId}/preview`} />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
                <Film size={22} className="text-mist-400" />
                <p className="text-xs text-mist-400">
                  {clip.status === "rendering" ? `Rendering · ${Math.round((clip.progress ?? 0) * 100)}%` : "Not rendered yet"}
                </p>
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-2 border-t border-ink-700/70 p-3">
            {clip.status === "rendered" ? (
              <>
                <a className="btn btn-ghost text-xs" href={`/api/clips/${clipId}/file?download=true`}>
                  <Download size={13} /> Download MP4
                </a>
                <a className="btn btn-ghost text-xs" href={`/api/clips/${clipId}/captions/srt`}>
                  <Download size={13} /> SRT
                </a>
              </>
            ) : null}
            <button type="button" className="btn btn-quiet text-xs" onClick={() => setShowCommand(!showCommand)}>
              <Terminal size={13} /> {showCommand ? "Hide" : "Show"} FFmpeg command
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <section className="card p-4">
            <h2 className="flex items-center gap-2 text-sm font-bold text-mist-200">
              <Sparkles size={15} className="text-amber-glow" /> WHY THIS CLIP?
            </h2>
            <p className="mt-2 text-sm italic leading-relaxed text-mist-300">“{clip.hook}”</p>
            <ul className="mt-3 space-y-1.5">
              {clip.why?.map((reason, index) => (
                <li key={index} className="flex gap-2 text-xs leading-relaxed text-mist-300">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-flare-500" />
                  {reason}
                </li>
              ))}
            </ul>
            {clip.summary ? <p className="mt-3 text-xs leading-relaxed text-mist-400">{clip.summary}</p> : null}
            {clip.factors ? (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {Object.entries(clip.factors)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 8)
                  .map(([name, value]) => (
                    <span key={name} className="chip" title={name}>
                      {name.replace(/_/g, " ")} {(value as number).toFixed(2)}
                    </span>
                  ))}
              </div>
            ) : null}
          </section>

          <section className="card space-y-4 p-4">
            <h2 className="flex items-center gap-2 text-sm font-bold text-mist-200">
              <Layers size={15} /> Edit plan
            </h2>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <span className="label">Title</span>
                <input className="input" value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} />
              </div>
              <div>
                <span className="label">Layout</span>
                <select className="select" value={draft.layout} onChange={(event) => setDraft({ ...draft, layout: event.target.value })}>
                  {LAYOUTS.map((layout) => (
                    <option key={layout} value={layout}>
                      {layout.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <span className="label">Start (s)</span>
                <input
                  className="input"
                  type="number"
                  step="0.1"
                  value={draft.start}
                  onChange={(event) => setDraft({ ...draft, start: Number(event.target.value) })}
                />
              </div>
              <div>
                <span className="label">End (s)</span>
                <input
                  className="input"
                  type="number"
                  step="0.1"
                  value={draft.end}
                  onChange={(event) => setDraft({ ...draft, end: Number(event.target.value) })}
                />
              </div>
              <div>
                <span className="label">Split ratio (speaker share)</span>
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
              <div>
                <span className="label">Caption preset</span>
                <select
                  className="select"
                  value={draft.caption_preset}
                  onChange={(event) => {
                    setDraft({ ...draft, caption_preset: event.target.value });
                    loadPreview(event.target.value);
                  }}
                >
                  <option value="">Project default</option>
                  {PRESETS.map((preset) => (
                    <option key={preset} value={preset}>
                      {preset.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button type="button" className="btn btn-primary" onClick={save} disabled={busy}>
                <Save size={14} /> Save &amp; rebuild
              </button>
              <span className="text-[11px] text-mist-400">
                Saving re-runs silence detection, framing and caption planning for this clip only.
              </span>
            </div>
          </section>

          <section className="grid gap-4 sm:grid-cols-2">
            <div className="card p-4">
              <h3 className="flex items-center gap-2 text-sm font-bold text-mist-200">
                <Rows3 size={15} /> Pacing
              </h3>
              <p className="mt-2 text-xs text-mist-400">
                {timeline?.segments?.length ?? 1} kept segment{(timeline?.segments?.length ?? 1) === 1 ? "" : "s"} · {removedSeconds.toFixed(1)}s of
                silence removed
              </p>
              <ul className="mt-2 space-y-0.5 text-[11px] text-mist-400">
                {(timeline?.notes ?? []).map((note, index) => (
                  <li key={index}>· {note}</li>
                ))}
                {(clip.plan?.notes ?? []).map((note, index) => (
                  <li key={`p${index}`}>· {note}</li>
                ))}
              </ul>
            </div>
            <div className="card p-4">
              <h3 className="flex items-center gap-2 text-sm font-bold text-mist-200">
                <Info size={15} /> Assets
              </h3>
              <ul className="mt-2 space-y-1 text-[11px] text-mist-400">
                <li>Gameplay: {clip.plan?.gameplay?.name ?? "none selected"}</li>
                <li>B-roll: {clip.plan?.broll?.name ?? "none selected"}</li>
                <li>Music: {clip.plan?.music?.name ?? "no music track"}</li>
                <li>
                  Reframing: {clip.plan?.crop ? `${clip.plan.crop.mode ?? "static"} · ${clip.plan.crop.crop?.join("×") ?? ""}` : "full frame"}
                </li>
              </ul>
            </div>
          </section>

          {preview ? (
            <section className="card p-4">
              <h3 className="text-sm font-bold text-mist-200">Caption preview · {preview.theme?.preset}</h3>
              <ol className="mt-2 space-y-1.5">
                {preview.lines?.slice(0, 8).map((line: any) => (
                  <li key={line.index} className="mono text-xs text-mist-200">
                    <span className="mr-2 text-mist-400">{clock(line.start)}</span>
                    {line.text}
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          {showCommand ? (
            <section className="card p-4">
              <h3 className="text-sm font-bold text-mist-200">FFmpeg command</h3>
              <pre className="mono mt-2 max-h-52 overflow-auto rounded-lg bg-ink-950/80 p-3 text-[10px] leading-relaxed text-mist-300 scroll-thin">
                {command || "unavailable"}
              </pre>
            </section>
          ) : null}
        </div>
      </section>
    </div>
  );
}
