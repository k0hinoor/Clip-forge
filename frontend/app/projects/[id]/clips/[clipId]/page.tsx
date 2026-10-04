"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Ban, Copy, Download, Eye, Film, Loader2, Play, Sparkles, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { useLiveRefresh } from "@/lib/live";
import { CaptionStyleSelect, LAYOUT_OPTIONS, Toggle } from "@/components/Controls";
import { useSystem } from "@/components/System";
import { useToast } from "@/components/Toast";
import { ProgressBar, Spinner } from "@/components/Progress";
import { clock, duration, mediaUrl, statusChip } from "@/lib/format";
import type { ClipAssetOptions, ClipDetail, TimelineSegment } from "@/lib/types";

type Draft = {
  title: string;
  start: number;
  end: number;
  layout: string;
  split_ratio: number;
  caption_preset: string;
  captions_enabled: boolean;
  remove_silence: boolean;
  auto_zoom: boolean;
  aspect_ratio: string;
  music_enabled: boolean;
  gameplay_asset_id: string;
  broll_asset_id: string;
  music_asset_id: string;
};

type Player = "clip" | "source";

function draftFrom(detail: ClipDetail): Draft {
  return {
    title: detail.title,
    start: detail.start,
    end: detail.end,
    layout: detail.plan?.layout ?? "podcast",
    split_ratio: detail.plan?.split_ratio ?? 65,
    caption_preset: detail.captions?.theme?.preset ?? "",
    captions_enabled: detail.captions_enabled !== false,
    remove_silence: detail.settings?.remove_silence ?? true,
    auto_zoom: detail.settings?.auto_zoom ?? true,
    aspect_ratio: detail.settings?.aspect_ratio ?? "9:16",
    music_enabled: detail.settings?.music_enabled ?? false,
    gameplay_asset_id: detail.plan?.gameplay?.id ?? "",
    broll_asset_id: detail.plan?.broll?.id ?? "",
    music_asset_id: detail.plan?.music?.id ?? "",
  };
}

/** Map a time in the rendered clip back to the source (silence removal shortens clips). */
function outputToSource(time: number, segments: TimelineSegment[], fallbackStart: number): number {
  for (const segment of segments) {
    if (time <= segment.out_end + 0.001) {
      return segment.src_start + Math.max(0, time - segment.out_start);
    }
  }
  const last = segments[segments.length - 1];
  return last ? last.src_end : fallbackStart + time;
}

export default function ClipEditorPage() {
  const params = useParams<{ id: string; clipId: string }>();
  const { id: projectId, clipId } = params;
  const router = useRouter();
  const toast = useToast();
  const { features } = useSystem();
  const video = useRef<HTMLVideoElement>(null);

  const [clip, setClip] = useState<ClipDetail | null>(null);
  const [assets, setAssets] = useState<ClipAssetOptions>({ gameplay: [], broll: [], music: [] });
  const [busy, setBusy] = useState<string>("");
  const [player, setPlayer] = useState<Player>("clip");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [dirty, setDirty] = useState(false);
  const [missing, setMissing] = useState(false);
  const dirtyRef = useRef(false);
  dirtyRef.current = dirty;

  const load = useCallback(
    async (silent = false, resetDraft = false) => {
      try {
        const detail = await api.clip(clipId);
        setClip(detail);
        // Never overwrite edits the user has not saved yet.
        if (resetDraft || !dirtyRef.current) {
          setDraft(draftFrom(detail));
          setDirty(false);
        }
      } catch (error: any) {
        if (error?.status === 404) setMissing(true);
        else if (!silent) toast.fail(error, "Could not open this clip.");
      }
    },
    [clipId, toast],
  );

  useEffect(() => {
    load(false, true);
    api.clipAssets(clipId).then(setAssets).catch(() => undefined);
  }, [clipId, load]);

  useLiveRefresh(() => load(true), {
    projectId,
    match: (event) => !event.clip_id || event.clip_id === clipId,
    throttleMs: 1000,
    intervalMs: 15000,
  });

  const edit = (patch: Partial<Draft>) => {
    setDraft((current) => (current ? { ...current, ...patch } : current));
    setDirty(true);
  };

  const run = async (label: string, action: () => Promise<void>) => {
    setBusy(label);
    try {
      await action();
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy("");
    }
  };

  const changes = (): Record<string, unknown> => {
    if (!clip || !draft) return {};
    const base = draftFrom(clip);
    const patch: Record<string, unknown> = {};
    (Object.keys(draft) as (keyof Draft)[]).forEach((key) => {
      if (draft[key] !== base[key]) patch[key] = draft[key];
    });
    if (patch.caption_preset === "") delete patch.caption_preset;
    return patch;
  };

  const save = () =>
    run("save", async () => {
      const patch = changes();
      if (!Object.keys(patch).length) {
        setDirty(false);
        return;
      }
      await api.updateClip(clipId, patch);
      toast.ok("Saved", "Render again to apply the changes to the video.");
      await load(true, true);
    });

  const render = () =>
    run("render", async () => {
      if (dirty) {
        const patch = changes();
        if (Object.keys(patch).length) await api.updateClip(clipId, patch);
      }
      await api.renderClip(clipId, { export: features.local_paths });
      toast.ok("Render queued", "Follow it here or in the Queue.");
      await load(true, true);
    });

  const preview = () =>
    run("preview", async () => {
      await api.previewClip(clipId);
      toast.ok("Preview queued", "A fast low-resolution version appears in a moment.");
    });

  const cancel = () =>
    run("cancel", async () => {
      const result = await api.cancelClip(clipId);
      toast.ok(result.cancelled ? "Render cancelled" : "Nothing to cancel");
      await load(true);
    });

  const regenerate = () =>
    run("regenerate", async () => {
      if (dirty && !window.confirm("Regenerating discards your unsaved edits. Continue?")) return;
      const result = await api.regenerateClip(clipId, { window: 120 });
      toast.ok(result.changed ? "Found a stronger moment nearby" : "This was already the strongest moment nearby", result.explanation);
      await load(true, true);
    });

  const duplicate = () =>
    run("duplicate", async () => {
      const copy = await api.duplicateClip(clipId);
      toast.ok("Clip duplicated");
      router.push(`/projects/${projectId}/clips/${copy.id}`);
    });

  const remove = () =>
    run("delete", async () => {
      if (!window.confirm("Delete this clip and its rendered files?")) return;
      await api.deleteClip(clipId);
      toast.ok("Clip deleted");
      router.push(`/projects/${projectId}`);
    });

  const markTime = (which: "start" | "end") => {
    const current = video.current?.currentTime;
    if (current === undefined || !clip) return;
    // The source player runs on source time; the rendered clip has its own
    // (possibly shortened) timeline that maps back through the segments.
    const value = player === "source" ? current : outputToSource(current, clip.plan?.timeline?.segments ?? [], clip.start);
    edit({ [which]: Number(value.toFixed(2)) } as Partial<Draft>);
  };

  if (missing) {
    return (
      <div className="card flex flex-col items-center gap-2 px-6 py-12 text-center">
        <p className="text-sm font-medium text-text-2">This clip no longer exists</p>
        <Link href={`/projects/${projectId}`} className="btn btn-secondary btn-sm mt-2">
          <ArrowLeft size={12} /> Back to the project
        </Link>
      </div>
    );
  }
  if (!clip || !draft) return <Spinner label="Loading clip…" />;

  const chip = clip.status === "pending" && clip.has_render ? { label: "Edited · re-render", className: "chip chip-warn" } : statusChip(clip.status);
  const working = clip.status === "rendering" || clip.status === "queued";
  const captionLines = clip.captions?.lines ?? [];
  const version = clip.rendered_at ?? clip.file_size;
  const clipSource = clip.render_url ? mediaUrl(clip.render_url, version) : clip.preview_url ? mediaUrl(clip.preview_url, Date.parse(clip.created_at)) : "";
  const showSource = player === "source" || !clipSource;
  const splitLayout = ["split", "broll", "gameplay"].includes(draft.layout);

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <Link href={`/projects/${projectId}`} className="inline-flex items-center gap-1 text-xs text-text-3 hover:text-text-2">
            <ArrowLeft size={12} /> {clip.project?.title || "Project"}
          </Link>
          <h1 className="mt-1 truncate text-lg font-semibold tracking-tight text-text">
            Clip {String(clip.index).padStart(2, "0")} · {clip.title}
          </h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-text-3">
            <span className="mono">
              {clock(clip.start)}–{clock(clip.end)}
            </span>
            <span>{duration(clip.duration)}</span>
            <span className="chip">{clip.category_label || clip.category}</span>
            <span className={chip.className}>{chip.label}</span>
            <span className="mono">score {clip.score.toFixed(0)}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {working ? (
            <button type="button" className="btn btn-secondary" onClick={cancel} disabled={Boolean(busy)}>
              <Ban size={14} /> Cancel render
            </button>
          ) : (
            <button type="button" className="btn btn-secondary" onClick={preview} disabled={Boolean(busy)} title="Fast low-resolution render">
              <Eye size={14} /> Preview
            </button>
          )}
          <button type="button" className="btn btn-primary" onClick={render} disabled={Boolean(busy) || working}>
            {busy === "render" ? <Loader2 size={14} className="animate-spin" /> : <Film size={14} />} {dirty ? "Save & render" : "Render"}
          </button>
        </div>
      </header>

      {clip.status === "failed" && clip.error?.message ? (
        <div className="card border-bad/40 px-4 py-3 text-[13px] text-bad">{clip.error.message}</div>
      ) : clip.error?.message ? (
        <div className="card border-warn/40 px-4 py-3 text-[13px] text-warn">{clip.error.message}</div>
      ) : null}

      <section className="grid gap-4 lg:grid-cols-[minmax(0,20rem)_1fr]">
        <div className="space-y-3">
          <div className="card overflow-hidden">
            <div className="flex border-b border-line text-[12px]">
              {(
                [
                  ["clip", clip.render_url ? "Rendered clip" : "Preview"],
                  ["source", "Source"],
                ] as [Player, string][]
              ).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setPlayer(key)}
                  disabled={key === "clip" && !clipSource}
                  className={`flex-1 px-3 py-2 font-medium disabled:opacity-40 ${
                    (key === "source") === showSource ? "bg-surface-2 text-text" : "text-text-3 hover:text-text-2"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className={`${showSource ? "aspect-video" : "aspect-[9/16]"} bg-black`}>
              {showSource ? (
                <video
                  key="source"
                  ref={video}
                  className="h-full w-full object-contain"
                  controls
                  preload="metadata"
                  src={`/api/projects/${projectId}/source#t=${Math.max(0, clip.start).toFixed(2)}`}
                />
              ) : (
                <video key={clipSource} ref={video} className="h-full w-full object-contain" controls preload="metadata" src={clipSource} />
              )}
            </div>
            {working ? (
              <div className="flex items-center gap-2 border-t border-line px-3 py-2">
                <ProgressBar value={clip.progress ?? 0} />
                <span className="shrink-0 text-[11px] text-text-3">
                  {clip.status === "queued" ? "queued" : `${Math.round((clip.progress ?? 0) * 100)}%`}
                </span>
              </div>
            ) : null}
            <div className="flex flex-wrap gap-2 border-t border-line p-3">
              {clip.render_url ? (
                <>
                  <a className="btn btn-secondary btn-sm" href={`${clip.render_url}?download=true`} download>
                    <Download size={13} /> MP4
                  </a>
                  <a className="btn btn-ghost btn-sm" href={`/api/clips/${clipId}/captions/srt`} download>
                    <Download size={13} /> SRT
                  </a>
                </>
              ) : (
                <span className="text-[11px] text-text-3">Render to get the MP4 and its SRT.</span>
              )}
            </div>
          </div>

          <div className="card flex flex-wrap gap-2 p-3">
            <button type="button" className="btn btn-ghost btn-sm" onClick={regenerate} disabled={Boolean(busy) || working}>
              {busy === "regenerate" ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />} Regenerate
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={duplicate} disabled={Boolean(busy)}>
              <Copy size={13} /> Duplicate
            </button>
            <button type="button" className="btn btn-ghost btn-sm ml-auto text-bad" onClick={remove} disabled={Boolean(busy)}>
              <Trash2 size={13} /> Delete
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <div className="card p-4">
            <h2 className="panel-title">Trim</h2>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {(["start", "end"] as const).map((which) => (
                <div key={which}>
                  <span className="label">{which === "start" ? "Start" : "End"} (seconds in the source)</span>
                  <div className="flex gap-2">
                    <input
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={draft[which]}
                      onChange={(event) => edit({ [which]: Number(event.target.value) } as Partial<Draft>)}
                    />
                    <button type="button" className="btn btn-secondary" onClick={() => markTime(which)} title="Use the player position">
                      Here
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-text-3">
              Length before silence removal {duration(Math.max(0, draft.end - draft.start))} · “Here” reads the player position
              {showSource ? "" : " (mapped back to the source)"}.
            </p>
          </div>

          <div className="card p-4">
            <h2 className="panel-title">Look</h2>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <label className="block sm:col-span-2">
                <span className="label">Title</span>
                <input className="input" value={draft.title} onChange={(event) => edit({ title: event.target.value })} />
              </label>
              <label className="block">
                <span className="label">Layout</span>
                <select className="select" value={draft.layout} onChange={(event) => edit({ layout: event.target.value })}>
                  {LAYOUT_OPTIONS.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="label">Aspect ratio</span>
                <select className="select" value={draft.aspect_ratio} onChange={(event) => edit({ aspect_ratio: event.target.value })}>
                  <option value="9:16">9:16 vertical</option>
                  <option value="1:1">1:1 square</option>
                  <option value="16:9">16:9 landscape</option>
                </select>
              </label>
              {splitLayout ? (
                <label className="block">
                  <span className="label">Speaker share</span>
                  <select className="select" value={draft.split_ratio} onChange={(event) => edit({ split_ratio: Number(event.target.value) })}>
                    {[50, 55, 60, 65, 70, 75].map((ratio) => (
                      <option key={ratio} value={ratio}>
                        {ratio} / {100 - ratio}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {draft.layout === "split" || draft.layout === "gameplay" ? (
                <AssetSelect
                  label="Gameplay footage"
                  value={draft.gameplay_asset_id}
                  options={assets.gameplay}
                  empty="Import gameplay in the Library first"
                  onChange={(value) => edit({ gameplay_asset_id: value })}
                />
              ) : null}
              {draft.layout === "broll" ? (
                <AssetSelect
                  label="B-roll footage"
                  value={draft.broll_asset_id}
                  options={assets.broll}
                  empty="Import B-roll in the Library first"
                  onChange={(value) => edit({ broll_asset_id: value })}
                />
              ) : null}
              <label className="block">
                <span className="label">Caption style</span>
                <CaptionStyleSelect
                  value={draft.caption_preset}
                  disabled={!draft.captions_enabled}
                  placeholder="Keep the current style"
                  onChange={(value) => edit({ caption_preset: value })}
                />
              </label>
            </div>

            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <Toggle label="Burn in captions" value={draft.captions_enabled} onChange={(value) => edit({ captions_enabled: value })} />
              <Toggle label="Remove silences" value={draft.remove_silence} onChange={(value) => edit({ remove_silence: value })} />
              <Toggle label="Punch-ins" value={draft.auto_zoom} onChange={(value) => edit({ auto_zoom: value })} />
              <Toggle label="Background music" value={draft.music_enabled} onChange={(value) => edit({ music_enabled: value })} />
            </div>
            {draft.music_enabled ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <AssetSelect
                  label="Music track"
                  value={draft.music_asset_id}
                  options={assets.music}
                  empty="Import music in the Library first"
                  onChange={(value) => edit({ music_asset_id: value })}
                />
              </div>
            ) : null}

            <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-3">
              <button type="button" className="btn btn-secondary" onClick={save} disabled={Boolean(busy) || !dirty}>
                {busy === "save" ? <Loader2 size={14} className="animate-spin" /> : null}
                Save changes
              </button>
              {dirty ? (
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => {
                    setDraft(draftFrom(clip));
                    setDirty(false);
                  }}
                >
                  Discard
                </button>
              ) : null}
              <span className="text-[11px] text-text-3">
                {dirty ? "Unsaved changes." : "Saving rebuilds pacing, framing and captions for this clip."}
              </span>
            </div>
          </div>

          <div className="card p-4">
            <h2 className="panel-title">Captions · {captionLines.length} lines</h2>
            {captionLines.length === 0 ? (
              <p className="mt-2 text-xs text-text-3">
                {draft.captions_enabled ? "No words in this range — adjust the trim." : "Captions are switched off for this clip."}
              </p>
            ) : (
              <ol className="mt-2 max-h-56 space-y-1 overflow-y-auto scroll-thin">
                {captionLines.map((line) => (
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
            {clip.hook ? <p className="mt-2 text-[13px] italic leading-relaxed text-text-2">“{clip.hook}”</p> : null}
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
                {clip.plan.notes.map((note, index) => (
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

function AssetSelect({
  label,
  value,
  options,
  empty,
  onChange,
}: {
  label: string;
  value: string;
  options: { id: string; name: string; category: string }[];
  empty: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      {options.length ? (
        <select className="select" value={value} onChange={(event) => onChange(event.target.value)}>
          <option value="">Pick automatically</option>
          {options.map((asset) => (
            <option key={asset.id} value={asset.id}>
              {asset.name} · {asset.category}
            </option>
          ))}
        </select>
      ) : (
        <Link href="/library" className="block rounded-md border border-dashed border-line px-3 py-2 text-[12px] text-text-3 hover:text-text-2">
          {empty}
        </Link>
      )}
    </label>
  );
}
