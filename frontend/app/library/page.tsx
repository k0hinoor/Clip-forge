"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FolderInput, Loader2, RefreshCw, Trash2, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { useSystem } from "@/components/System";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { bytes, duration } from "@/lib/format";
import type { Asset, AssetKind, AssetLibrary } from "@/lib/types";

const KINDS: { key: AssetKind; label: string; accept: string }[] = [
  { key: "gameplay", label: "Gameplay", accept: "video/*,.mp4,.mov,.mkv,.webm,.m4v" },
  { key: "broll", label: "B-roll", accept: "video/*,image/*,.mp4,.mov,.mkv,.webm,.jpg,.png" },
  { key: "music", label: "Music", accept: "audio/*,.mp3,.wav,.m4a,.aac,.ogg,.opus,.flac" },
];

const pretty = (value: string) => value.replace(/_/g, " ");

export default function LibraryPage() {
  const toast = useToast();
  const { features } = useSystem();
  const [kind, setKind] = useState<AssetKind>("gameplay");
  const [category, setCategory] = useState("");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [library, setLibrary] = useState<AssetLibrary | null>(null);
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const payload = await api.assets(kind, category);
      setAssets(payload.assets);
      setLibrary(payload.library);
      setCategories(payload.categories ?? {});
    } catch (error) {
      toast.fail(error, "Could not read the library.");
    }
  }, [kind, category, toast]);

  useEffect(() => {
    load();
  }, [load]);

  const guarded = async (action: () => Promise<void>) => {
    setBusy(true);
    try {
      await action();
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const upload = (file: File) =>
    guarded(async () => {
      setProgress(0);
      try {
        await api.uploadAsset(file, kind, category || "general", file.name.replace(/\.[^.]+$/, ""), setProgress);
      } finally {
        if (fileInput.current) fileInput.current.value = "";
      }
      toast.ok("Added to the library", file.name);
      await load();
    });

  const importPath = () =>
    guarded(async () => {
      if (!path.trim()) return;
      await api.importAssetPath({ path: path.trim(), kind, category: category || "general", copy: true });
      toast.ok("Imported");
      setPath("");
      await load();
    });

  const rescan = () =>
    guarded(async () => {
      const result = await api.scanAssets();
      toast.ok(result.imported.length ? `${result.imported.length} new files found` : "No new files in the asset folders");
      await load();
    });

  const toggle = (asset: Asset) =>
    guarded(async () => {
      await api.updateAsset(asset.id, { enabled: !asset.enabled });
      await load();
    });

  const remove = (asset: Asset) =>
    guarded(async () => {
      if (!window.confirm(`Remove “${asset.name}” from the library?`)) return;
      await api.deleteAsset(asset.id);
      toast.ok("Removed");
      await load();
    });

  const kindInfo = KINDS.find((item) => item.key === kind) ?? KINDS[0];
  const kindCategories = categories[kind] ?? [];

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-semibold tracking-tight text-text">Library</h1>
        <p className="mt-0.5 max-w-2xl text-xs text-text-3">
          Background footage and music for the split-screen layouts. Clipforge never downloads copyrighted material — add files
          you are licensed to use.
        </p>
      </header>

      <section className="card space-y-3 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="block">
            <span className="label">Kind</span>
            <select
              className="select"
              value={kind}
              onChange={(event) => {
                setKind(event.target.value as AssetKind);
                setCategory("");
              }}
            >
              {KINDS.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="label">{kind === "music" ? "Mood" : "Category"}</span>
            <select className="select" value={category} onChange={(event) => setCategory(event.target.value)}>
              <option value="">All</option>
              {kindCategories.map((item) => (
                <option key={item} value={item}>
                  {pretty(item)}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className="btn btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
            {progress !== null ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />} Upload {kindInfo.label.toLowerCase()}
          </button>
          <button type="button" className="btn btn-ghost btn-icon" onClick={rescan} disabled={busy} aria-label="Rescan the asset folders" title="Rescan the asset folders">
            <RefreshCw size={14} />
          </button>
          <input
            ref={fileInput}
            type="file"
            accept={kindInfo.accept}
            hidden
            onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])}
          />
        </div>
        {progress !== null ? (
          <div className="flex items-center gap-2">
            <ProgressBar value={progress} className="max-w-sm" />
            <span className="text-[11px] text-text-3">{progress < 1 ? `Uploading · ${Math.round(progress * 100)}%` : "Processing…"}</span>
          </div>
        ) : null}
        <p className="text-[11px] text-text-3">
          New uploads go into {category ? <span className="mono">{pretty(category)}</span> : "the selected category (or general)"}. The category
          helps the picker match footage to each clip.
        </p>

        {features.local_paths ? (
          <div className="flex flex-wrap items-end gap-3 border-t border-line pt-3">
            <label className="block min-w-[14rem] flex-1">
              <span className="label">Import a file from this computer</span>
              <input
                className="input"
                placeholder="/footage/subway_surfer/clip_04.mp4"
                value={path}
                onChange={(event) => setPath(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && importPath()}
                spellCheck={false}
              />
            </label>
            <button type="button" className="btn btn-secondary" onClick={importPath} disabled={busy || !path.trim()}>
              <FolderInput size={14} /> Import
            </button>
          </div>
        ) : null}
      </section>

      {library?.counts ? (
        <p className="flex flex-wrap gap-2 text-[11px] text-text-3">
          <span className="chip">gameplay {library.counts.gameplay}</span>
          <span className="chip">b-roll {library.counts.broll}</span>
          <span className="chip">music {library.counts.music}</span>
          <span className="chip">{duration(library.total_seconds)} of media</span>
        </p>
      ) : null}

      {assets.length === 0 ? (
        <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
          <p className="text-sm font-medium text-text-2">No {kindInfo.label.toLowerCase()} {category ? `in ${pretty(category)}` : "files"} yet</p>
          <p className="max-w-md text-xs text-text-3">
            Upload here{features.local_paths ? (
              <>
                , or drop files into <span className="mono">assets/{kind}/&lt;category&gt;/</span> in the data folder and press rescan
              </>
            ) : null}
            .
          </p>
        </div>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {assets.map((asset) => (
            <li key={asset.id} className={`card overflow-hidden ${asset.enabled ? "" : "opacity-60"}`}>
              {asset.kind === "music" ? (
                <div className="flex h-20 items-center justify-center border-b border-line bg-surface-2 px-3">
                  <audio className="w-full" controls preload="none" src={asset.stream_url} />
                </div>
              ) : (
                <video className="h-32 w-full bg-black object-cover" muted controls preload="metadata" src={asset.stream_url} />
              )}
              <div className="space-y-1.5 p-3">
                <p className="truncate text-[13px] font-medium text-text" title={asset.filename}>
                  {asset.name}
                </p>
                <p className="flex flex-wrap gap-2 text-[11px] text-text-3">
                  <span className="chip">{pretty(asset.category)}</span>
                  <span>{bytes(asset.size_bytes)}</span>
                  {asset.duration ? <span>{duration(asset.duration)}</span> : null}
                  {asset.width ? (
                    <span>
                      {asset.width}×{asset.height}
                    </span>
                  ) : null}
                </p>
                <div className="flex items-center gap-2 pt-1">
                  <label className="flex items-center gap-1.5 text-[11px] text-text-2">
                    <input type="checkbox" className="accent-accent" checked={asset.enabled} onChange={() => toggle(asset)} disabled={busy} />
                    Use in clips
                  </label>
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon ml-auto"
                    onClick={() => remove(asset)}
                    disabled={busy}
                    aria-label={`Remove ${asset.name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
