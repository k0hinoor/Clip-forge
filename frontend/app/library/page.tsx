"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FolderInput, RefreshCw, Trash2, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { bytes } from "@/lib/format";
import type { Asset } from "@/lib/types";

const KINDS = [
  { key: "gameplay", label: "Gameplay" },
  { key: "broll", label: "B-roll" },
  { key: "music", label: "Music" },
];

export default function LibraryPage() {
  const toast = useToast();
  const [kind, setKind] = useState("gameplay");
  const [category, setCategory] = useState("");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [library, setLibrary] = useState<any>(null);
  const [folders, setFolders] = useState<any>(null);
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const payload = await api.assets(kind, category);
      setAssets(payload.assets);
      setLibrary(payload.library);
      if (!folders) setFolders(await api.gameplayAssets().then((result) => result.folders));
    } catch (error) {
      toast.fail(error, "Could not read the library.");
    }
  }, [kind, category, folders, toast]);

  useEffect(() => {
    load();
  }, [load]);

  const upload = async (file: File) => {
    setBusy(true);
    try {
      await api.uploadAsset(file, kind, category || file.name.split(/[_\-. ]/)[0] || "general", file.name);
      toast.ok("Added to the library");
      await load();
    } catch (error) {
      toast.fail(error, "The file could not be added.");
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const importPath = async () => {
    if (!path.trim()) return;
    setBusy(true);
    try {
      await api.importAssetPath({ path: path.trim(), kind, category: category || "general", copy: true });
      toast.ok("Imported");
      setPath("");
      await load();
    } catch (error) {
      toast.fail(error, "That path could not be imported.");
    } finally {
      setBusy(false);
    }
  };

  const categories: string[] = folders?.[kind] ?? [];

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-semibold tracking-tight text-text">Library</h1>
        <p className="mt-0.5 max-w-2xl text-xs text-text-3">
          Background footage and music for the split-screen layouts. Clipforge never downloads copyrighted material — add
          files you are licensed to use.
        </p>
      </header>

      <section className="card flex flex-wrap items-end gap-3 p-4">
        <div>
          <span className="label">Kind</span>
          <select
            className="select"
            value={kind}
            onChange={(event) => {
              setKind(event.target.value);
              setCategory("");
            }}
          >
            {KINDS.map((item) => (
              <option key={item.key} value={item.key}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <span className="label">Category</span>
          <select className="select" value={category} onChange={(event) => setCategory(event.target.value)}>
            <option value="">All</option>
            {categories.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
        <div className="min-w-[14rem] flex-1">
          <span className="label">Import a file from this machine</span>
          <input
            className="input"
            placeholder="/footage/subway_surfer/clip_04.mp4"
            value={path}
            onChange={(event) => setPath(event.target.value)}
            spellCheck={false}
          />
        </div>
        <button type="button" className="btn btn-secondary" onClick={importPath} disabled={busy}>
          <FolderInput size={14} /> Import
        </button>
        <button type="button" className="btn btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
          <Upload size={14} /> Upload
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-icon"
          onClick={async () => {
            await api.scanAssets();
            await load();
          }}
          aria-label="Rescan folders"
        >
          <RefreshCw size={14} />
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="video/*,audio/*"
          hidden
          onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])}
        />
      </section>

      {library?.counts ? (
        <p className="flex flex-wrap gap-2 text-[11px] text-text-3">
          <span className="chip">gameplay {library.counts.gameplay}</span>
          <span className="chip">b-roll {library.counts.broll}</span>
          <span className="chip">music {library.counts.music}</span>
          <span className="chip">{bytes(library.total_bytes ?? 0)}</span>
        </p>
      ) : null}

      {assets.length === 0 ? (
        <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
          <p className="text-sm font-medium text-text-2">No {kind} files yet</p>
          <p className="max-w-md text-xs text-text-3">
            Upload here, or drop files into <span className="mono">assets/{kind}/&lt;category&gt;/</span> in the data folder
            and press rescan. The category name helps the picker choose fitting backgrounds.
          </p>
        </div>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {assets.map((asset) => (
            <li key={asset.id} className="card overflow-hidden">
              {asset.kind === "music" ? (
                <div className="flex h-20 items-center justify-center border-b border-line bg-surface-2 px-3">
                  <audio className="w-full" controls src={`/api/assets/${asset.id}/file`} />
                </div>
              ) : (
                <video className="h-32 w-full object-cover" muted preload="metadata" src={`/api/assets/${asset.id}/file`} />
              )}
              <div className="space-y-1.5 p-3">
                <p className="truncate text-[13px] font-medium text-text">{asset.name}</p>
                <p className="flex flex-wrap gap-2 text-[11px] text-text-3">
                  <span className="chip">{asset.category}</span>
                  <span>{bytes(asset.size)}</span>
                  {asset.duration ? <span>{asset.duration.toFixed(1)}s</span> : null}
                  {asset.width ? (
                    <span>
                      {asset.width}×{asset.height}
                    </span>
                  ) : null}
                </p>
                <div className="flex items-center gap-1 pt-1">
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon"
                    onClick={async () => {
                      if (!window.confirm(`Remove ${asset.name}?`)) return;
                      await api.deleteAsset(asset.id);
                      await load();
                    }}
                    aria-label={`Remove ${asset.name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                  <span className="ml-auto truncate text-[10px] text-text-3">{asset.licence || asset.source || "local file"}</span>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
