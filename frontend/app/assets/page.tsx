"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FolderInput, LibraryBig, RefreshCw, Star, Trash2, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { bytes } from "@/lib/format";
import type { Asset } from "@/lib/types";

const KINDS = [
  { key: "gameplay", label: "Gameplay" },
  { key: "broll", label: "B-roll" },
  { key: "music", label: "Music" },
];

export default function AssetsPage() {
  const toast = useToast();
  const [kind, setKind] = useState("gameplay");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [library, setLibrary] = useState<any>(null);
  const [folders, setFolders] = useState<any>(null);
  const [category, setCategory] = useState("");
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
      toast.fail(error, "Could not read the asset library.");
    }
  }, [kind, category, folders, toast]);

  useEffect(() => {
    load();
  }, [load]);

  const upload = async (file: File) => {
    setBusy(true);
    try {
      await api.uploadAsset(file, kind, category || file.name.split(/[_\-. ]/)[0] || "general", file.name);
      toast.ok("Asset added", "Only import footage you have the rights to use.");
      await load();
    } catch (error) {
      toast.fail(error, "The asset could not be added.");
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
    <div className="space-y-5">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-xl font-black text-mist-200">Asset library</h1>
          <p className="mt-0.5 text-xs text-mist-400">
            CLIPFORGE never downloads copyrighted material. Add your own gameplay, B-roll and licensed music here.
          </p>
        </div>
        <button type="button" className="btn btn-ghost" onClick={async () => { await api.scanAssets(); await load(); }}>
          <RefreshCw size={14} /> Rescan folders
        </button>
      </header>

      <section className="card flex flex-wrap items-end gap-3 p-4">
        <div>
          <span className="label">Kind</span>
          <select className="select" value={kind} onChange={(event) => { setKind(event.target.value); setCategory(""); }}>
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
        <div className="min-w-[16rem] flex-1">
          <span className="label">Import from a folder on this PC</span>
          <input
            className="input"
            placeholder="D:\\footage\\subway_surfer\\clip_04.mp4"
            value={path}
            onChange={(event) => setPath(event.target.value)}
          />
        </div>
        <button type="button" className="btn btn-ghost" onClick={importPath} disabled={busy}>
          <FolderInput size={14} /> Import path
        </button>
        <button type="button" className="btn btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
          <Upload size={14} /> Upload
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
        <div className="flex flex-wrap gap-2 text-[11px] text-mist-400">
          <span className="chip">gameplay {library.counts.gameplay}</span>
          <span className="chip">b-roll {library.counts.broll}</span>
          <span className="chip">music {library.counts.music}</span>
          <span className="chip">total {bytes(library.total_bytes ?? 0)}</span>
        </div>
      ) : null}

      {assets.length === 0 ? (
        <div className="card flex flex-col items-center gap-2 p-10 text-center">
          <LibraryBig size={24} className="text-mist-400" />
          <p className="text-sm font-semibold text-mist-300">No {kind} assets yet</p>
          <p className="max-w-lg text-xs text-mist-400">
            Drop files into <span className="mono">{`assets/${kind}/<category>/`}</span> inside your CLIPFORGE data folder,
            or upload them here. Categories such as subway_surfer, temple_run, parkour or satisfying help the picker choose
            fitting backgrounds.
          </p>
        </div>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {assets.map((asset) => (
            <li key={asset.id} className="card-tight overflow-hidden">
              {asset.kind === "music" ? (
                <div className="flex h-24 items-center justify-center bg-ink-900">
                  <audio className="w-[92%]" controls src={`/api/assets/${asset.id}/file`} />
                </div>
              ) : (
                <video className="h-32 w-full bg-black object-cover" muted preload="metadata" src={`/api/assets/${asset.id}/file`} />
              )}
              <div className="space-y-1.5 p-3">
                <p className="truncate text-sm font-semibold text-mist-200">{asset.name}</p>
                <p className="flex flex-wrap gap-2 text-[11px] text-mist-400">
                  <span className="chip">{asset.category}</span>
                  <span>{bytes(asset.size)}</span>
                  {asset.duration ? <span>{asset.duration.toFixed(1)}s</span> : null}
                  {asset.width ? <span>{asset.width}×{asset.height}</span> : null}
                </p>
                <div className="flex items-center gap-1 pt-1">
                  <button
                    type="button"
                    className={`btn btn-quiet p-1.5 ${asset.favorite ? "text-amber-glow" : ""}`}
                    onClick={async () => {
                      await api.updateAsset(asset.id, { favorite: !asset.favorite });
                      await load();
                    }}
                    aria-label="Favourite"
                  >
                    <Star size={14} />
                  </button>
                  <button
                    type="button"
                    className="btn btn-quiet p-1.5"
                    onClick={async () => {
                      if (!window.confirm(`Remove ${asset.name}?`)) return;
                      await api.deleteAsset(asset.id);
                      await load();
                    }}
                    aria-label="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                  <span className="ml-auto text-[10px] text-mist-400">{asset.licence || asset.source || "local file"}</span>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
