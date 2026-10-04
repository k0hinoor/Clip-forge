"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import type { Diagnostics, HardwareReport, SettingsSchemaField, SystemStatus } from "@/lib/types";

/** The sections the page shows, and the settings each one owns. */
const GROUPS: { key: string; label: string; hint: string; fields: string[] }[] = [
  {
    key: "clips",
    label: "Clips",
    hint: "How moments are found and how long they are.",
    fields: [
      "clip_mode",
      "min_clip_seconds",
      "target_clip_seconds",
      "max_clip_seconds",
      "min_score",
      "max_clips",
      "remove_silence",
      "silence_min_duration",
      "silence_threshold_db",
      "auto_zoom",
      "smart_reframe",
      "speaker_tracking",
    ],
  },
  {
    key: "output",
    label: "Output",
    hint: "Frame size, frame rate and encoder quality.",
    fields: ["aspect_ratio", "output_width", "output_height", "output_fps", "render_preset", "crf", "audio_bitrate_kbps", "hw_accel"],
  },
  {
    key: "captions",
    label: "Captions",
    hint: "Subtitles burned into every render.",
    fields: ["captions_enabled", "translate_captions", "translation_language"],
  },
  {
    key: "audio",
    label: "Audio",
    hint: "Voice treatment and loudness.",
    fields: ["normalize_loudness", "target_lufs", "true_peak_db", "voice_boost", "voice_gain_db", "ducking", "music_enabled", "music_volume"],
  },
  {
    key: "transcription",
    label: "Transcription",
    hint: "Local speech-to-text used to find the moments.",
    fields: [
      "whisper_model",
      "whisper_device",
      "whisper_compute_type",
      "language_hint",
      "diarization",
      "max_speakers",
      "word_alignment",
      "llm_enabled",
      "ollama_base_url",
      "ollama_model",
    ],
  },
  {
    key: "sources",
    label: "Downloads & uploads",
    hint: "How source videos are fetched and stored.",
    fields: [
      "max_download_height",
      "prefer_mp4",
      "download_concurrency",
      "max_source_hours",
      "cookies_path",
      "proxy",
      "uploads_enabled",
      "max_upload_gb",
      "cache_downloads",
    ],
  },
  {
    key: "storage",
    label: "Storage & files",
    hint: "Where renders go and how long things are kept.",
    fields: ["export_dir", "export_filename_template", "auto_open_folder", "keep_source_video", "cache_transcripts", "cleanup_days", "max_cache_gb"],
  },
  {
    key: "advanced",
    label: "Advanced",
    hint: "Everything else, including ffmpeg paths and worker counts.",
    fields: [],
  },
];

export default function SettingsPage() {
  const toast = useToast();
  const [schema, setSchema] = useState<Record<string, SettingsSchemaField>>({});
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [hardware, setHardware] = useState<HardwareReport | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [active, setActive] = useState("clips");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [settingsPayload, hardwarePayload, statusPayload, diagnosticsPayload] = await Promise.all([
        api.settings(),
        api.hardware(),
        api.status(),
        api.diagnostics(),
      ]);
      setSchema(settingsPayload.schema);
      setValues(settingsPayload.settings);
      setHardware(hardwarePayload);
      setStatus(statusPayload);
      setDiagnostics(diagnosticsPayload);
    } catch (error) {
      toast.fail(error, "Could not load settings.");
    }
  }, [toast]);

  useEffect(() => {
    load();
  }, [load]);

  const save = async (patch: Record<string, unknown>) => {
    setBusy(true);
    try {
      const payload = await api.updateSettings(patch);
      setValues(payload.settings);
      toast.ok("Saved");
    } catch (error) {
      toast.fail(error, "That setting was rejected.");
    } finally {
      setBusy(false);
    }
  };

  const shown = useMemo(() => {
    const group = GROUPS.find((item) => item.key === active);
    if (!group) return [];
    if (group.key === "advanced") {
      const claimed = new Set(GROUPS.flatMap((item) => item.fields));
      return Object.values(schema).filter((field) => !claimed.has(field.name) && field.type !== "object");
    }
    return group.fields.map((name) => schema[name]).filter(Boolean);
  }, [active, schema]);

  const update = (field: SettingsSchemaField, raw: unknown) => {
    let value: unknown = raw;
    if (field.type === "integer") value = Number.parseInt(String(raw), 10) || 0;
    else if (field.type === "number") value = Number.parseFloat(String(raw)) || 0;
    save({ [field.name]: value });
  };

  const group = GROUPS.find((item) => item.key === active);

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-lg font-semibold tracking-tight text-text">Settings</h1>
          <p className="mt-0.5 text-xs text-text-3">Saved on this machine. Changes apply to the next analysis or render.</p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => api.resetSettings().then(load)}>
          <RefreshCw size={14} /> Restore defaults
        </button>
      </header>

      <section className="card p-4">
        <h2 className="panel-title">This machine</h2>
        <dl className="mt-2 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
          <Row label="CPU" value={hardware?.hardware.cpu?.name ?? "detecting…"} />
          <Row
            label="Memory"
            value={`${hardware?.hardware.cpu?.logical_cores ?? "?"} threads · ${hardware?.hardware.memory?.total_gb ?? "?"} GB`}
          />
          <Row
            label="GPU"
            value={
              hardware?.hardware.gpu?.available
                ? `${hardware.hardware.gpu.devices?.[0]?.name ?? hardware.hardware.gpu.vendor}${hardware.hardware.gpu.cuda ? " · CUDA" : ""}`
                : "none — CPU encoding"
            }
          />
          <Row label="Disk free" value={`${hardware?.hardware.disk?.free_gb ?? "?"} GB`} />
          <Row
            label="FFmpeg"
            value={
              status?.ffmpeg?.available
                ? `${status.ffmpeg.version?.split(" ")[0] ?? "ready"}${status.ffmpeg.libass ? " · libass" : " · no libass (captions cannot be burned in)"}`
                : "missing"
            }
          />
          <Row label="Speech-to-text" value={status?.ai?.faster_whisper ? "faster-whisper" : "not installed"} />
          <Row label="Local LLM" value={values.llm_enabled ? String(values.ollama_model ?? "enabled") : "disabled"} />
          <Row label="Data folder" value={diagnostics?.paths?.data_dir ?? status?.data_dir ?? ""} mono />
        </dl>
        {status?.notes?.length ? (
          <ul className="mt-3 space-y-1 border-t border-line pt-2">
            {status.notes.map((note, index) => (
              <li key={index} className="text-[11px] text-warn" title={note.detail}>
                · {note.title}
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      <nav className="flex flex-wrap gap-1 border-b border-line">
        {GROUPS.map((item) => (
          <button
            key={item.key}
            type="button"
            onClick={() => setActive(item.key)}
            className={`-mb-px border-b-2 px-3 py-2 text-[13px] font-medium transition-colors ${
              active === item.key ? "border-accent text-text" : "border-transparent text-text-3 hover:text-text-2"
            }`}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <section className="card p-4">
        {group?.hint ? <p className="mb-3 text-xs text-text-3">{group.hint}</p> : null}
        {shown.length === 0 ? (
          <p className="text-xs text-text-3">Nothing here yet.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {shown.map((field) => (
              <Field key={field.name} field={field} value={values[field.name] ?? field.value} onCommit={(value) => update(field, value)} />
            ))}
          </div>
        )}
        {busy ? <p className="mt-3 text-[11px] text-text-3">Saving…</p> : null}
      </section>

      {diagnostics?.errors?.length ? (
        <section className="card p-4">
          <h2 className="panel-title">Recent errors</h2>
          <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto scroll-thin">
            {diagnostics.errors.slice(0, 12).map((error, index) => (
              <li key={index} className="text-[11px] text-text-3">
                <span className="mono mr-2">{error.at?.slice(11, 19)}</span>
                {error.message}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex gap-2">
      <dt className="w-28 shrink-0 text-text-3">{label}</dt>
      <dd className={`min-w-0 flex-1 truncate text-text-2 ${mono ? "mono text-[11px]" : ""}`} title={value}>
        {value}
      </dd>
    </div>
  );
}

function Field({
  field,
  value,
  onCommit,
}: {
  field: SettingsSchemaField;
  value: unknown;
  onCommit: (value: unknown) => void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);

  if (field.type === "boolean") {
    return (
      <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-surface-2 px-3 py-2">
        <span className="min-w-0">
          <span className="block text-[13px] font-medium text-text">{field.label}</span>
          {field.help ? <span className="block truncate text-[11px] text-text-3">{field.help}</span> : null}
        </span>
        <input type="checkbox" className="h-4 w-4 accent-accent" checked={Boolean(value)} onChange={(event) => onCommit(event.target.checked)} />
      </label>
    );
  }

  if (field.type === "enum") {
    const options = field.options.map((option) => (typeof option === "string" ? option : String((option as any).name ?? "")));
    return (
      <label className="block">
        <span className="label">{field.label}</span>
        <select className="select" value={String(value ?? "")} onChange={(event) => onCommit(event.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option || "(auto)"}
            </option>
          ))}
        </select>
      </label>
    );
  }

  if (field.type === "number" || field.type === "integer") {
    return (
      <label className="block">
        <span className="label">{field.label}</span>
        <input
          className="input"
          type="number"
          step={field.type === "integer" ? 1 : 0.1}
          min={field.ge ?? field.gt}
          max={field.le ?? field.lt}
          value={Number(draft ?? 0)}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => onCommit(draft)}
        />
      </label>
    );
  }

  return (
    <label className="block">
      <span className="label">{field.label}</span>
      <input
        className="input"
        value={String(draft ?? "")}
        placeholder={field.default ? String(field.default) : ""}
        spellCheck={false}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => onCommit(draft)}
      />
    </label>
  );
}
