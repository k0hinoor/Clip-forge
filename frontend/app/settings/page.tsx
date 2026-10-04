"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Cpu, Gauge, HardDrive, RefreshCw, Save, Server, Terminal, Wand2 } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { bytes } from "@/lib/format";
import type { Diagnostics, HardwareReport, SettingsSchemaField, SystemStatus } from "@/lib/types";

const SECTION_LABELS: Record<string, string> = {
  general: "General",
  ai: "AI & analysis",
  transcription: "Transcription",
  video: "Video & clips",
  captions: "Captions",
  gameplay: "Gameplay & layouts",
  audio: "Audio",
  export: "Export",
  storage: "Storage",
  advanced: "Advanced",
};

export default function SettingsPage() {
  const toast = useToast();
  const [schema, setSchema] = useState<Record<string, SettingsSchemaField>>({});
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [hardware, setHardware] = useState<HardwareReport | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [templates, setTemplates] = useState<any[]>([]);
  const [ollama, setOllama] = useState<{ models: string[]; error?: string }>({ models: [] });
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [busy, setBusy] = useState(false);
  const [activeSection, setActiveSection] = useState("video");

  const load = useCallback(async () => {
    try {
      const [schemaPayload, settingsPayload, hardwarePayload, statusPayload, templatePayload, diagnosticsPayload] =
        await Promise.all([
          api.settingsSchema(),
          api.settings(),
          api.hardware(),
          api.status(),
          api.templates(),
          api.diagnostics(),
        ]);
      setSchema(schemaPayload.schema);
      setValues(settingsPayload.settings);
      setHardware(hardwarePayload);
      setStatus(statusPayload);
      setTemplates(templatePayload.templates);
      setDiagnostics(diagnosticsPayload);
      setOllama(await api.ollamaModels().catch((error) => ({ models: [], error: String(error?.message ?? error) })));
    } catch (error) {
      toast.fail(error, "Could not load settings.");
    }
  }, [toast]);

  useEffect(() => {
    load();
  }, [load]);

  const sections = useMemo(() => {
    const grouped: Record<string, SettingsSchemaField[]> = {};
    for (const field of Object.values(schema)) {
      grouped[field.section] = grouped[field.section] ?? [];
      grouped[field.section].push(field);
    }
    return grouped;
  }, [schema]);

  const save = async (patch: Record<string, unknown>) => {
    setBusy(true);
    try {
      const payload = await api.updateSettings(patch);
      setValues(payload.settings);
      toast.ok("Settings saved");
    } catch (error) {
      toast.fail(error, "That setting was rejected.");
    } finally {
      setBusy(false);
    }
  };

  const update = (field: SettingsSchemaField, raw: unknown) => {
    let value: unknown = raw;
    if (field.type === "integer") value = Number.parseInt(String(raw), 10) || 0;
    else if (field.type === "number") value = Number.parseFloat(String(raw)) || 0;
    save({ [field.name]: value });
  };

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-xl font-black text-mist-200">Settings</h1>
          <p className="mt-0.5 text-xs text-mist-400">Stored locally in SQLite · nothing is uploaded anywhere.</p>
        </div>
        <button type="button" className="btn btn-ghost" onClick={() => api.resetSettings().then(load)}>
          <RefreshCw size={14} /> Restore defaults
        </button>
      </header>

      {/* ------------------------------------------------------ hardware */}
      <section className="grid gap-3 lg:grid-cols-3">
        <div className="card p-4">
          <h2 className="flex items-center gap-2 text-sm font-bold text-mist-200">
            <Cpu size={15} /> This machine
          </h2>
          <ul className="mt-2 space-y-1 text-xs text-mist-400">
            <li>{hardware?.hardware.cpu?.name ?? "detecting…"}</li>
            <li>
              {hardware?.hardware.cpu?.logical_cores ?? "?"} threads
              ({hardware?.hardware.cpu?.physical_cores ?? "?"} cores) · {hardware?.hardware.memory?.total_gb ?? "?"} GB RAM
            </li>
            <li>
              GPU:{" "}
              {hardware?.hardware.gpu?.available
                ? `${hardware.hardware.gpu.devices?.[0]?.name ?? hardware.hardware.gpu.vendor}${hardware.hardware.gpu.cuda ? " (CUDA)" : ""}`
                : "none detected - CPU encoding"}
            </li>
            <li>
              Disk: {hardware?.hardware.disk?.free_gb ?? "?"} GB free of {hardware?.hardware.disk?.total_gb ?? "?"} GB
            </li>
          </ul>
        </div>
        <div className="card p-4">
          <h2 className="flex items-center gap-2 text-sm font-bold text-mist-200">
            <Wand2 size={15} /> Recommended
          </h2>
          <ul className="mt-2 space-y-1 text-xs text-mist-400">
            <li>Whisper model: {hardware?.hardware.recommended?.whisper_model ?? "…"}</li>
            <li>
              Device: {hardware?.hardware.recommended?.whisper_device ?? "…"} ·{" "}
              {hardware?.hardware.recommended?.hw_accel === "none" ? "CPU encode" : hardware?.hardware.recommended?.hw_accel}
            </li>
            <li className="pt-1 text-[11px] leading-relaxed">
              {hardware?.hardware.recommended?.concurrency ?? 1} clip render{hardware?.hardware.recommended?.concurrency === 1 ? "" : "s"} at a
              time
            </li>
          </ul>
        </div>
        <div className="card p-4">
          <h2 className="flex items-center gap-2 text-sm font-bold text-mist-200">
            <Gauge size={15} /> Services
          </h2>
          <ul className="mt-2 space-y-1 text-xs text-mist-400">
            <li>FFmpeg: {status?.ffmpeg?.available ? `${status.ffmpeg.version || "ready"}${status.ffmpeg.libass ? " · libass" : ""}` : "missing"}</li>
            <li>Speech-to-text: {status?.ai?.faster_whisper ? "faster-whisper ready" : "not installed"}</li>
            <li>Vision: {status?.ai?.opencv ? "OpenCV ready" : "optional, not installed"}</li>
            <li>
              Ollama:{" "}
              {!values.llm_enabled
                ? "disabled in AI & analysis"
                : ollama.models.length
                  ? `${ollama.models.length} model${ollama.models.length === 1 ? "" : "s"} · ${String(values.llm_model ?? "")}`
                  : ollama.error || "not reachable"}
            </li>
          </ul>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              className="btn btn-quiet text-xs"
              onClick={async () => {
                const result = await api
                  .testOllama("", String(values.llm_model ?? ""))
                  .catch(() => null);
                const models = await api.ollamaModels().catch(() => ({ models: [], error: "unreachable" }));
                setOllama(models);
                toast.ok(result ? "Ollama answered" : "Ollama check finished", models.error);
              }}
            >
              <Server size={13} /> Test Ollama
            </button>
            <button type="button" className="btn btn-quiet text-xs" onClick={() => api.logs("render", 200).then((payload) => toast.push({ kind: "info", title: `${payload.lines.length} render log lines`, hint: payload.lines.slice(-3).join(" ⏎ ") }))}>
              <Terminal size={13} /> Recent render log
            </button>
          </div>
          {ollama.models?.length ? <p className="mt-2 text-[11px] text-mist-400">Installed models: {ollama.models.join(", ")}</p> : null}
        </div>
      </section>

      {/* ------------------------------------------------------ templates */}
      <section className="card p-4">
        <h2 className="flex items-center gap-2 text-sm font-bold text-mist-200">
          <HardDrive size={15} /> Templates
        </h2>
        <div className="mt-2 flex flex-wrap gap-2">
          {templates.map((template) => (
            <button
              key={template.id}
              type="button"
              className="btn btn-ghost text-xs"
              title={template.description}
              onClick={async () => {
                const payload = await api.applyTemplate(template.id).catch((error) => {
                  toast.fail(error);
                  return null;
                });
                if (payload) {
                  toast.ok(`${template.name} applied`);
                  setValues(payload.settings ?? values);
                  await load();
                }
              }}
            >
              {template.name}
            </button>
          ))}
          {templates.length === 0 ? <p className="text-xs text-mist-400">No saved templates yet.</p> : null}
        </div>
      </section>

      {/* ------------------------------------------------------- sections */}
      <div className="flex flex-wrap gap-1">
        {Object.keys(sections).map((section) => (
          <button
            key={section}
            type="button"
            className={`btn ${activeSection === section ? "btn-ghost" : "btn-quiet"} text-xs`}
            onClick={() => setActiveSection(section)}
          >
            {SECTION_LABELS[section] ?? section}
          </button>
        ))}
      </div>

      <section className="card p-4">
        <div className="grid gap-4 md:grid-cols-2">
          {(sections[activeSection] ?? []).map((field) => (
            <Field
              key={field.name}
              field={field}
              value={values[field.name] ?? field.value}
              onCommit={(value) => update(field, value)}
            />
          ))}
        </div>
        <p className="mt-4 flex items-center gap-2 text-[11px] text-mist-400">
          <Save size={12} /> Changes save immediately and apply to the next analysis or render.
        </p>
      </section>

      <section className="card p-4">
        <h2 className="text-sm font-bold text-mist-200">Storage</h2>
        <ul className="mt-2 space-y-1 text-xs text-mist-400">
          <li>Data directory: <span className="mono">{diagnostics?.paths?.data_dir ?? status?.data_dir}</span></li>
          <li>Exports: <span className="mono">{diagnostics?.paths?.exports}</span></li>
          <li>Database: <span className="mono">{diagnostics?.paths?.database}</span></li>
          {status?.ai?.faster_whisper_version ? <li>faster-whisper {status.ai.faster_whisper_version}</li> : null}
        </ul>
        {status?.notes?.length ? (
          <ul className="mt-2 space-y-1 text-[11px] text-amber-glow">
            {status.notes.map((note, index) => (
              <li key={index} title={note.detail}>
                · {note.title}
              </li>
            ))}
          </ul>
        ) : null}
      </section>
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
      <div className="flex items-center justify-between gap-3 rounded-lg border border-ink-700/70 p-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-mist-200">{field.label}</p>
          {field.help ? <p className="mt-0.5 text-[11px] text-mist-400">{field.help}</p> : null}
        </div>
        <button
          type="button"
          onClick={() => onCommit(!value)}
          className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${value ? "bg-signal-500/80" : "bg-ink-600"}`}
        >
          <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-mist-200 ${value ? "left-[1.15rem]" : "left-0.5"}`} />
        </button>
      </div>
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
        {field.help ? <span className="mt-1 block text-[11px] text-mist-400">{field.help}</span> : null}
      </label>
    );
  }

  if (field.type === "number" || field.type === "integer") {
    const min = field.ge ?? field.gt ?? 0;
    const max = field.le ?? (min + (field.type === "integer" ? 100 : 10));
    const step = field.type === "integer" ? 1 : (max - min) / 200;
    return (
      <label className="block">
        <span className="label">
          {field.label} · <span className="mono text-mist-300">{Number(draft).toFixed(field.type === "integer" ? 0 : 1)}</span>
        </span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={Number(draft) || min}
          onChange={(event) => setDraft(event.target.value)}
          onMouseUp={() => onCommit(draft)}
          onTouchEnd={() => onCommit(draft)}
          className="w-full accent-flare-500"
        />
        {field.help ? <span className="mt-1 block text-[11px] text-mist-400">{field.help}</span> : null}
      </label>
    );
  }

  if (field.type === "object") {
    return (
      <div className="rounded-lg border border-ink-700/70 p-3">
        <p className="text-xs font-semibold text-mist-200">{field.label}</p>
        <p className="mt-1 text-[11px] text-mist-400">
          Configured from the caption editor on a clip (presets, fonts, colours, animation).
        </p>
      </div>
    );
  }

  return (
    <label className="block">
      <span className="label">{field.label}</span>
      <input
        className="input"
        value={String(draft ?? "")}
        placeholder={field.default ? String(field.default) : ""}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => onCommit(draft)}
      />
      {field.help ? <span className="mt-1 block text-[11px] text-mist-400">{field.help}</span> : null}
    </label>
  );
}
