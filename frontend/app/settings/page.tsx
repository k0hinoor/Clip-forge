"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { HardDrive, Loader2, PlugZap, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { CaptionStyleSelect } from "@/components/Controls";
import { useSystem } from "@/components/System";
import { useToast } from "@/components/Toast";
import { bytes } from "@/lib/format";
import type { CaptionPlan, CaptionPresetInfo, CleanupReport, Diagnostics, HardwareReport, SettingsSchemaField } from "@/lib/types";

/** The sections the page shows, and the settings each one owns. */
const GROUPS: { key: string; label: string; hint: string; fields: string[] }[] = [
  {
    key: "clips",
    label: "Clips",
    hint: "How moments are found and how long they are.",
    fields: [
      "clip_mode",
      "min_score",
      "max_clips",
      "min_clip_seconds",
      "target_clip_seconds",
      "max_clip_seconds",
      "remove_silence",
      "silence_min_duration",
      "silence_threshold_db",
      "auto_zoom",
      "smart_reframe",
      "speaker_tracking",
    ],
  },
  {
    key: "layout",
    label: "Layout",
    hint: "Default framing and the background footage used by split layouts.",
    fields: [
      "layout",
      "split_ratio",
      "gameplay_enabled",
      "gameplay_mode",
      "gameplay_category",
      "gameplay_pace",
      "gameplay_volume",
      "broll_enabled",
      "broll_category",
    ],
  },
  {
    key: "output",
    label: "Output",
    hint: "Frame size, frame rate and encoder quality.",
    fields: ["aspect_ratio", "output_width", "output_height", "output_fps", "allow_60fps", "render_preset", "crf", "video_bitrate_kbps", "audio_bitrate_kbps", "hw_accel"],
  },
  {
    key: "captions",
    label: "Captions",
    hint: "Subtitles burned into every render.",
    fields: ["captions_enabled", "translate_captions"],
  },
  {
    key: "audio",
    label: "Audio",
    hint: "Voice treatment, loudness and background music.",
    fields: [
      "normalize_loudness",
      "target_lufs",
      "true_peak_db",
      "voice_boost",
      "voice_gain_db",
      "noise_reduction",
      "music_enabled",
      "music_mood",
      "music_volume",
      "ducking",
      "ducking_db",
    ],
  },
  {
    key: "transcription",
    label: "Transcription & AI",
    hint: "Speech-to-text used to find the moments, and the optional local LLM.",
    fields: [
      "whisper_model",
      "whisper_device",
      "whisper_compute_type",
      "whisper_beam_size",
      "language_hint",
      "whisper_initial_prompt",
      "diarization",
      "max_speakers",
      "word_alignment",
      "keep_source_audio",
      "llm_enabled",
      "ollama_base_url",
      "ollama_model",
    ],
  },
  {
    key: "sources",
    label: "Downloads & uploads",
    hint: "How source videos are fetched and accepted.",
    fields: ["max_download_height", "prefer_mp4", "download_concurrency", "max_source_hours", "cookies_path", "proxy", "uploads_enabled", "max_upload_gb", "cache_downloads"],
  },
  {
    key: "storage",
    label: "Storage",
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

const DESKTOP_ONLY = new Set(["auto_open_folder"]);
const WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"];

export default function SettingsPage() {
  const toast = useToast();
  const { features, status, refresh: refreshStatus } = useSystem();
  const [schema, setSchema] = useState<Record<string, SettingsSchemaField>>({});
  const [values, setValues] = useState<Record<string, any>>({});
  const [presets, setPresets] = useState<Record<string, CaptionPresetInfo>>({});
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const [hardware, setHardware] = useState<HardwareReport | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [active, setActive] = useState("clips");
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      const [settingsPayload, hardwarePayload, diagnosticsPayload, assetPayload] = await Promise.all([
        api.settings(),
        api.hardware(),
        api.diagnostics(),
        api.assets().catch(() => null),
      ]);
      setSchema(settingsPayload.schema);
      setValues(settingsPayload.settings);
      setPresets(settingsPayload.caption_presets ?? {});
      setHardware(hardwarePayload);
      setDiagnostics(diagnosticsPayload);
      if (assetPayload) setCategories(assetPayload.categories ?? {});
    } catch (error) {
      toast.fail(error, "Could not load settings.");
    }
  }, [toast]);

  useEffect(() => {
    load();
  }, [load]);

  const save = useCallback(
    async (patch: Record<string, unknown>) => {
      setSaving(true);
      try {
        const payload = await api.updateSettings(patch);
        setValues(payload.settings);
        toast.ok("Saved");
        refreshStatus();
      } catch (error) {
        toast.fail(error, "That setting was rejected.");
      } finally {
        setSaving(false);
      }
    },
    [toast, refreshStatus],
  );

  const reset = async () => {
    if (!window.confirm("Restore every setting to its default? Your projects and files are not affected.")) return;
    try {
      const payload = await api.resetSettings();
      setValues(payload.settings);
      toast.ok("Defaults restored");
      await load();
    } catch (error) {
      toast.fail(error);
    }
  };

  const shown = useMemo(() => {
    const group = GROUPS.find((item) => item.key === active);
    if (!group) return [];
    const visible = (field: SettingsSchemaField | undefined): field is SettingsSchemaField =>
      Boolean(field) && field!.type !== "object" && (features.local_paths || !DESKTOP_ONLY.has(field!.name));
    if (group.key === "advanced") {
      const claimed = new Set(GROUPS.flatMap((item) => item.fields));
      return Object.values(schema).filter((field) => !claimed.has(field.name) && visible(field));
    }
    return group.fields.map((name) => schema[name]).filter(visible);
  }, [active, schema, features.local_paths]);

  const suggestions: Record<string, string[]> = {
    whisper_model: WHISPER_MODELS,
    gameplay_category: categories.gameplay ?? [],
    broll_category: categories.broll ?? [],
    music_mood: categories.music ?? [],
  };

  const group = GROUPS.find((item) => item.key === active);
  const memory = hardware?.hardware.memory;

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-lg font-semibold tracking-tight text-text">Settings</h1>
          <p className="mt-0.5 text-xs text-text-3">Changes apply to the next analysis or render.</p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={reset}>
          <RefreshCw size={14} /> Restore defaults
        </button>
      </header>

      <section className="card p-4">
        <h2 className="panel-title">This machine</h2>
        <dl className="mt-2 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
          <Row
            label="CPU"
            value={hardware ? `${hardware.hardware.cpu?.name ?? "unknown"} · ${hardware.hardware.cpu?.logical_cores ?? "?"} threads` : "detecting…"}
          />
          <Row
            label="Memory"
            value={memory ? `${memory.total_gb} GB${memory.container_limited ? " (container limit)" : ""} · ${memory.available_gb} GB free` : "detecting…"}
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
          <Row label="Speech-to-text" value={status?.ai?.faster_whisper ? `faster-whisper ${status.ai.faster_whisper_version ?? ""}`.trim() : "not installed"} />
          <Row label="Recommended model" value={hardware?.hardware.recommended?.whisper_model ?? "—"} />
          <Row label="Data folder" value={diagnostics?.paths?.data_dir ?? status?.data_dir ?? ""} mono />
        </dl>
        {status?.notes?.length ? (
          <ul className="mt-3 space-y-1 border-t border-line pt-2">
            {status.notes.map((note, index) => (
              <li key={index} className={`text-[11px] ${note.level === "error" ? "text-bad" : note.level === "warning" ? "text-warn" : "text-text-3"}`}>
                · <span className="font-medium">{note.title}</span> — {note.detail}
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
        {shown.length === 0 && active !== "captions" ? (
          <p className="text-xs text-text-3">{Object.keys(schema).length ? "Nothing here." : "Loading…"}</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {shown.map((field) => (
              <Field
                key={field.name}
                field={field}
                value={values[field.name] ?? field.value}
                suggestions={suggestions[field.name]}
                onCommit={(value) => save({ [field.name]: value })}
              />
            ))}
          </div>
        )}
        {active === "captions" && values.caption ? (
          <CaptionStyleEditor theme={values.caption} presets={presets} onSave={(caption) => save({ caption })} />
        ) : null}
        {active === "transcription" ? <OllamaCheck baseUrl={values.ollama_base_url ?? ""} model={values.ollama_model ?? ""} /> : null}
        {active === "storage" ? <StorageCleanup days={values.cleanup_days ?? 14} /> : null}
        {saving ? <p className="mt-3 text-[11px] text-text-3">Saving…</p> : null}
      </section>

      {diagnostics?.errors?.length ? (
        <section className="card p-4">
          <h2 className="panel-title">Recent errors</h2>
          <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto scroll-thin">
            {diagnostics.errors.slice(0, 12).map((error, index) => (
              <li key={index} className="text-[11px] text-text-3">
                <span className="mono mr-2">{error.at?.slice(5, 19).replace("T", " ")}</span>
                {error.file ? <span className="chip mr-2">{error.file}</span> : null}
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
      <dt className="w-32 shrink-0 text-text-3">{label}</dt>
      <dd className={`min-w-0 flex-1 truncate text-text-2 ${mono ? "mono text-[11px]" : ""}`} title={value}>
        {value}
      </dd>
    </div>
  );
}

function Help({ text }: { text?: string }) {
  return text ? <span className="mt-1 block text-[11px] leading-snug text-text-3">{text}</span> : null;
}

function Field({
  field,
  value,
  suggestions,
  onCommit,
}: {
  field: SettingsSchemaField;
  value: unknown;
  suggestions?: string[];
  onCommit: (value: unknown) => void;
}) {
  const [draft, setDraft] = useState<string>(value === null || value === undefined ? "" : String(value));
  useEffect(() => setDraft(value === null || value === undefined ? "" : String(value)), [value]);
  const disabled = Boolean(field.readonly);
  const help = disabled ? `${field.help ? `${field.help} ` : ""}Fixed by the server administrator.` : field.help;

  if (field.type === "boolean") {
    return (
      <label className="flex items-start justify-between gap-3 rounded-md border border-line bg-surface-2 px-3 py-2">
        <span className="min-w-0">
          <span className="block text-[13px] font-medium text-text">{field.label}</span>
          <Help text={help} />
        </span>
        <input
          type="checkbox"
          className="mt-0.5 h-4 w-4 shrink-0 accent-accent"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(event) => onCommit(event.target.checked)}
        />
      </label>
    );
  }

  if (field.type === "enum" || (suggestions && suggestions.length && field.name !== "whisper_model")) {
    const options =
      field.type === "enum"
        ? field.options.map((option) => (typeof option === "string" ? option : String((option as any).name ?? "")))
        : suggestions ?? [];
    return (
      <label className="block">
        <span className="label">{field.label}</span>
        <select className="select" value={String(value ?? "")} disabled={disabled} onChange={(event) => onCommit(event.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option ? option.replace(/_/g, " ") : "(auto)"}
            </option>
          ))}
        </select>
        <Help text={help} />
      </label>
    );
  }

  if (field.type === "number" || field.type === "integer") {
    const commit = () => {
      if (draft.trim() === "") {
        setDraft(String(value ?? ""));
        return;
      }
      const parsed = field.type === "integer" ? Number.parseInt(draft, 10) : Number.parseFloat(draft);
      if (Number.isNaN(parsed)) {
        setDraft(String(value ?? ""));
        return;
      }
      if (parsed !== Number(value)) onCommit(parsed);
    };
    return (
      <label className="block">
        <span className="label">{field.label}</span>
        <input
          className="input"
          type="number"
          step={field.type === "integer" ? 1 : "any"}
          min={field.ge ?? field.gt}
          max={field.le ?? field.lt}
          value={draft}
          disabled={disabled}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => event.key === "Enter" && (event.target as HTMLInputElement).blur()}
        />
        <Help text={help} />
      </label>
    );
  }

  const listId = suggestions?.length ? `suggest-${field.name}` : undefined;
  return (
    <label className="block">
      <span className="label">{field.label}</span>
      <input
        className="input"
        value={draft}
        list={listId}
        disabled={disabled}
        placeholder={field.default ? String(field.default) : ""}
        spellCheck={false}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => draft !== String(value ?? "") && onCommit(draft)}
        onKeyDown={(event) => event.key === "Enter" && (event.target as HTMLInputElement).blur()}
      />
      {listId ? (
        <datalist id={listId}>
          {suggestions!.map((option) => (
            <option key={option} value={option} />
          ))}
        </datalist>
      ) : null}
      <Help text={help} />
    </label>
  );
}

// ------------------------------------------------------------------ captions

const POSITIONS = ["top", "middle", "lower-middle", "bottom"];
const ANIMATIONS = ["none", "pop", "karaoke", "word_by_word", "slide_up"];

function CaptionStyleEditor({
  theme,
  presets,
  onSave,
}: {
  theme: Record<string, any>;
  presets: Record<string, CaptionPresetInfo>;
  onSave: (caption: Record<string, unknown>) => void;
}) {
  const [draft, setDraft] = useState<Record<string, any>>(theme);
  const [sample, setSample] = useState<CaptionPlan | null>(null);
  useEffect(() => setDraft(theme), [theme]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      api
        .captionStylePreview("", draft)
        .then(setSample)
        .catch(() => setSample(null));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [draft]);

  const changed = JSON.stringify(draft) !== JSON.stringify(theme);
  const set = (patch: Record<string, unknown>) => setDraft((current) => ({ ...current, ...patch }));
  const applyPreset = (preset: string) => setDraft((current) => ({ ...current, ...(presets[preset]?.theme ?? {}), preset }));
  const line = sample?.lines?.[0];

  return (
    <div className="mt-4 grid gap-4 border-t border-line pt-4 lg:grid-cols-[1fr_14rem]">
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block sm:col-span-2">
          <span className="label">Caption style</span>
          <CaptionStyleSelect value={draft.preset ?? ""} presets={presets} onChange={applyPreset} />
          <Help text={presets[draft.preset]?.description} />
        </label>
        <label className="block">
          <span className="label">Font</span>
          <input className="input" value={draft.font ?? ""} onChange={(event) => set({ font: event.target.value })} spellCheck={false} />
        </label>
        <label className="block">
          <span className="label">Font size · {draft.font_size}</span>
          <input type="range" min={28} max={110} value={draft.font_size ?? 68} onChange={(event) => set({ font_size: Number(event.target.value) })} className="w-full accent-accent" />
        </label>
        <label className="block">
          <span className="label">Position</span>
          <select className="select" value={draft.position} onChange={(event) => set({ position: event.target.value })}>
            {POSITIONS.map((option) => (
              <option key={option} value={option}>
                {option.replace("-", " ")}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="label">Animation</span>
          <select className="select" value={draft.animation} onChange={(event) => set({ animation: event.target.value })}>
            {ANIMATIONS.map((option) => (
              <option key={option} value={option}>
                {option.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="label">Text colour</span>
          <input type="color" className="h-9 w-full cursor-pointer rounded-md border border-line bg-surface-2" value={draft.primary_color ?? "#FFFFFF"} onChange={(event) => set({ primary_color: event.target.value.toUpperCase() })} />
        </label>
        <label className="block">
          <span className="label">Highlight colour</span>
          <input type="color" className="h-9 w-full cursor-pointer rounded-md border border-line bg-surface-2" value={draft.highlight_color ?? "#FFD400"} onChange={(event) => set({ highlight_color: event.target.value.toUpperCase() })} />
        </label>
        <label className="block">
          <span className="label">Words per line · {draft.max_words_per_line}</span>
          <input type="range" min={1} max={8} value={draft.max_words_per_line ?? 3} onChange={(event) => set({ max_words_per_line: Number(event.target.value) })} className="w-full accent-accent" />
        </label>
        <label className="flex items-center gap-2 self-end text-[13px] text-text-2">
          <input type="checkbox" className="accent-accent" checked={Boolean(draft.uppercase)} onChange={(event) => set({ uppercase: event.target.checked })} />
          Uppercase
        </label>
        <div className="flex items-center gap-2 sm:col-span-2">
          <button type="button" className="btn btn-primary" disabled={!changed} onClick={() => onSave(draft)}>
            Save caption style
          </button>
          {changed ? (
            <button type="button" className="btn btn-ghost" onClick={() => setDraft(theme)}>
              Discard
            </button>
          ) : null}
        </div>
      </div>

      <div className="relative aspect-[9/16] overflow-hidden rounded-md border border-line bg-gradient-to-b from-surface-2 to-black">
        <div
          className={`absolute inset-x-3 text-center font-black leading-tight ${
            draft.position === "top" ? "top-[12%]" : draft.position === "middle" ? "top-1/2 -translate-y-1/2" : draft.position === "bottom" ? "bottom-[8%]" : "bottom-[24%]"
          }`}
          style={{
            fontFamily: `"${draft.font}", ${(draft.fallback_fonts ?? []).map((font: string) => `"${font}"`).join(", ")}, sans-serif`,
            fontSize: `${Math.max(10, Math.round((draft.font_size ?? 68) / 4.2))}px`,
            color: draft.primary_color,
            textTransform: draft.uppercase ? "uppercase" : "none",
            textShadow: `0 0 ${Math.max(1, (draft.outline_width ?? 4) / 2)}px ${draft.outline_color ?? "#000"}, 0 1px 2px #000`,
          }}
        >
          {(line?.words ?? []).map((word, index) => (
            <span key={index} style={{ color: index === 1 ? draft.highlight_color : undefined }}>
              {word.text}{" "}
            </span>
          ))}
          {!line ? <span>Your captions</span> : null}
        </div>
        <span className="absolute bottom-1 right-2 text-[9px] text-text-3">preview</span>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------- ollama

function OllamaCheck({ baseUrl, model }: { baseUrl: string; model: string }) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ available: boolean; error: string; version: string; models: string[] } | null>(null);

  const test = async () => {
    setBusy(true);
    try {
      const payload = await api.testOllama(baseUrl, model);
      setResult({ ...payload.status, models: payload.models ?? [] });
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const hasModel = result?.models?.some((name) => name === model || name.startsWith(`${model}:`));
  return (
    <div className="mt-4 flex flex-wrap items-start gap-3 border-t border-line pt-3">
      <button type="button" className="btn btn-secondary btn-sm" onClick={test} disabled={busy}>
        {busy ? <Loader2 size={13} className="animate-spin" /> : <PlugZap size={13} />} Test Ollama connection
      </button>
      {result ? (
        <div className="min-w-0 flex-1 text-[11px]">
          {result.available ? (
            <p className="text-good">
              Connected{result.version ? ` · Ollama ${result.version}` : ""}.{" "}
              {hasModel ? `“${model}” is installed.` : <span className="text-warn">“{model}” is not pulled yet (ollama pull {model}).</span>}
            </p>
          ) : (
            <p className="text-bad">Not reachable: {result.error || "no response"}. Analysis still works without it.</p>
          )}
          {result.models?.length ? <p className="mt-1 text-text-3">Installed models: {result.models.join(", ")}</p> : null}
        </div>
      ) : (
        <p className="text-[11px] text-text-3">Checks {baseUrl || "the configured URL"} without saving anything.</p>
      )}
    </div>
  );
}

// ------------------------------------------------------------------- storage

function StorageCleanup({ days }: { days: number }) {
  const toast = useToast();
  const [usage, setUsage] = useState<Record<string, number> | null>(null);
  const [renders, setRenders] = useState(false);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<CleanupReport | null>(null);

  const loadUsage = useCallback(() => {
    api
      .storage()
      .then((payload) => setUsage(payload.usage))
      .catch(() => setUsage(null));
  }, []);

  useEffect(() => {
    loadUsage();
  }, [loadUsage]);

  const run = async () => {
    setBusy(true);
    try {
      const preview = await api.cleanup({ renders, dry_run: true });
      const removable = preview.projects_removed.length + preview.renders_removed + preview.source_files_removed + preview.cache.removed_files;
      if (!removable) {
        toast.ok("Nothing to clean up", `Nothing is older than ${days} days.`);
        setReport(preview);
        return;
      }
      const summary = [
        preview.cache.removed_files ? `${preview.cache.removed_files} cached downloads` : "",
        preview.projects_removed.length ? `${preview.projects_removed.length} unfinished projects` : "",
        preview.renders_removed ? `${preview.renders_removed} rendered clips` : "",
        preview.source_files_removed ? `${preview.source_files_removed} source files` : "",
      ]
        .filter(Boolean)
        .join(", ");
      if (!window.confirm(`Remove ${summary}? This frees about ${bytes(preview.freed_bytes)}.`)) return;
      const done = await api.cleanup({ renders });
      setReport(done);
      setUsage(done.usage);
      toast.ok("Cleanup finished", `${bytes(done.freed_bytes)} freed`);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-4 space-y-3 border-t border-line pt-3">
      <h3 className="panel-title flex items-center gap-1.5">
        <HardDrive size={13} /> Disk usage
      </h3>
      {usage ? (
        <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
          {Object.entries(usage).map(([area, size]) => (
            <div key={area} className="flex gap-2">
              <dt className="w-20 capitalize text-text-3">{area}</dt>
              <dd className="mono text-text-2">{bytes(size)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-xs text-text-3">Measuring…</p>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" className="btn btn-secondary btn-sm" onClick={run} disabled={busy}>
          {busy ? <Loader2 size={13} className="animate-spin" /> : null} Clean up now
        </button>
        <label className="flex items-center gap-2 text-[12px] text-text-2">
          <input type="checkbox" className="accent-accent" checked={renders} onChange={(event) => setRenders(event.target.checked)} />
          Also delete rendered clips older than {days} days (they can be re-rendered)
        </label>
      </div>
      <p className="text-[11px] text-text-3">
        Removes cached downloads and draft, failed or cancelled projects older than {days} days, and trims the download cache to its
        budget. Projects with running jobs are never touched.
      </p>
      {report && !report.dry_run ? (
        <p className="text-[11px] text-good">
          Freed {bytes(report.freed_bytes)} · {report.projects_removed.length} projects · {report.renders_removed} renders ·{" "}
          {report.cache.removed_files} cache files
        </p>
      ) : null}
    </div>
  );
}
