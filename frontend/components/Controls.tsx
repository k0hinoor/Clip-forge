"use client";

// Small form controls shared by the Studio, the clip editor and Settings.

import type { CaptionPresetInfo } from "@/lib/types";

/** Built-in caption presets (labels mirror backend/clipforge/constants.py). */
export const CAPTION_STYLES: [string, string][] = [
  ["bold_creator", "Bold Creator"],
  ["minimal", "Minimal"],
  ["karaoke", "Karaoke"],
  ["cinematic", "Cinematic"],
  ["highlight", "Highlight"],
  ["documentary", "Documentary"],
];

export const LAYOUT_OPTIONS: [string, string][] = [
  ["podcast", "Full frame"],
  ["blur", "Blurred background"],
  ["cinematic", "Cinematic crop"],
  ["split", "Split screen (gameplay)"],
  ["gameplay", "Gameplay background"],
  ["broll", "B-roll split"],
];

export function CaptionStyleSelect({
  value,
  onChange,
  disabled,
  presets,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  /** Presets reported by the API; the built-in list is used until they load. */
  presets?: Record<string, CaptionPresetInfo>;
  /** Optional first entry with an empty value (e.g. "Keep current style"). */
  placeholder?: string;
}) {
  const entries: [string, string][] = presets && Object.keys(presets).length
    ? Object.entries(presets).map(([key, info]) => [key, info.label || key])
    : CAPTION_STYLES;
  return (
    <select className="select" value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
      {placeholder !== undefined ? <option value="">{placeholder}</option> : null}
      {entries.map(([key, label]) => (
        <option key={key} value={key}>
          {label}
        </option>
      ))}
    </select>
  );
}

export function Toggle({
  label,
  hint,
  value,
  onChange,
  disabled,
}: {
  label: string;
  hint?: string;
  value: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      disabled={disabled}
      className="flex items-center justify-between gap-3 rounded-md border border-line bg-surface-2 px-3 py-2 text-left disabled:opacity-50"
      aria-pressed={value}
    >
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-text">{label}</span>
        {hint ? <span className="block truncate text-[11px] text-text-3">{hint}</span> : null}
      </span>
      <span className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${value ? "bg-accent" : "bg-line-strong"}`}>
        <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${value ? "left-3.5" : "left-0.5"}`} />
      </span>
    </button>
  );
}
