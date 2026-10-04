export function clock(seconds: number | undefined | null, withHours = false): string {
  const total = Math.max(0, Math.floor(seconds ?? 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const base = `${String(minutes).padStart(withHours || hours ? 2 : 1, "0")}:${String(secs).padStart(2, "0")}`;
  return hours ? `${hours}:${base}` : base;
}

export function duration(seconds: number | undefined | null): string {
  const value = seconds ?? 0;
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

export function bytes(size: number | undefined | null): string {
  const value = size ?? 0;
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let index = -1;
  let scaled = value;
  do {
    scaled /= 1024;
    index += 1;
  } while (scaled >= 1024 && index < units.length - 1);
  return `${scaled.toFixed(scaled >= 10 ? 0 : 1)} ${units[index]}`;
}

export function when(value: string | undefined | null): string {
  if (!value) return "—";
  const date = new Date(value.endsWith("Z") || value.includes("+") ? value : `${value}Z`);
  if (Number.isNaN(date.getTime())) return String(value);
  const diff = (Date.now() - date.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return date.toLocaleDateString();
}

export function scoreTone(score: number): string {
  if (score >= 82) return "text-good";
  if (score >= 70) return "text-warn";
  return "text-text-2";
}

export function statusChip(status: string): { label: string; className: string } {
  switch (status) {
    case "rendered":
      return { label: "Rendered", className: "chip chip-good" };
    case "rendering":
      return { label: "Rendering", className: "chip chip-warn" };
    case "running":
      return { label: "Analysing", className: "chip chip-warn" };
    case "queued":
      return { label: "Queued", className: "chip" };
    case "failed":
      return { label: "Failed", className: "chip chip-bad" };
    case "ready":
      return { label: "Ready", className: "chip chip-good" };
    case "succeeded":
      return { label: "Done", className: "chip chip-good" };
    case "cancelled":
      return { label: "Cancelled", className: "chip" };
    case "pending":
      return { label: "Not rendered", className: "chip" };
    case "draft":
      return { label: "Draft", className: "chip" };
    default:
      return { label: status ? status.charAt(0).toUpperCase() + status.slice(1) : "Draft", className: "chip" };
  }
}

/** A project that is waiting for or running its analysis. */
export function isAnalysing(status: string | undefined): boolean {
  return status === "queued" || status === "running";
}

/** Human label for a job row. */
export function jobLabel(job: { kind: string; clip_index?: number; clip_title?: string; project_title?: string }): string {
  const kind =
    job.kind === "analyze"
      ? "Analysis"
      : job.kind === "render_clip"
        ? "Render"
        : job.kind === "render_preview"
          ? "Preview"
          : job.kind === "scan_assets"
            ? "Library scan"
            : job.kind.replace(/_/g, " ");
  const clip = job.clip_index ? `clip ${String(job.clip_index).padStart(2, "0")}${job.clip_title ? ` · ${job.clip_title}` : ""}` : "";
  return [kind, clip || job.project_title].filter(Boolean).join(" — ");
}

/** Cache-busting media URL: a re-render must never show the previous file. */
export function mediaUrl(url: string | null | undefined, version?: string | number | null): string {
  if (!url) return "";
  return version ? `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(String(version))}` : url;
}

export function progressLabel(stage: string, progress: number): string {
  const percent = Math.round((progress ?? 0) * 100);
  return `${percent}% · ${stage || "working"}`;
}
