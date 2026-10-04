"use client";

import { Loader2 } from "lucide-react";

export function ProgressBar({ value, className = "" }: { value: number; className?: string }) {
  const percent = Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)));
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-ink-700/70 ${className}`}>
      <div
        className="h-full rounded-full bg-gradient-to-r from-flare-500 to-amber-glow transition-[width] duration-500"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

export function StageList({
  stages,
}: {
  stages: { key: string; label: string; status: string; progress: number; message: string }[];
}) {
  return (
    <ol className="space-y-1.5">
      {stages.map((stage) => {
        const tone =
          stage.status === "done"
            ? "text-signal-400"
            : stage.status === "active"
              ? "text-amber-glow"
              : stage.status === "failed"
                ? "text-flare-400"
                : "text-mist-400";
        return (
          <li key={stage.key} className="flex items-start gap-2 text-xs">
            <span className={`mt-0.5 w-4 shrink-0 ${tone}`}>
              {stage.status === "done" ? "●" : stage.status === "active" ? <Loader2 size={12} className="animate-spin" /> : "○"}
            </span>
            <span className="min-w-0 flex-1">
              <span className={`font-semibold ${tone}`}>{stage.label}</span>
              {stage.message ? <span className="ml-2 text-mist-400">{stage.message}</span> : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-mist-400">
      <Loader2 size={13} className="animate-spin" />
      {label}
    </span>
  );
}
