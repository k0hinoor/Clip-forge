"use client";

import { Check, Loader2, X } from "lucide-react";

export function ProgressBar({ value, className = "" }: { value: number; className?: string }) {
  const percent = Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)));
  return (
    <div className={`h-1 w-full overflow-hidden rounded-full bg-line ${className}`}>
      <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${percent}%` }} />
    </div>
  );
}

export type Stage = {
  key: string;
  label: string;
  detail?: string;
  state: "pending" | "running" | "done" | "failed";
  progress?: number;
};

/** The analysis pipeline, one row per stage, using the state the API reports. */
export function StageList({ stages }: { stages: Stage[] }) {
  return (
    <ol className="space-y-1">
      {stages.map((stage) => {
        const tone =
          stage.state === "done"
            ? "text-good"
            : stage.state === "running"
              ? "text-accent"
              : stage.state === "failed"
                ? "text-bad"
                : "text-text-3";
        return (
          <li key={stage.key} className="flex items-center gap-2 text-xs">
            <span className={`grid h-3.5 w-3.5 shrink-0 place-items-center ${tone}`}>
              {stage.state === "done" ? (
                <Check size={12} />
              ) : stage.state === "failed" ? (
                <X size={12} />
              ) : stage.state === "running" ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <span className="h-1 w-1 rounded-full bg-current" />
              )}
            </span>
            <span className={stage.state === "pending" ? "text-text-3" : "text-text-2"}>{stage.label}</span>
            {stage.state === "running" && stage.detail ? (
              <span className="truncate text-text-3">{stage.detail}</span>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-text-3">
      <Loader2 size={13} className="animate-spin" />
      {label}
    </span>
  );
}
