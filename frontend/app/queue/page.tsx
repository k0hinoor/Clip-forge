"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Ban, ListVideo, RotateCcw, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { useLiveRefresh } from "@/lib/live";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { jobLabel, statusChip, when } from "@/lib/format";
import type { Job, QueueState } from "@/lib/types";

export default function QueuePage() {
  const toast = useToast();
  const [queue, setQueue] = useState<QueueState | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(
    async (silent = false) => {
      try {
        const payload = await api.queue();
        setQueue(payload.queue);
      } catch (error) {
        if (!silent) toast.fail(error, "Could not read the queue.");
      }
    },
    [toast],
  );

  useEffect(() => {
    load();
  }, [load]);

  useLiveRefresh(() => load(true), { match: (event) => event.type.startsWith("job."), throttleMs: 1000, intervalMs: 10000 });

  const act = async <T,>(action: () => Promise<T>, done: string | ((result: T) => string)) => {
    setBusy(true);
    try {
      const result = await action();
      toast.ok(typeof done === "function" ? done(result) : done);
      await load(true);
    } catch (error) {
      toast.fail(error);
    } finally {
      setBusy(false);
    }
  };

  const counts = queue?.counts;
  const sections: { title: string; jobs: Job[] }[] = [
    { title: "Running", jobs: queue?.running ?? [] },
    { title: "Waiting", jobs: queue?.queued ?? [] },
    { title: "Failed", jobs: queue?.failed ?? [] },
    { title: "Cancelled", jobs: queue?.cancelled ?? [] },
    { title: "Finished", jobs: queue?.recent ?? [] },
  ].filter((section) => section.jobs.length);
  const finished = (queue?.recent.length ?? 0) + (queue?.cancelled.length ?? 0);

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-lg font-semibold tracking-tight text-text">Queue</h1>
          <p className="mt-0.5 text-xs text-text-3">
            {counts ? `${counts.running} running · ${counts.queued} waiting · ${counts.failed} failed` : "loading…"}
          </p>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => act(() => api.retryFailed(), (result) => `${result.retried} job${result.retried === 1 ? "" : "s"} re-queued`)}
          disabled={busy || !counts?.failed}
        >
          <RotateCcw size={14} /> Retry failed
        </button>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={() => act(() => api.purgeJobs(), "Finished jobs cleared")}
          disabled={busy || finished === 0}
          title="Remove finished and cancelled jobs from the list (failed ones stay)"
        >
          <Trash2 size={14} /> Clear finished
        </button>
      </header>

      {sections.length === 0 ? (
        <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
          <ListVideo size={20} className="text-text-3" />
          <p className="mt-1 text-sm font-medium text-text-2">{queue ? "The queue is empty" : "Loading…"}</p>
          <p className="max-w-sm text-xs text-text-3">Analyses and renders run here in the background, one after another.</p>
        </div>
      ) : (
        sections.map((section) => (
          <section key={section.title} className="space-y-1.5">
            <h2 className="panel-title">
              {section.title} · {section.jobs.length}
            </h2>
            <ul className="card divide-y divide-line overflow-hidden">
              {section.jobs.map((job) => (
                <JobRow
                  key={job.id}
                  job={job}
                  disabled={busy}
                  onCancel={() => act(() => api.cancelJob(job.id), "Cancelling")}
                  onRetry={() => act(() => api.retryJob(job.id), "Re-queued")}
                />
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  );
}

function JobRow({ job, disabled, onCancel, onRetry }: { job: Job; disabled: boolean; onCancel: () => void; onRetry: () => void }) {
  const chip = statusChip(job.status);
  const href = job.clip_id ? `/projects/${job.project_id}/clips/${job.clip_id}` : job.project_id ? `/projects/${job.project_id}` : "";
  const label = jobLabel(job);
  return (
    <li className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className={chip.className}>{chip.label}</span>
        {href ? (
          <Link href={href} className="min-w-0 flex-1 truncate text-[13px] text-text hover:text-accent">
            {label}
          </Link>
        ) : (
          <span className="min-w-0 flex-1 truncate text-[13px] text-text">{label}</span>
        )}
        {job.project_title && job.clip_id ? <span className="hidden truncate text-[11px] text-text-3 sm:inline">{job.project_title}</span> : null}
        <span className="mono text-[11px] text-text-3">{when(job.finished_at || job.started_at || job.created_at)}</span>
        {job.status === "queued" || job.status === "running" ? (
          <button type="button" className="btn btn-ghost btn-icon" onClick={onCancel} disabled={disabled || job.cancel_requested} aria-label="Cancel job">
            <Ban size={14} />
          </button>
        ) : job.status === "failed" || job.status === "cancelled" ? (
          <button type="button" className="btn btn-ghost btn-icon" onClick={onRetry} disabled={disabled} aria-label="Retry job">
            <RotateCcw size={14} />
          </button>
        ) : null}
      </div>

      {job.status === "running" ? (
        <div className="mt-2 flex items-center gap-2">
          <ProgressBar value={job.progress} className="max-w-xs" />
          <span className="truncate text-[11px] text-text-3">
            {Math.round(job.progress * 100)}% · {job.cancel_requested ? "cancelling…" : job.message || job.stage || "working"}
          </span>
        </div>
      ) : null}

      {job.status === "failed" && job.error?.message ? (
        <p className="mt-1.5 text-[11px] text-bad">
          {job.error.message}
          {job.error.hint ? <span className="text-text-3"> — {job.error.hint}</span> : null}
        </p>
      ) : null}
    </li>
  );
}
