"use client";

import { useCallback, useEffect, useState } from "react";
import { Ban, ListVideo, Play, RotateCcw, Trash2 } from "lucide-react";
import { api, subscribeEvents } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { when } from "@/lib/format";
import type { Job, QueueState } from "@/lib/types";

export default function QueuePage() {
  const toast = useToast();
  const [queue, setQueue] = useState<QueueState | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);

  const load = useCallback(
    async (silent = false) => {
      try {
        const [queuePayload, jobPayload] = await Promise.all([api.queue(), api.jobs(150)]);
        setQueue(queuePayload.queue);
        setJobs(jobPayload.jobs);
      } catch (error) {
        if (!silent) toast.fail(error, "Could not read the render queue.");
      }
    },
    [toast],
  );

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 3000);
    const unsubscribe = subscribeEvents(() => load(true));
    return () => {
      window.clearInterval(timer);
      unsubscribe();
    };
  }, [load]);

  const act = async (kind: "cancel" | "retry" | "retryFailed" | "purge", jobId = "") => {
    try {
      if (kind === "cancel") await api.cancelJob(jobId);
      if (kind === "retry") await api.retryJob(jobId);
      if (kind === "retryFailed") await api.retryFailed();
      if (kind === "purge") await api.purgeJobs();
      toast.ok("Done");
      await load(true);
    } catch (error) {
      toast.fail(error);
    }
  };

  const counts = queue?.counts;
  const finished = jobs.filter((job) => ["succeeded", "failed", "cancelled"].includes(job.status));
  const rows = [...(queue?.running ?? []), ...(queue?.queued ?? []), ...finished];

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-lg font-semibold tracking-tight text-text">Queue</h1>
          <p className="mt-0.5 text-xs text-text-3">
            {counts ? `${counts.running} running · ${counts.queued} waiting · ${counts.failed} failed` : "loading…"}
          </p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => act("retryFailed")} disabled={!counts?.failed}>
          <RotateCcw size={14} /> Retry failed
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => act("purge")}>
          <Trash2 size={14} /> Clear finished
        </button>
      </header>

      {rows.length === 0 ? (
        <div className="card flex flex-col items-center gap-1 px-6 py-12 text-center">
          <ListVideo size={20} className="text-text-3" />
          <p className="mt-1 text-sm font-medium text-text-2">The queue is empty</p>
          <p className="max-w-sm text-xs text-text-3">
            Queue clips from a project and they render here one after another, in the background.
          </p>
        </div>
      ) : (
        <ul className="card divide-y divide-line overflow-hidden">
          {rows.map((job) => (
            <li key={job.id} className="px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="chip">{job.kind.replace(/_/g, " ")}</span>
                <span className="min-w-0 flex-1 truncate text-[13px] text-text">{job.label || job.message || job.id}</span>
                <span className="mono text-[11px] text-text-3">{when(job.created_at)}</span>
                {["queued", "running"].includes(job.status) ? (
                  <button type="button" className="btn btn-ghost btn-icon" onClick={() => act("cancel", job.id)} aria-label="Cancel job">
                    <Ban size={14} />
                  </button>
                ) : (
                  <button type="button" className="btn btn-ghost btn-icon" onClick={() => act("retry", job.id)} aria-label="Retry job">
                    <Play size={14} />
                  </button>
                )}
              </div>

              {job.status === "running" ? (
                <div className="mt-2 flex items-center gap-2">
                  <ProgressBar value={job.progress} className="max-w-xs" />
                  <span className="truncate text-[11px] text-text-3">
                    {Math.round(job.progress * 100)}% · {job.stage || "working"}
                  </span>
                </div>
              ) : null}

              {job.error_message ? (
                <p className="mt-1.5 text-[11px] text-bad">
                  {job.error_message}
                  {job.error_hint ? <span className="text-text-3"> — {job.error_hint}</span> : null}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
