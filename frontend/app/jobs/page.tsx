"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Ban, ListVideo, Play, RotateCcw, Trash2, Users } from "lucide-react";
import { api, subscribeEvents } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { ProgressBar } from "@/components/Progress";
import { when } from "@/lib/format";
import type { Job, QueueState } from "@/lib/types";

export default function JobsPage() {
  const toast = useToast();
  const [queue, setQueue] = useState<QueueState | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [workers, setWorkers] = useState<any>(null);

  const load = useCallback(
    async (silent = false) => {
      try {
        const [queuePayload, jobPayload, workerPayload] = await Promise.all([api.queue(), api.jobs(150), api.workers()]);
        setQueue(queuePayload.queue);
        setWorkers(workerPayload);
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

  const act = async (kind: "cancel" | "retry" | "retryFailed" | "purge" | "start" | "stop", jobId = "") => {
    try {
      if (kind === "cancel") await api.cancelJob(jobId);
      if (kind === "retry") await api.retryJob(jobId);
      if (kind === "retryFailed") await api.retryFailed();
      if (kind === "purge") await api.purgeJobs();
      if (kind === "start") await api.startWorkers();
      if (kind === "stop") await api.stopWorkers();
      toast.ok("Done");
      await load(true);
    } catch (error) {
      toast.fail(error);
    }
  };

  const counts = queue?.counts;

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h1 className="text-xl font-black text-mist-200">Render queue</h1>
          <p className="mt-0.5 text-xs text-mist-400">
            {counts ? `${counts.running} running · ${counts.queued} waiting · ${counts.failed} failed` : "loading…"}
          </p>
        </div>
        <span className="chip">
          <Users size={13} /> {workers ? `${workers.active ?? 0}/${workers.workers ?? 0} workers` : "…"}
        </span>
        <button type="button" className="btn btn-ghost" onClick={() => act("retryFailed")}>
          <RotateCcw size={14} /> Retry failed
        </button>
        <button type="button" className="btn btn-quiet" onClick={() => act("purge")}>
          <Trash2 size={14} /> Clear finished
        </button>
      </header>

      <section className="grid gap-3 sm:grid-cols-3">
        <Stat label="Running" value={counts?.running ?? 0} tone="text-amber-glow" />
        <Stat label="Queued" value={counts?.queued ?? 0} />
        <Stat label="Failed" value={counts?.failed ?? 0} tone={counts?.failed ? "text-flare-400" : undefined} />
      </section>

      {queue && queue.running.length + queue.queued.length === 0 && jobs.length === 0 ? (
        <div className="card flex flex-col items-center gap-2 p-10 text-center">
          <ListVideo size={24} className="text-mist-400" />
          <p className="text-sm font-semibold text-mist-300">The queue is empty</p>
          <p className="text-xs text-mist-400">Queue clips from a project and they will render here, one after another.</p>
        </div>
      ) : null}

      <ul className="space-y-2">
        {[...(queue?.running ?? []), ...(queue?.queued ?? []), ...jobs.filter((job) => ["finished", "failed", "cancelled"].includes(job.status))].map((job) => (
          <li key={job.id} className="card-tight p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="chip">{job.kind.replace(/_/g, " ")}</span>
              <span className="min-w-0 flex-1 truncate text-sm text-mist-200">{job.label || job.message || job.id}</span>
              <span className="mono text-[11px] text-mist-400">{when(job.created_at)}</span>
              {["queued", "running"].includes(job.status) ? (
                <button type="button" className="btn btn-quiet p-1.5" onClick={() => act("cancel", job.id)} aria-label="Cancel job">
                  <Ban size={14} />
                </button>
              ) : (
                <button type="button" className="btn btn-quiet p-1.5" onClick={() => act("retry", job.id)} aria-label="Retry job">
                  <Play size={14} />
                </button>
              )}
            </div>
            {job.status === "running" ? (
              <div className="mt-2 space-y-1">
                <ProgressBar value={job.progress} />
                <p className="text-[11px] text-mist-400">
                  {Math.round(job.progress * 100)}% · {job.stage || "working"} {job.message ? `· ${job.message}` : ""}
                </p>
              </div>
            ) : null}
            {job.error_message ? (
              <p className="mt-1.5 text-[11px] text-flare-400">
                {job.error_message} {job.error_hint ? <span className="text-mist-400">— {job.error_hint}</span> : null}
              </p>
            ) : null}
            {job.log?.length ? (
              <details className="mt-2">
                <summary className="cursor-pointer text-[11px] text-mist-400">Show {job.log.length} log lines</summary>
                <ul className="mono mt-1 max-h-40 space-y-0.5 overflow-y-auto text-[10px] text-mist-400 scroll-thin">
                  {job.log.map((line, index) => (
                    <li key={index}>
                      {line.stage ? `[${line.stage}] ` : ""}
                      {line.message}
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </li>
        ))}
      </ul>

      <p className="text-[11px] text-mist-400">
        Renders run in the background: keep this tab open or not, the work continues in the local worker process.{" "}
        <Link href="/" className="text-flare-400">
          Back to the studio
        </Link>
      </p>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone?: string }) {
  return (
    <div className="card p-4">
      <p className="text-[11px] uppercase tracking-widest text-mist-400">{label}</p>
      <p className={`mt-1 text-2xl font-black ${tone ?? "text-mist-200"}`}>{value}</p>
    </div>
  );
}
