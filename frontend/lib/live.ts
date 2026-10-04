"use client";

import { useEffect, useRef } from "react";
import { subscribeEvents, type StreamEvent } from "./api";

type LiveOptions = {
  /** Only events for this project (plus global ones). */
  projectId?: string;
  /** Ignore events this returns false for. */
  match?: (event: StreamEvent) => boolean;
  /** Minimum gap between two refreshes caused by events. */
  throttleMs?: number;
  /** Fallback polling interval, in case the event stream is interrupted. */
  intervalMs?: number;
};

/**
 * Re-run ``refresh`` when the backend reports relevant changes.
 *
 * Progress events arrive several times a second while a job runs; refreshes
 * are throttled so a page never floods the API, and a slow poll covers the
 * moments the stream is reconnecting.
 */
export function useLiveRefresh(refresh: (event?: StreamEvent) => void, options: LiveOptions = {}): void {
  const { projectId = "", throttleMs = 1000, intervalMs = 15000 } = options;
  const refreshRef = useRef(refresh);
  const matchRef = useRef(options.match);
  refreshRef.current = refresh;
  matchRef.current = options.match;

  useEffect(() => {
    let timer: number | undefined;
    let pending: StreamEvent | undefined;
    let last = 0;

    const fire = () => {
      timer = undefined;
      last = Date.now();
      refreshRef.current(pending);
      pending = undefined;
    };

    const unsubscribe = subscribeEvents((event) => {
      if (matchRef.current && !matchRef.current(event)) return;
      pending = event;
      if (timer !== undefined) return;
      timer = window.setTimeout(fire, Math.max(0, throttleMs - (Date.now() - last)));
    }, projectId);
    const poll = window.setInterval(() => refreshRef.current(), intervalMs);

    return () => {
      unsubscribe();
      window.clearInterval(poll);
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [projectId, throttleMs, intervalMs]);
}

/** Events that change what a project page shows. */
export function isProjectEvent(event: StreamEvent): boolean {
  return /^(job|clip|project|analysis)\./.test(event.type);
}
