"use client";

// Friendly, stack-trace-free error surface. Every backend error already carries
// a human message and a hint; this just shows them and never the raw exception.

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { ApiFailure } from "@/lib/api";

type ToastKind = "error" | "success" | "info";
type Toast = { id: number; kind: ToastKind; title: string; hint?: string; detail?: string };

type ToastApi = {
  push: (toast: Omit<Toast, "id">) => void;
  fail: (error: unknown, fallback?: string) => void;
  ok: (title: string, hint?: string) => void;
};

const ToastContext = createContext<ToastApi | null>(null);

const ICONS: Record<ToastKind, ReactNode> = {
  error: <AlertTriangle size={18} />,
  success: <CheckCircle2 size={18} />,
  info: <Info size={18} />,
};

const TONES: Record<ToastKind, string> = {
  error: "border-flare-500/50 bg-flare-500/10 text-flare-400",
  success: "border-signal-500/50 bg-signal-500/10 text-signal-400",
  info: "border-ink-600 bg-ink-800 text-mist-200",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (toast: Omit<Toast, "id">) => {
      const id = Date.now() + Math.random();
      setToasts((current) => [...current.slice(-3), { ...toast, id }]);
      const lifetime = toast.kind === "error" ? 12000 : 5000;
      window.setTimeout(() => dismiss(id), lifetime);
    },
    [dismiss],
  );

  const fail = useCallback(
    (error: unknown, fallback = "Something went wrong.") => {
      if (error instanceof ApiFailure) {
        push({ kind: "error", title: error.message, hint: error.hint });
        return;
      }
      const message = error instanceof Error ? error.message : fallback;
      push({ kind: "error", title: message, hint: "Check Settings → Diagnostics for more detail." });
    },
    [push],
  );

  const ok = useCallback((title: string, hint?: string) => push({ kind: "success", title, hint }), [push]);

  const value = useMemo(() => ({ push, fail, ok }), [push, fail, ok]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-5 right-5 z-50 flex w-[min(94vw,420px)] flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`slide-in card-tight flex items-start gap-3 border p-3 shadow-2xl backdrop-blur ${TONES[toast.kind]}`}
          >
            <div className="mt-0.5 shrink-0">{ICONS[toast.kind]}</div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-mist-200">{toast.title}</p>
              {toast.hint ? <p className="mt-1 text-xs leading-relaxed text-mist-400">{toast.hint}</p> : null}
            </div>
            <button type="button" className="btn-quiet -m-1 p-1" onClick={() => dismiss(toast.id)} aria-label="Dismiss">
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used inside <ToastProvider>");
  }
  return context;
}
