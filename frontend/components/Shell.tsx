"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Cpu, Film, FolderCog, LayoutDashboard, LibraryBig, ListVideo, Settings } from "lucide-react";
import { api } from "@/lib/api";
import type { SystemStatus } from "@/lib/types";

const NAV = [
  { href: "/", label: "Studio", icon: LayoutDashboard },
  { href: "/jobs", label: "Render queue", icon: ListVideo },
  { href: "/assets", label: "Assets", icon: LibraryBig },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .status()
        .then((value) => alive && setStatus(value))
        .catch(() => alive && setStatus(null));
    load();
    const timer = window.setInterval(load, 20000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const queue = status?.queue;

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-ink-700/70 bg-ink-900/60 px-4 py-6 backdrop-blur md:flex">
        <Link href="/" className="mb-7 flex items-center gap-2 px-1">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-flare-500 to-amber-glow text-sm font-black text-ink-950">
            CF
          </span>
          <span className="leading-tight">
            <span className="block text-sm font-black tracking-wide text-mist-200">CLIPFORGE</span>
            <span className="block text-[10px] font-semibold uppercase tracking-[0.18em] text-flare-400">AI studio</span>
          </span>
        </Link>
        <nav className="flex flex-1 flex-col gap-1">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link key={item.href} href={item.href} className={`btn ${active ? "btn-ghost" : "btn-quiet"} justify-start`}>
                <Icon size={16} />
                {item.label}
                {item.href === "/jobs" && queue && queue.queued + queue.running > 0 ? (
                  <span className="ml-auto chip">
                    {queue.running > 0 ? `${queue.running} running` : `${queue.queued} queued`}
                  </span>
                ) : null}
              </Link>
            );
          })}
        </nav>
        <div className="mt-4 space-y-2 border-t border-ink-700/70 pt-4 text-[11px] text-mist-400">
          <p className="flex items-center gap-1.5">
            <Cpu size={12} /> {status?.hardware?.os ?? "detecting hardware…"}
          </p>
          <p className="flex items-center gap-1.5">
            <Film size={12} /> FFmpeg {status?.ffmpeg?.available ? status.ffmpeg.version || "ready" : "missing"}
          </p>
          <p className="flex items-center gap-1.5">
            <FolderCog size={12} /> {status?.data_dir ? status.data_dir.split(/[\\/]/).slice(-2).join("/") : ""}
          </p>
          {status?.ready === false ? <p className="text-flare-400">Not ready</p> : null}
        </div>
      </aside>

      <main className="min-w-0 flex-1">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-ink-700/70 bg-ink-950/80 px-5 py-3 backdrop-blur md:hidden">
          <Link href="/" className="flex items-center gap-2">
            <span className="grid h-7 w-7 place-items-center rounded-md bg-gradient-to-br from-flare-500 to-amber-glow text-xs font-black text-ink-950">
              CF
            </span>
            <span className="text-sm font-black text-mist-200">CLIPFORGE</span>
          </Link>
          <nav className="ml-auto flex gap-1">
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <Link key={item.href} href={item.href} className="btn btn-quiet p-2" aria-label={item.label}>
                  <Icon size={16} />
                </Link>
              );
            })}
          </nav>
        </header>
        <div className="mx-auto w-full max-w-6xl px-5 py-6">{children}</div>
      </main>
    </div>
  );
}
