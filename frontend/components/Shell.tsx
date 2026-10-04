"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Film, FolderOpen, ListVideo, Settings, SlidersHorizontal } from "lucide-react";
import { api } from "@/lib/api";
import type { SystemStatus } from "@/lib/types";

const NAV = [
  { href: "/", label: "Studio", icon: SlidersHorizontal },
  { href: "/queue", label: "Queue", icon: ListVideo },
  { href: "/library", label: "Library", icon: FolderOpen },
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

  const busy = (status?.queue?.queued ?? 0) + (status?.queue?.running ?? 0);

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-line bg-surface px-3 py-4 md:flex">
        <Link href="/" className="mb-6 flex items-center gap-2 px-2">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-[11px] font-bold text-white">CF</span>
          <span className="text-[13px] font-semibold tracking-tight text-text">Clipforge</span>
        </Link>

        <nav className="flex flex-1 flex-col gap-0.5">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 rounded-md px-2 py-1.5 text-[13px] font-medium transition-colors ${
                  active ? "bg-surface-2 text-text" : "text-text-3 hover:bg-surface-2 hover:text-text-2"
                }`}
              >
                <Icon size={15} />
                {item.label}
                {item.href === "/queue" && busy > 0 ? (
                  <span className="ml-auto chip">{busy}</span>
                ) : null}
              </Link>
            );
          })}
        </nav>

        <div className="space-y-1 border-t border-line pt-3 text-[11px] text-text-3">
          <p className="flex items-center gap-1.5">
            <Film size={12} />
            {status?.ffmpeg?.available ? `FFmpeg ${status.ffmpeg.version?.split(" ")[0] ?? "ready"}` : "FFmpeg missing"}
          </p>
          <p className="truncate" title={status?.data_dir ?? ""}>
            {status?.data_dir ? status.data_dir.split(/[\\/]/).slice(-2).join("/") : "—"}
          </p>
        </div>
      </aside>

      <main className="min-w-0 flex-1">
        <header className="sticky top-0 z-30 flex items-center gap-1 border-b border-line bg-bg px-4 py-2.5 md:hidden">
          <Link href="/" className="mr-2 flex items-center gap-2">
            <span className="grid h-6 w-6 place-items-center rounded bg-accent text-[10px] font-bold text-white">CF</span>
            <span className="text-[13px] font-semibold text-text">Clipforge</span>
          </Link>
          <nav className="ml-auto flex gap-0.5">
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <Link key={item.href} href={item.href} className="btn btn-ghost btn-icon" aria-label={item.label}>
                  <Icon size={16} />
                </Link>
              );
            })}
          </nav>
        </header>
        <div className="mx-auto w-full max-w-5xl px-4 py-6 md:px-6">{children}</div>
      </main>
    </div>
  );
}
