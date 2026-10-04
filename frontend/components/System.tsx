"use client";

// Backend status shared by every page: whether the API is reachable, what this
// deployment allows (local paths only on the desktop app) and the queue size.

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import type { SystemStatus } from "@/lib/types";

type SystemState = {
  status: SystemStatus | null;
  /** True once the first request finished and failed. */
  offline: boolean;
  features: SystemStatus["features"];
  refresh: () => void;
};

const NO_FEATURES: SystemStatus["features"] = { local_paths: false, open_folder: false };

const SystemContext = createContext<SystemState>({
  status: null,
  offline: false,
  features: NO_FEATURES,
  refresh: () => undefined,
});

export function SystemProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(() => {
    api
      .status()
      .then((value) => {
        setStatus(value);
        setOffline(false);
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 20000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const value = useMemo(
    () => ({ status, offline, features: status?.features ?? NO_FEATURES, refresh }),
    [status, offline, refresh],
  );
  return <SystemContext.Provider value={value}>{children}</SystemContext.Provider>;
}

export function useSystem(): SystemState {
  return useContext(SystemContext);
}
