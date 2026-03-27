import { create } from "zustand";
import type { StatusUpdate } from "@/hooks/useSSE";

export type IngestionPhase = "idle" | "running" | "complete" | "error";

export interface RepoEntry {
  url: string;
  username: string;
  pat: string;
}

interface IngestionState {
  phase: IngestionPhase;
  requestId: string | null;
  currentUpdate: StatusUpdate | null;
  errorMessage: string | null;
  statsVersion: number;
  updateVersion: number;
  /** The repo URL(s) that were ingested (comma-separated if multiple) */
  repoUrl: string | null;
  /** Optional credentials for private repos */
  username: string | null;
  pat: string | null;

  setRunning: (requestId: string) => void;
  setRepo: (repoUrl: string, username?: string, pat?: string) => void;
  applyUpdate: (update: StatusUpdate) => void;
  setComplete: () => void;
  setError: (message: string) => void;
  reset: () => void;
}

function loadStoredRepo(): { repoUrl: string | null; username: string | null; pat: string | null } {
  try {
    const raw = localStorage.getItem("stakgraph_repo");
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore
  }
  return { repoUrl: null, username: null, pat: null };
}

export const useIngestion = create<IngestionState>((set) => {
  const stored = loadStoredRepo();
  return {
  phase: "idle",
  requestId: null,
  currentUpdate: null,
  errorMessage: null,
  statsVersion: 0,
  updateVersion: 0,
  repoUrl: stored.repoUrl,
  username: stored.username,
  pat: stored.pat,

  setRunning: (requestId) =>
    set({
      phase: "running",
      requestId,
      currentUpdate: null,
      errorMessage: null,
      statsVersion: 0,
    }),
  setRepo: (repoUrl, username, pat) => {
    set({
      repoUrl,
      username: username || null,
      pat: pat || null,
    });
    try {
      localStorage.setItem(
        "stakgraph_repo",
        JSON.stringify({ repoUrl, username: username || null, pat: pat || null }),
      );
    } catch {
      // localStorage unavailable
    }
  },
  applyUpdate: (update) => {
    set((state) => {
      const prev = state.currentUpdate;
      const merged: StatusUpdate = {
        status: update.status || prev?.status || "",
        message: update.message || prev?.message || "",
        step: update.step > 0 ? update.step : (prev?.step ?? 0),
        total_steps:
          update.total_steps > 0
            ? update.total_steps
            : (prev?.total_steps ?? 16),
        progress: update.progress > 0 ? update.progress : (prev?.progress ?? 0),
        stats: update.stats ?? prev?.stats ?? undefined,
        step_description:
          update.step_description || prev?.step_description || undefined,
      };

      const isComplete =
        update.status === "complete" || update.status === "Complete";
      const isError =
        update.status === "error" ||
        update.status === "Error" ||
        update.status === "Failed";

      if (isComplete) {
        return { currentUpdate: merged, phase: "complete" as IngestionPhase, updateVersion: state.updateVersion + 1 };
      }
      if (isError) {
        return {
          currentUpdate: merged,
          phase: "error" as IngestionPhase,
          errorMessage: update.message,
          updateVersion: state.updateVersion + 1,
        };
      }
      const statsVersion =
        update.stats != null ? state.statsVersion + 1 : state.statsVersion;
      return { currentUpdate: merged, statsVersion, updateVersion: state.updateVersion + 1 };
    });
  },
  setComplete: () => set({ phase: "complete" }),
  setError: (message) => set({ phase: "error", errorMessage: message }),
  reset: () => {
    set({
      phase: "idle",
      requestId: null,
      currentUpdate: null,
      errorMessage: null,
      statsVersion: 0,
      updateVersion: 0,
      repoUrl: null,
      username: null,
      pat: null,
    });
    try { localStorage.removeItem("stakgraph_repo"); } catch { /* */ }
  },
}});
