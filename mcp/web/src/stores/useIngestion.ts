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

  setRunning: (requestId: string) => void;
  applyUpdate: (update: StatusUpdate) => void;
  setComplete: () => void;
  setError: (message: string) => void;
  reset: () => void;
}

export const useIngestion = create<IngestionState>((set) => ({
  phase: "idle",
  requestId: null,
  currentUpdate: null,
  errorMessage: null,

  setRunning: (requestId) => set({ phase: "running", requestId, currentUpdate: null, errorMessage: null }),
  applyUpdate: (update) => {
    set({ currentUpdate: update });
    if (update.status === "complete" || update.status === "Complete") {
      set({ phase: "complete" });
    } else if (update.status === "error" || update.status === "Error" || update.status === "Failed") {
      set({ phase: "error", errorMessage: update.message });
    }
  },
  setComplete: () => set({ phase: "complete" }),
  setError: (message) => set({ phase: "error", errorMessage: message }),
  reset: () => set({ phase: "idle", requestId: null, currentUpdate: null, errorMessage: null }),
}));
