// Per-endpoint hooks. One hook per route is the contract — pages
// import a hook and don't think about cache keys, polling intervals,
// or retry policy. Those decisions live here, in one place, so
// "should the dashboard poll every 30s" stays a one-line edit.

import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";
import type {
  HistogramCostResponse,
  MeResponse,
  RunDetailResponse,
  SpendByAgentResponse,
  SpendByUserResponse,
  Window,
  Bucket,
  Dimension,
} from "./types";

// ─── /me ─────────────────────────────────────────────────────────────
// Fires once at boot, plus on tab refocus (Tanstack default). Cheap
// enough that the second-order traffic is irrelevant.

export function useMe() {
  return useQuery({
    queryKey: ["me"],
    queryFn: () => apiFetch<MeResponse>("/me"),
    retry: false, // 401 short-circuits to login; no value in retrying.
    staleTime: 30_000,
  });
}

// ─── /spend/by-agent ────────────────────────────────────────────────
//
// Polled at 30s on the dashboard's rankings tables and the Agents
// page. Polling pauses automatically when the tab is hidden.

export function useSpendByAgent(window: Window) {
  return useQuery({
    queryKey: ["spend", "by-agent", window],
    queryFn: () =>
      apiFetch<SpendByAgentResponse>(
        `/spend/by-agent?window=${encodeURIComponent(window)}`
      ),
    refetchInterval: 30_000,
    staleTime: 10_000,
  });
}

// ─── /spend/by-user ─────────────────────────────────────────────────

export function useSpendByUser(window: Window) {
  return useQuery({
    queryKey: ["spend", "by-user", window],
    queryFn: () =>
      apiFetch<SpendByUserResponse>(
        `/spend/by-user?window=${encodeURIComponent(window)}`
      ),
    refetchInterval: 30_000,
    staleTime: 10_000,
  });
}

// ─── /histogram/cost ────────────────────────────────────────────────
//
// 60s poll. The chart is the heaviest single query in the dashboard —
// 30s would re-render on every panel and feels jumpy without buying
// the operator any new signal.

export interface HistogramCostArgs {
  window: Window;
  bucket: Bucket;
  dimension: Dimension;
  /** Optional agent name to scope to a single line (AgentDetail).
   *  Phase 8 honours this client-side by filtering the response —
   *  the backend doesn't yet accept a metadata filter on histograms.
   *  Documented as a known limitation in the AgentDetail page. */
  agentName?: string;
}

export function useHistogramCost(args: HistogramCostArgs) {
  const { window, bucket, dimension } = args;
  return useQuery({
    queryKey: ["histogram", "cost", window, bucket, dimension],
    queryFn: () =>
      apiFetch<HistogramCostResponse>(
        `/histogram/cost?window=${encodeURIComponent(
          window
        )}&bucket=${encodeURIComponent(bucket)}&dimension=${encodeURIComponent(
          dimension
        )}`
      ),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

// ─── /runs/:id ──────────────────────────────────────────────────────
//
// Run detail is historical — no polling. The user explicitly hits
// the page; refreshing the data is a manual page reload.

export function useRunDetail(runID: string | undefined) {
  return useQuery({
    queryKey: ["runs", runID],
    queryFn: () =>
      apiFetch<RunDetailResponse>(`/runs/${encodeURIComponent(runID!)}`),
    enabled: !!runID,
    staleTime: Infinity,
  });
}
