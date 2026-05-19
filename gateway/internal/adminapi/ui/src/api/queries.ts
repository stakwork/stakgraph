// Per-endpoint hooks. One hook per route is the contract — pages
// import a hook and don't think about cache keys, polling intervals,
// or retry policy. Those decisions live here, in one place, so
// "should the dashboard poll every 30s" stays a one-line edit.

import { useQueries, useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";
import type {
  AgentBudgetResponse,
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

// ─── /agents/:name/budget ───────────────────────────────────────────
//
// 30s poll — the cap doesn't change often, but the spent-against-cap
// number wants to feel live, especially during the demo as an
// operator fires calls and watches the bar fill.

export function useAgentBudget(name: string | undefined) {
  return useQuery({
    queryKey: ["agents", name, "budget"],
    queryFn: () =>
      apiFetch<AgentBudgetResponse>(
        `/agents/${encodeURIComponent(name!)}/budget`
      ),
    enabled: !!name,
    refetchInterval: 30_000,
    staleTime: 10_000,
  });
}

// `useAgentBudgets` fans out per-agent fetches for the Agents list
// page — one query per row. Tanstack's `useQueries` is the right
// primitive: variable-length list of queries, all subject to the
// rules-of-hooks at the call boundary, results indexed positionally.
//
// Returns a name → response map for ergonomic lookup in the table
// cell renderer. `undefined` for in-flight / missing.
export function useAgentBudgets(names: string[]) {
  const queries = useQueries({
    queries: names.map((n) => ({
      queryKey: ["agents", n, "budget"],
      queryFn: () =>
        apiFetch<AgentBudgetResponse>(
          `/agents/${encodeURIComponent(n)}/budget`
        ),
      refetchInterval: 30_000,
      staleTime: 10_000,
    })),
  });
  const out: Record<string, AgentBudgetResponse | undefined> = {};
  names.forEach((n, i) => {
    out[n] = queries[i]?.data;
  });
  return out;
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
