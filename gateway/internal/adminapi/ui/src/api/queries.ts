// Per-endpoint hooks. One hook per route is the contract — pages
// import a hook and don't think about cache keys, polling intervals,
// or retry policy. Those decisions live here, in one place, so
// "should the dashboard poll every 30s" stays a one-line edit.

import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { apiFetch, ApiCallError } from "./client";
import type {
  AgentBudgetResponse,
  AgentCatalogResponse,
  CatalogListResponse,
  CallDetailResponse,
  HistogramCostResponse,
  MeResponse,
  RunDetailResponse,
  SpendByAgentResponse,
  SpendByAgentUserResponse,
  SpendByUserResponse,
  TrustOrg,
  TrustStatus,
  UserDetailResponse,
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

export function useSpendByAgent(window: Window, userID?: string) {
  // Optional userID scopes the rollup to one person's calls —
  // used by the UserDetail page to render "agents this user
  // invoked". Server-side filter so we don't fan out per row.
  const params = new URLSearchParams({ window });
  if (userID) params.set("user_id", userID);
  return useQuery({
    queryKey: ["spend", "by-agent", window, userID ?? ""],
    queryFn: () =>
      apiFetch<SpendByAgentResponse>(`/spend/by-agent?${params.toString()}`),
    refetchInterval: 30_000,
    staleTime: 10_000,
  });
}

// ─── /spend/by-agent-user ───────────────────────────────────────────
//
// Single-pass (agent × user) crossing for the Canvas page. One
// round-trip replaces N parallel by-agent?user_id=… calls. 30s poll
// mirrors the other rollups so all three feel synchronized.

export function useSpendByAgentUser(window: Window) {
  return useQuery({
    queryKey: ["spend", "by-agent-user", window],
    queryFn: () =>
      apiFetch<SpendByAgentUserResponse>(
        `/spend/by-agent-user?window=${encodeURIComponent(window)}`
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
  /** Optional metadata filters narrowing the histogram to one
   *  user or one agent's contribution. Both translate to
   *  `metadata.<dim>=<value>` filters on the backend (see
   *  observability.go `metadataFilterFromQuery`). Either or both
   *  may be set. */
  userID?: string;
  agentName?: string;
}

export function useHistogramCost(args: HistogramCostArgs) {
  const { window, bucket, dimension, userID, agentName } = args;
  const params = new URLSearchParams({
    window,
    bucket,
    dimension,
  });
  if (userID) params.set("user_id", userID);
  if (agentName) params.set("agent_name", agentName);
  return useQuery({
    queryKey: [
      "histogram",
      "cost",
      window,
      bucket,
      dimension,
      userID ?? "",
      agentName ?? "",
    ],
    queryFn: () =>
      apiFetch<HistogramCostResponse>(`/histogram/cost?${params.toString()}`),
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

// ─── /agents/catalog (list) ─────────────────────────────────────────
//
// The whole registry — every catalog agent, traffic or not. The Agents
// list page unions this with spend-by-agent so seeded agents that have
// never been invoked still appear. Slow cadence (changes on deploy).
// 503 (neo4j not wired) ⇒ null; the page falls back to spend-only.

export function useAgentCatalogList() {
  return useQuery({
    queryKey: ["agents", "catalog-list"],
    queryFn: async (): Promise<CatalogListResponse | null> => {
      try {
        return await apiFetch<CatalogListResponse>("/agents/catalog");
      } catch (e) {
        if (e instanceof ApiCallError && e.status === 503) {
          return null; // catalog not configured on this swarm
        }
        throw e;
      }
    },
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
    retry: false,
  });
}

// ─── /agents/:name/catalog ──────────────────────────────────────────
//
// What the agent is _made of_ (prompts/tools/skills), pushed into the
// neo4j catalog by hive / prompt-manager / goose pods. Slow cadence:
// the catalog changes on deploy, not per second, so a long stale time
// and no aggressive poll. Two non-error "empty" states are folded into
// the data:
//
//   - 404 (agent has no catalog node yet) ⇒ resolves to `null`; the
//     page renders empty tabs rather than an error.
//   - 503 (neo4j not wired on this swarm) ⇒ propagates as an
//     ApiCallError with code "catalog_unavailable"; the page renders a
//     "catalog not wired" notice. Distinguished so the copy can differ.

export function useAgentCatalog(name: string | undefined) {
  return useQuery({
    queryKey: ["agents", name, "catalog"],
    queryFn: async (): Promise<AgentCatalogResponse | null> => {
      try {
        return await apiFetch<AgentCatalogResponse>(
          `/agents/${encodeURIComponent(name!)}/catalog`
        );
      } catch (e) {
        if (e instanceof ApiCallError && e.status === 404) {
          return null; // no catalog for this agent (yet)
        }
        throw e; // 503 "catalog_unavailable" and real failures bubble up
      }
    },
    enabled: !!name,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
    retry: false,
  });
}

// ─── PATCH /agents/:name/{tools,skills} (toggle enabled) ────────────
//
// The one mutable piece of catalog state the gateway owns (Hive seeds
// the palette; the operator toggles which tools/skills are active). On
// success we invalidate this agent's catalog query so the switch
// reflects the persisted state. Optimistic update keeps the toggle
// snappy and rolls back if the PATCH fails. Tools and skills share the
// exact same shape, so one internal hook drives both `view`s.

interface ChildToggleArgs {
  source: string;
  name: string;
  enabled: boolean;
}

function useToggleCatalogChild(agentName: string, view: "tools" | "skills") {
  const qc = useQueryClient();
  const key = ["agents", agentName, "catalog"];
  return useMutation({
    mutationFn: (args: ChildToggleArgs) =>
      apiFetch<{ name: string; source: string; enabled: boolean }>(
        `/agents/${encodeURIComponent(agentName)}/${view}`,
        { method: "PATCH", body: args },
      ),
    onMutate: async (args) => {
      await qc.cancelQueries({ queryKey: key });
      const prev = qc.getQueryData<AgentCatalogResponse | null>(key);
      if (prev) {
        const flip = <T extends ChildToggleArgs>(items: T[]) =>
          items.map((it) =>
            it.source === args.source && it.name === args.name
              ? { ...it, enabled: args.enabled }
              : it,
          );
        qc.setQueryData<AgentCatalogResponse | null>(key, {
          ...prev,
          ...(view === "tools"
            ? { tools: flip(prev.tools) }
            : { skills: flip(prev.skills) }),
        });
      }
      return { prev };
    },
    onError: (_err, _args, ctx) => {
      if (ctx?.prev !== undefined) qc.setQueryData(key, ctx.prev);
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: key });
    },
  });
}

export function useToggleSkill(agentName: string) {
  return useToggleCatalogChild(agentName, "skills");
}

export function useToggleTool(agentName: string) {
  return useToggleCatalogChild(agentName, "tools");
}

// ─── /users/:id ─────────────────────────────────────────────────────
//
// User detail page. 30s poll so the operator can fire a call in a
// terminal and watch the dashboard pick it up. `userID` may be the
// Hive UUID (production) or a friendly string (`u_alice`) in dev —
// the backend treats it as opaque either way.

export function useUserDetail(userID: string | undefined, window: Window) {
  return useQuery({
    queryKey: ["users", userID, window],
    queryFn: () =>
      apiFetch<UserDetailResponse>(
        `/users/${encodeURIComponent(userID!)}?window=${encodeURIComponent(window)}`
      ),
    enabled: !!userID,
    refetchInterval: 30_000,
    staleTime: 10_000,
  });
}

// ─── /trust/:org_id ─────────────────────────────────────────────────
//
// Reads one org's trust-registry entry. Used by the Provenance card
// on RunDetail to render "Authorized by <org>" with the pubkey + a
// verification badge.
//
// 404 (org not in registry) is a meaningful state — the UI renders
// "⚠ Not in trust registry" rather than a generic error. We swallow
// the 404 here so Tanstack's `data === null` carries that meaning.
//
// The endpoint is cookieOrBearer per server.go (read-only trust
// data is non-sensitive — pubkeys / issuer URLs are public-by-design).

export function useTrustOrg(orgID: string | undefined) {
  return useQuery({
    queryKey: ["trust", orgID],
    queryFn: async (): Promise<TrustOrg | null> => {
      try {
        return await apiFetch<TrustOrg>(
          `/trust/${encodeURIComponent(orgID!)}`
        );
      } catch (e) {
        // 404 ⇒ unknown org. Surface as `null` so the UI can
        // render the "not in registry" badge without confusing
        // it for a fetch failure.
        if (e instanceof ApiCallError && e.status === 404) {
          return null;
        }
        throw e;
      }
    },
    enabled: !!orgID,
    staleTime: 5 * 60_000, // org records change rarely
    retry: false,
  });
}

// ─── /trust/status ──────────────────────────────────────────────────
//
// Surfaces the swarm's self-identity (`realm_id`) plus the trusted
// org list. The Provenance card on RunDetail reads `realm_id` to
// render "this swarm processes realm w1" — phase 11 moved the realm
// off per-row metadata and onto this single, signed-out status
// surface. Long stale time: the value changes only when an operator
// hits PUT /_plugin/trust/realm_id, which is human-scale (workspace
// provisioning, debugging).

export function useTrustStatus() {
  return useQuery({
    queryKey: ["trust", "status"],
    queryFn: () => apiFetch<TrustStatus>("/trust/status"),
    staleTime: 5 * 60_000,
    retry: false,
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

// ─── /runs/:run_id/calls/:call_id ───────────────────────────────────
//
// Per-call drill-down. Fired only when the operator clicks a row in
// the RunDetail call log — `enabled` gates the request on a non-null
// callID, so closing the drawer doesn't keep the query alive but
// switching between rows pulls cleanly out of cache.
//
// `staleTime: Infinity` — once fetched, a single call's body never
// changes. Bifrost's row is immutable post-write.
export function useRunCall(
  runID: string | undefined,
  callID: string | undefined,
) {
  return useQuery({
    queryKey: ["runs", runID, "calls", callID],
    queryFn: () =>
      apiFetch<CallDetailResponse>(
        `/runs/${encodeURIComponent(runID!)}/calls/${encodeURIComponent(callID!)}`,
      ),
    enabled: !!runID && !!callID,
    staleTime: Infinity,
    retry: false, // 404 on cross-run / unknown id is meaningful; no retry
  });
}
