// Per-endpoint hooks. One hook per route is the contract — pages
// import a hook and don't think about cache keys, polling intervals,
// or retry policy. Those decisions live here, in one place, so
// "should the dashboard poll every 30s" stays a one-line edit.

import {
  queryOptions,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { apiFetch, ApiCallError, getErrorMessage } from "./client";
import type {
  AgentBudgetResponse,
  AgentCatalogResponse,
  AgentEvalsResponse,
  AgentStateResponse,
  KillAgentResponse,
  KillRunResponse,
  RunStateResponse,
  EvalRefResponse,
  EvalSetDetailResponse,
  CatalogListResponse,
  CallDetailResponse,
  HistogramCostResponse,
  MeResponse,
  RunDetailResponse,
  SpendByAgentResponse,
  SpendByAgentUserResponse,
  SpendByUserResponse,
  TlogStatusResponse,
  UserDetailResponse,
} from "./types";
import type {
  TrustOrg,
  TrustStatus,
  Window,
  Bucket,
  Dimension,
} from "./manual";

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

// ─── hot state: /runs/:id/state · /agents/:name/state ──────────────
//
// Phase-9 live state over the phase-6 Redis routes. Both endpoints
// 503 when the swarm has no Redis. That is a property of the swarm,
// not a transient failure, so the hooks fold it into `data === null`
// (pages render an inline "hot state unavailable" note and disable
// the kill switch) and slow polling to a 60s retry so the card
// recovers on its own once the link is up. `undefined` = in flight.
//
// Cadence (phase 9 "Data-fetching contract"), all at the hook level:
//
//   run state    2s while the run is in flight, 30s once terminal,
//                500ms for KILL_BOOST_MS right after a kill/unkill so
//                the operator watches the flag flip.
//   agent state  10s on AgentDetail ("agent current bucket state"),
//                30s per row on the Agents list (list cadence).
//
// "Is the run in flight" is the page's call — it has the call log
// and the step counter (see RunDetail's LiveStateCard); the hook just
// takes the boolean.

const KILL_BOOST_MS = 30_000;

// "<kind>:<id>" → epoch-ms until which that target's /state polls at
// 500ms. Module-level rather than React state so the mutation hooks
// and the state hooks share it without threading props through the
// pages. Entries lapse on read.
const killBoostUntil = new Map<string, number>();

function boosted(key: string): boolean {
  const until = killBoostUntil.get(key);
  if (until === undefined) return false;
  if (Date.now() >= until) {
    killBoostUntil.delete(key);
    return false;
  }
  return true;
}

function boost(key: string) {
  killBoostUntil.set(key, Date.now() + KILL_BOOST_MS);
}

async function fetchHotState<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiCallError && e.status === 503) {
      return null; // redis not configured on this swarm
    }
    throw e;
  }
}

const runStateKey = (runID: string) => ["runs", runID, "state"] as const;
const agentStateKey = (name: string) => ["agents", name, "state"] as const;

export function useRunState(
  runID: string | undefined,
  opts: { inFlight?: boolean } = {},
) {
  const { inFlight = false } = opts;
  return useQuery({
    queryKey: runStateKey(runID ?? ""),
    queryFn: () =>
      fetchHotState<RunStateResponse>(
        `/runs/${encodeURIComponent(runID!)}/state`,
      ),
    enabled: !!runID,
    refetchInterval: (q) => {
      if (q.state.data === null) return 60_000; // redis off: slow retry
      if (runID && boosted("run:" + runID)) return 500;
      return inFlight ? 2_000 : 30_000;
    },
    staleTime: 0,
    retry: false, // 503 is folded into data; 400 (bad id) won't improve
  });
}

// Shared by useAgentState (detail page) and useAgentStates (list
// fan-out) so both observe the same cache entry — a kill from the
// detail page updates the list's badge for free.
function agentStateOptions(name: string, idleInterval: number) {
  return queryOptions({
    queryKey: agentStateKey(name),
    queryFn: () =>
      fetchHotState<AgentStateResponse>(
        `/agents/${encodeURIComponent(name)}/state`,
      ),
    refetchInterval: (q) => {
      if (q.state.data === null) return 60_000;
      if (boosted("agent:" + name)) return 500;
      return idleInterval;
    },
    staleTime: 0,
    retry: false,
  });
}

export function useAgentState(name: string | undefined) {
  return useQuery({
    ...agentStateOptions(name ?? "", 10_000),
    enabled: !!name,
  });
}

// Per-row fan-out for the Agents list — one /state query per visible
// agent, same idiom as useAgentBudgets. The list is small; a batch
// endpoint is a later optimisation. `data`: `null` ⇒ redis off (every
// row will be null in that case), `undefined` ⇒ in flight. `error`
// carries a per-row failure (a 400 on a name the kill routes reject)
// so the column can show "—" with a reason instead of a forever "…".
export function useAgentStates(names: string[]) {
  const queries = useQueries({
    queries: names.map((n) => agentStateOptions(n, 30_000)),
  });
  const data: Record<string, AgentStateResponse | null | undefined> = {};
  const error: Record<string, string | undefined> = {};
  names.forEach((n, i) => {
    data[n] = queries[i]?.data;
    error[n] = queries[i]?.isError ? getErrorMessage(queries[i].error) : undefined;
  });
  return { data, error };
}

// ─── POST/DELETE /runs/:id/kill · /agents/:name/kill ────────────────
//
// Destructive mutations (phase 9 "Destructive"): no optimistic update
// — the operator clicks Kill to *see* the kill land, so the badge
// only flips once /state says so. On settle we invalidate the
// matching state query and arm the 500ms poll boost. Agent kills
// invalidate `["agents", name, "state"]`, which is the same cache
// entry the Agents list's per-row column observes, so the list
// updates too. `apiFetch` already sends the `X-Bifrost-CSRF` header
// on every request; nothing extra is needed for the cookie session.

export function useKillRun(runID: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<KillRunResponse>(`/runs/${encodeURIComponent(runID)}/kill`, {
        method: "POST",
      }),
    retry: false,
    onSuccess: () => boost("run:" + runID),
    onSettled: () => qc.invalidateQueries({ queryKey: runStateKey(runID) }),
  });
}

export function useUnkillRun(runID: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<void>(`/runs/${encodeURIComponent(runID)}/kill`, {
        method: "DELETE",
      }),
    retry: false,
    onSuccess: () => boost("run:" + runID),
    onSettled: () => qc.invalidateQueries({ queryKey: runStateKey(runID) }),
  });
}

export function useKillAgent(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<KillAgentResponse>(
        `/agents/${encodeURIComponent(name)}/kill`,
        { method: "POST" },
      ),
    retry: false,
    onSuccess: () => boost("agent:" + name),
    onSettled: () => qc.invalidateQueries({ queryKey: agentStateKey(name) }),
  });
}

export function useUnkillAgent(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<void>(`/agents/${encodeURIComponent(name)}/kill`, {
        method: "DELETE",
      }),
    retry: false,
    onSuccess: () => boost("agent:" + name),
    onSettled: () => qc.invalidateQueries({ queryKey: agentStateKey(name) }),
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

// ─── /tlog/status ───────────────────────────────────────────────────
//
// Phase-12 transparency log, as the gateway itself sees it: leaf
// count, root, per-boot log key, newest leaf, up/down. Local facts
// only — witnessing is Hive's — and never the leaves or the STH
// signature, which stay on the bearer-only /tlog/sth witness route.
//
// 10s poll: the agent hot-state card's cadence (useAgentState). The
// card should visibly tick as leaves land without the run-state 2s
// burst. The route answers 200 even when the log is disabled
// (`healthy: false` + `error`), so an error from this hook is a real
// fetch failure and the card renders "Unavailable", not "Disabled".

export function useTlogStatus() {
  return useQuery({
    queryKey: ["tlog", "status"],
    queryFn: () => apiFetch<TlogStatusResponse>("/tlog/status"),
    refetchInterval: 10_000,
    staleTime: 5_000,
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

// ─── Evals (agent-detail tab) ───────────────────────────────────────
//
// Reads (list-for-agent, set detail) hit neo4j directly through the
// gateway. Writes (create / link / update / delete / run) are
// delegated by the gateway to Hive, so a mutation's success is only
// as fresh as the subsequent read — we invalidate the relevant query
// on settle rather than optimistically patching.
//
// A 503 "hive_not_connected" from a write means this swarm hasn't had
// its Hive callback config pushed yet; the error bubbles up so the
// EvalsView can show a "connect to Hive" notice.

const AGENT_EVALS_KEY = (name: string) => ["agents", name, "evals"];
const EVAL_SET_KEY = (setId: string) => ["evals", setId];

export function useAgentEvals(name: string | undefined) {
  return useQuery({
    queryKey: ["agents", name, "evals"],
    queryFn: () =>
      apiFetch<AgentEvalsResponse>(`/agents/${encodeURIComponent(name!)}/evals`),
    enabled: !!name,
    staleTime: 60_000,
    retry: false,
  });
}

export function useEvalSet(setId: string | undefined) {
  return useQuery({
    queryKey: ["evals", setId],
    queryFn: () =>
      apiFetch<EvalSetDetailResponse>(`/evals/${encodeURIComponent(setId!)}`),
    enabled: !!setId,
    staleTime: 30_000,
    retry: false,
  });
}

// Create a new eval set and link it to this agent (or link an existing
// one when `setId` is provided). Backed by POST /agents/:name/evals.
export function useCreateOrLinkEvalSet(agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { name?: string; description?: string; setId?: string }) =>
      apiFetch<EvalRefResponse>(`/agents/${encodeURIComponent(agentName)}/evals`, {
        method: "POST",
        body: {
          name: args.name,
          description: args.description,
          set_id: args.setId,
        },
      }),
    onSettled: () =>
      qc.invalidateQueries({ queryKey: AGENT_EVALS_KEY(agentName) }),
  });
}

// Remove a set from this agent (unlink only — the set itself survives).
export function useUnlinkEvalSet(agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (setId: string) =>
      apiFetch<void>(
        `/agents/${encodeURIComponent(agentName)}/evals/${encodeURIComponent(setId)}`,
        { method: "DELETE" },
      ),
    onSettled: () =>
      qc.invalidateQueries({ queryKey: AGENT_EVALS_KEY(agentName) }),
  });
}

// Delete a set outright (via Hive) and refresh the agent's list.
export function useDeleteEvalSet(agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (setId: string) =>
      apiFetch<void>(`/evals/${encodeURIComponent(setId)}`, { method: "DELETE" }),
    onSettled: () =>
      qc.invalidateQueries({ queryKey: AGENT_EVALS_KEY(agentName) }),
  });
}

export function useUpdateEvalSet(setId: string, agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { name?: string; description?: string }) =>
      apiFetch<void>(`/evals/${encodeURIComponent(setId)}`, {
        method: "PATCH",
        body: args,
      }),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: EVAL_SET_KEY(setId) });
      qc.invalidateQueries({ queryKey: AGENT_EVALS_KEY(agentName) });
    },
  });
}

interface RequirementWrite {
  name?: string;
  description?: string;
  prompt_snippet?: string;
  desirable_cases?: string[];
  undesirable_cases?: string[];
}

export function useCreateRequirement(setId: string, agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: RequirementWrite) =>
      apiFetch<EvalRefResponse>(
        `/evals/${encodeURIComponent(setId)}/requirements`,
        { method: "POST", body: args },
      ),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: EVAL_SET_KEY(setId) });
      qc.invalidateQueries({ queryKey: AGENT_EVALS_KEY(agentName) });
    },
  });
}

export function useUpdateRequirement(setId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { reqId: string } & RequirementWrite) => {
      const { reqId, ...body } = args;
      return apiFetch<void>(
        `/evals/${encodeURIComponent(setId)}/requirements/${encodeURIComponent(reqId)}`,
        { method: "PATCH", body },
      );
    },
    onSettled: () => qc.invalidateQueries({ queryKey: EVAL_SET_KEY(setId) }),
  });
}

export function useDeleteRequirement(setId: string, agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (reqId: string) =>
      apiFetch<void>(
        `/evals/${encodeURIComponent(setId)}/requirements/${encodeURIComponent(reqId)}`,
        { method: "DELETE" },
      ),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: EVAL_SET_KEY(setId) });
      qc.invalidateQueries({ queryKey: AGENT_EVALS_KEY(agentName) });
    },
  });
}

// Dispatch an eval run for a requirement. Hive fires the Stakwork
// workflow; outputs land back in Jarvis, so we invalidate the set
// detail on settle to pick up the new results on the next poll.
export function useRunRequirement(setId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { reqId: string; agent?: string }) =>
      apiFetch<{ project_ids?: unknown[] }>(
        `/evals/${encodeURIComponent(setId)}/requirements/${encodeURIComponent(args.reqId)}/run`,
        { method: "POST", body: { agent: args.agent } },
      ),
    onSettled: () => qc.invalidateQueries({ queryKey: EVAL_SET_KEY(setId) }),
  });
}
