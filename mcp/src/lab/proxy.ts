/**
 * `POST /gitree/process` and `POST /repo/describe` as lab workflow launches
 * (docs/plans/gitree-to-workflows.md).
 *
 * A request that names an actor — `x-strut-actor` beside a matching
 * `x-api-token`, the rule `labAuth` applies to `/lab` — launches the lab's
 * `bootstrap-then-process` / `describe-code` as that actor through
 * `strut.run`: a tracked run, visible in /lab, cancellable, resumed by strut
 * after a restart, billed to the actor. The response and `/progress` keep
 * the old shape, so hive and standalone change nothing but the header.
 *
 * A request with no actor — mcp's own SPA (its JWT carries no `sub`), dev
 * mode, any older caller — goes to the old handler unchanged. Not a
 * rollout shim: it is the SPA's path until the SPA is reworked, and it is
 * what makes the mcp, hive and standalone deploys order-independent.
 *
 * The GitHub token never enters a run. A `token` in the query is stored as
 * the actor's `GITHUB_TOKEN` secret (the `PUT /actors/:actor/secrets/:name`
 * hive already calls before its own dispatches, here in-process), and the
 * concept steps read it through `ctx.services.secrets`, which strut binds
 * to the run's principal. A run's input and every step's config are written
 * to the run log; a secret is not.
 *
 * One sweep per repo: two concurrent `bootstrap-then-process` runs of one
 * repo would race on its checkpoint. The in-flight map is the first check;
 * on a miss, strut's own run list is the second — a run strut auto-resumed
 * after a restart is found there and attached to, never launched twice.
 * `describe-code` needs neither: it re-reads "nodes without a description"
 * on every batch.
 */

import type { Request as ExpressRequest, Response as ExpressResponse } from "express";
import { generateRunId, type RunEvent, type RunResult, type RunStore } from "strut";
import * as asyncReqs from "../graph/reqs.js";
import { hasApiKeyForProvider } from "../aieo/src/provider.js";

export const PROCESS_WORKFLOW = "bootstrap-then-process";
export const DESCRIBE_WORKFLOW = "describe-code";
/** The cheap describe model, passed when the deployment has an OpenRouter
 *  key (`describe_nodes_agent` picks the same one); the workflow's own
 *  default is `haiku`, which every deployment can call. */
export const DESCRIBE_MODEL = "openrouter/qwen/qwen3-coder-30b-a3b-instruct";

/** What the proxies use of the lab strut (a `Strut<LabServices>` satisfies it). */
export interface ProxyStrut {
  /** `strut.run`: the tracked launch — a controller, `run.start` with the
   *  actor and principal, auto-resume. */
  run(
    workflow: string,
    input?: unknown,
    opts?: { actor?: string; runId?: string; params?: Record<string, unknown> },
  ): Promise<RunResult>;
  app: { fetch(request: globalThis.Request): globalThis.Response | Promise<globalThis.Response> };
  store: Pick<RunStore, "getRunEvents" | "tailEvents">;
}

export type Handler = (req: ExpressRequest, res: ExpressResponse) => unknown;

export interface ProxyDeps {
  /** The lab strut, built on first use (`labStrut` in mount.ts). */
  strut(): Promise<ProxyStrut>;
  /** The old handlers, for a request with no actor. */
  fallback: { process: Handler; describe: Handler };
  /** Whether the deployment can call the OpenRouter describe model. */
  hasOpenRouterKey?: () => boolean;
}

// ── The actor ─────────────────────────────────────────────────────────────

interface HeaderSource {
  header(name: string): string | undefined;
}

/** The actor a request names: `x-strut-actor`, honored only beside a
 *  matching `x-api-token` (the token proves the caller is hive or
 *  standalone) — `labAuthorized`'s rule. The SPA's Bearer JWT, `?token=`,
 *  Basic, and dev mode with no `API_TOKEN` name nobody. */
export function actorOf(req: HeaderSource): string | undefined {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) return undefined;
  if (req.header("x-api-token") !== apiToken) return undefined;
  const actor = req.header("x-strut-actor")?.trim();
  return actor || undefined;
}

// ── Inputs ────────────────────────────────────────────────────────────────

const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);

/** `owner` + `repo`, or both parsed from `repo_url` (https, ssh, or bare
 *  `owner/repo`) — what `gitree_process` accepts. */
export function parseOwnerRepo(q: { owner?: unknown; repo?: unknown; repo_url?: unknown }): { owner: string; repo: string } | undefined {
  let owner = str(q.owner);
  let repo = str(q.repo);
  const url = str(q.repo_url);
  if (url && (!owner || !repo)) {
    let clean = url.replace(/\.git$/, "");
    const ssh = clean.match(/^git@[^:]+:(.+)$/);
    if (ssh) clean = ssh[1]!;
    else if (/^https?:\/\//.test(clean)) clean = clean.replace(/^https?:\/\/[^/]+\//, "");
    const parts = clean.split("/").filter(Boolean);
    if (parts.length >= 2) {
      owner = parts[0];
      repo = parts[1];
    }
  }
  return owner && repo ? { owner, repo } : undefined;
}

export interface DescribeLaunch {
  input: { repo_url?: string; file_paths: string[]; cost_limit: number };
  params: { batch_size: number; concurrency: number; model?: string };
}

/** The `describe-code` launch for a `/repo/describe` body, or the 400 it
 *  earns. `embed` is ignored (the embedding pass embeds); `apiKey` is
 *  refused (a key in a request has no place in a billed run). */
export function parseDescribeBody(
  body: Record<string, unknown>,
  hasOpenRouterKey: () => boolean,
): { ok: true; launch: DescribeLaunch } | { ok: false; error: string } {
  if (str(body.apiKey)) return { ok: false, error: "apiKey is not accepted on a billed run; the run uses the actor's delegation or the deployment's key" };
  const cost_limit = parseFloat(String(body.cost_limit ?? "0.5"));
  if (isNaN(cost_limit) || cost_limit <= 0) return { ok: false, error: "Invalid cost_limit. Must be a positive number." };
  const batch_size = parseInt(String(body.batch_size ?? "50"), 10);
  if (isNaN(batch_size) || batch_size <= 0) return { ok: false, error: "Invalid batch_size. Must be a positive integer." };
  const concurrency = parseInt(String(body.concurrency ?? "5"), 10);
  if (isNaN(concurrency) || concurrency <= 0) return { ok: false, error: "Invalid concurrency. Must be a positive integer." };
  const file_paths = Array.isArray(body.file_paths) ? body.file_paths.filter((p): p is string => typeof p === "string") : [];
  const repo_url = str(body.repo_url);
  const model = str(body.model) ?? (hasOpenRouterKey() ? DESCRIBE_MODEL : undefined);
  return {
    ok: true,
    launch: {
      input: { file_paths, cost_limit, ...(repo_url ? { repo_url } : {}) },
      params: { batch_size, concurrency, ...(model ? { model } : {}) },
    },
  };
}

// ── Results ───────────────────────────────────────────────────────────────

export function mapProcessResult(r: RunResult, repoId: string) {
  return { status: "success", message: `Processed ${repoId}`, run_id: r.runId, output: r.output ?? null };
}

export function mapDescribeResult(r: RunResult) {
  const o = (r.output ?? {}) as { processed?: number; failed?: number; cost?: number; stopped?: string };
  return {
    success: true,
    processed: o.processed ?? 0,
    total_cost: o.cost ?? 0,
    failed: o.failed ?? 0,
    stopped: o.stopped ?? "done",
    run_id: r.runId,
  };
}

function settle(request_id: string, r: RunResult, map: (r: RunResult) => unknown): void {
  if (r.status === "success") asyncReqs.finishReq(request_id, map(r));
  else if (r.status === "cancelled") asyncReqs.failReq(request_id, { message: "run cancelled", run_id: r.runId });
  else asyncReqs.failReq(request_id, { message: r.error?.message ?? "run failed", run_id: r.runId });
}

/** A `RunResult` from the terminal event of a run followed by its tail. */
function resultOf(runId: string, e: RunEvent | undefined): RunResult {
  if (!e) return { runId, status: "error", error: { message: "run log ended without a result" } };
  if (e.type === "run.end") return { runId, status: "success", output: e.output };
  if (e.type === "run.cancelled") return { runId, status: "cancelled" };
  return { runId, status: "error", error: e.error ?? { message: "run failed" } };
}

// ── In-flight sweeps ──────────────────────────────────────────────────────

interface InFlight {
  request_id: string;
  run_id: string;
}

/** `owner/repo` → the sweep running for it. */
const inFlight = new Map<string, InFlight>();
/** A launch in progress for a repo (between the run-list check and the
 *  map entry), so two requests in the same tick share one outcome. */
const launching = new Map<string, Promise<{ entry: InFlight; attached: boolean }>>();

/** Is a sweep of `repo` (`owner/repo`; any repo when omitted) in flight?
 *  What `GET /gitree/concepts` reports as `processing`. */
export function isRepoInFlight(repo?: string): boolean {
  return repo ? inFlight.has(repo) : inFlight.size > 0;
}

/** Test-only. */
export function _resetInFlight(): void {
  inFlight.clear();
  launching.clear();
}

function release(repoId: string, runId: string): void {
  if (inFlight.get(repoId)?.run_id === runId) inFlight.delete(repoId);
}

/** In-process requests to the lab strut. The lab strut has no key of its
 *  own; `Authorization` rides along in case a deployment gives it one. */
function labRequest(path: string, init?: RequestInit): globalThis.Request {
  const key = process.env.STRUT_API_KEY;
  const headers = new Headers(init?.headers);
  if (key) headers.set("authorization", `Bearer ${key}`);
  return new Request(`http://lab${path}`, { ...init, headers });
}

async function putActorSecret(strut: ProxyStrut, actor: string, name: string, value: string): Promise<void> {
  const res = await strut.app.fetch(
    labRequest(`/actors/${encodeURIComponent(actor)}/secrets/${name}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value }),
    }),
  );
  if (!res.ok) {
    const detail = (await res.text().catch(() => "")).slice(0, 200);
    throw new Error(`storing the actor's ${name} failed: ${res.status} ${detail}`);
  }
}

const LIVE = new Set(["running", "pausing", "paused", "cancelling"]);
/** How many summary-less runs to look inside for the repo. */
const RECENT_RUNS = 25;

/** A `bootstrap-then-process` run of `repoId` that strut holds live — one it
 *  auto-resumed after a restart, which this process never launched. A
 *  `stale` run (cut off and not resumed) is not one. */
async function findLiveRun(strut: ProxyStrut, repoId: string): Promise<string | undefined> {
  const res = await strut.app.fetch(labRequest(`/workflows/${PROCESS_WORKFLOW}/runs`));
  if (!res.ok) return undefined;
  const runs = (await res.json()) as Array<{ runId: string; status?: string }>;
  let seen = 0;
  for (const run of runs) {
    if (!LIVE.has(run.status ?? "")) continue;
    if (++seen > RECENT_RUNS) break;
    const start = (await strut.store.getRunEvents(PROCESS_WORKFLOW, run.runId)).find((e) => e.type === "run.start");
    const input = start?.input as { owner?: unknown; repo?: unknown } | undefined;
    if (input && `${input.owner}/${input.repo}` === repoId) return run.runId;
  }
  return undefined;
}

/** Track a run this process did not launch: a fresh request id that settles
 *  from the run's tail. */
function attach(strut: ProxyStrut, repoId: string, runId: string): InFlight {
  const request_id = asyncReqs.startReq();
  const entry = { request_id, run_id: runId };
  inFlight.set(repoId, entry);
  void (async () => {
    let terminal: RunEvent | undefined;
    for await (const e of strut.store.tailEvents(PROCESS_WORKFLOW, runId)) {
      if (e.type === "run.end" || e.type === "run.error" || e.type === "run.cancelled") {
        terminal = e;
        break;
      }
    }
    settle(request_id, resultOf(runId, terminal), (r) => mapProcessResult(r, repoId));
  })()
    .catch((e) => asyncReqs.failReq(request_id, e))
    .finally(() => release(repoId, runId));
  return entry;
}

async function processRun(
  strut: ProxyStrut,
  actor: string,
  owner: string,
  repo: string,
  token: string | undefined,
): Promise<{ entry: InFlight; attached: boolean }> {
  const repoId = `${owner}/${repo}`;
  const held = inFlight.get(repoId);
  if (held) return { entry: held, attached: true };
  const pending = launching.get(repoId);
  if (pending) return { entry: (await pending).entry, attached: true };

  const launch = (async () => {
    const resumed = await findLiveRun(strut, repoId);
    if (resumed) return { entry: attach(strut, repoId, resumed), attached: true };
    if (token) await putActorSecret(strut, actor, "GITHUB_TOKEN", token);
    const request_id = asyncReqs.startReq();
    const run_id = generateRunId();
    const entry = { request_id, run_id };
    inFlight.set(repoId, entry);
    strut
      .run(PROCESS_WORKFLOW, { owner, repo }, { actor, runId: run_id })
      .then(
        (r) => settle(request_id, r, (x) => mapProcessResult(x, repoId)),
        (e) => asyncReqs.failReq(request_id, e),
      )
      .finally(() => release(repoId, run_id));
    return { entry, attached: false };
  })();
  launching.set(repoId, launch);
  try {
    return await launch;
  } finally {
    launching.delete(repoId);
  }
}

// ── The handlers ──────────────────────────────────────────────────────────

export function createProxies(deps: ProxyDeps): { process: Handler; describe: Handler } {
  const hasOpenRouterKey = deps.hasOpenRouterKey ?? (() => hasApiKeyForProvider("openrouter"));

  const gitreeProcess: Handler = async (req, res) => {
    const actor = actorOf(req);
    if (!actor) return deps.fallback.process(req, res);
    const target = parseOwnerRepo(req.query as Record<string, unknown>);
    if (!target) {
      res.status(400).json({ error: "Missing owner/repo or repo_url" });
      return;
    }
    try {
      const strut = await deps.strut();
      const { entry, attached } = await processRun(strut, actor, target.owner, target.repo, str(req.query.token));
      res.json({ request_id: entry.request_id, status: "pending", run_id: entry.run_id, ...(attached ? { attached: true } : {}) });
    } catch (e) {
      console.error("[lab/proxy] gitree/process launch failed:", e);
      res.status(500).json({ error: e instanceof Error ? e.message : String(e) });
    }
  };

  const repoDescribe: Handler = async (req, res) => {
    const actor = actorOf(req);
    if (!actor) return deps.fallback.describe(req, res);
    const parsed = parseDescribeBody((req.body ?? {}) as Record<string, unknown>, hasOpenRouterKey);
    if (!parsed.ok) {
      res.status(400).json({ error: parsed.error });
      return;
    }
    try {
      const strut = await deps.strut();
      const request_id = asyncReqs.startReq();
      const run_id = generateRunId();
      strut
        .run(DESCRIBE_WORKFLOW, parsed.launch.input, { actor, runId: run_id, params: parsed.launch.params })
        .then(
          (r) => settle(request_id, r, mapDescribeResult),
          (e) => asyncReqs.failReq(request_id, e),
        );
      res.json({ request_id, status: "pending", run_id, message: "Started description generation job" });
    } catch (e) {
      console.error("[lab/proxy] repo/describe launch failed:", e);
      res.status(500).json({ error: e instanceof Error ? e.message : String(e) });
    }
  };

  return { process: gitreeProcess, describe: repoDescribe };
}
