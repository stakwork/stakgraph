import type { HttpResponse, StepContext } from "strut";

/**
 * Shared by the pod/* steps (a seeded helper; the registry skips a leading
 * `_`). A pod is a hive sandbox: a container with the workspace's
 * repositories, a dev server, an IDE and staklink — a control server on the
 * pod that runs a coding agent (goose), resets repositories, diffs, pushes
 * and tests. Hive's pool manager hands one out (`pod/claim`) and takes it
 * back (`pod/release`); everything else talks to staklink at the pod's
 * `control` URL, as the pod, with its password.
 *
 * THE PASSWORD IS A CREDENTIAL. It never rides in a step's output, the run
 * log or an agent's context: `pod/claim` returns it SEALED (AES-GCM under a
 * key derived from HIVE_API_KEY — whoever holds that key can claim pods
 * anyway, so the seal adds no trust and needs no second secret), and every
 * other step takes `sealed` and opens it in-process, for the one request.
 */

export type PodCtx = StepContext<any>;

async function need(ctx: PodCtx, name: string, what: string): Promise<string> {
  const v = await ctx.services?.secrets?.get(name);
  if (!v) throw new Error(`${name} secret is not set — ${what}`);
  return String(v);
}

/** Hive's pool manager, as the org: the base URL and the key header. */
export async function hiveApi(ctx: PodCtx): Promise<{ base: string; headers: Record<string, string> }> {
  const base = await need(ctx, "HIVE_URL", "the hive this strut claims pods from (its base URL)");
  const key = await need(ctx, "HIVE_API_KEY", "an org-scoped hive API key with pool-manager access");
  return { base: base.replace(/\/+$/, ""), headers: { "x-api-token": key } };
}

// ── the sealed password ───────────────────────────────────────────────────

const enc = new TextEncoder();
const b64 = (u: Uint8Array) => Buffer.from(u).toString("base64");

async function sealKey(ctx: PodCtx, usage: "encrypt" | "decrypt"): Promise<CryptoKey> {
  const secret = await need(ctx, "HIVE_API_KEY", "the pod password is sealed under it");
  const raw = await crypto.subtle.digest("SHA-256", enc.encode("strut-pod-seal:v1:" + secret));
  return crypto.subtle.importKey("raw", raw, "AES-GCM", false, [usage]);
}

export async function seal(ctx: PodCtx, password: string): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await sealKey(ctx, "encrypt");
  const ct = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, enc.encode(password)));
  return `v1.${b64(iv)}.${b64(ct)}`;
}

export async function unseal(ctx: PodCtx, sealed: string): Promise<string> {
  const [v, ivB, ctB] = String(sealed).split(".");
  if (v !== "v1" || !ivB || !ctB) throw new Error("`sealed` is not a sealed pod password (pod/claim returns one)");
  try {
    const key = await sealKey(ctx, "decrypt");
    const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv: Buffer.from(ivB, "base64") }, key, Buffer.from(ctB, "base64"));
    return new TextDecoder().decode(pt);
  } catch {
    throw new Error("could not open the sealed pod password (sealed under another HIVE_API_KEY?)");
  }
}

// ── staklink, the pod's control server ────────────────────────────────────

export const brief = (v: unknown, max = 400): string => {
  const s = typeof v === "string" ? v : (JSON.stringify(v) ?? String(v));
  return s.length > max ? `${s.slice(0, max)}…` : s;
};

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** One request to the pod's control server, as the pod. An error names the
 *  route, never the password; a 401 says what it means. */
export async function podCall(
  ctx: PodCtx,
  control: string,
  sealed: string,
  path: string,
  init: { method: string; body?: unknown; timeout?: number } = { method: "GET" },
): Promise<HttpResponse> {
  const password = await unseal(ctx, sealed);
  const url = `${String(control).replace(/\/+$/, "")}${path}`;
  let res: HttpResponse;
  try {
    res = await ctx.services.http(url, {
      method: init.method,
      headers: { Authorization: `Bearer ${password}` },
      ...(init.body !== undefined ? { body: init.body } : {}),
      ...(init.timeout ? { timeout: init.timeout } : {}),
    });
  } catch (err) {
    const why = message(err).split(password).join("[password]");
    throw new Error(`${init.method} ${url} failed — ${why}; is the pod's control URL reachable?`);
  }
  if (res.status === 401) {
    throw new Error(`${init.method} ${url}: 401 — the pod rejected its password (released and re-claimed? claim again)`);
  }
  return res;
}

/** Staklink's one job shape: a request submitted now, polled until it settles. */
export interface Progress {
  status: "pending" | "completed" | "failed";
  body: any;
}

export async function submit(
  ctx: PodCtx,
  control: string,
  sealed: string,
  path: string,
  method: string,
  body?: unknown,
): Promise<{ requestId: string } & Progress> {
  const res = await podCall(ctx, control, sealed, path, { method, body });
  if (res.status === 409) throw new Error(`${method} ${path}: 409 — the pod is busy with an earlier request; wait for it to finish`);
  if (!res.ok) throw new Error(`${method} ${path}: ${res.status} — ${brief(res.body)}`);
  const b: any = res.body ?? {};
  if (!b.request_id) throw new Error(`${method} ${path} returned no request_id: ${brief(b)}`);
  const status: Progress["status"] = b.status === "completed" || b.status === "failed" ? b.status : "pending";
  return { requestId: String(b.request_id), status, body: b };
}

export async function progress(ctx: PodCtx, control: string, sealed: string, requestId: string): Promise<Progress> {
  const path = `/script_progress?request_id=${encodeURIComponent(requestId)}`;
  const res = await podCall(ctx, control, sealed, path);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status} — ${brief(res.body)}`);
  const b: any = res.body ?? {};
  if (b.status !== "pending" && b.status !== "completed" && b.status !== "failed") {
    throw new Error(`GET ${path}: unexpected status ${JSON.stringify(b.status)}`);
  }
  return { status: b.status, body: b };
}

/** Wait for a submitted request to settle. Checks run control between polls,
 *  so a cancelled run stops waiting at once instead of at the deadline. */
export async function settle(
  ctx: PodCtx,
  control: string,
  sealed: string,
  first: { requestId: string } & Progress,
  opts: { pollMs: number; timeoutMs: number; what: string },
): Promise<Progress> {
  let p: Progress = first;
  const deadline = Date.now() + opts.timeoutMs;
  while (p.status === "pending") {
    if (Date.now() >= deadline) {
      throw new Error(`${opts.what}: still running after ${Math.round(opts.timeoutMs / 1000)} s (request ${first.requestId})`);
    }
    await new Promise((r) => setTimeout(r, opts.pollMs));
    await ctx.control?.checkpoint();
    p = await progress(ctx, control, sealed, first.requestId);
  }
  return p;
}

/** What a failed request says went wrong. */
export function failure(body: any): string {
  const e = body?.error ?? body?.result?.error;
  if (typeof e === "string") return e;
  if (e?.message) return String(e.message);
  return e ? brief(e) : "no error message";
}

/** Head + tail of a long text, the middle cut, so a result stays readable. */
export function capText(s: string, max: number): string {
  if (s.length <= max) return s;
  const half = Math.floor(max / 2);
  return `${s.slice(0, half)}\n… [${s.length - max} chars cut] …\n${s.slice(-half)}`;
}

// ── the pod's agent ───────────────────────────────────────────────────────

/** This run's LLM gateway grant — what the pod's agent calls the model with,
 *  so its spend lands on this run's principal like any step's. */
export async function llmGrant(ctx: PodCtx): Promise<{ apiKey: string; baseUrl: string; headers?: Record<string, string> }> {
  const llmAuth = ctx.services?.llmAuth;
  if (typeof llmAuth !== "function") {
    throw new Error("this strut has no LLM gateway (Mothership) configured; a pod's agent only ever calls the model through one, billed to this run");
  }
  const grant = await llmAuth({
    kind: "step",
    provider: "anthropic",
    runId: ctx.runId,
    workflow: ctx.path.split("/")[0],
    stepPath: ctx.path,
    ...(ctx.actor ? { actor: ctx.actor } : {}),
    ...(ctx.principal ? { principal: ctx.principal } : {}),
  });
  if (!grant?.apiKey || !grant?.baseUrl) {
    throw new Error("the LLM gateway has no delegation on file for this run's principal: nobody to bill the pod's agent to");
  }
  return grant;
}

export interface AgentAsk {
  prompt: string;
  system?: string | undefined;
  repoName?: string | undefined;
  model?: string | undefined;
  session?: string | undefined;
}

/** The body of a `POST /agent`: the ask, the session (the run's job unless
 *  given, so a job's pod agent remembers across turns) and the grant. The
 *  grant's macaroon rides INSIDE the key, `<vk>.<macaroon>`: goose's anthropic
 *  provider sends no custom headers, so an `x-macaroon` header handed to
 *  staklink never reached the gateway (2026-10-07, the first live pod job:
 *  `401 x-macaroon header is required`). The gateway's wrapper splits the key
 *  back into the two headers (stakgraph gateway/wrapper/authsplit.go). The
 *  grant's other headers — the `x-bf-dim-*` dims — still go as headers, for
 *  the providers goose carries them on. */
export async function agentBody(ctx: PodCtx, ask: AgentAsk): Promise<{ body: Record<string, unknown>; session: string | null }> {
  const grant = await llmGrant(ctx);
  const session = ask.session ?? ctx.job ?? null;
  const headers: Record<string, string> = {};
  let macaroon: string | undefined;
  for (const [k, v] of Object.entries(grant.headers ?? {})) {
    if (k.toLowerCase() === "x-macaroon") macaroon = v;
    else headers[k] = v;
  }
  const apiKey = macaroon ? `${grant.apiKey}.${macaroon}` : grant.apiKey;
  const body: Record<string, unknown> = { prompt: ask.prompt, apiKey, baseUrl: grant.baseUrl };
  if (ask.system !== undefined) body["system"] = ask.system;
  if (ask.repoName !== undefined) body["repoName"] = ask.repoName;
  if (ask.model !== undefined) body["model"] = ask.model;
  if (session !== null) body["session"] = session;
  if (Object.keys(headers).length) body["headers"] = headers;
  return { body, session };
}

/** A goose request's outcome, as the agent-shaped steps report it. */
export function agentOutcome(p: Progress, cap: number) {
  const r = p.body?.result;
  const text = typeof r === "string" ? r : typeof r?.result === "string" ? r.result : r != null ? JSON.stringify(r) : null;
  return {
    status: p.status,
    output: text == null ? null : capText(String(text), cap),
    summary: typeof r?.summary === "string" ? r.summary : null,
    usage: r?.usage ?? null,
    model: typeof r?.model === "string" ? r.model : null,
    error: p.status === "failed" ? failure(p.body) : null,
  };
}

// ── git, as the run ───────────────────────────────────────────────────────

export interface GitCredentials {
  provider: "github";
  auth_type: "pat";
  auth_data: { token: string; username: string };
}

/** The run's GitHub identity for staklink's git calls: the token under
 *  `secretName` (the actor's first, then the deployment's) and its login.
 *  Undefined when the run has none. */
export async function gitCredentials(ctx: PodCtx, secretName: string): Promise<GitCredentials | undefined> {
  const token = await ctx.services?.secrets?.get(secretName);
  if (!token) return undefined;
  const res = await ctx.services.http("https://api.github.com/user", {
    method: "GET",
    headers: { Authorization: `Bearer ${token}`, "User-Agent": "strut-pod" },
  });
  const login = (res.body as any)?.login;
  if (!res.ok || !login) throw new Error(`GET https://api.github.com/user: ${res.status} — ${secretName} is not a working GitHub token`);
  return { provider: "github", auth_type: "pat", auth_data: { token: String(token), username: String(login) } };
}
