/**
 * The lab actor — who a `/lab` request is from, as strut's opaque `actor`
 * string (plans/mothership-cost-control.md §2, §5) — and its scope: what it
 * may do there (strut plans/federation.md §3). `labAuth` (mount.ts) decides
 * both per credential and stashes them on the Node request; strut's
 * `resolveActor` / `resolveScope` hooks (createLabStrut.ts) read them back.
 *
 * Why a stash and not a header: the Express → Hono bridge is
 * `@hono/node-server`'s `getRequestListener`, which builds the Hono `Request`
 * from `incoming.rawHeaders` — anything written to `req.headers` after the
 * fact is invisible to strut. The listener does hand Hono `{ incoming,
 * outgoing }` as its env, and `incoming` IS the Express `req` (the same
 * object), so a property set here is readable there.
 */

const LAB_ACTOR = Symbol.for("stakgraph.lab.actor");

type Stash = { [LAB_ACTOR]?: string };

/** Record the actor on the Node request (`undefined` clears it). */
export function stashLabActor(req: object, actor: string | undefined): void {
  if (actor) (req as Stash)[LAB_ACTOR] = actor;
  else delete (req as Stash)[LAB_ACTOR];
}

/** The actor `labAuth` stashed on this Node request, if any. */
export function labActorOf(incoming: unknown): string | undefined {
  if (!incoming || typeof incoming !== "object") return undefined;
  const v = (incoming as Stash)[LAB_ACTOR];
  return typeof v === "string" && v ? v : undefined;
}

/**
 * strut's `resolveActor` hook for the lab: the stash, read through
 * `c.env.incoming`. Off the bridge (a test calling `app.fetch`, a smoke
 * script) there is no env and so no actor — never a guess.
 */
export function resolveLabActor(c: { env?: unknown }): string | undefined {
  const env = c.env;
  if (!env || typeof env !== "object") return undefined;
  return labActorOf((env as { incoming?: unknown }).incoming);
}

const LAB_SCOPE = Symbol.for("stakgraph.lab.scope");

type ScopeStash = { [LAB_SCOPE]?: "peer" };

/** Record that this request is a PEER's — another strut, holding a
 *  `lab:peer` token (`undefined` clears it: full, as every other credential). */
export function stashLabScope(req: object, scope: "peer" | undefined): void {
  if (scope) (req as ScopeStash)[LAB_SCOPE] = scope;
  else delete (req as ScopeStash)[LAB_SCOPE];
}

/**
 * strut's `resolveScope` hook for the lab: `"peer"` when `labAuth` stashed a
 * `lab:peer` token's grant, else `undefined` — strut's `full`, as before.
 * Strut enforces what a peer may do (its one gate and the control routes);
 * the lab only says who is one.
 */
export function resolveLabScope(c: { env?: unknown }): "peer" | undefined {
  const env = c.env;
  if (!env || typeof env !== "object") return undefined;
  const incoming = (env as { incoming?: unknown }).incoming;
  if (!incoming || typeof incoming !== "object") return undefined;
  return (incoming as ScopeStash)[LAB_SCOPE] === "peer" ? "peer" : undefined;
}
