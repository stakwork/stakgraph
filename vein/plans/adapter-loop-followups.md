# Adapter loop — follow-ups

Context: we shipped the "safe inner loop" for authoring integration adapters —
the standard `http`/`secrets` capabilities (`capabilities.ts`), record/replay
cassettes (`cassette.ts`), single-step running (`run-step.ts` + `POST
/steps/:type/run` + the `run_step` agent tool), and routed the core `http` step
through `ctx.services.http`. This doc tracks what we deliberately deferred.

Status of the shipped slice: done, 372 tests green. Everything below is NOT yet
built.

---

## 1. Credentials / secrets UI + store (the headline)

**Why deferred:** the `secrets` *seam* is in (`ctx.services.secrets.get(name)`,
env-backed by default), which is the part that costs nothing and future-proofs
adapters. A real credential *system* is a big, security-sensitive subsystem whose
requirements we'll only know after ~10 real adapters. Don't build the vault
before the adapters that consume it.

**What it needs (later):**
- A `SecretsCapability` implementation backed by a real store (encrypted at rest)
  instead of `process.env`. Swappable behind the existing interface — **zero
  adapter changes** (same move as `RunStore`).
- A UI to manage secrets: list which names exist, set/rotate a value, see which
  steps reference which secret. Per-deployment scope first; per-publisher /
  per-tenant later.
- Wiring: inject the store-backed `secrets` into `createVein`'s services bag
  (replacing/overriding the env default).
- **Coupling already handled:** the cassette recorder scrubs secret VALUES (read
  through `services.secrets`) to `{{secret:NAME}}`, so a real store doesn't change
  the leak-free guarantee.
- Open question: how does the chat agent obtain a credential to RECORD the first
  live cassette? Probably a UI prompt ("this adapter needs STRIPE_KEY") rather
  than the agent ever seeing the value. After the first record, replay needs no
  creds, so this only matters once per integration.

**Decision rule unchanged:** secrets always flow through `ctx.services.secrets`,
never `process.env`, in adapter steps.

---

## 2. Cassette management UI

**Why deferred:** the engine half (record/replay, persistence under
`steps/_cassettes/<name>.json`) is enough to make the loop work via the agent
tool + endpoint.

**What it needs (later):**
- View a step's cassette(s): the recorded `(key, args) → result` entries.
- Delete / re-record a cassette (e.g. when the upstream API changed shape).
- Multiple named scenarios per step (`cassetteName`) surfaced in the UI.
- Show in the StepRunFlyout whether a run_step was live / recorded / replayed.

---

## 3. `run_step` in the web UI

**Why deferred:** the agent uses the `run_step` tool and the `POST
/steps/:type/run` endpoint is callable; the browser UI just doesn't expose a
button yet.

**What it needs:** (per AGENTS.md "adding an API endpoint")
- `web/src/api.ts`: typed `runStep(type, { config, input, params, cassette })`.
- A "Run this step" affordance in `StepEditFlyout` / the Add Step flow, with a
  config editor and a record/replay toggle, showing `{ status, output, events }`.
- `/steps` is already proxied by the Vite dev config, so no proxy change.

---

## 4. SDK allowlist (mostly a docs/convention item)

**State:** already works — the registry loader does a bare dynamic `import()`,
so an adapter CAN `import` a vendor SDK *if the host pre-installed it* (pro users
consume vein as a lib and can add deps). The agent is told to prefer raw REST and
only import a pre-installed package.

**Optional later:** bake a curated allowlist (e.g. `stripe`, `@octokit/rest`)
into the host `package.json` and surface it to the agent in the prompt. Note: an
SDK does its own networking, so it bypasses `services.http` and is NOT cassette-
recordable — prefer exposing such an SDK as a *service* if record/replay matters.

---

## 5. Verify mcp `/lab` with the new default services

**Why:** `createVein` now auto-merges `standardServices()` (`{ http, secrets }`)
into the default services bag (was `{}`). It's additive and consumer-overridable,
and the one test asserting the old empty default was updated.

**To do:** confirm mcp's `/lab` (which copies vein as a `file:../vein` dep and may
introspect `vein.services`) still boots and behaves. Check for any code that
assumed an empty default services bag.

---

## 6. Live agent end-to-end check

**Why:** we tested the mechanics (runSingleStep, cassette, endpoint, core http
rerouting) but NOT an actual LLM turn driving `create_step → run_step(record) →
run_step(replay) → edit_step`.

**To do:** one real chat run (needs `ANTHROPIC_API_KEY`): ask the agent to build
an adapter for a public API, watch it use `ctx.services.http` (it can read
`get_step("http")` as the reference) and the cassette loop. This is the real
end-to-end validation of the feature.

---

## 7. Type safety for adapter authoring (the REAL win: typecheck-in-the-loop)

**Rejected:** defaulting `defineStep`'s `TServices` to `VeinCapabilities` (so
`ctx.services.http`/`secrets` type without annotation). It's a breaking type
change for consumers — mcp lab steps cast `ctx.services as ConceptServices`, and
`VeinCapabilities` (disjoint from those bags) makes that a `TS2352` ("convert to
unknown first"). Marginal benefit anyway: the runner erases services to
`unknown` (author-time only), `res.body` stays `unknown`, and custom steps load
via tsx with NO typecheck — so the LLM gets zero benefit. Kept `unknown` default;
`VeinCapabilities` is exported for OPT-IN annotation
(`defineStep<"t", In, Out, VeinCapabilities>(…)`).

**The actually-useful version:** add a **typecheck to `create_step` / `edit_step`**
so the AGENT gets real diagnostics. After writing the source, compile it (e.g.
`ts.transpileModule` won't catch type errors — need a real `Program` or
`tsc --noEmit` against the file with a synthetic `services` type) and return any
errors to the model so it can fix `ctx.services.htttp`, a wrong config field, or
a bad return shape BEFORE publishing. This is the only path that gives the LLM
type feedback (it can't get it from tsx at runtime). Pairs well with run_step:
typecheck → run_step(record) → run_step(replay).

## 8. Cassette matching robustness (only if needed)

Current matching is exact `(key, stableJson(args))` with in-order consumption of
repeated identical calls. Possible later needs: ignore volatile headers/fields,
TTL / staleness, partial-arg matching. Don't build until a real adapter needs it.
