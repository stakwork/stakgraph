GOAL: AS SIMPLE AS POSSIBLE!

# vein

Minimal workflow engine with HTTP API and web UI. See `SPEC.md` for
the full design spec (sections 1-15). This file covers how to work
on the codebase.

## Stack

| Concern     | Choice                                                                     |
| ----------- | -------------------------------------------------------------------------- |
| Engine      | TypeScript (Node 22+), Zod for schemas, custom expression evaluator        |
| HTTP        | Hono + @hono/node-server                                                   |
| Persistence | Filesystem (JSONL events + JSON summaries). Swappable via `RunStore` iface |
| Web UI      | Preact + Vite + system-canvas-react. Vanilla CSS, no Tailwind              |
| Tests       | Node native test runner (`node:test`) via tsx                              |
| LLM step    | Vercel AI SDK (ai + @ai-sdk/anthropic + @ai-sdk/openai) — lazy-loaded      |
| AI builder  | Vercel AI SDK `ToolLoopAgent` + Anthropic, streamed to UI via `/chat` SSE  |

## Layout

```
vein/
├── SPEC.md                # full design spec — read this first
├── package.json           # engine deps (hono, zod, ai sdk)
├── tsconfig.json          # strict, Node16 module, types: ["node"]
├── src/
│   ├── core.ts            # flow(), step(), defineStep(), services bag, all types
│   ├── expr.ts            # {{ }} template evaluator (~460 LOC recursive descent)
│   ├── runner.ts          # execution engine: DAG (topological), retry, onError, control flow
│   ├── store.ts           # RunStore interface + FileRunStore + MemoryRunStore
│   ├── workspace.ts       # WorkspaceManager: versioning, _metadata.json, YAML loading
│   ├── createVein.ts      # createVein() factory: Hono HTTP API + SSE run streaming + /chat + static serving; injectable registry/store/services
│   ├── server.ts          # thin wrapper over createVein() (getApp/startServer) — default filesystem-backed server
│   ├── auth.ts            # requireApiKey middleware + warnIfUnconfigured (VEIN_API_KEY shared secret)
│   ├── index.ts           # barrel export — createVein (primary entry), createRegistry, coreRegistry, all types
│   ├── steps/
│   │   ├── core/          # 8 built-in steps: http, log, if, loop, foreach, subflow, llm, wait (static import)
│   │   ├── lib/           # built-in domain integrations (github/fetch-pr, ...) — dynamic import
│   │   └── registry.ts    # auto-discovery: buildRegistry() core (static) + lib (dynamic) + workspace custom/ (dynamic); createRegistry() for in-code steps
│   ├── ai/                # AI workflow-builder backend (used by POST /chat)
│   │   ├── index.ts       # barrel export
│   │   ├── prompts.ts     # SYSTEM prompt + buildSystem(deps) (pre-seeds steps tree)
│   │   ├── tools.ts       # buildTools(deps): list_steps, search_steps, get_step,
│   │   │                  #                   create_workflow, run_workflow
│   │   ├── stepHelpers.ts # lsSteps / searchSteps / readStepSource (filesystem-style browser)
│   │   └── schemaHelpers.ts # Zod → FieldDesc[] (for get_step schema rendering)
│   └── *.test.ts          # 298 tests across 12 files
└── web/
    ├── package.json       # preact, system-canvas, vite
    ├── vite.config.ts     # preact preset, dev proxy to :3000 (/workflows, /steps, /chat, /health)
    ├── index.html
    └── src/
        ├── main.tsx       # entry: renders <App/>
        ├── app.tsx        # shell: sidebar, topbar, canvas, events panel, dialog/flyout orchestration
        ├── api.ts         # typed fetch wrapper for all API endpoints (+ SSE/chat stream parser)
        ├── flow-to-canvas.ts  # Flow → CanvasData; STEP_COLORS → categories; childRefForStep/stepWorkflow (container nav)
        ├── helpers.ts     # normalizeSteps, formatJson, etc.
        ├── icons.tsx      # inline SVG icons
        ├── storage.ts     # crash-safe localStorage wrapper (UI prefs, session state)
        ├── components/
        │   ├── AddStepDialog.tsx     # searchable Add Step picker (core / lib / custom)
        │   ├── ChatFlyout.tsx        # AI workflow-builder chat (streams from /chat)
        │   ├── ConfigField.tsx       # field renderer driven by Zod-derived FieldDesc
        │   ├── CreateDialog.tsx      # new-workflow dialog
        │   ├── EventsPanel.tsx       # bottom run-events panel
        │   ├── EventsResizer.tsx     # drag handle for events panel height
        │   ├── Markdown.tsx          # tiny markdown renderer for chat output
        │   ├── RunInputPopover.tsx   # input-payload editor when triggering a run
        │   ├── StepEditFlyout.tsx    # edit step (id / type / config / depends / options)
        │   ├── StepRunFlyout.tsx     # view a step's run I/O (leaf: input/output; container: aggregate summary)
        │   └── FlyoutResizer.tsx     # drag handle for flyout width
        └── styles/
            ├── base.css       # palette (CSS vars on :root), reset, type
            └── components.css # shell grid, sidebar, flyout, dialog, events, chat, badges
```

## Running

```bash
# Engine
cd vein
npm install
npm test                    # 298 tests, ~330ms
npm run dev                 # starts Hono server on :3000

# Web UI (dev mode with HMR)
cd vein/web
npm install
npm run dev                 # Vite on :5173, proxies API to :3000

# Web UI (production build, served by engine)
cd vein/web && npm run build  # outputs web/dist/
cd vein && npm run dev        # serves API + UI on :3000
```

## Environment

| Variable            | Default        | Description                          |
| ------------------- | -------------- | ------------------------------------ |
| `VEIN_WORKSPACE`    | `./workspace`  | Persistent volume for workflows/runs |
| `VEIN_PORT`         | `3000`         | HTTP server port                     |
| `VEIN_API_KEY`      | (unset)        | Deployment-scoped shared secret. See "Auth" below. |
| `VEIN_LLM_PROVIDER` | `anthropic`    | Default LLM provider for llm step    |
| `VEIN_LLM_MODEL`    | (per-provider) | Override model name                  |

## Auth

`VEIN_API_KEY` is a **deployment-scoped shared secret**. Set it on every
container in the compose (vein and any service that registers steps).

- **Unset (dev mode):** registration mutations (`POST /steps`,
  `DELETE /steps/:name`, `DELETE /steps?publisher=X`) are unauthenticated.
  Vein logs a one-time warning at boot so the lax posture is visible.
  `GET /steps` and workflow execution are always public.
- **Set (production):** the gated endpoints require
  `Authorization: Bearer <VEIN_API_KEY>`. Anything else returns `401`.

The same secret authenticates **both directions** within a deployment:

1. Mcp → vein: registers steps with `Authorization: Bearer $VEIN_API_KEY`.
2. Step files inside vein → mcp: read `process.env.VEIN_API_KEY` and send
   it on callbacks to `mcp/gitree/cmd` (mcp validates the same value).

This is sufficient when both services live in the same trust domain
(same compose, same private network). Per-publisher keys can be added
later without breaking this contract — they'd be additive env vars
(`VEIN_API_KEY_<NAMESPACE>`) checked in addition to the shared key.

## Key concepts

- **`createVein()` is the primary entry point** (`src/createVein.ts`).
  It builds a configured instance — Hono `app`, `workspace`, `store`,
  `services`, plus `run()`/`listen()`/`rebuildRegistry()` helpers — and
  mounts every route (`/workflows`, `/steps`, `/chat`, `/health`, static
  UI). Everything is injectable via `VeinOptions`: pass your own
  `registry` (disables filesystem step discovery + publishing), `store`
  (e.g. `MemoryRunStore`), `services` bag, or toggle `serveUi`/
  `enableChat`. `server.ts` is just a thin default wrapper
  (`getApp`/`startServer`) over `createVein()` with filesystem defaults.

- **`services` bag** is a consumer-defined capabilities object exposed to
  every step via `ctx.services` (typed by `defineStep`, untyped at
  runtime). Inject env-specific implementations (Neo4j vs in-memory,
  real vs fake LLM) without touching workflows or the registry. Threaded
  through `createVein({ services })` / `vein.run(wf, input, { services })`
  / `runWorkflow(flow, input, registry, { services })`.

- **`createRegistry(steps)`** builds a registry from in-code step defs
  layered on core + lib (no filesystem custom/ discovery) — the
  library-usage counterpart to `buildRegistry(workspacePath)`. User
  steps may shadow core/lib steps (with a warning); duplicates throw.

- **Workflows are YAML**. One format everywhere — no `.ts` workflows,
  no JSON workflows. The API accepts either a `steps` array (server
  serializes to YAML) or a raw `yaml` string. Files are stored as
  `<version>.yaml` in the workflow directory. `js-yaml` for parsing.

- **`params` = the experiment surface.** A workflow may declare a
  top-level `params:` block of tunable default knobs (prompts,
  thresholds, sample sizes). Step configs reference them via
  `{{ params.* }}`, parsed in `workspace.ts:loadFlowYaml` and seeded
  into runner scope as `{ ...flow.params, ...runOverride }` (see
  `executeFlow` in `runner.ts`). Deliberately distinct from `input`:
  `input` is the run *subject* (validated, no defaults), `params` is
  *how* it's processed (all defaults, sparsely overridden per run via
  `POST /run { params }` / `runWorkflow({ params })`). Override
  precedence: step Zod `.default()` < `params` default < per-run
  override. Subflows use their own `params`; the parent override does
  not propagate. This is what lets overnight experiments sweep 100
  prompt variants as 100 **runs** (logged in `run.json`) rather than
  100 workflow versions — promote a winner by editing the `params`
  default and publishing one new version.

- **DAG execution via `depends`**. Steps have an optional `depends`
  field (`string | string[]`). If set, the step waits for those
  dependencies before running. If omitted, the step implicitly
  depends on the previous step in the array (sequential by default).
  `depends: []` means no dependencies — the step runs immediately
  in parallel with others. The runner launches all steps via
  `Promise.all`, each waiting on its own deps internally.

- **No `parallel` step type**. Parallelism is expressed purely via
  `depends`. Two steps with the same `depends` run concurrently.
  A downstream step with `depends: ["a", "b"]` waits for both.

- **Control flow steps** (`if`, `loop`, `subflow`) are intercepted
  by the runner before registry lookup. Their config is NOT
  pre-resolved by `executeStep` — they manage their own template
  resolution (see `SELF_RESOLVING_STEPS` set in `runner.ts`). This
  is because `loop`'s `until` references `$current` which doesn't
  exist until iteration time.

- **Expression evaluator** (`expr.ts`) is a one-pass recursive
  descent parser. It does NOT short-circuit `&&`/`||` — both sides
  are always evaluated. No function calls in v1.

- **The global regex issue**: `TEMPLATE_RE` is a module-level
  `/g` regex. `resolveTemplate` uses a non-regex check
  (`indexOf("{{")`) for the single-expression fast path to avoid
  the greedy backtracking bug with multi-segment templates.

- **`_metadata.json`** in each workflow dir tracks versions and the
  active version. In `steps/custom/_metadata.json` it tracks
  per-step `{ active, versions, publisher? }` keyed by full
  slash-name (e.g. `"gitree/save-feature"`), where `versions` maps
  each version id (`v1`, `v2`, …) to `{ createdAt, description?, hash }`.
  The engine can run without these files but the UI needs them.

- **Custom steps are versioned** (like workflows). `publishStep`
  is keyed by content hash (`src/version.ts`) but labeled with
  sequential `vN` ids: identical content re-activates the existing
  version (no-op if already active), changed content publishes the
  next `vN`. The active version's source is materialized at
  `steps/custom/<name>.ts` (what the registry loads); every version
  is archived under `steps/_history/<name>/<vid>.ts` for rollback.
  Endpoints: `GET /steps/:type/versions`, `GET /steps/:type/version/:version`,
  `PUT /steps/:type/active`. Versioning is disabled when the registry
  is injected at construction time.

- **Step registration is filesystem-based.** External services
  register steps by `POST /steps { name, code, description?, publisher? }`.
  Names may be nested (`"gitree/save-feature"` writes
  `steps/custom/gitree/save-feature.ts`) and helper files use a
  leading `_` (`"gitree/_shared"`) — these are saved and importable
  by sibling steps but skipped by registry discovery. A service
  cleans up on shutdown via `DELETE /steps?publisher=X`, which
  removes every step it owns and prunes empty namespace dirs.
  Namespaces are pure naming convention (just slashes in the name) —
  no ownership enforcement, last writer wins.

- **`buildRegistry()` returns `{ registry, sources }`.** The
  `sources` map records where each step was loaded from
  (`"core" | "lib" | "custom"`). Always use this map instead of
  guessing from the name — a custom step like `gitree/save-feature`
  has a slash but is `custom`, not `lib`. The `/steps` endpoint
  uses this map.

- **Runs are stored per-workflow** under
  `workflows/<name>/runs/<runId>/`. Run IDs are millisecond
  timestamps (not UUIDs) for natural sort order and easy
  pagination. Listing runs for a workflow is a single `readdir`.

- **`RunStore.append/finalize`** take `(workflow, runId, ...)`
  — the workflow name is the first param. `MemoryRunStore` keys
  by `"workflow/runId"` internally; use `store.getEvents(wf, id)`
  and `store.getSummary(wf, id)` in tests.

- **Flyout has two modes**: when no run is selected, clicking a
  canvas node opens the step editor (edit id, type, config). When
  a run is selected, it opens run results (input, output, error,
  duration). No backdrop — flyout stays open when clicking between
  nodes for smooth transitions. Clicking is always **inspect**;
  entering a container's children is the **arrow's** job (below).

- **Container nodes & navigation** (`flow-to-canvas.ts` +
  `app.tsx`). `subflow`/`foreach`/`loop` steps that target a child
  workflow (`stepWorkflow()` resolves a name — a subflow's
  `config.workflow`, or a foreach/loop whose `body` is a subflow)
  get a navigable **ref arrow** (`childRefForStep` sets
  `node.ref = "wf:<name>"`; `refCorner: "topRight"` so it clears the
  left-aligned name label). The arrow means different things by mode
  (we use system-canvas's `externalNavigation` so a ref click only
  fires `onNavigate`, never the lib's internal drill):
  - **Edit view** → **go-to-definition**: `handleNavigate` switches
    the selected workflow (the target is a standalone, separately
    editable workflow — not an inline sub-canvas).
  - **Run view** → **drill into this run's nested execution**: a
    read-only sub-canvas built from the child's steps + the run's
    events **re-keyed** to the nested path prefix. Subflows execute
    inline under one `runId` with hierarchical event paths
    (`wf/subflowId/childId`, `wf/foreachId#i/...`), so
    `viewEvents` strips the `<prefix>/` and re-prefixes with the
    child workflow name — the existing path-based status overlay +
    flyout lookups then work unchanged at any depth. A `runDrill`
    frame stack + breadcrumb bar handles back-navigation. foreach/
    loop frames carry an **iteration count + selected `iter`**
    (`framePrefix` appends `#<iter>`), surfaced as a dropdown in the
    drill bar (`countIterations` derives N from the events).
  - **Container I/O is the "summary"**: a leaf's flyout shows its
    `input → output`; a container's shows its aggregate (subflow:
    child input → child result; foreach: items array → results
    array). The runner emits these as the container's `step.start`
    input (`subflow` → `config.input`, `foreach` → resolved
    `items`); `loop` has no natural input.

- **`canvas` and `flyoutEvents` are derived** (`useMemo`) from a
  **view context** — root (`selectedWf` + `localSteps` + `events`)
  or the deepest `runDrill` child (its workflow + steps + re-keyed
  events). There is no imperative `rebuildCanvas`; setting
  `localSteps`/`events`/`runDrill` recomputes the canvas. A
  `running` flag drives the pending-status overlay during a live
  streamed run (events are still empty at run start).

- **AI workflow builder** (`src/ai/` + `POST /chat`). The chat
  flyout streams a `ToolLoopAgent` (Vercel AI SDK + Anthropic) that
  can browse step types (`list_steps`, `search_steps`, `get_step`),
  publish workflows (`create_workflow`), and test them
  (`run_workflow`). The system prompt is built per-request by
  `buildSystem(deps)` which pre-seeds the available steps tree so
  the model can skip the initial `list_steps` calls. Tool inputs
  and step finish events are logged server-side with a `[chat <id>]`
  prefix. The browser parses the AI SDK UI-message stream protocol
  in `web/src/api.ts:chat()` and dispatches text deltas, tool
  calls, and tool results back to `ChatFlyout`.

## Conventions

- **Vanilla CSS** with custom properties. Two files only:
  `base.css` (palette + reset) and `components.css` (all
  component classes). No JS styles, no CSS-in-JS, no Tailwind.
  Same pattern as `gateway/internal/adminapi/ui/`.

- **`system-canvas`** for flow visualization **and editing**. The
  canvas is `editable={true}` when not viewing a run. Each step
  type is a theme **category** (`step-http`, `step-log`, etc.)
  with a `header` slot showing the uppercase type and a run-status
  slot (checkmark = success, X = error, dot = running, clock =
  pending) — `topRight` for leaves, `bottomRight` for container
  nodes (whose `topRight` carries the ref arrow). Container
  categories also add a `body` text slot (the workflow name, small,
  left-aligned, ellipsized) and set `refCorner: "topRight"`. Nodes
  carry `customData: { stepId, stepIndex, status, stepType }`.
  Interactive features: drag-to-connect creates `depends` edges,
  the "+" FAB opens a searchable Add Step dialog (core/lib/custom),
  delete key removes nodes/edges. `panMode="trackpad"` for
  Figma-style two-finger-scroll panning.

- **`system-canvas` is a sibling library** at
  `/Users/evanfeenstra/code/sphinx2/system-canvas` (its own repo,
  published to npm in lockstep as `system-canvas` +
  `system-canvas-react`). vein consumes the **published** version
  (`web/package.json`). Features vein relies on: `node.ref` +
  `externalNavigation` (ref click fires `onNavigate` only, no
  internal drill), per-node/category `refCorner`, category `slots`.
  To land a lib change: edit the lib, `npm run build`, push to
  `main` (auto-release bumps the patch), then `npm install
  system-canvas@<v> system-canvas-react@<v>` in `web/` and rebuild.
  (For local iteration you can copy the built `dist/` into
  `web/node_modules/...`, but a fresh `npm install` overwrites it.)

- **Add Step dialog** opens from the canvas FAB button. Fetches
  all available step types from the `/steps` API, groups them
  by source (core / library / custom), and supports type-ahead
  search. Selecting a type adds the step with `depends: []` and
  opens the flyout for configuration.

- **Tests** use `node:test` + `assert/strict`. Each test file
  creates its own helper step definitions (echo, value, fail,
  counter, flakey). `MemoryRunStore` for unit tests,
  `FileRunStore` with tmp dirs for integration tests.

- **Step definitions** use `defineStep()` which returns the def
  as-is (identity function for type inference). `AnyStepDef` is
  the type-erased version used in the runtime registry.

## When adding a new core step

1. Create `src/steps/core/<name>.ts` with a `defineStep()` default export.
2. Import it in `src/steps/registry.ts` and add to `CORE_STEPS`.
3. If it's control flow, add to `SELF_RESOLVING_STEPS` in `runner.ts`
   and handle it in `dispatchStep`.
4. Add a color entry in `STEP_COLORS` in `web/src/flow-to-canvas.ts`
   (categories are auto-generated from this map via `buildCategories()`;
   the Add Step dialog discovers types via the `/steps` API).
5. Write tests in the appropriate test file.
6. Run `npm test` and `cd web && npx tsc --noEmit && npx vite build`.

## When adding an API endpoint

1. Add the route in `src/createVein.ts` (inside the `createVein()`
   factory, where all routes are mounted). Watch route ordering —
   specific paths like `/workflows/:name/flow` must come BEFORE
   catch-all params like `/workflows/:name/:version`.
2. Add the typed function in `web/src/api.ts`.
3. Wire it into `web/src/app.tsx`.
4. The Vite dev proxy in `web/vite.config.ts` only proxies known
   prefixes (`/workflows`, `/steps`, `/chat`, `/health`). Runs are
   under `/workflows/` so they're already proxied. SSE responses
   get `cache-control: no-cache` + `x-accel-buffering: no` injected
   by the proxy. Add new prefixes if needed.

## When modifying the web UI

- Edit `web/src/styles/components.css` for styling — never use JS
  style objects.
- The shell layout is a CSS grid:
  `grid-template-areas: "sidebar topbar" / "sidebar canvas" / "sidebar events"`.
- The flyout is `position: fixed` on the right, no backdrop (so
  canvas clicks pass through for node-to-node transitions).
- Rebuild with `cd web && npm run build` before testing against the
  engine server.
- vein is also embedded by **mcp** under `/lab` (it consumes vein as a
  copied `file:../vein` dep, UI bundled into `web/dist`). After a web
  change, `mcp`'s `yarn dev` runs `refresh-vein` (rebuild vein + web,
  reinstall into mcp) before starting on `:3355` — so changes only reach
  `/lab` after that, not on a bare vite rebuild.
