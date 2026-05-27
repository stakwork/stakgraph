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

## Layout

```
vein/
├── SPEC.md                # full design spec — read this first
├── package.json           # engine deps (hono, zod, ai sdk)
├── tsconfig.json          # strict, Node16 module, types: ["node"]
├── src/
│   ├── core.ts            # flow(), step(), defineStep(), all types
│   ├── expr.ts            # {{ }} template evaluator (~300 LOC recursive descent)
│   ├── runner.ts          # execution engine: sequential, retry, onError, control flow
│   ├── store.ts           # RunStore interface + FileRunStore + MemoryRunStore
│   ├── workspace.ts       # WorkspaceManager: versioning, _metadata.json, JSON + .ts loading
│   ├── server.ts          # Hono HTTP API + static file serving (entry point)
│   ├── index.ts           # barrel export
│   ├── steps/
│   │   ├── core/          # 8 built-in steps: http, log, if, loop, parallel, subflow, llm, wait
│   │   └── registry.ts    # auto-discovery: merges core + workspace lib/ + custom/
│   ├── *.test.ts          # 188 tests across 6 files
└── web/
    ├── package.json       # preact, system-canvas, vite
    ├── vite.config.ts     # preact preset, dev proxy to :3000
    ├── index.html
    └── src/
        ├── main.tsx       # entry: renders <App/>
        ├── app.tsx        # main app: sidebar, canvas, flyout, events panel
        ├── api.ts         # typed fetch wrapper for all API endpoints
        ├── flow-to-canvas.ts  # Flow → CanvasData converter for system-canvas
        └── styles/
            ├── base.css       # palette (CSS vars on :root), reset, type
            └── components.css # shell grid, sidebar, flyout, dialog, events, badges
```

## Running

```bash
# Engine
cd vein
npm install
npm test                    # 196 tests, ~200ms
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
| `VEIN_LLM_PROVIDER` | `anthropic`    | Default LLM provider for llm step    |
| `VEIN_LLM_MODEL`    | (per-provider) | Override model name                  |

## Key concepts

- **Workflows are YAML**. One format everywhere — no `.ts` workflows,
  no JSON workflows. The API accepts either a `steps` array (server
  serializes to YAML) or a raw `yaml` string. Files are stored as
  `<version>.yaml` in the workflow directory. `js-yaml` for parsing.

- **Control flow steps** (`if`, `loop`, `parallel`, `subflow`) are
  intercepted by the runner before registry lookup. Their config is
  NOT pre-resolved by `executeStep` — they manage their own template
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
  active version. In each step dir it tracks descriptions. The
  engine can run without these files but the UI needs them.

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
  nodes for smooth transitions.

## Conventions

- **Vanilla CSS** with custom properties. Two files only:
  `base.css` (palette + reset) and `components.css` (all
  component classes). No JS styles, no CSS-in-JS, no Tailwind.
  Same pattern as `gateway/internal/adminapi/ui/`.

- **`system-canvas`** for flow visualization. Each step type is
  a theme **category** (`step-http`, `step-log`, etc.) with a
  `header` slot showing the uppercase type and a `topRight`
  custom slot for run status indicators (checkmark = success,
  X = error, dot = running, clock = pending). Node `text` is
  the step name (left-aligned by the category layout). Nodes
  carry `customData: { stepId, stepIndex, status }` so click
  handlers can map back to steps.

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
4. Add to `STEP_TYPES` array in `web/src/app.tsx` (for the type dropdown).
5. Add a color entry in `STEP_COLORS` in `web/src/flow-to-canvas.ts`
   (categories are auto-generated from this map via `buildCategories()`).
6. Write tests in the appropriate test file.
7. Run `npm test` and `cd web && npx tsc --noEmit && npx vite build`.

## When adding an API endpoint

1. Add the route in `src/server.ts`. Watch route ordering — specific
   paths like `/workflows/:name/flow` must come BEFORE catch-all
   params like `/workflows/:name/:version`.
2. Add the typed function in `web/src/api.ts`.
3. Wire it into `web/src/app.tsx`.
4. The Vite dev proxy in `web/vite.config.ts` only proxies known
   prefixes (`/workflows`, `/steps`, `/health`). Runs are under
   `/workflows/` so they're already proxied. Add new prefixes if needed.

## When modifying the web UI

- Edit `web/src/styles/components.css` for styling — never use JS
  style objects.
- The shell layout is a CSS grid:
  `grid-template-areas: "sidebar topbar" / "sidebar canvas" / "sidebar events"`.
- The flyout is `position: fixed` on the right, no backdrop (so
  canvas clicks pass through for node-to-node transitions).
- Rebuild with `cd web && npm run build` before testing against the
  engine server.
