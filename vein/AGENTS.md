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
│   ├── core.ts            # flow(), step(), defineStep(), all types
│   ├── expr.ts            # {{ }} template evaluator (~460 LOC recursive descent)
│   ├── runner.ts          # execution engine: DAG (topological), retry, onError, control flow
│   ├── store.ts           # RunStore interface + FileRunStore + MemoryRunStore
│   ├── workspace.ts       # WorkspaceManager: versioning, _metadata.json, YAML loading
│   ├── server.ts          # Hono HTTP API + SSE run streaming + /chat + static file serving
│   ├── index.ts           # barrel export
│   ├── steps/
│   │   ├── core/          # 7 built-in steps: http, log, if, loop, subflow, llm, wait (static import)
│   │   ├── lib/           # built-in domain integrations (github/, ...) — dynamic import
│   │   └── registry.ts    # auto-discovery: core (static) + lib (dynamic) + workspace custom/ (dynamic)
│   ├── ai/                # AI workflow-builder backend (used by POST /chat)
│   │   ├── index.ts       # barrel export
│   │   ├── prompts.ts     # SYSTEM prompt + buildSystem(deps) (pre-seeds steps tree)
│   │   ├── tools.ts       # buildTools(deps): list_steps, search_steps, get_step,
│   │   │                  #                   create_workflow, run_workflow
│   │   ├── stepHelpers.ts # lsSteps / searchSteps / readStepSource (filesystem-style browser)
│   │   └── schemaHelpers.ts # Zod → FieldDesc[] (for get_step schema rendering)
│   └── *.test.ts          # 199 tests across 7 files
└── web/
    ├── package.json       # preact, system-canvas, vite
    ├── vite.config.ts     # preact preset, dev proxy to :3000 (/workflows, /steps, /chat, /health)
    ├── index.html
    └── src/
        ├── main.tsx       # entry: renders <App/>
        ├── app.tsx        # shell: sidebar, topbar, canvas, events panel, dialog/flyout orchestration
        ├── api.ts         # typed fetch wrapper for all API endpoints (+ SSE/chat stream parser)
        ├── flow-to-canvas.ts  # Flow → CanvasData converter; STEP_COLORS → canvas categories
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
        │   └── StepRunFlyout.tsx     # view a single step's input/output/error/duration in a run
        └── styles/
            ├── base.css       # palette (CSS vars on :root), reset, type
            └── components.css # shell grid, sidebar, flyout, dialog, events, chat, badges
```

## Running

```bash
# Engine
cd vein
npm install
npm test                    # 199 tests, ~200ms
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
  with a `header` slot showing the uppercase type and a `topRight`
  custom slot for run status indicators (checkmark = success,
  X = error, dot = running, clock = pending). Nodes carry
  `customData: { stepId, stepIndex, status }`. Interactive
  features: drag-to-connect creates `depends` edges, the "+"
  FAB opens a searchable Add Step dialog (core/lib/custom),
  delete key removes nodes/edges. `panMode="trackpad"` for
  Figma-style two-finger-scroll panning.

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

1. Add the route in `src/server.ts`. Watch route ordering — specific
   paths like `/workflows/:name/flow` must come BEFORE catch-all
   params like `/workflows/:name/:version`.
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
