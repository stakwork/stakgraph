# Mini Workflow Engine — Spec

A minimal, LLM-friendly workflow engine. Workflows are YAML. Steps are TypeScript files. Runs are persisted to disk as JSONL. Ships with an HTTP API and a web UI.

---

## 1. Goals

- Workflows are YAML files — easy for LLMs and humans to read, edit, and compose.
- Trivially renderable in a visual UI (boxes + nested boxes). Ships with a built-in web UI using system-canvas.
- Steps are TypeScript code (`defineStep()`) — the leaf nodes that actually do things.
- Huge workflows scale via child workflows (`subflow`) — one workflow per file.
- Runs are observable: every step start/end/error is logged to JSONL on disk.
- HTTP API for all operations: publish, run, inspect, manage versions.

---

## 2. File Layout

The engine has two layers: the **engine** (immutable, baked into a Docker image) and the **workspace** (mutable, user content on a persistent volume).

### 2.1 Engine (immutable)

```
src/
  core.ts               # flow(), step(), defineStep(), types
  runner.ts             # execution engine
  expr.ts               # template expression evaluator
  store.ts              # RunStore interface + filesystem implementation
  workspace.ts          # workspace manager (discovery, versioning)
  server.ts             # Hono HTTP API + static file serving
  steps/
    core/               # built-ins, never modified (statically imported)
      http.ts
      if.ts
      loop.ts
      subflow.ts
      log.ts
      llm.ts
      wait.ts
    lib/                # built-in domain integrations (dynamically imported)
      github/
        fetch-pr.ts
      neo4j/
        query.ts
      slack/
        send-message.ts
    registry.ts         # auto-discovers and registers all step types
web/                    # Preact + system-canvas UI (built to web/dist/)
  src/
    app.tsx             # main app: sidebar, canvas, event log
    api.ts              # typed API client
    flow-to-canvas.ts   # Flow → CanvasData converter for visualization
    styles/
      base.css          # palette, reset, custom properties
      components.css    # shell layout, sidebar, badges, dialog, events
```

### 2.2 Workspace (mutable, persistent volume)

Set via `VEIN_WORKSPACE` env var (default: `./workspace`). This is a mounted volume in Docker.

```
$VEIN_WORKSPACE/
  workflows/
    deploy/
      _metadata.json    # { "active": "v2", "versions": { ... } }
      v1.yaml           # workflow definition
      v2.yaml
      runs/             # runs scoped to this workflow
        1748374800123/  # timestamp-based run ID (ms)
          events.jsonl
          run.json
        1748374856789/
          events.jsonl
          run.json
    poll-and-notify/
      _metadata.json
      v1.yaml
      runs/
  steps/
    custom/             # user/LLM-created steps, generated at runtime
      _metadata.json
      my-scorer.ts
      parse-diff.ts
```

- **`steps/core/`** (engine): built-in control flow and primitives. Frozen — never edited by users or LLMs. Statically imported, so they're always loaded.
- **`steps/lib/`** (engine): built-in domain steps organized by integration/service (e.g. `github/`, `neo4j/`, `slack/`). Shipped with the engine but **dynamically imported** — heavy dependencies (e.g. `@octokit/rest`) are only pulled in when a workflow actually uses the step.
- **`steps/custom/`** (workspace): user-created or LLM-generated steps. An LLM can create a new step by writing a `.ts` file here; it's immediately usable in workflows. Dynamically imported.
- Each workflow lives in its own directory under `workflows/`, with versioned `.yaml` files and a `_metadata.json`.
- Child workflows are referenced by name via the `subflow` step type.

---

## 3. Authoring API

### 3.1 `flow(name, input, steps)`

```ts
import { z } from "zod";
import { flow, step } from "../core";

export default flow("deploy", {
  input: z.object({ service: z.string() }),
  steps: [
    step("kick", "http", { url: "/deploy", method: "POST" }),
    step("done", "log", { message: "deployed {{ input.service }}" }),
  ],
});
```

- `name`: unique workflow name (string).
- `input`: Zod schema for the workflow's input. Required (use `z.object({})` if none).
- `steps`: ordered array of steps. By default, steps run sequentially (each implicitly depends on the previous). Steps with explicit `depends` form a DAG — steps sharing the same dependency run concurrently. Workflow's return value is the **last step's output** (by array order).

### 3.2 `step(id, type, config, options?)`

```ts
step("check", "http", { url: "{{ input.url }}" });
step("fast", "http", { url: "/fast" }, { depends: [] });           // no deps = run immediately
step("merge", "log", { message: "done" }, { depends: ["a", "b"] }); // wait for a & b
```

- `id`: unique within the enclosing `flow` or branch. String, `[a-zA-Z_][a-zA-Z0-9_]*`.
- `type`: a key in the step registry. Type-checked.
- `config`: shape determined by the step's input schema. Type-checked.
- `depends`: optional. `string | string[]`. If omitted, the step implicitly depends on the previous step in the array (sequential). `depends: []` means no dependencies — runs immediately in parallel. `depends: "stepId"` or `depends: ["a", "b"]` waits for those steps.

### 3.3 `defineStep(...)` (authoring a step type)

```ts
// steps/http.ts
import { z } from "zod";
import { defineStep } from "../core";

export default defineStep({
  type: "http",
  input: z.object({
    url: z.string(),
    method: z.enum(["GET", "POST", "PUT", "DELETE"]).default("GET"),
    body: z.any().optional(),
    headers: z.record(z.string()).optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const res = await fetch(cfg.url, {
      method: cfg.method,
      body: cfg.body ? JSON.stringify(cfg.body) : undefined,
      headers: cfg.headers,
    });
    return { status: res.status, body: await res.json() };
  },
});
```

Step registry (`steps/index.ts`) imports each step and exposes a typed union; `step()`'s type/config args are inferred from this registry.

---

## 4. Step Type Resolution

Step type names are derived from a file's path relative to its base directory:

- `"http"` → `src/steps/core/http.ts` (engine, static import)
- `"github/fetch-pr"` → `src/steps/lib/github/fetch-pr.ts` (engine, dynamic import)
- `"neo4j/query"` → `src/steps/lib/neo4j/query.ts` (engine, dynamic import)
- `"my-scorer"` → `<workspace>/steps/custom/my-scorer.ts` (workspace, dynamic import)
- `"utils/parse-diff"` → `<workspace>/steps/custom/utils/parse-diff.ts` (workspace, dynamic import)

Resolution order: `core/` → `lib/` → `custom/`. A name in a higher tier cannot shadow a lower one — the registry logs a warning and skips the collision. Names with a `/` are namespaced (typically lib steps); flat names without a `/` are typically core or custom.

The registry auto-discovers `lib/` and `custom/` at startup using dynamic `import()`, so each step's dependencies are only resolved when it is actually loaded. Core steps are statically imported and always present. Adding a file to any directory registers a new step type — no manual wiring needed.

### 4.1 Core Steps

All core steps live in `steps/core/`. These are built-in control flow primitives and basic utilities.

#### 4.1.1 `http`

Config: `{ url, method?, body?, headers? }` (templates allowed in all string fields).
Output: `{ status, body }`.

#### 4.1.2 `log`

Config: `{ message: string, level?: "info"|"warn"|"error" }`.
Output: the resolved message string.

#### 4.1.3 `if`

Config: `{ cond: string /* {{ expr }} */ }`.
Output: the boolean result of evaluating `cond`.

`if` is a **gate**, not a container. Downstream steps branch by declaring
`depends: <if-id>` and `when: true` or `when: false`. A step's `when` must
match the boolean output of the gate it depends on, or the step is skipped.
Skipped steps cascade: a step whose dependencies are **all** skipped is itself
skipped. A step with mixed real and skipped deps (a fan-in) still runs.

```yaml
- id: check
  type: if
  config:
    cond: "{{ input.fast }}"
- id: quick
  type: log
  config: { message: "fast path" }
  depends: check
  when: true
- id: slow
  type: log
  config: { message: "slow path" }
  depends: check
  when: false
- id: done                       # fan-in: runs after whichever branch ran
  type: log
  config: { message: "branch complete" }
  depends: [quick, slow]
```

#### 4.1.4 `loop`

Config:

```ts
{
  until: string,        // {{ expr }} evaluated against the body's last output
  maxIterations: number,
  delayMs?: number,
  body: Step,
}
```

- Iteration N runs `body`. Inside `body`'s config, `{{ $current }}` is the **previous iteration's** output (undefined on iteration 0).
- After each iteration, `until` is evaluated; if true, loop exits.
- If `maxIterations` is reached without `until` becoming true, the loop **fails** (errors the run).
- Output: the **last iteration's** output.

#### 4.1.5 `subflow`

Config: `{ workflow: string, version?: string, input: object }`.

- Looks up the workflow by `workflow` (name) in the workspace and runs it as a child.
- If `version` is omitted, the workflow's active version is used.
- Input must satisfy the child workflow's `input` schema (validated at run).
- Output: that workflow's output (its last step's output).
- Child runs share the parent `runId` and write events into the same `events.jsonl` with a `path` prefix (see §7).
- Subflows reference **published** workflows by name — they are not defined inline.

#### 4.1.6 `llm`

Config:

```ts
{
  prompt: string,       // template string
  schema?: z.ZodSchema, // optional structured output schema
  provider?: string,    // e.g. "anthropic", "openai" — defaults to env
  model?: string,       // override model
}
```

- Calls an LLM with the resolved prompt. If `schema` is provided, uses structured output (e.g. `generateObject`).
- Output: `{ text: string }` (no schema) or the parsed object (with schema).

### 4.2 Lib Steps

Lib steps live in `src/steps/lib/<namespace>/` inside the engine itself. They are reusable domain-specific integrations (GitHub, Neo4j, Slack, …) that ship with vein but are kept out of the static dependency graph. Each file is a `defineStep(...)` export, same as core steps.

Lib steps are loaded with **dynamic `import()`** at registry build time. The practical consequence: a workflow that doesn't use the `github/*` steps never resolves `@octokit/rest`. This keeps the cold-start surface small even as the lib catalog grows to many integrations.

Step type name = `"<namespace>/<filename>"` (e.g. `"github/fetch-pr"`).

Example:

```ts
// src/steps/lib/github/fetch-pr.ts
import { z } from "zod";
import { Octokit } from "@octokit/rest";
import { defineStep } from "../../../core.js";

export default defineStep({
  type: "github/fetch-pr",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    pull_number: z.number().int().positive(),
    token: z.string().optional(),
  }),
  output: z.object({ markdown: z.string(), pr: z.any() }),
  async run(cfg) {
    const octokit = new Octokit({ auth: cfg.token ?? process.env["GITHUB_TOKEN"] });
    // …fetch + format
  },
});
```

Usage in a workflow:

```yaml
- id: pr
  type: github/fetch-pr
  config:
    owner: "{{ input.owner }}"
    repo: "{{ input.repo }}"
    pull_number: "{{ input.prNumber }}"
```

Because lib steps are part of the engine source tree, adding a new one is a code change — there is no `POST /steps` route for lib. Users contribute lib steps via PRs.

### 4.3 Custom Steps

Custom steps live in `<workspace>/steps/custom/`. They are created by users or LLMs at runtime — the `POST /steps` endpoint writes a `.ts` file there and the registry picks it up on the next rebuild. Same dynamic-import loading as lib.

Same `defineStep(...)` format as core and lib steps. Step type name = the filename without extension (e.g. `"my-scorer"` for `steps/custom/my-scorer.ts`). Subdirectories within `custom/` are allowed and follow the same `"dir/name"` naming convention.

---

## 5. Expression Language (`{{ ... }}`)

A small, deterministic evaluator. **Not** `eval`. Implemented from scratch (~100 LOC) or via `expr-eval`.

### 5.1 Scope

Available bindings inside a step config:

- `input` — the workflow's input object.
- `<stepId>` — output of any previously completed step at the same level.
- `$current` — only inside a `loop` body; previous iteration's output (undefined on iteration 0).

### 5.2 Syntax

- A config field's value is either a **literal** (non-string, or a string with no `{{`) or a **template string** containing one or more `{{ expr }}` segments.
- If the entire string is a single `{{ expr }}`, the result keeps the expression's type (object, number, etc.).
- Otherwise, segments are stringified and concatenated.

### 5.3 Supported expression grammar

- Property access: `foo.bar.baz`, `foo["bar"]`, `arr[0]`.
- Literals: numbers, strings (single or double quotes), `true`, `false`, `null`.
- Operators: `=== !== == != < <= > >= && || !  + - * / %`.
- Ternary: `a ? b : c`.
- Function calls: **none** in v1.

Examples:

```
{{ input.url }}
{{ poll.body.status === "complete" }}
{{ fan.left.count + fan.right.count }}
{{ items[0].name }}
```

### 5.4 Resolution

Before running a step, the runner walks its `config` recursively and resolves every string. Unresolved references (`foo.bar` where `foo` is undefined) throw a `TemplateError` and fail the step.

---

## 6. Error Handling

Every step accepts two optional fields at the **step level** (not inside `config`):

```ts
step(
  "check",
  "http",
  { url: "/health" },
  {
    retry: { max: 3, delayMs: 1000 },
    onError: step("notify-fail", "http", { url: "/alert", method: "POST" }),
  },
);
```

- `retry`: if `run` throws, retry up to `max` times with `delayMs` between attempts. Default: no retry.
- `onError`: if all retries fail (or no retry), run this single fallback step. Its output becomes this step's output. The original error is available as `{{ $error }}` inside the fallback's config only.
- If neither is set and the step throws, the run **fails** (the `Promise.all` in the DAG executor rejects and the error propagates).

---

## 7. Persistence & Logging

### 7.1 Run identity

A run is created with a millisecond-timestamp `runId` (e.g. `1748374800123`). Timestamp IDs give natural sort order and easy pagination (runs after X). Runs are stored under the workflow directory:

```
workflows/<name>/runs/<runId>/events.jsonl
workflows/<name>/runs/<runId>/run.json     (written at end)
```

This scoping means listing runs for a workflow is a single `readdir` — no scanning across all workflows.

### 7.2 Event format (JSONL, append-only)

One JSON object per line. Common fields:

```ts
{
  ts: "2026-05-26T12:34:56.789Z",
  runId: "uuid",
  path: "deploy/fan.left/check",   // slash-separated step path including subflows/branches
  type: "step.start" | "step.end" | "step.error" | "step.retry" | "run.start" | "run.end" | "run.error",
  stepType?: "http",
  input?: any,        // resolved config
  output?: any,
  error?: { message: string, stack?: string },
  durationMs?: number,
  iteration?: number, // for loop bodies
}
```

- `path` uniquely identifies a step instance across subflows, parallel branches, and loop iterations. Loop iterations are appended as `#0`, `#1`, … (e.g. `deploy/wait/check#3`).

### 7.3 `run.json` (final summary)

```ts
{
  runId: "uuid",
  workflow: "deploy",
  startedAt, finishedAt, durationMs,
  status: "success" | "error",
  input: any,
  output?: any,
  error?: { message, stack },
}
```

### 7.4 Storage interface

Default writer is filesystem. The runner depends on an interface:

```ts
interface RunStore {
  append(runId: string, event: object): Promise<void>;
  finalize(runId: string, summary: object): Promise<void>;
}
```

Swap in S3, Postgres, etc. without changing the runner.

---

## 8. Type Safety

The registry is the single source of truth. It is built at server startup by `buildRegistry(workspacePath?)` in `src/steps/registry.ts`:

- **Core** steps are statically imported (always present, always in the bundle).
- **Lib** steps are discovered by walking `src/steps/lib/` and `await import()`-ing each `.ts` file. Their step name is derived from the path (e.g. `lib/github/fetch-pr.ts` → `"github/fetch-pr"`).
- **Custom** steps are discovered the same way under `<workspace>/steps/custom/`.

Sketch:

```ts
// src/steps/registry.ts
import http from "./core/http.js";
// …other core imports

const CORE_STEPS = { http, if: ifStep, loop, subflow, log, llm, wait };
const LIB_DIR = join(dirname(fileURLToPath(import.meta.url)), "lib");

export async function buildRegistry(workspacePath?: string): Promise<StepRegistry> {
  const registry: StepRegistry = { ...CORE_STEPS };
  await loadStepsFrom(LIB_DIR, registry, "lib");
  if (workspacePath) {
    await loadStepsFrom(join(workspacePath, "steps", "custom"), registry, "custom");
  }
  return registry;
}
```

`loadStepsFrom` recursively walks the directory, derives the step name from the relative path, and dynamically imports the file. A collision with an already-registered name is skipped with a warning, so higher tiers (custom) cannot shadow lower ones (lib, core).

`step()` is generic over `keyof Registry`, so:

```ts
step("a", "http", { url: "x" }); // OK
step("a", "http", { uri: "x" }); // TS error
step("a", "htttp", { url: "x" }); // TS error
step("a", "github/fetch-prs", { owner: "x" }); // OK — namespaced type
step("a", "github/fetch-prs", { user: "x" }); // TS error
```

Cross-step template references are **not** statically checked (they are strings). They are validated at **load time** by parsing each template and resolving against the known step ids — this happens in `flow()` before any execution, and throws on unknown references.

---

## 9. Runner Contract

```ts
async function runWorkflow(
  flow: Flow,
  input: unknown,
  opts?: { runId?: string; store?: RunStore },
): Promise<{
  runId: string;
  status: "success" | "error";
  output?: any;
  error?: any;
}>;
```

Behavior:

1. Generate `runId` if not provided.
2. Validate `input` against `flow.input`. Validation failure → `run.error`, no steps run.
3. Emit `run.start`.
4. Build dependency graph from `depends` fields (implicit sequential if omitted).
5. Launch all steps via `Promise.all`. Each step waits for its own dependencies internally before executing:
   - Resolve templates in `config`.
   - Emit `step.start`.
   - Run the step (with retry/onError as configured).
   - Emit `step.end` or `step.error`.
   - Store output keyed by step id for later templates at this scope.
6. Workflow output = last step's output (by array order).
7. Emit `run.end` (or `run.error`).
8. Call `store.finalize(...)`.

Scope rules:

- Each `flow` (top-level or subflow) has its own step-output scope.
- A step inside a subflow **cannot** reference steps in the parent.
- Inputs to subflows must pass through their `input` config explicitly.

---

## 10. Workspace & Deployment

### 10.1 Workspace directory

The engine discovers user content from a single workspace root, configured via:

```
VEIN_WORKSPACE=/data/vein
```

Default: `./workspace` relative to the engine's working directory. In Docker, this is a mounted persistent volume.

The engine's built-in `steps/core/` and `steps/lib/` both live inside the engine image. At startup, the registry merges them with custom steps loaded from `<workspace>/steps/custom/`. Lib steps are bundled with the engine but only dynamically imported on demand, so their dependencies don't bloat cold start.

### 10.2 Docker model

```dockerfile
# Engine image — immutable
COPY src/ /app/src/
# Workspace — mounted at runtime
VOLUME /data/vein
ENV VEIN_WORKSPACE=/data/vein
```

Users never modify the engine image. They publish workflows and steps by writing files to the workspace volume.

---

## 11. Versioning & `_metadata.json`

### 11.1 Workflow versioning

Each workflow lives in its own directory. Versions are separate `.yaml` files. A `_metadata.json` tracks which version is active.

```
workflows/
  deploy/
    _metadata.json
    v1.yaml
    v2.yaml
  poll-and-notify/
    _metadata.json
    v1.yaml
```

**`_metadata.json`** for workflows:

```json
{
  "active": "v2",
  "versions": {
    "v1": { "createdAt": "2026-05-20T10:00:00Z", "description": "initial deploy flow" },
    "v2": { "createdAt": "2026-05-27T14:30:00Z", "description": "added retry on kick" },
    "v3": { "createdAt": "2026-05-27T16:00:00Z", "description": "draft — not yet active" }
  }
}
```

- **`active`**: the version key the runner uses when you say "run deploy." Must match a key in `versions`.
- **`versions`**: metadata about each version. The engine doesn't strictly need this to run — it can resolve `active` to `<active>.ts` — but it enables UI listing, audit, and rollback.
- **Publishing a new version**: write the `.yaml` file, add an entry to `versions`, set `active` to the new key. The API handles this automatically.
- **Rollback**: change `active` back to a previous version key (`PUT /workflows/:name/active`).

When the runner resolves a workflow name:
1. Look up `workflows/<name>/_metadata.json`.
2. Read `active` field → load `workflows/<name>/<active>.yaml`.

#### YAML workflow format

```yaml
name: deploy
steps:
  - id: kick
    type: http
    config:
      url: "/deploy/{{ input.service }}"
      method: POST
  - id: done
    type: log
    config:
      message: "deployed {{ input.service }}"
```

Steps can include an optional `depends` field for DAG execution:

```yaml
name: enrich
steps:
  - id: profile
    type: http
    config:
      url: "/profile/{{ input.id }}"
  - id: orders
    type: http
    config:
      url: "/orders/{{ input.id }}"
    depends: []                          # no deps = runs in parallel with profile
  - id: save
    type: log
    config:
      message: "done"
    depends: [profile, orders]           # waits for both
```

YAML workflows accept any input (equivalent to `z.any()`). The engine parses the YAML at load time and constructs a `Flow` object.

### 11.2 Step metadata

Step directories use `_metadata.json` for discoverability and documentation, not versioning (steps are not versioned in v1 — they are single files).

**`_metadata.json`** for step directories:

```json
{
  "steps": {
    "fetch-prs": { "createdAt": "2026-05-20T10:00:00Z", "description": "List PRs from GitHub" },
    "fetch-commit": { "createdAt": "2026-05-25T12:00:00Z", "description": "Fetch a single commit" }
  }
}
```

The engine can function without these files — it discovers steps by scanning `.ts` files. The metadata is for UI rendering, LLM context, and documentation.

### 11.3 WorkspaceManager API

The engine exposes programmatic helpers used by the HTTP server:

```ts
interface WorkspaceManager {
  // Workflows
  listWorkflows(): Promise<WorkflowInfo[]>;
  getWorkflow(name: string): Promise<Flow>;
  getWorkflowSource(name: string, version: string): Promise<string>; // raw YAML
  publishWorkflow(name: string, version: string, content: { steps: any[] } | string, description?: string): Promise<void>;
  setActiveVersion(name: string, version: string): Promise<void>;

  // Steps
  listSteps(): Promise<StepInfo[]>;
  publishStep(namespace: string, name: string, code: string, description?: string): Promise<void>;
}
```

---

## 12. HTTP API

The engine ships with an HTTP server (Hono) that exposes all operations. Set `VEIN_PORT` (default: `3000`).

### 12.1 Workflows

| Method | Path                           | Description                                                      |
| ------ | ------------------------------ | ---------------------------------------------------------------- |
| GET    | `/workflows`                   | List all workflows with metadata                                 |
| GET    | `/workflows/:name`             | Get workflow metadata (versions, active)                         |
| GET    | `/workflows/:name/flow`        | Get parsed flow structure (JSON) for the active version          |
| GET    | `/workflows/:name/:version`    | Get workflow YAML source for a specific version                  |
| POST   | `/workflows/:name`             | Publish new version: `{ version, steps }` or `{ version, yaml }` |
| PUT    | `/workflows/:name/active`      | Set active version: `{ version }`                                |
| POST   | `/workflows/:name/run`         | Run active version: `{ input?, runId? }`                         |
| POST   | `/workflows/:name/:version/run`| Run specific version: `{ input?, runId? }`                       |

### 12.2 Steps

| Method | Path     | Description                                                       |
| ------ | -------- | ----------------------------------------------------------------- |
| GET    | `/steps` | List all steps (core + lib + workspace custom)                    |
| POST   | `/steps` | Publish step: `{ namespace, name, code, description? }`           |

### 12.3 Runs & Logs

Runs are scoped to workflows. Run IDs are millisecond timestamps.

| Method | Path                                      | Description                                    |
| ------ | ----------------------------------------- | ---------------------------------------------- |
| GET    | `/workflows/:name/runs`                   | List runs for a workflow (newest first)         |
| GET    | `/workflows/:name/runs/:runId`            | Get run summary (run.json)                      |
| GET    | `/workflows/:name/runs/:runId/events`     | Get all events as JSON array                    |

### 12.4 Health

| Method | Path      | Description                                    |
| ------ | --------- | ---------------------------------------------- |
| GET    | `/health` | Returns workspace path and registered step count |

---

## 13. Web UI

The engine serves a built-in web UI at the root path (`/`). Stack: Preact + system-canvas-react. Vanilla CSS, no Tailwind. ~75 KB gzipped total.

### 13.1 Features

- **Workflow list** — sidebar showing all workflows with active version badges.
- **Create workflow** — dialog with YAML editor and pre-filled example.
- **Flow graph** — system-canvas viewport rendering the workflow's steps as a DAG. Step types are color-coded (green=http, blue=log, yellow=if, purple=loop, cyan=subflow). Edges from `depends`. Topological layer layout (same-depth steps side by side).
- **Visual editing** — canvas is editable when not viewing a run. Drag-to-connect creates `depends` edges. "+" FAB opens searchable Add Step dialog (core/lib/custom). Delete key removes nodes/edges. Click node to edit in flyout.
- **Run workflow** — button to execute the active version. Accepts input JSON.
- **Run history** — sidebar list of runs with status badges (green/red/yellow).
- **Event log** — bottom panel showing the JSONL event stream for the selected run, color-coded by event type.
- **Run status overlay** — canvas nodes show green (success), red (error), or yellow (running) when viewing a run.

### 13.2 Development

```bash
# Dev mode (vite + API proxy)
cd vein && npm run dev        # API on :3000
cd vein/web && npm run dev    # UI on :5173 (proxies API to :3000)

# Production (single server)
cd vein/web && npm run build  # build to web/dist/
cd vein && npm run dev        # serves API + UI on :3000
```

### 13.3 Flow visualization & editing

The `flow-to-canvas.ts` module converts a `Flow` object into a `CanvasData` for system-canvas. Layout uses topological layers — steps at the same dependency depth are placed side by side.

- **Edges** come from `depends` (or implicit sequential)
- **`if`** → a normal node like any other. Edges from an `if` gate to steps with a `when` field are labeled `true` or `false`.
- **`loop`** → enlarged node showing body step name

The canvas is `editable={true}` when not viewing a run:

- **Drag-to-connect** between nodes creates `depends` edges
- **"+" FAB** opens a searchable Add Step dialog (core/lib/custom types)
- **Delete key** removes selected nodes/edges
- **Click node** opens the step edit flyout (YAML editor with id, type, config, depends)

---

## 14. Out of Scope (v1)

- Function values in configs.
- Function calls inside template expressions.
- Cancellation / pause / resume.
- Distributed execution.

---

## 15. Minimal Examples

### 15.1 Poll until ready, then webhook

```ts
import { z } from "zod";
import { flow, step } from "../core";

export default flow("poll-and-notify", {
  input: z.object({ pollUrl: z.string(), webhookUrl: z.string() }),
  steps: [
    step("poll", "loop", {
      until: "{{ $current.body.status === 'complete' }}",
      maxIterations: 30,
      delayMs: 2000,
      body: step("check", "http", { url: "{{ input.pollUrl }}" }),
    }),
    step("notify", "http", {
      url: "{{ input.webhookUrl }}",
      method: "POST",
      body: "{{ poll.body.result }}",
    }),
  ],
});
```

### 15.2 Parallel branches with merge (DAG)

```ts
export default flow("enrich", {
  input: z.object({ id: z.string() }),
  steps: [
    step("profile", "http", { url: "/profile/{{ input.id }}" }),
    step("orders", "http", { url: "/orders/{{ input.id }}" }, { depends: [] }),
    step("save", "http", {
      url: "/save",
      method: "POST",
      body: { profile: "{{ profile }}", orders: "{{ orders }}" },
    }, { depends: ["profile", "orders"] }),
  ],
});
```

### 15.3 Child workflow (subflow)

Subflows reference a published workflow by name. The child must already exist in the workspace.

```ts
export default flow("deploy", {
  input: z.object({ service: z.string() }),
  steps: [
    step("kick", "http", {
      url: "/deploy/{{ input.service }}",
      method: "POST",
    }),
    step("tell", "subflow", {
      workflow: "notify",          // name of a published workflow
      // version: "v2",            // optional; defaults to active version
      input: { message: "deployed {{ input.service }}" },
    }),
  ],
});
```

### 15.4 Publishing via API

```bash
# Create a workflow (steps array — server serializes to YAML)
curl -X POST http://localhost:3000/workflows/hello \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "v1",
    "steps": [
      { "id": "greet", "type": "log", "config": { "message": "Hello {{ input.name }}!" } },
      { "id": "fetch", "type": "http", "config": { "url": "https://httpbin.org/json" } }
    ],
    "description": "A hello world workflow"
  }'

# Or publish raw YAML
curl -X POST http://localhost:3000/workflows/hello \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "v1",
    "yaml": "name: hello\nsteps:\n  - id: greet\n    type: log\n    config:\n      message: \"Hello {{ input.name }}!\"\n"
  }'

# Run it
curl -X POST http://localhost:3000/workflows/hello/run \
  -H 'Content-Type: application/json' \
  -d '{ "input": { "name": "World" } }'

# View the run events
curl http://localhost:3000/workflows/hello/runs/<runId>/events
```
