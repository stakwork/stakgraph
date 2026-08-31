# Generic storage — one boundary per layer, filesystem as one implementation

## Problem

AGENTS.md claims "Persistence: Filesystem … Swappable via `RunStore` iface".
That's half-true, and the half that's false blocks any non-filesystem
backend (the concrete target: a Neo4j-backed workspace for graph-native
deployments). Storage is five separate layers, in three states of health:

| Layer         | Today                                          | Swappable?          |
| ------------- | ---------------------------------------------- | ------------------- |
| Run events    | `RunStore` iface (`store.ts`)                  | **No** — see below  |
| Workflows/steps | `WorkspaceManager` concrete class (869 lines) | **No** — no iface   |
| Chats         | `ChatStore` iface + File/Memory impls          | Yes                 |
| Secrets       | `SecretStore` iface + File/Memory impls        | Yes                 |
| Artifacts/cassettes/shell cwd | raw paths off `workspace.path` | N/A — see "local dirs" |

The three specific failures:

1. **`RunStore` is write-only.** The interface is `append` + `finalize`;
   every read (`listRuns`, `getRunSummary`, `getRunEvents`, `tailEvents`)
   lives on the concrete `FileRunStore`. `createVein.ts` has eight
   `store instanceof FileRunStore` guards that return `501` for any other
   store — run listing, run lookup, event lookup, SSE streaming, durable
   resume, and both promotion endpoints all capability-gate on the concrete
   class. `authoring.ts` papers over the same gap with its own
   `RunReadStore` + `asReadStore` feature-detection. A custom store today
   gets a vein that can launch runs but never read them back.

2. **`WorkspaceManager` has no interface.** It's injectable into
   `createVein` but *typed as the class*, so "swap it" means structurally
   cloning ~25 methods with no compiler-checked contract. (The repo already
   knows the right shape: `runner.ts`'s `SubflowResolver` is a 2-method
   interface the manager happens to satisfy.)

3. **`workspace.path` leaks a filesystem root to ~15 call sites.** Step
   discovery (`buildRegistry`), step source reads
   (`readStepSourceFromDisk`, `ai/stepHelpers`), cassette paths, the
   artifacts capability, the agent step's shell cwd, and a raw
   `_metadata.json` read in `GET /workflows/:name` all reach through the
   workspace object to its directory. A graph-backed workspace has no
   `.path` to give them.

There is also one cross-layer leak in the other direction:
`WorkspaceManager.lastRunAt` reads the run store's on-disk layout
(`workflows/<name>/runs/` dir names) directly — workspace code depending on
`FileRunStore`'s private file format.

## Non-goals

- **Implementing the Neo4j backend.** This plan makes the boundary real and
  proves it with the existing File + Memory impls. The graph impl is a
  follow-up (sketch in §7).
- **Making artifacts/cassettes/shell graph-backed.** These are inherently
  local (blobs, dev capture files, a working directory for spawned
  processes). They get an explicit local root, not an abstraction (§4).
- **New features.** Behavior with the default filesystem wiring is
  byte-identical after every step.

## 1. Widen `RunStore` to the full contract

Fold the read half into the interface (`store.ts`):

```ts
export interface RunStore {
  // writes (unchanged)
  append(workflow: string, runId: string, event: RunEvent): Promise<void>;
  finalize(workflow: string, runId: string, summary: RunSummary): Promise<void>;
  // reads (promoted from FileRunStore)
  listRuns(workflow: string): Promise<string[]>;           // newest first
  getRunSummary(workflow: string, runId: string): Promise<RunSummary | null>;
  getRunEvents(workflow: string, runId: string): Promise<RunEvent[]>;
  // history → live tail; closes after a terminal event unless reopened
  // (run.resumed) or opts.stillLive says a resume is in flight.
  tailEvents(workflow: string, runId: string, opts?: TailOpts): AsyncGenerator<RunEvent>;
  // newest run's start time (epoch ms) or null — replaces
  // WorkspaceManager.lastRunAt's cross-layer dir read.
  lastRunAt(workflow: string): Promise<number | null>;
}
```

- The tail contract (history from event 0, then follow, terminal /
  `reopens` / `stillLive` semantics per RUN_CONTROL_SPEC §5.2) becomes
  interface documentation, not `FileRunStore` internals.
- Ship a generic `tailFromPolling(store, workflow, runId, opts)` helper
  built on repeated `getRunEvents` + an index cursor, so a backend without
  a native tail (memory, a database) implements only the five data methods
  and delegates `tailEvents` to the helper. `FileRunStore` keeps its
  byte-offset `tailJsonl` (cheaper — no full re-read per poll).
- `MemoryRunStore` implements the full interface (it already holds the
  arrays; the reads are ~15 lines). It stops being a write-only test stub
  and becomes a complete ephemeral backend — memory-mode vein gets run
  history, SSE reattach, resume, and promotions for free.
- Delete all eight `instanceof FileRunStore` guards in `createVein.ts` and
  the 501 branches. Delete `authoring.ts`'s `RunReadStore` / `asReadStore`
  / `NO_RUN_HISTORY_ERROR` — it takes `RunStore`.
- `lastRunAt` moves from `WorkspaceManager` to the store; `listWorkflows`
  gains access via a store handed to it (§2 — the workspace iface method
  takes the value, or `createVein` composes the two; pick composition:
  `listWorkflows` returns entries without `lastRunAt`, `createVein`'s
  `GET /workflows` decorates from `store.lastRunAt`). Keeps the layers
  ignorant of each other.

This step is standalone-valuable (memory-mode becomes fully functional;
`createVein.ts` shrinks) and everything later depends on it.

## 2. Extract `WorkspaceStore` from `WorkspaceManager`

New interface in `workspace.ts`, mechanically derived from the public
surface of the class:

```ts
export interface WorkspaceStore {
  // workflows
  listWorkflows(): Promise<WorkflowListEntry[]>;
  getWorkflow(name): Promise<Flow>;
  getWorkflowVersion(name, version): Promise<Flow>;
  getWorkflowSource(name, version): Promise<string>;
  getWorkflowHash(name, version?): Promise<string | null>;
  getWorkflowMetadata(name): Promise<WorkflowMetadata | null>;   // NEW — see below
  createWorkflow(...): Promise<{ name, version }>;
  publishWorkflow(...): Promise<void>;
  publishWorkflowByContent(...): Promise<...>;
  setWorkflowCategory(name, category): Promise<void>;
  setActiveVersion(name, version): Promise<void>;
  setParam(...): Promise<...>;
  deleteWorkflow(name): Promise<boolean>;
  // steps
  listSteps(filter?): Promise<StepListEntry[]>;
  publishStep(...): Promise<...>;
  listStepVersions(name): Promise<StepVersionsResult>;
  getStepVersionSource(name, version): Promise<string>;
  setActiveStepVersion(name, version): Promise<void>;
  deleteStep(name): Promise<boolean>;
  deleteStepsByPublisher(publisher): Promise<string[]>;
  // step source + materialization (§3)
  getStepSource(type): Promise<{ code: string; origin: StepSource } | null>;
  materializeCustomSteps(): Promise<string>;   // dir importable by buildRegistry
}
```

- `WorkspaceManager` is renamed `FileWorkspaceStore`; keep
  `export { FileWorkspaceStore as WorkspaceManager }` so embedders don't
  break. `VeinOptions.workspace`, `Vein.workspace`, `AiDeps.workspace`,
  `stepHelpers`, `prompts.ts`, `authoring.ts` all retype to the interface.
- `getWorkflowMetadata` replaces `createVein.ts`'s raw `_metadata.json`
  `readFile` in `GET /workflows/:name` (the one route that bypasses the
  manager entirely today).
- `SubflowResolver` in `runner.ts` stays as-is (it's the narrow view the
  runner needs; `WorkspaceStore` extends it structurally).
- The interface conformance suite (§6) is written against this.

## 3. Step source and code loading

Custom steps are *executable code* — the one place storage and filesystem
genuinely can't be fully separated, because Node's module loader imports
files. The boundary that works for every backend:

- **`getStepSource(type)`** — read a step's source text (core / lib /
  custom tiers). File impl wraps today's `readStepSourceFromDisk`; a graph
  impl serves custom from the graph and still reads core/lib from vein's
  own install dir (those ship with the engine and are backend-independent).
- **`materializeCustomSteps()`** — ensure every *active* custom step exists
  as an importable file and return the directory root. File impl returns
  `<root>/steps/custom` (already materialized — publish writes it). A graph
  impl writes active sources to a scratch dir (content-hash named, so
  re-materialization is cheap and idempotent) and returns that.
- `buildRegistry` changes signature from `buildRegistry(workspacePath)` to
  `buildRegistry(customDir)` — callers pass
  `await workspace.materializeCustomSteps()`. Core/lib discovery is
  untouched (engine-relative, not workspace-relative).
- `ai/stepHelpers`'s `lsSteps` / `searchSteps` / `readStepSource` browse
  through `getStepSource` + `listSteps` instead of walking
  `deps.workspace.path`. (The "filesystem-style browser" presentation for
  the AI keeps its ls/glob *shape* — it's just fed from the interface.)

## 4. Local dirs: stop overloading `workspace.path`

The remaining `.path` consumers don't want the *workspace* — they want *a
local directory*. Give them one explicitly. `createVein` grows a resolved
`dataDir` (default: the same `VEIN_WORKSPACE` root, so file-backed
deployments see zero change):

| Consumer                              | Today                                  | After                          |
| ------------------------------------- | -------------------------------------- | ------------------------------ |
| artifacts capability                  | `join(workspace.path, "artifacts")`    | `join(dataDir, "artifacts")`   |
| cassette record/replay paths          | `cassettePath(workspace.path, …)`      | `cassettePath(dataDir, …)`     |
| agent step / chat shell cwd + scratch | `workspace.path`                       | `dataDir`                      |
| authoring cassette + custom-step peek | `deps.workspace.path`                  | `deps.dataDir` / iface methods |

- `VeinOptions.dataDir?: string` overrides. A graph-backed deployment
  points it at any scratch volume; losing it loses blobs/cassettes/scratch
  but no workspace records — that's the explicit contract.
- After this step `WorkspaceStore` needs no `path` getter. Keep `path` on
  `FileWorkspaceStore` only (impl detail); nothing in `createVein`,
  `authoring`, or `ai/` touches it.

## 5. Defaults and wiring

`createVein` default selection today keys off `store instanceof
MemoryRunStore` to pick memory chat/secret stores. Replace the scattered
instanceof checks with one resolved mode at the top:

```ts
const fileBacked = opts.workspace ? opts.workspace instanceof FileWorkspaceStore
                                  : true;  // default workspace is file-backed
```

…used *only* for choosing unspecified defaults (store/chatStore/
secretStore follow the workspace kind). Capability gating by instanceof is
gone entirely (§1). Passing any custom impl explicitly always wins.

## 6. Conformance tests

New `storage-conformance.test.ts`: one suite of behavioral tests
parameterized over `(makeWorkspace, makeRunStore, makeChatStore,
makeSecretStore)` factories, run against **file** (tmpdir) and **memory**
impls. Covers: publish/version/activate/dedup round-trips, run
append→read→tail (incl. terminal + reopen), partial summaries from a
finalize-less log, chat turn tail, secret list-never-values. A future
`Neo4jWorkspaceStore` passes by running the same suite against a test
container — the suite *is* the spec of the boundary.

Plus: full existing suite green after every numbered step (each step is a
separate commit; §1 and §2+3+4 are independently shippable).

## 7. The Neo4j follow-up (sketch, out of scope here)

What the boundary buys, and the recommended split:

- **Graph-backed:** `Neo4jWorkspaceStore` — workflows, versions, steps,
  metadata as nodes; `ACTIVE`, `VERSION_OF`, `DEPENDS_ON`, `PUBLISHED_BY`
  edges. This is where the graph pays: "which workflows use step X",
  version lineage, promotion ancestry (EVOLVE_SPEC) become one-hop queries.
  Custom step source lives as node properties; `materializeCustomSteps`
  writes them to `dataDir` scratch at boot/rebuild.
- **Not graph-backed (recommendation):** run events. High-volume
  append-only with byte-offset tailing is the wrong shape for Neo4j; the
  file `RunStore` (or later Postgres) composes fine beside a graph
  workspace — the layers are independently pluggable, which is the point.
- Chats/secrets: either; low value, small surface.

## Sequencing vs. `workspace-files-and-includes`

**This plan lands first.** The files plan adds `publishFile` /
`getFile` / draft accumulation to "WorkspaceManager" — after this plan
those land as `WorkspaceStore` interface methods (+ conformance cases +
file impl) from day one, instead of being extracted in a second pass.
`@@include` expansion inside `publishWorkflowByContent` is
backend-independent and unaffected.

## Step order

1. **Widen `RunStore`** (reads + tail + `lastRunAt` + polling-tail helper);
   full `MemoryRunStore`; delete the eight guards + `authoring.ts`
   feature-detect; move `lastRunAt` decoration into `GET /workflows`.
2. **Extract `WorkspaceStore`**; rename class to `FileWorkspaceStore`
   (aliased); add `getWorkflowMetadata`; retype all consumers.
3. **Step source boundary**: `getStepSource` + `materializeCustomSteps`;
   `buildRegistry(customDir)`; rewire `ai/stepHelpers`.
4. **`dataDir`**: artifacts/cassettes/shell/scratch off the explicit local
   root; remove every `workspace.path` read outside `workspace.ts`.
5. **Default wiring** cleanup (§5) + **conformance suite** (§6).
6. (Follow-up plan) `Neo4jWorkspaceStore` against the conformance suite.
