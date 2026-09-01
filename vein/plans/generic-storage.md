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

The goal is a graph that holds the *skeleton* of everything — the domain
knowledge, the workflow/step logic, AND the usage of both — with heavy
payloads hanging off it by reference. The split that serves that:

- **Graph-backed:** `Neo4jWorkspaceStore` — workflows, versions, steps,
  metadata as nodes (`VeinWorkflow`, `VeinWorkflowVersion`, `VeinStep`,
  `VeinStepVersion` — see the label registry below); `ACTIVE_VERSION`,
  `VERSION_OF`, `USES_STEP`, `DEPENDS_ON`, `PUBLISHED_BY` edges. This is
  where the graph pays: "which workflows use step X", version lineage,
  promotion ancestry (EVOLVE_SPEC) become one-hop queries.
  Custom step source lives as node properties; `materializeCustomSteps`
  writes them to `dataDir` scratch at boot/rebuild.
- **Run events — two tiers, not "not graph-backed."** The queries we
  actually want over runs ("which prompt versions touch which subgraph",
  "how did 10k traces perform against Concept X", "which runs promoted
  which param") are queries over structure and linkage, not raw token
  streams. So:
  - *Raw log stays append-only and pluggable* (file, later Postgres).
    High-volume events with byte-offset tailing are the wrong shape for
    Neo4j node properties; stuffing full tool I/O and transcripts into the
    graph makes it slow without adding a queryable edge. The raw store
    remains the store of record for SSE tailing, resume, and replay.
  - *A graph projection is built on top:* `VeinRun`, `VeinTurn`,
    `VeinAgentSession`, `VeinToolCall` nodes;
    `EXECUTED (VeinRun→VeinWorkflowVersion)`,
    `ACCESSED (VeinToolCall→Concept | any node)`, `PROMOTED_FROM` edges;
    the run's `params` snapshot lives as properties on `VeinRun` (it's a
    value bag, not a relationship). Nodes carry summaries + a pointer
    back to the raw log, never full payloads.
    The raw materials already exist: `run.start` records `workflowHash`
    and `params`, and `promotes` declares the output→param evolution
    mapping — Run→version→prompt-knob linkage is derivable today.
  - *Recommended shape: a post-hoc projector*, i.e. a consumer of any
    `RunStore` — §1's widened read interface (`listRuns`/`getRunEvents`)
    is precisely what an ingester needs. It can run as a vein workflow
    itself, batch or streaming, and can be re-run to rebuild/enrich the
    projection as the edge vocabulary evolves. Zero coupling to the hot
    path. (Alternative: a `Neo4jRunStore` that dual-writes — file append,
    projection on `finalize` — if projection lag ever matters.)
- **Chats: projected (or fully graph-backed), not "low value."** A chat
  turn is where a human's intent enters the system;
  `VeinChat→SPAWNED→VeinRun→…→ACCESSED→Concept` is the provenance chain
  a reflection loop reads. Volume is low enough that chats could live
  entirely in the graph; at minimum they get projected alongside runs
  with the spawn edge.
- Secrets: either; small surface, no linkage value.

### Label registry (checked against jarvis `schema_library.py`)

The target graph is jarvis's Neo4j, whose schema library already defines
153 node types — including an entire `Workflow` domain (`Workflow`,
`Workflow_version`, `Run`, `Run_step`, `Turn`, `Agent`, `AgentSession`,
`Prompt`) that belongs to a **different workflow engine**. We do NOT
reuse those labels: same-name nodes with different semantics and
node_keys would corrupt both engines' queries. Rule, following the
existing `Hive*` precedent for a separate product family: **every vein
node label is `Vein`-prefixed, domain `Vein`**, and every edge label is
verified absent from the library before use. `Concept` (and other
domain-knowledge nodes) are jarvis's — we point edges AT them, never
redefine them.

Node labels (all verified unused in the library):

| Label                 | What it is                                              |
| --------------------- | ------------------------------------------------------- |
| `VeinWorkflow`        | A workflow by name (the stable identity)                |
| `VeinWorkflowVersion` | One content-hashed version of a workflow                |
| `VeinStep`            | A published step type (custom tier)                     |
| `VeinStepVersion`     | One version of a step's source                          |
| `VeinRun`             | One run (projected: status, timings, params, log ref)   |
| `VeinAgentSession`    | One agent-step execution inside a run                   |
| `VeinToolCall`        | One tool call inside an agent session                   |
| `VeinChat`            | A long-lived chat                                       |
| `VeinTurn`            | One turn of a chat                                      |

Edge labels (all verified absent; child→parent direction for `IN_*`):

| Edge             | From → To                                   |
| ---------------- | ------------------------------------------- |
| `VERSION_OF`     | `VeinWorkflowVersion`/`VeinStepVersion` → parent |
| `ACTIVE_VERSION` | `VeinWorkflow`/`VeinStep` → its active version |
| `USES_STEP`      | `VeinWorkflowVersion` → `VeinStep`          |
| `DEPENDS_ON`     | `VeinWorkflowVersion` → `VeinWorkflow` (subflow) |
| `PUBLISHED_BY`   | `VeinStepVersion` → `Person` (optional)     |
| `EXECUTED`       | `VeinRun` → `VeinWorkflowVersion`           |
| `PROMOTED_FROM`  | `VeinWorkflowVersion` → `VeinRun` (param promotion lineage) |
| `IN_RUN`         | `VeinAgentSession` → `VeinRun`              |
| `IN_SESSION`     | `VeinToolCall` → `VeinAgentSession`         |
| `SPAWNED`        | `VeinChat` → `VeinRun`                      |
| `IN_CHAT`        | `VeinTurn` → `VeinChat`                     |
| `ACCESSED`       | `VeinToolCall` → `Concept` (or any graph node) — provenance |

Near-collision notes (why some obvious names were rejected): the library
already uses `TOUCHES`, `DERIVED_FROM`, `USES`, `HAS_TURN`,
`HAS_SESSION`, `HAS_PROMPT`, `CALLS`, `TRIGGERED_BY` — hence `ACCESSED`
(not `TOUCHED`/`TOUCHES`), `PROMOTED_FROM` (not `DERIVED_FROM`, which is
ProposedFix fix-lineage), `USES_STEP` (not `USES`), and the `IN_*`
child→parent family (not `HAS_*`, heavily overloaded). Any new label
added later gets the same check against `schema_library.py` first.

This registry is the shared vocabulary — the conformance suite's future
graph cases, the projector, and EVOLVE_SPEC all speak it.

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

## v2: Provenance convention (the gap the projection can't close alone)

Storage backends are not what blocks "which prompts touch which parts of
the graph" — **provenance capture is**. Today a graph-touching tool call
(e.g. a `jarvis/*` step exposed via `agentTools`) is logged as
`stepType: "tool:jarvis/search"` with I/O truncated to 1500 chars for
event-log readability (`summarizeForEvent` in `steps/core/agent.ts`).
Which Concept nodes that call touched is recorded nowhere structured —
a projector would have to parse prose. Without this, the `ACCESSED` edge
has no data to run on.

The convention (small, additive — v2 work, after the boundary lands):

- **Graph-touching steps return touched node refs in a standard field**
  on their output — e.g. `output._nodes: [{ id, label }]` (or a
  well-known key negotiated with jarvis). Steps that don't touch the
  graph simply omit it.
- **`wrapToolsWithEmit` preserves `_nodes` untruncated** in the emitted
  `step.end` event, exempt from `summarizeForEvent` — it's the one part
  of tool output that must survive into the log verbatim, because it's
  data for the projector, not a preview for humans.
- **The projector turns `_nodes` into `ACCESSED` edges**
  (`VeinToolCall→Concept`), completing the chain:
  `VeinChat→SPAWNED→VeinRun→EXECUTED→VeinWorkflowVersion` and
  `VeinRun←IN_RUN←VeinAgentSession←IN_SESSION←VeinToolCall→ACCESSED→Concept`.
  That chain is what makes
  self-evolution queryable: evaluate trace cohorts against the subgraph
  they touched, and trace a prompt/param version's blast radius through
  the domain graph.
- Same convention applies to chat-mode agents (their tool calls emit
  into the chat observability stream) so chat provenance projects
  identically.

Non-goal even in v2: inferring touched nodes from unstructured output.
If a step doesn't report refs, it gets no `ACCESSED` edges — explicit
over clever.
