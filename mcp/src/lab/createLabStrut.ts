import {
  createStrut,
  createMothership,
  WorkspaceManager,
  FileRunStore,
  FileChatStore,
  FileSecretStore,
  graphWorkspaceRequested,
  graphWorkspaceFromEnv,
  seedConcepts,
  STRUT_CONCEPTS,
  type WorkspaceStore,
  type Strut,
  type RunResult,
} from "strut";
import {
  buildConceptServices,
  type ConceptServices,
  type BuildServicesOptions,
} from "./concepts/services.js";
import { seedConceptWorkflows, seedConceptSteps } from "./concepts/seed.js";
import { EVAL_CONCEPTS, seedEvalSteps } from "./eval/seed.js";
import { seedJarvisSteps } from "./jarvis/seed.js";
import { seedGaiaSteps, seedGaiaWorkflows } from "./gaia/seed.js";
import { seedArtifactSteps } from "./artifacts/seed.js";
import { seedWfbenchSteps, seedWfbenchWorkflows } from "./wfbench/seed.js";
import { CODE_CONCEPTS, seedCodeWorkflows } from "./code/seed.js";
import { JOB_CONCEPTS, JOB_SEED } from "./job/seed.js";
import { ORCHESTRATOR_SEED, seedOrchestratorSteps } from "./orchestrator/seed.js";
import { POD_CONCEPTS, seedPodSteps } from "./pods/seed.js";
import { POD_SETUP_CONCEPTS, seedPodSetupSteps, seedPodSetupWorkflows } from "./pod-setup/seed.js";
import { JANITOR_CONCEPTS, seedJanitorSteps, seedJanitorWorkflows } from "./janitor/seed.js";
import { seedExploreSteps, seedExploreWorkflows } from "./explore/seed.js";
import { LOGS_CONCEPTS, seedLogsSteps, seedLogsWorkflows } from "./logs/seed.js";
import { buildLogsServices, type LogsServices } from "./logs/service.js";
import { seedCallsWorkflows } from "./calls/seed.js";
import { seedTribesWorkflows } from "./tribes/seed.js";
import { anchorConcepts } from "./concept-anchor.js";
import { retireSteps, retireWorkflows } from "./seed-opts.js";
import { buildGaiaServices, type GaiaServices } from "./gaia/service.js";
import { resolveLabActor, resolveLabScope } from "./actor.js";
import { env } from "../env.js";

/** Steps the lab used to seed and strut now ships as lib steps of the same
 *  name (`browser/_shared` was the seeded helper the others imported). */
const MOVED_TO_STRUT = [
  ...["create-spreadsheet", "update-values", "batch-update-values", "get-values", "add-sheet", "import-spreadsheet"].map((n) => `sheets/${n}`),
  ...["_shared", "open", "goto", "back", "close", "snapshot", "text", "evaluate", "observe", "click", "fill", "type", "press", "select", "hover", "scroll", "wait", "screenshot", "capture", "verify", "walkthrough", "state"].map((n) => `browser/${n}`),
];

/** Steps and workflows the lab used to seed and no longer ships at all: the
 *  gitsee setup profiler (its QA harness, boot gate and eval/optimize stack)
 *  and the Harvey LAB grader with its deliver pipeline, both unused. Seeding
 *  is additive, so a deployed swarm keeps them until they are retired here
 *  (seed-opts.ts). The lists stay as long as a swarm might still hold one;
 *  missing names are a no-op. */
const RETIRED_STEPS = [
  ...["clone-workspace", "score-setup", "verify-setup", "boot-and-exercise", "stage-setup", "boot", "browser-open", "browser-snapshot", "browser-click", "browser-fill", "browser-press", "browser-observe", "assess-ui", "read-logs", "finalize-setup", "capture-edits"].map((n) => `gitsee/${n}`),
  ...["get-task", "evaluate", "digest-results", "normalize-documents", "graph-sub-agent", "ingest-state", "drafter-plan", "validate-deliverables", "filter-contested", "merge-disputes", "generate-docx", "generate-xlsx",
    // retired by the harvey seeder before it went
    "pack-result", "aggregate-scores", "build-eval-chain", "criterion-refs"].map((n) => `harvey/${n}`),
];
const RETIRED_WORKFLOWS = [
  ...["explore-services", "setup-and-run", "eval", "eval-score", "eval-reflect", "optimize"].map((n) => `gitsee-${n}`),
  ...["produce", "run", "candidate-run", "evolve-gen", "evolve", "ingest-doc", "knowledge", "draft", "judge-criterion", "dispute-criterion", "score", "deliver"].map((n) => `harvey-${n}`),
];

/**
 * Lets a step run other workflows (and read their params) from inside a run —
 * the `eval/optimize` loop uses this to eval/reflect across generations. A leaf
 * step has no runner of its own, so we hand it a closure over `strut.run`. Set
 * AFTER the strut is built (it closes over the instance); see `createLabStrut`.
 */
export interface OptimizerCapability {
  run(
    name: string,
    input: unknown,
    opts?: {
      paramOverrides?: Record<string, Record<string, unknown>>;
      /** The calling step's `ctx.runId` — links the nested run's controller
       *  under the launching run's, so cancelling/pausing an evolve run
       *  reaches its generation/candidate runs (RUN_CONTROL_SPEC §2.2). */
      parentRunId?: string;
    },
  ): Promise<RunResult>;
  getParams(name: string): Promise<Record<string, unknown>>;
}

/**
 * The merged capabilities bag for ALL lab experiments. Each experiment's
 * steps cast `ctx.services` to whatever subset they need (services is
 * untyped at runtime), so a single merged bag serves every experiment.
 * Extend this as experiments are added.
 */
export interface LabServices extends ConceptServices {
  /** Run-sub-workflows capability for the optimize loop. Injected post-build. */
  optimizer?: OptimizerCapability;
  /** GAIA LAB scoring: runs the REAL leaderboard scorer.py (python3
   *  subprocess) against the validation gold in the pinned dataset checkout
   *  (GAIA_DIR). In-code on purpose — the grader and the gold must stay
   *  outside the agent-editable surface; gaia/* steps are thin plumbing. */
  gaia?: GaiaServices;
  /** The logs agent's tools for the `logs` workflow (lab/logs/service.ts):
   *  `/logs/agent`'s fetches and texts plus search / read / insights, over
   *  this host's AWS credentials. In code because a seeded step imports
   *  `strut` alone; the logs/* steps are thin plumbing over it. */
  logs?: LogsServices;
  /** Generic per-run teardown hook called by the strut runner in a `finally`
   *  (success AND error). Nothing in the lab sets it today; a caller-built
   *  bag may. (The `browser/*` steps' browser is strut's own
   *  `services.browser`, built from BROWSER_* — createStrut chains its
   *  teardown after this.) */
  onRunEnd?(runId: string): Promise<void>;
}

export interface CreateLabStrutOptions extends BuildServicesOptions {
  /** Pre-built merged services bag. If omitted, built from env. */
  services?: LabServices;
  /** Workspace dir for all lab workflows/runs. Defaults to
   *  `STRUT_LAB_WORKSPACE` or `./lab-workspace`. */
  workspacePath?: string;
  /** Serve the strut web UI (true when run standalone on its own port;
   *  the Express `/lab` mount passes false). Defaults to true. */
  serveUi?: boolean;
}

/**
 * Build THE single lab strut instance: one workspace, one UI, one merged
 * services bag. Experiments are just groups of workflows/steps inside it —
 * not separate servers.
 *
 * Steps and workflows are seeded into the workspace as content-hash–versioned
 * artifacts and discovered from disk (no in-code registry injection), so they
 * are editable + versioned through the strut API/UI.
 *
 * Adding an experiment = seed its step + workflow templates here and merge its
 * services into the bag.
 */
export async function createLabStrut(
  opts: CreateLabStrutOptions = {},
): Promise<Strut<LabServices>> {
  // Merged services bag. Today concepts + the GAIA grader; spread
  // additional experiments' bags here as they're added.
  const services: LabServices =
    opts.services ?? (await buildConceptServices(opts));

  // GAIA LAB grader — in-code, NOT seeded (see gaia/service.ts). Only added
  // when absent, to respect a caller-provided bag. Construction is cheap;
  // GAIA_DIR is checked at call time (loud per-run error).
  if (!services.gaia) {
    services.gaia = buildGaiaServices();
  }
  // The logs workflow's tools (lab/logs/service.ts). Cheap to build; AWS is
  // reached on a tool's first call.
  if (!services.logs) {
    services.logs = buildLogsServices();
  }

  const workspacePath =
    opts.workspacePath ??
    env["STRUT_LAB_WORKSPACE"] ??
    "./lab-workspace";

  // Seed each experiment's workflow + step templates into the workspace
  // BEFORE building the strut, so the registry's discovery picks up the
  // seeded steps. Steps are self-contained custom steps (not injected
  // in-code) — content-hash reconciled (see seed-opts.ts: a changed template
  // wins, an unchanged one leaves UI/API edits active), editable + versioned
  // via the strut API/UI. No `registry` is passed, so createStrut discovers core + lib +
  // these custom steps via `workspace.materializeCustomSteps()` (and step
  // publishing is enabled).
  //
  // Workflows/steps live in Neo4j by default (strut's Neo4jWorkspaceStore;
  // NEO4J_HOST/USER/PASSWORD, same defaults as this host's own client);
  // STRUT_WORKSPACE_BACKEND=fs keeps them on disk instead. Runs, chats,
  // secrets, artifacts, and cassettes stay on disk under `workspacePath`
  // either way.
  const graphBacked = graphWorkspaceRequested();
  const workspace: WorkspaceStore = graphBacked
    ? (await graphWorkspaceFromEnv(env, { dataDir: workspacePath })).workspace
    : new WorkspaceManager(workspacePath);
  if (graphBacked) console.log(`[lab] strut workspace: Neo4jWorkspaceStore (data dir: ${workspacePath})`);
  // Generic, domain-agnostic eval primitives (steps). The concept-specific eval
  // WORKFLOWS that wire them (concepts-eval*) are seeded with the concepts
  // experiment below.
  await seedEvalSteps(workspace);
  await seedConceptWorkflows(workspace);
  await seedConceptSteps(workspace);
  // jarvis knowledge-graph steps (self-contained; reach Jarvis over
  // ctx.services.http with JARVIS_URL/API_TOKEN from ctx.services.secrets).
  // Grantable to agents via agentTools: ["jarvis/*"].
  await seedJarvisSteps(workspace);
  // orchestrator — the lab's one step of its own, `concept/create`: a new
  // Concept composed (node, PARENT_OF, the workspace anchor by kind), gated
  // and granted to the orchestrator by ORCHESTRATOR_SEED below. Seeded
  // before createStrut like every custom step, so discovery finds it.
  await seedOrchestratorSteps(workspace);
  // gaia harness (thin plumbing over services.gaia; grant gaia/evaluate —
  // and every gaia/* — only to harness workflows, never the producing agent).
  await seedGaiaSteps(workspace);
  await seedGaiaWorkflows(workspace);
  // generic artifact plumbing (artifacts/dir — bridge runId → path for cwd).
  await seedArtifactSteps(workspace);
  // wfbench — the Workflow Editor Agent Benchmark harness (stakwork 58313's
  // twin; plans/wfbench-harness.md). Pure plumbing steps; never granted to
  // an agent (the author gets meta/* only).
  await seedWfbenchSteps(workspace);
  await seedWfbenchWorkflows(workspace);
  // code — code changes as strut workflows (strut plans/code-change.md):
  // hive's propose_code_change preview, run by strut's own agent. YAML only;
  // every step is strut's (git/checkout, agent, git/diff, pack).
  await seedCodeWorkflows(workspace);
  // job — the job agent as a workflow (strut plans/jobs.md §5) is STRUT's:
  // createStrut seeds `src/workflows/job.yaml` at boot; the lab passes only
  // its delta (`job: JOB_SEED` below — the pod/* grants and the Hive
  // workspace paragraph). Never seeded from here as well: one seeder per name.
  // janitor — the engine (`graph-janitor`: YAML only, one automation per
  // mandate).
  // janitors: the plan / readback / revert-plan steps the engine writes through, then the two engine workflows.
  await seedJanitorSteps(workspace);
  await seedJanitorWorkflows(workspace);
  // explore — the knowledge-graph explorer (strut plans/federation.md
  // §2.2): the workflow another strut asks this one to run, a read-only
  // agent over the graph's read steps answering { answer, sources }, told
  // which repositories the graph holds parsed (explore/repositories).
  await seedExploreSteps(workspace);
  await seedExploreWorkflows(workspace);
  // logs — the logs agent (`/logs/agent`) as a workflow the orchestrator
  // runs here or on a peer swarm: the endpoint's tools through services.logs,
  // its bash replaced by logs/search, logs/read and logs/insights.
  await seedLogsSteps(workspace);
  await seedLogsWorkflows(workspace);
  // calls — what Hive runs here about its calls: `call-transcript`, the
  // workflow its CALL_TRANSCRIPT_STRUT_WORKFLOW_NAME names, which gets a
  // call's whole transcript when the call ends. A placeholder for now. YAML
  // only.
  await seedCallsWorkflows(workspace);
  // tribes — what Hive runs here about the Sphinx tribes linked to its org:
  // `tribe-messages`, the workflow its TRIBE_MESSAGES_STRUT_WORKFLOW_NAME
  // names, which gets what was said in one, a batch at a time. A placeholder
  // for now. YAML only.
  await seedTribesWorkflows(workspace);
  // sheets/* and browser/* are strut's lib steps now (strut#200, #201), and
  // strut seeds the three browser workflows (same names, same YAML). The
  // lab's seeded copies of the steps are retired: a workspace custom step
  // shadows a lib step of the same name, so until then a swarm ran the lab's.
  // The workflows are NOT retired: that would delete strut's own.
  await retireSteps(workspace, MOVED_TO_STRUT, "lab");
  // gitsee and harvey are gone (steps, workflows, services): what a swarm
  // still holds of them is retired, and their runs on disk are left alone.
  await retireSteps(workspace, RETIRED_STEPS, "lab");
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "lab");
  // pods: hive sandboxes as steps, `pod/*` — the job agent's tools for a
  // change that must run to be judged (claim → latest → agent → push →
  // release; the pod is a hold on the job between turns).
  await seedPodSteps(workspace);
  // pod setup: a workspace's first pod config, drafted from its repositories
  // (strut plans/onboarding.md) — identify-services and its two code steps;
  // the onboarding job runs it, then runs the result in a pod.
  await seedPodSetupSteps(workspace);
  await seedPodSetupWorkflows(workspace);
  // Mothership cost control (plans/mothership-cost-control.md §5) — strut's
  // opt-in module; core knows only the two hooks below. Hive pushes one
  // standing macaroon per user to `PUT /lab/llm/delegations/:actor`; from
  // then on that user's `agent`/`llm` steps and chat turns go through the
  // gateway, billed to user × workflow × step and capped per run. With no
  // delegation on file the hook returns undefined and steps call the
  // providers directly, as before. `mothership.json` sits beside
  // `secrets.json` under `workspacePath` in both workspace modes (the volume
  // sphinx-swarm already binds), encrypted under STRUT_SECRET_KEY — which
  // index.ts derives from API_TOKEN when unset.
  const mothership = createMothership({ dataDir: workspacePath });

  const strut = await createStrut<LabServices>({
    workspace,
    services,
    serveUi: opts.serveUi ?? true,
    llmAuth: mothership.llmAuth,
    // Who a request is from: what mount.ts's `labAuth` stashed on the Node
    // request — the mint-token JWT's `sub`, or hive's `x-strut-actor` on an
    // `x-api-token` call. Nothing off the Express bridge (smoke scripts,
    // tests calling `app.fetch`): no actor, so no owner stamp and direct keys.
    resolveActor: resolveLabActor,
    // And what it may do: `peer` for a `lab:peer` token — another strut,
    // which strut lets read, launch, and control the runs a peer launched.
    resolveScope: resolveLabScope,
    // No `chatSystem`: strut's default puts the `Workflow Builder` Concept's
    // page in the builder's prompt (it seeds the page; the lab's kinds hang
    // under it, below).
    // The job agent's deployment delta (job/seed.ts): `pod/*` and the page
    // instruments on top of strut's grants, and which Hive workspace this
    // strut belongs to after strut's prompt — merged into the `job` template
    // strut seeds at boot.
    job: JOB_SEED,
    // The orchestrator's (orchestrator/seed.ts): the graph reads, the
    // seven graph writes on this swarm's graph and `concept/create`, the
    // writes GATED (`approve`: a call records a proposal hive's card
    // decides), the meta/* read + run tools, and which Hive workspace this
    // strut belongs to plus what a Concept is here — merged into strut's
    // `orchestrator` template the same way.
    orchestrator: ORCHESTRATOR_SEED,
    // A non-file workspace would otherwise default the run/chat/secret
    // stores to memory — pin them to disk so history survives restarts.
    ...(graphBacked
      ? {
          dataDir: workspacePath,
          store: new FileRunStore(workspacePath),
          chatStore: new FileChatStore(workspacePath),
          secretStore: new FileSecretStore(workspacePath),
          // The builder's graph tools: `graph_get`, which its section of the
          // prompt opens pages with, and the read-only `graph_query`.
          ...(workspace.graph ? { graph: workspace.graph } : {}),
        }
      : {}),
  });
  // The delegation routes (`/llm/delegations`, `/llm/mothership`; mutations
  // behind strut's requireApiKey) and the workspace the hook reads run caps
  // from. Once, after createStrut.
  mothership.mount(strut);

  // Concept files — graph DATA, reconciled per node by a source stamp the way
  // SEED_OPTS reconciles versions, with strut's `seedConcepts`. Strut seeds
  // its own pages inside createStrut — `Workflow Builder` (what the AI
  // builder reads about this deployment) and `Job` (what the job agent opens
  // every turn), with its kinds `App` and `Workflow Build` (a builder job)
  // under them — so the lab's pages, which hang under them by name, seed
  // AFTER: `Janitor` under `Workflow Builder` and `Benchmark App` under its
  // `App`; `Code Change`, `Explore`,
  // `Logs`, `Onboarding`, `Plan Mode` and `Verify` under `Job`, under `Code Change`
  // the `Pod` way, and under `Pod Config` the `Pod Setup` pages
  // identify-services loads. Strut's set rides along (all kept, nothing
  // written) so its pages are owned in this call too: their PARENT_OF edges
  // to ours follow the files, and they are anchored with ours. No mandate
  // ships: those are each workspace's. A no-op on the fs workspace; a page
  // that will not seed is a warning, never a failed boot.
  try {
    const outcomes = await seedConcepts(workspace.graph, [STRUT_CONCEPTS, JANITOR_CONCEPTS, EVAL_CONCEPTS, JOB_CONCEPTS, CODE_CONCEPTS, LOGS_CONCEPTS, POD_CONCEPTS, POD_SETUP_CONCEPTS]);
    await anchorConcepts(workspace.graph, outcomes);
  } catch (err) {
    console.warn("[concepts] could not seed the lab's Concept pages:", err instanceof Error ? err.message : err);
  }

  // Inject the run-sub-workflows capability now that the instance exists.
  // CRITICAL: mutate `strut.services` — the EFFECTIVE bag createStrut built by
  // spreading our `services` into a fresh object (standardServices +
  // artifacts + ours) — NOT the local `services`, which runs never see
  // again. Mutating the local bag here silently broke every consumer of
  // `services.optimizer` (eval/optimize, eval/evolve-loop): steps threw
  // "requires a services.optimizer capability" at run time. This is what
  // lets the optimize/evolve loops run sub-workflows.
  const optimizer: LabServices["optimizer"] = {
    run: (name, input, runOpts) => strut.run(name, input, runOpts),
    getParams: async (name) => (await strut.workspace.getWorkflow(name)).params ?? {},
  };
  (strut.services as LabServices).optimizer = optimizer;
  services.optimizer = optimizer; // keep the caller's bag consistent too

  return strut;
}
