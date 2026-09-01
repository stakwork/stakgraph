import {
  createVein,
  WorkspaceManager,
  type Vein,
  type RunResult,
} from "vein";
import {
  buildConceptServices,
  type ConceptServices,
  type BuildServicesOptions,
} from "./concepts/services.js";
import { seedConceptWorkflows, seedConceptSteps } from "./concepts/seed.js";
import { seedEvalSteps } from "./eval/seed.js";
import { seedGitseeWorkflows, seedGitseeSteps } from "./gitsee/seed.js";
import { seedJarvisSteps } from "./jarvis/seed.js";
import { seedSheetsSteps } from "./sheets/seed.js";
import { seedHarveySteps, seedHarveyWorkflows } from "./harvey/seed.js";
import { seedGaiaSteps, seedGaiaWorkflows } from "./gaia/seed.js";
import { seedArtifactSteps } from "./artifacts/seed.js";
import { buildHarveyServices, type HarveyServices } from "./harvey/service.js";
import { buildGaiaServices, type GaiaServices } from "./gaia/service.js";
import { buildGitseeServices, type GitseeServices } from "./gitsee/services/index.js";

/**
 * Lets a step run other workflows (and read their params) from inside a run —
 * the `eval/optimize` loop uses this to eval/reflect across generations. A leaf
 * step has no runner of its own, so we hand it a closure over `vein.run`. Set
 * AFTER the vein is built (it closes over the instance); see `createLabVein`.
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
  /** Gitsee QA harness: per-run browser + stack session managers + vision judge,
   *  reached by the gitsee tool-steps via `ctx.services.gitsee.*`. */
  gitsee?: GitseeServices;
  /** Harvey LAB verification: subprocess-runs the REAL legal-benchmark eval
   *  from the pinned harvey-labs checkout (HARVEY_LABS_DIR). In-code on
   *  purpose — the grader must stay outside the agent-editable surface. */
  harvey?: HarveyServices;
  /** GAIA LAB scoring: runs the REAL leaderboard scorer.py (python3
   *  subprocess) against the validation gold in the pinned dataset checkout
   *  (GAIA_DIR). In-code on purpose — the grader and the gold must stay
   *  outside the agent-editable surface; gaia/* steps are thin plumbing. */
  gaia?: GaiaServices;
  /** Generic per-run teardown hook called by the vein runner in a `finally`
   *  (success AND error). Disposes a run's gitsee browser + booted stack. */
  onRunEnd?(runId: string): Promise<void>;
}

export interface CreateLabVeinOptions extends BuildServicesOptions {
  /** Pre-built merged services bag. If omitted, built from env. */
  services?: LabServices;
  /** Workspace dir for all lab workflows/runs. Defaults to
   *  `VEIN_LAB_WORKSPACE` or `./lab-workspace`. */
  workspacePath?: string;
  /** Serve the vein web UI (true when run standalone on its own port;
   *  the Express `/lab` mount passes false). Defaults to true. */
  serveUi?: boolean;
}

/**
 * Build THE single lab vein instance: one workspace, one UI, one merged
 * services bag. Experiments are just groups of workflows/steps inside it —
 * not separate servers.
 *
 * Steps and workflows are seeded into the workspace as content-hash–versioned
 * artifacts and discovered from disk (no in-code registry injection), so they
 * are editable + versioned through the vein API/UI.
 *
 * Adding an experiment = seed its step + workflow templates here and merge its
 * services into the bag.
 */
export async function createLabVein(
  opts: CreateLabVeinOptions = {},
): Promise<Vein<LabServices>> {
  // Merged services bag. Today concepts + the gitsee QA harness; spread
  // additional experiments' bags here as they're added.
  const services: LabServices =
    opts.services ?? (await buildConceptServices(opts));

  // Gitsee harness: per-run browser + stack managers + vision judge, plus the
  // generic per-run teardown hook. Wired even when `opts.services` was supplied
  // (so a custom bag still gets the gitsee harness + teardown) — only added when
  // absent, to respect a caller that intentionally provided its own gitsee bag.
  if (!services.gitsee) {
    const { gitsee, disposeRun } = buildGitseeServices();
    services.gitsee = gitsee;
    const priorOnRunEnd = services.onRunEnd?.bind(services);
    services.onRunEnd = async (runId: string) => {
      if (priorOnRunEnd) await priorOnRunEnd(runId);
      await disposeRun(runId);
    };
  }

  // Harvey LAB grader — in-code, NOT seeded (see harvey/service.ts). Only
  // added when absent, to respect a caller-provided bag. Construction is
  // cheap; HARVEY_LABS_DIR is checked at call time (loud per-run error).
  if (!services.harvey) {
    services.harvey = buildHarveyServices();
  }

  // GAIA LAB grader — same in-code, NOT-seeded discipline (see
  // gaia/service.ts). GAIA_DIR is checked at call time.
  if (!services.gaia) {
    services.gaia = buildGaiaServices();
  }

  const workspacePath =
    opts.workspacePath ??
    process.env["VEIN_LAB_WORKSPACE"] ??
    "./lab-workspace";

  // Seed each experiment's workflow + step templates into the workspace
  // BEFORE building the vein, so the registry's disk discovery picks up the
  // seeded steps. Steps are now self-contained custom steps on disk (not
  // injected in-code) — content-hash reconciled, editable + versioned via the
  // vein API/UI. No `registry` is passed, so createVein discovers core + lib +
  // these custom steps from `workspace.path` (and step publishing is enabled).
  const workspace = new WorkspaceManager(workspacePath);
  // Generic, domain-agnostic eval primitives (steps). The concept-specific eval
  // WORKFLOWS that wire them (concepts-eval*) are seeded with the concepts
  // experiment below.
  await seedEvalSteps(workspace);
  await seedConceptWorkflows(workspace);
  await seedConceptSteps(workspace);
  // gitsee experiment: self-contained steps (no services bag needed).
  await seedGitseeWorkflows(workspace);
  await seedGitseeSteps(workspace);
  // jarvis knowledge-graph steps (self-contained; reach Jarvis over
  // ctx.services.http with JARVIS_URL/API_TOKEN from ctx.services.secrets).
  // Grantable to agents via agentTools: ["jarvis/*"].
  await seedJarvisSteps(workspace);
  // google sheets steps (self-contained; reach the Sheets/Drive REST APIs
  // over ctx.services.http with GOOGLE_SERVICE_ACCOUNT_JSON /
  // GOOGLE_DRIVE_FOLDER_ID from ctx.services.secrets). Grantable to agents
  // via agentTools: ["sheets/*"].
  await seedSheetsSteps(workspace);
  // harvey verification steps (thin plumbing over services.harvey; grant
  // harvey/evaluate only to harness workflows, never the producing agent).
  await seedHarveySteps(workspace);
  await seedHarveyWorkflows(workspace);
  // gaia harness (thin plumbing over services.gaia; grant gaia/evaluate —
  // and every gaia/* — only to harness workflows, never the producing agent).
  await seedGaiaSteps(workspace);
  await seedGaiaWorkflows(workspace);
  // generic artifact plumbing (artifacts/dir — bridge runId → path for cwd).
  await seedArtifactSteps(workspace);

  const vein = await createVein<LabServices>({
    workspace,
    services,
    serveUi: opts.serveUi ?? true,
  });

  // Inject the run-sub-workflows capability now that the instance exists.
  // CRITICAL: mutate `vein.services` — the EFFECTIVE bag createVein built by
  // spreading our `services` into a fresh object (standardServices +
  // artifacts + ours) — NOT the local `services`, which runs never see
  // again. Mutating the local bag here silently broke every consumer of
  // `services.optimizer` (eval/optimize, eval/evolve-loop): steps threw
  // "requires a services.optimizer capability" at run time. This is what
  // lets the optimize/evolve loops run sub-workflows.
  const optimizer: LabServices["optimizer"] = {
    run: (name, input, runOpts) => vein.run(name, input, runOpts),
    getParams: async (name) => (await vein.workspace.getWorkflow(name)).params ?? {},
  };
  (vein.services as LabServices).optimizer = optimizer;
  services.optimizer = optimizer; // keep the caller's bag consistent too

  return vein;
}
