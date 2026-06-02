import {
  createVein,
  WorkspaceManager,
  type Vein,
} from "vein";
import {
  buildConceptServices,
  type ConceptServices,
  type BuildServicesOptions,
} from "./concepts/services.js";
import { seedConceptWorkflows, seedConceptSteps } from "./concepts/seed.js";

/**
 * The merged capabilities bag for ALL lab experiments. Each experiment's
 * steps cast `ctx.services` to whatever subset they need (services is
 * untyped at runtime), so a single merged bag serves every experiment.
 * Extend this as experiments are added.
 */
export interface LabServices extends ConceptServices {}

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
  // Merged services bag. Today just concepts; spread additional
  // experiments' bags here as they're added.
  const services: LabServices =
    opts.services ?? (await buildConceptServices(opts));

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
  await seedConceptWorkflows(workspace);
  await seedConceptSteps(workspace);

  const vein = await createVein<LabServices>({
    workspace,
    services,
    serveUi: opts.serveUi ?? true,
  });

  return vein;
}
