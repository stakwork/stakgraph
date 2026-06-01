import {
  createVein,
  createRegistry,
  WorkspaceManager,
  type Vein,
} from "vein";
import { conceptSteps } from "./concepts/steps/index.js";
import {
  buildConceptServices,
  type ConceptServices,
  type BuildServicesOptions,
} from "./concepts/services.js";
import { seedConceptWorkflows } from "./concepts/seed.js";

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
 * Build THE single lab vein instance: one workspace, one UI, one
 * registry composed of every experiment's steps, one merged services
 * bag. Experiments are just groups of workflows/steps inside it — not
 * separate servers.
 *
 * Adding an experiment = register its steps in the registry, merge its
 * services, and seed its workflow templates here.
 */
export async function createLabVein(
  opts: CreateLabVeinOptions = {},
): Promise<Vein<LabServices>> {
  // Merged services bag. Today just concepts; spread additional
  // experiments' bags here as they're added.
  const services: LabServices =
    opts.services ?? (await buildConceptServices(opts));

  // Registry = vein core + lib + every experiment's steps.
  const registry = await createRegistry([...conceptSteps]);

  const workspacePath =
    opts.workspacePath ??
    process.env["VEIN_LAB_WORKSPACE"] ??
    "./lab-workspace";

  const vein = await createVein<LabServices>({
    workspace: new WorkspaceManager(workspacePath),
    registry,
    services,
    serveUi: opts.serveUi ?? true,
  });

  // Seed each experiment's workflow templates (idempotent).
  await seedConceptWorkflows(vein.workspace);

  return vein;
}
