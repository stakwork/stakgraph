import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { JobSeedOptions } from "strut";
import type { ConceptSet } from "../concept-seed.js";

/**
 * job — the job agent as a strut workflow (strut `plans/jobs.md`). A host
 * (hive's Jamie chat) launches it with a `job` id on the launch and a
 * prompt: `job/dir` hands the agent ONE directory for the life of the job,
 * `session: "{{ input.session || $job }}"` continues one thread across every
 * turn, the agent writes its deliverables as files there and names them in
 * its structured output, and strut's `run.end` callback carries them
 * resolved to links (`/jobs/<job>/files/<path>`).
 *
 * The WORKFLOW is strut's — `src/workflows/job.yaml`, seeded at boot by
 * `createStrut` (content-hash reconciled, like SEED_OPTS; a changed template
 * wins, an unchanged one leaves a workspace-side edit active). Since
 * 2026-10-13 this module ships only the lab's DELTA on it, passed as
 * `createStrut({ job: JOB_SEED })` — the twin of `chatSystem`: grants
 * appended to `params.tools`, a section appended to the agent's system
 * prompt after strut's. Exactly one seeder per name: publishing `job` from
 * here as well would leave whichever template changed last active.
 *
 *  - `tools`: the `pod/*` steps (pods/seed.ts) — hive sandboxes, for a
 *    change that must run to be judged; a claimed pod is a hold on the job,
 *    released with it, its frontend the hold's url — and `browser/verify`
 *    (browser/seed.ts), the page instrument a claim's check names (strut
 *    plans/job-claims.md §4), a tool too, so the agent sees what the check
 *    will see; `browser/walkthrough`, the same with a frame per step, for
 *    a feature that is a flow (strut plans/live-artifacts.md §4). Strut's own list (graph reads, the meta/* read + run tools,
 *    the claim tools + meta/verify-job + http/verify, strut/run-workflow,
 *    the job/* reads) is already in the YAML.
 *  - `system`: open the `Job` Concept first (JOB_CONCEPTS, below) — WHICH
 *    workflow or way does which kind of work is graph data, read every turn,
 *    never prompt text: `Code Change` names `code-change-pr` and its child
 *    `Pod` says when a sandbox is warranted, `Plan Mode` says how to plan.
 */
export const JOB_SEED: JobSeedOptions = {
  tools: ["pod/*", "browser/verify", "browser/walkthrough"],
  system: `Before you act on a message, open the \`Job\` Concept: graph/graph-get
with node_type "Concept", name "Job", children "PARENT_OF". Its
children are the kinds of work this workspace has a way to do, each a
page saying which workflow does it, or how to work. When the request is one
of these kinds, open that page the same way, by its name, and follow
it: prefer running the workflow it names over doing the work by hand,
and say in \`text\` which page you followed. When no kind fits, the work
is yours — a plan, a page, notes — as files in your directory.`,
};

const HERE = dirname(fileURLToPath(import.meta.url));
/** The `Job` page the job agent opens every turn, and under it `Plan Mode`, `Explore`, `Verify` and `Captcha`; `Code Change` (CODE_CONCEPTS) is a child too. */
export const JOB_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/job/concepts/" };
