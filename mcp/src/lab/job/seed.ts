import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { ConceptSet, JobSeedOptions } from "strut";

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
 *  - `system`: what this deployment adds to strut's prompt (which already
 *    opens the `Job` Concept — strut's page — before acting; WHICH workflow
 *    or way does which kind of work is its children, graph data, never
 *    prompt text: `Code Change` names `code-change-pr` and its child `Pod`
 *    says when a sandbox is warranted, `Plan Mode` says how to plan). Here:
 *    that this strut belongs to one Hive workspace and every other is a
 *    peer, and that looking at a page is `browser-explore`, never a browser
 *    of the agent's own — both were the `Job` page's until strut shipped a
 *    generic one. The second moves into strut's `job.yaml` prompt with
 *    `browser/*`: drop it from here in the strut bump that carries it, or
 *    lab swarms read it twice.
 */
export const JOB_SEED: JobSeedOptions = {
  tools: ["pod/*", "browser/verify", "browser/walkthrough"],
  system: `This strut belongs to one Hive workspace: your message's first line names
it, \`@<slug>\`. Every other workspace of the org is a peer, with its own
strut, graph and repositories, named by its slug: \`@<slug>\`.
\`strut/run-workflow { peer: "<slug>", workflow, input }\` runs a workflow
there and returns its output; its files stay there. When a request names a
workspace, that is where its code lives.

Seeing what a page shows is a workflow too. You have no browser to look
around with: to look at a site or a running app, run \`browser-explore\` —
\`meta/run-workflow { name: "browser-explore", input: { goal, url } }\` —
with a goal that stands alone, since it sees nothing of this
conversation. It drives the deployment's browser, looks for itself and
answers \`{ answer, evidence: [{ path, url }], pages }\`; on a job its
screenshots land in your directory as \`shots/NNN.png\`, and each \`url\` is
a link to report as an image artifact. Never install or launch a browser
yourself (playwright, puppeteer, a headless chrome from bash): what it
captures is a file you cannot see, and it runs on the host, not in the
browser set aside for this.`,
};

const HERE = dirname(fileURLToPath(import.meta.url));
/** The pages under strut's `Job` page: `Explore`, `Onboarding`, `Plan Mode` and `Verify`; `Code Change` (CODE_CONCEPTS) is a child too. */
export const JOB_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/job/concepts/" };
