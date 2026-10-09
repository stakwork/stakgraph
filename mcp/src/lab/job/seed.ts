import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import type { ConceptSet } from "../concept-seed.js";
import { SEED_OPTS, retireWorkflows } from "../seed-opts.js";

/**
 * job — the job agent as a strut workflow (strut `plans/jobs.md`, Jobs V1).
 * A host (hive's Jamie chat) launches it with a `job` id on the launch and
 * a prompt: `job/dir` hands the agent ONE directory for the life of the
 * job, `session: "{{ input.session || $job }}"` continues one thread across
 * every turn (the job's, unless the launch names another), the
 * agent writes its deliverables as files there and names them in its
 * structured output, and strut's `run.end` callback carries them resolved
 * to links (`/jobs/<job>/files/<path>`). Launched WITHOUT a job (strut's
 * Run button) the same YAML is a one-shot in the run's artifact directory.
 *
 * `params` are the evolvable surface — `system`, `model`, `maxSteps`,
 * `tools` (registry steps granted on top of the agent's built-ins — files,
 * bash, web_search, web_fetch: graph reads, the meta/* read + run tools,
 * so a turn can run any workflow as a child run under the job, and the
 * pod/* tools, a hive sandbox the job holds between turns — WHICH one
 * for which kind of work is the `Job` Concept tree's (JOB_CONCEPTS),
 * read every turn, never this prompt's; authoring is a later line) — so a job learns to do something new by a new version
 * of this workflow on the swarm, never by a change on the host. NO custom
 * steps: every step is strut's, so this seeder ships YAML only. Seeded
 * UNSTAMPED (no publisher → not "ai"), content-hash reconciled (SEED_OPTS):
 * a changed committed copy wins at boot, an unchanged one leaves a
 * workspace-side edit active.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "job",
    description:
      "A job turn: an agent working in the job's directory with a thread that remembers every earlier turn (job/dir → agent with session: {{ input.session || $job }} → pack). Launch it with `job` on the launch — POST /workflows/job/run { job, input: { prompt }, callback } — and the same job again to revise the same files; without a job it is a one-shot in the run's artifact directory. Input: { prompt, workspace? (the slug of the hive workspace this strut belongs to — the agent's message starts with it; another workspace of the org is a peer, @<slug>), session? (the agent's thread: the job id unless given — pass one to work on the job's files with a fresh thread, or a second beside the first), repos? (repository URLs checked out into the job's directory, one subdirectory each, kept turn to turn) }. Output: { text, artifacts: [{ id, kind?, title, label?, summary?, path | url | content }], ask?: { message }, cost, session }; the callback carries `artifacts` resolved to links. params.tools (graph reads; meta/list-workflows, get-workflow, run-workflow, get-run; strut/run-workflow — a workflow on a peer, another workspace's strut by its slug, e.g. `explore` there; job/list, get, read — earlier jobs by words, what one produced, one of its files; pod/* — a kind's page under the `Job` Concept says how to do it, e.g. Explore → run `explore` here or on a peer and keep what it found as a note in the job's directory; Code Change → run `code-change-pr` as a child run under the job, briefed from the notes, or its child Pod → claim a hive sandbox for the job and let the agent in it make the change; Plan Mode → a plan.html to approve first) and params.system are the evolvable surface.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
/** The `Job` page the job agent opens every turn, and under it `Plan Mode`; `Code Change` (CODE_CONCEPTS) is a child too. */
export const JOB_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/job/concepts/" };

export async function seedJobWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, "job", undefined, SEED_OPTS);
      if (changed) console.log(`[job] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[job] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "job");
}
