import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS, retireWorkflows } from "../seed-opts.js";

/**
 * calls — what Hive runs on this swarm about its calls. `call-transcript`
 * gets a call's whole transcript once the call ends (sphinx-voice
 * plans/call-transcript.md): Hive's POST /api/calls/transcript launches it
 * when CALL_TRANSCRIPT_STRUT_WORKFLOW_NAME names it, with the transcript as
 * the run's attachment. For now a placeholder that returns what arrived, so
 * every swarm has the workflow Hive names and a call's transcript can be
 * checked on the run. NO custom steps: YAML only. Seeded UNSTAMPED,
 * content-hash reconciled (SEED_OPTS), like code.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "call-transcript",
    description:
      "A Hive call's whole transcript, sent once when the call ends, as the run's attachment attachments/call-transcript.jsonl (one JSON object per line: n, at, speaker, identity?, source voice | chat | jamie, text). Launched by Hive's POST /api/calls/transcript as the person who started the call. A placeholder: it returns what arrived. Input: { room, workspace, conversation, startedAt, endedAt, lines, speakers, file }. Output: { file, room, workspace, conversation, startedAt, endedAt, lines, speakers }.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedCallsWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, "calls", undefined, SEED_OPTS);
      if (changed) console.log(`[calls] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[calls] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "calls");
}
