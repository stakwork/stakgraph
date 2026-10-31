import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS, retireWorkflows } from "../seed-opts.js";

/**
 * tribes — what Hive runs on this swarm about the Sphinx tribes linked to
 * its org. `tribe-messages` gets what was said in one, a batch at a time
 * (stakwork/hive docs/plans/jamie-in-tribes.md, "The log"): Hive's POST
 * /api/tribes/log launches it when TRIBE_MESSAGES_STRUT_WORKFLOW_NAME names
 * it, with the batch as the run's attachment. For now a placeholder that
 * returns what arrived, so every swarm has the workflow Hive names and a
 * tribe's messages can be checked on the run. NO custom steps: YAML only.
 * Seeded UNSTAMPED, content-hash reconciled (SEED_OPTS), like code.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "tribe-messages",
    description:
      "A batch of messages from a Sphinx tribe linked to this org, as the run's attachment attachments/tribe-messages.jsonl (one JSON object per line, oldest first: id, time, alias, owner, mine, text, replyTo?, thread?). Launched by Hive's POST /api/tribes/log as the person who linked the tribe. A placeholder: it returns what arrived. Input: { tribe, workspace, from, to, messages, aliases, file }. Output: { file, tribe, workspace, from, to, messages, aliases }.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedTribesWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, "tribes", undefined, SEED_OPTS);
      if (changed) console.log(`[tribes] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[tribes] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "tribes");
}
