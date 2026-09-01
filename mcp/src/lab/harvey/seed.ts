import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Harvey LAB verification steps — THIN plumbing only. The actual grader is
 * the in-code `harvey` service (`service.ts`), which subprocess-runs the real
 * eval from the harvey-labs checkout; editing these seeded steps can only
 * break plumbing, never change how grading works. That split is deliberate:
 * the benchmark verifier must stay outside the agent-editable surface.
 *
 * - `harvey/get-task` — instructions + documents, rubric stripped. Safe for
 *   producing agents.
 * - `harvey/evaluate` — stage deliverables + run the real eval. Grant ONLY
 *   to harness workflows, never to the producing agent.
 * - `harvey/pack-result` — echo combiner (assemble workflow outputs; onError
 *   fallbacks).
 * - `harvey/digest-results` — aggregate graded results into the propose
 *   digest (verdict channel only; see the step header).
 *
 * The hill-climb loop itself is the GENERIC `eval/evolve-loop` (seeded by
 * eval/seed.ts) — harvey-evolve wires it with harvey's gen workflow and
 * pass-rate fitness.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "get-task.ts", type: "harvey/get-task" },
  { file: "evaluate.ts", type: "harvey/evaluate" },
  { file: "pack-result.ts", type: "harvey/pack-result" },
  { file: "digest-results.ts", type: "harvey/digest-results" },
  // harvey-deliver pipeline steps (standalone production-style pipeline —
  // rubric as input; NOT part of the benchmark harness): intake, drafting
  // plan, scoring plumbing, and the pinned read-only graph sub-agent.
  { file: "normalize-documents.ts", type: "harvey/normalize-documents" },
  { file: "graph-sub-agent.ts", type: "harvey/graph-sub-agent" },
  { file: "ingest-state.ts", type: "harvey/ingest-state" },
  { file: "drafter-plan.ts", type: "harvey/drafter-plan" },
  { file: "validate-deliverables.ts", type: "harvey/validate-deliverables" },
  { file: "filter-contested.ts", type: "harvey/filter-contested" },
  { file: "aggregate-scores.ts", type: "harvey/aggregate-scores" },
  { file: "merge-disputes.ts", type: "harvey/merge-disputes" },
  { file: "build-eval-chain.ts", type: "harvey/build-eval-chain" },
  { file: "criterion-refs.ts", type: "harvey/criterion-refs" },
  // deliverable generation (pandoc / openpyxl) — grantable agent tools so the
  // production prompts' harvey_generate_docx/_xlsx calls work verbatim.
  { file: "generate-docx.ts", type: "harvey/generate-docx" },
  { file: "generate-xlsx.ts", type: "harvey/generate-xlsx" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

// Order matters only for readability — all four are plain publishes. The
// produce/grade split (EVOLVE_SPEC §5): harvey-produce is the swappable
// candidate unit, harvey-run/harvey-candidate-run are the grading harnesses,
// harvey-evolve is the authoring harness over all of them. All seeded
// UNSTAMPED (no publisher arg → not "ai"), so the meta surface can read but
// never edit, run, or overwrite them.
const SEED_WORKFLOWS = [
  "harvey-produce",
  "harvey-run",
  "harvey-candidate-run",
  "harvey-evolve-gen",
  "harvey-evolve",
  // The deliver pipeline (standalone; rubric as input — never a harvey-run
  // candidate). Sub-workflows first for readability; all plain publishes.
  "harvey-ingest-doc",
  "harvey-knowledge",
  "harvey-draft",
  "harvey-judge-criterion",
  "harvey-dispute-criterion",
  "harvey-score",
  "harvey-deliver",
];

/**
 * Expand `@@include(FILE.md)` marker lines with the contents of
 * `prompts/FILE.md`, indented to the marker's own indentation — so a marker
 * inside a YAML literal block (`prompt: |`) splices a multi-KB prompt body in
 * as valid YAML. Keeps the big deliver-pipeline prompts as clean, diffable
 * markdown files instead of 40KB YAML scalars; publishWorkflowByContent
 * hashes the EXPANDED yaml, so editing a prompt file re-seeds its workflows.
 * Unknown includes throw (a silently-missing prompt would seed a broken
 * workflow).
 */
/**
 * The prompt files are VERBATIM copies of the stakwork production prompts,
 * which name the repo-agent's jarvis tools (`jarvis_graph_search`, …). The
 * lab pipeline runs on the vein-native `graph/*` steps (same shapes, backed
 * by vein's own Neo4j graph backend — no jarvis process), whose agent tool
 * names are `graph_*`. Translate the tool names at seed time so the prompt
 * files stay byte-identical to production while the agents call the tools
 * they were actually granted.
 */
const JARVIS_TOOL_NAME =
  /\bjarvis_(get_ontology_type|get_ontology|graph_search|graph_get_batched|graph_get|graph_neighbors|register_namespace|create_node|edit_node|create_triplet|create_batch_triplet)\b/g;
export function translateToolNames(text: string): string {
  return text.replace(JARVIS_TOOL_NAME, "graph_$1");
}

async function expandIncludes(yaml: string): Promise<string> {
  const promptsDir = join(HERE, "prompts");
  const lines = yaml.split("\n");
  const out: string[] = [];
  for (const line of lines) {
    const m = line.match(/^([ \t]*)@@include\(([^)]+)\)\s*$/);
    if (!m) {
      out.push(line);
      continue;
    }
    const [, indent, file] = m;
    const body = translateToolNames(await readFile(join(promptsDir, file), "utf-8"));
    for (const bodyLine of body.replace(/\n$/, "").split("\n")) {
      out.push(bodyLine.length > 0 ? indent + bodyLine : "");
    }
  }
  return out.join("\n");
}

export async function seedHarveyWorkflows(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await expandIncludes(await readFile(join(dir, `${name}.yaml`), "utf-8"));
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, "harvey-seed", "harvey");
      if (changed) console.log(`[harvey] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[harvey] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
}

export async function seedHarveySteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "harvey-seed");
      if (changed) console.log(`[harvey] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[harvey] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
