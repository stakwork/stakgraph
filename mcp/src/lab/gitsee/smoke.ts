import "dotenv/config";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import yaml from "js-yaml";
import { coreRegistry } from "vein";
import cloneStep from "./steps/clone-workspace.js";

/**
 * Throwaway end-to-end smoke test for the gitsee experiment — runs the gitsee
 * clone step + the vein-core `agent` step directly (real git clone + real
 * Anthropic call), feeding the exact `params` from gitsee-explore-services.yaml.
 * No vein server / Neo4j / seeding needed.
 *
 * Clones a WORKSPACE (one or more repos as siblings). Pass repos as
 * "owner/repo" args (defaults to a single heroku/node-js-getting-started):
 *   npx tsx src/lab/gitsee/smoke.ts [workspace] [owner/repo ...]
 */
const HERE = dirname(fileURLToPath(import.meta.url));
const workspace = process.argv[2] || "heroku-node";
const repoArgs = process.argv.slice(3);
const repos = (repoArgs.length ? repoArgs : ["heroku/node-js-getting-started"]).map((s) => {
  const [owner, repo] = s.split("/");
  return { owner, repo };
});

const wf = yaml.load(
  readFileSync(join(HERE, "workflows", "gitsee-explore-services.yaml"), "utf-8"),
) as { params: Record<string, any> };
const p = wf.params;

const ctx = { runId: "smoke", path: "smoke", services: {}, emit: () => {} } as any;

async function main() {
  console.log(`\n=== clone workspace "${workspace}": ${repos.map((r) => r.owner + "/" + r.repo).join(", ")} ===`);
  const cloneCfg = (cloneStep.input as any).parse({ workspace, repos, token: process.env.GITHUB_TOKEN });
  const { workspacePath } = (await cloneStep.run(cloneCfg, ctx)) as { workspacePath: string };
  console.log("workspacePath:", workspacePath);

  console.log(`\n=== explore via core agent (model ${p.model}) ===`);
  const agentStep = coreRegistry()["agent"];
  const exploreCfg = (agentStep.input as any).parse({
    cwd: workspacePath,
    prompt: p.prompt,
    system: p.system,
    finalAnswer: p.finalAnswer,
    fileLines: p.fileLines,
    model: p.model,
  });
  const out = (await agentStep.run(exploreCfg, ctx)) as { result: string; steps: number };

  console.log(`\n=== RESULT (${out.steps} steps) ===\n`);
  console.log(out.result);
}

main().catch((e) => {
  console.error("SMOKE FAILED:", e);
  process.exit(1);
});
