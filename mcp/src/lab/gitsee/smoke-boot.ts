import "dotenv/config";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import yaml from "js-yaml";
import { coreRegistry } from "vein";
import cloneStep from "./steps/clone-workspace.js";
import bootExerciseStep from "./steps/boot-and-exercise.js";

/**
 * Throwaway end-to-end smoke for the boot-and-exercise loop. Clones a workspace,
 * runs the core `agent` to PRODUCE an initial setup (reusing the exact
 * gitsee-explore-services prompts), then runs `gitsee/boot-and-exercise` to BOOT
 * + DRIVE + FIX it until the frontend works. No vein server / Neo4j / seeding.
 *
 *   npx tsx src/lab/gitsee/smoke-boot.ts [workspace] [owner/repo ...]
 *
 * Needs ANTHROPIC_API_KEY + git + docker on PATH and `npx playwright install
 * chromium`. Set KEEP_UP=1 to leave the stack running for inspection.
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
  console.log(`\n=== clone "${workspace}": ${repos.map((r) => r.owner + "/" + r.repo).join(", ")} ===`);
  const cloneCfg = (cloneStep.input as any).parse({ workspace, repos, token: process.env.GITHUB_TOKEN });
  const { workspacePath } = (await cloneStep.run(cloneCfg, ctx)) as { workspacePath: string };
  console.log("workspacePath:", workspacePath);

  console.log(`\n=== produce initial setup via core agent (model ${p.model}) ===`);
  const agentStep = coreRegistry()["agent"];
  const exploreCfg = (agentStep.input as any).parse({
    cwd: workspacePath,
    prompt: p.prompt,
    system: p.system,
    finalAnswer: p.finalAnswer,
    model: p.model,
  });
  const produced = (await agentStep.run(exploreCfg, ctx)) as { result: string };
  console.log(produced.result);

  console.log(`\n=== boot-and-exercise ===`);
  const beCfg = (bootExerciseStep.input as any).parse({
    workspacePath,
    setup: produced.result,
    keepUp: !!process.env.KEEP_UP,
  });
  const out = (await bootExerciseStep.run(beCfg, ctx)) as Record<string, unknown>;

  console.log(`\n=== RESULT (booted=${out.booted} working=${out.working} iterations=${out.iterations} $${out.cost}) ===\n`);
  console.log(out.report);
  console.log(`\n=== FINAL SETUP ===\n${out.setup}`);
  if (out.changed) console.log(`\n=== REPO DIFF (${(out.changedRepos as string[]).join(", ")}) ===\n${out.diff}`);
  if (out.screenshotPath) console.log(`\nscreenshot: ${out.screenshotPath}`);
}

main().catch((e) => {
  console.error("SMOKE FAILED:", e);
  process.exit(1);
});
