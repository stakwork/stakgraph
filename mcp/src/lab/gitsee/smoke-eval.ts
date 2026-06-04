import "dotenv/config";
import { join } from "node:path";
import { createLabVein } from "../createLabVein.js";

/**
 * Throwaway integration test for the gitsee EVAL harness: builds a real lab vein
 * (seeds the gitsee workflows/steps) and runs `gitsee-eval` end-to-end —
 * produce (clone workspace + explore via gitsee-explore-services) → score
 * (eval/score with the gold-files rubric). Validates subflow wiring + scoring,
 * not just the leaf steps. Needs Neo4j up + ANTHROPIC_API_KEY + GITHUB_TOKEN.
 *
 * Runs the first matching workspace from gitsee-optimize's dataset (its gold),
 * selected by label (default "heroku-node"):
 *   npx tsx src/lab/gitsee/smoke-eval.ts [workspace-label]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import yaml from "js-yaml";

const label = process.argv[2] || "heroku-node";

async function main() {
  // The canonical lab workspace (gitignored, inside the mcp tree so the seeded
  // steps' bare `import "vein"` resolves against mcp/node_modules — the
  // AGENTS.md workspace gotcha).
  const workspacePath = join(process.cwd(), "lab-workspace");
  console.log("workspace:", workspacePath);
  const vein = await createLabVein({ serveUi: false, workspacePath });

  // Pull the workspace entry (repos + gold) from the optimize dataset.
  const here = dirname(fileURLToPath(import.meta.url));
  const opt = yaml.load(
    readFileSync(join(here, "workflows", "gitsee-optimize.yaml"), "utf-8"),
  ) as { params: { dataset: Array<{ label: string; repos: any[]; expected: string }> } };
  const entry = opt.params.dataset.find((d) => d.label === label);
  if (!entry) throw new Error(`no dataset entry with label "${label}"`);

  console.log(`\n=== run gitsee-eval on workspace "${label}" (${entry.repos.length} repo(s)) ===`);
  const res = await vein.run("gitsee-eval", {
    label: entry.label,
    repos: entry.repos,
    expected: entry.expected,
    token: process.env.GITHUB_TOKEN,
  });

  console.log("\nstatus:", res.status);
  if (res.status !== "success") {
    console.error("error:", JSON.stringify(res.error, null, 2));
    process.exit(1);
  }
  const o = res.output as any;
  console.log(`score: ${o.score} (recall ${o.recall}, precision ${o.precision})`);
  console.log("missing:", o.missing);
  console.log("spurious:", o.spurious);
  console.log("\nmarkdown:\n" + o.markdown);
  process.exit(0);
}

main().catch((e) => {
  console.error("SMOKE-EVAL FAILED:", e);
  process.exit(1);
});
