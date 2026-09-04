import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { runVerification } from "./agent.js";

const SCRATCH = process.env.SCRATCH!;
const rep = process.env.REP || "1";

async function main() {
  const cell = JSON.parse(readFileSync(`${SCRATCH}/cells-tough.json`, "utf8"))[0];
  const verdict: any = await runVerification({
    taskId: cell.id,
    taskPrompt: cell.prompt,
    diff: cell.diff,
    appUrl: "N/A — this is a goose CLI change, not a web app. Use verify_run_command to run the goose CLI and prove the fix.",
    notes: null,
    model: process.env.AGENT_MODEL || "anthropic/claude-opus-5",
    apiKey: process.env.ANTHROPIC_API_KEY!,
    sessionId: `tough-${rep}-${Date.now()}`,
    maxTurns: 40,
  });
  mkdirSync(`${SCRATCH}/toughrep${rep}`, { recursive: true });
  writeFileSync(`${SCRATCH}/toughrep${rep}/${cell.id}.json`, JSON.stringify(verdict, null, 2));
  const kinds = [...new Set((verdict.evidence || []).map((e: any) => e.kind))].join(",");
  console.log(`RESULT rep${rep} overall=${verdict.overall} kinds=[${kinds}]`);
  process.exit(0);
}

main().catch((e) => {
  console.error("tough error:", e?.message ?? e);
  process.exit(1);
});
