// Spike D4/D5 runner: a GENERIC agent (opus-5, discipline-only prompt) that
// verifies a change using ONLY the shared MCP tools, routed through the exact
// same dispatch as server.ts (stagehand.call / verify.* / tagEvidence). No
// bespoke verification method is baked into the agent.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { ToolLoopAgent, stepCountIs, tool, jsonSchema, StopCondition } from "ai";
import { getModelDetails, getProviderOptions } from "../../aieo/src/provider.js";
import * as stagehand from "../stagehand/tools.js";
import * as verify from "./index.js";

const SCRATCH = process.env.SCRATCH!;
const APP = process.env.APP_URL || "http://localhost:3000";
const KEY = process.env.ANTHROPIC_API_KEY!;
const MODEL = process.env.AGENT_MODEL || "anthropic/claude-opus-5";
const CELL_ID = process.env.CELL!;

const SYSTEM = `You are verifying whether a specific TASK actually works in a running application. You did not write this code; you audit it. You never modify anything.

Use the tools available to exercise the running app and gather evidence, then call submit_verdict to end.

SCOPE: Judge ONLY what THIS task claims. Do not invent or verify additional claims the task did not make — e.g. do not check database persistence for a task that is only about a network call. Extra observations belong in observations[], never as claims that can fail the task.

DISCIPLINE (the core):
- Mark a claim "works" ONLY if you CAPTURED proof for it and cite the evidence id (evN) a probe tool returned, in that claim's proof[]. A works claim with no captured evidence id is rejected.
- "It looks right" or "the UI said success" is NOT proof — confirm the underlying behavior (the actual network status, the console, the persisted data).
- If you cannot tell, use unknown. If it is broken, say why. Be honest; an unjustified "works" is the worst outcome.
- submit_verdict is the only way to finish.`;

function toText(result: any): string {
  const items = (result?.content as Array<{ type: string; text?: string }>) || [];
  const parts: string[] = [];
  for (const it of items) {
    if (it.type === "text" && it.text) parts.push(it.text);
    else if (it.type === "image") parts.push("[image captured]");
  }
  return parts.join("\n") || "(no output)";
}

function buildTools(sessionId: string) {
  const defs = [...stagehand.TOOLS, ...verify.VERIFY_TOOLS];
  const out: Record<string, any> = {};
  for (const t of defs) {
    const name = t.name;
    out[name] = tool({
      description: t.description as string,
      inputSchema: jsonSchema(t.inputSchema as any),
      execute: async (args: any) => {
        let result: any;
        if (name.startsWith("stagehand_")) {
          result = verify.tagEvidence(sessionId, name, await stagehand.call(name, args || {}, sessionId));
        } else if (name === verify.HttpRequestTool.name) {
          result = await verify.httpRequest(sessionId, args || {});
        } else if (name === verify.SampleTool.name) {
          result = await verify.sampleUrl(sessionId, args || {});
        } else if (name === verify.DbQueryTool.name) {
          result = await verify.dbQuery(sessionId, args || {});
        } else if (name === verify.SubmitVerdictTool.name) {
          result = await verify.submitVerdict(sessionId, args || {});
        } else {
          result = { content: [{ type: "text", text: `unknown tool ${name}` }] };
        }
        return toText(result);
      },
    });
  }
  return out;
}

const stopAfterVerdict: StopCondition<any> = ({ steps }) => {
  for (const s of steps as any[])
    for (const item of s.content || [])
      if (item.type === "tool-call" && item.toolName === "submit_verdict") return true;
  return false;
};

async function main() {
  const cells = JSON.parse(readFileSync(`${SCRATCH}/${process.env.CELLS_FILE || "cells.json"}`, "utf8"));
  const cell = cells.find((c: any) => c.id === CELL_ID);
  if (!cell) throw new Error(`cell ${CELL_ID} not found`);

  const sessionId = `spike-${cell.id}-${Date.now()}`;
  const { model, modelId, provider } = getModelDetails(MODEL, KEY, undefined, undefined, undefined, 300000);

  const agent = new ToolLoopAgent<never, any>({
    model,
    instructions: SYSTEM,
    tools: buildTools(sessionId),
    stopWhen: [stopAfterVerdict, stepCountIs(50) as StopCondition<any>],
    providerOptions: getProviderOptions(provider as any, undefined, modelId) as any,
    maxOutputTokens: 64000,
  });

  const userPrompt =
    `Task to verify: ${cell.prompt}\n\n` +
    `The application is running at ${APP}. The change is exercised at ${APP}${cell.path}. ` +
    `Verify whether the task actually works, capturing evidence, then call submit_verdict.`;

  const started = Date.now();
  try {
    await agent.generate({ prompt: userPrompt });
  } catch (err: any) {
    console.error(`[spike] ${cell.id} generate error: ${err?.message ?? String(err)}`);
  }

  const verdict: any = verify.getVerdict(sessionId) || {
    overall: "unknown",
    claims: [],
    observations: ["agent ended without submit_verdict"],
    summary: "no verdict",
    evidence: [],
  };
  verdict.taskId = cell.id;
  verdict.startedAt = new Date(started).toISOString();
  verdict.finishedAt = new Date().toISOString();

  const outDir = process.env.OUT_DIR || "verdicts-goose";
  mkdirSync(`${SCRATCH}/${outDir}`, { recursive: true });
  writeFileSync(`${SCRATCH}/${outDir}/${cell.id}.json`, JSON.stringify(verdict, null, 2));
  const kinds = [...new Set((verdict.evidence || []).map((e: any) => e.kind))].join(",");
  const secs = ((Date.now() - started) / 1000).toFixed(0);
  console.log(`RESULT ${cell.id} overall=${verdict.overall} kinds=[${kinds}] (${secs}s)`);
  process.exit(0);
}

main().catch((e) => {
  console.error("[spike] fatal:", e?.message ?? e);
  process.exit(1);
});
