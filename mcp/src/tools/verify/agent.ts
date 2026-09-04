import { ToolLoopAgent, stepCountIs, tool, jsonSchema, StopCondition } from "ai";
import { getModelDetails, getProviderOptions } from "../../aieo/src/provider.js";
import * as stagehand from "../stagehand/tools.js";
import * as verify from "./index.js";
import { selectConcept, conceptHint } from "./concepts.js";

const SYSTEM = `You are verifying whether a specific TASK actually works in a running application. You did not write this code; you audit it. You never modify anything.

Use the tools available to exercise the running app and gather evidence, then call submit_verdict to end.

SCOPE: Judge ONLY what THIS task claims. Do not invent or verify additional claims the task did not make — e.g. do not check database persistence for a task that is only about a network call. Extra observations belong in observations[], never as claims that can fail the task.

DISCIPLINE (the core):
- Mark a claim "works" ONLY if you CAPTURED proof for it and cite the evidence id (evN) a probe tool returned, in that claim's proof[]. A works claim with no captured evidence id is rejected.
- "It looks right" or "the UI said success" is NOT proof — confirm the underlying behavior (the actual network status, the console, the persisted data).
- If you cannot tell, use unknown. If it is broken, say why. Be honest; an unjustified "works" is the worst outcome.
- submit_verdict is the only way to finish.`;

export interface VerifyInput {
  taskId: string;
  taskPrompt: string;
  diff: string;
  appUrl: string;
  notes?: string | null;
  model: string;
  apiKey: string;
  sessionId: string;
  maxTurns?: number;
}

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

export async function runVerification(input: VerifyInput): Promise<any> {
  const started = Date.now();
  const concept = selectConcept(input.taskPrompt, input.diff || "");
  const { model, modelId, provider } = getModelDetails(input.model, input.apiKey, undefined, undefined, undefined, 300000);

  const agent = new ToolLoopAgent<never, any>({
    model,
    instructions: SYSTEM,
    tools: buildTools(input.sessionId),
    stopWhen: [stopAfterVerdict, stepCountIs(input.maxTurns ?? 50) as StopCondition<any>],
    providerOptions: getProviderOptions(provider as any, undefined, modelId) as any,
    maxOutputTokens: 64000,
  });

  const diffBlock = input.diff ? `\n\nThe diff of the change:\n${input.diff.slice(0, 6000)}` : "";
  const notesBlock = input.notes ? `\nNotes: ${input.notes}` : "";
  const userPrompt =
    `Task to verify: ${input.taskPrompt}\n\n` +
    `The application is running at ${input.appUrl}.${notesBlock}${diffBlock}\n\n` +
    `${conceptHint(concept)}\n\n` +
    `Verify whether the task actually works, capturing evidence, then call submit_verdict.`;

  console.error(`[verify] taskId=${input.taskId} concept=${concept.id} model=${modelId}`);
  try {
    await agent.generate({ prompt: userPrompt });
  } catch (err: any) {
    console.error(`[verify] taskId=${input.taskId} generate error: ${err?.message ?? String(err)}`);
  }

  const verdict: any = verify.getVerdict(input.sessionId) || {
    overall: "unknown",
    claims: [],
    observations: ["agent ended without submit_verdict"],
    summary: "no verdict",
    evidence: [],
  };
  verdict.taskId = input.taskId;
  verdict.startedAt = new Date(started).toISOString();
  verdict.finishedAt = new Date().toISOString();
  verify.resetVerifySession(input.sessionId);
  return verdict;
}
