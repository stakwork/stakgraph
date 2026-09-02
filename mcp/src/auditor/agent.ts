import { ToolLoopAgent, StopCondition, stepCountIs } from "ai";
import { getModelDetails, getProviderOptions } from "../aieo/src/provider.js";
import { AuditBrowser } from "./browser.js";

function maxOutputTokensFor(provider?: string): number {
  const env = Number(process.env.MAX_OUTPUT_TOKENS);
  if (env > 0) return env;
  return provider === "anthropic" ? 128_000 : 64_000;
}
import { getAuditorTools, AuditorTools, END_OF_AUDIT } from "./tools.js";
import { AUDITOR_SYSTEM_PROMPT } from "./prompt.js";
import {
  AuditJob,
  EvidenceCollector,
  EvidenceKind,
  EvidenceRecord,
  Verdict,
} from "./types.js";

const DEFAULT_MAX_TURNS = 40;

const RUN_TIMEOUT_MS =
  parseInt(process.env.AUDIT_RUN_TIMEOUT_MS || "", 10) || 900_000;
const MODEL_TIMEOUT_MS =
  parseInt(process.env.AUDIT_MODEL_TIMEOUT_MS || "", 10) || 300_000;

function createCollector(): EvidenceCollector {
  const records: EvidenceRecord[] = [];
  return {
    records,
    verdict: undefined,
    push(kind: EvidenceKind, summary: string, data?: string): string {
      const id = `ev${records.length + 1}`;
      records.push({ id, kind, summary, data: data ?? "" });
      return id;
    },
  };
}

function hasEndMarker(): StopCondition<AuditorTools> {
  return ({ steps }) => {
    for (const step of steps) {
      for (const item of step.content) {
        if (item.type === "text" && item.text?.includes(END_OF_AUDIT)) {
          return true;
        }
      }
    }
    return false;
  };
}

export interface PreparedAuditor {
  run(): Promise<Verdict>;
}

export function prepareAuditor(
  job: AuditJob,
  maxTurns: number = DEFAULT_MAX_TURNS,
): PreparedAuditor {
  const startedAt = new Date().toISOString();
  const collector = createCollector();
  const logs: string[] = [];

  const browser = new AuditBrowser(job.model);

  const abortController = new AbortController();

  const modelName = job.model.model ?? job.model.provider;
  const { model, modelId, provider } = getModelDetails(
    modelName,
    job.model.apiKey,
    job.model.host,
    undefined,
    abortController.signal,
    MODEL_TIMEOUT_MS,
  );

  const tools = getAuditorTools({ deck: job.deck, collector, browser });

  const stopWhen: StopCondition<AuditorTools>[] = [
    hasEndMarker(),
    stepCountIs(maxTurns) as StopCondition<AuditorTools>,
  ];

  const agent = new ToolLoopAgent<never, AuditorTools>({
    model,
    instructions: AUDITOR_SYSTEM_PROMPT,
    tools,
    stopWhen,
    stopSequences: [END_OF_AUDIT],
    providerOptions: getProviderOptions(provider as any, undefined, modelId) as any,
    maxOutputTokens: maxOutputTokensFor(provider),
    onStepFinish: (sf) => {
      for (const item of sf.content) {
        if (item.type === "tool-call") {
          logs.push(`tool_call ${item.toolName}`);
          console.log(`[auditor] taskId=${job.taskId} tool_call ${item.toolName}`);
        } else if (item.type === "text" && item.text) {
          const line = item.text.slice(0, 200).replace(/\n/g, " ");
          logs.push(`text ${line}`);
        }
      }
    },
  });

  async function run(): Promise<Verdict> {
    const timer = setTimeout(() => abortController.abort(), RUN_TIMEOUT_MS);
    let runError: string | undefined;

    const userPrompt =
      `Audit task ${job.taskId}. Load the task and its diff, the feature context, and the map, ` +
      `then exercise the RUNNING app to reach an evidence-backed verdict. ` +
      `Capture proof for anything you mark works, and call submit_verdict when done.`;

    console.log(
      `[auditor] run_start taskId=${job.taskId} model=${modelId} provider=${provider} appUrl=${job.deck.map.appUrl}`,
    );

    try {
      await agent.generate({
        prompt: userPrompt,
        abortSignal: abortController.signal,
      });
    } catch (err: any) {
      runError = err?.message ?? String(err);
      console.error(`[auditor] run_error taskId=${job.taskId}: ${runError}`);
    } finally {
      clearTimeout(timer);
      await browser.close().catch(() => {});
    }

    const finishedAt = new Date().toISOString();
    const submitted = collector.verdict;

    if (submitted) {
      return {
        taskId: job.taskId,
        overall: submitted.overall,
        claims: submitted.claims,
        observations: submitted.observations,
        summary: submitted.summary,
        evidence: collector.records,
        startedAt,
        finishedAt,
        ...(runError ? { error: runError } : {}),
      };
    }

    return {
      taskId: job.taskId,
      overall: "unknown",
      claims: [],
      observations: [
        `Auditor ended without submitting a verdict after ${logs.length} logged step events.`,
        ...(collector.records.length > 0
          ? [`${collector.records.length} evidence records were captured.`]
          : []),
      ],
      summary:
        "The audit ended before submit_verdict was called; no honest verdict could be produced.",
      evidence: collector.records,
      startedAt,
      finishedAt,
      error: runError ?? "no verdict submitted",
    };
  }

  return { run };
}
