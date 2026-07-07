import { ToolLoopAgent, ModelMessage } from "ai";
import { addUsage, ModelName, getModelDetails, getProviderOptions, normalizeUsage } from "../aieo/src/index.js";
import { get_log_tools } from "./tools.js";
import { ContextResult } from "../tools/types.js";
import { redactSecrets, redactSecretsDeep } from "./redact.js";
import {
  extractFinalAnswer,
  extractMessagesFromSteps,
  createHasEndMarkerCondition,
} from "../repo/utils.js";
import {
  createSession as createNewSession,
  loadSessionMessages,
  appendMessages,
  appendSessionEnd,
  appendStepMeta,
  sessionExists,
  saveSessionConfig,
  saveSessionMetadata,
  SessionConfig,
  StepMeta,
} from "../repo/session.js";
import { StakworkRunSummary } from "./types.js";

const SYSTEM = `You are a log analysis agent. You have tools to fetch logs from various sources (CloudWatch, Quickwit, etc.) and write them to local files, plus a bash tool to search and analyze those files.

Your workflow:
1. If the user mentions a specific log group or service, fetch those logs directly.
2. If unsure which log group to use, call list_cloudwatch_groups to discover available groups.
3. If a Stakwork run has agentLogs, use fetch_agent_log to read the full agent log content for a specific agent.
4. After fetching, use bash to search and analyze the log files (rg, grep, awk, sort, uniq, wc, head, tail, etc.).
5. Synthesize your findings into a clear answer.

Tips:
- Fetch logs first, then search. Don't try to search before fetching.
- For large log volumes, use specific filter patterns when fetching to reduce noise.
- Look for patterns: recurring errors, timestamps clustering, correlated events.
- When debugging an issue, search for both the error itself and surrounding context.

TIME BUDGET (very important): You have roughly 8 minutes of wall-clock time total. Work efficiently:
- Plan a focused approach up front; don't fetch more than you need. Prefer narrow filter patterns, specific log streams, and shorter time windows over broad fetches.
- Avoid rabbit holes and redundant tool calls. Each fetch/search should have a clear purpose.
- Every tool result ends with a "[time budget: ...]" line showing how much time has elapsed and how much is left. Watch it. Once only a couple of minutes remain (or you have enough to answer), STOP investigating and write up what you've found.
- It is far better to return a partial-but-useful answer than to run out of time with nothing. If you couldn't fully confirm something, say so and report your best findings, leads, and next steps.
- You MUST always finish by producing a final answer with the [END_OF_ANSWER] marker — never end without one.

CRITICAL: When you are ready to provide your final answer, output your complete response followed by [END_OF_ANSWER] on a new line.

Example format:
Your complete analysis here.

[END_OF_ANSWER]`;

export interface LogAgentOptions {
  modelName?: ModelName;
  apiKey?: string;
  baseUrl?: string;
  logs?: boolean;
  sessionId?: string;
  sessionConfig?: SessionConfig;
  stakworkApiKey?: string;
  stakworkRuns?: StakworkRunSummary[];
  logsDir: string;
  printAgentProgress?: boolean;
  source?: string;
  /** Custom HTTP headers attached to every LLM endpoint request (provider-level). */
  headers?: Record<string, string>;
  /** Abort signal to cancel the in-flight run (and forward into tools). */
  abortSignal?: AbortSignal;
  /** Optional caller-supplied metadata persisted as a sidecar at session-create time. */
  _metadata?: unknown;
}

export async function log_agent_context(
  prompt: string,
  opts: LogAgentOptions
): Promise<ContextResult> {
  const startTime = Date.now();
  const redactOpts = { literals: [opts.stakworkApiKey, opts.apiKey].filter(Boolean) as string[] };
  const { model, provider, modelId } = getModelDetails(opts.modelName, opts.apiKey, opts.baseUrl, opts.headers);
  console.log("===> log_agent model", model);

  const tools = get_log_tools({
    logsDir: opts.logsDir,
    stakworkApiKey: opts.stakworkApiKey,
    stakworkRuns: opts.stakworkRuns,
    abortSignal: opts.abortSignal,
    startTime,
  });

  const hasEndMarker = createHasEndMarkerCondition<typeof tools>();

  for (const t of Object.keys(tools)) {
    console.log("===> log tool", t);
  }

  const stepMetas: StepMeta[] = [];
  let cumInput = 0;
  let cumOutput = 0;
  let turnIndex = 1;

  const agent = new ToolLoopAgent({
    model,
    instructions: SYSTEM,
    tools,
    providerOptions: getProviderOptions(provider, undefined, modelId) as any,
    stopWhen: hasEndMarker,
    stopSequences: ["[END_OF_ANSWER]"],
    onStepFinish: (sf) => {
      logStepMaybe(redactSecretsDeep(sf.content, redactOpts), opts.printAgentProgress);
      const usage = normalizeUsage(sf.usage);
      cumInput += usage.inputTokens ?? 0;
      cumOutput += usage.outputTokens ?? 0;
      stepMetas.push({
        step: stepMetas.length,
        turn: turnIndex,
        usage,
        cumulativeInput: cumInput,
        cumulativeOutput: cumOutput,
        toolCalls: (sf.toolCalls ?? []).map((tc: { toolName: string }) => tc.toolName),
        timestamp: new Date().toISOString(),
      });
    },
  });

  // Session handling (after instructions are final so we can persist them)
  let sessionId: string | undefined;
  let previousMessages: ModelMessage[] = [];

  if (opts.sessionId) {
    if (sessionExists(opts.sessionId)) {
      sessionId = opts.sessionId;
      previousMessages = loadSessionMessages(sessionId);
    } else {
      sessionId = createNewSession(opts.sessionId, SYSTEM, opts.source);
      // systemOverride carries the fixed SYSTEM prompt (not a caller override) —
      // shape aligns with repo-agent's SessionInitConfig per contract requirement.
      saveSessionConfig(sessionId, {
        model: modelId,
        provider,
        systemOverride: SYSTEM,
        sessionConfig: opts.sessionConfig,
        source: opts.source,
        temperature: 0, // required field on SessionInitConfig — must always be set
        tools: Object.fromEntries(
          Object.entries(tools).map(([name, t]) => [name, (t as any).description ?? ""])
        ),
        providerConfig: getProviderOptions(provider, undefined, modelId) as any,
        baseUrl: opts.baseUrl,
      });
      if (opts._metadata !== undefined) {
        saveSessionMetadata(sessionId, opts._metadata);
      }
    }
    turnIndex = previousMessages.filter((m) => m.role === "user").length + 2;
  }

  const userMessage: ModelMessage = { role: "user", content: prompt };

  const abortSignal = opts.abortSignal;
  let result;
  try {
    if (previousMessages.length > 0) {
      result = await agent.generate({
        messages: [...previousMessages, userMessage],
        ...(abortSignal ? { abortSignal } : {}),
      });
    } else {
      result = await agent.generate({
        prompt,
        ...(abortSignal ? { abortSignal } : {}),
      });
    }
  } catch (err) {
    if (sessionId) {
      const { modelId, provider } = getModelDetails(opts.modelName, opts.apiKey);
      await appendSessionEnd(sessionId, {
        end_time: new Date().toISOString(),
        model: modelId,
        provider,
        duration_ms: Date.now() - startTime,
        status: "error",
        error_message: err instanceof Error ? err.message : String(err),
      });
    }
    throw err;
  }

  const { steps, totalUsage } = result;
  const usage = stepMetas.length > 0
    ? normalizeUsage(addUsage(...stepMetas.map((step) => step.usage)))
    : normalizeUsage(totalUsage);

  // Save to session if enabled
  if (sessionId) {
    const newMessages = extractMessagesFromSteps(
      userMessage,
      steps,
      opts.sessionConfig
    );
    appendMessages(sessionId, redactSecretsDeep(newMessages, redactOpts) as ModelMessage[]);
    appendStepMeta(sessionId, redactSecretsDeep(stepMetas, redactOpts) as StepMeta[]);
    const { modelId, provider } = getModelDetails(opts.modelName, opts.apiKey);
    appendSessionEnd(sessionId, {
      end_time: new Date().toISOString(),
      model: modelId,
      provider,
      status: "success",
      token_usage: usage,
    });
  }

  const final = extractFinalAnswer(steps);

  const endTime = Date.now();
  const duration = endTime - startTime;
  console.log(
    `log_agent completed in ${duration}ms (${(duration / 1000).toFixed(2)}s)`
  );

  const redactedAnswer = redactSecrets(final.answer, redactOpts);
  return {
    final: redactedAnswer,
    tool_use: final.tool_use,
    content: redactedAnswer,
    usage,
    logs: opts.logs ? JSON.stringify(redactSecretsDeep(steps, redactOpts), null, 2) : undefined,
    sessionId,
  };
}

export function logStepMaybe(contents: any, printAgentProgress?: boolean) {
  if (!printAgentProgress) return;
  console.log("===> logStep", JSON.stringify(contents, null, 2));
}