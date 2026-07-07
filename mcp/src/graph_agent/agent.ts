import {
  ToolLoopAgent,
  ModelMessage,
  StopCondition,
  ToolSet,
  StepResult,
  stepCountIs,
} from "ai";
import {
  addUsage,
  normalizeUsage,
  getModelDetails,
  getProviderOptions,
  ModelName,
} from "../aieo/src/index.js";
import {
  createSession,
  appendSessionEnd,
  loadSession,
  loadSessionMessages,
  appendMessages,
  appendStepMeta,
  sessionExists,
  saveSessionConfig,
  saveSessionMetadata,
  SessionConfig,
  StepMeta,
} from "../repo/session.js";
import {
  extractFinalAnswer,
  createHasEndMarkerCondition,
  extractMessagesFromSteps,
  truncateOldToolResults,
} from "../repo/utils.js";
import { get_graph_tools, GraphToolsConfig } from "./tools.js";

type GraphTools = ReturnType<typeof get_graph_tools>;
import { GRAPH_AGENT_SYSTEM_PROMPT, buildContextualSystemPrompt } from "./prompts/graph.js";
import { randomUUID } from "crypto";

export interface GraphAgentOptions {
  prompt: string | ModelMessage[];
  modelName?: ModelName;
  apiKey?: string;
  baseUrl?: string;
  sessionId?: string;
  sessionConfig?: SessionConfig;
  abortSignal?: AbortSignal;
  onStepEvent?: (content: any[]) => void;
  maxTurns?: number;
  // Forwarded L402/Authorization token for jarvis tool calls
  authToken?: string;
  /** Custom HTTP headers attached to every LLM endpoint request (provider-level). */
  headers?: Record<string, string>;
  /** Optional context to scope the agent to a specific node and its neighbourhood. */
  context?: { selectedRefId: string; nodeType: string; title?: string };
  /** Optional caller-supplied metadata persisted as a sidecar at session-create time only. */
  _metadata?: unknown;
}

/** Returns true if the error was caused by an AbortSignal. */
function isAbortError(err: unknown): boolean {
  if (!err) return false;
  if (err instanceof Error) {
    if (err.name === "AbortError") return true;
    const cause: any = (err as any).cause;
    if (cause && cause.name === "AbortError") return true;
    if (/abort/i.test(err.message)) return true;
  }
  return false;
}

function logStep(content: any[]): void {
  for (const item of content) {
    if (item.type === "tool-call") {
      console.log(`[graph_agent] tool_call: ${item.toolName}`);
    } else if (item.type === "text" && item.text) {
      console.log(`[graph_agent] text: ${item.text.slice(0, 120).replace(/\n/g, " ")}...`);
    }
  }
}

interface PreparedGraphAgent {
  agent: ToolLoopAgent<never, GraphTools>;
  modelId: string;
  provider: string;
  finalPrompt: string | ModelMessage[];
  previousMessages: ModelMessage[];
  userMessage: ModelMessage;
  sessionId: string | undefined;
  sessionConfig: SessionConfig | undefined;
  startTime: number;
  stepMetas: StepMeta[];
  turnIndex: number;
  abortSignal: AbortSignal | undefined;
  requestId: string;
}

async function prepareGraphAgent(
  opts: GraphAgentOptions,
): Promise<PreparedGraphAgent> {
  const {
    prompt,
    modelName,
    apiKey: apiKeyIn,
    baseUrl,
    sessionId: inputSessionId,
    sessionConfig,
    abortSignal,
    onStepEvent,
    maxTurns,
    authToken,
    headers,
    context,
  } = opts;

  const startTime = Date.now();
  const requestId = randomUUID();

  const { model, apiKey: _apiKey, provider, contextLimit, modelId } = getModelDetails(
    modelName,
    apiKeyIn,
    baseUrl,
    headers,
  );
  console.log(`[graph_agent] model=${modelId} provider=${provider} contextLimit=${contextLimit}`);

  const systemPrompt = context
    ? buildContextualSystemPrompt(context)
    : GRAPH_AGENT_SYSTEM_PROMPT;

  const toolConfig: GraphToolsConfig = { authToken };
  const tools = get_graph_tools(toolConfig);

  const hasEndMarker = createHasEndMarkerCondition<typeof tools>();

  const stopConditions: StopCondition<GraphTools>[] = [hasEndMarker];
  if (typeof maxTurns === "number" && maxTurns > 0) {
    stopConditions.push(stepCountIs(maxTurns) as StopCondition<GraphTools>);
  }
  const stopWhen: StopCondition<GraphTools> | StopCondition<GraphTools>[] =
    stopConditions.length === 1 ? stopConditions[0] : stopConditions;

  // Session setup
  let sessionId: string | undefined;
  let previousMessages: ModelMessage[] = [];
  let hasSystemTurn = false;

  if (inputSessionId) {
    if (sessionExists(inputSessionId)) {
      sessionId = inputSessionId;
      hasSystemTurn = loadSession(sessionId)[0]?.role === "system";
      previousMessages = loadSessionMessages(sessionId);
    } else {
      sessionId = createSession(inputSessionId, systemPrompt, "graph_agent");
      hasSystemTurn = true;
      // Persist config sidecar at session-create time so GET /repo/agent/session returns non-null config.
      // Note: providerConfig is descriptive only — graph_agent's ToolLoopAgent does not pass providerOptions at runtime.
      saveSessionConfig(sessionId, {
        model: modelId,
        provider,
        // systemOverride here carries the resolved system prompt (context-aware or default).
        systemOverride: systemPrompt,
        sessionConfig: opts.sessionConfig,
        source: "graph_agent",
        maxTurns: opts.maxTurns,
        temperature: 0, // required field on SessionInitConfig — must always be set
        tools: Object.fromEntries(
          Object.entries(tools).map(([name, t]) => [name, (t as any).description ?? ""])
        ),
        providerConfig: getProviderOptions(provider as any, undefined, modelId),
        baseUrl: opts.baseUrl,
      });
      // Persist optional caller-supplied metadata only at create time (not on resume).
      if (opts._metadata !== undefined) {
        saveSessionMetadata(sessionId, opts._metadata);
      }
    }
  }

  if (context) {
    console.log(
      `[graph_agent] contextual_run ref_id=${context.selectedRefId} nodeType=${context.nodeType} sessionId=${sessionId ?? "none"}`,
    );
  }

  const stepMetas: StepMeta[] = [];
  let cumInput = 0;
  let cumOutput = 0;

  const turnIndex =
    previousMessages.filter((m) => m.role === "user").length +
    (hasSystemTurn ? 2 : 1);

  const agent = new ToolLoopAgent({
    model,
    instructions: systemPrompt,
    tools,
    stopWhen,
    stopSequences: ["[END_OF_ANSWER]"],
    onStepFinish: (sf) => {
      logStep(sf.content);
      if (onStepEvent) {
        try { onStepEvent(sf.content); } catch (_) {}
      }
      const u = normalizeUsage(sf.usage);
      cumInput += u.inputTokens ?? 0;
      cumOutput += u.outputTokens ?? 0;
      stepMetas.push({
        step: stepMetas.length,
        turn: turnIndex,
        usage: u,
        cumulativeInput: cumInput,
        cumulativeOutput: cumOutput,
        toolCalls: (sf.toolCalls ?? []).map((tc: { toolName: string }) => tc.toolName),
        timestamp: new Date().toISOString(),
      });
    },
    prepareStep: async ({ steps, messages }) => {
      const lastStep = steps.length > 0 ? steps[steps.length - 1] : null;
      const inputTokens = lastStep?.usage?.inputTokens ?? 0;
      const truncated = await truncateOldToolResults(messages, inputTokens, contextLimit);
      if (truncated === messages) return undefined;
      return { messages: truncated };
    },
  });

  const userMessageContent =
    typeof prompt === "string"
      ? prompt
      : (prompt.find((m) => m.role === "user")?.content || "");

  const userMessage: ModelMessage = {
    role: "user",
    content: userMessageContent as string,
  };

  return {
    agent,
    modelId,
    provider,
    finalPrompt: prompt,
    previousMessages,
    userMessage,
    sessionId,
    sessionConfig,
    startTime,
    stepMetas,
    turnIndex,
    abortSignal,
    requestId,
  };
}

function buildCallParams(prepared: PreparedGraphAgent) {
  const { finalPrompt, previousMessages, userMessage, provider, modelId, abortSignal } = prepared;
  const providerOptions = getProviderOptions(provider as any, undefined, modelId);
  const base = abortSignal ? { providerOptions, abortSignal } : { providerOptions };

  if (previousMessages.length > 0) {
    const messagesToSend =
      typeof finalPrompt === "string"
        ? [...previousMessages, userMessage]
        : [...previousMessages, ...finalPrompt];
    return { messages: messagesToSend, ...base };
  }
  if (typeof finalPrompt === "string") {
    return { prompt: finalPrompt, ...base };
  }
  return { messages: finalPrompt, ...base };
}

export async function get_context(opts: GraphAgentOptions): Promise<{
  answer: string;
  cited_ref_ids: string[];
  usage: Record<string, any>;
  sessionId: string | undefined;
}> {
  const prepared = await prepareGraphAgent(opts);
  const { agent, modelId, provider, sessionId, sessionConfig, userMessage, startTime, stepMetas } =
    prepared;

  console.log(`[graph_agent] graph_agent_run_start requestId=${prepared.requestId} sessionId=${sessionId ?? "none"} model=${modelId}`);

  let result: Awaited<ReturnType<typeof agent.generate>>;
  try {
    result = await agent.generate(buildCallParams(prepared));
  } catch (err) {
    const aborted = isAbortError(err);
    const duration = Date.now() - startTime;
    console.log(
      `[graph_agent] graph_agent_run_end requestId=${prepared.requestId} sessionId=${sessionId ?? "none"} model=${modelId} duration=${duration}ms status=${aborted ? "aborted" : "error"}`,
    );
    if (sessionId) {
      await appendSessionEnd(sessionId, {
        end_time: new Date().toISOString(),
        model: modelId,
        provider,
        duration_ms: duration,
        status: aborted ? "aborted" : "error",
        error_message: err instanceof Error ? err.message : String(err),
      });
    }
    throw err;
  }

  const { steps, totalUsage } = result;
  const usage = stepMetas.length > 0
    ? normalizeUsage(addUsage(...stepMetas.map((s) => s.usage)))
    : normalizeUsage(totalUsage);

  const duration = Date.now() - startTime;

  // Persist session
  if (sessionId) {
    const newMessages = extractMessagesFromSteps(userMessage, steps as unknown as StepResult<ToolSet>[], sessionConfig);
    appendMessages(sessionId, newMessages);
    appendStepMeta(sessionId, stepMetas);
    await appendSessionEnd(sessionId, {
      end_time: new Date().toISOString(),
      model: modelId,
      provider,
      duration_ms: duration,
      status: "success",
      token_usage: usage,
    });
  }

  const final = extractFinalAnswer(steps as unknown as StepResult<ToolSet>[]);
  let answer = final.answer || "";
  let cited_ref_ids: string[] = [];

  // Try to parse the structured JSON response
  try {
    const jsonMatch = answer.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      if (parsed.answer) answer = parsed.answer;
      if (Array.isArray(parsed.cited_ref_ids)) cited_ref_ids = parsed.cited_ref_ids;
    }
  } catch (_) {
    // If JSON parsing fails, use the raw answer
  }

  console.log(
    `[graph_agent] graph_agent_run_end requestId=${prepared.requestId} sessionId=${sessionId ?? "none"} model=${modelId} duration=${duration}ms status=success inputTokens=${usage.inputTokens ?? 0} outputTokens=${usage.outputTokens ?? 0}`,
  );

  return {
    answer,
    cited_ref_ids,
    usage: { ...usage, model: modelId, provider },
    sessionId,
  };
}

export async function stream_context(opts: GraphAgentOptions) {
  const prepared = await prepareGraphAgent(opts);
  const { agent, modelId, provider, sessionId, sessionConfig, userMessage, startTime, stepMetas } =
    prepared;

  console.log(`[graph_agent] graph_agent_run_start (stream) requestId=${prepared.requestId} sessionId=${sessionId ?? "none"} model=${modelId}`);

  const streamResult = await agent.stream(buildCallParams(prepared));

  return {
    streamResult,
    async finalizeSession() {
      if (!sessionId) return;
      try {
        const steps = await (streamResult as any).steps;
        const usage = await (streamResult as any).usage;
        const duration = Date.now() - startTime;
        const newMessages = extractMessagesFromSteps(
          userMessage,
          steps ?? [],
          sessionConfig,
        );
        appendMessages(sessionId, newMessages);
        appendStepMeta(sessionId, stepMetas);
        const stepUsage =
          stepMetas.length > 0
            ? normalizeUsage(addUsage(...stepMetas.map((s) => s.usage)))
            : normalizeUsage(usage);
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: duration,
          status: "success",
          token_usage: stepUsage,
        });
        console.log(
          `[graph_agent] graph_agent_run_end (stream) requestId=${prepared.requestId} sessionId=${sessionId} model=${modelId} duration=${duration}ms status=success`,
        );
      } catch (e) {
        const aborted = isAbortError(e);
        if (aborted) {
          console.log(`[graph_agent] Stream aborted requestId=${prepared.requestId}`);
        } else {
          console.error(`[graph_agent] Failed to finalize session:`, e);
        }
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status: aborted ? "aborted" : "error",
          error_message: e instanceof Error ? e.message : String(e),
        }).catch(() => {});
        console.log(
          `[graph_agent] graph_agent_run_end (stream) requestId=${prepared.requestId} sessionId=${sessionId} model=${modelId} status=${aborted ? "aborted" : "error"}`,
        );
      }
    },
  };
}
