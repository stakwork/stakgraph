import { ToolLoopAgent, ModelMessage } from "ai";
import { ModelName, getModelDetails } from "../aieo/src/index.js";
import { get_log_tools } from "./tools.js";
import { ContextResult } from "../tools/types.js";
import {
  extractFinalAnswer,
  extractMessagesFromSteps,
  createHasEndMarkerCondition,
} from "../repo/utils.js";
import {
  createSession as createNewSession,
  loadSession,
  appendMessages,
  sessionExists,
  SessionConfig,
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

CRITICAL: When you are ready to provide your final answer, output your complete response followed by [END_OF_ANSWER] on a new line.

Example format:
Your complete analysis here.

[END_OF_ANSWER]`;

export interface LogAgentOptions {
  modelName?: ModelName;
  apiKey?: string;
  logs?: boolean;
  sessionId?: string;
  sessionConfig?: SessionConfig;
  stakworkApiKey?: string;
  stakworkRuns?: StakworkRunSummary[];
  logsDir: string;
  printAgentProgress?: boolean;
}

export async function log_agent_context(
  prompt: string,
  opts: LogAgentOptions
): Promise<ContextResult> {
  const startTime = Date.now();
  const { model } = getModelDetails(opts.modelName, opts.apiKey);
  console.log("===> log_agent model", model);

  // Session handling
  let sessionId: string | undefined;
  let previousMessages: ModelMessage[] = [];

  if (opts.sessionId) {
    if (sessionExists(opts.sessionId)) {
      sessionId = opts.sessionId;
      previousMessages = loadSession(sessionId);
    } else {
      sessionId = createNewSession(opts.sessionId);
    }
  }

  const tools = get_log_tools({
    logsDir: opts.logsDir,
    stakworkApiKey: opts.stakworkApiKey,
    stakworkRuns: opts.stakworkRuns,
  });

  const hasEndMarker = createHasEndMarkerCondition<typeof tools>();

  for (const t of Object.keys(tools)) {
    console.log("===> log tool", t);
  }

  const agent = new ToolLoopAgent({
    model,
    instructions: SYSTEM,
    tools,
    stopWhen: hasEndMarker,
    stopSequences: ["[END_OF_ANSWER]"],
    onStepFinish: (sf) => logStepMaybe(sf.content, opts.printAgentProgress), // dont log logs_agent logs!
  });

  const userMessage: ModelMessage = { role: "user", content: prompt };

  let result;
  if (previousMessages.length > 0) {
    result = await agent.generate({
      messages: [...previousMessages, userMessage],
    });
  } else {
    result = await agent.generate({ prompt });
  }

  const { steps, totalUsage } = result;

  // Save to session if enabled
  if (sessionId) {
    const newMessages = extractMessagesFromSteps(
      userMessage,
      steps,
      opts.sessionConfig
    );
    appendMessages(sessionId, newMessages);
  }

  const final = extractFinalAnswer(steps);

  const endTime = Date.now();
  const duration = endTime - startTime;
  console.log(
    `log_agent completed in ${duration}ms (${(duration / 1000).toFixed(2)}s)`
  );

  return {
    final: final.answer,
    tool_use: final.tool_use,
    content: final.answer,
    usage: {
      inputTokens: totalUsage.inputTokens || 0,
      outputTokens: totalUsage.outputTokens || 0,
      totalTokens: totalUsage.totalTokens || 0,
    },
    logs: opts.logs ? JSON.stringify(steps, null, 2) : undefined,
    sessionId,
  };
}

export function logStepMaybe(contents: any, printAgentProgress?: boolean) {
  if (!printAgentProgress) return;
  console.log("===> logStep", JSON.stringify(contents, null, 2));
}