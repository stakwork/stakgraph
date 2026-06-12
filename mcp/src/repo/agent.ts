import {
  generateText,
  Output,
  ToolLoopAgent,
  ModelMessage,
  StopCondition,
  ToolSet,
  jsonSchema,
  stepCountIs,
} from "ai";
import {
  addUsage,
  ModelName,
  getModelDetails,
  getProviderOptions,
  normalizeUsage,
} from "../aieo/src/index.js";
import { get_tools, ToolsConfig, SkillsConfig, GgnnConfig, MessagesRef, ProvenanceCollector } from "./tools.js";
import { SKILLS } from "./skills.js";
import { type SubAgent, subAgentRepoNames } from "./subagent.js";
import { ContextResult } from "../tools/types.js";
import {
  logStep,
  extractFinalAnswer,
  createHasEndMarkerCondition,
  createHasAskQuestionsCondition,
  ensureAdditionalPropertiesFalse,
  extractMessagesFromSteps,
  deepParseJsonStrings,
  truncateOldToolResults,
  extractLeadingJsonObject,
  matchesSchemaShape,
  collectEnumConstraints,
} from "./utils.js";
import { LanguageModel } from "ai";
import {
  createSession as createNewSession,
  appendSessionEnd,
  loadSession,
  loadSessionMessages,
  appendMessages,
  appendStepMeta,
  appendSearchProvenance,
  sessionExists,
  SessionConfig,
  StepMeta,
} from "./session.js";
import { McpServer, getMcpTools } from "./mcpServers.js";
import {
  resolveCurrentTurnAttachments,
  cacheAttachments,
  buildImageParts,
  buildPlaceholderParts,
  rehydrateMessages,
  lastUserText,
  CachedAttachment,
} from "./attachments.js";
import type { TextPart } from "ai";

function SYSTEM_PROMPT_END(qs: boolean) {
  const normalEnd = `CRITICAL: When you are ready to provide your final answer, output your complete response followed by [END_OF_ANSWER] on a new line. Don't start your answer with preamble like "Ok! I have all the information I need. Let me create a plan...". Just start with your answer.

  Write your answer directly as text and end with [END_OF_ANSWER].`;
  
  const qsEnd = `CRITICAL: When you finish exploring, you MUST do ONE of these:
  
  1. Write your complete answer as text, then output [END_OF_ANSWER] on a new line
  2. Call ask_clarifying_questions tool with format: { "questions": [...] }
  
  Call ask_clarifying_questions when:
  - The user's query is too general
  - You can provide a better answer by first gathering more information from the user
  - Your technical exploration has revealed multiple possible approaches, and you want the user's input on which to choose
  - You want to validate a proposed flow or architecture before proceeding (use mermaid or comparison_table questionArtifact)
  
  WIDGET TYPES:
  - Basic questions: Simple string options like ["Option A", "Option B"]
  - Diagram confirmation: Use questionArtifact with type "mermaid" to show a flow diagram
  - Comparison table: Use questionArtifact with type "comparison_table" to compare approaches with pros/cons
  - Color picker: Use questionArtifact with type "color_swatch" to show a color picker
  
  Otherwise, provide your answer directly followed by [END_OF_ANSWER]. Don't start your answer with preamble like "Ok! I have all the information I need. Let me create a plan...". Just write your answer.`

  return qs ? qsEnd : normalEnd;
}

function DEFAULT_SYSTEM(toolsConfig?: ToolsConfig) {

  const learn_concepts = toolsConfig?.learn_concept || toolsConfig?.list_concepts || toolsConfig?.learn_concepts

  const qs = toolsConfig?.ask_clarifying_questions ? true : false;

  return `You are a code exploration assistant with access to a **code knowledge graph**. Use graph tools whenever possible — they are faster, more precise, and understand code relationships.

Try to match the tone of the user. If the user is asking a technical question, research deeper, and respond with technical details. If the user's question is high-level (non-specific), then do not answer with too much detail!

${learn_concepts ? "Use list_concepts and learn_concept tools first, to learn about high-level features in the codebase." : ""}

### Graph Tools
- \`repo_overview\` — Use only for broad orientation or architecture questions; it returns a compact, de-noised repo tree.
- \`stakgraph_search\` — Search by keyword/semantic/hybrid. Returns compact results (name, file, ref_id, description). Use \`node_types\` to filter (e.g. \`["Endpoint"]\`, \`["Function"]\`, \`["Datamodel"]\`, \`["UnitTest"]\`). Use \`include_patterns\` to scope results — by file type (\`["**/*.ts"]\`), directory (\`["src/auth/**"]\`), or repo (\`["owner/repo/**"]\`). Use \`exclude_patterns\` to remove noise (\`["__tests__", "dist"]\`). Always scope to the target repo when multiple repos are in the graph.
- \`stakgraph_map\` — Trace relationships from a node. Use \`direction: "up"\` for callers, \`"down"\` for callees.
- \`stakgraph_code\` — Read source code of a specific node. Pass \`ref_id\` from search results or \`name\` + \`node_type\`.

### File Tools

- \`file_summary\` — Summarize a file's main code entities and their relationships.
- \`fulltext_search\` — Exact string matching via ripgrep.

### Bash Hints

- Full file: \`cat -n path/to/file\`
- Line range (e.g. 50-75): \`sed -n '50,75p' path/to/file\`
- Recursive search: \`rg -n "pattern" path/to/dir\`
- Filenames only: \`rg -l "pattern" path/to/dir\`
- By language: \`rg -n -t py "pattern" path/to/dir\`
- With context: \`rg -n -C5 "pattern" path/to/file\`
- Whole word match: \`rg -nw "pattern" path/to/dir\`
- Exclude paths: \`rg -n -g "!*.test.*" "pattern" path/to/dir\`
- Limit results: \`rg -n -m3 "pattern" path/to/dir\`
- Find files by name: \`find path/to/dir -name "*.py" -type f\`
- Directory overview: \`tree -L 2 path/to/dir\`

## Rules
The prompt prepended to your instructions tells you which repos are graph-backed and which are bash-only. Apply these rules per repo accordingly.

**For graph-backed repos:**
- If the user already named a feature, endpoint, file, model, or flow, skip \`repo_overview\` and go straight to \`stakgraph_search\`.
- After \`stakgraph_search\` returns results with ref_ids, your NEXT call MUST be \`stakgraph_code\` on one of those ref_ids — not bash, not another search.
- Do NOT use bash to search code (rg, grep, find). Use \`stakgraph_search\` instead.
- Do NOT use bash to read source files (cat, sed, head) when you have a ref_id. Use \`stakgraph_code\` instead.
- Use \`stakgraph_map\` to trace callers (direction: "up") or callees (direction: "down") — do not follow imports with bash.
- Search returned irrelevant nodes? Refine with \`node_types\` or a more specific query — do NOT fall back to bash.

**For bash-only repos (not in the graph):**
- Use bash and fulltext_search freely. Graph tools will return nothing.

**Always:**
- Stop calling tools as soon as you have enough information to answer the question. More calls rarely improve a complete answer.

## Workflow
1. Check the repo context prepended to your prompt — identify which repos are graph-backed.
2. For broad architecture questions only, \`repo_overview\` can help you orient on the repo tree.
3. \`stakgraph_search\` → find relevant nodes (returns names, ref_ids, descriptions — NOT full code). Scope to the target repo with \`include_patterns: ["owner/repo/**"]\` when multiple repos are available.
4. \`stakgraph_code\` → read source of each relevant node using its ref_id.
5. \`stakgraph_map\` → trace callers/callees when you need to follow a chain.
6. \`bash\` → only for config/env files, directory listings, or bash-only repos.

## Patterns
- **Find endpoints** → \`stakgraph_search({ query: "bounty", node_types: ["Endpoint"] })\`
- **How does X work?** → \`stakgraph_search({ query: "X" })\` → \`stakgraph_code({ ref_id: "..." })\` on each result
- **What calls Y?** → \`stakgraph_map({ name: "Y", node_type: "Function", direction: "up" })\`
- **List data models** → \`stakgraph_search({ query: "model", node_types: ["Datamodel"] })\`
- **Find tests** → \`stakgraph_search({ query: "X", node_types: ["UnitTest", "IntegrationTest"] })\`
- **Multi-repo: scope to one repo** → \`stakgraph_search({ query: "auth", include_patterns: ["stakwork/hive/**"] })\`

${SYSTEM_PROMPT_END(qs)}
`;
};

async function structureFinalAnswer(
  finalPrompt: string | ModelMessage[],
  finalAnswer: string,
  schema: { [key: string]: any },
  model: LanguageModel,
  provider: string
): Promise<any> {
  // Fast path: the agent often already emits a schema-shaped JSON object
  // (optionally followed by markdown, e.g. `{...}\n\n---\n\n## Report`).
  // If so, parse it directly and skip the structuring LLM call entirely —
  // a round-trip through the model risks "correcting" enum/literal values
  // (e.g. flipping `type: "user_question"` to `type: "run_debug"`)
  const direct =
    typeof finalAnswer === "string"
      ? extractLeadingJsonObject(finalAnswer)
      : finalAnswer;
  if (direct && matchesSchemaShape(direct, schema)) {
    console.log("===> structureFinalAnswer: using agent's structured answer as-is");
    return deepParseJsonStrings(direct);
  }

  const msgs: ModelMessage[] = [];

  // Handle finalPrompt - if it's ModelMessage[], push all messages, otherwise create a user message
  if (Array.isArray(finalPrompt)) {
    msgs.push(...finalPrompt);
  } else {
    msgs.push({ role: "user", content: finalPrompt });
  }

  // Add the assistant's answer and the structuring request.
  // IMPORTANT: this is a pure reformatting step, NOT a regeneration step.
  // Without strict instructions the model "rephrases" opaque literals (file
  // paths, ids, urls) into plausible-looking but fabricated values, causing
  // the structured `content` to diverge from the free-text `final_answer`.
  // Build an explicit reminder for every enum field so the model copies the
  // value already present in the answer instead of re-classifying it.
  const enumConstraints = collectEnumConstraints(schema);
  const enumRules =
    enumConstraints.length > 0
      ? "\n- ENUM FIELDS: For each field below, copy the EXACT value already present in the answer above. " +
        "Do NOT re-classify, re-categorize, or 'correct' it based on the answer's content — even if another " +
        "allowed value seems more fitting. The value in the answer is authoritative.\n" +
        enumConstraints
          .map((c) => `  - \`${c.path}\` must be one of: ${JSON.stringify(c.values)}`)
          .join("\n") +
        "\n"
      : "";

  msgs.push(
    { role: "assistant", content: finalAnswer },
    {
      role: "user",
      content:
        "Reformat the answer above into the following JSON format. " +
        "This is a formatting task ONLY — do not add, summarize, re-word, or invent anything.\n\n" +
        "STRICT RULES:\n" +
        "- Copy every value directly from the answer above. Do NOT generate new values.\n" +
        "- File paths, URLs, IDs, names, and other literals MUST be copied character-for-character. " +
        "Never normalize, guess, or substitute a 'typical-looking' value.\n" +
        "- If a field's value is not explicitly present in the answer above, leave it empty/null rather than inventing one." +
        enumRules +
        "\n\nJSON format:\n" +
        JSON.stringify(schema),
    }
  );

  const normalizedSchema = ensureAdditionalPropertiesFalse(schema);

  const { output } = await generateText({
    model,
    prompt: msgs,
    providerOptions: getProviderOptions(provider as any, "fast") as any,
    output: Output.object({ schema: jsonSchema(normalizedSchema) }),
  });

  return deepParseJsonStrings(output ?? {});
}

export interface GetContextOptions {
  modelName?: ModelName;
  apiKey?: string;
  baseUrl?: string;
  pat?: string | undefined;
  toolsConfig?: ToolsConfig;
  systemOverride?: string;
  schema?: { [key: string]: any };
  logs?: boolean;
  // Session support
  sessionId?: string; // Use existing session or create new one with this ID
  sessionConfig?: SessionConfig; // Truncation settings
  // MCP servers
  mcpServers?: McpServer[]; // External MCP servers to load tools from
  // Multi-repo support: list of "owner/repo" strings
  repos?: string[];
  // Skills support
  skills?: SkillsConfig;
  // Sub-agents: remote agent instances this agent can delegate to
  subAgents?: SubAgent[];
  // GGNN integration
  ggnn?: GgnnConfig;
  // Real-time step event callback (for SSE streaming)
  onStepEvent?: (content: any[]) => void;
  // Source label persisted to the session file
  source?: string;
  // Write messages to the session but don't load prior messages as context
  isolatedContext?: boolean;
  // Abort signal for cancelling in-flight requests
  abortSignal?: AbortSignal;
  // Maximum number of agent steps (tool-call turns) before stopping
  maxTurns?: number;
  // Image attachment URLs (e.g. uploaded screenshots) for the current turn
  attachments?: string[];
  // Custom HTTP headers attached to every LLM endpoint request (provider-level)
  headers?: Record<string, string>;
}

interface PreparedAgent {
  agent: ToolLoopAgent<never, ToolSet>;
  model: LanguageModel;
  modelId: string;
  provider: string;
  finalPrompt: string | ModelMessage[];
  previousMessages: ModelMessage[];
  userMessage: ModelMessage;
  storageUserMessage: ModelMessage;
  sessionId: string | undefined;
  sessionConfig: SessionConfig | undefined;
  startTime: number;
  stepMetas: StepMeta[];
  turnIndex: number;
  provenanceCollector: ProvenanceCollector;
  abortSignal: AbortSignal | undefined;
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

async function prepareAgent(
  prompt: string | ModelMessage[],
  repoPath: string,
  opts: GetContextOptions,
): Promise<PreparedAgent> {
  const {
    modelName,
    pat,
    toolsConfig,
    systemOverride,
    sessionId: inputSessionId,
    sessionConfig,
    mcpServers,
    apiKey: apiKeyIn,
    baseUrl,
    repos,
    skills,
    subAgents,
    ggnn,
    onStepEvent,
  } = opts;
  const startTime = Date.now();
  const { model, apiKey, provider, contextLimit, modelId } = getModelDetails(modelName, apiKeyIn, baseUrl, opts.headers);
  console.log("===> model", model, "contextLimit", contextLimit);

  const messagesRef: MessagesRef = { current: [] };
  const provenanceCollector: ProvenanceCollector = { entries: [] };
  let tools = await get_tools(
    repoPath,
    apiKey,
    pat,
    toolsConfig,
    provider,
    repos,
    subAgents,
    ggnn,
    messagesRef,
    provenanceCollector,
  );

  // Load and merge MCP server tools if configured
  let orgAgentToolNames: string[] = [];
  if (mcpServers && mcpServers.length > 0) {
    const mcpTools = await getMcpTools(mcpServers);
    tools = { ...tools, ...mcpTools };
    console.log(`[MCP] Merged ${Object.keys(mcpTools).length} MCP tools`);
    // Detect any "*_org_agent" tools (e.g. stakwork_org_agent, evanfeenstra_org_agent)
    orgAgentToolNames = Object.keys(mcpTools).filter(name => /_org_agent$/i.test(name));
  }

  let instructions = systemOverride || DEFAULT_SYSTEM(toolsConfig);

  // If an org_agent tool is available from MCP, inject a strong hint to use it for unclear/high-level questions.
  if (orgAgentToolNames.length > 0) {
    const toolList = orgAgentToolNames.map(n => `\`${n}\``).join(", ");
    const single = orgAgentToolNames.length === 1;
    const primary = orgAgentToolNames[0];
    const orgAgentBlock = `\n\nORG AGENT TOOL${single ? "" : "S"}:
You have access to ${single ? "an org agent tool" : "org agent tools"}: ${toolList}.

**Purpose:** ${single ? "This tool provides" : "These tools provide"} HIGH-LEVEL, ORG-WIDE context that is NOT available in this swarm (e.g. product goals, business context, R+D, organizational priorities, related projects, external systems). The specific source code of the repos you are exploring already lives here — do NOT ask the org agent about it.

**When to call ${single ? `\`${primary}\`` : "one of these tools"}:**
- The user's question is ambiguous, vague, or high-level and you need broader context to interpret it.
- You need org-wide / product-level / business background that isn't in the code.
- Call it FIRST in these cases, before graph or bash tools.

**How to phrase the query — CRITICAL:**
- Ask ONLY for the high-level/general context you are missing.
- Do NOT mention specific repos, files, functions, classes, endpoints, models, or code symbols from the user's question.
- Do NOT ask it to look up implementation details — those live HERE in the swarm and you'll find them with graph/bash tools.
- Frame the query generically — as if the org agent has no access to this codebase (because it shouldn't be looking at code at all).
- Do NOT ask it to do a deep dive - ask it to be brief and concise.

After getting high-level context back, use graph/bash tools on the actual codebase to answer the specific question.`;
    instructions += orgAgentBlock;
  }

  // console.log("INSTRUCTIONS", instructions);

  // Append sub-agent instructions if any sub-agents are registered
  if (subAgents && subAgents.length > 0) {
    const validSubAgents = subAgents.filter(sa => sa.name && sa.url && sa.apiToken);
    if (validSubAgents.length > 0) {
      const agentList = validSubAgents
        .map(sa => {
          const desc = sa.description || `Delegates to the "${sa.name}" sub-agent`;
          const repos = subAgentRepoNames(sa);
          const reposPart = repos.length > 0 ? ` (repos: ${repos.join(", ")})` : "";
          return `  - @${sa.name}: ${desc}${reposPart}`;
        })
        .join('\n');
      const subAgentBlock = `\n\nSUB-AGENTS:
You have access to the following sub-agents as tools:
${agentList}
If the user's prompt mentions a sub-agent with an @mention (e.g. "@${validSubAgents[0].name}"), use that sub-agent tool to handle the relevant part of the request. Pass it a focused, specific question or task based on the user's prompt.`;
      instructions += subAgentBlock;
    }
  }

  // Append skills instructions if any skills are active
  const activeSkills = Object.entries(skills || {})
    .filter(([, enabled]) => enabled)
    .map(([name]) => name);

  if (activeSkills.length > 0) {
    const inlineSkills = activeSkills.filter(name => !name.includes("/") && SKILLS[name]);
    const pathSkills = activeSkills.filter(name => name.includes("/") || !SKILLS[name]);

    if (inlineSkills.length > 0) {
      const inlineBlock = inlineSkills.map(name => SKILLS[name]).join('\n\n');
      instructions += `\n\n${inlineBlock}`;
    }

    if (pathSkills.length > 0) {
      const pathBlock = `\n\nSKILLS INSTRUCTIONS:
Before starting your main task, use the bash tool to load your active skills into context:
${pathSkills
    .map(name => `  - Run: ls ~/.agents/skills/${name}/\n  - Then: cat ~/.agents/skills/${name}/SKILL.md`)
    .join('\n')}
Apply the guidance from each skill throughout your response.`;
      instructions += pathBlock;
    }
  }

  const hasEndMarker = createHasEndMarkerCondition<typeof tools>();
  const hasAskQuestions = createHasAskQuestionsCondition<typeof tools>();

  const stopConditions: StopCondition<ToolSet>[] = [hasEndMarker];
  let finalPrompt: string | ModelMessage[] = prompt;

  if (toolsConfig?.ask_clarifying_questions) {
    stopConditions.push(hasAskQuestions);
  }

  // Resolve image attachments for THIS turn (body field preferred, else the
  // <attachments> tag on the last user message) and strip the tag from text.
  const { urls: attachmentUrls, cleanedPrompt } = resolveCurrentTurnAttachments(
    finalPrompt,
    opts.attachments,
  );
  finalPrompt = cleanedPrompt;

  if (typeof opts.maxTurns === "number" && opts.maxTurns > 0) {
    stopConditions.push(stepCountIs(opts.maxTurns));
  }

  const stopWhen: StopCondition<ToolSet> | StopCondition<ToolSet>[] =
    stopConditions.length === 1 ? stopConditions[0] : stopConditions;

  // Derive repo label from opts.repos or repoPath
  const repoLabel = opts.repos && opts.repos.length > 0
    ? opts.repos.join(", ")
    : repoPath.replace(/\/+$/, "").split("/").slice(-2).join("/");

  // Session handling (after instructions are fully assembled so we can persist them)
  let sessionId: string | undefined;
  let previousMessages: ModelMessage[] = [];
  let hasSystemTurn = false;

  if (inputSessionId) {
    if (sessionExists(inputSessionId)) {
      sessionId = inputSessionId;
      hasSystemTurn = loadSession(sessionId)[0]?.role === "system";
      previousMessages = opts.isolatedContext ? [] : loadSessionMessages(sessionId);
    } else {
      sessionId = createNewSession(inputSessionId, instructions, opts.source, repoLabel);
      hasSystemTurn = true;
    }
  }

  // Rehydrate cached image attachments from prior turns (placeholder -> bytes)
  // so the model can keep seeing earlier screenshots on follow-up turns.
  if (sessionId && previousMessages.length > 0) {
    previousMessages = rehydrateMessages(previousMessages, sessionId);
  }

  // Download + cache this turn's attachments once.
  let cachedAttachments: CachedAttachment[] = [];
  if (attachmentUrls.length > 0) {
    cachedAttachments = await cacheAttachments(attachmentUrls, sessionId, opts.abortSignal);
  }

  for (const tool of Object.keys(tools)) {
    console.log("===> tool", tool, "===>", tools[tool].description);
  }

  const stepMetas: StepMeta[] = [];
  let cumInput = 0;
  let cumOutput = 0;
  const turnIndex =
    previousMessages.filter((m) => m.role === "user").length +
    (hasSystemTurn ? 2 : 1);

  const agent = new ToolLoopAgent({
    model,
    instructions,
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
      messagesRef.current = messages as ModelMessage[];
      const lastStep = steps.length > 0 ? steps[steps.length - 1] : null;
      const inputTokens = lastStep?.usage?.inputTokens ?? 0;
      const truncated = await truncateOldToolResults(messages, inputTokens, contextLimit);
      if (truncated === messages) return undefined;
      return { messages: truncated };
    },
  });

  // The message SENT to the model carries real image bytes; the message
  // PERSISTED to the session carries only `attachment://` placeholders.
  let userMessage: ModelMessage; // sent to the model
  let storageUserMessage: ModelMessage; // persisted to the session

  if (cachedAttachments.length > 0) {
    const turnText =
      typeof finalPrompt === "string"
        ? finalPrompt
        : lastUserText(finalPrompt) ?? (prompt as string);
    const textPart: TextPart = { type: "text", text: turnText };
    userMessage = {
      role: "user",
      content: [textPart, ...buildImageParts(cachedAttachments)],
    };
    storageUserMessage = {
      role: "user",
      content: [textPart, ...buildPlaceholderParts(cachedAttachments)],
    };
    // Send as a single multimodal user turn (buildCallParams handles arrays).
    finalPrompt = [userMessage];
  } else {
    const userMessageContent =
      typeof finalPrompt === "string"
        ? finalPrompt
        : finalPrompt.find((m) => m.role === "user")?.content || prompt;
    userMessage = { role: "user", content: userMessageContent as string };
    storageUserMessage = userMessage;
  }

  return {
    agent,
    model,
    modelId,
    provider,
    finalPrompt,
    previousMessages,
    userMessage,
    storageUserMessage,
    sessionId,
    sessionConfig,
    startTime,
    stepMetas,
    turnIndex,
    provenanceCollector,
    abortSignal: opts.abortSignal,
  };
}

/** Build the generate/stream call params from the prepared agent state. */
function buildCallParams(prepared: PreparedAgent) {
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

export async function get_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  opts: GetContextOptions
): Promise<ContextResult> {
  const prepared = await prepareAgent(prompt, repoPath, opts);
  const {
    agent,
    model,
    modelId,
    provider,
    finalPrompt,
    sessionId,
    sessionConfig,
    storageUserMessage,
    startTime,
    stepMetas,
    provenanceCollector,
  } = prepared;
  const { schema } = opts;

  let result;
  try {
    result = await agent.generate(buildCallParams(prepared));
  } catch (err) {
    const aborted = isAbortError(err);
    if (sessionId) {
      await appendSessionEnd(sessionId, {
        end_time: new Date().toISOString(),
        model: modelId,
        provider,
        duration_ms: Date.now() - startTime,
        status: aborted ? "aborted" : "error",
        error_message: err instanceof Error ? err.message : String(err),
      });
    }
    throw err;
  }

  const { steps, totalUsage } = result;
  const usage = stepMetas.length > 0
    ? normalizeUsage(addUsage(...stepMetas.map((step) => step.usage)))
    : normalizeUsage(totalUsage);

  const endTime = Date.now();
  const duration = endTime - startTime;

  // Save to session if enabled
  if (sessionId) {
    const newMessages = extractMessagesFromSteps(
      storageUserMessage,
      steps,
      sessionConfig
    );
    appendMessages(sessionId, newMessages);
    appendStepMeta(sessionId, stepMetas);
    if (provenanceCollector.entries.length > 0) {
      appendSearchProvenance(sessionId, provenanceCollector.entries);
    }

    await appendSessionEnd(sessionId, {
      end_time: new Date().toISOString(),
      model: modelId,
      provider,
      duration_ms: duration,
      status: "success",
      token_usage: usage,
    });
  }

  const final = extractFinalAnswer(steps);

  let finalAnswer = final.answer;
  if (schema) {
    console.log("===> structuring final answer with schema", schema);
    finalAnswer = await structureFinalAnswer(
      finalPrompt,
      finalAnswer,
      schema,
      model,
      provider
    );
  }
  console.log(
    `⏱️ get_context completed in ${duration}ms (${(duration / 1000).toFixed(
      2
    )}s)`
  );

  return {
    final: final.answer,
    tool_use: final.tool_use,
    content: finalAnswer,
    usage: {
      ...usage,
      model: modelId,
      provider,
    },
    logs: opts.logs ? JSON.stringify(steps, null, 2) : undefined,
    sessionId,
  };
}

/**
 * Streaming variant of get_context.
 * Returns an object with the stream result and a finalizeSession() function
 * that must be called after the stream is fully consumed to persist the session.
 */
export async function stream_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  opts: GetContextOptions
) {
  const prepared = await prepareAgent(prompt, repoPath, opts);
  const {
    sessionId,
    sessionConfig,
    storageUserMessage,
    modelId,
    provider,
    startTime,
    stepMetas,
    provenanceCollector,
  } = prepared;
  const streamResult = await prepared.agent.stream(buildCallParams(prepared));

  return {
    streamResult,
    async finalizeSession() {
      if (!sessionId) return;
      try {
        const steps = await (streamResult as any).steps;
        const usage = await (streamResult as any).usage;
        const endTime = Date.now();
        const duration = endTime - startTime;
        const newMessages = extractMessagesFromSteps(
          storageUserMessage,
          steps ?? [],
          sessionConfig,
        );
        appendMessages(sessionId, newMessages);
        appendStepMeta(sessionId, stepMetas);
        if (provenanceCollector.entries.length > 0) {
          appendSearchProvenance(sessionId, provenanceCollector.entries);
        }
        const stepUsage = stepMetas.length > 0
          ? normalizeUsage(addUsage(...stepMetas.map((step) => step.usage)))
          : normalizeUsage(usage);
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: duration,
          status: "success",
          token_usage: stepUsage,
        });
      } catch (e) {
        const aborted = isAbortError(e);
        if (aborted) {
          console.log("[stream_context] Stream aborted by client");
        } else {
          console.error("[stream_context] Failed to finalize session:", e);
        }
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status: aborted ? "aborted" : "error",
          error_message: e instanceof Error ? e.message : String(e),
        }).catch(() => {});
      }
    },
  };
}

/*
curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "hi, how are you?"
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "please call the bash tool to make sure it works. List my docker containers currently running."
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "please call the web_search tool to make sure it works. Search for Evan Feenstra online. Then call final_answer to say the answer.",
    "toolsConfig": {
      "web_search": ""
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST -H "Content-Type: application/json" \
  -d '{"request_id":"09e94379-78d4-4d02-b032-3868a7322cc9"}' \
  http://localhost:3355/repo/agent/abort

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "i want to build a user inbox to show recent activity and notifications. Ask me clarifying questions.",
    "toolsConfig": {
      "ask_clarifying_questions": true
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "i want to build a sub-account feature. please tell me a brief technical architecture for this.",
    "toolsConfig": {
      "ask_clarifying_questions": true
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/stakgraph",
    "prompt": "use the file_summary tool to find out what mcp/src/repo/agent.ts does"
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "What workflows are available?",
    "mcpServers": [
      {
        "name": "stak",
        "url": "https://mcp.stakwork.com/mcp",
        "token": "xxx",
        "toolFilter": ["GetWorkflows"]
      }
    ]
  }' \
  "http://localhost:3355/repo/agent"

curl "http://localhost:3355/progress?request_id=5b0e0339-e616-48fc-bd98-8676a556b689"

curl -X POST \
  -H "Content-Type: application/json" \
  -d @mcp/test-results/test.json \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "123sef8sehf8shefs",
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "what do you see? <attachments>https://png.pngtree.com/background/20240824/original/pngtree-blue-and-purple-neon-star-3d-art-background-with-a-cool-picture-image_10210904.jpg</attachments>"
  }' \
  "http://localhost:3355/repo/agent"
  
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "tell me a joke"
  }' \
  "http://localhost:3355/repo/agent"
*/
