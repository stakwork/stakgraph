import {
  generateText,
  Output,
  ToolLoopAgent,
  ModelMessage,
  StopCondition,
  ToolSet,
  jsonSchema,
} from "ai";
import {
  ModelName,
  getModelDetails,
} from "../aieo/src/index.js";
import { get_tools, ToolsConfig, SkillsConfig, GgnnConfig, MessagesRef } from "./tools.js";
import { SKILLS } from "./skills.js";
import { type SubAgent } from "./subagent.js";
import { ContextResult } from "../tools/types.js";
import {
  appendTextToPrompt,
  logStep,
  extractFinalAnswer,
  createHasEndMarkerCondition,
  createHasAskQuestionsCondition,
  ensureAdditionalPropertiesFalse,
  extractMessagesFromSteps,
  deepParseJsonStrings,
  truncateOldToolResults,
} from "./utils.js";
import { LanguageModel } from "ai";
import {
  createSession as createNewSession,
  loadSessionMessages,
  appendMessages,
  sessionExists,
  SessionConfig,
} from "./session.js";
import { McpServer, getMcpTools } from "./mcpServers.js";

function DEFAULT_SYSTEM(toolsConfig?: ToolsConfig) {

  const learn_concepts = toolsConfig?.learn_concept || toolsConfig?.list_concepts || toolsConfig?.learn_concepts

  return `You are a code exploration assistant with access to a **code knowledge graph**. Use graph tools whenever possible — they are faster, more precise, and understand code relationships.

Try to match the tone of the user. If the user is asking a technical question, research deeper, and respond with technical details. If the user's question is high-level (non-specific), then do not answer with too much detail!

${learn_concepts ? "Use list_concepts and learn_concept tools first, to learn about high-level features in the codebase." : ""}

### Graph Tools
- \`repo_overview\` — Use only for broad orientation or architecture questions; it returns a compact, de-noised repo tree.
- \`stakgraph_search\` — Search by keyword/semantic/hybrid. Returns compact results (name, file, ref_id, description). Use \`node_types\` to filter (e.g. \`["Endpoint"]\`, \`["Function"]\`, \`["DataModel"]\`, \`["UnitTest"]\`).
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
3. \`stakgraph_search\` → find relevant nodes (returns names, ref_ids, descriptions — NOT full code).
4. \`stakgraph_code\` → read source of each relevant node using its ref_id.
5. \`stakgraph_map\` → trace callers/callees when you need to follow a chain.
6. \`bash\` → only for config/env files, directory listings, or bash-only repos.

## Patterns
- **Find endpoints** → \`stakgraph_search({ query: "bounty", node_types: ["Endpoint"] })\`
- **How does X work?** → \`stakgraph_search({ query: "X" })\` → \`stakgraph_code({ ref_id: "..." })\` on each result
- **What calls Y?** → \`stakgraph_map({ name: "Y", node_type: "Function", direction: "up" })\`
- **List data models** → \`stakgraph_search({ query: "model", node_types: ["DataModel"] })\`
- **Find tests** → \`stakgraph_search({ query: "X", node_types: ["UnitTest", "IntegrationTest"] })\`

CRITICAL: When you are ready to provide your final answer, output your complete response followed by [END_OF_ANSWER] on a new line. Don't start your answer with preamble like "Ok! I have all the information I need. Let me create a plan...". Just start with your answer.

Write your answer directly as text and end with [END_OF_ANSWER].`;

};

const ASK_CLARIFYING_QUESTIONS_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt.

CRITICAL: When you finish exploring, you MUST do ONE of these:

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

Otherwise, provide your answer directly followed by [END_OF_ANSWER]. Don't start your answer with preamble like "Ok! I have all the information I need. Let me create a plan...". Just write your answer.`;

async function structureFinalAnswer(
  finalPrompt: string | ModelMessage[],
  finalAnswer: string,
  schema: { [key: string]: any },
  model: LanguageModel
): Promise<any> {
  const msgs: ModelMessage[] = [];

  // Handle finalPrompt - if it's ModelMessage[], push all messages, otherwise create a user message
  if (Array.isArray(finalPrompt)) {
    msgs.push(...finalPrompt);
  } else {
    msgs.push({ role: "user", content: finalPrompt });
  }

  // Add the assistant's answer and the structuring request
  msgs.push(
    { role: "assistant", content: finalAnswer },
    {
      role: "user",
      content:
        "Great! Please rephrase your answer in the following JSON format: " +
        JSON.stringify(schema),
    }
  );

  const normalizedSchema = ensureAdditionalPropertiesFalse(schema);

  const { output } = await generateText({
    model,
    prompt: msgs,
    output: Output.object({ schema: jsonSchema(normalizedSchema) }),
  });

  return deepParseJsonStrings(output ?? {});
}

export interface GetContextOptions {
  modelName?: ModelName;
  apiKey?: string;
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
}

interface PreparedAgent {
  agent: ToolLoopAgent<never, ToolSet>;
  model: LanguageModel;
  modelId: string;
  provider: string;
  finalPrompt: string | ModelMessage[];
  previousMessages: ModelMessage[];
  userMessage: ModelMessage;
  sessionId: string | undefined;
  sessionConfig: SessionConfig | undefined;
  startTime: number;
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
    repos,
    skills,
    subAgents,
    ggnn,
    onStepEvent,
  } = opts;
  const startTime = Date.now();
  const { model, apiKey, provider, contextLimit, modelId } = getModelDetails(modelName, apiKeyIn);
  console.log("===> model", model, "contextLimit", contextLimit);

  const messagesRef: MessagesRef = { current: [] };
  let tools = await get_tools(repoPath, apiKey, pat, toolsConfig, provider, repos, subAgents, ggnn, messagesRef);

  // Load and merge MCP server tools if configured
  if (mcpServers && mcpServers.length > 0) {
    const mcpTools = await getMcpTools(mcpServers);
    tools = { ...tools, ...mcpTools };
    console.log(`[MCP] Merged ${Object.keys(mcpTools).length} MCP tools`);
  }

  let instructions = systemOverride || DEFAULT_SYSTEM(toolsConfig);

  // Append sub-agent instructions if any sub-agents are registered
  if (subAgents && subAgents.length > 0) {
    const validSubAgents = subAgents.filter(sa => sa.name && sa.url && sa.apiToken);
    if (validSubAgents.length > 0) {
      const agentList = validSubAgents
        .map(sa => `  - @${sa.name}: ${sa.description || `Delegates to the "${sa.name}" sub-agent`}`)
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

  let stopWhen: StopCondition<ToolSet> | StopCondition<ToolSet>[] =
    hasEndMarker;
  let finalPrompt: string | ModelMessage[] = prompt;

  if (toolsConfig?.ask_clarifying_questions) {
    instructions = ASK_CLARIFYING_QUESTIONS_SYSTEM;
    stopWhen = [hasEndMarker, hasAskQuestions];

    finalPrompt = appendTextToPrompt(
      prompt,
      " After exploring a bit, ask clarifying questions if needed."
    );
  }

  // Session handling (after instructions are fully assembled so we can persist them)
  let sessionId: string | undefined;
  let previousMessages: ModelMessage[] = [];

  if (inputSessionId) {
    if (sessionExists(inputSessionId)) {
      sessionId = inputSessionId;
      previousMessages = loadSessionMessages(sessionId);
    } else {
      sessionId = createNewSession(inputSessionId, instructions);
    }
  }

  for (const tool of Object.keys(tools)) {
    console.log("===> tool", tool, "===>", tools[tool].description);
  }

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

  const userMessageContent =
    typeof finalPrompt === "string"
      ? finalPrompt
      : finalPrompt.find((m) => m.role === "user")?.content || prompt;
  const userMessage: ModelMessage = {
    role: "user",
    content: userMessageContent as string,
  };

  return { agent, model, modelId, provider, finalPrompt, previousMessages, userMessage, sessionId, sessionConfig, startTime };
}

/** Build the generate/stream call params from the prepared agent state. */
function buildCallParams(prepared: PreparedAgent) {
  const { finalPrompt, previousMessages, userMessage } = prepared;
  if (previousMessages.length > 0) {
    const messagesToSend =
      typeof finalPrompt === "string"
        ? [...previousMessages, userMessage]
        : [...previousMessages, ...finalPrompt];
    return { messages: messagesToSend };
  }
  if (typeof finalPrompt === "string") {
    return { prompt: finalPrompt };
  }
  return { messages: finalPrompt };
}

export async function get_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  opts: GetContextOptions
): Promise<ContextResult> {
  const prepared = await prepareAgent(prompt, repoPath, opts);
  const { agent, model, modelId, provider, finalPrompt, sessionId, sessionConfig, userMessage, startTime } = prepared;
  const { schema } = opts;

  const result = await agent.generate(buildCallParams(prepared));

  const { steps, totalUsage } = result;

  // Save to session if enabled
  if (sessionId) {
    // Extract messages from this turn (user + assistant + tool)
    const newMessages = extractMessagesFromSteps(
      userMessage,
      steps,
      sessionConfig
    );
    appendMessages(sessionId, newMessages);
  }

  const final = extractFinalAnswer(steps);

  let finalAnswer = final.answer;
  if (schema) {
    console.log("===> structuring final answer with schema", schema);
    finalAnswer = await structureFinalAnswer(
      finalPrompt,
      finalAnswer,
      schema,
      model
    );
  }

  const endTime = Date.now();
  const duration = endTime - startTime;
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
      inputTokens: totalUsage.inputTokens || 0,
      outputTokens: totalUsage.outputTokens || 0,
      totalTokens: totalUsage.totalTokens || 0,
      model: modelId,
      provider,
    },
    logs: opts.logs ? JSON.stringify(steps, null, 2) : undefined,
    sessionId,
  };
}

/**
 * Streaming variant of get_context.
 * Returns a StreamTextResult that can be piped to an HTTP response
 * via .toUIMessageStreamResponse().
 */
export async function stream_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  opts: GetContextOptions
) {
  const prepared = await prepareAgent(prompt, repoPath, opts);
  return prepared.agent.stream(buildCallParams(prepared));
}

/*
curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"

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
*/
