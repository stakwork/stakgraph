import { ModelMessage, ToolSet, StepResult, StopCondition } from "ai";
import { SessionConfig, truncateToolResult } from "./session.js";
import { createByModelName, TikTokenizer } from "@microsoft/tiktokenizer";

const TRUNCATED = "<TRUNCATED>";
const SLIMMED_PREFIX = "[SLIMMED";
const CONTEXT_THRESHOLD = 0.9;
const RECENT_TOOL_RESULTS = 6;

// Singleton tokenizer instance (lazy-initialized)
let _tokenizer: TikTokenizer | null = null;
let _tokenizerPromise: Promise<TikTokenizer> | null = null;

export async function getTokenizer(): Promise<TikTokenizer> {
  if (_tokenizer) return _tokenizer;
  if (!_tokenizerPromise) {
    _tokenizerPromise = createByModelName("gpt-4").then((t) => {
      _tokenizer = t;
      return t;
    });
  }
  return _tokenizerPromise;
}

/**
 * Count tokens using the real tokenizer (sync, after init).
 * Falls back to a conservative estimate if tokenizer isn't ready yet.
 */
function countTokens(text: string): number {
  if (_tokenizer) {
    return _tokenizer.encode(text).length;
  }
  // Conservative fallback: ~3 chars per token (errs on the side of overcounting)
  return Math.ceil(text.length / 3);
}

export function createHasEndMarkerCondition<
  T extends ToolSet
>(): StopCondition<T> {
  return ({ steps }) => {
    for (const step of steps) {
      for (const item of step.content) {
        if (item.type === "text" && item.text?.includes("[END_OF_ANSWER]")) {
          return true;
        }
      }
    }
    return false;
  };
}

export function createHasAskQuestionsCondition<
  T extends ToolSet
>(): StopCondition<T> {
  return ({ steps }) => {
    for (const step of steps) {
      for (const item of step.content) {
        // Check for successful ask_clarifying_questions call
        if (
          item.type === "tool-result" &&
          item.toolName === "ask_clarifying_questions"
        ) {
          return true;
        }
      }
    }
    return false;
  };
}

export function ensureAdditionalPropertiesFalse(schema: {
  [key: string]: any;
}): { [key: string]: any } {
  const result = { ...schema };

  if (result.type === "object" && result.additionalProperties === undefined) {
    result.additionalProperties = false;
  }

  // Recursively process properties
  if (result.properties) {
    result.properties = Object.keys(result.properties).reduce((acc, key) => {
      acc[key] = ensureAdditionalPropertiesFalse(result.properties[key]);
      return acc;
    }, {} as { [key: string]: any });
  }

  // Recursively process array items
  if (result.items) {
    result.items = ensureAdditionalPropertiesFalse(result.items);
  }

  // Recursively process anyOf, allOf, oneOf
  if (result.anyOf) {
    result.anyOf = result.anyOf.map(ensureAdditionalPropertiesFalse);
  }
  if (result.allOf) {
    result.allOf = result.allOf.map(ensureAdditionalPropertiesFalse);
  }
  if (result.oneOf) {
    result.oneOf = result.oneOf.map(ensureAdditionalPropertiesFalse);
  }

  return result;
}

export function logStep(contents: any): void {
  if (!Array.isArray(contents)) return;
  for (const item of contents) {
    if (item.type === "tool-call") {
      console.log(`[repo_agent] tool_call: ${item.toolName}`);
    } else if (item.type === "text" && item.text) {
      console.log(`[repo_agent] text: ${item.text.slice(0, 120).replace(/\n/g, " ")}...`);
    }
  }
}

export function appendTextToPrompt(
  prompt: string | ModelMessage[],
  textToAppend: string
): string | ModelMessage[] {
  if (typeof prompt === "string") {
    return prompt + textToAppend;
  }

  if (!Array.isArray(prompt)) {
    return prompt;
  }

  // Find the last user message and append to it
  const modifiedPrompt = [...prompt];
  for (let i = modifiedPrompt.length - 1; i >= 0; i--) {
    const message = modifiedPrompt[i];
    if (message.role === "user") {
      if (typeof message.content === "string") {
        modifiedPrompt[i] = {
          ...message,
          content: message.content + textToAppend,
        } as ModelMessage;
      } else if (Array.isArray(message.content)) {
        // If content is an array of parts, append to the last text part
        const contentCopy = [...message.content];
        for (let j = contentCopy.length - 1; j >= 0; j--) {
          const part = contentCopy[j];
          if (part.type === "text") {
            contentCopy[j] = {
              type: "text",
              text: part.text + textToAppend,
            };
            break;
          }
        }
        modifiedPrompt[i] = {
          ...message,
          content: contentCopy,
        } as ModelMessage;
      }
      break;
    }
  }

  return modifiedPrompt;
}

export interface FinalAnswerResult {
  answer: any;
  tool_use?: string;
}

export function extractFinalAnswer(
  steps: StepResult<ToolSet>[]
): FinalAnswerResult {
  // Search for ask_clarifying_questions tool result (highest priority)
  for (let i = steps.length - 1; i >= 0; i--) {
    const askQuestionsResult = steps[i].content.find(
      (c) =>
        c.type === "tool-result" && c.toolName === "ask_clarifying_questions"
    );
    if (askQuestionsResult) {
      return {
        answer: (askQuestionsResult as any).output,
        tool_use: "ask_clarifying_questions",
      };
    }
  }

  // Look for text with [END_OF_ANSWER] sequence (search all text)
  let allText = "";
  for (const step of steps) {
    for (const item of step.content) {
      if (item.type === "text" && item.text) {
        allText += item.text;
      }
    }
  }

  const endMarkerIndex = allText.indexOf("[END_OF_ANSWER]");
  if (endMarkerIndex !== -1) {
    const answer = allText.substring(0, endMarkerIndex).trim();
    if (answer) {
      return {
        answer,
        tool_use: "text_with_end_marker",
      };
    }
  }

  // Fallback: collect all text after the last tool call
  let lastToolStepIndex = -1;
  let lastToolContentIndex = -1;

  // Find the last tool-call or tool-result
  for (let i = steps.length - 1; i >= 0; i--) {
    for (let j = steps[i].content.length - 1; j >= 0; j--) {
      const item = steps[i].content[j];
      if (item.type === "tool-call" || item.type === "tool-result") {
        lastToolStepIndex = i;
        lastToolContentIndex = j;
        break;
      }
    }
    if (lastToolStepIndex !== -1) break;
  }

  // Collect all text after the last tool
  let textAfterLastTool = "";
  let startCollecting = false;

  for (let i = 0; i < steps.length; i++) {
    for (let j = 0; j < steps[i].content.length; j++) {
      const item = steps[i].content[j];

      // Start collecting after we've passed the last tool
      if (i === lastToolStepIndex && j === lastToolContentIndex) {
        startCollecting = true;
        continue;
      }

      if (startCollecting && item.type === "text" && item.text) {
        textAfterLastTool += item.text;
      }
    }
  }

  const trimmedTextAfterLastTool = textAfterLastTool.trim();
  if (trimmedTextAfterLastTool) {
    console.warn(
      "No [END_OF_ANSWER] marker or ask_clarifying_questions detected; falling back to text after last tool call."
    );
    return {
      answer: trimmedTextAfterLastTool,
    };
  }

  // If no tools were found, fall back to all text
  const trimmedAllText = allText.trim();
  if (trimmedAllText) {
    console.warn(
      "No tools found; falling back to all text."
    );
    return {
      answer: trimmedAllText,
    };
  }

  return { answer: "" };
}

/**
 * Convert a user message + generateText steps into ModelMessage[] for session storage.
 * This captures the full turn: user message, assistant responses, and tool results.
 */
export function extractMessagesFromSteps(
  userMessage: ModelMessage,
  steps: StepResult<ToolSet>[],
  sessionConfig?: SessionConfig
): ModelMessage[] {
  const messages: ModelMessage[] = [userMessage];

  // Use the SDK's own accumulated response messages from the last step.
  // These are produced by toResponseMessages() which correctly handles
  // provider-executed tools (web_search, code_execution, etc.) by keeping
  // their tool-results in the assistant message and client-executed
  // tool-results in the tool message. Manually reconstructing from
  // step.content loses providerExecuted flags and deferred result ordering,
  // causing "tool_use ids found without tool_result blocks" errors on
  // session replay with the Anthropic API.
  const lastStep = steps[steps.length - 1];
  if (!lastStep) return messages;

  const responseMessages = lastStep.response.messages as ModelMessage[];
  for (const msg of responseMessages) {
    if (sessionConfig?.truncateToolResults && msg.role === "tool") {
      // Truncate client-executed tool results for storage efficiency
      const truncatedContent = (msg.content as any[]).map((part: any) => {
        if (part.type === "tool-result" && part.output) {
          const output = part.output;
          if (output.type === "text" && typeof output.value === "string") {
            return {
              ...part,
              output: {
                ...output,
                value: truncateToolResult(
                  part.toolName,
                  output.value,
                  sessionConfig
                ),
              },
            };
          }
        }
        return part;
      });
      messages.push({ ...msg, content: truncatedContent } as ModelMessage);
    } else {
      messages.push(msg);
    }
  }

  return messages;
}

/**
 * Strip trailing garbage characters after the last matching bracket.
 * LLMs sometimes append extra chars like `\n"` after valid JSON.
 */
function stripTrailingGarbage(s: string): string {
  if (s.startsWith("[")) {
    const lastBracket = s.lastIndexOf("]");
    if (lastBracket > 0) return s.substring(0, lastBracket + 1);
  } else if (s.startsWith("{")) {
    const lastBrace = s.lastIndexOf("}");
    if (lastBrace > 0) return s.substring(0, lastBrace + 1);
  }
  return s;
}

/**
 * Estimate token count for a string using real tokenizer or conservative fallback.
 */
function estimateTokens(text: string): number {
  return countTokens(text);
}

/**
 * Estimate the total tokens in a tool-result message part's output.
 */
function getToolResultSize(part: any): number {
  if (!part?.output) return 0;
  if (part.output.type === "text" && typeof part.output.value === "string") {
    return estimateTokens(part.output.value);
  }
  if (part.output.type === "json") {
    return estimateTokens(JSON.stringify(part.output.value));
  }
  return 0;
}

/**
 * Check if a tool result part is already truncated or slimmed.
 */
function isAlreadyTruncated(part: any): boolean {
  if (part?.output?.type !== "text") return false;
  const v = part.output.value;
  return v === TRUNCATED || (typeof v === "string" && v.startsWith(SLIMMED_PREFIX));
}

function slimSearchResult(text: string): string {
  try {
    const items = JSON.parse(text);
    if (!Array.isArray(items)) return text;
    const slimmed = items.map((item: any) => ({
      name: item.name,
      node_type: item.node_type,
      file: item.file,
      ref_id: item.ref_id,
    }));
    return `${SLIMMED_PREFIX} ${items.length} search results — descriptions stripped, re-call stakgraph_code with ref_id to read source]\n${JSON.stringify(slimmed)}`;
  } catch {
    return text;
  }
}

function slimCodeResult(text: string): string {
  const lines = text.split("\n");
  if (lines.length <= 5) return text;
  const sig = lines.slice(0, 3).join("\n");
  return `${SLIMMED_PREFIX} source code ${lines.length} lines — showing signature only, re-call stakgraph_code to read full source]\n${sig}\n[... ${lines.length - 3} more lines]`;
}

function slimBashResult(text: string): string {
  const lines = text.split("\n");
  if (lines.length <= 8) return text;
  const head = lines.slice(0, 3).join("\n");
  const tail = lines.slice(-3).join("\n");
  return `${SLIMMED_PREFIX} bash output ${lines.length} lines — showing first/last 3 lines]\n${head}\n[... ${lines.length - 6} lines omitted ...]\n${tail}`;
}

function slimFulltextResult(text: string): string {
  const lines = text.split("\n");
  if (lines.length <= 6) return text;
  const kept = lines.slice(0, 5).join("\n");
  return `${SLIMMED_PREFIX} fulltext_search ${lines.length} matches — showing top 5]\n${kept}\n[... ${lines.length - 5} more matches]`;
}

function slimRepoOverview(text: string): string {
  return `${SLIMMED_PREFIX} repo_overview was here — call repo_overview to refresh]`;
}

export function slimToolResult(toolName: string, output: string): string {
  switch (toolName) {
    case "stakgraph_search":
    case "vector_search":
      return slimSearchResult(output);
    case "stakgraph_code":
      return slimCodeResult(output);
    case "bash":
      return slimBashResult(output);
    case "fulltext_search":
      return slimFulltextResult(output);
    case "repo_overview":
      return slimRepoOverview(output);
    default:
      return output;
  }
}

/**
 * Estimate total tokens in a messages array by serializing the entire message.
 * We JSON.stringify each message to capture all structural overhead (role markers,
 * tool IDs, tool names, content part framing, etc.) — not just the text payloads.
 */
export function estimateMessagesTokens(messages: ModelMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    total += estimateTokens(JSON.stringify(msg));
  }
  return total;
}

/**
 * Collect all tool-result locations in the messages array.
 * Returns them in order (earliest first).
 */
function collectToolResults(
  messages: ModelMessage[]
): { msgIdx: number; partIdx: number; toolName: string; tokens: number }[] {
  const results: { msgIdx: number; partIdx: number; toolName: string; tokens: number }[] = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i] as any;
    if (msg.role !== "tool" || !Array.isArray(msg.content)) continue;
    for (let j = 0; j < msg.content.length; j++) {
      const part = msg.content[j];
      if (part.type === "tool-result" && !isAlreadyTruncated(part)) {
        results.push({
          msgIdx: i,
          partIdx: j,
          toolName: part.toolName ?? "unknown",
          tokens: getToolResultSize(part),
        });
      }
    }
  }
  return results;
}

function getToolResultText(part: any): string | undefined {
  if (part?.output?.type === "text" && typeof part.output.value === "string") {
    return part.output.value;
  }
  if (part?.output?.type === "json") {
    try { return JSON.stringify(part.output.value); } catch { return undefined; }
  }
  return undefined;
}

function isErrorOutput(part: any): boolean {
  const text = getToolResultText(part);
  if (!text) return false;
  return /^error[:\s]/i.test(text) || /failed|exception|traceback/i.test(text.slice(0, 200));
}

/**
 * Proactively slim old tool results to reduce context size while preserving
 * message structure (no cache busting). The most recent RECENT_TOOL_RESULTS
 * tool results and any error outputs are left untouched. Everything older
 * gets per-tool-type compression: search results lose descriptions, code
 * keeps only signatures, bash keeps head/tail, repo_overview is replaced
 * with a refresh hint.
 *
 * Runs on every step via prepareStep. Always safe — slimming preserves
 * the message count, roles, and tool-call IDs.
 */
export function slimOldToolResults(messages: ModelMessage[]): ModelMessage[] {
  const allResults = collectToolResults(messages);
  if (allResults.length <= RECENT_TOOL_RESULTS) return messages;

  const slimmable = allResults.slice(0, allResults.length - RECENT_TOOL_RESULTS);
  const result = [...messages];
  let slimmedCount = 0;

  for (const { msgIdx, partIdx, toolName } of slimmable) {
    const part = (messages[msgIdx] as any).content[partIdx];
    if (isErrorOutput(part)) continue;

    const text = getToolResultText(part);
    if (!text) continue;

    const slimmed = slimToolResult(toolName, text);
    if (slimmed === text) continue;

    if (result[msgIdx] === messages[msgIdx]) {
      const msg = messages[msgIdx] as any;
      result[msgIdx] = { ...msg, content: [...msg.content] } as ModelMessage;
    }
    (result[msgIdx] as any).content[partIdx] = {
      ...part,
      output: { type: "text" as const, value: slimmed },
    };
    slimmedCount++;
  }

  if (slimmedCount > 0) {
    console.log(
      `[context] slimmed ${slimmedCount} old tool result(s), protected last ${RECENT_TOOL_RESULTS}`
    );
  }
  return slimmedCount > 0 ? result : messages;
}

/**
 * Emergency truncation: nuke old tool results with <TRUNCATED> when
 * context usage exceeds 90% of the window. This is the last resort
 * after slimming has already run.
 */
export async function truncateOldToolResults(
  messages: ModelMessage[],
  inputTokens: number,
  contextLimit: number
): Promise<ModelMessage[]> {
  await getTokenizer().catch(() => {});

  const estimated = estimateMessagesTokens(messages);
  const lastInputTokens = inputTokens > 0 ? inputTokens : estimated;
  const threshold = contextLimit * CONTEXT_THRESHOLD;

  console.log(
    `[truncate] providerTokens=${inputTokens} estimated=${estimated} threshold=${Math.round(threshold)} contextLimit=${contextLimit} messages=${messages.length} tokenizer=${_tokenizer ? 'real' : 'fallback'}`
  );

  // First pass: proactive slimming (always runs, preserves structure)
  let current = slimOldToolResults(messages);

  // Re-estimate after slimming
  const estimatedAfterSlim = estimateMessagesTokens(current);
  if (lastInputTokens < threshold && estimatedAfterSlim < threshold) {
    return current;
  }

  // Second pass: emergency nuke (only if still over 90% after slimming)
  const worstCase = Math.max(lastInputTokens, estimatedAfterSlim);
  const excess = worstCase - threshold;
  let tokensToFree = Math.ceil(excess * 1.1);

  console.warn(
    `[context] EMERGENCY truncation after slimming. worstCase=${worstCase} excess=${excess} tokensToFree=${tokensToFree}`
  );

  const candidates = collectToolResults(current)
    .filter((c) => c.tokens > 0);

  if (candidates.length === 0) {
    console.log(`[truncate] WARNING: over limit but no tool results to truncate!`);
    return current;
  }

  const result = [...current];
  let freedTokens = 0;
  let truncatedCount = 0;

  for (const { msgIdx, partIdx, tokens } of candidates) {
    if (freedTokens >= tokensToFree) break;

    if (result[msgIdx] === current[msgIdx]) {
      const msg = current[msgIdx] as any;
      result[msgIdx] = { ...msg, content: [...msg.content] } as ModelMessage;
    }
    const part = (result[msgIdx] as any).content[partIdx];
    (result[msgIdx] as any).content[partIdx] = {
      ...part,
      output: { type: "text" as const, value: TRUNCATED },
    };

    freedTokens += tokens;
    truncatedCount++;
  }

  const estimatedAfter = estimateMessagesTokens(result);
  console.log(
    `[truncate] Freed ~${Math.round(freedTokens)} tokens by truncating ${truncatedCount} tool result(s). estimatedAfter=${estimatedAfter}`
  );

  return result;
}

export function deepParseJsonStrings(value: any): any {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      // Try parsing as-is first
      try {
        const parsed = JSON.parse(trimmed);
        if (typeof parsed === "object" && parsed !== null) {
          return deepParseJsonStrings(parsed);
        }
      } catch {
        // Try again after stripping trailing garbage (e.g. `\n"`)
        const cleaned = stripTrailingGarbage(trimmed);
        if (cleaned !== trimmed) {
          try {
            const parsed = JSON.parse(cleaned);
            if (typeof parsed === "object" && parsed !== null) {
              return deepParseJsonStrings(parsed);
            }
          } catch {
            // Still not valid JSON — leave as-is
          }
        }
      }
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((v: any) => deepParseJsonStrings(v));
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, deepParseJsonStrings(v)])
    );
  }
  return value;
}

/**
 * Extract the first balanced top-level JSON object from a string.
 * Agents often emit `{...}\n\n---\n\n## markdown` — this grabs the leading object.
 * Returns the parsed object, or null if none is found / it isn't valid JSON.
 */
export function extractLeadingJsonObject(text: string): any | null {
  const trimmed = text.trim();
  const start = trimmed.indexOf("{");
  if (start === -1) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < trimmed.length; i++) {
    const ch = trimmed[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === "{") {
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0) {
        const candidate = trimmed.slice(start, i + 1);
        try {
          const parsed = JSON.parse(candidate);
          return typeof parsed === "object" && parsed !== null ? parsed : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/**
 * Returns true if `obj` already satisfies the top-level shape of `schema`:
 * it is a plain object, contains every `required` key, and introduces no keys
 * outside `properties`. Used to short-circuit the structuring LLM call when the
 * agent already produced a schema-shaped object.
 */
export function matchesSchemaShape(
  obj: any,
  schema: { [key: string]: any }
): boolean {
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
    return false;
  }
  if (!schema || schema.type !== "object" || !schema.properties) {
    return false;
  }
  const propKeys = Object.keys(schema.properties);
  const objKeys = Object.keys(obj);
  if (objKeys.length === 0) return false;
  // No unknown keys.
  for (const key of objKeys) {
    if (!propKeys.includes(key)) return false;
  }
  // All required keys present.
  const required: string[] = Array.isArray(schema.required)
    ? schema.required
    : [];
  for (const key of required) {
    if (!(key in obj)) return false;
  }
  return true;
}

/**
 * Walk a JSON schema and collect every enum constraint as a `path -> values`
 * pair. Used to remind the structuring model that enum/categorical fields must
 * be copied verbatim from the answer, not re-classified.
 */
export function collectEnumConstraints(
  schema: { [key: string]: any },
  pathPrefix = ""
): { path: string; values: any[] }[] {
  const out: { path: string; values: any[] }[] = [];
  if (!schema || typeof schema !== "object") return out;

  if (Array.isArray(schema.enum)) {
    out.push({ path: pathPrefix || "(root)", values: schema.enum });
  }
  if (schema.properties) {
    for (const key of Object.keys(schema.properties)) {
      const childPath = pathPrefix ? `${pathPrefix}.${key}` : key;
      out.push(...collectEnumConstraints(schema.properties[key], childPath));
    }
  }
  if (schema.items) {
    out.push(...collectEnumConstraints(schema.items, `${pathPrefix}[]`));
  }
  for (const combiner of ["anyOf", "allOf", "oneOf"] as const) {
    if (Array.isArray(schema[combiner])) {
      for (const sub of schema[combiner]) {
        out.push(...collectEnumConstraints(sub, pathPrefix));
      }
    }
  }
  return out;
}
