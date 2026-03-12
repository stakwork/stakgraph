import { ModelMessage, ToolSet, StepResult, StopCondition } from "ai";
import { SessionConfig, truncateToolResult } from "./session.js";
import { createByModelName, TikTokenizer } from "@microsoft/tiktokenizer";

const TRUNCATED = "<TRUNCATED>";
const CONTEXT_THRESHOLD = 0.9;

// Singleton tokenizer instance (lazy-initialized)
let _tokenizer: TikTokenizer | null = null;
let _tokenizerPromise: Promise<TikTokenizer> | null = null;

async function getTokenizer(): Promise<TikTokenizer> {
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

export function logStep(contents: any) {
  console.log("===> logStep", JSON.stringify(contents, null, 2));
  return;
  if (!Array.isArray(contents)) return;
  for (const content of contents) {
    if (content.type === "tool-call" && content.toolName !== "final_answer") {
      console.log("TOOL CALL:", content.toolName, ":", content.input);
    }
    console.log("CONTENT:", JSON.stringify(content, null, 2));
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
 * Check if a tool result part is already truncated.
 */
function isAlreadyTruncated(part: any): boolean {
  if (part?.output?.type === "text" && part.output.value === TRUNCATED) {
    return true;
  }
  return false;
}

/**
 * Estimate total tokens in a messages array by serializing all content.
 */
export function estimateMessagesTokens(messages: ModelMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    const m = msg as any;
    if (typeof m.content === "string") {
      total += estimateTokens(m.content);
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (part.type === "text" && typeof part.text === "string") {
          total += estimateTokens(part.text);
        } else if (part.type === "tool-result") {
          total += getToolResultSize(part);
        } else if (part.type === "tool-call") {
          total += estimateTokens(JSON.stringify(part.input ?? ""));
        } else {
          total += estimateTokens(JSON.stringify(part));
        }
      }
    }
  }
  return total;
}

/**
 * Truncate old tool results from the beginning of a messages array
 * to bring token usage under the context limit threshold (90%).
 *
 * @param messages - The full messages array about to be sent to the model.
 * @param inputTokens - Actual input tokens from the provider (from the last step),
 *                       or 0 if unavailable (in which case we estimate from messages).
 * @param contextLimit - The model's context window size.
 *
 * Returns a new messages array (or the original if no truncation needed).
 * Tool results are replaced with "<TRUNCATED>" starting from the earliest
 * messages, skipping the most recent tool results.
 */
export async function truncateOldToolResults(
  messages: ModelMessage[],
  inputTokens: number,
  contextLimit: number
): Promise<ModelMessage[]> {
  // Eagerly initialize the tokenizer so countTokens() is accurate
  await getTokenizer().catch(() => {});

  // Use provider-reported tokens if available, otherwise estimate
  const estimated = estimateMessagesTokens(messages);
  const lastInputTokens = inputTokens > 0
    ? inputTokens
    : estimated;
  const threshold = contextLimit * CONTEXT_THRESHOLD;
  // Also check our own estimate in case the provider count is stale
  if (lastInputTokens < threshold && estimated < threshold) {
    return messages;
  }

  const worstCase = Math.max(lastInputTokens, estimated);
  const excess = worstCase - threshold;
  // Add 10% safety margin to account for estimation inaccuracy
  let tokensToFree = Math.ceil(excess * 1.1);

  console.log(
    `[truncate] Input tokens ${lastInputTokens} >= ${Math.round(threshold)} (90% of ${contextLimit}). ` +
    `Need to free ~${Math.round(tokensToFree)} tokens.`
  );

  // Collect all truncatable tool-result locations (earliest first).
  // Each entry: { msgIdx, partIdx, tokens }
  const candidates: { msgIdx: number; partIdx: number; tokens: number }[] = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i] as any;
    if (msg.role !== "tool") continue;
    if (!Array.isArray(msg.content)) continue;
    for (let j = 0; j < msg.content.length; j++) {
      const part = msg.content[j];
      if (part.type === "tool-result" && !isAlreadyTruncated(part)) {
        const tokens = getToolResultSize(part);
        if (tokens > 0) {
          candidates.push({ msgIdx: i, partIdx: j, tokens });
        }
      }
    }
  }

  if (candidates.length === 0) {
    return messages;
  }

  // Deep copy only the messages we need to modify
  const result = [...messages];
  let freedTokens = 0;

  // Truncate from the beginning (oldest tool results first)
  for (const { msgIdx, partIdx, tokens } of candidates) {
    if (freedTokens >= tokensToFree) break;

    // Lazy deep-copy the message on first mutation
    if (result[msgIdx] === messages[msgIdx]) {
      const msg = messages[msgIdx] as any;
      result[msgIdx] = {
        ...msg,
        content: [...msg.content],
      } as ModelMessage;
    }
    const msgContent = (result[msgIdx] as any).content;
    const part = msgContent[partIdx];
    // Replace with truncated marker
    msgContent[partIdx] = {
      ...part,
      output: { type: "text" as const, value: TRUNCATED },
    };

    freedTokens += tokens;
  }

  let truncatedCount = 0;
  let sum = 0;
  for (const c of candidates) {
    if (sum >= tokensToFree) break;
    sum += c.tokens;
    truncatedCount++;
  }

  console.log(
    `[truncate] Freed ~${Math.round(freedTokens)} estimated tokens by truncating ${truncatedCount} tool result(s).`
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
