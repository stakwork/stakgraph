import { ModelMessage, ToolSet, StepResult, StopCondition } from "ai";
import { SessionConfig, truncateToolResult } from "./session.js";

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

  for (const step of steps) {
    // Build assistant message content
    const assistantContent: any[] = [];

    // Extract text parts
    for (const item of step.content) {
      if (item.type === "text" && item.text) {
        assistantContent.push({ type: "text", text: item.text });
      }
    }

    // Extract tool calls
    for (const item of step.content) {
      if (item.type === "tool-call") {
        assistantContent.push({
          type: "tool-call",
          toolCallId: item.toolCallId,
          toolName: item.toolName,
          args: item.input,
        });
      }
    }

    // Add assistant message if there's content
    if (assistantContent.length > 0) {
      messages.push({
        role: "assistant",
        content: assistantContent,
      });
    }

    // Extract tool results into tool message
    const toolResults: any[] = [];
    for (const item of step.content) {
      if (item.type === "tool-result") {
        let result = item.output;

        // Truncate if config provided
        if (sessionConfig?.truncateToolResults && typeof result === "string") {
          result = truncateToolResult(item.toolName, result, sessionConfig);
        }

        toolResults.push({
          type: "tool-result",
          toolCallId: item.toolCallId,
          toolName: item.toolName,
          result,
        });
      }
    }

    if (toolResults.length > 0) {
      messages.push({
        role: "tool",
        content: toolResults,
      });
    }
  }

  return messages;
}
