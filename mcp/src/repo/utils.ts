import { ModelMessage, ToolSet, StepResult, StopCondition } from "ai";

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

export function logStep(contents: any) {
  // console.log("===> logStep", JSON.stringify(contents, null, 2));
  // return;
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

  // Look for text with [END_OF_ANSWER] sequence
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

  // Fallback to last text if no marker found
  let lastText = "";
  for (const step of steps) {
    for (const item of step.content) {
      if (item.type === "text" && item.text && item.text.trim().length > 0) {
        lastText = item.text.trim();
      }
    }
  }

  if (lastText) {
    console.warn(
      "No [END_OF_ANSWER] marker or ask_clarifying_questions detected; falling back to last text."
    );
    return {
      answer: lastText,
    };
  }

  return { answer: "" };
}
