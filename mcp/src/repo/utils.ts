import { ModelMessage, ToolSet, StepResult } from "ai";

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
  let lastText = "";
  let foundFinalAnswerCall = false;
  let textAfterFinalAnswer = "";

  // Iterate through all steps to find final_answer calls and text
  for (const step of steps) {
    for (const item of step.content) {
      // Track if we found a final_answer tool call
      if (item.type === "tool-call" && item.toolName === "final_answer") {
        foundFinalAnswerCall = true;
        textAfterFinalAnswer = ""; // Reset text after finding tool call
      }

      // Capture text after final_answer call
      if (
        foundFinalAnswerCall &&
        item.type === "text" &&
        item.text &&
        item.text.trim().length > 0
      ) {
        textAfterFinalAnswer = item.text.trim();
      }

      // Keep track of last text as general fallback
      if (item.type === "text" && item.text && item.text.trim().length > 0) {
        lastText = item.text.trim();
      }
    }
  }

  // Search for ask_clarifying_questions or final_answer tool result (reverse order for efficiency)
  for (let i = steps.length - 1; i >= 0; i--) {
    // Check for ask_clarifying_questions first (higher priority)
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

    // Then check for final_answer tool result
    const finalAnswerResult = steps[i].content.find(
      (c) => c.type === "tool-result" && c.toolName === "final_answer"
    );
    if (finalAnswerResult) {
      const output = (finalAnswerResult as any).output;
      // Only return if output is not empty, otherwise fall through to text after tool call
      if (output && output.trim()) {
        return {
          answer: output,
          tool_use: "final_answer",
        };
      }
    }
  }

  // If final_answer was called but failed, use text that came after it
  if (foundFinalAnswerCall && textAfterFinalAnswer) {
    return {
      answer: textAfterFinalAnswer,
      tool_use: "final_answer",
    };
  }

  // Fallback to last text if neither tool was found
  if (lastText) {
    console.warn(
      "No final_answer or ask_clarifying_questions tool call detected; falling back to last text."
    );
    return {
      answer: `${lastText}\n\n(Note: Model did not invoke final_answer tool; using last text as final answer.)`,
    };
  }

  return { answer: "" };
}
