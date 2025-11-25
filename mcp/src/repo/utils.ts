import { ModelMessage, ToolSet, StepResult } from "ai";

export function logStep(contents: any) {
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

export function extractFinalAnswer(steps: StepResult<ToolSet>[]): string {
  let lastText = "";

  // Collect last text content as fallback
  for (const step of steps) {
    for (const item of step.content) {
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
      return (askQuestionsResult as any).output;
    }

    // Then check for final_answer
    const finalAnswerResult = steps[i].content.find(
      (c) => c.type === "tool-result" && c.toolName === "final_answer"
    );
    if (finalAnswerResult) {
      return (finalAnswerResult as any).output;
    }
  }

  // Fallback to last text if neither tool was found
  if (lastText) {
    console.warn(
      "No final_answer or ask_clarifying_questions tool call detected; falling back to last text."
    );
    return `${lastText}\n\n(Note: Model did not invoke final_answer tool; using last text as final answer.)`;
  }

  return "";
}
