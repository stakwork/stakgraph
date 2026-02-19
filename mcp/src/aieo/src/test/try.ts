import { generateText, type ModelMessage } from "ai";

import * as aieo from "../../dist/index.js";

import * as dotenv from "dotenv";

dotenv.config({ path: "../../.env" });

async function doTheThing() {
  try {
    const systemMessage: ModelMessage = {
      role: "system",
      content: "you are an expert developer",
    };
    const content = "what is your name?";
    const userMessageContent: ModelMessage = {
      role: "user",
      content,
    };
    const messages = [systemMessage, userMessageContent];

    const provider: aieo.Provider = "anthropic";

    const model = aieo.getModel(provider, {
      // modelName: "opus",
      logger: aieo.consoleLogger("aieo"),
    });

    const res = await generateText({
      model,
      messages,
    });

    console.log("Response:", res);
  } catch (error) {
    console.log("Model error:", error);
  }
}

doTheThing().catch((error) => {
  console.error("Error occurred while calling model:", error);
});
