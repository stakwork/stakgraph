import { type ModelMessage } from "ai";

import * as aieo from "../../dist/index.js";

import * as dotenv from "dotenv";

dotenv.config({ path: "../.env" });

async function doTheThing() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("Missing ANTHROPIC_API_KEY");
    return;
  }
  try {
    const systemMessage: ModelMessage = {
      role: "system",
      content: "you are an expert developer",
    };
    const userMessageContent: ModelMessage = {
      role: "user",
      content:
        "please add a comment on top of src/store.ts explain wha the file does",
    };
    const messages = [systemMessage, userMessageContent];

    const provider: aieo.Provider = "claude_code";
    const res = await aieo.callModel({
      provider,
      apiKey,
      messages,
      cwd: "/Users/evanfeenstra/code/sphinx2/stakgraph/mcp/aieo",
    });
    console.log("Response:", res);
  } catch (error) {
    console.log("Model error:", error);
  }
}

doTheThing().catch((error) => {
  console.error("Error occurred while calling model:", error);
});
