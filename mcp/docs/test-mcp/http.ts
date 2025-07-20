import { experimental_createMCPClient } from "ai";
import * as dotenv from "dotenv";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { randomUUID } from "crypto";

dotenv.config();

async function runAgent() {
  try {
    const client = await experimental_createMCPClient({
      transport: new StreamableHTTPClientTransport(
        new URL("http://localhost:3000/mcp"),
        {
          requestInit: {
            headers: {
              authorization: `Bearer asdfasdf`,
              "x-session-id": randomUUID(),
            },
          },
        }
      ),
    });

    const tools = await client.tools();
    console.log("MCP tools available:", Object.keys(tools));

    const result = await tools.stagehand_navigate.execute(
      {
        url: "https://community.sphinx.chat/leaderboard",
      },
      {
        toolCallId: "1",
        messages: [],
      }
    );
    console.log(result);

    const result2 = await tools.stagehand_observe.execute(
      {
        instruction: "describe the page",
      },
      {
        toolCallId: "2",
        messages: [],
      }
    );
    console.log(result2);
  } catch (error) {
    console.error("Error setting up MCP client:", error);
  }
}

runAgent();
