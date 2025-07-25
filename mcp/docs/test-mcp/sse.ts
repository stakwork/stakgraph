import { experimental_createMCPClient } from "ai";
import * as dotenv from "dotenv";

dotenv.config();

async function runAgent() {
  try {
    const client = await experimental_createMCPClient({
      transport: {
        type: "sse",
        url: "https://repo2graph.swarm38.sphinx.chat/sse",
        // url: "http://localhost:3000/sse",
        headers: {
          Authorization: `Bearer ${process.env.API_TOKEN}`,
          "mcp-session-id": "my-session-id-123",
        },
      },
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
        instruction: "find the name of the #1 user on the leaderboard",
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
