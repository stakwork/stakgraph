import { experimental_createMCPClient } from "ai";
import * as dotenv from "dotenv";

dotenv.config();

async function runAgent() {
  try {
    const client = await experimental_createMCPClient({
      transport: {
        type: "sse",
        // url: "https://repo2graph.swarm38.sphinx.chat/sse",
        url: "http://localhost:3000/sse",
        headers: {
          Authorization: `Bearer ${process.env.API_TOKEN}`,
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

    // const instruction = "click on the 'Ivana' user and describe the profile page that appears"
    const instruction =
      "find the bounties section, and filter by Assigned. Then click on the first bounty and give a one sentence solution summary";

    const result2 = await tools.stagehand_agent.execute(
      {
        instruction,
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
