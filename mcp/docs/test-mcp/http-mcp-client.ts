import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import * as dotenv from "dotenv";

dotenv.config();

async function runAgent() {
  try {
    const transport = new StreamableHTTPClientTransport(
      new URL("http://localhost:3000/mcp"),
      {
        requestInit: {
          headers: {
            "mcp-session-id": "test-session-123",
            authorization: `Bearer asdfasdf`,
          },
        },
      }
    );

    const client = new Client({
      name: "test-client",
      version: "1.0.0",
    }, {
      capabilities: {}
    });

    await client.connect(transport);

    const result = await client.callTool({
      name: "stagehand_navigate",
      arguments: { url: "https://github.com/anthropics/anthropic-sdk-python" }
    });
    console.log("Navigate 1:", result);

    const result2 = await client.callTool({
      name: "stagehand_network_activity",
      arguments: {}
    });
    console.log("Network activity after navigation:", result2);

    const result4 = await client.callTool({
      name: "stagehand_act",
      arguments: { action: "Click on the README.md file" }
    });
    console.log("act:", result4);

    await new Promise((resolve) => setTimeout(resolve, 5000));

    const result5 = await client.callTool({
      name: "stagehand_network_activity",
      arguments: {}
    });
    console.log("Network activity after act:", result5);

    await client.close();
  } catch (error) {
    console.error("Error setting up MCP client:", error);
  }
}

runAgent();