

import { createMCPClient } from "@ai-sdk/mcp";

const MCP_URL = process.env.MCP_URL || "http://localhost:3000/mcp?apiKey=hive_cmmc_aVKYoL7BLSNIRqakjJhSb4KWAfdT9AJz";

async function testMcpServerDirect() {
  console.log("=== MCP Server Direct Test ===");
  console.log(`URL: ${MCP_URL}`);
  console.log("");

  try {
    const headers: Record<string, string> = {};

    console.log("Creating MCP client...");
    const mcpClient = await createMCPClient({
      transport: {
        type: "http",
        url: MCP_URL,
        headers,
      },
    });

    console.log("Listing tools...");
    const tools = await mcpClient.tools();
    const toolNames = Object.keys(tools);
    console.log(`Found ${toolNames.length} tools:`);
    toolNames.forEach((name) => {
      console.log(`  - ${name}`);
    });
    console.log("");

    await mcpClient.close();
    console.log("\nMCP client closed.");
  } catch (error) {
    console.error("Error:", error);
  }
}

async function main() {
  await testMcpServerDirect();
}

main();
