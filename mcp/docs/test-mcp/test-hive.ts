

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

    // list_features
    const list_features = toolNames.find((name) =>
      name.toLowerCase().includes("list_features")
    );
    if (list_features) {
      console.log(`Testing tool: ${list_features}`);
      const tool = tools[list_features];
      const result = await tool.execute({}, {toolCallId: "1", messages: []});
      console.log("list_features result:", JSON.stringify(result, null, 2));
    }

    const read_feature = toolNames.find((name) =>
      name.toLowerCase().includes("read_feature")
    );
    if (read_feature) {
      console.log(`Testing tool: ${read_feature}`);
      const tool = tools[read_feature];
      const result = await tool.execute({featureId: "cmmcdyh0v0001jp9pll4s5kcz"}, {toolCallId: "2", messages: []});
      console.log("read_feature result:", JSON.stringify(result, null, 2));
    }

    const create_feature = toolNames.find((name) =>
      name.toLowerCase().includes("create_feature")
    );
    if (create_feature) {
      console.log(`Testing tool: ${create_feature}`);
      const tool = tools[create_feature];
      const result = await tool.execute({title: "asdfasdf", brief: "Test Feature", requirements: "Test Description"}, {toolCallId: "3", messages: []});
      console.log("create_feature result:", JSON.stringify(result, null, 2));
    }

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
