import { createMCPClient } from "@ai-sdk/mcp";
import { getMcpTools, McpServer } from "../src/repo/mcpServers.js";

const MCP_URL = process.env.MCP_URL || "https://mcp.stakwork.com/mcp";
const MCP_TOKEN = process.env.MCP_TOKEN || "";

async function testMcpServerDirect() {
  console.log("=== MCP Server Direct Test ===");
  console.log(`URL: ${MCP_URL}`);
  console.log(`Token: ${MCP_TOKEN ? "[SET]" : "[NOT SET]"}`);
  console.log("");

  try {
    const headers: Record<string, string> = {};
    if (MCP_TOKEN) {
      headers.Authorization = `Bearer ${MCP_TOKEN}`;
    }

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

    // Test GetSkills if it exists
    const getSkillsTool = toolNames.find((name) =>
      name.toLowerCase().includes("getskills")
    );
    if (getSkillsTool) {
      console.log(`Testing tool: ${getSkillsTool}`);
      const tool = tools[getSkillsTool];

      try {
        // Call the tool's execute function directly
        console.log("Calling tool.execute...");
        const result = await (tool as any).execute({}, {});
        console.log("Raw result:");
        console.log(JSON.stringify(result, null, 2));
      } catch (execError) {
        console.error("Execute error:", execError);
      }
    }

    await mcpClient.close();
    console.log("\nMCP client closed.");
  } catch (error) {
    console.error("Error:", error);
  }
}

async function testViaMcpServers() {
  console.log("\n=== Test via getMcpTools (like agent does) ===\n");

  const mcpServers: McpServer[] = [
    {
      name: "stak",
      url: MCP_URL,
      token: MCP_TOKEN || undefined,
      toolFilter: ["GetSkills"],
    },
  ];

  try {
    console.log("Calling getMcpTools...");
    const tools = await getMcpTools(mcpServers);
    console.log(`Got ${Object.keys(tools).length} tools:`, Object.keys(tools));

    const skillsTool = tools["stak_GetSkills"];
    if (skillsTool) {
      console.log("\nTesting stak_GetSkills...");
      console.log("Tool type:", typeof skillsTool);
      console.log("Tool keys:", Object.keys(skillsTool));

      try {
        console.log("Calling execute...");
        const result = await (skillsTool as any).execute({}, {});
        console.log("Result type:", typeof result);
        console.log("Result is null:", result === null);
        console.log("Result is undefined:", result === undefined);
        console.log("Raw result:", JSON.stringify(result, null, 2));
      } catch (execError) {
        console.error("Execute error:", execError);
      }
    } else {
      console.log("stak_GetSkills not found in tools");
    }
  } catch (error) {
    console.error("getMcpTools error:", error);
  }
}

async function testWithAgent() {
  console.log("\n=== Test with ToolLoopAgent (full simulation) ===\n");

  const { ToolLoopAgent } = await import("ai");
  const { getModelDetails } = await import("../src/aieo/src/index.js");

  const mcpServers: McpServer[] = [
    {
      name: "stak",
      url: MCP_URL,
      token: MCP_TOKEN || undefined,
      toolFilter: ["GetSkills"],
    },
  ];

  try {
    const tools = await getMcpTools(mcpServers);
    console.log("Tools loaded:", Object.keys(tools));

    const { model } = getModelDetails("claude-sonnet");

    const agent = new ToolLoopAgent({
      model,
      instructions: "You are a helpful assistant. Use the available tools to answer questions. When done, say [END_OF_ANSWER]",
      tools,
      stopSequences: ["[END_OF_ANSWER]"],
      onStepFinish: (sf) => {
        console.log("\n--- Step finished ---");
        console.log("Content:", JSON.stringify(sf.content, null, 2));
      },
    });

    console.log("\nGenerating with agent...");
    const result = await agent.generate({
      prompt: "Call GetSkills to list available skills, then summarize what you found. [END_OF_ANSWER]",
    });

    console.log("\n--- Agent result ---");
    console.log("Steps count:", result.steps.length);
    for (const step of result.steps) {
      console.log("\nStep toolCalls:", step.toolCalls?.length || 0);
      console.log("Step toolResults:", step.toolResults?.length || 0);
      if (step.toolResults) {
        for (const tr of step.toolResults) {
          console.log("  Tool result:", JSON.stringify(tr, null, 2).slice(0, 500));
        }
      }
    }
  } catch (error) {
    console.error("Agent test error:", error);
  }
}

async function main() {
  await testMcpServerDirect();
  await testViaMcpServers();
  await testWithAgent();
}

main();
