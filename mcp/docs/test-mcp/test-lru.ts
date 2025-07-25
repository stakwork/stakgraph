import { experimental_createMCPClient } from "ai";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { randomUUID } from "crypto";

async function testLRUEviction() {
  console.log("🧪 Testing LRU Eviction: Creating 27 sessions to trigger eviction...\n");
  
  const sessions: string[] = [];
  
  for (let i = 1; i <= 27; i++) {
    const sessionId = randomUUID();
    sessions.push(sessionId);
    
    console.log(`📱 Creating session ${i}/27: ${sessionId.substring(0, 8)}...`);
    
    try {
      const client = await experimental_createMCPClient({
        transport: new StreamableHTTPClientTransport(
          new URL("http://localhost:3000/mcp"),
          {
            requestInit: {
              headers: {
                authorization: `Bearer asdfasdf`,
                "x-session-id": sessionId,
              },
            },
          }
        ),
      });

      // Initialize stagehand for this session by calling a tool
      const tools = await client.tools();
      await tools.stagehand_navigate.execute(
        { url: "data:text/html,<html><body><h1>Session " + i + "</h1></body></html>" },
        { toolCallId: `test-${i}`, messages: [] }
      );
      
      console.log(`   ✅ Session ${i} stagehand initialized`);
      
      // Small delay to ensure different timestamps
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      console.error(`   ❌ Failed to create session ${i}:`, error);
    }
  }
  
  console.log("\n🔍 LRU Test complete!");
  console.log("📊 Check server logs for eviction messages starting around session 26");
  console.log("🎯 Expected: Sessions 1-2 should be evicted when sessions 26-27 are created");
}

testLRUEviction().catch(console.error);