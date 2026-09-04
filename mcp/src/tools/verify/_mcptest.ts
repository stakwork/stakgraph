import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function main() {
  const transport = new StdioClientTransport({
    command: "npx",
    args: ["tsx", "/Users/fayekelmith/Kelmith/Projects/stakwork/stakgraph/mcp/src/tools/verify/_stdio.ts"],
    env: {
      ...process.env,
      USE_STAGEHAND: "true",
      STAGEHAND_MODEL: "anthropic/claude-opus-5",
    } as Record<string, string>,
    stderr: "inherit",
  });
  const client = new Client({ name: "mcptest", version: "1.0.0" }, { capabilities: {} });
  await client.connect(transport);
  const tools = await client.listTools();
  console.error("tools:", tools.tools.map((t) => t.name).join(","));

  const call = async (name: string, args: any) => {
    const t0 = Date.now();
    try {
      const r = await client.callTool({ name, arguments: args });
      console.error(`${name} OK (${((Date.now() - t0) / 1000).toFixed(1)}s):`, JSON.stringify(r).slice(0, 160));
    } catch (err: any) {
      console.error(`${name} ERROR (${((Date.now() - t0) / 1000).toFixed(1)}s):`, err?.message ?? String(err));
    }
  };
  await call("stagehand_navigate", { url: "http://localhost:3000/audit-lab/network-broken" });
  await call("stagehand_act", { action: "Click the Save button" });
  await call("stagehand_network_activity", {});
  await call("verify_http_request", { url: "http://localhost:3000/api/audit-lab/net-fail", method: "POST" });
  await client.close().catch(() => {});
  process.exit(0);
}

main().catch((e) => {
  console.error("mcptest fatal:", e?.message ?? e);
  process.exit(1);
});
