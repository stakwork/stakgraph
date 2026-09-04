import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { graphServer } from "../server.js";

async function main() {
  const transport = new StdioServerTransport();
  await graphServer.connect(transport);
}

main().catch((e) => {
  console.error("[verify-stdio] fatal:", e?.message ?? e);
  process.exit(1);
});
