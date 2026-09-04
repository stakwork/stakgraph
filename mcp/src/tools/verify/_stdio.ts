// stdout is the MCP protocol channel — route all app logging to stderr so it
// cannot corrupt the JSON-RPC stream. Must run before any other module loads,
// so the imports below are dynamic.
console.log = (...a: unknown[]) => console.error(...a);
console.info = (...a: unknown[]) => console.error(...a);
console.warn = (...a: unknown[]) => console.error(...a);

const { StdioServerTransport } = await import("@modelcontextprotocol/sdk/server/stdio.js");
const { graphServer } = await import("../server.js");

const transport = new StdioServerTransport();
await graphServer.connect(transport);
