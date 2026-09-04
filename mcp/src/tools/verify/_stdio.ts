// stdout is the MCP JSON-RPC channel. Anything the app (browser launch,
// stagehand, dependencies) writes to stdout corrupts the protocol and a strict
// client (goose/rmcp) drops the transport. Guard stdout so only JSON-RPC frames
// (lines starting with '{') pass through; route everything else to stderr.
const realStdoutWrite = process.stdout.write.bind(process.stdout);
process.stdout.write = ((chunk: any, ...rest: any[]): boolean => {
  const s = typeof chunk === "string" ? chunk : Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk);
  if (s.startsWith("{")) return realStdoutWrite(chunk, ...rest);
  return (process.stderr.write as any)(chunk, ...rest);
}) as typeof process.stdout.write;

console.log = (...a: unknown[]) => console.error(...a);
console.info = (...a: unknown[]) => console.error(...a);
console.warn = (...a: unknown[]) => console.error(...a);

const { StdioServerTransport } = await import("@modelcontextprotocol/sdk/server/stdio.js");
const { graphServer } = await import("../server.js");

const transport = new StdioServerTransport();
await graphServer.connect(transport);
