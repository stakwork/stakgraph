// Stdio MCP transport for the verification tools. A generic agent (e.g. goose,
// launched by staklink in the pod) spawns this as an MCP extension and drives
// the shared verify tools + submit_verdict over stdio.
//
// stdout is the MCP JSON-RPC channel: anything the app writes there (browser
// launch, stagehand, deps) corrupts the protocol for a strict client, so guard
// stdout to pass only JSON-RPC frames and route everything else to stderr.
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
