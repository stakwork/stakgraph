import { Express } from 'express';
import { createMcpHandler } from 'mcp-handler';
import { registerTools } from './tools.js';
import { createExpressAdapter } from './utils.js';

export function mcp_routes(app: Express) {
  const handler = createMcpHandler(registerTools, {}, {
    basePath: "/mcp",
    verboseLogs: true
  });

  app.all('/mcp*', createExpressAdapter(handler));
}
