import { createMCPClient } from '@ai-sdk/mcp';

const server = 'https://mcp.stakwork.com/mcp';
const token = process.env.API_TOKEN || 'asdfasdf';

async function run() {

    const mcpClient = await createMCPClient({
    transport: {
        type: 'http',
        url: server,
        headers: { Authorization: `Bearer ${token}` },
    },
    });

    const tools = await mcpClient.tools();

    console.log(tools);
    console.log((tools.GetWorkflows.inputSchema as any).jsonSchema);
}

run();