import { createMCPClient } from '@ai-sdk/mcp';

const server = 'http://localhost:3355/mcp/mcp';
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
}

run();