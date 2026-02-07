import { createMCPClient } from '@ai-sdk/mcp';

const server = process.env.MCP_URL || 'http://localhost:3355/mcp';
const token = process.env.API_TOKEN || 'asdfasdf';

async function run() {
    const mcpClient = await createMCPClient({
        transport: {
            type: 'http',
            url: server,
            headers: { 
                Authorization: `Bearer ${token}`,
                // 'x-session-id': randomUUID(),  // Force new session
            },
        }
    });

    const tools = await mcpClient.tools();
    console.log('Available tools:', Object.keys(tools));

    // Execute search_logs tool
    const searchLogsTool = tools['search_logs'];
    if (searchLogsTool && searchLogsTool.execute) {
        console.log('Executing search_logs...');
        const result = await searchLogsTool.execute({
            query: 'path:pod-repair',
            max_hits: 10
        }, {
            toolCallId: '1',
            messages: [],
        });
        console.log('search_logs result:', JSON.stringify(result, null, 2));
    } else {
        console.log('search_logs tool not found');
    }

    await mcpClient.close();
}

run();