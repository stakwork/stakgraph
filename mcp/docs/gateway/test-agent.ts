// End-to-end smoke test:
//   1. Mint a fresh virtual key in Bifrost.
//   2. POST to MCP's /repo/agent with that VK as `apiKey` in the request body.
//      MCP forwards it as the Anthropic `x-api-key` header on its way through
//      Bifrost (see mcp/src/aieo/src/provider.ts).
//   3. Print the response.
//   4. Tell the user how to verify in mcp.log + bifrost logs.
//
// Assumes:
//   - Bifrost is up:   docker compose up -d   (in this directory)
//   - MCP is running:  LLM_GATEWAY_URL=http://localhost:8181 yarn dev   (in ../../)
//
// Usage:
//   tsx test-agent.ts
//   tsx test-agent.ts "your prompt here"

import { createVirtualKey, MCP_URL, ping } from "./lib.js";

const REPO_URL = process.env.REPO_URL ?? "https://github.com/stakwork/hive";

async function main() {
  await ping();

  const prompt = process.argv[2] ?? "hi, how are you?";
  console.error(`[gateway-test] minting VK...`);
  const vk = await createVirtualKey({
    name: `agent-test-${Date.now()}`,
    description: "Per-run VK from test-agent.ts",
    budget: { max_limit: 2.0, reset_duration: "1d" },
    rate_limit: {
      request_max_limit: 30,
      request_reset_duration: "1m",
    },
  });
  console.error(`[gateway-test] VK ${vk.id} value=${vk.value}`);

  const body = {
    repo_url: REPO_URL,
    prompt,
    apiKey: vk.value,
    // model defaults to whatever LLM_PROVIDER is set in the MCP env (anthropic -> sonnet).
  };

  console.error(`[gateway-test] POST ${MCP_URL}/repo/agent with prompt: "${prompt}"`);
  const t0 = Date.now();
  const res = await fetch(`${MCP_URL}/repo/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const text = await res.text();
  let parsed: any;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = text;
  }
  console.error(`[gateway-test] -> ${res.status} in ${elapsed}s`);

  if (typeof parsed === "object" && parsed) {
    // Trim the noisy fields so the answer is readable.
    const { answer, status, request_id, sessionId, session_id, error } = parsed;
    console.log(
      JSON.stringify(
        { status, request_id, sessionId: sessionId ?? session_id, error, answer },
        null,
        2,
      ),
    );
  } else {
    console.log(parsed);
  }

  console.error("");
  console.error("[gateway-test] verify routing:");
  console.error("  tail -n 20 ../../mcp.log | grep LLM_GATEWAY");
  console.error("  docker logs --tail 50 bifrost | grep -i anthropic");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
