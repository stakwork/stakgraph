# Bifrost gateway (local)

Companion to [`mcp/docs/plans/llm-gateway-and-runaway-enforcement.md`](../plans/llm-gateway-and-runaway-enforcement.md).
Runs Bifrost locally on `http://localhost:8181` so MCP can be tested with
`LLM_GATEWAY_URL=http://localhost:8181`.

## Layout

- `docker-compose.yml` — Bifrost service, host port `8181 -> container 8080`.
- `data/config.json` — seed config: Anthropic + OpenAI + OpenRouter + Gemini
  providers, API keys read from `env.*`, `config_store` enabled (SQLite) so the
  Web UI works. (Bifrost calls Google's public Gemini API `gemini`; the MCP
  client side calls the same thing `google`.)
- `data/config.db`, `data/logs.db` — created on first boot; gitignored.
- `.env` — symlinked to `../../.env` (the MCP `.env`) so docker-compose picks up
  `ANTHROPIC_API_KEY` etc. without copy-paste.
- `lib.ts` — small Bifrost governance API client (TypeScript / `fetch`).
- `create-vk.ts` — mints a virtual key, prints its `sk-bf-...` value.
- `list-vks.ts` — lists VKs; pass `--delete` to wipe all of them.
- `test-agent.ts` — mints a fresh VK and hits `/repo/agent` with it.

## Quick start

```bash
# 1. Bring up Bifrost
cd mcp/docs/gateway
docker compose up -d
docker logs -f bifrost   # (optional) watch it boot

# 2. Open the UI to confirm providers loaded
open http://localhost:8181

# 3. Make sure MCP is running with LLM_GATEWAY_URL set (from mcp/)
#    LLM_GATEWAY_URL=http://localhost:8181 yarn dev > mcp.log

# 4. End-to-end smoke test: creates a VK and calls /repo/agent
#    (use the mcp project's tsx so transitive deps resolve)
cd mcp
npx tsx docs/gateway/test-agent.ts "hi, how are you?"

# 5. Verify
tail -n 50 mcp.log | grep -E 'LLM_GATEWAY|apiKeyPrefix'
#   ↑ should show
#       [LLM_GATEWAY] routing anthropic via http://localhost:8181/anthropic/v1
#       apiKeyPrefix: 'sk-bf-...'

docker logs --tail 50 bifrost | grep "/anthropic/v1/messages"
#   ↑ should show a 200 from user_agent="ai/... ai-sdk/provider-utils/..."

# 6. (optional) list / clean up VKs
npx tsx docs/gateway/list-vks.ts
npx tsx docs/gateway/list-vks.ts --delete
```

## Notes

- `enforce_auth_on_inference` is **off** in `config.json`, so requests without a
  VK still pass through (per rollout step 2 in the plan). Switch it on once all
  callers send a VK.
- The provider keys live in the host `.env` (read by docker-compose) and are
  passed into the container as env vars; `config.json` references them via
  `env.ANTHROPIC_API_KEY` etc. — they are NOT baked into `config.db`.
- To reset everything (drop the SQLite stores so `config.json` is re-read on
  next boot):
  ```bash
  docker compose down
  rm -f data/config.db* data/logs.db*
  docker compose up -d
  ```
