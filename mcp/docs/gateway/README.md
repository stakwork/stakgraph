# Bifrost gateway (local)

Companion to [`mcp/docs/plans/llm-gateway-and-runaway-enforcement.md`](../plans/llm-gateway-and-runaway-enforcement.md).
Runs Bifrost locally on `http://localhost:8181` so MCP can be tested with
`LLM_GATEWAY_URL=http://localhost:8181`.

## Layout

- `docker-compose.yml` — Bifrost service (image pinned to `maximhq/bifrost:v2.5.1`),
  host port `8181 -> container 8080`.
- `data/config.json` — seed config: Anthropic + OpenAI + OpenRouter + Gemini
  + xAI providers, API keys read from `env.*`, `config_store` enabled (SQLite)
  so the Web UI works. (Bifrost calls Google's public Gemini API `gemini`; the
  MCP client side calls the same thing `google`. xAI has no dedicated Bifrost
  route; the MCP client sends Grok through `/openai/v1` with an `xai/`-prefixed
  model id.)
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

- `enforce_auth_on_inference` is `true` in `config.json` (this note used to say
  "off"; the file was already `true`), so requests without a VK are refused
  with 401 (verified on v2.5.0). Set it to `false` to let unauthenticated
  callers through.
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
- The image is pinned to `v2.5.1`. From Bifrost 2.2.6 on, `/api/*` (the VK
  scripts here) needs a setup token while no dashboard admin exists:
  `BIFROST_SETUP_TOKEN` in `docker-compose.yml` (default `dev-setup-token`) and
  `lib.ts` sends it as `X-Bifrost-Setup-Token`. Inference without a VK is
  refused by `enforce_auth_on_inference: true` in `config.json`.

## Anthropic workload identity federation (optional)

Not used by default (the quickstart authenticates with `ANTHROPIC_API_KEY`).
Upstream Bifrost (2.5.0+) can instead exchange a workload identity token for
a short-lived Anthropic access token. In `data/config.json`, replace the
Anthropic key (a key that has a `value` ignores `workload_identity`, so drop
`value`):

```json
{
  "name": "anthropic-wif",
  "models": ["*"],
  "weight": 1.0,
  "anthropic_key_config": {
    "workload_identity": {
      "federation_rule_id": "env.ANTHROPIC_FEDERATION_RULE_ID",
      "organization_id": "env.ANTHROPIC_ORGANIZATION_ID",
      "service_account_id": "env.ANTHROPIC_SERVICE_ACCOUNT_ID",
      "workspace_id": "env.ANTHROPIC_WORKSPACE_ID",
      "identity_source": { "type": "file", "path": "/var/run/secrets/anthropic.com/token" }
    }
  }
}
```

(`workspace_id` is optional.) Pass those `ANTHROPIC_*` variables to the
container in `docker-compose.yml`.

Bifrost does **not** fetch the identity token itself: it only reads the JWT
from `identity_source` (`file` is re-read on every exchange; `inline` is
fixed) and posts it to `https://api.anthropic.com/v1/oauth/token`. Something
else must put a JWT in that file and keep it fresh, mounted into the
container under `/var/run/secrets` or `/run/secrets` (other directories need
`BIFROST_OIDC_ALLOWED_CREDENTIAL_DIRS`), e.g. a Kubernetes projected service
account token or a sidecar. If the token carries a `jti`, Anthropic accepts it
once, so rewrite the file before Bifrost's next exchange. Note that
`config.json` only reconciles into an existing `config.db` by key name: keep
the key `name` to replace the old key, or reset the stores as above.

For the AWS EC2 instance profile case, `gateway/` in this repo ships a token
helper and wrapper that do this for you (see `gateway/README.md`,
"Anthropic workload identity federation").
