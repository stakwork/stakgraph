#!/usr/bin/env bash
#
# smoke-test.sh — observability smoke test for the stakgraph LLM gateway.
#
# Goal: prove that with only Bifrost's built-in logging plugin and the
# `x-bf-dim-*` header convention, we can answer:
#
#   "How much money was spent per agent name, broken down by time
#    period, across the gateway?"
#
# This script generates ~15 chat-completion requests with a deliberate
# traffic matrix designed to stress every interesting dimension:
#
#   - Multiple agent names (browser / coder / chat / reviewer)
#   - Multiple users (alice / bob / carol) and workspaces (w1 / w2)
#   - Spread across ≥2 minute buckets via sleeps
#   - Multiple providers (anthropic / openai / openrouter) and models
#   - One streaming request
#   - One error request (bad model name)
#   - One request with NO dim headers (graceful absence check)
#
# Hard ceiling: < $0.02 of real spend (cheap models, max_tokens=20).
#
# Usage:
#   bash gateway/scripts/smoke-test.sh
#
# After the script completes, inspect rows with:
#   docker run --rm -v stakgraph-gateway-data:/data alpine:3.23 \
#     sh -c "apk add -q sqlite >/dev/null && sqlite3 /data/logs.db \
#       \"SELECT json_extract(metadata,'\$.agent-name') AS agent,
#                ROUND(SUM(cost),6) AS spend
#         FROM logs
#         WHERE created_at >= datetime('now','-15 minutes')
#           AND json_extract(metadata,'\$.agent-name') IS NOT NULL
#         GROUP BY agent ORDER BY spend DESC;\""
#
# The DB lives in the `stakgraph-gateway-data` docker volume (named volume
# instead of bind mount — see plans/bifrost-log-drop-debug.md for why).

set -u  # don't use -e: we expect at least one request to fail (the error row)
        # and want to keep going

GW="${GW:-http://localhost:8181}"
RUN_PREFIX="${RUN_PREFIX:-r_smoke_$(date +%s)}"

# --- helpers ---------------------------------------------------------------

new_uuid() {
  # Portable enough: prefer /proc, fall back to python.
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen | tr 'A-Z' 'a-z'
  else
    python3 -c 'import uuid; print(uuid.uuid4())'
  fi
}

# call_llm <agent> <user> <workspace> <session> <provider/model> <prompt>
#
# Sends one non-streaming chat completion with the full dim header set,
# capped at 20 output tokens. Prints a one-line summary per call.
call_llm() {
  local agent="$1" user="$2" workspace="$3" session="$4" model="$5" prompt="$6"
  local run_id; run_id="$(new_uuid)"
  local started; started=$(date -u +%H:%M:%S)
  local status
  status=$(curl -sS -o /tmp/smoke_resp.json -w '%{http_code}' "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "x-bf-dim-run-id: $run_id" \
    -H "x-bf-dim-session-id: $session" \
    -H "x-bf-dim-agent-name: $agent" \
    -H "x-bf-dim-realm-id: $workspace" \
    -H "x-bf-dim-user-id: $user" \
    -H "x-bf-dim-deployment: smoke-test" \
    -d "{\"model\":\"$model\",\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}")
  printf '%s  %-9s  %-9s  %-3s  %-39s  run_id=%s  http=%s\n' \
    "$started" "$agent" "$user" "$workspace" "$model" "$run_id" "$status"
}

# call_llm_stream <agent> <user> <workspace> <session> <provider/model> <prompt>
#
# Sends one streaming request. Bifrost streams SSE; we only need the
# response to terminate cleanly so the final-chunk accumulator writes
# the cost into the logs row.
call_llm_stream() {
  local agent="$1" user="$2" workspace="$3" session="$4" model="$5" prompt="$6"
  local run_id; run_id="$(new_uuid)"
  local started; started=$(date -u +%H:%M:%S)
  # --no-buffer + drain to /dev/null. We don't care about the content,
  # only that the stream completes and bifrost emits a final chunk.
  curl -sS --no-buffer -o /tmp/smoke_stream.txt -w 'http=%{http_code}\n' \
    "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    -H "x-bf-dim-run-id: $run_id" \
    -H "x-bf-dim-session-id: $session" \
    -H "x-bf-dim-agent-name: $agent" \
    -H "x-bf-dim-realm-id: $workspace" \
    -H "x-bf-dim-user-id: $user" \
    -H "x-bf-dim-deployment: smoke-test" \
    -d "{\"model\":\"$model\",\"max_tokens\":20,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}" \
  | sed "s/^/  $started  $agent  $user  $workspace  STREAM run_id=$run_id /"
}

# call_llm_no_dims: same shape as call_llm but sends ZERO x-bf-dim-* headers,
# to prove the absence-of-dims case still produces a graceful logs row.
call_llm_no_dims() {
  local model="$1" prompt="$2"
  local started; started=$(date -u +%H:%M:%S)
  local status
  status=$(curl -sS -o /tmp/smoke_resp.json -w '%{http_code}' "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}")
  printf '%s  %-9s  %-9s  %-3s  %-39s  run_id=%s  http=%s\n' \
    "$started" "(no-dims)" "-" "-" "$model" "-" "$status"
}

# --- preflight -------------------------------------------------------------

if ! curl -sf -o /dev/null "$GW/api/health"; then
  echo "gateway not reachable at $GW/api/health — start it with: cd gateway && make docker-up" >&2
  exit 1
fi

echo "Smoke test starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) against $GW"
echo
printf '%-8s  %-9s  %-9s  %-3s  %-39s  %s\n' \
  'time(UTC)' 'agent' 'user' 'ws' 'provider/model' 'identifiers'
printf -- '-%.0s' {1..110}; echo

# Stable session-ids: one per (user, workspace) pair, so the "one session,
# many runs" case is testable.
SESS_ALICE_W1="s_alice_w1_$(date +%s)"
SESS_BOB_W1="s_bob_w1_$(date +%s)"
SESS_BOB_W2="s_bob_w2_$(date +%s)"
SESS_CAROL_W1="s_carol_w1_$(date +%s)"
SESS_CAROL_W2="s_carol_w2_$(date +%s)"

# Cheap models, all available in the gateway today (verified via /v1/models).
M_HAIKU="anthropic/claude-haiku-4-5-20251001"
M_MINI="openai/gpt-4o-mini"
M_NANO="openai/gpt-4.1-nano"
M_KIMI="openrouter/moonshotai/kimi-k2-0905"

# --- batch 1: minute N ----------------------------------------------------

echo "# batch 1 — minute N"
call_llm browser   u_alice  w1  "$SESS_ALICE_W1" "$M_HAIKU" 'reply in 3 words'
call_llm browser   u_alice  w1  "$SESS_ALICE_W1" "$M_MINI"  'reply in 3 words'
call_llm browser   u_bob    w1  "$SESS_BOB_W1"   "$M_HAIKU" 'reply in 3 words'
call_llm coder     u_alice  w1  "$SESS_ALICE_W1" "$M_NANO"  'one-line haiku'
call_llm coder     u_alice  w1  "$SESS_ALICE_W1" "$M_HAIKU" 'one-line haiku'
call_llm chat      u_bob    w2  "$SESS_BOB_W2"   "$M_MINI"  'two-word greeting'

# sleep enough to land the next batch in a different minute bucket
echo
echo "# sleeping 25s to cross a minute boundary..."
sleep 25

# --- batch 2: minute N+1 --------------------------------------------------

echo
echo "# batch 2 — minute N+1"
call_llm chat      u_carol  w2  "$SESS_CAROL_W2" "$M_HAIKU" 'two-word greeting'
call_llm reviewer  u_carol  w1  "$SESS_CAROL_W1" "$M_NANO"  'reply in 3 words'
call_llm reviewer  u_carol  w1  "$SESS_CAROL_W1" "$M_KIMI"  'reply in 3 words'

echo
echo "# streaming request (verify final-chunk cost lands)"
call_llm_stream chat u_alice w1 "$SESS_ALICE_W1" "$M_HAIKU" 'count: 1 2 3'

echo
echo "# error request (bad model name -> status != success)"
call_llm browser   u_alice  w1  "$SESS_ALICE_W1" \
  'anthropic/this-model-does-not-exist' 'reply in 3 words'

echo
echo "# request with NO dim headers (absence-of-dims must be graceful)"
call_llm_no_dims "$M_HAIKU" 'reply in 3 words'

echo
echo "Done. Wait ~5s for streaming flush, then query the logs volume."
echo
echo "Headline query (DB lives in the stakgraph-gateway-data docker volume):"
echo "  docker run --rm -v stakgraph-gateway-data:/data alpine:3.23 \\"
echo "    sh -c \"apk add -q sqlite >/dev/null && sqlite3 /data/logs.db \\"
echo "      \\\"SELECT json_extract(metadata,'\\\$.agent-name') AS agent,"
echo "              ROUND(SUM(cost),6) AS spend"
echo "       FROM logs WHERE created_at >= datetime('now','-15 minutes')"
echo "         AND json_extract(metadata,'\\\$.agent-name') IS NOT NULL"
echo "       GROUP BY agent ORDER BY spend DESC;\\\"\""
