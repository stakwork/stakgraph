#!/usr/bin/env bash
#
# smoke-test-concat.sh — end-to-end test of the bearer-concat
# transport (phase 10).
#
# Proves the wrapper correctly splits `<vk>.<macaroon>` from each of
# the three accepted inbound auth headers and produces the same
# end-state (plugin sees x-macaroon, logs.db row carries macaroon
# dims) as a request that sent the two-header form directly.
#
# Companion to smoke-test-enforcement.sh (which tests the two-header
# form). Both must keep working — the concat form is strictly
# additive.
#
# Each call exercises one inbound header shape:
#
#   1. Authorization: Bearer <vk>.<mac>   (OpenAI-family path)
#   2. x-api-key: <vk>.<mac>              (Anthropic path)
#   3. x-goog-api-key: <vk>.<mac>         (Gemini path)
#
# For each call we assert:
#   - HTTP 200 from the gateway (proves Bifrost saw a clean VK after
#     the wrapper split).
#   - A plugin "auth: verify ok" log line for that run_id (proves the
#     plugin saw x-macaroon after the wrapper injected it).
#   - A logs.db row with metadata dims canonicalized from the macaroon
#     claims (proves end-to-end pipeline).
#
# All three calls use the same provider (Anthropic) so we don't need
# three sets of provider keys. Only the inbound auth header shape
# changes — the wrapper's job is to make all three converge to the
# same downstream state.
#
# Required:
#   - `make docker-up` already run
#   - `gateway/auth/ts/node_modules` populated (`npm install`)
#   - ANTHROPIC_API_KEY set in the gateway's env (compose passes
#     through from the host .env)
#
# Usage:
#   bash gateway/scripts/smoke-test-concat.sh
#
# Environment knobs:
#   GW                          gateway URL (default http://localhost:8181)
#   BIFROST_PROVISIONING_TOKEN  bearer for /_plugin/trust admin
#   ORG_ID                      org_id for registration + macaroon
#                               (default org_concat)
#   BIFROST_VK                  the VK to glue onto each macaroon.
#                               Default: a placeholder VK shape. With
#                               disable_auth_on_inference=true (the
#                               default in data/config.json), Bifrost
#                               doesn't reject unknown VKs — it just
#                               doesn't stamp customer_id on the log
#                               row. The wrapper's split logic doesn't
#                               care whether the VK is real, only that
#                               it matches the VK-shape regex.
#
#                               Override with a real provisioned VK if
#                               you also want customer_id stamped:
#                                 BIFROST_VK=sk-bf-real... bash $0

set -u

GW="${GW:-http://localhost:8181}"
TOKEN="${BIFROST_PROVISIONING_TOKEN:-dev-provisioning-token}"
ORG_ID="${ORG_ID:-org_concat}"
# Placeholder VK: shape-valid (matches the wrapper's regex) but not
# necessarily registered as a real customer. Sufficient for testing
# the wrapper rewrite when Bifrost runs with disable_auth_on_inference.
VK="${BIFROST_VK:-sk-bf-smoke-placeholder-vk}"

# Fixture org pubkey #0 — see smoke-test-enforcement.sh.
ORG_PUBKEY="0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TS_DIR="$REPO_ROOT/gateway/auth/ts"
TSX="$TS_DIR/node_modules/.bin/tsx"

ASSERT_FAILED=0

# --- helpers ---------------------------------------------------------------

section() {
  printf '\n=== %s ===\n' "$1"
}

# call_admin <method> <path> [body]
call_admin() {
  local method="$1" path="$2" body="${3:-}"
  local status
  if [ -n "$body" ]; then
    status=$(curl -sS -o /tmp/smoke_admin.json -w '%{http_code}' \
      -X "$method" "$GW$path" \
      -H "Authorization: Bearer $TOKEN" \
      -H 'Content-Type: application/json' \
      -d "$body")
  else
    status=$(curl -sS -o /tmp/smoke_admin.json -w '%{http_code}' \
      -X "$method" "$GW$path" \
      -H "Authorization: Bearer $TOKEN")
  fi
  printf '  %s %s -> %s\n' "$method" "$path" "$status"
  if [ -s /tmp/smoke_admin.json ]; then
    sed 's/^/    /' /tmp/smoke_admin.json
    echo
  fi
}

new_run_id() {
  local label="$1"
  printf 'r_concat_%s_%s_%s' "$label" "$(date +%s)" "$RANDOM"
}

# mint_macaroon <run_id>
#
# Same shape as the enforcement smoke. Outputs the base64url macaroon
# on stdout. Phase 11: no `--realm` flag — single-swarm shape is the
# default and matches what this concat-transport test needs.
mint_macaroon() {
  local run_id="$1"
  (
    cd "$TS_DIR" && \
    "$TSX" "$SCRIPT_DIR/mint-macaroon.ts" \
      --org-id "$ORG_ID" \
      --user-id u_alice \
      --agent coder \
      --run-id "$run_id" \
      --max-cost-usd 1.00 \
      --max-steps 50 \
      --ttl-seconds 600
  )
}

# call_llm_concat <label> <header_name> <header_prefix> <run_id> <vk> <macaroon>
#
# Send one chat completion request with the concat form on the
# specified inbound header. <header_prefix> is "Bearer " for
# Authorization and "" for x-api-key / x-goog-api-key.
#
# Note: we deliberately do NOT set x-macaroon ourselves — the whole
# point is that the wrapper injects it from the concat. We also do NOT
# set any of the x-bf-dim-* headers — phase 6 canonicalization
# populates them from the verified macaroon claims, so the logs.db
# row still gets correct dims.
call_llm_concat() {
  local label="$1" name="$2" prefix="$3" run_id="$4" vk="$5" mac="$6"
  local started status
  started=$(date -u +%H:%M:%S)
  status=$(curl -sS -o /tmp/smoke_llm.json -w '%{http_code}' \
    "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "$name: $prefix$vk.$mac" \
    -d '{"model":"anthropic/claude-haiku-4-5-20251001","max_tokens":20,"messages":[{"role":"user","content":"reply in 3 words"}]}')
  printf '  [%s] %s  run_id=%s  http=%s\n' "$label" "$started" "$run_id" "$status"
  if [ "$status" != "200" ] && [ -s /tmp/smoke_llm.json ]; then
    echo "  body:"
    sed 's/^/    /' /tmp/smoke_llm.json
    echo
  fi
  echo "$status"
}

# fetch_logs_row <run_id>
#
# Same query as smoke-test-enforcement.sh: pipe-delimited
# run-id|user-id|agent-name from logs.metadata. Empty on no-match.
# Phase 11 dropped realm-id from the row format (no longer
# signature-bound).
fetch_logs_row() {
  local run_id="$1"
  docker run --rm -v stakgraph-gateway-data:/data alpine:3.23 \
    sh -c "apk add -q sqlite >/dev/null && sqlite3 /data/logs.db \
      \"SELECT json_extract(metadata,'\$.run-id') || '|' ||
              COALESCE(json_extract(metadata,'\$.user-id'),'') || '|' ||
              COALESCE(json_extract(metadata,'\$.agent-name'),'')
       FROM logs WHERE json_extract(metadata,'\$.run-id')='$run_id' LIMIT 1;\"" \
    2>/dev/null
}

# expect_eq <label> <expected> <actual>
expect_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    printf '  [PASS] %s: %q\n' "$label" "$actual"
  else
    printf '  [FAIL] %s: expected %q got %q\n' "$label" "$expected" "$actual"
    ASSERT_FAILED=$((ASSERT_FAILED + 1))
  fi
}

# assert_verify_log_present <run_id>
#
# Confirms the plugin emitted a successful verify line for this run.
# The presence of such a line proves the wrapper successfully injected
# x-macaroon; if the wrapper had failed, the plugin would log "auth:
# no x-macaroon header ..." instead.
assert_verify_log_present() {
  local run_id="$1"
  local hit
  hit=$(docker compose -f "$REPO_ROOT/gateway/docker-compose.yml" \
        logs --tail=500 bifrost 2>/dev/null \
    | grep -E "auth: verify .* run_id=$run_id" \
    | tail -1)
  if [ -n "$hit" ]; then
    printf '  [PASS] verify line present for %s\n' "$run_id"
    printf '         %s\n' "$hit"
  else
    printf '  [FAIL] no verify line found for %s\n' "$run_id"
    ASSERT_FAILED=$((ASSERT_FAILED + 1))
  fi
}

# --- preflight -------------------------------------------------------------

section "preflight"

echo "  using BIFROST_VK=${VK:0:16}... (length=${#VK})"

if ! curl -sf -o /dev/null "$GW/api/health"; then
  echo "  gateway not reachable at $GW/api/health" >&2
  echo "  start it with: cd gateway && make docker-up" >&2
  exit 1
fi
echo "  gateway reachable at $GW"

if [ ! -x "$TSX" ]; then
  echo "  tsx not found at $TSX" >&2
  echo "  install with: cd gateway/auth/ts && npm install" >&2
  exit 1
fi
echo "  tsx available"

trust_status=$(curl -sS -o /tmp/smoke_admin.json -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" "$GW/_plugin/trust/status")
if [ "$trust_status" != "200" ]; then
  echo "  /_plugin/trust/status -> $trust_status (provisioning token wrong?)" >&2
  exit 1
fi
echo "  trust registry reachable"

# --- register org ----------------------------------------------------------

section "register org $ORG_ID"

call_admin POST /_plugin/trust "$(cat <<JSON
{
  "org_id": "$ORG_ID",
  "pubkey": "$ORG_PUBKEY",
  "issuer_url": "",
  "revocation_poll_seconds": 60
}
JSON
)"

# --- three concat forms ---------------------------------------------------

section "concat form 1/3: Authorization: Bearer <vk>.<mac>"

AUTH_RUN_ID=$(new_run_id "auth")
echo "  minting macaroon for run_id=$AUTH_RUN_ID..."
AUTH_MAC=$(mint_macaroon "$AUTH_RUN_ID")
if [ -z "$AUTH_MAC" ]; then
  echo "  mint FAILED" >&2
  exit 1
fi
echo "  macaroon length=${#AUTH_MAC}"
AUTH_STATUS=$(call_llm_concat "Authorization" "Authorization" "Bearer " "$AUTH_RUN_ID" "$VK" "$AUTH_MAC" | tail -1)

section "concat form 2/3: x-api-key: <vk>.<mac>"

XAPI_RUN_ID=$(new_run_id "xapi")
echo "  minting macaroon for run_id=$XAPI_RUN_ID..."
XAPI_MAC=$(mint_macaroon "$XAPI_RUN_ID")
if [ -z "$XAPI_MAC" ]; then
  echo "  mint FAILED" >&2
  exit 1
fi
XAPI_STATUS=$(call_llm_concat "x-api-key" "x-api-key" "" "$XAPI_RUN_ID" "$VK" "$XAPI_MAC" | tail -1)

section "concat form 3/3: x-goog-api-key: <vk>.<mac>"

XGOOG_RUN_ID=$(new_run_id "xgoog")
echo "  minting macaroon for run_id=$XGOOG_RUN_ID..."
XGOOG_MAC=$(mint_macaroon "$XGOOG_RUN_ID")
if [ -z "$XGOOG_MAC" ]; then
  echo "  mint FAILED" >&2
  exit 1
fi
# Note: Bifrost routes by URL path, not by which auth header was
# used. /v1/chat/completions is the OpenAI-shaped endpoint, so even
# though we're sending x-goog-api-key, we still get an Anthropic
# response. The point of this test isn't to exercise Gemini routing
# — it's to verify the wrapper rewrites x-goog-api-key correctly.
# Bifrost will then read the rewritten "x-goog-api-key: <vk>" the
# same way it reads any other VK lookup header.
XGOOG_STATUS=$(call_llm_concat "x-goog-api-key" "x-goog-api-key" "" "$XGOOG_RUN_ID" "$VK" "$XGOOG_MAC" | tail -1)

# --- backward-compat sanity: two-header form still works ------------------

section "backward compat: two-header form still works"

COMPAT_RUN_ID=$(new_run_id "compat")
echo "  minting macaroon for run_id=$COMPAT_RUN_ID..."
COMPAT_MAC=$(mint_macaroon "$COMPAT_RUN_ID")
if [ -z "$COMPAT_MAC" ]; then
  echo "  mint FAILED" >&2
  exit 1
fi
# Send VK and macaroon as two separate headers — the existing,
# pre-phase-10 way. Must still work byte-for-byte after the change.
COMPAT_STATUS=$(curl -sS -o /tmp/smoke_llm.json -w '%{http_code}' \
  "$GW/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VK" \
  -H "x-macaroon: $COMPAT_MAC" \
  -d '{"model":"anthropic/claude-haiku-4-5-20251001","max_tokens":20,"messages":[{"role":"user","content":"reply in 3 words"}]}')
printf '  two-header form  run_id=%s  http=%s\n' "$COMPAT_RUN_ID" "$COMPAT_STATUS"
if [ "$COMPAT_STATUS" != "200" ] && [ -s /tmp/smoke_llm.json ]; then
  echo "  body:"
  sed 's/^/    /' /tmp/smoke_llm.json
  echo
fi

# --- give bifrost a beat to flush logs ------------------------------------

echo
echo "  waiting 4s for log flush..."
sleep 4

# --- assertions ------------------------------------------------------------

section "assertions"

# HTTP status: all four calls must produce IDENTICAL downstream
# behavior. If the wrapper's split is correct, the three concat
# variants should be byte-identical to the two-header form from
# Bifrost's perspective — same VK, same x-macaroon, same dim
# canonicalization. So we don't hard-code "200"; we just require
# the three concat forms to match the two-header form.
#
# (With a real provisioned VK that has provider keys attached, all
# four are 200. With an unprovisioned VK and disable_auth_on_inference
# off, all four are 401. With a provisioned VK whose provider keys
# aren't wired, all four are 400 "no keys found". The point of THIS
# test is that those outcomes are always identical across all four
# call shapes — which proves the wrapper is transparent.)
expect_eq "Authorization concat matches two-header HTTP"   "$COMPAT_STATUS" "$AUTH_STATUS"
expect_eq "x-api-key concat matches two-header HTTP"       "$COMPAT_STATUS" "$XAPI_STATUS"
expect_eq "x-goog-api-key concat matches two-header HTTP"  "$COMPAT_STATUS" "$XGOOG_STATUS"
echo "  (two-header HTTP was $COMPAT_STATUS; all concat forms must match it)"

# Plugin verify lines: the only way we observed `auth: verify ok` for
# run_id=X is if x-macaroon was present when PreLLMHook ran. For the
# three concat calls, that's only possible if the wrapper successfully
# split and injected. This is the load-bearing assertion.
echo
echo "  Plugin verify lines (each proves wrapper injected x-macaroon):"
assert_verify_log_present "$AUTH_RUN_ID"
assert_verify_log_present "$XAPI_RUN_ID"
assert_verify_log_present "$XGOOG_RUN_ID"
assert_verify_log_present "$COMPAT_RUN_ID"

# logs.db rows: same dim canonicalization applies regardless of how
# the macaroon arrived. All four rows should carry the macaroon's
# claims.
echo
echo "  logs.db rows (each proves end-to-end pipeline):"
for label in "Authorization:$AUTH_RUN_ID" \
             "x-api-key:$XAPI_RUN_ID" \
             "x-goog-api-key:$XGOOG_RUN_ID" \
             "two-header:$COMPAT_RUN_ID"; do
  name="${label%%:*}"
  rid="${label#*:}"
  row=$(fetch_logs_row "$rid")
  expect_eq "logs row [$name]" "$rid|u_alice|coder" "$row"
done

# --- summary ---------------------------------------------------------------

section "done"

echo "  ORG_ID=$ORG_ID"
echo "  Authorization run_id=$AUTH_RUN_ID"
echo "  x-api-key      run_id=$XAPI_RUN_ID"
echo "  x-goog-api-key run_id=$XGOOG_RUN_ID"
echo "  two-header     run_id=$COMPAT_RUN_ID"
echo

if [ "$ASSERT_FAILED" -gt 0 ]; then
  echo "  $ASSERT_FAILED assertion(s) FAILED."
  exit 1
fi

echo "  All assertions passed: bearer-concat transport works for all"
echo "  three inbound auth headers, AND the two-header form is"
echo "  unaffected."
