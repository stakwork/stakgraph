#!/usr/bin/env bash
#
# smoke-test-phase-11.sh — end-to-end smoke test for the phase-11
# symmetric-recursive-authorization changes.
#
# Exercises every new surface introduced by phase 11 without depending
# on LLM provider keys. The plugin's verify pipeline runs even when
# the downstream LLM call fails, so we read the "auth: verify ok ..."
# / "auth: verify FAIL ..." lines out of the container logs to assert
# behavior.
#
# Surfaces covered
# ----------------
#   1. PUT /_plugin/trust/realm_id — set, clear, bad shape, GET 405,
#      no-auth 401.
#   2. /status surfaces realm_id when set (omitempty otherwise).
#   3. Single-swarm path: no swarm realm_id + no macaroon realm_budgets
#      → membership check is a no-op, verify ok.
#   4. Multi-realm permitted path: swarm realm_id IS a key in
#      macaroon.realm_budgets → verify ok with non-empty
#      permitted_realms.
#   5. Multi-realm NOT permitted: swarm realm_id NOT in
#      macaroon.realm_budgets → "realm_not_permitted" in the log line.
#   6. Configuration error: multi-realm macaroon arrives at a swarm
#      with no realm_id → "realm_not_configured" in the log line.
#   7. Swarm-id-only: swarm has realm_id but macaroon has no
#      realm_budgets → verify ok (the org didn't scope per-realm).
#
# All paths run against the gateway in shadow mode (the default for
# the local docker-compose); the log line tells us what the adapter
# would have done in enforce mode. Promoting these checks to "block
# the request and assert HTTP status" is a one-line flip of
# enforce_macaroons in gateway/data/config.json.
#
# Usage
# -----
#   bash gateway/scripts/smoke-test-phase-11.sh
#
# Required: the gateway must be up (`make docker-up`) AND
# `gateway/auth/ts/node_modules` must be installed (`npm install`
# inside that dir).
#
# Environment knobs
# -----------------
#   GW                            gateway URL (default http://localhost:8181)
#   BIFROST_PROVISIONING_TOKEN    bearer for /_plugin/trust admin
#                                 (default matches docker-compose.yml)
#   ORG_ID                        org_id used for registration + mint
#                                 (default org_p11)

set -u

GW="${GW:-http://localhost:8181}"
TOKEN="${BIFROST_PROVISIONING_TOKEN:-dev-provisioning-token}"
ORG_ID="${ORG_ID:-org_p11}"

# Fixture org pubkey #0 — matches priv 0x…01 used by mint-macaroon.ts.
# See gateway/auth/fixtures/keys.json.
ORG_PUBKEY="0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TS_DIR="$REPO_ROOT/gateway/auth/ts"
TSX="$TS_DIR/node_modules/.bin/tsx"
COMPOSE_FILE="$REPO_ROOT/gateway/docker-compose.yml"

ASSERT_FAILED=0

section() { printf '\n=== %s ===\n' "$1"; }

# expect_eq <label> <expected> <actual>
expect_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    printf '  [PASS] %s\n' "$label"
  else
    printf '  [FAIL] %s\n    expected: %q\n    actual:   %q\n' "$label" "$expected" "$actual"
    ASSERT_FAILED=$((ASSERT_FAILED + 1))
  fi
}

# expect_contains <label> <needle> <haystack>
expect_contains() {
  local label="$1" needle="$2" haystack="$3"
  if echo "$haystack" | grep -qF "$needle"; then
    printf '  [PASS] %s\n' "$label"
  else
    printf '  [FAIL] %s\n    needle:   %q\n    haystack: %q\n' "$label" "$needle" "$haystack"
    ASSERT_FAILED=$((ASSERT_FAILED + 1))
  fi
}

# mint <run_id> [realm_budgets_json]
#
# Wraps mint-macaroon.ts. Returns the base64url macaroon on stdout.
mint() {
  local run_id="$1"
  local realm_budgets="${2:-}"
  if [ -n "$realm_budgets" ]; then
    (
      cd "$TS_DIR" && \
      "$TSX" "$SCRIPT_DIR/mint-macaroon.ts" \
        --org-id "$ORG_ID" --user-id u_alice --agent coder \
        --run-id "$run_id" --max-cost-usd 1.00 --max-steps 50 --ttl-seconds 600 \
        --realm-budgets "$realm_budgets"
    )
  else
    (
      cd "$TS_DIR" && \
      "$TSX" "$SCRIPT_DIR/mint-macaroon.ts" \
        --org-id "$ORG_ID" --user-id u_alice --agent coder \
        --run-id "$run_id" --max-cost-usd 1.00 --max-steps 50 --ttl-seconds 600
    )
  fi
}

# call_with_macaroon <run_id> <macaroon>
#
# Send /v1/chat/completions with the macaroon. Without API keys the
# call fails upstream, but the plugin's verify pipeline runs first,
# which is what we want to inspect.
call_with_macaroon() {
  local run_id="$1" mac="$2"
  curl -sS -o /tmp/p11_llm.json -w '%{http_code}' \
    "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "x-macaroon: $mac" \
    -d '{"model":"anthropic/claude-haiku-4-5-20251001","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}' \
    > /dev/null
}

# verify_log_for <run_id>
#
# Returns the single "auth: verify ..." line for the given run_id,
# or empty string if not found. We tail enough lines that consecutive
# test calls don't push earlier results out of the window.
verify_log_for() {
  local run_id="$1"
  docker compose -f "$COMPOSE_FILE" logs --tail=500 bifrost 2>/dev/null \
    | grep -E "auth: verify .* run_id=$run_id" \
    | tail -1
}

# set_realm_id <value>  ("" to clear)
set_realm_id() {
  local v="$1"
  curl -sS -X PUT \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"realm_id\":\"$v\"}" \
    "$GW/_plugin/trust/realm_id" > /dev/null
}

# status_realm_id — print just the realm_id from /_plugin/trust/status
# (empty string when omitted from response).
status_realm_id() {
  curl -sS -H "Authorization: Bearer $TOKEN" "$GW/_plugin/trust/status" \
    | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('realm_id',''))"
}

# --- preflight -------------------------------------------------------------

section "preflight"

if ! curl -sf -o /dev/null "$GW/_plugin/health"; then
  echo "  gateway not reachable at $GW/_plugin/health" >&2
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

# Probe the trust admin surface — confirms BIFROST_PROVISIONING_TOKEN
# is wired.
trust_status=$(curl -sS -o /tmp/p11_admin.json -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" "$GW/_plugin/trust/status")
if [ "$trust_status" != "200" ]; then
  echo "  /_plugin/trust/status -> $trust_status" >&2
  exit 1
fi
echo "  trust registry reachable"

# --- register org ---------------------------------------------------------

section "register org $ORG_ID with fixture pubkey"

curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"org_id\":\"$ORG_ID\",\"pubkey\":\"$ORG_PUBKEY\",\"issuer_url\":\"\",\"revocation_poll_seconds\":60}" \
  "$GW/_plugin/trust" > /dev/null
echo "  org $ORG_ID registered (idempotent)"

# Make sure we start each scenario from a known state. The realm_id
# endpoint is the only mutable swarm-identity state.
set_realm_id ""
echo "  swarm realm_id cleared (initial state)"

# --- 1. realm_id admin endpoint ------------------------------------------

section "1. /_plugin/trust/realm_id admin endpoint"

# PUT happy path
resp=$(curl -sS -X PUT \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"realm_id":"w1"}' "$GW/_plugin/trust/realm_id")
expect_contains "PUT w1 returns ok" '"ok":true' "$resp"
expect_contains "PUT w1 returns realm_id" '"realm_id":"w1"' "$resp"

expect_eq "/status surfaces realm_id=w1" "w1" "$(status_realm_id)"

# Bad shape
status=$(curl -sS -o /tmp/p11_r.json -w '%{http_code}' -X PUT \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"realm_id":"has/slash"}' "$GW/_plugin/trust/realm_id")
expect_eq "bad realm_id → 400" "400" "$status"

# Clear
resp=$(curl -sS -X PUT \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"realm_id":""}' "$GW/_plugin/trust/realm_id")
expect_contains "clear returns ok" '"ok":true' "$resp"
expect_contains "clear returns empty realm_id" '"realm_id":""' "$resp"
expect_eq "/status omits realm_id after clear" "" "$(status_realm_id)"

# Method enforcement
status=$(curl -sS -o /tmp/p11_r.json -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" "$GW/_plugin/trust/realm_id")
expect_eq "GET on realm_id → 405" "405" "$status"

# Auth enforcement
status=$(curl -sS -o /tmp/p11_r.json -w '%{http_code}' -X PUT \
  -H "Content-Type: application/json" -d '{"realm_id":"w1"}' \
  "$GW/_plugin/trust/realm_id")
expect_eq "no auth → 401" "401" "$status"

# --- 2. single-swarm path (no realm fields anywhere) ---------------------

section "2. single-swarm path: no swarm realm_id, no macaroon realm_budgets"

set_realm_id ""
RUN_ID="r_p11simple_$(date +%s)_$RANDOM"
MAC=$(mint "$RUN_ID")
call_with_macaroon "$RUN_ID" "$MAC"
sleep 1
line=$(verify_log_for "$RUN_ID")
expect_contains "single-swarm: verify ok logged" "verify ok" "$line"
expect_contains "single-swarm: empty permitted_realms" "permitted_realms=[]" "$line"

# --- 3. multi-realm permitted --------------------------------------------

section "3. multi-realm permitted: swarm.realm_id ∈ macaroon.realm_budgets"

set_realm_id "w1"
RUN_ID="r_p11permit_$(date +%s)_$RANDOM"
MAC=$(mint "$RUN_ID" '{"w1":{"max_total_usd":50},"w2":{"max_total_usd":20}}')
call_with_macaroon "$RUN_ID" "$MAC"
sleep 1
line=$(verify_log_for "$RUN_ID")
expect_contains "multi-realm permitted: verify ok logged" "verify ok" "$line"
expect_contains "multi-realm permitted: permitted_realms surfaced" "permitted_realms=[w1 w2]" "$line"

# --- 4. multi-realm NOT permitted ----------------------------------------

section "4. multi-realm NOT permitted: swarm.realm_id ∉ macaroon.realm_budgets"

set_realm_id "w3"  # swarm claims w3, macaroon authorizes w1+w2
RUN_ID="r_p11deny_$(date +%s)_$RANDOM"
MAC=$(mint "$RUN_ID" '{"w1":{"max_total_usd":50},"w2":{"max_total_usd":20}}')
call_with_macaroon "$RUN_ID" "$MAC"
sleep 1
line=$(verify_log_for "$RUN_ID")
expect_contains "multi-realm denied: verify FAIL logged" "verify FAIL" "$line"
expect_contains "multi-realm denied: realm_not_permitted code" "code=realm_not_permitted" "$line"

# --- 5. macaroon realms only, swarm has no realm_id ----------------------

section "5. macaroon has realm_budgets but swarm has no realm_id"

set_realm_id ""  # swarm has no identity
RUN_ID="r_p11misconfigured_$(date +%s)_$RANDOM"
MAC=$(mint "$RUN_ID" '{"w1":{"max_total_usd":50}}')
call_with_macaroon "$RUN_ID" "$MAC"
sleep 1
line=$(verify_log_for "$RUN_ID")
expect_contains "misconfigured: verify FAIL logged" "verify FAIL" "$line"
expect_contains "misconfigured: realm_not_configured code" "code=realm_not_configured" "$line"

# --- 6. swarm realm_id but no macaroon realm_budgets ---------------------

section "6. swarm has realm_id but macaroon has no realm_budgets"

set_realm_id "w1"
RUN_ID="r_p11nomac_$(date +%s)_$RANDOM"
MAC=$(mint "$RUN_ID")  # no --realm-budgets
call_with_macaroon "$RUN_ID" "$MAC"
sleep 1
line=$(verify_log_for "$RUN_ID")
expect_contains "swarm-id only: verify ok logged" "verify ok" "$line"
expect_contains "swarm-id only: empty permitted_realms" "permitted_realms=[]" "$line"

# --- 7. cross-realm attenuation (just confirm verify still works) --------
#
# We don't have an attenuator binary in the tree, so this section just
# rounds-out coverage by minting a multi-realm macaroon that partially
# overlaps with the swarm's realm_id and confirming the verifier
# accepts it. The unit tests in gateway/auth/{go,ts}/ cover the
# attenuation HMAC path byte-for-byte against fixtures
# 02-one-attenuation, 03-two-attenuations, 07-cross-realm-attenuation.

section "7. multi-realm permitted with subset realms"

set_realm_id "w2"  # swarm is w2; macaroon authorizes w1+w2
RUN_ID="r_p11subset_$(date +%s)_$RANDOM"
MAC=$(mint "$RUN_ID" '{"w1":{"max_total_usd":50},"w2":{"max_total_usd":20}}')
call_with_macaroon "$RUN_ID" "$MAC"
sleep 1
line=$(verify_log_for "$RUN_ID")
expect_contains "subset permitted: verify ok logged" "verify ok" "$line"
expect_contains "subset permitted: permitted_realms=[w1 w2]" "permitted_realms=[w1 w2]" "$line"

# --- restore ---------------------------------------------------------------

section "cleanup"
set_realm_id ""
echo "  swarm realm_id cleared"

# --- summary --------------------------------------------------------------

section "summary"

if [ "$ASSERT_FAILED" -eq 0 ]; then
  echo "  ALL ASSERTIONS PASSED"
  exit 0
fi
echo "  $ASSERT_FAILED assertion(s) FAILED"
exit 1
