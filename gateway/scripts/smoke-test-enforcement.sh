#!/usr/bin/env bash
#
# smoke-test-enforcement.sh — end-to-end macaroon enforcement smoke test.
#
# Companion to smoke-test.sh (which only exercises the observability /
# dim-header surface). This script proves the *cryptographic identity*
# pipeline works end-to-end:
#
#   1. Register an org's pubkey in the gateway's trust registry via
#      POST /_plugin/trust.
#   2. Mint a fresh wire-format macaroon (TS, via mint-macaroon.ts)
#      signed by the matching fixture private keys.
#   3. Make a real LLM call carrying both `x-macaroon` and the dim
#      headers — same shape callers will use in production.
#   4. Surface the verification outcome:
#        - plugin logs (auth: verify ok / FAIL …)
#        - logs.db row (metadata dims land alongside the cost/latency)
#
# Today (phase 4-5 landed, phase 6 enforcement not yet wired into the
# hot path) the plugin runs in *shadow mode*: it verifies macaroons
# and logs the decision but does not reject calls. That means a
# successful smoke run looks like:
#
#   - HTTP 200 from the gateway
#   - `auth: verify ok mode=shadow org=org_smoke user=u_alice ...`
#     in the bifrost container logs
#   - a logs row written with the dim-header values present
#
# Later iterations of this script will add:
#   - attenuation chain (parent → child macaroons)
#   - enforce-mode (BIFROST_PLUGIN_ENFORCE_MACAROONS=true → 401 on bad
#     macaroon, 200 on good)
#   - per-run cost cap exceeded (Redis-side accumulator check)
#   - kill switch (POST /_plugin/runs/:id/kill → next call 402s)
#
# Usage
# -----
#   bash gateway/scripts/smoke-test-enforcement.sh
#
# Required: the gateway must be up (`make docker-up`) AND
# `gateway/auth/ts/node_modules` must be installed (`npm install`
# inside that dir; done as part of fixture regeneration too).
#
# Environment knobs
# -----------------
#   GW                            gateway URL (default http://localhost:8181)
#   BIFROST_PROVISIONING_TOKEN    bearer for /_plugin/trust admin
#                                 (default matches docker-compose.yml)
#   ORG_ID                        org_id for registration + macaroon
#                                 (default org_smoke)

set -u  # don't use -e: we want to keep going past partial failures so
        # the operator can read the full report at the end

GW="${GW:-http://localhost:8181}"
TOKEN="${BIFROST_PROVISIONING_TOKEN:-dev-provisioning-token}"
ORG_ID="${ORG_ID:-org_smoke}"

# Fixture org pubkey #0 — matches priv 0x…01 used by mint-macaroon.ts.
# See gateway/auth/fixtures/keys.json. Hardcoded here (rather than
# parsed from the file) so the script stays self-contained for ops
# debugging in an unfamiliar checkout.
ORG_PUBKEY="0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"

# Repo root, relative to this script. All paths below derived from
# this so the script works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TS_DIR="$REPO_ROOT/gateway/auth/ts"
TSX="$TS_DIR/node_modules/.bin/tsx"

# --- helpers ---------------------------------------------------------------

# section <title>
section() {
  printf '\n=== %s ===\n' "$1"
}

# call_admin <method> <path> [body]
#
# Bearer-authed call to a /_plugin/* admin route. Prints status + body
# but doesn't fail the script on non-2xx — we want to see all errors at
# the end, not stop at the first.
call_admin() {
  local method="$1" path="$2" body="${3:-}"
  local status http_body
  if [ -n "$body" ]; then
    http_body=$(curl -sS -o /tmp/smoke_admin.json -w '%{http_code}' \
      -X "$method" "$GW$path" \
      -H "Authorization: Bearer $TOKEN" \
      -H 'Content-Type: application/json' \
      -d "$body")
  else
    http_body=$(curl -sS -o /tmp/smoke_admin.json -w '%{http_code}' \
      -X "$method" "$GW$path" \
      -H "Authorization: Bearer $TOKEN")
  fi
  status="$http_body"
  printf '  %s %s -> %s\n' "$method" "$path" "$status"
  if [ -s /tmp/smoke_admin.json ]; then
    sed 's/^/    /' /tmp/smoke_admin.json
    echo
  fi
}

# new_run_id — fresh id per call so accumulator state doesn't carry
# across smoke runs.
new_run_id() {
  printf 'r_enforce_%s_%s' "$(date +%s)" "$RANDOM"
}

# mint_macaroon <run_id> [realm_budgets_json]
#
# Calls mint-macaroon.ts and emits the b64url macaroon on stdout. Any
# tsx errors land on stderr and we propagate the non-zero exit.
#
# Phase 11: the singular `--realm` flag is gone. Pass the optional
# second arg (a JSON `realm_budgets` map) to mint a multi-swarm
# macaroon — both the UA and the invocation receive that scoping.
# Omit it for the simple single-swarm shape.
mint_macaroon() {
  local run_id="$1"
  local realm_budgets="${2:-}"
  # Run tsx from the TS package dir so node module resolution picks
  # up the @noble/* / canonicalize deps. The mint script resolves
  # fixture paths relative to its own __dirname, so it works fine
  # from any cwd.
  if [ -n "$realm_budgets" ]; then
    (
      cd "$TS_DIR" && \
      "$TSX" "$SCRIPT_DIR/mint-macaroon.ts" \
        --org-id "$ORG_ID" \
        --user-id u_alice \
        --agent coder \
        --run-id "$run_id" \
        --max-cost-usd 1.00 \
        --max-steps 50 \
        --ttl-seconds 600 \
        --realm-budgets "$realm_budgets"
    )
  else
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
  fi
}

# call_llm_with_macaroon <run_id> <macaroon-b64>
#
# One non-streaming chat completion with both x-macaroon AND the
# current x-bf-dim-* convention. Cheap model, max_tokens=20.
call_llm_with_macaroon() {
  local run_id="$1" macaroon="$2"
  local started status
  started=$(date -u +%H:%M:%S)
  status=$(curl -sS -o /tmp/smoke_llm.json -w '%{http_code}' \
    "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "x-macaroon: $macaroon" \
    -H "x-bf-dim-run-id: $run_id" \
    -H "x-bf-dim-session-id: s_smoke_enforce" \
    -H "x-bf-dim-agent-name: coder" \
    -H "x-bf-dim-user-id: u_alice" \
    -H "x-bf-dim-deployment: smoke-enforcement" \
    -d '{"model":"anthropic/claude-haiku-4-5-20251001","max_tokens":20,"messages":[{"role":"user","content":"reply in 3 words"}]}')
  printf '  %s  run_id=%s  http=%s\n' "$started" "$run_id" "$status"
  if [ "$status" != "200" ] && [ -s /tmp/smoke_llm.json ]; then
    echo "  body:"
    sed 's/^/    /' /tmp/smoke_llm.json
    echo
  fi
}

# call_llm_mismatched <macaroon-b64> <lie-run-id> <lie-user> <lie-agent>
#
# Send the macaroon (with truthful claims) but DELIBERATELY MISMATCHED
# x-bf-dim-* headers. Used by the canonicalization test below: after
# the fix, the plugin must drop the lying header values and stamp the
# macaroon's claims into logs.metadata. Before the fix, the headers
# win and the logs row records the lies.
#
# Phase 11: realm-id is no longer signature-bound, so we don't try to
# canonicalize it — the three remaining bound dims are run, user,
# agent.
call_llm_mismatched() {
  local macaroon="$1" lie_run="$2" lie_user="$3" lie_agent="$4"
  local started status
  started=$(date -u +%H:%M:%S)
  status=$(curl -sS -o /tmp/smoke_llm.json -w '%{http_code}' \
    "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "x-macaroon: $macaroon" \
    -H "x-bf-dim-run-id: $lie_run" \
    -H "x-bf-dim-user-id: $lie_user" \
    -H "x-bf-dim-agent-name: $lie_agent" \
    -H "x-bf-dim-session-id: s_smoke_mismatch" \
    -H "x-bf-dim-deployment: smoke-mismatch" \
    -d '{"model":"anthropic/claude-haiku-4-5-20251001","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}')
  printf '  %s  headers={run=%s user=%s agent=%s}  http=%s\n' \
    "$started" "$lie_run" "$lie_user" "$lie_agent" "$status"
  if [ "$status" != "200" ] && [ -s /tmp/smoke_llm.json ]; then
    echo "  body:"
    sed 's/^/    /' /tmp/smoke_llm.json
    echo
  fi
}

# fetch_logs_row <run-id>
#
# Pulls one row from the gateway's logs.db whose metadata.run-id
# matches the argument. Output format: pipe-delimited
# run-id|user-id|agent-name. Empty string on no-match.
#
# Used by the mismatch assertion: after the canonicalization fix the
# row keyed by the MACAROON's run-id is the only one that exists; the
# row keyed by the HEADER's run-id is gone (proving the headers were
# overwritten).
#
# Phase 11: dropped the realm-id column from the row format because
# realm-id is no longer a signature-bound dim (the macaroon doesn't
# carry a singular realm). Every row in this swarm's logs.db is by
# definition for this swarm's realm.
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
#
# Tiny assertion helper. Sets the global ASSERT_FAILED counter; the
# script prints a final summary and exits non-zero if any fail. We
# don't `set -e` because we want to see ALL mismatches in one run.
ASSERT_FAILED=0
expect_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    printf '  [PASS] %s: %q\n' "$label" "$actual"
  else
    printf '  [FAIL] %s: expected %q got %q\n' "$label" "$expected" "$actual"
    ASSERT_FAILED=$((ASSERT_FAILED + 1))
  fi
}

# --- preflight -------------------------------------------------------------

section "preflight"

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
echo "  tsx available at $TSX"

# Probe the trust admin surface — confirms BIFROST_PROVISIONING_TOKEN
# is wired (without it, every admin call below would 401).
trust_status=$(curl -sS -o /tmp/smoke_admin.json -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" "$GW/_plugin/trust/status")
if [ "$trust_status" != "200" ]; then
  echo "  /_plugin/trust/status -> $trust_status (token wrong or admin server down)" >&2
  echo "  current BIFROST_PROVISIONING_TOKEN=${TOKEN:-<unset>}" >&2
  exit 1
fi
echo "  trust registry reachable; current contents:"
sed 's/^/    /' /tmp/smoke_admin.json
echo

# --- register org ----------------------------------------------------------

section "register org $ORG_ID with fixture pubkey"

# Idempotent on org_id (registry.Upsert) — re-running the script just
# rewrites the same entry.
call_admin POST /_plugin/trust "$(cat <<JSON
{
  "org_id": "$ORG_ID",
  "pubkey": "$ORG_PUBKEY",
  "issuer_url": "",
  "revocation_poll_seconds": 60
}
JSON
)"

# Confirm it landed.
call_admin GET "/_plugin/trust/$ORG_ID"

# --- mint + call -----------------------------------------------------------

section "mint macaroon and make LLM call"

RUN_ID=$(new_run_id)
echo "  minting macaroon for run_id=$RUN_ID..."
MACAROON=$(mint_macaroon "$RUN_ID")
mint_status=$?
if [ $mint_status -ne 0 ] || [ -z "$MACAROON" ]; then
  echo "  mint FAILED (exit=$mint_status)" >&2
  exit 1
fi
echo "  macaroon minted (${#MACAROON} chars)"
echo "  first 80 chars: ${MACAROON:0:80}..."
echo

echo "  calling /v1/chat/completions with x-macaroon..."
call_llm_with_macaroon "$RUN_ID" "$MACAROON"

# Give bifrost-core's logging plugin a beat to flush the row to disk.
echo "  waiting 3s for log flush..."
sleep 3

# --- inspect results -------------------------------------------------------

section "inspect plugin logs"
echo "  Look for 'auth: verify ok mode=shadow ...' lines for this run:"
echo
echo "    docker compose -f gateway/docker-compose.yml logs --tail=200 bifrost \\"
echo "      | grep -E 'auth:|run_id=$RUN_ID'"
echo
# Best-effort inline grep if docker compose is on PATH and the
# containers are running under the standard project name. Errors are
# silenced so a missing compose CLI doesn't fail the script.
if command -v docker >/dev/null 2>&1; then
  docker compose -f "$REPO_ROOT/gateway/docker-compose.yml" logs --tail=200 bifrost 2>/dev/null \
    | grep -E "auth:|$RUN_ID" \
    | sed 's/^/    /' \
    || echo "    (no matching log lines — run the grep above manually)"
fi

section "inspect logs.db row"
echo "  The logs row carries the dim headers in metadata. Look for"
echo "  the row with run-id=$RUN_ID:"
echo
echo "    docker run --rm -v stakgraph-gateway-data:/data alpine:3.23 \\"
echo "      sh -c \"apk add -q sqlite >/dev/null && sqlite3 /data/logs.db \\"
echo "        \\\"SELECT json_extract(metadata,'\\\$.run-id') AS run_id,"
echo "                 json_extract(metadata,'\\\$.user-id') AS user,"
echo "                 json_extract(metadata,'\\\$.agent-name') AS agent,"
echo "                 model, ROUND(cost,6) AS cost, status"
echo "          FROM logs WHERE json_extract(metadata,'\\\$.run-id')='$RUN_ID';\\\"\""

# Best-effort inline run if docker is available.
if command -v docker >/dev/null 2>&1; then
  echo
  echo "  inline result:"
  docker run --rm -v stakgraph-gateway-data:/data alpine:3.23 \
    sh -c "apk add -q sqlite >/dev/null && sqlite3 /data/logs.db \
      \"SELECT json_extract(metadata,'\$.run-id') AS run_id,
              json_extract(metadata,'\$.user-id') AS user,
              json_extract(metadata,'\$.agent-name') AS agent,
              model,
              ROUND(cost,6) AS cost,
              status
       FROM logs WHERE json_extract(metadata,'\$.run-id')='$RUN_ID';\"" \
    2>/dev/null \
    | sed 's/^/    /' \
    || echo "    (no row found — the call may have failed before logging)"
fi

# --- mismatch test: macaroon claims must override caller headers ----------
#
# The dim headers and the macaroon caveats overlap on four fields
# (run-id, user-id, agent-name, realm-id). The macaroon is
# cryptographically verified; the headers are caller-controlled and
# could be anything. The plugin MUST treat the macaroon as
# authoritative — otherwise a caller could log spending under any
# user's name by stamping a header.
#
# This section fires one request where the headers DELIBERATELY LIE
# about every signature-bound dim. We then read the logs.db row back
# and assert it carries the MACAROON's values, not the headers'.
#
# Today (pre-fix) this section is expected to fail: headers win and
# logs.db records the lies. After llm_prehook.go canonicalizes dims
# from claims, the assertions pass.

section "mismatch test: macaroon dims override header lies"

MISMATCH_RUN_ID=$(new_run_id)
LIE_RUN_ID="r_HEADER_LIE_${MISMATCH_RUN_ID}"
LIE_USER="u_LIAR"
LIE_AGENT="malicious-agent"

echo "  minting macaroon for truthful run_id=$MISMATCH_RUN_ID..."
MISMATCH_MACAROON=$(mint_macaroon "$MISMATCH_RUN_ID")
mint_status=$?
if [ $mint_status -ne 0 ] || [ -z "$MISMATCH_MACAROON" ]; then
  echo "  mint FAILED (exit=$mint_status)" >&2
  exit 1
fi

echo "  calling /v1/chat/completions with mismatched headers..."
call_llm_mismatched "$MISMATCH_MACAROON" "$LIE_RUN_ID" "$LIE_USER" "$LIE_AGENT"

echo "  waiting 3s for log flush..."
sleep 3

echo
echo "  Plugin verify line (claims from macaroon, NOT headers):"
docker compose -f "$REPO_ROOT/gateway/docker-compose.yml" logs --tail=50 bifrost 2>/dev/null \
  | grep -E "auth: verify .* run_id=$MISMATCH_RUN_ID" \
  | tail -1 \
  | sed 's/^/    /' \
  || echo "    (no verify line found — adapter may not have processed this call)"

echo
echo "  logs.db row keyed by macaroon run_id=$MISMATCH_RUN_ID:"
mac_row=$(fetch_logs_row "$MISMATCH_RUN_ID")
if [ -n "$mac_row" ]; then
  printf '    %s\n' "$mac_row"
else
  printf '    (no row — headers won, see below)\n'
fi

echo
echo "  logs.db row keyed by header lie run_id=$LIE_RUN_ID:"
lie_row=$(fetch_logs_row "$LIE_RUN_ID")
if [ -n "$lie_row" ]; then
  printf '    %s\n' "$lie_row"
else
  printf '    (no row — macaroon won)\n'
fi

echo
echo "  Assertions (post-canonicalization-fix expected):"
# Post-fix: the macaroon-keyed row exists and carries macaroon values.
# Pre-fix:  this row does not exist; the lie-keyed row does.
expect_eq "macaroon row exists" \
  "$MISMATCH_RUN_ID|u_alice|coder" \
  "$mac_row"
# Post-fix: the lie-keyed row does not exist (headers were overwritten).
# Pre-fix:  this row exists with all three lies.
expect_eq "lie row does not exist" \
  "" \
  "$lie_row"

section "done"

echo "  Smoke run complete."
echo "    truthful RUN_ID=$RUN_ID"
echo "    mismatch macaroon-side run_id=$MISMATCH_RUN_ID  header-lie run_id=$LIE_RUN_ID"
echo "    ORG_ID=$ORG_ID"
echo

if [ "$ASSERT_FAILED" -gt 0 ]; then
  echo "  $ASSERT_FAILED assertion(s) FAILED — see [FAIL] lines above."
  echo "  This is expected pre-fix: headers currently override macaroon claims"
  echo "  in logs.metadata. Apply the canonicalization fix in"
  echo "  gateway/internal/hooks/llm_prehook.go and re-run."
  exit 1
fi

echo "  All assertions passed: macaroon claims are authoritative over"
echo "  caller-supplied x-bf-dim-* headers for signature-bound dims."
