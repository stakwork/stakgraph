#!/usr/bin/env bash
# Log payload offload (BIFROST_LOGS_S3_*), end to end: the real gateway
# image against MinIO, with real provider calls.
#
#   make docker-build
#   ENV_FILE=/path/to/provider.env scripts/logs-s3-e2e.sh
#
# ENV_FILE holds ANTHROPIC_API_KEY (passed to the container with
# --env-file, never printed). IMAGE defaults to stakgraph-gateway:dev,
# PORT to 18181. Everything it starts is named logs-s3-e2e-* (its own
# network, volumes, MinIO, Redis and gateway) and removed on exit
# unless KEEP=1; the stakgraph-gateway compose stack is not touched.
#
# What it shows, in order:
#   off      vars unset: config.json == seed, payloads in logs.db (these
#            rows are the "existing logs.db" for the next phase)
#   on       same volume, offload on, credentials of a MinIO user that
#            holds ONLY the README's policy:
#            a. objects at <prefix>/logs/YYYY/MM/DD/HH/<id>.json.gz
#            b. the SQLite rows' payload columns stripped
#            c. /api/logs/{id} and /_plugin/runs/:run/calls/:id full
#            d. plugin spend rollups == SUM(cost) in logs.db
#            f. pre-offload rows still read in full
#   off-again vars unset over the offloaded volume
#   bad      g. a bad bucket / bad keys / unreachable endpoint / one key
#            only: the gateway still boots and serves, without offload
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-stakgraph-gateway:dev}
ENV_FILE=${ENV_FILE:-$HERE/../.env}
PORT=${PORT:-18181}
MODEL=${MODEL:-anthropic/claude-haiku-4-5}

P=logs-s3-e2e
NET=$P-net
MINIO=$P-minio
REDIS=$P-redis
GW=$P-gw
VOL=$P-data
VOL_BAD=$P-data-bad
SQLITE_IMG=$P-sqlite:tmp
BASE=http://127.0.0.1:$PORT

BUCKET=fleet-logs
PREFIX=e2e/swarm-test
MINIO_ROOT=e2e-root
MINIO_ROOT_PASS=$(openssl rand -hex 16)
GW_KEY_ID=gateway-e2e
GW_SECRET=$(openssl rand -hex 16)
ADMIN_USER=admin
ADMIN_PASS=e2e-admin-$(openssl rand -hex 6)
TOKEN=e2e-provisioning-$(openssl rand -hex 6)

[ -f "$ENV_FILE" ] || { echo "ENV_FILE=$ENV_FILE not found" >&2; exit 2; }
for t in docker jq curl openssl; do command -v $t >/dev/null || { echo "need $t" >&2; exit 2; }; done

FAILS=0
check() { # check <name> <0|1 ok> [detail]
  if [ "$2" = 1 ]; then echo "  PASS  $1"; else echo "  FAIL  $1${3:+  -- $3}"; FAILS=$((FAILS + 1)); fi
}
section() { echo; echo "== $*"; }
is() { [ "$@" ] && echo 1 || echo 0; }

cleanup() {
  [ "${KEEP:-}" = 1 ] && { echo "KEEP=1: leaving $P-* in place"; return; }
  docker rm -f "$GW" "$MINIO" "$REDIS" >/dev/null 2>&1 || true
  docker volume rm "$VOL" "$VOL_BAD" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  docker rmi "$SQLITE_IMG" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mc() { docker run --rm --network "$NET" -e "MC_HOST_m=http://$MINIO_ROOT:$MINIO_ROOT_PASS@$MINIO:9000" minio/mc:latest "$@"; }
sql() { docker run --rm -v "$1":/data "$SQLITE_IMG" sqlite3 /data/logs.db "$2"; }
admin() { curl -sS -u "$ADMIN_USER:$ADMIN_PASS" -H 'Content-Type: application/json' "$@"; }
plugin() { curl -sS -H "Authorization: Bearer $TOKEN" "$@"; }

# start_gw <volume> [docker run args...]: start the gateway, wait for
# /health, print the seconds it took.
start_gw() {
  local vol=$1; shift
  docker rm -f "$GW" >/dev/null 2>&1 || true
  local t0=$SECONDS
  docker run -d --name "$GW" --network "$NET" -p "127.0.0.1:$PORT:8181" \
    --env-file "$ENV_FILE" \
    -e BIFROST_ADMIN_USER="$ADMIN_USER" -e BIFROST_ADMIN_PASS="$ADMIN_PASS" \
    -e BIFROST_PROVISIONING_TOKEN="$TOKEN" \
    -e BIFROST_PLUGIN_REDIS_URL="redis://$REDIS:6379/0" \
    -v "$vol":/app/data "$@" "$IMAGE" >/dev/null
  for _ in $(seq 1 400); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ]; then
      BOOT_SECS=$((SECONDS - t0)); return 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$GW")" != true ]; then
      echo "gateway exited:"; docker logs "$GW" 2>&1 | tail -30; return 1
    fi
    sleep 0.5
  done
  echo "gateway not healthy after 200s"; docker logs "$GW" 2>&1 | tail -30; return 1
}
stop_gw() { docker stop -t 20 "$GW" >/dev/null; docker rm "$GW" >/dev/null; }
gwlog() { docker logs "$GW" 2>&1; }
seed_equal() { docker exec "$GW" cmp -s /app/config.json.seed /app/data/config.json && echo 1 || echo 0; }

VK=""
mint_vk() {
  local cust
  cust=$(admin -X POST "$BASE/api/governance/customers" -d '{"name":"logs-s3-e2e","budget":{"max_limit":5,"reset_duration":"1d"}}' | jq -r '.customer.id')
  VK=$(admin -X POST "$BASE/api/governance/virtual-keys" \
    -d "{\"name\":\"logs-s3-e2e\",\"customer_id\":\"$cust\",\"provider_configs\":[{\"provider\":\"anthropic\",\"allowed_models\":[\"*\"],\"key_ids\":[\"*\"]}]}" \
    | jq -r '.virtual_key.value')
  check "virtual key minted" "$(is -n "$VK" -a "$VK" != null)"
}

# chat <model> <run_id> <prompt> [stream]: prints the HTTP status
chat() {
  local stream=false; [ "${4:-}" = stream ] && stream=true
  curl -sS -o /dev/null -w '%{http_code}' "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' -H "x-bf-vk: $VK" \
    -H "x-bf-dim-run-id: $2" -H 'x-bf-dim-agent-name: logs-s3-e2e' -H 'x-bf-dim-user-id: u-e2e' \
    -d "{\"model\":\"$1\",\"stream\":$stream,\"max_tokens\":40,\"messages\":[{\"role\":\"system\",\"content\":\"You are terse. SYSTEM-MARKER-$2\"},{\"role\":\"user\",\"content\":\"first turn $2\"},{\"role\":\"assistant\",\"content\":\"ok\"},{\"role\":\"user\",\"content\":\"Say hi in three words. $3\"}]}"
}

# wait_run <run_id> <n>: wait until n logs of a run are listed and done
wait_run() {
  for _ in $(seq 1 60); do
    local got
    got=$(admin "$BASE/api/logs?limit=50&metadata_run-id=$1" | jq '[.logs[] | select(.status != "processing")] | length')
    [ "${got:-0}" -ge "$2" ] && return 0
    sleep 1
  done
  echo "  (run $1: fewer than $2 finished logs after 60s)"
}
log_ids() { admin "$BASE/api/logs?limit=50&metadata_run-id=$1" | jq -r '.logs[].id'; }

# ─────────────────────────────────────────────────────────────────────
section "setup: network, MinIO (bucket + least-privilege user), Redis, sqlite helper"
docker network create "$NET" >/dev/null
docker run -d --name "$MINIO" --network "$NET" -e MINIO_ROOT_USER="$MINIO_ROOT" -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASS" \
  minio/minio:latest server /data >/dev/null
docker run -d --name "$REDIS" --network "$NET" redis:7-alpine >/dev/null
printf 'FROM alpine:3.23\nRUN apk add --no-cache sqlite\n' | docker build -q -t "$SQLITE_IMG" - >/dev/null
for _ in $(seq 1 30); do mc ready m >/dev/null 2>&1 && break; sleep 1; done
mc mb -q "m/$BUCKET" >/dev/null
# The README's policy, verbatim in effect: ListBucket on the bucket (the
# boot HeadBucket), object read/write/delete/tagging under the prefix.
POLICY=$(cat <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":["arn:aws:s3:::$BUCKET"]},
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:PutObjectTagging","s3:GetObjectTagging"],
  "Resource":["arn:aws:s3:::$BUCKET/$PREFIX/*"]}]}
EOF
)
docker run --rm --network "$NET" -e "MC_HOST_m=http://$MINIO_ROOT:$MINIO_ROOT_PASS@$MINIO:9000" --entrypoint sh minio/mc:latest -c \
  "echo '$POLICY' > /tmp/p.json && mc admin policy create m gateway-logs /tmp/p.json >/dev/null && mc admin user add m $GW_KEY_ID $GW_SECRET >/dev/null && mc admin policy attach m gateway-logs --user $GW_KEY_ID >/dev/null"
echo "  MinIO user $GW_KEY_ID holds only policy gateway-logs on s3://$BUCKET/$PREFIX/"

S3_ENV=(-e BIFROST_LOGS_S3_BUCKET=$BUCKET -e BIFROST_LOGS_S3_PREFIX=$PREFIX -e BIFROST_LOGS_S3_REGION=us-east-1
        -e BIFROST_LOGS_S3_ENDPOINT=http://$MINIO:9000
        -e BIFROST_LOGS_S3_ACCESS_KEY_ID=$GW_KEY_ID -e BIFROST_LOGS_S3_SECRET_ACCESS_KEY=$GW_SECRET)

# ─────────────────────────────────────────────────────────────────────
section "off: vars unset (today's behaviour)"
start_gw "$VOL"
echo "  healthy after ${BOOT_SECS}s"
check "config.json == seed, byte for byte" "$(seed_equal)"
check "wrapper logs offload off" "$(gwlog | grep -q 'logs offload: off' && echo 1 || echo 0)"
mint_vk
check "chat 200 (pre-offload)" "$(is "$(chat "$MODEL" r-pre 'pre')" = 200)"
check "chat error (bad model) logged too" "$(is "$(chat anthropic/claude-no-such-model r-pre 'pre-err')" != 200)"
wait_run r-pre 2
PRE_IDS=$(log_ids r-pre)
PRE_OK=$(admin "$BASE/api/logs?limit=50&metadata_run-id=r-pre" | jq -r '.logs[] | select(.status=="success") | .id' | head -1)
echo "  pre-offload rows:"
sql "$VOL" "SELECT '    '||id, status, has_object, length(input_history), length(output_message), length(params) FROM logs WHERE json_extract(metadata,'\$.\"run-id\"')='r-pre';"
check "pre-offload success row keeps its payload in SQLite" \
  "$(is "$(sql "$VOL" "SELECT has_object=0 AND length(output_message)>0 AND input_history LIKE '%SYSTEM-MARKER%' FROM logs WHERE id='$PRE_OK';")" = 1)"
stop_gw

# ─────────────────────────────────────────────────────────────────────
section "on: same volume, offload to MinIO"
start_gw "$VOL" "${S3_ENV[@]}"
echo "  healthy after ${BOOT_SECS}s"
gwlog | grep -E 'logs offload|logs store initialized|skills object store' | sed 's/^/    /'
CFG=$(docker exec "$GW" cat /app/data/config.json)
SEED=$(docker exec "$GW" cat /app/config.json.seed)
check "config.json = seed + logs_store (every other key identical)" \
  "$(is "$(echo "$CFG" | jq -S 'del(.logs_store)')" = "$(echo "$SEED" | jq -S .)")"
echo "    diff seed config.json:"
diff <(echo "$SEED") <(echo "$CFG") | sed 's/^/      /' || true
check "config.json carries env.* references, not the key values" \
  "$(is "$(echo "$CFG" | grep -c -e "$GW_SECRET" -e "$GW_KEY_ID")" = 0)"
check "no fallback" "$(gwlog | grep -q 'WITHOUT logs offload' && echo 0 || echo 1)"
check "chat 200" "$(is "$(chat "$MODEL" r-post 'post')" = 200)"
check "chat 200 (stream)" "$(is "$(chat "$MODEL" r-post 'post-stream' stream)" = 200)"
check "chat error (bad model)" "$(is "$(chat anthropic/claude-no-such-model r-post 'post-err')" != 200)"
wait_run r-post 3
sleep 2 # uploads are async after the row is written
POST_IDS=$(log_ids r-post)
POST_OK=$(admin "$BASE/api/logs?limit=50&metadata_run-id=r-post" | jq -r '.logs[] | select(.status=="success") | .id' | head -1)
POST_ERR=$(admin "$BASE/api/logs?limit=50&metadata_run-id=r-post" | jq -r '.logs[] | select(.status=="error") | .id' | head -1)

echo "  a. objects:"
OBJS=$(mc ls -r "m/$BUCKET/$PREFIX/logs/")
echo "$OBJS" | sed 's/^/    /'
for id in $POST_IDS; do
  check "a. object for $id at $PREFIX/logs/YYYY/MM/DD/HH/$id.json.gz" \
    "$(echo "$OBJS" | grep -qE "[0-9]{4}/[0-9]{2}/[0-9]{2}/[0-9]{2}/$id\.json\.gz" && echo 1 || echo 0)"
done
for id in $PRE_IDS; do
  check "a. no object for pre-offload $id (existing rows are not migrated)" "$(echo "$OBJS" | grep -q "$id" && echo 0 || echo 1)"
done
KEY=$(echo "$OBJS" | grep -oE "[0-9/]+$POST_OK\.json\.gz" | head -1)
STAT=$(mc stat --json "m/$BUCKET/$PREFIX/logs/$KEY")
check "a. object stored gzip (Content-Encoding: gzip)" "$(echo "$STAT" | jq -r '.metadata["Content-Encoding"] // .metadata["content-encoding"]' | grep -q gzip && echo 1 || echo 0)"
echo "    tags: $(mc tag list --json "m/$BUCKET/$PREFIX/logs/$KEY" | jq -c '.tagset')"
TMPD=$(mktemp -d)
mc cat "m/$BUCKET/$PREFIX/logs/$KEY" > "$TMPD/obj"
check "a. object body is gzip (1f 8b)" "$(is "$(head -c 2 "$TMPD/obj" | od -An -tx1 | tr -d ' ')" = 1f8b)"
echo "    $KEY: $(wc -c < "$TMPD/obj" | tr -d ' ') bytes stored"
OBJ=$( (gunzip -c < "$TMPD/obj" 2>/dev/null) || cat "$TMPD/obj")
rm -rf "$TMPD"
check "a. object holds the full payload (system prompt, output, params)" \
  "$(echo "$OBJ" | jq -e '(.input_history|contains("SYSTEM-MARKER-r-post")) and (.output_message|length>0) and (.params|length>0)' >/dev/null && echo 1 || echo 0)"

echo "  b. SQLite rows (has_object, len input_history / output_message / params / raw_response / error_details, summary):"
sql "$VOL" "SELECT '    '||id, status, has_object, length(input_history), length(output_message), length(params), length(raw_response), length(error_details), substr(content_summary,1,40) FROM logs WHERE json_extract(metadata,'\$.\"run-id\"') IN ('r-pre','r-post') ORDER BY timestamp;"
check "b. success row: has_object=1, output/params stripped, input_history = last user message only" \
  "$(is "$(sql "$VOL" "SELECT has_object=1 AND coalesce(output_message,'')='' AND coalesce(params,'')='' AND input_history NOT LIKE '%SYSTEM-MARKER%' AND input_history LIKE '%three words%' FROM logs WHERE id='$POST_OK';")" = 1)"
check "b. error row: has_object=1, error_details stripped" \
  "$(is "$(sql "$VOL" "SELECT has_object=1 AND coalesce(error_details,'')='' FROM logs WHERE id='$POST_ERR';")" = 1)"
check "b. tokens and cost stay in the row" \
  "$(is "$(sql "$VOL" "SELECT total_tokens>0 AND cost>0 FROM logs WHERE id='$POST_OK';")" = 1)"

echo "  c. detail reads:"
D=$(admin "$BASE/api/logs/$POST_OK")
check "c. /api/logs/{id}: input_history has the system prompt again" "$(echo "$D" | jq -e '.input_history | tostring | contains("SYSTEM-MARKER-r-post")' >/dev/null && echo 1 || echo 0)"
check "c. /api/logs/{id}: output_message and params back" "$(echo "$D" | jq -e '(.output_message != null) and (.params != null)' >/dev/null && echo 1 || echo 0)"
DE=$(admin "$BASE/api/logs/$POST_ERR")
check "c. /api/logs/{id} (error row): error_details back" "$(echo "$DE" | jq -e '.error_details != null' >/dev/null && echo 1 || echo 0)"
PD=$(plugin "$BASE/_plugin/runs/r-post/calls/$POST_OK")
check "c. /_plugin/runs/r-post/calls/{id}: full input_history and output_message" \
  "$(echo "$PD" | jq -e '(tostring|contains("SYSTEM-MARKER-r-post")) and (.output_message != null)' >/dev/null && echo 1 || echo 0)" \
  "$(echo "$PD" | head -c 300)"
echo "    plugin detail keys: $(echo "$PD" | jq -c 'keys')"

echo "  d. rollups:"
DB_SUM=$(sql "$VOL" "SELECT printf('%.8f', coalesce(sum(cost),0)) FROM logs WHERE json_extract(metadata,'\$.\"agent-name\"')='logs-s3-e2e';")
DB_N=$(sql "$VOL" "SELECT count(*) FROM logs WHERE json_extract(metadata,'\$.\"agent-name\"')='logs-s3-e2e';")
SP=$(plugin "$BASE/_plugin/spend/by-agent?agent_name=logs-s3-e2e&window=24h")
echo "    /_plugin/spend/by-agent: $(echo "$SP" | jq -c .)   logs.db: sum(cost)=$DB_SUM n=$DB_N"
check "d. spend/by-agent total_cost == SUM(cost) in logs.db" \
  "$(echo "$SP" | jq --arg s "$DB_SUM" '((.total_cost - ($s|tonumber)) | fabs) < 1e-9' | grep -q true && echo 1 || echo 0)"
check "d. spend/by-agent request_count == rows in logs.db" "$(is "$(echo "$SP" | jq '.request_count')" = "$DB_N")"
RUN=$(plugin "$BASE/_plugin/runs/r-post")
echo "    /_plugin/runs/r-post stats: $(echo "$RUN" | jq -c '.stats')"
RUN_SUM=$(sql "$VOL" "SELECT printf('%.8f', coalesce(sum(cost),0)) FROM logs WHERE json_extract(metadata,'\$.\"run-id\"')='r-post';")
check "d. run r-post cost == SUM(cost) of its rows" \
  "$(echo "$RUN" | jq --arg s "$RUN_SUM" '((.stats.total_cost - ($s|tonumber)) | fabs) < 1e-9' | grep -q true && echo 1 || echo 0)" \
  "stats=$(echo "$RUN" | jq -c '.stats') sum=$RUN_SUM"

echo "  f. pre-offload rows:"
DP=$(admin "$BASE/api/logs/$PRE_OK")
check "f. /api/logs/{pre id}: full payload from SQLite" "$(echo "$DP" | jq -e '(.input_history|tostring|contains("SYSTEM-MARKER-r-pre")) and (.output_message != null)' >/dev/null && echo 1 || echo 0)"
PDP=$(plugin "$BASE/_plugin/runs/r-pre/calls/$PRE_OK")
check "f. /_plugin/runs/r-pre/calls/{id}: full payload" "$(echo "$PDP" | jq -e 'tostring|contains("SYSTEM-MARKER-r-pre")' >/dev/null && echo 1 || echo 0)"
check "no objectstore warnings" "$(gwlog | grep -q 'objectstore:' && echo 0 || echo 1)" "$(gwlog | grep 'objectstore:' | head -3)"
stop_gw

# ─────────────────────────────────────────────────────────────────────
section "off-again: vars unset over the offloaded volume"
start_gw "$VOL"
check "config.json == seed again" "$(seed_equal)"
check "chat 200, payload back in SQLite" "$(is "$(chat "$MODEL" r-after 'after')" = 200)"
wait_run r-after 1
AFTER=$(log_ids r-after | head -1)
check "e. new row keeps its payload in SQLite (has_object=0)" \
  "$(is "$(sql "$VOL" "SELECT has_object=0 AND length(output_message)>0 FROM logs WHERE id='$AFTER';")" = 1)"
DO=$(admin "$BASE/api/logs/$POST_OK")
echo "    offloaded row read with offload off: input_history=$(echo "$DO" | jq -c '.input_history' | head -c 160) output_message=$(echo "$DO" | jq -c '.output_message' | head -c 60)"
check "f'. pre-offload row still full" "$(admin "$BASE/api/logs/$PRE_OK" | jq -e '.output_message != null' >/dev/null && echo 1 || echo 0)"
stop_gw

# ─────────────────────────────────────────────────────────────────────
section "g. bad destinations: the gateway must still boot and serve"
bad_case() { # bad_case <name> <docker env args...>
  local name=$1; shift
  echo "  -- $name"
  if ! start_gw "$VOL_BAD" "$@"; then check "g. $name: gateway boots" 0; return; fi
  echo "    healthy after ${BOOT_SECS}s"
  gwlog | grep -E '\[wrapper\].*(ERROR|logs offload)|failed to (ping|create) (skills )?object store|failed to bootstrap' | sed -E 's/^/    /' | cut -c1-260
  check "g. $name: gateway healthy" 1
  check "g. $name: config.json == seed after fallback" "$(seed_equal)"
  check "g. $name: chat 200" "$(is "$(chat "$MODEL" "r-bad-$name" bad)" = 200)"
  stop_gw
}
start_gw "$VOL_BAD" >/dev/null && mint_vk && stop_gw
bad_case missing-bucket "${S3_ENV[@]}" -e BIFROST_LOGS_S3_BUCKET=no-such-bucket
bad_case wrong-secret "${S3_ENV[@]}" -e BIFROST_LOGS_S3_SECRET_ACCESS_KEY=wrong
bad_case one-key-only -e BIFROST_LOGS_S3_BUCKET=$BUCKET -e BIFROST_LOGS_S3_ACCESS_KEY_ID=$GW_KEY_ID
bad_case no-region "${S3_ENV[@]}" -e BIFROST_LOGS_S3_REGION=
bad_case dns-fail "${S3_ENV[@]}" -e BIFROST_LOGS_S3_ENDPOINT=http://no-such-host.invalid:9000
bad_case blackhole "${S3_ENV[@]}" -e BIFROST_LOGS_S3_ENDPOINT=http://10.255.255.1:9000

echo
if [ "$FAILS" = 0 ]; then echo "ALL PASS"; else echo "$FAILS FAILED"; exit 1; fi
