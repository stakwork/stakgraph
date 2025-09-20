#!/bin/bash
# filepath: /Users/fayekelmith/Kelmith/Projects/stakgraph/standalone/tests/local_e2e.sh

REPO_URL="https://github.com/fayekelmith/demo-repo.git"
EXPECTED_MAP_ONE="./standalone/tests/maps/actual-map-response.html"
ACTUAL_MAP_ONE="./standalone/tests/maps/map-response.html"
EXPECTED_MAP_TWO="./standalone/tests/maps/actual-map-response-two.html"
ACTUAL_MAP_TWO="./standalone/tests/maps/map-response-two.html"
EXPECTED_REPO_MAP="./standalone/tests/maps/actual-repo-map-response.html"
ACTUAL_REPO_MAP="./standalone/tests/maps/map-repomap-response.html"
COMMIT="ebb64dd615de5c6c83f4fceb170773f28c06cbea"


docker compose -f mcp/neo4j.yaml down -v
rm -rf ./mcp/.neo4j
docker compose -f mcp/neo4j.yaml up -d

echo "Waiting for Neo4j to be healthy..."
until docker inspect --format "{{json .State.Health.Status }}" neo4j.sphinx | grep -q "healthy"; do
  echo "Neo4j is not ready yet..."
  sleep 5
done
echo "Neo4j is healthy!"

# Start the Rust Server (background)
export USE_LSP=false
cargo run --bin standalone --features neo4j > standalone.log 2>&1 &
RUST_PID=$!


# Wait for Rust server to be ready
echo "Waiting for Rust server on :7799..."
until curl -s http://localhost:7799/fetch-repos > /dev/null; do
  sleep 2
done
echo "Rust server is ready!"

# 3. Start the Nodejs Server 
cd mcp
yarn install 
yarn run dev > server.log 2>&1 &
NODE_PID=$!
cd ..


# Wait for Node server to be ready
echo "Waiting for Node server on :3355..."
until curl -s http://localhost:3355 > /dev/null; do
  sleep 2
done
echo "Node server is ready!"

# 4. INGEST repo data asynchronously
echo "Ingesting data from $REPO_URL"
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"repo_url\": \"$REPO_URL\", \"commit\": \"$COMMIT\"}" http://localhost:7799/ingest_async)
REQUEST_ID=$(echo "$RESPONSE" | grep -o '"request_id":"[^"]*' | grep -o '[^"]*$')

if [ -z "$REQUEST_ID" ]; then
  echo "Failed to get request_id from /ingest_async response: $RESPONSE"
  kill $RUST_PID $NODE_PID
  exit 1
fi

echo "Polling status for request_id: $REQUEST_ID"
while true; do
  STATUS_JSON=$(curl -s http://localhost:7799/status/$REQUEST_ID)
  STATUS=$(echo "$STATUS_JSON" | grep -o '"status":"[^"]*' | grep -o '[^"]*$')
  if [ "$STATUS" = "Complete" ]; then
    echo "Ingest complete!"
    break
  elif [[ "$STATUS" == Failed* ]]; then
    echo "Ingest failed: $STATUS_JSON"
    kill $RUST_PID $NODE_PID
    exit 1
  else
    echo "Still processing... ($STATUS)"
    sleep 3
  fi
done

# 5. Query /map endpoint
curl "http://localhost:3355/map?name=App&node_type=Function" -o "$ACTUAL_MAP_ONE"



grep -v '^<pre>' "$ACTUAL_MAP_ONE" | grep -v '^</pre>' | grep -v 'Total tokens:' > /tmp/actual_clean.html
grep -v '^<pre>' "$EXPECTED_MAP_ONE" | grep -v '^</pre>' | grep -v 'Total tokens:' > /tmp/expected_clean.html

if diff --color=always -u /tmp/expected_clean.html /tmp/actual_clean.html; then
  echo "✅ Output matches expected (structure-sensitive)"
else
  echo "❌ Output does not match expected"
  kill $RUST_PID $NODE_PID
  exit 1
fi


# Sync the graph and make another request
echo "Syncing the graph to latest commit..."
SYNC_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"repo_url\": \"$REPO_URL\"}" http://localhost:7799/sync_async)
SYNC_REQUEST_ID=$(echo "$SYNC_RESPONSE" | grep -o '"request_id":"[^"]*' | grep -o '[^"]*$')

if [ -z "$SYNC_REQUEST_ID" ]; then
  echo "Failed to get request_id from /sync_async response: $SYNC_RESPONSE"
  kill $RUST_PID $NODE_PID
  exit 1
fi

echo "Polling status for sync_request_id: $SYNC_REQUEST_ID"
while true; do
  SYNC_STATUS_JSON=$(curl -s http://localhost:7799/status/$SYNC_REQUEST_ID)
  SYNC_STATUS=$(echo "$SYNC_STATUS_JSON" | grep -o '"status":"[^"]*' | grep -o '[^"]*$')
  if [ "$SYNC_STATUS" = "Complete" ]; then
    echo "Sync complete!"
    break
  elif [[ "$SYNC_STATUS" == Failed* ]]; then
    echo "Sync failed: $SYNC_STATUS_JSON"
    kill $RUST_PID $NODE_PID
    exit 1
  else
    echo "Still syncing... ($SYNC_STATUS)"
    sleep 3
  fi
done

# --- Query for NewPerson for main Function ---
curl "http://localhost:3355/map?name=NewPerson&node_type=Function" -o "$ACTUAL_MAP_TWO"

grep -v '^<pre>' "$ACTUAL_MAP_TWO" | grep -v '^</pre>' | grep -v 'Total tokens:'  > /tmp/sorted_actual_two.html
grep -v '^<pre>' "$EXPECTED_MAP_TWO" | grep -v '^</pre>' | grep -v 'Total tokens:'  > /tmp/sorted_expected_two.html

if diff --color=always -u /tmp/sorted_expected_two.html /tmp/sorted_actual_two.html; then
  echo "✅ main Function output matches expected (order-insensitive)"
else
  echo "❌ main Function output does not match expected"
  kill $RUST_PID $NODE_PID
  exit 1
fi

curl "http://localhost:3355/repo_map?name=fayekelmith/demo-repo" -o "$ACTUAL_REPO_MAP"

grep -v '^<pre>' "$ACTUAL_REPO_MAP" | grep -v '^</pre>' | grep -v 'Total tokens:'  > /tmp/actual_repo_map_clean.html
grep -v '^<pre>' "$EXPECTED_REPO_MAP" | grep -v '^</pre>' | grep -v 'Total tokens:' > /tmp/expected_repo_map_clean.html

if diff --color=always -u /tmp/expected_repo_map_clean.html /tmp/actual_repo_map_clean.html; then
  echo "✅ Repo map output matches expected (structure-sensitive)"
else
  echo "❌ Repo map output does not match expected"
  kill $RUST_PID $NODE_PID
  exit 1
fi

#  Cleanup
kill $RUST_PID $NODE_PID
