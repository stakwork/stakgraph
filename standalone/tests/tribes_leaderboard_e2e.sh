#!/bin/bash

REPO_URLS="https://github.com/stakwork/sphinx-tribes.git,https://github.com/stakwork/sphinx-tribes-frontend.git"


# COMMITS=(
#   "1ae6eeea7cedbdd25c769b063a133673b66a79f3"      # sphinx-tribes
#   "c34119b044c3f55859cb650c2c105609209e9f04" # sphinx-tribes-frontend
# )

EXPECTED_MAP="./standalone/tests/maps/actual-leaderboard-map-response.html"
ACTUAL_MAP="./standalone/tests/maps/leaderboard-map-response.html"

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


# # Wait for Rust server to be ready
echo "Waiting for Rust server on :7799..."
until curl -s http://localhost:7799/fetch-repos > /dev/null; do
  sleep 5
done
echo "Rust server is ready!"

# 3. Start the Nodejs Server 
cd mcp
yarn install 
yarn run dev > server.log 2>&1 &
NODE_PID=$!
cd ..


# Wait for Node server to be ready
echo "Waiting for Node server on :3000..."
until curl -s http://localhost:3000 > /dev/null; do
  sleep 2
done
echo "Node server is ready!"

# Ingest each repo at the specified commit
echo "Ingesting repositories: $REPO_URLS"
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"repo_url\": \"$REPO_URLS\"}" http://localhost:7799/ingest_async)
REQUEST_ID=$(echo "$RESPONSE" | grep -o '"request_id":"[^"]*' | grep -o '[^"]*$')

if [ -z "$REQUEST_ID" ]; then
  echo "Failed to get request_id for repositories"
  kill $RUST_PID $NODE_PID
  exit 1
fi

# Monitor the single ingestion process
while true; do
  STATUS_JSON=$(curl -s http://localhost:7799/status/$REQUEST_ID)
  STATUS=$(echo "$STATUS_JSON" | grep -o '"status":"[^"]*' | grep -o '[^"]*$')
  
  if [ "$STATUS" = "Complete" ]; then
    echo "Ingest complete for both repositories!"
    break
  elif [[ "$STATUS" == Failed* ]]; then
    echo "Ingest failed: $STATUS_JSON"
    kill $RUST_PID $NODE_PID
    exit 1
  else
    echo "Still processing repositories... ($STATUS)"
    sleep 10
  fi
done

# Query the map endpoint for LeaderboardPage
curl "http://localhost:3000/map?node_type=Function&name=LeaderboardPage" -o "$EXPECTED_MAP"

# Clean and compare output
grep -v '^<pre>' "$ACTUAL_MAP" | grep -v '^</pre>' | grep -v 'Total tokens:' > /tmp/actual_leaderboard_clean.html
grep -v '^<pre>' "$EXPECTED_MAP" | grep -v '^</pre>' | grep -v 'Total tokens:' > /tmp/expected_leaderboard_clean.html

if diff --color=always -u /tmp/expected_leaderboard_clean.html /tmp/actual_leaderboard_clean.html; then
  echo "✅ LeaderboardPage output matches expected"
else
  echo "❌ LeaderboardPage output does not match expected"
  kill $RUST_PID $NODE_PID
  exit 1
fi

# Cleanup
kill $RUST_PID $NODE_PID