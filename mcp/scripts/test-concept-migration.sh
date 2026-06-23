#!/usr/bin/env bash
#
# Spin up a local Neo4j, then run the Feature -> Concept migration test harness
# against it (scripts/test-concept-migration.ts).
#
# Usage:
#   ./scripts/test-concept-migration.sh           # start neo4j, run test
#   KEEP=1 ./scripts/test-concept-migration.sh    # keep seeded data after
#   NO_DOCKER=1 ./scripts/test-concept-migration.sh  # use an already-running neo4j
#
set -euo pipefail

cd "$(dirname "$0")/.."

export NEO4J_HOST="${NEO4J_HOST:-localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-testtest}"

if [ "${NO_DOCKER:-}" != "1" ]; then
  echo "==> Starting Neo4j (docker compose -f neo4j.yaml)..."
  docker compose -f neo4j.yaml up -d

  echo "==> Waiting for Neo4j to become healthy..."
  for i in $(seq 1 60); do
    status=$(docker inspect --format='{{.State.Health.Status}}' neo4j.sphinx 2>/dev/null || echo "starting")
    if [ "$status" = "healthy" ]; then
      echo "    Neo4j is healthy."
      break
    fi
    if [ "$i" = "60" ]; then
      echo "    Timed out waiting for Neo4j to be healthy (status: $status)." >&2
      exit 1
    fi
    sleep 2
  done
fi

echo "==> Running migration test harness..."
npx tsx scripts/test-concept-migration.ts
