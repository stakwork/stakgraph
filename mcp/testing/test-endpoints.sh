#!/bin/bash
set -euo pipefail

BASE_URL="${MCP_URL:-http://localhost:3355}"
PASS=0
FAIL=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

assert_true() {
  local desc="$1"
  local val="$2"
  TOTAL=$((TOTAL + 1))
  if [ "$val" = "true" ]; then
    PASS=$((PASS + 1))
    echo -e "  ${GREEN}PASS: $desc${NC}"
  else
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}FAIL: $desc (got: $val)${NC}"
  fi
}

assert_eq() {
  local desc="$1"
  local actual="$2"
  local expected="$3"
  TOTAL=$((TOTAL + 1))
  if [ "$actual" = "$expected" ]; then
    PASS=$((PASS + 1))
    echo -e "  ${GREEN}PASS: $desc${NC}"
  else
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}FAIL: $desc (expected: $expected, got: $actual)${NC}"
  fi
}

echo -e "${YELLOW}=== MCP Integration Tests against $BASE_URL ===${NC}"
echo ""

# ─── Group 1: /nodes — Type queries ───────────────────────────────────────────

echo -e "${YELLOW}--- /nodes endpoint ---${NC}"

ENDPOINTS=$(curl -s "$BASE_URL/nodes?node_type=Endpoint&output=json")
assert_eq "Endpoint count = 3" "$(echo "$ENDPOINTS" | jq 'length')" "3"
assert_true "Known endpoint /people exists" "$(echo "$ENDPOINTS" | jq 'any(.[]; .properties.name == "/people")')"
assert_true "Known endpoint /person exists" "$(echo "$ENDPOINTS" | jq 'any(.[]; .properties.name == "/person")')"
assert_true "Known endpoint /person/{id} exists" "$(echo "$ENDPOINTS" | jq 'any(.[]; .properties.name == "/person/{id}")')"
assert_true "Endpoint verbs correct" "$(echo "$ENDPOINTS" | jq '[.[].properties.verb] | sort == ["GET","GET","POST"]')"

FUNCTIONS=$(curl -s "$BASE_URL/nodes?node_type=Function&output=json")
FUNC_COUNT=$(echo "$FUNCTIONS" | jq 'length')
assert_true "Function count > 10" "$(echo "$FUNCTIONS" | jq 'length > 10')"
assert_true "Go function NewRouter exists" "$(echo "$FUNCTIONS" | jq 'any(.[]; .properties.name == "NewRouter")')"
assert_true "Go function GetPeople exists" "$(echo "$FUNCTIONS" | jq 'any(.[]; .properties.name == "GetPeople")')"
assert_true "React function App exists" "$(echo "$FUNCTIONS" | jq 'any(.[]; .properties.name == "App")')"
assert_true "React function People exists" "$(echo "$FUNCTIONS" | jq 'any(.[]; .properties.name == "People")')"

assert_eq "Limit=3 returns 3" "$(curl -s "$BASE_URL/nodes?node_type=Function&limit=3&output=json" | jq 'length')" "3"
assert_eq "Invalid label returns empty" "$(curl -s "$BASE_URL/nodes?node_type=BOGUS&output=json" | jq 'length')" "0"

CLASSES=$(curl -s "$BASE_URL/nodes?node_type=Class&output=json")
assert_true "Class count >= 1" "$(echo "$CLASSES" | jq 'length >= 1')"
assert_true "Class database exists" "$(echo "$CLASSES" | jq 'any(.[]; .properties.name == "database")')"

DMS=$(curl -s "$BASE_URL/nodes?node_type=Datamodel&output=json")
assert_true "Datamodel count >= 1" "$(echo "$DMS" | jq 'length >= 1')"
assert_true "Datamodel Person exists" "$(echo "$DMS" | jq 'any(.[]; .properties.name == "Person")')"

PAGES=$(curl -s "$BASE_URL/nodes?node_type=Page&output=json")
assert_eq "Page count = 2" "$(echo "$PAGES" | jq 'length')" "2"

REQUESTS=$(curl -s "$BASE_URL/nodes?node_type=Request&output=json")
assert_true "Request count >= 2" "$(echo "$REQUESTS" | jq 'length >= 2')"

echo ""

# ─── Group 2: POST /nodes ─────────────────────────────────────────────────────

echo -e "${YELLOW}--- POST /nodes ---${NC}"

POST_ENDPOINTS=$(curl -s -X POST "$BASE_URL/nodes" -H 'Content-Type: application/json' \
  -d '{"node_type":"Endpoint","output":"json"}')
assert_eq "POST Endpoint count = 3" "$(echo "$POST_ENDPOINTS" | jq 'length')" "3"

REF_ID=$(curl -s "$BASE_URL/nodes?node_type=Endpoint&limit=1&output=json" | jq -r '.[0].ref_id')
if [ -n "$REF_ID" ] && [ "$REF_ID" != "null" ]; then
  REFID_RESULT=$(curl -s -X POST "$BASE_URL/nodes" -H 'Content-Type: application/json' \
    -d "{\"ref_ids\":[\"$REF_ID\"],\"output\":\"json\"}")
  REFID_COUNT=$(echo "$REFID_RESULT" | jq 'length')
  assert_true "ref_ids lookup returns >= 1" "$([ "$REFID_COUNT" -ge 1 ] && echo true || echo false)"
else
  echo -e "  ${RED}FAIL: Could not get ref_id for ref_ids test${NC}"
  FAIL=$((FAIL + 1))
  TOTAL=$((TOTAL + 1))
fi

POST_LIMIT=$(curl -s -X POST "$BASE_URL/nodes" -H 'Content-Type: application/json' \
  -d '{"node_type":"Function","limit":2,"output":"json"}')
assert_eq "POST limit=2 returns 2" "$(echo "$POST_LIMIT" | jq 'length')" "2"

echo ""

# ─── Group 3: /graph ──────────────────────────────────────────────────────────

echo -e "${YELLOW}--- /graph endpoint ---${NC}"

GRAPH=$(curl -s "$BASE_URL/graph?limit=50")
assert_true "Graph status = Success" "$(echo "$GRAPH" | jq '.status == "Success"')"
assert_true "Graph has nodes" "$(echo "$GRAPH" | jq '(.nodes | length) > 0')"
assert_true "Graph meta has Function" "$(echo "$GRAPH" | jq '.meta.node_types | index("Function") != null')"
assert_true "Graph meta has Endpoint" "$(echo "$GRAPH" | jq '.meta.node_types | index("Endpoint") != null')"

GRAPH_TOTAL=$(curl -s "$BASE_URL/graph?limit=10&limit_mode=total")
assert_true "limit_mode=total respects limit" "$(echo "$GRAPH_TOTAL" | jq '(.nodes | length) <= 10')"

GRAPH_EDGES=$(curl -s "$BASE_URL/graph?limit=100&edges=true")
assert_true "Graph has edges (with edges=true)" "$(echo "$GRAPH_EDGES" | jq '(.edges | length) > 0')"

echo ""

# ─── Group 4: /search — Fulltext ──────────────────────────────────────────────

echo -e "${YELLOW}--- /search endpoint ---${NC}"

assert_true "Search NewRouter finds results" \
  "$(curl -s "$BASE_URL/search?query=NewRouter&output=json&limit=5" | jq 'length > 0')"

assert_true "Search GetPeople finds results" \
  "$(curl -s "$BASE_URL/search?query=GetPeople&output=json&limit=5" | jq 'length > 0')"

assert_true "Search NewPerson finds results" \
  "$(curl -s "$BASE_URL/search?query=NewPerson&output=json&limit=5" | jq 'length > 0')"

FILTERED=$(curl -s "$BASE_URL/search?query=Person&node_types=Function&output=json&limit=10")
assert_true "Search with node_types=Function returns only Functions" \
  "$(echo "$FILTERED" | jq 'if length == 0 then true else all(.[]; .node_type == "Function") end')"

assert_eq "Nonsense search returns empty" \
  "$(curl -s "$BASE_URL/search?query=zzzzqqqwwwxxx123&output=json" | jq 'length')" "0"

SNIPPET=$(curl -s "$BASE_URL/search?query=NewRouter&output=snippet&limit=1")
if echo "$SNIPPET" | grep -q "snippet"; then
  assert_true "Snippet output contains <snippet>" "true"
else
  assert_true "Snippet output contains <snippet>" "false"
fi

echo ""

# ─── Group 5: /edges ──────────────────────────────────────────────────────────

echo -e "${YELLOW}--- /edges endpoint ---${NC}"

HANDLER_EDGES=$(curl -s "$BASE_URL/edges?edge_type=HANDLER&output=json")
assert_true "HANDLER edges exist" "$(echo "$HANDLER_EDGES" | jq 'length > 0')"

CONTAINS_EDGES=$(curl -s "$BASE_URL/edges?edge_type=CONTAINS&output=json")
assert_true "CONTAINS edges exist" "$(echo "$CONTAINS_EDGES" | jq 'length > 0')"

echo ""

# ─── Group 6: /repos ──────────────────────────────────────────────────────────

echo -e "${YELLOW}--- /repos endpoint ---${NC}"

REPOS=$(curl -s "$BASE_URL/repos")
assert_true "At least 1 repo" "$(echo "$REPOS" | jq 'length >= 1')"
assert_true "demo-repo in repos" "$(echo "$REPOS" | jq 'any(.[]; .name | contains("demo-repo"))')"

echo ""

# ─── Group 7: /map ────────────────────────────────────────────────────────────

echo -e "${YELLOW}--- /map endpoint ---${NC}"

MAP=$(curl -s "$BASE_URL/map?name=App&node_type=Function")
if echo "$MAP" | grep -q "App"; then
  assert_true "Map for App contains App" "true"
else
  assert_true "Map for App contains App" "false"
fi

echo ""

# ─── Group 8: Cross-language validation ───────────────────────────────────────

echo -e "${YELLOW}--- Cross-language validation ---${NC}"

ALL_FUNCS=$(curl -s "$BASE_URL/nodes?node_type=Function&output=json")
assert_true "Has Go functions (.go files)" \
  "$(echo "$ALL_FUNCS" | jq 'any(.[]; .properties.file | endswith(".go"))')"
assert_true "Has React functions (.tsx files)" \
  "$(echo "$ALL_FUNCS" | jq 'any(.[]; .properties.file | endswith(".tsx"))')"

assert_true "Requests reference frontend paths" \
  "$(curl -s "$BASE_URL/nodes?node_type=Request&output=json" | jq 'any(.[]; .properties.name | contains("/person"))')"

echo ""

# ─── Summary ──────────────────────────────────────────────────────────────────

echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo -e "  Total: $TOTAL  Passed: ${GREEN}$PASS${NC}  Failed: ${RED}$FAIL${NC}"
echo -e "${YELLOW}═══════════════════════════════════════${NC}"

if [ "$FAIL" -gt 0 ]; then
  echo -e "${RED}FAILED${NC}"
  exit 1
else
  echo -e "${GREEN}ALL PASSED${NC}"
  exit 0
fi
