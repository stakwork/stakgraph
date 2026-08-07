// Backfill: normalize AgentSession.node_key to STRING.
//
// External clients supply their own session ids. When one sent a numeric id as
// a JSON number (e.g. {"sessionId": 151395375}), the value reached Cypher
// untouched and MERGE stored `node_key` as a Float (1.51395375E8) rather than
// the string "151395375".
//
// Cypher equality is type-strict, so GET_AGENT_SESSION_QUERY
// (`MATCH (n:AgentSession {node_key: $session_id})`, called with a string id
// from the URL path) never matched these nodes. LIST_AGENT_SESSIONS_QUERY
// matches on label only and stringifies afterwards, so it kept working — which
// is why affected sessions showed correct token usage in /api/sessions and
// all-zeros in /api/sessions/:id.
//
// The code fix (String() coercion at the route boundary, mcp/src/repo/index.ts)
// stops new nodes from being written this way. This script repairs the ones
// already in the graph.
//
// The type test is `NOT toString(n.node_key) = n.node_key` rather than
// valueType(): it needs no version floor, and cross-type `=` yields false in
// Cypher rather than erroring. String keys compare equal to their own
// toString() and are skipped; numeric keys don't and are selected.
//
// Run against prod with cypher-shell:
//   cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
//     -f mcp/scripts/backfill-agent-session-node-key.cypher

// ---------------------------------------------------------------------------
// STEP 1 — dry run. Count and preview affected nodes. Writes nothing.
// ---------------------------------------------------------------------------
MATCH (n:AgentSession)
WHERE n.node_key IS NOT NULL AND NOT toString(n.node_key) = n.node_key
RETURN count(n) AS affected;

MATCH (n:AgentSession)
WHERE n.node_key IS NOT NULL AND NOT toString(n.node_key) = n.node_key
RETURN n.node_key                        AS current_key,
       toString(toInteger(n.node_key))   AS will_become,
       n.total_tokens                    AS total_tokens,
       n.model                           AS model
ORDER BY n.start_time DESC
LIMIT 20;

// Collision check — MUST return 0 rows before running STEP 2. A row means a
// string-keyed node already occupies the id a numeric node would rename onto;
// merge or delete those by hand first, or the graph gains duplicate keys.
MATCH (n:AgentSession)
WHERE n.node_key IS NOT NULL AND NOT toString(n.node_key) = n.node_key
WITH DISTINCT toString(toInteger(n.node_key)) AS target
MATCH (s:AgentSession)
WHERE s.node_key = target
RETURN target, count(s) AS existing_string_nodes;

// ---------------------------------------------------------------------------
// STEP 2 — the write. Run only after STEP 1 looks right.
//
// toInteger() before toString() matters: these are stored as Floats, so a bare
// toString() yields "1.51395375E8" instead of "151395375". `n.name` is updated
// too because buildRunFromNode() falls back to it when node_key is absent.
// ---------------------------------------------------------------------------
MATCH (n:AgentSession)
WHERE n.node_key IS NOT NULL AND NOT toString(n.node_key) = n.node_key
WITH n, toString(toInteger(n.node_key)) AS fixed
SET n.node_key = fixed, n.name = fixed
RETURN count(n) AS rewritten;

// ---------------------------------------------------------------------------
// STEP 3 — verify. Both counts must be 0.
// ---------------------------------------------------------------------------
MATCH (n:AgentSession)
WHERE n.node_key IS NOT NULL AND NOT toString(n.node_key) = n.node_key
RETURN count(n) AS remaining_non_string;

// Catches a toString()-without-toInteger() mistake ("1.51395375E8").
MATCH (n:AgentSession)
WHERE n.node_key CONTAINS 'E'
RETURN count(n) AS scientific_notation_leaked;
