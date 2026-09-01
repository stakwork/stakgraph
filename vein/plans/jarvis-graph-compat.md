# Jarvis-compatible graph writes from TypeScript — no jarvis in the loop

## Goal

Vein's `Neo4jWorkspaceStore` + run/chat projector (generic-storage.md §7)
talk to Neo4j **directly over bolt from TypeScript**. Jarvis is not a
dependency — but every byte we write follows jarvis's conventions, so a
jarvis instance pointed at the same database later treats the `Vein`
domain as native: its search finds our nodes, its UI renders them, its
seeder and migrations don't fight ours.

"100% compatible" means: **identical graph state** to what jarvis's own
write path would have produced — same labels, same generic properties,
same node_key composition, same embedding vectors, same indexes/
constraints, same schema meta-graph. It does NOT mean reimplementing
jarvis's HTTP API, Redis queue, scratchpad, radar, Elasticsearch, or
entity-resolution machinery (see "Explicitly out of scope").

Everything below was extracted from jarvis-backend source with
file:line references; treat those as the authority when implementing.

## Implementation notes (start here in a fresh session)

- **Read first**: this doc + the "Label registry" section of
  `generic-storage.md` (the node/edge vocabulary — do not invent
  labels).
- **Source-of-truth repos**: jarvis-backend lives at
  `/Users/evan/code/sphinx/jarvis-backend` (all `file:line` refs below
  point into it). The lab step tools defining the read surface (§7):
  `mcp/src/lab/jarvis/steps/` in this repo.
- **Branch, never main**: implement on a feature branch + PR.
- **Deps**: `neo4j-driver` (bolt) and
  `npm install @huggingface/transformers@4.1.1` (chosen for ARM+Intel
  Mac cross-compat). Config via env/secret store: `NEO4J_URI`,
  `NEO4J_USER`, `NEO4J_PASSWORD`, optional `VEIN_GRAPH_NAMESPACE`
  (default `"default"`).
- **Build order** (each step testable before the next):
  1. `bolt.ts` + `vein-schemas.ts` + `schema-seed.ts` — then assert
     seeding idempotence (run twice, graph diff empty).
  2. `node-writer.ts` + the §6 validation gate.
  3. `embeddings.ts` — golden-vector parity FIRST, then the boot sweep.
  4. `edge-writer.ts`.
  5. `search.ts` (§7 hybrid search + get/neighbors).
  6. `graph/*` step twins of the `jarvis/*` steps.
  The vein-native boundary work (generic-storage §1–5) is a parallel,
  independent track; the two meet only in `Neo4jWorkspaceStore` + the
  projector, which come after both.
- **Parity fixtures** (check into the test tree, produced once):
  - Embedding golden vector: with jarvis's Python env
    (`sentence-transformers`), encode a fixed string —
    `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("vein golden parity fixture").tolist()`
    → JSON fixture; TS side must reach cosine ≥ 0.999 against it.
  - Node-write + search-response snapshots: run jarvis-backend locally
    (or hit the swarm deployment), perform one `POST /v2/nodes` and one
    `GET /v2/nodes?q=...`, dump the resulting node's full property map
    (via bolt) and the HTTP response as fixtures.
- **Vein schema attributes** (§5 lists only node_key/index): derive
  each type's full attribute map from vein's existing core types —
  `RunSummary`/`RunEvent` (`src/core.ts`, `src/store.ts`) for
  `VeinRun`/`VeinAgentSession`/`VeinToolCall`, `ChatMeta`/chat events
  (`src/chat-store.ts`) for `VeinChat`/`VeinTurn`, the workspace
  types (`src/workspace.ts`) for the workflow/step schemas. Every
  node_key token must be a required (non-`?`) attribute; previews
  capped ~500 chars; full payloads by `log_ref` pointer only.
- **Known open questions** (resolve by reading jarvis source when
  reached, don't guess): how `POST /namespace` persists registrations
  (§7 table); seed `Person` or defer `PUBLISHED_BY` (§4 item 6).

## Deployment modes

- **Standalone**: a fresh Neo4j only vein writes to. Our bootstrap must
  create everything jarvis would have (Thing schema, constraints,
  indexes), so jarvis can mount later with zero surprises.
- **Shared**: jarvis already runs on the DB. Everything global exists;
  our bootstrap is idempotent (`IF NOT EXISTS` / guarded MERGE) and
  add-only, exactly like jarvis's own seeder. One caveat: jarvis holds
  in-process caches (node-type cache, 600s `/schema` HTTP cache, 60s
  domain-visibility cache) — external bolt writes don't invalidate
  them; new Vein types appear to a running jarvis after restart or a
  `/schema` POST/PUT.

## Module plan (all in `vein/src/graph/`)

| Module            | Responsibility                                          |
| ----------------- | ------------------------------------------------------- |
| `bolt.ts`         | neo4j-driver session/tx helpers                         |
| `schema-seed.ts`  | Vein domain registration (schema meta-graph + indexes)  |
| `node-writer.ts`  | jarvis-dialect node MERGE (labels, stamps, node_key)    |
| `edge-writer.ts`  | jarvis-dialect edge MERGE (edge_key, alias rewrite)     |
| `embeddings.ts`   | local MiniLM vectors via `@huggingface/transformers`    |
| `vein-schemas.ts` | the Vein schema library (TS mirror of a jarvis library) |
| `search.ts`       | hybrid search + get/neighbors reads (§7)                |

## 1. Node writes (`node-writer.ts`)

The one Cypher template (jarvis: `schema_node_helper.py:238-268,610`):

```cypher
MERGE (node:`<Type>`:Node:Data_Bank:Domain_vein {node_key: $node_key, namespace: $namespace})
 ON CREATE SET  node.p1=$p1, node.p2=$p2, ...
 ON MATCH SET node.p1=$p1, ...
 RETURN node
```

Label set = `` `Type` `` + `:Node` + `:Data_Bank` + `:Domain_<domain.lower()>`
(so ours is always `:Domain_vein`). Type label backticked, others not.

Generic stamps (exact formats matter):

| Property               | Value                                                     |
| ---------------------- | --------------------------------------------------------- |
| `ref_id`               | `crypto.randomUUID()` — lowercase uuid4 with dashes       |
| `date_added_to_graph`  | epoch **milliseconds** as a Neo4j **Integer** (`neo4j.int(Date.now())`) |
| `namespace`            | `"default"` unless configured                             |
| `node_key`             | see composition below                                     |
| `Data_Bank`            | search text — see §2                                      |
| `_search_fields_used`  | string array of the fields that built `Data_Bank`         |

Asymmetry to preserve: any attribute our schema declares `datetime` is
normalized to epoch **seconds** (int) before write
(`schema_validation.py:281-308`), while `date_added_to_graph` is epoch
**ms**. Never write ISO strings for datetime attrs.

**node_key composition** (`schema_validation.py:350-399`) — implement
verbatim:

1. Split the schema's `node_key` spec on `-` (e.g. `veinrun-run_id`).
2. Token 0 = the type name: `.toLowerCase()` then strip
   `[^a-zA-Z0-9\s]` (so `VeinRun` → `veinrun`).
3. Each later token = the named property's value:
   `String(v).trim()` → remove spaces (`replace(/ /g,"")`) →
   `.toLowerCase()` → strip `[^a-zA-Z0-9\s]`. Property lookup is
   case-insensitive; a missing property is an error.
4. Join with `-`. If the result exceeds **200 chars** (and has ≥2
   parts): `parts[0] + "-" + sha256(parts.slice(1).join("-")).hex().slice(0,32)`.

**Property-writing rules** (`schema_node_helper.py:251-257`): skip any
property whose `String(value).length === 0` (empty string never
written); property names must be bare Cypher identifiers (we control
our schemas, so enforce `^[a-zA-Z_][a-zA-Z0-9_]*$` at schema-authoring
time); flat properties only — no JSON `properties` blob.

**Duplicate semantics**: jarvis pre-checks
`MATCH (n:`Type`) WHERE n.node_key=$k AND n.namespace=$ns` and on hit
does NOT write (default), or updates-in-place preserving `ref_id`,
`node_key`, `namespace`, `date_added_to_graph` (`reprocess`), or
restores if soft-deleted. Our writer implements two modes:
`create` (no-op on existing, return existing ref_id) and `upsert`
(reprocess semantics — SET everything except the preserved five). The
projector uses `upsert` (runs get re-projected); the workspace store
uses `create` + explicit update paths. Never let `ON MATCH SET`
overwrite `ref_id`/`date_added_to_graph` — put those only in the
ON CREATE clause (jarvis has a latent race bug here; we fix it, the
resulting state is what jarvis intends).

Soft delete = `SET n.is_deleted = true`; readers filter
`(n.is_deleted IS NULL OR n.is_deleted = false)`. Never DETACH DELETE
except an explicit destructive admin path. **Restore semantics**
(mirroring jarvis `schema_node_helper.py:620-657`): a `create` that
hits a soft-deleted node with the same node_key restores it —
`is_deleted = false`, new payload applied, preserving `ref_id`,
`node_key`, `namespace`, `date_added_to_graph`.

**Idempotent projector writes**: stamp `unique_source_id` on projected
nodes (e.g. `"veinrun:<runId>"`, `"veintoolcall:<runId>:<path>:<seq>"`)
— jarvis already indexes it (`node_unique_source_id_index`) and stamps
it onto edges when both endpoints share the value, so re-projection
and cross-store reconciliation get a cheap identity probe for free.

**Batching**: the projector writes many nodes per run — use the UNWIND
form of the same MERGE (one statement per type per batch, same
resulting state), mirroring jarvis's bulk paths.

## 2. `Data_Bank` search text + embeddings (`embeddings.ts`)

**Text construction** (`schema_node_helper.py:141-234`): take the
schema's `index` field list **in declared order**, keep values that are
present and non-blank after `String(v).trim()`, and join with `"\n"` —
no field-name prefixes. Store as the `Data_Bank` property; store the
used field names as `_search_fields_used`. (We always declare a real
`index` list on every Vein schema, so the priority-list fallback path
never applies to us — don't implement it.)

**Model** — jarvis uses `sentence-transformers/all-MiniLM-L6-v2` with
SentenceTransformer defaults: 384 dims, **mean pooling,
L2-normalized**, truncation at **256 tokens** (not 512). The TS
equivalent:

```bash
npm install @huggingface/transformers@4.1.1
```

```ts
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const out = await extractor(text, { pooling: "mean", normalize: true });
// truncation: tokenizer must cap at 256 tokens (model_max_length override)
```

Written as `n.text_embeddings` (LIST OF FLOAT, 384). Cross-arch note:
ONNX runtime via transformers.js is the same graph on ARM and Intel
Macs; assert `dim === 384` at startup and add a golden-vector test
(fixed string → cosine ≥ 0.999 vs. a checked-in Python-produced
vector) to prove parity with jarvis's encoder.

**No Redis queue.** Jarvis's "pending embedding" state is purely
`text_embeddings IS NULL` — no flag property. So we embed in-process
(await inline, or a tiny in-memory async batcher) and remain fully
compatible: a jarvis backfill would simply find nothing NULL. Never
invent a pending-marker property.

**Crash-safe backfill sweep** (replaces the durability the queue gave
jarvis): a crash between the node MERGE and the vector write leaves
`text_embeddings` NULL permanently. At every boot of the graph
backend, sweep and heal:

```cypher
MATCH (n:Domain_vein)
WHERE n.Data_Bank IS NOT NULL AND n.text_embeddings IS NULL
RETURN n.ref_id, n.Data_Bank
```

→ embed in batches → `SET n.text_embeddings` (matching jarvis's own
NULL-scan backfill idiom, `migration.py:857-861`). Idempotent, cheap
when clean.

**Do not implement**: `edge_text_embeddings`; metaphone3.
Per-property `vector_index` embeddings (`{stem}_embeddings`) ARE
needed for the two types that declare `vector_index` (§7 —
`input_q`/`output_q` search); no others.

## 3. Edge writes (`edge-writer.ts`)

Canonical semantics (`bulk_edge_helper.py:243-254`), reproducible in
plain Cypher without APOC since our edge type is static per statement:

```cypher
MATCH (source:Data_Bank {ref_id: $src})
MATCH (target:Data_Bank {ref_id: $tgt})
OPTIONAL MATCH (source)-[:IS_ALIAS]->(sa)
OPTIONAL MATCH (target)-[:IS_ALIAS]->(ta)
WITH COALESCE(sa, source) AS ns, COALESCE(ta, target) AS nt
MERGE (ns)-[r:`<EDGE_TYPE>` {edge_key: $edge_key}]->(nt)
ON CREATE SET r += $on_create
RETURN r.ref_id = $on_create.ref_id AS created
```

- Endpoints matched by **`ref_id` on `:Data_Bank`** (the only unique
  ref_id constraint), never node_key.
- The `IS_ALIAS` COALESCE rewrite is mandatory in shared mode (jarvis
  node-merge parks aliases); harmless in standalone.
- `edge_key = edgeType.toLowerCase()` (no Vein edge schema declares an
  `edge_key` pattern — matching jarvis, where effectively none do).
- Invariant: **one edge per (src, type, tgt)**. On-match is a no-op —
  existing edges are never mutated.
- On-create stamps: `ref_id` (uuid4), `edge_key`, `weight: 1` (int),
  `date_added_to_graph` (epoch ms int). **No `namespace` on edges** —
  scoping goes through the source node.
- Edge soft delete = `SET r.is_muted = true`.
- Edge attributes beyond the stamps are unvalidated passthrough (same
  as jarvis).

Do not implement: temporal/bitemporal edges (`valid_at` etc. — note
those are epoch *seconds as floats* if we ever do), `SCRATCHPAD_EDGE`,
`edge_text` backfills, GDS `SIMILAR`.

## 4. Vein domain registration (`schema-seed.ts`)

Schemas ARE graph nodes: `(:Schema {type})`, global (no namespace),
never labeled `:Node`/`:Data_Bank`, with **attributes flattened as
top-level properties** next to the core keys (`type`, `parent`,
`domain`, `node_key`, `index` (a real LIST), `icon`, `shape`,
`primary_color`, `secondary_color`, `title_key`, `description_key`,
`type_description`, `ref_id`). There is no attributes JSON blob.
Attribute type grammar: `string|boolean|int|float|complex|datetime|list`
with optional `?` prefix. Attribute names must not be `type`, `parent`,
`node_key`, or `index` (reserved), and must not contain `-`.

**A domain is registered by existing**: jarvis derives the domain list
from `MATCH (s:Schema) WHERE s.domain IS NOT NULL RETURN DISTINCT
toLower(s.domain)`. Writing one Vein schema node with `domain: "Vein"`
registers the domain. No root schema type named "Vein" is needed
(precedent: General, Hive, Scratchpad).

Idempotent bootstrap, per boot of the vein graph backend:

1. **`Thing` root** (standalone mode): mirror jarvis's `thing_schema`
   (`default_schemas.py:11-33`) exactly —
   `MERGE (s:Schema {type:"Thing"}) ON CREATE SET s = $thing, s.ref_id = $uuid`.
   MERGE means a jarvis-seeded Thing is left untouched.
2. **Each Vein schema**: guard
   `MATCH (n:Schema) WHERE toLower(n.type) = toLower($t) RETURN n`
   (no `is_deleted` filter — soft-deleted blocks re-create, same as
   jarvis's seeder); on miss
   `CREATE (s:Schema) SET s = $flat, s.ref_id = $uuid`. **Add-only
   reconcile** on hit: set only keys absent on the live node (never
   overwrite `type`/`parent`/`ref_id`; changing `index`/`domain` later
   is an explicit migration, not seeding).
3. **`CHILD_OF`**:
   `MERGE (child)-[:CHILD_OF {ref_id:$e}]->(parent:Schema {type:"Thing"})`
   — plus the redundant `parent` scalar already on the child. Required:
   type resolution, inheritance of Thing's attrs (`description`,
   `weight`, `is_muted`, `unique_source_id`, `image_url`), and edge-
   schema ancestor-walk all depend on it.
4. **Per-type constraint + index**:
   `CREATE CONSTRAINT unique_<type.lower()>_node_key IF NOT EXISTS FOR (n:`Type`) REQUIRE (n.node_key, n.namespace) IS UNIQUE`
   and `CREATE INDEX IF NOT EXISTS FOR (n:`Type`) ON (n.node_key)`.
5. **Global objects** (standalone; `IF NOT EXISTS` makes them safe in
   shared):
   `CREATE CONSTRAINT unique_node_key_global IF NOT EXISTS FOR (n:Node) REQUIRE (n.node_key, n.namespace) IS UNIQUE`;
   `CREATE CONSTRAINT IF NOT EXISTS FOR (n:Data_Bank) REQUIRE n.ref_id IS UNIQUE`.
6. **Edge schemas** — one relationship per edge in the label registry
   (generic-storage.md), between the two `:Schema` endpoint nodes:
   `MERGE (source)-[r:EDGE_TYPE]->(target) SET r = $props` with
   `props = {ref_id}`. Edge type must match `^[A-Z][A-Z0-9_]*$`. Both
   endpoints must exist first (a miss silently no-ops — assert the
   MERGE returned a row). For `ACCESSED`, the declared target is
   `Thing` (provenance can point at any node; the ancestor walk makes
   a Thing→Thing or VeinToolCall→Thing declaration cover all types).
   `PUBLISHED_BY` targets `Person` — in standalone mode seed jarvis's
   `Person` schema too, or (simpler) drop the edge and keep publisher
   as a property until shared mode.
7. **Vector index** (NOT auto-created by jarvis at startup — the one
   piece a new domain must create itself):
   ```cypher
   CREATE VECTOR INDEX `domain_vein_vector_index` IF NOT EXISTS
   FOR (n:`Domain_vein`) ON n.text_embeddings
   OPTIONS { indexConfig: { `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine' } }
   ```
   Also `text_embeddings_vector_index` on `:Data_Bank` (same options)
   in standalone mode.
8. **Fulltext**: jarvis (re)builds `domain_vein_attribute_index` (+
   `_v2` english-analyzer sibling, which its queries actually use) on
   its next boot from the schema `index` fields. For standalone search
   parity, create the `_v2` form ourselves:
   `CREATE FULLTEXT INDEX domain_vein_attribute_index_v2 IF NOT EXISTS
   FOR (n:`Domain_vein`) ON EACH [<sorted searchable attrs> + node_key]
   OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}`.
9. **Migration ledger stamp** so jarvis's runner sees our seeding as
   done work, not conflict:
   `MERGE (m:Migration {migration_id:"vein_domain_seed_v1"}) SET m.executed_at = timestamp()`
   (plus `CREATE CONSTRAINT migration_id_unique IF NOT EXISTS ...` in
   standalone).

Footguns (both learned from jarvis's own migration 115):
- Never let "Vein" land in `About.hidden_domains`, and never parent a
  Vein type under a hidden type — either silently withholds the
  `Domain_vein` label and drops every node out of all domain indexes.
- The `index` list on each schema directly controls what enters
  `Data_Bank` + `text_embeddings`. Keep large payloads (event-log
  pointers, raw output snapshots) OUT of `index`.

## 5. Vein schema `index`/`node_key` choices (`vein-schemas.ts`)

All `parent: "Thing"`, `domain: "Vein"`. Node types and edges per the
label registry in generic-storage.md. Keys:

| Type                  | node_key spec                                   | index (→ search + embedding text)        |
| --------------------- | ----------------------------------------------- | ---------------------------------------- |
| `VeinWorkflow`        | `veinworkflow-name`                             | `[name, description]`                    |
| `VeinWorkflowVersion` | `veinworkflowversion-name-content_hash`         | `[name, description]`                    |
| `VeinStep`            | `veinstep-step_type`                            | `[step_type, description]`               |
| `VeinStepVersion`     | `veinstepversion-step_type-content_hash`        | `[step_type, description]`               |
| `VeinRun`             | `veinrun-run_id`                                | `[workflow_name, status, summary]`       |
| `VeinAgentSession`    | `veinagentsession-run_id-path`                  | `[prompt_preview, result_preview]`       |
| `VeinToolCall`        | `veintoolcall-run_id-path-seq`                  | `[tool_name, input_preview]`             |
| `VeinChat`            | `veinchat-chat_id`                              | `[title, summary]`                       |
| `VeinTurn`            | `veinturn-chat_id-turn`                         | `[user_text_preview]`                    |

Rules baked in: every node_key token is a required (non-`?`) attribute;
`*_preview` fields are truncated (~500 chars) copies specifically so
search/embeddings stay light while full payloads stay in the run log
(pointer property `log_ref`); no raw transcripts, tool I/O, or step
source in any `index` list. Full step/workflow source lives as a
non-indexed node property (like Concept's `docs` predecessor
`documentation` — deliberately invisible to retrieval).

Optionally spread jarvis's `USAGE_ATTRIBUTES` shape
(`usage_count: "?int"`, `usage_count_30d: "?int"`) onto `VeinWorkflow`/
`VeinStep` — jarvis never writes these, external writers do, so vein
owns updating them; jarvis's `?sort=usage` and search tiebreak read
them for free.

## 6. Validation — no invalid/unexpected node reaches the graph

Every write goes through the writer; the writer validates BEFORE
building any Cypher. Mirror of jarvis's two-stage gate
(`assert_is_valid_schema_of_type` `schema_validation.py:58-180` +
`validate_by_schema` `:184-278`), tightened where jarvis is loose —
strictness is safe in one direction only: anything vein accepts must
also pass jarvis's validators, never the reverse.

Write-time gate (reject with a typed error, nothing written):

1. **Closed type set** — `type` must be one of the nine registered
   Vein schemas. No dynamic types, no scratchpad fallback, no
   `create_schema_if_missing`. (Jarvis resolves types case-
   insensitively against `db.labels()`; we don't — exact match on our
   registry.)
2. **Required attributes present** — every non-`?` attribute in the
   schema must be present and non-null.
3. **Unknown attributes rejected** — every key in the payload must be
   declared in the schema (jarvis rejects these too, `:161-178`).
   This is what makes "unexpected" nodes impossible, not just invalid
   ones.
4. **Type checks per the grammar** — `string|boolean|int|float|
   datetime|list` (+`?`). Stricter than jarvis on two Python quirks we
   do NOT replicate: booleans are not ints, and int-where-float is
   coerced explicitly rather than passed through. `datetime` accepts
   ISO string or epoch number and always normalizes to epoch seconds
   (int) before write.
5. **node_key integrity** — all node_key tokens are required attrs
   (enforced at schema-author time, mirroring `valid_node_key`
   `:311-338`: token 0 = type.lower(), no optional/reserved fields),
   so a node can never be written with an incomplete key.
6. **Edge validation, stricter than jarvis** — edge type must be in
   the label-registry vocabulary (closed set — no equivalent of
   jarvis's 63-type `EDGE_TYPES` bypass), and (source label, edge,
   target label) must match a registry row (`ACCESSED` alone accepts
   any target). Both endpoints must resolve by `ref_id` — a MATCH
   miss is an error, not a silent zero-row no-op.

DB-level backstop (catches bugs in the writer itself and concurrent
races): the `(node_key, namespace)` per-type uniqueness constraints,
global `:Node` key constraint, and `:Data_Bank` ref_id constraint from
§4. Honest limit: Neo4j property *type/existence* constraints are
Enterprise-only, so attribute-level enforcement lives in the app layer
— same as jarvis. Anyone with raw bolt credentials can bypass both
vein's and jarvis's validation equally; the guarantee is "nothing
invalid gets in **through vein**", and the conformance suite's
graph-lint case (§8) exists to detect out-of-band junk: scan
`Domain_vein` nodes and re-run the validator against stored state,
assert zero violations.

## 7. Read surface — scoped by the lab step tools

The functional contract for "vein without jarvis" is NOT jarvis's whole
API — it is exactly what the `jarvis/*` steps in
`mcp/src/lab/jarvis/steps/` touch. Those steps keep their input/output
shapes; their backing moves from HTTP calls to the TS graph layer
(vein-native `graph/*` step twins, so workflows swap by step type).
Everything in jarvis NOT reached by these tools stays out of scope.

| Step(s)                                          | jarvis endpoint                          | TS implementation needed |
| ------------------------------------------------ | ---------------------------------------- | ------------------------ |
| `create-node`, `create-triplet`, `create-batch-triplet` | `POST /v2/nodes`, `POST /v2/edges` | already §1/§3            |
| `edit-node`                                      | `POST /v2/nodes/:ref_id`                 | §1 upsert (reprocess semantics) |
| `register-namespace`                             | `POST/GET /namespace`                    | trivial: idempotent namespace registry (mirror jarvis's storage — verify whether it's a node or derived from distinct `n.namespace` before building) |
| `get-ontology`, `get-ontology-type`              | `GET /v2/schema[/{type}]`                | list `:Schema` nodes + inherited attrs via `CHILD_OF` walk (read side of §4) |
| `graph-get`, `graph-get-batched`                 | `GET /v2/nodes/:ref_id` + `/connection-counts` | fetch by ref_id; response envelope = properties minus `GENERIC_NODE_PROPERTIES`; `{EDGE_TYPE: count}` map |
| `graph-neighbors`                                | `GET /v2/nodes/:ref_id?expand=edges&...` | bounded neighbor traversal: `edge_type`/`node_type`/`exclude_node_type` filters, `limit` with `sort_by=importance`, per-neighbor edge counts |
| `graph-search`                                   | `GET /v2/nodes?q&input_q&output_q&type&domains&namespace&include_edge_counts` | **hybrid search — see below** |

### Hybrid search (the one algorithm to port faithfully)

`graph-search` depends on jarvis's fused retrieval
(`node_service_v2.py`); the exact recipe:

- **Keyword retriever**: fulltext query against the `_v2`
  english-analyzer index (`data_bank_attribute_index_v2`, or
  `domain_<d>_attribute_index_v2` when domain-filtered).
- **Semantic retriever**: embed the query with the same MiniLM encoder
  (no prefix/instruction), `db.index.vector.queryNodes(<index>, k, emb)`
  against `text_embeddings_vector_index` / `domain_*_vector_index`;
  `k = 50`, similarity floor `0.4` (Neo4j cosine score = `(1+cos)/2`).
- **Field-scoped retrievers** (`input_q` / `output_q`): vector query
  against `{label.lower()}_{stem}_vector_index` over
  `{stem}_embeddings`.
- **Fusion**: weighted Reciprocal Rank Fusion, `RRF_K = 60`; weights
  fulltext `1.15`, semantic `1.0`, field-scoped `1.2`.
- **Tiebreak**: bucket normalized scores by epsilon, then sort within a
  bucket by `usage_count_30d` desc → `usage_count` desc → best rank →
  `ref_id`.
- Filters: `type` (label list), `domains` (validated against the
  distinct-domain registry), `namespace`, soft-delete/mute exclusion.

Consequence for §2/§5 (supersedes §2's "do not implement
`vector_index`"): `input_q`/`output_q` only work on types whose schema
declares `vector_index` — so `VeinWorkflowVersion` and `VeinStep`
declare `vector_index: ["input_schema", "output_schema"]`, we write
`input_embeddings`/`output_embeddings` (embedding the rendered form
`"Input:\n{text}"` / `"Output:\n{text}"`, per `schema_embedding.py:30-41`),
and create the per-label vector indexes
(`veinworkflowversion_input_vector_index`, …, 384/cosine) at seed time.

### What this scoping proves unnecessary

Radar, review/merge queues, scratchpad promotion, legal/matter
endpoints, temporal facts API, Hive endpoints, stats/decay/GDS,
Elasticsearch, whinx alias ingest, the ontology-agent — none are
reachable from the lab steps. They are jarvis application features,
not graph-dialect obligations; jarvis can keep providing them on top
of the shared DB whenever it's mounted.

## 8. Explicitly out of scope (jarvis tolerates absence)

Redis embedding queue (we embed in-process) · scratchpad fallback ·
`smart_reinsert` / `force_delete` flows · temporal edges ·
Elasticsearch (optional in jarvis, effectively unused for node search) ·
metaphone3 · GDS algorithms · radar/linker (only triggered by radar) ·
alias/entity ingest (we only *honor* `IS_ALIAS` on edge writes) ·
`edge_text_embeddings` · jarvis's HTTP API surface · the legacy
non-`_v2` fulltext variants.

## 9. Conformance additions

Extend the storage-conformance suite (generic-storage.md §6) with a
jarvis-dialect suite run against a Neo4j test container:

- node write → assert exact label set, stamp types (ms-int vs s-int),
  `Data_Bank` join, `_search_fields_used`, node_key for tricky inputs
  (spaces, punctuation, >200 chars → hash form, case-insensitive
  property lookup).
- duplicate write → no-op preserving ref_id; upsert preserving the
  five protected properties.
- edge write → single edge per (src,type,tgt), stamps, alias rewrite.
- embedding golden-vector parity test (cosine vs. Python-produced
  fixture ≥ 0.999).
- bootstrap idempotence: run seeding twice, diff graph state = empty;
  run against a jarvis-seeded dump, assert no jarvis-owned property
  changed.
- validation gate (§6): missing required attr, unknown attr, wrong
  attr type, unregistered node/edge type, edge with unresolvable
  endpoint — each rejected with nothing written (assert graph
  unchanged). Plus the graph-lint case: re-validate all stored
  `Domain_vein` nodes against the schemas, zero violations.
- search parity (§7): seed a small corpus, run each lab step's
  scenario against the TS layer, and assert result-set/rank agreement
  with jarvis on the same dump (fixture-based — record jarvis's
  responses once, don't require a live jarvis in CI).
