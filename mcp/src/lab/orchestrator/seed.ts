import type { OrchestratorSeedOptions } from "strut";

/**
 * orchestrator — one turn of hive's Jamie canvas chat as a strut workflow
 * (strut `plans/proposals.md` §9). The WORKFLOW is strut's —
 * `src/workflows/orchestrator.yaml`, seeded at boot by `createStrut` exactly
 * as `job` is (content-hash reconciled; a changed template wins, an
 * unchanged one leaves a workspace-side edit active). This module ships
 * only the lab's DELTA on it, passed as `createStrut({ orchestrator:
 * ORCHESTRATOR_SEED })` beside `job: JOB_SEED`: grants appended to
 * `params.tools`, the gated ones appended to `params.approve`, a section
 * appended to the agent's system prompt after strut's. Exactly one seeder
 * per name.
 *
 *  - `tools`: strut's `graph/*` steps on this swarm's Neo4j — the reads
 *    (search, get, get-batched, neighbors, the two ontology reads) and the
 *    seven writes hive's `propose_*` tools were (plans/proposals.md §6b) —
 *    plus the `meta/*` read + run tools, so a turn runs the lab's seeded
 *    workflows (`explore`) the way the job agent does. Strut's own list
 *    (memory/*, job/start, continue, list, get, read) is already in the
 *    YAML. Granted BY NAME, never `graph/*`: the glob would hand out
 *    `graph/create-schema` and `graph/edit-edge` ungated.
 *  - `approve`: exactly the seven writes. A call to one does not run the
 *    step: strut records a PROPOSAL, hive draws the card, and the step runs
 *    in code when the person accepts (`POST /proposals/:id/decide`). The
 *    reads stay immediate.
 *  - `system`: what only this deployment knows — which Hive workspace this
 *    strut belongs to (`@<slug>` on the message's first line, as the job
 *    seed says), what a Concept is on this graph (`docs` carries the body;
 *    jarvis's `documentation` is deprecated and unindexed), that a NEW
 *    Concept is ONE batch-triplet proposal placing it under its parent
 *    (`PARENT_OF`) and anchoring it to the workspace node (`HiveWorkspace
 *    -PROCESS-> Concept`, the edge hive's `handleApproval` wrote and
 *    `concept-anchor.ts` writes for seeded pages — nothing writes it for a
 *    proposal but the proposal itself), and that graph writes are
 *    proposed, never made: the decision reaches the agent on a later turn
 *    as a `[proposal-result]` line (strut's `proposalResultLines`).
 */

/** The graph reads: immediate, on this swarm's graph. */
export const GRAPH_READS = [
  "graph/graph-search",
  "graph/graph-get",
  "graph/graph-get-batched",
  "graph/graph-neighbors",
  "graph/get-ontology",
  "graph/get-ontology-type",
] as const;

/** The seven writes hive's graph and Concept proposals become (strut
 *  plans/proposals.md §6b) — every one gated. */
export const GRAPH_WRITES = [
  "graph/create-node",
  "graph/edit-node",
  "graph/create-triplet",
  "graph/create-batch-triplet",
  "graph/move-node",
  "graph/delete-edge",
  "graph/delete-node",
] as const;

/** The meta/* read + run tools: run a seeded workflow (`explore`) inside the turn. */
export const META_TOOLS = ["meta/list-workflows", "meta/get-workflow", "meta/run-workflow"] as const;

export const ORCHESTRATOR_SEED: OrchestratorSeedOptions = {
  tools: [...GRAPH_READS, ...GRAPH_WRITES, ...META_TOOLS],
  approve: [...GRAPH_WRITES],
  system: `This strut belongs to one Hive workspace: your message's first line names
it, \`@<slug>\`. Its knowledge graph is the graph/* tools': read it freely
(graph/graph-search, graph/graph-get, graph/graph-neighbors, the ontology
reads) and search before you write, so a change goes to the node that
exists, by ref_id, never to a duplicate.

Every graph write is PROPOSED, never made: graph/create-node, edit-node,
create-triplet, create-batch-triplet, move-node, delete-edge and
delete-node each record a proposal the person sees as a card and answer
{ proposal, status: "pending" }. Nothing changes until they accept, so say
in \`text\` what you proposed, never that it is done. Their decision
reaches you on a later turn as a \`[proposal-result]\` line before the
message — applied, rejected (with their reason) or failed — and that is
when to build on it or try another way. Give each proposal a rationale
the person can judge it by.

A Concept is a node of type Concept: \`name\` (its key), a one-line
\`description\`, and its documentation in \`docs\` (markdown; never the
deprecated \`documentation\`). Concepts form a tree by PARENT_OF edges, and
every one hangs from this workspace's HiveWorkspace node by a PROCESS
edge. A new Concept is ONE proposal, graph/create-batch-triplet, with the
Concept inline (node_type "Concept", the same node_data in both) as the
target of two triplets: its parent Concept, by ref_id, -PARENT_OF-> it,
and the HiveWorkspace node (graph/graph-search, node type HiveWorkspace)
-PROCESS-> it. Changing a Concept is graph/edit-node on its ref_id with
the changed fields in node_data, \`docs\` whole as it should read.

The workflows this strut has are yours to run inside the turn:
meta/list-workflows lists them, meta/get-workflow shows one's input,
meta/run-workflow runs it and returns its output. \`explore\` answers a
question about this workspace's code or graph ({ prompt, repos? }) with
its sources; prefer it to reading the graph node by node when the question
is open. \`logs\` answers a question about this swarm's logs ({ prompt }):
its CloudWatch groups, its pods, its app's requests; for another
workspace's swarm, strut/run-workflow { peer, workflow: "logs", input: {
prompt } }. Work that needs files, the web or many steps is a job's.`,
};
