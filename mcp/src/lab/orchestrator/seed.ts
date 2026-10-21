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
 *    plus the `meta/*` read + run tools, so a turn runs the lab's lookups
 *    (`explore`, `logs`) the way the job agent does. Strut's own list
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
 *    as a `[proposal-result]` line (strut's `proposalResultLines`). And
 *    which workflows are the turn's LOOKUPS here (`explore`, `logs`, the
 *    latter on a peer too): strut's prompt draws the line — the turn
 *    answers from what is known, asks for a decision, orients with a
 *    lookup; what must be found out or made is a job's — and this section
 *    says when a lookup's answer is the reply (the whole answer; explore's
 *    `confidence: high`) and that otherwise it goes into the job's brief,
 *    which names a kind of work: a child of the `Job` Concept (rendered at
 *    the end of the prompt by strut's `context/concept` step where it has
 *    one; graph/graph-get with `children: "PARENT_OF"` lists them where
 *    not), whose page the turn may open first. A posture, not a recipe: no
 *    count of lookups or steps.
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

/** The meta/* read + run tools: run a lookup (`explore`, `logs`) inside the turn. */
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

\`explore\` and \`logs\` are the LOOKUPS this deployment grants the turn:
workflows that return an answer, run with meta/run-workflow
(meta/get-workflow shows one's input). \`explore\` ({ prompt, repos? })
reads this workspace's code or graph and answers { answer, sources,
confidence }; \`logs\` ({ prompt }) reads this swarm's logs, and
strut/run-workflow { peer, workflow: "logs", input: { prompt } } another
workspace's. Use them to orient: which workspace holds the code, what
something is called, whether it exists. A lookup's answer is your reply
only when it is the whole answer — explore's \`confidence\` is the hint:
\`high\` is an answer, anything lower is a brief — and otherwise what it
found goes into a job's prompt, for the job to start from. The kinds of
work a job can do here are the \`Job\` Concept's children, listed at the
end of this prompt or, when they are not, by graph/graph-get { node_type:
"Concept", name: "Job", children: "PARENT_OF" }; you may open a kind's
page the same way before writing the brief, so the brief names the kind
and gives what its page asks for. Anything that must be found out or made
is a job's, whatever its size.`,
};
