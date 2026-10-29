import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";
import { WorkspaceManager, buildRegistry, orchestratorWorkflowYaml, ORCHESTRATOR_WORKFLOW } from "strut";
import { CONCEPT_WRITES, GRAPH_READS, GRAPH_WRITES, META_TOOLS, ORCHESTRATOR_SEED, seedOrchestratorSteps } from "./seed.js";

/** The seven writes hive's graph and Concept proposals become (strut
 *  plans/proposals.md §6b) plus the lab's own `concept/create` — the
 *  committed list, so a write slipping out of the gate, or a ninth slipping
 *  in, fails here. */
const GATED = [
  "graph/create-node",
  "graph/edit-node",
  "graph/create-triplet",
  "graph/create-batch-triplet",
  "graph/move-node",
  "graph/delete-edge",
  "graph/delete-node",
  "concept/create",
];

/** What strut's own template gates, ahead of the lab's delta: `app/grant`,
 *  a person's yes to an app talking to this assistant (strut
 *  src/apps/grants.ts). */
const STRUT_GATED = ["app/grant"];

// The mcp dir, like the real lab-workspace: a seeded step's bare `strut`
// import resolves via mcp/node_modules from there.
const MCP_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

interface Params {
  tools: string[];
  approve: string[];
  system: string;
}

async function merged(): Promise<{ name: string; params: Params; steps: Array<{ id: string; type: string; config?: Record<string, unknown> }> }> {
  return yaml.load(await orchestratorWorkflowYaml(ORCHESTRATOR_SEED)) as any;
}

describe("orchestrator seed (the lab's delta on strut's template)", () => {
  it("gates exactly the seven graph writes and concept/create (beside strut's own), every one also granted", async () => {
    const { name, params } = await merged();
    assert.equal(name, ORCHESTRATOR_WORKFLOW);
    assert.deepEqual([...params.approve].sort(), [...STRUT_GATED, ...GATED].sort());
    assert.deepEqual([...GRAPH_WRITES, ...CONCEPT_WRITES].sort(), [...GATED].sort());
    assert.deepEqual(ORCHESTRATOR_SEED.approve, [...GRAPH_WRITES, ...CONCEPT_WRITES]);
    assert.deepEqual([...CONCEPT_WRITES], ["concept/create"]);
    // The agent step warns and gates nothing for an `approve` type outside
    // agentTools: each gated type must be a grant too.
    for (const t of params.approve) assert.ok(params.tools.includes(t), `approve "${t}" is not granted`);
    assert.deepEqual(new Set(params.tools).size, params.tools.length, "a tool granted twice");
  });

  it("grants the reads, the writes, concept/create and the meta/* read + run tools by name, never a glob", async () => {
    const { params } = await merged();
    for (const t of [...GRAPH_READS, ...GRAPH_WRITES, ...CONCEPT_WRITES, ...META_TOOLS]) assert.ok(params.tools.includes(t), `missing grant ${t}`);
    // No glob: `graph/*` would hand out create-schema and edit-edge ungated.
    assert.deepEqual(params.tools.filter((t) => t.includes("*")), []);
    // Strut's own grants survive the merge.
    for (const t of ["job/start", "job/continue", "job/list", "app/grant"]) assert.ok(params.tools.includes(t), `strut's ${t} dropped`);
    // What the orchestrator must NOT have: the job agent's shell-bearing
    // surfaces, authoring, and strut-only ungated writes.
    for (const t of ["graph/create-schema", "graph/edit-edge", "meta/publish-workflow", "meta/create-step", "pod/*"]) {
      assert.ok(!params.tools.includes(t), `${t} granted`);
    }
  });

  it("every granted type is a step of the lab's registry: strut's, plus the one this seeder publishes", async () => {
    const { params } = await merged();
    const root = await mkdtemp(join(MCP_DIR, ".lab-orchestrator-seed-"));
    try {
      const ws = new WorkspaceManager(root);
      await seedOrchestratorSteps(ws);
      assert.deepEqual((await ws.listSteps()).map((s) => s.type), [...CONCEPT_WRITES]);
      const { registry } = await buildRegistry(await ws.materializeCustomSteps());
      assert.deepEqual(params.tools.filter((t) => !registry[t]), []);
      // Strut's own registry does not have it: it is the lab's.
      const { registry: struts } = await buildRegistry();
      assert.deepEqual([...CONCEPT_WRITES].filter((t) => struts[t]), []);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("the agent step reads the grants and the gate from params", async () => {
    const { steps } = await merged();
    const turn = steps.find((s) => s.type === "agent");
    assert.ok(turn, "no agent step");
    assert.equal(turn.config?.["agentTools"], "{{ params.tools }}");
    assert.equal(turn.config?.["approve"], "{{ params.approve }}");
  });

  it("appends the deployment's section after strut's prompt", async () => {
    const { params } = await merged();
    const ours = params.system.indexOf("@<slug>");
    const struts = params.system.indexOf("job/start");
    assert.ok(struts >= 0 && ours > struts, "the lab's section must follow strut's");
    // Strut's prompt sends a workflow build to a builder job; the lab's kinds (the `Job` children) include strut's `Workflow Build`.
    const builder = params.system.indexOf("`workflow: builder`");
    assert.ok(builder >= 0 && builder < ours, "strut's builder-job sentence must survive the merge, before the lab's section");
    // What the section tells the agent: proposed, never made; the Concept
    // shape (docs, the PARENT_OF tree, the HiveWorkspace anchor) and that a
    // new one is ONE concept/create proposal; the lookups.
    for (const needle of ["[proposal-result]", "PROPOSED", "`docs`", "PARENT_OF", "HiveWorkspace", "concept/create", "`explore`", "`logs`", "meta/run-workflow"]) {
      assert.ok(params.system.includes(needle), `system lacks ${needle}`);
    }
    assert.ok(params.system.endsWith("\n"));
  });

  it("a new Concept is one concept/create proposal, never the batch-triplet recipe; a change stays graph/edit-node", async () => {
    const { params } = await merged();
    const para = params.system.slice(params.system.indexOf("A Concept is a node of type Concept"), params.system.indexOf("are the LOOKUPS"));
    assert.ok(para.length > 0 && para.length < params.system.length, "no Concept paragraph");
    for (const needle of [
      "A NEW Concept is ONE\nproposal, concept/create",
      // Its input: name, docs, description, the parent by ref_id or exact name, the kind.
      "its name, docs, description, its parent",
      "a ref_id from graph/graph-search, or its exact name",
      "preference, best_practice, gotcha, process or general",
      // The step does the placing and the anchoring: the model carries no recipe.
      "places it under its parent and anchors it to the workspace itself",
      "never graph/create-node or a triplet for a new Concept",
      "Changing a\nConcept is graph/edit-node on its ref_id",
      "`docs` whole as it should read",
    ]) {
      assert.ok(para.includes(needle), `Concept paragraph lacks ${JSON.stringify(needle)}`);
    }
    // The old recipe is gone: no inline Concept in a batch triplet, no PROCESS edge the model writes itself.
    assert.doesNotMatch(params.system, /Concept inline|the same node_data in both|-PROCESS-> it|node type HiveWorkspace/);
    assert.doesNotMatch(para, /create-batch-triplet|create-triplet/);
    // concept/create is in the PROPOSED list with the seven.
    assert.match(params.system, /delete-node\nand concept\/create each record a proposal/);
  });

  it("the lookups orient; what must be found out or made is a job's, briefed by kind", async () => {
    const { params } = await merged();
    const section = params.system.slice(params.system.indexOf("are the LOOKUPS"));
    assert.ok(section.length < params.system.length, "no LOOKUPS paragraph");
    // Both lookups with their real shapes, logs on a peer too.
    for (const needle of [
      "{ prompt, repos? }",
      "{ answer, sources,\nconfidence }",
      "`logs` ({ prompt })",
      'strut/run-workflow { peer, workflow: "logs", input: { prompt } }',
      // A lookup's answer is the reply only when it is whole; explore's confidence is the hint.
      "only when it is the whole answer",
      "`high` is an answer, anything lower is a brief",
      "goes into a job's prompt",
      // The kinds are the Job Concept's children, rendered or read.
      "the `Job` Concept's children",
      'name: "Job", children: "PARENT_OF" }',
      "names the kind",
      "found out or made\nis a job's, whatever its size",
    ]) {
      assert.ok(section.includes(needle), `section lacks ${JSON.stringify(needle)}`);
    }
    // A posture, not a recipe: the old pull toward exploring an open question
    // inside the turn is gone, and no count of lookups or steps creeps in.
    assert.doesNotMatch(params.system, /node by node|when the question\s+is open/);
    assert.doesNotMatch(section, /\b(once|twice|(one|two|three|a few|\d+) (lookups?|explores?|steps?|calls?|runs?))\b/i);
  });
});
