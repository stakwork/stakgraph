import { describe, it } from "node:test";
import assert from "node:assert/strict";
import yaml from "js-yaml";
import { buildRegistry, orchestratorWorkflowYaml, ORCHESTRATOR_WORKFLOW } from "strut";
import { GRAPH_READS, GRAPH_WRITES, META_TOOLS, ORCHESTRATOR_SEED } from "./seed.js";

/** The seven writes hive's graph and Concept proposals become (strut
 *  plans/proposals.md §6b) — the committed list, so a write slipping out of
 *  the gate, or an eighth slipping in, fails here. */
const GATED = [
  "graph/create-node",
  "graph/edit-node",
  "graph/create-triplet",
  "graph/create-batch-triplet",
  "graph/move-node",
  "graph/delete-edge",
  "graph/delete-node",
];

interface Params {
  tools: string[];
  approve: string[];
  system: string;
}

async function merged(): Promise<{ name: string; params: Params; steps: Array<{ id: string; type: string; config?: Record<string, unknown> }> }> {
  return yaml.load(await orchestratorWorkflowYaml(ORCHESTRATOR_SEED)) as any;
}

describe("orchestrator seed (the lab's delta on strut's template)", () => {
  it("gates exactly the seven graph writes, every one also granted", async () => {
    const { name, params } = await merged();
    assert.equal(name, ORCHESTRATOR_WORKFLOW);
    assert.deepEqual([...params.approve].sort(), [...GATED].sort());
    assert.deepEqual([...GRAPH_WRITES].sort(), [...GATED].sort());
    assert.deepEqual(ORCHESTRATOR_SEED.approve, [...GRAPH_WRITES]);
    // The agent step warns and gates nothing for an `approve` type outside
    // agentTools: each gated type must be a grant too.
    for (const t of params.approve) assert.ok(params.tools.includes(t), `approve "${t}" is not granted`);
    assert.deepEqual(new Set(params.tools).size, params.tools.length, "a tool granted twice");
  });

  it("grants the reads, the writes and the meta/* read + run tools by name, never a glob", async () => {
    const { params } = await merged();
    for (const t of [...GRAPH_READS, ...GRAPH_WRITES, ...META_TOOLS]) assert.ok(params.tools.includes(t), `missing grant ${t}`);
    // No glob: `graph/*` would hand out create-schema and edit-edge ungated.
    assert.deepEqual(params.tools.filter((t) => t.includes("*")), []);
    // Strut's own grants survive the merge.
    for (const t of ["memory/recall", "job/start", "job/continue", "job/list"]) assert.ok(params.tools.includes(t), `strut's ${t} dropped`);
    // What the orchestrator must NOT have: the job agent's shell-bearing
    // surfaces, authoring, and strut-only ungated writes.
    for (const t of ["graph/create-schema", "graph/edit-edge", "meta/publish-workflow", "meta/create-step", "pod/*"]) {
      assert.ok(!params.tools.includes(t), `${t} granted`);
    }
  });

  it("every granted type is a step of the lab's registry", async () => {
    const { params } = await merged();
    const { registry } = await buildRegistry();
    assert.deepEqual(params.tools.filter((t) => !registry[t]), []);
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
    // What the section tells the agent: proposed, never made; the Concept
    // shape (docs, PARENT_OF, the PROCESS anchor in one batch triplet); the
    // lookups.
    for (const needle of ["[proposal-result]", "PROPOSED", "`docs`", "PARENT_OF", "PROCESS", "HiveWorkspace", "graph/create-batch-triplet", "`explore`", "`logs`", "meta/run-workflow"]) {
      assert.ok(params.system.includes(needle), `system lacks ${needle}`);
    }
    assert.ok(params.system.endsWith("\n"));
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
