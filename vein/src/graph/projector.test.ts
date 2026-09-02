import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import type { RunEvent } from "../core.js";
import { MemoryRunStore } from "../store.js";
import { MemoryChatStore } from "../chat-store.js";
import { openGraphBackend, type GraphBackend } from "./backend.js";
import { seedVeinDomain } from "./schema-seed.js";
import { testGraphConfig, wipeGraph } from "./test-util.js";
import { Neo4jWorkspaceStore } from "./workspace-store.js";
import { messageText, preview, projectAll, projectChats, projectRunEvents, projectRuns, spawnedRunIds } from "./projector.js";

const cfg = testGraphConfig();
let backend: GraphBackend;

const WF = "harvey-deliver";
const RUN = "1788307097627";
const T0 = Date.parse("2026-09-01T20:00:00Z");
const ts = (i: number) => new Date(T0 + i * 1000).toISOString();
const ev = (i: number, type: RunEvent["type"], path: string, extra: Partial<RunEvent> = {}): RunEvent => ({
  ts: ts(i),
  runId: RUN,
  path,
  type,
  ...extra,
});

/** A run with one agent step that made two tool calls, then finished. */
function sampleEvents(workflowHash: string): RunEvent[] {
  return [
    ev(0, "run.start", WF, { input: { q: "deliver" }, workflowHash, params: { model: "m" } }),
    ev(1, "step.start", `${WF}/plan`, { stepType: "agent", input: { prompt: "Plan the delivery", model: "claude" } }),
    ev(2, "step.start", `${WF}/plan/001-search`, { stepType: "tool:graph/graph-search", input: { query: "docs" } }),
    ev(3, "step.end", `${WF}/plan/001-search`, { stepType: "tool:graph/graph-search", output: { hits: 3 }, durationMs: 800 }),
    ev(4, "step.start", `${WF}/plan/002-read`, { stepType: "tool:read", input: { id: "x" } }),
    ev(5, "step.error", `${WF}/plan/002-read`, { stepType: "tool:read", error: { message: "not found" }, durationMs: 10 }),
    ev(6, "step.end", `${WF}/plan`, { stepType: "agent", output: "Plan: ship it", durationMs: 5000 }),
    ev(7, "step.start", `${WF}/ship`, { stepType: "log" }),
    ev(8, "step.end", `${WF}/ship`, { stepType: "log", output: "shipped" }),
    ev(9, "run.end", WF, { output: { delivered: 60 } }),
  ];
}

describe("projectRunEvents (pure)", () => {
  it("derives run, session, and tool-call nodes with previews and log refs", () => {
    const p = projectRunEvents(WF, RUN, sampleEvents("abc123def456"), null)!;
    assert.equal(p.run.type, "VeinRun");
    assert.equal(p.run.data["status"], "success");
    assert.equal(p.run.data["workflow_hash"], "abc123def456");
    assert.equal(p.run.data["params_json"], '{"model":"m"}');
    assert.equal(p.run.data["log_ref"], `${WF}/${RUN}`);
    assert.equal(p.run.data["input_preview"], '{"q":"deliver"}');
    assert.equal(p.workflowHash, "abc123def456");

    assert.equal(p.sessions.length, 1);
    assert.equal(p.sessions[0]!.data["prompt_preview"], "Plan the delivery");
    assert.equal(p.sessions[0]!.data["result_preview"], "Plan: ship it");
    assert.equal(p.sessions[0]!.data["model"], "claude");
    assert.equal(p.sessions[0]!.data["duration_ms"], 5000);

    assert.deepEqual(
      p.toolCalls.map((t) => [t.node.data["seq"], t.node.data["tool_name"], t.node.data["error_message"], t.sessionPath]),
      [
        [1, "graph/graph-search", undefined, `${WF}/plan`],
        [2, "read", "not found", `${WF}/plan`],
      ],
    );
  });

  it("uses the summary when present, and marks a finalize-less log stale", () => {
    const events = sampleEvents("h").slice(0, 3);
    assert.equal(projectRunEvents(WF, RUN, events, null)!.run.data["status"], "stale");
    const withSummary = projectRunEvents(WF, RUN, events, {
      runId: RUN,
      workflow: WF,
      startedAt: ts(0),
      finishedAt: ts(9),
      durationMs: 9000,
      status: "error",
      input: {},
      error: { message: "boom" },
    })!;
    assert.equal(withSummary.run.data["status"], "error");
    assert.equal(withSummary.run.data["error_message"], "boom");
    assert.equal(withSummary.run.data["summary"], "error: boom");
    assert.equal(projectRunEvents(WF, RUN, [], null), null);
  });

  it("preview caps length; messageText + spawnedRunIds read transcripts", () => {
    assert.equal(preview("x".repeat(600))!.length, 500);
    assert.equal(preview({ a: 1 }), '{"a":1}');
    assert.equal(preview(undefined), undefined);
    assert.equal(messageText([{ type: "text", text: "hi" }, { type: "text", text: "there" }]), "hi\nthere");
    assert.equal(messageText("plain"), "plain");
    assert.deepEqual(
      spawnedRunIds([
        { role: "user", content: "go" },
        { role: "tool", content: [{ type: "tool-result", toolName: "run_workflow", output: { type: "json", value: { runId: "1", status: "success" } } }] },
        { role: "tool", content: [{ type: "tool-result", toolName: "list_runs", output: { runs: [{ runId: "99" }] } }] },
      ]),
      ["1"],
    );
  });
});

describe("projector (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let store: MemoryRunStore;
  let ws: Neo4jWorkspaceStore;

  before(async () => {
    backend = await openGraphBackend(cfg!, { embeddings: false, skipBoot: true });
  });
  after(async () => {
    await backend.close();
  });
  beforeEach(async () => {
    await wipeGraph(backend.bolt);
    await seedVeinDomain(backend.bolt);
    store = new MemoryRunStore();
    ws = new Neo4jWorkspaceStore(backend);
    await ws.publishWorkflow(WF, "v1", { steps: [{ id: "plan", type: "agent", config: {} }] });
  });

  const count = async (label: string) =>
    Number((await backend.bolt.run(`MATCH (n:\`${label}\`) RETURN count(n) AS c`))[0]!["c"]);
  const edges = async (edge: string) =>
    Number((await backend.bolt.run(`MATCH (:Data_Bank)-[r:\`${edge}\`]->(:Data_Bank) RETURN count(r) AS c`))[0]!["c"]);

  it("projects runs into VeinRun / VeinAgentSession / VeinToolCall with IN_RUN, IN_SESSION, EXECUTED edges", async () => {
    const hash = (await ws.getWorkflowHash(WF))!;
    for (const e of sampleEvents(hash)) await store.append(WF, RUN, e);
    await store.finalize(WF, RUN, {
      runId: RUN, workflow: WF, startedAt: ts(0), finishedAt: ts(9), durationMs: 9000, status: "success", input: { q: "deliver" }, output: { delivered: 60 },
    });

    const report = await projectRuns(backend, store, { workflows: [WF] });
    assert.deepEqual([report.runs, report.sessions, report.toolCalls, report.edges, report.skipped], [1, 1, 2, 4, 0]);
    assert.equal(await count("VeinRun"), 1);
    assert.equal(await count("VeinAgentSession"), 1);
    assert.equal(await count("VeinToolCall"), 2);
    assert.equal(await edges("IN_RUN"), 1);
    assert.equal(await edges("IN_SESSION"), 2);
    assert.equal(await edges("EXECUTED"), 1);

    const run = (await backend.bolt.run(`MATCH (r:VeinRun) RETURN properties(r) AS p`))[0]!["p"] as Record<string, unknown>;
    assert.equal(run["status"], "success");
    assert.equal(run["output_preview"], '{"delivered":60}');
    assert.equal(run["unique_source_id"], `veinrun:${RUN}`);
    assert.equal(run["started_at"], Math.floor(T0 / 1000));
    assert.equal(run["duration_ms"], 9000);

    // "which runs executed this version" is one hop.
    const rows = await backend.bolt.run(
      `MATCH (r:VeinRun)-[:EXECUTED]->(v:VeinWorkflowVersion)<-[:ACTIVE_VERSION]-(w:VeinWorkflow) RETURN w.name AS wf, r.run_id AS run`,
    );
    assert.deepEqual(rows, [{ wf: WF, run: RUN }]);
  });

  it("is idempotent, skips settled runs, and re-projects an unsettled run once it finalizes", async () => {
    const events = sampleEvents("h");
    for (const e of events.slice(0, 7)) await store.append(WF, RUN, e); // no terminal event yet
    let report = await projectRuns(backend, store, { workflows: [WF] });
    assert.equal(report.runs, 1);
    let run = (await backend.bolt.run(`MATCH (r:VeinRun) RETURN r.status AS s, r.ref_id AS id`))[0]!;
    assert.equal(run["s"], "stale");

    // Re-run with nothing new: the stale run is re-read (not settled), same nodes.
    report = await projectRuns(backend, store, { workflows: [WF] });
    assert.equal(report.runs, 1);
    assert.equal(await count("VeinRun"), 1);
    assert.equal(await count("VeinToolCall"), 2);
    assert.equal(await edges("IN_SESSION"), 2);

    for (const e of events.slice(7)) await store.append(WF, RUN, e);
    await store.finalize(WF, RUN, {
      runId: RUN, workflow: WF, startedAt: ts(0), finishedAt: ts(9), durationMs: 9000, status: "success", input: {},
    });
    report = await projectRuns(backend, store, { workflows: [WF] });
    const after = (await backend.bolt.run(`MATCH (r:VeinRun) RETURN r.status AS s, r.ref_id AS id`))[0]!;
    assert.equal(after["s"], "success");
    assert.equal(after["id"], run["id"], "upsert keeps the node identity");

    report = await projectRuns(backend, store, { workflows: [WF] });
    assert.deepEqual([report.runs, report.skipped], [0, 1], "settled runs are skipped");
    report = await projectRuns(backend, store, { workflows: [WF], skipSettled: false });
    assert.deepEqual([report.runs, report.skipped], [1, 0]);
    assert.equal(await count("VeinRun"), 1);
  });

  it("projects chats and turns with IN_CHAT edges and SPAWNED edges to runs the chat launched", async () => {
    for (const e of sampleEvents("h")) await store.append(WF, RUN, e);
    const chats = new MemoryChatStore();
    await chats.createChat({ id: "c1", title: "Deliver", model: "claude" });
    await chats.appendMessages("c1", [
      { role: "user", content: "run the delivery" },
      { role: "assistant", content: [{ type: "tool-call", toolName: "run_workflow", input: { name: WF } }] },
      { role: "tool", content: [{ type: "tool-result", toolName: "run_workflow", output: { type: "json", value: { runId: RUN, status: "success" } } }] },
      { role: "assistant", content: [{ type: "text", text: "Done — 60 delivered." }] },
      { role: "user", content: "thanks" },
      { role: "assistant", content: "Any time." },
    ]);

    const report = await projectAll(backend, { store, chatStore: chats, workflows: [WF] });
    assert.deepEqual([report.runs, report.chats, report.turns], [1, 1, 2]);
    assert.equal(await edges("IN_CHAT"), 2);
    assert.equal(await edges("SPAWNED"), 1);
    const turns = await backend.bolt.run(`MATCH (t:VeinTurn) RETURN t.turn AS n, t.user_text_preview AS u, t.assistant_text_preview AS a ORDER BY n`);
    assert.deepEqual(turns, [
      { n: 0, u: "run the delivery", a: "Done — 60 delivered." },
      { n: 1, u: "thanks", a: "Any time." },
    ]);
    const chain = await backend.bolt.run(`MATCH (c:VeinChat)-[:SPAWNED]->(r:VeinRun)<-[:IN_RUN]-(s:VeinAgentSession)<-[:IN_SESSION]-(t:VeinToolCall) RETURN count(t) AS c`);
    assert.equal(chain[0]!["c"], 2, "chat → run → session → tool call provenance chain");

    // Idempotent.
    await projectChats(backend, chats);
    assert.equal(await count("VeinChat"), 1);
    assert.equal(await count("VeinTurn"), 2);
    assert.equal(await edges("IN_CHAT"), 2);
  });
});
