import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WorkspaceManager, createStrut } from "strut";
import { seedCallsWorkflows } from "./seed.js";

/** What Hive's POST /api/calls/transcript launches (hive src/app/api/calls/transcript/route.ts). */
const TEXT = [
  { n: 1, at: "2026-10-27T14:03:12.481Z", speaker: "Alice", identity: "alice-7f3a", source: "voice", text: "Can we look at the login bug?" },
  { n: 2, at: "2026-10-27T14:03:26.730Z", speaker: "Jamie", source: "jamie", text: "Two PRs are open on auth." },
]
  .map((l) => JSON.stringify(l))
  .join("\n");
const INPUT = {
  room: "sphinx.call.-abc",
  workspace: "acme",
  conversation: null,
  startedAt: "2026-10-27T14:00:00.000Z",
  endedAt: "2026-10-27T14:05:00.000Z",
  lines: 2,
  speakers: ["Alice"],
  file: "call-transcript.jsonl",
};
const ATTACHMENT = { name: "call-transcript.jsonl", mediaType: "application/x-ndjson", data: Buffer.from(TEXT).toString("base64") };

describe("call-transcript workflow", () => {
  let root: string;
  let ws: WorkspaceManager;
  let strut: Awaited<ReturnType<typeof createStrut>>;

  before(async () => {
    root = await mkdtemp(join(tmpdir(), "calls-wf-"));
    ws = new WorkspaceManager(root);
    await seedCallsWorkflows(ws);
    strut = await createStrut({ workspace: ws, serveUi: false, enableChat: false, scheduler: false });
  });

  after(async () => {
    await strut.close();
    await rm(root, { recursive: true, force: true });
  });

  it("is seeded under calls and validates against the real registry", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "call-transcript");
    assert.ok(entry, "call-transcript is seeded");
    assert.equal(entry.category, "calls");
    const yaml = await ws.getWorkflowSource("call-transcript", entry.activeVersion);
    const v = await (strut.services as any).authoring.validateWorkflow(yaml, "call-transcript");
    assert.equal(v.ok, true, JSON.stringify(v.errors, null, 2));
  });

  it("reseeding the same YAML publishes nothing", async () => {
    const before = (await ws.listWorkflows()).find((w) => w.name === "call-transcript")!.activeVersion;
    await seedCallsWorkflows(ws);
    assert.equal((await ws.listWorkflows()).find((w) => w.name === "call-transcript")!.activeVersion, before);
  });

  it("takes Hive's launch as it is sent: the input (conversation null) and the transcript as the run's attachment", async () => {
    const res = await strut.app.request("/workflows/call-transcript/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: INPUT, attachments: [ATTACHMENT] }),
    });
    assert.equal(res.status, 202, await res.clone().text());
    const { runId } = (await res.json()) as { runId: string };

    let summary: any;
    for (let i = 0; i < 100; i++) {
      summary = await (await strut.app.request(`/workflows/call-transcript/runs/${runId}`)).json();
      if (!summary.partial && summary.status && summary.status !== "running") break;
      await new Promise((r) => setTimeout(r, 50));
    }
    assert.equal(summary.status, "success", JSON.stringify(summary.error));
    assert.deepEqual(summary.output, {
      file: "attachments/call-transcript.jsonl",
      room: INPUT.room,
      workspace: INPUT.workspace,
      conversation: null,
      startedAt: INPUT.startedAt,
      endedAt: INPUT.endedAt,
      lines: 2,
      speakers: ["Alice"],
    });

    const file = await strut.app.request(`/artifacts/${runId}/attachments/call-transcript.jsonl`);
    assert.equal(file.status, 200);
    assert.equal(await file.text(), TEXT);
  });
});
