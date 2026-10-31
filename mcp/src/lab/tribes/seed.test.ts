import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WorkspaceManager, createStrut } from "strut";
import { seedTribesWorkflows } from "./seed.js";

/** What Hive's POST /api/tribes/log launches (hive src/app/api/tribes/log/route.ts). */
const TEXT = [
  { id: "m-1", time: "2026-10-30T14:03:12.000Z", alias: "Alice", owner: true, mine: false, text: "Shipping the login fix today" },
  { id: "m-2", time: "2026-10-30T14:03:40.000Z", alias: "Jamie", owner: false, mine: true, text: "Two PRs are open on auth.", replyTo: "m-1", thread: "m-1" },
]
  .map((l) => JSON.stringify(l))
  .join("\n");
const INPUT = {
  tribe: { pubkey: "02" + "a".repeat(64), name: "Builders", host: "tribes.v2.sphinx.chat" },
  workspace: "acme",
  from: "2026-10-30T14:03:12.000Z",
  to: "2026-10-30T14:03:40.000Z",
  messages: 2,
  aliases: ["Alice"],
  file: "tribe-messages.jsonl",
};
const ATTACHMENT = { name: "tribe-messages.jsonl", mediaType: "application/x-ndjson", data: Buffer.from(TEXT).toString("base64") };

describe("tribe-messages workflow", () => {
  let root: string;
  let ws: WorkspaceManager;
  let strut: Awaited<ReturnType<typeof createStrut>>;

  before(async () => {
    root = await mkdtemp(join(tmpdir(), "tribes-wf-"));
    ws = new WorkspaceManager(root);
    await seedTribesWorkflows(ws);
    strut = await createStrut({ workspace: ws, serveUi: false, enableChat: false, scheduler: false });
  });

  after(async () => {
    await strut.close();
    await rm(root, { recursive: true, force: true });
  });

  it("is seeded under tribes and validates against the real registry", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "tribe-messages");
    assert.ok(entry, "tribe-messages is seeded");
    assert.equal(entry.category, "tribes");
    const yaml = await ws.getWorkflowSource("tribe-messages", entry.activeVersion);
    const v = await (strut.services as any).authoring.validateWorkflow(yaml, "tribe-messages");
    assert.equal(v.ok, true, JSON.stringify(v.errors, null, 2));
  });

  it("reseeding the same YAML publishes nothing", async () => {
    const before = (await ws.listWorkflows()).find((w) => w.name === "tribe-messages")!.activeVersion;
    await seedTribesWorkflows(ws);
    assert.equal((await ws.listWorkflows()).find((w) => w.name === "tribe-messages")!.activeVersion, before);
  });

  it("takes Hive's launch as it is sent: the input and the batch as the run's attachment", async () => {
    const res = await strut.app.request("/workflows/tribe-messages/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: INPUT, attachments: [ATTACHMENT] }),
    });
    assert.equal(res.status, 202, await res.clone().text());
    const { runId } = (await res.json()) as { runId: string };

    let summary: any;
    for (let i = 0; i < 100; i++) {
      summary = await (await strut.app.request(`/workflows/tribe-messages/runs/${runId}`)).json();
      if (!summary.partial && summary.status && summary.status !== "running") break;
      await new Promise((r) => setTimeout(r, 50));
    }
    assert.equal(summary.status, "success", JSON.stringify(summary.error));
    assert.deepEqual(summary.output, {
      file: "attachments/tribe-messages.jsonl",
      tribe: INPUT.tribe,
      workspace: INPUT.workspace,
      from: INPUT.from,
      to: INPUT.to,
      messages: 2,
      aliases: ["Alice"],
    });

    const file = await strut.app.request(`/artifacts/${runId}/attachments/tribe-messages.jsonl`);
    assert.equal(file.status, 200);
    assert.equal(await file.text(), TEXT);
  });
});
