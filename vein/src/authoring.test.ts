import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { createVein, type Vein } from "./createVein.js";
import { WorkspaceManager } from "./workspace.js";
import type { AuthoringCapability } from "./authoring.js";

/**
 * The authoring capability (services.authoring) + the meta/* lib steps —
 * EVOLVE_SPEC §5.2/§6: an in-workflow agent's author/test/inspect surface,
 * closed over the artifacts it publishes (publisher "ai").
 *
 * Authored step sources deliberately avoid `import "vein"`: the temp
 * workspace lives outside the package tree, where that specifier can't
 * resolve at dynamic-import time. A duck-typed `{ parse }` schema exercises
 * the same registry/load/run paths.
 */

// A loadable custom step with no imports (see note above).
const echoStep = (type: string, n: number) => `export default {
  type: ${JSON.stringify(type)},
  description: "echoes its config",
  input: { parse: (v) => v },
  output: { parse: (v) => v },
  run: async (cfg) => ({ echoed: cfg.msg, rev: ${n} }),
};
`;

const logFlow = (name: string, msg: string) =>
  `name: ${name}\nsteps:\n  - id: say\n    type: log\n    config:\n      message: "${msg}"\n`;

describe("authoring capability (the meta surface)", () => {
  let tempDir: string;
  let vein: Vein<Record<string, unknown>>;
  let authoring: AuthoringCapability;

  before(async () => {
    tempDir = join(tmpdir(), `vein-authoring-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
    vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      serveUi: false,
    });
    authoring = (vein.services as { authoring?: AuthoringCapability }).authoring!;
  });

  after(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("createVein auto-provides services.authoring and registers the meta/* lib steps", async () => {
    assert.ok(authoring, "services.authoring should be auto-provided");
    const registry = vein.getRegistry();
    for (const type of [
      "meta/list-steps",
      "meta/search-steps",
      "meta/get-step",
      "meta/create-step",
      "meta/edit-step",
      "meta/run-step",
      "meta/list-workflows",
      "meta/get-workflow",
      "meta/publish-workflow",
      "meta/run-workflow",
      "meta/list-runs",
      "meta/get-run",
      "meta/list-secrets",
    ]) {
      assert.ok(registry[type], `expected "${type}" in the registry`);
    }
    // The underscore helper is not a step.
    assert.equal(registry["meta/_shared"], undefined);
  });

  it("createStep → runStep → editStep is the inner loop, load-verified", async () => {
    const created = (await authoring.createStep("cand/echo", echoStep("cand/echo", 1))) as any;
    assert.equal(created.ok, true, JSON.stringify(created));
    assert.equal(created.version, "v1");
    assert.equal(created.loaded, true);

    const ran = (await authoring.runStep("cand/echo", { config: { msg: "hi" } })) as any;
    assert.equal(ran.status, "success", JSON.stringify(ran.error));
    assert.equal(ran.output.echoed, "hi");
    assert.equal(ran.output.rev, 1);

    const edited = (await authoring.editStep("cand/echo", echoStep("cand/echo", 2))) as any;
    assert.equal(edited.ok, true);
    assert.equal(edited.version, "v2");
    assert.equal(edited.changed, true);

    const ran2 = (await authoring.runStep("cand/echo", { config: { msg: "yo" } })) as any;
    assert.equal(ran2.output.rev, 2);
  });

  it("createStep hands back the load error for broken source (§5.3.4)", async () => {
    const broken = (await authoring.createStep("cand/broken", "export default { nope")) as any;
    assert.ok(!broken.ok, "a step that doesn't load must not report ok");
    assert.ok(broken.error, "the import failure must be handed back");
    assert.equal(broken.loaded, false);
  });

  it("editStep refuses steps the agent surface did not author", async () => {
    await vein.workspace.publishStep(
      "seeded/tool",
      echoStep("seeded/tool", 1),
      undefined,
      "harness-seed",
    );
    const res = (await authoring.editStep("seeded/tool", echoStep("seeded/tool", 2))) as any;
    assert.ok(res.error && /only edits steps it authored/.test(res.error), res.error);
  });

  it("publishWorkflow stamps publisher 'ai'; run + run-history are closed over the stamped set", async () => {
    // Candidate: published through the meta surface → stamped.
    const pub = (await authoring.publishWorkflow("cand-flow", logFlow("cand-flow", "v1"))) as any;
    assert.equal(pub.ok, true, JSON.stringify(pub));
    assert.equal(pub.version, "v1");
    assert.equal(pub.created, true);

    const listed = (await authoring.listWorkflows()) as any;
    const entry = listed.workflows.find((w: any) => w.name === "cand-flow");
    assert.equal(entry.publisher, "ai");

    // Idempotent republish; changed content bumps the version.
    const same = (await authoring.publishWorkflow("cand-flow", logFlow("cand-flow", "v1"))) as any;
    assert.equal(same.changed, false);
    const bumped = (await authoring.publishWorkflow("cand-flow", logFlow("cand-flow", "v2"))) as any;
    assert.equal(bumped.version, "v2");

    // Harness: created outside the meta surface → unstamped.
    await vein.workspace.createWorkflow("harness-flow", logFlow("harness-flow", "gold"));

    const overwrite = (await authoring.publishWorkflow(
      "harness-flow",
      logFlow("harness-flow", "evil"),
    )) as any;
    assert.ok(overwrite.error && /not agent-authored/.test(overwrite.error), overwrite.error);

    const runRefused = (await authoring.runWorkflow("harness-flow")) as any;
    assert.ok(runRefused.error && /not agent-authored/.test(runRefused.error));

    const historyRefused = (await authoring.listRuns("harness-flow")) as any;
    assert.ok(historyRefused.error && /not agent-authored/.test(historyRefused.error));

    // The stamped candidate runs, and its run history reads back.
    const result = (await authoring.runWorkflow("cand-flow", {})) as any;
    assert.equal(result.status, "success", JSON.stringify(result.error));
    assert.ok(result.runId);

    const runs = (await authoring.listRuns("cand-flow")) as any;
    assert.equal(runs.workflow, "cand-flow");
    assert.ok(runs.runs.length >= 1);

    const run = (await authoring.getRun("cand-flow", result.runId)) as any;
    assert.equal(run.summary.status, "success");
    assert.ok(Array.isArray(run.events) && run.events.length > 0);
    // Slim events carry no payloads by default.
    assert.equal(run.events[0].input, undefined);
  });

  it("runWorkflow sees a step authored moments before (fresh registry, §5.3.1)", async () => {
    const created = (await authoring.createStep("cand/echo2", echoStep("cand/echo2", 7))) as any;
    assert.equal(created.ok, true);

    const yaml = `name: cand-flow2\nsteps:\n  - id: e\n    type: cand/echo2\n    config:\n      msg: "fresh"\n`;
    const pub = (await authoring.publishWorkflow("cand-flow2", yaml)) as any;
    assert.equal(pub.ok, true);

    const result = (await authoring.runWorkflow("cand-flow2", {})) as any;
    assert.equal(result.status, "success", JSON.stringify(result.error));
    assert.equal((result.output as any).echoed, "fresh");
  });

  it("meta/* steps reach the capability through ctx.services inside a real run", async () => {
    const result = await vein.run({
      name: "meta-wiring-test",
      input: z.any(),
      steps: [
        { id: "pub", type: "meta/publish-workflow", config: {
          name: "cand-inrun",
          yaml: logFlow("cand-inrun", "from inside a run"),
        } },
        { id: "run", type: "meta/run-workflow", config: { name: "cand-inrun" }, depends: "pub" },
        { id: "history", type: "meta/list-runs", config: { name: "cand-inrun" }, depends: "run" },
      ],
    });
    assert.equal(result.status, "success", JSON.stringify(result.error));
    const out = result.output as any;
    assert.ok(out.runs.length >= 1, "the in-run candidate's run history should read back");

    // ...and the workflow published from inside the run is stamped.
    const listed = (await authoring.listWorkflows()) as any;
    const entry = listed.workflows.find((w: any) => w.name === "cand-inrun");
    assert.equal(entry.publisher, "ai");
  });

  it("listSecrets returns names only", async () => {
    await vein.secretStore.set("MY_TOKEN", "shh");
    const res = (await authoring.listSecrets()) as any;
    const names = res.secrets.map((s: any) => s.name);
    assert.ok(names.includes("MY_TOKEN"));
    assert.ok(!JSON.stringify(res).includes("shh"), "secret values must never surface");
  });
});
