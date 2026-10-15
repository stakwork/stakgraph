import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildRegistry, createStrut, MemoryRunStore, MemorySecretStore, WorkspaceManager } from "strut";
import { seedConceptSteps, seedConceptWorkflows } from "../seed.js";

/**
 * The GitHub token reaches the concept steps only through the run's
 * `secrets` (docs/plans/gitree-to-workflows.md): stored as the actor's
 * `GITHUB_TOKEN` before the launch, read by `clone-repo` and `fetch-changes`
 * inside it, and absent from every event the run log records — the seeded
 * `bootstrap-then-process`, run through a real strut over fake services.
 */

// The mcp dir, like the real lab-workspace: a seeded step's bare `strut`
// import resolves via mcp/node_modules from there.
const MCP_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..");
const TOKEN = "ghp_alice_secret_token_value";

const EXISTING = {
  id: "o/r/search",
  repo: "o/r",
  name: "Search",
  description: "Find things.",
  prNumbers: [],
  createdAt: new Date(0),
  lastUpdated: new Date(0),
};

function fakeServices(concepts: unknown[] = []) {
  const seen = { clone: [] as Array<string | undefined>, github: [] as Array<string | undefined>, usage: [] as Array<{ repo: string; total: number }> };
  const octokit = { paginate: async () => [], pulls: {}, repos: {} };
  const usage = (total: number) => ({ input: total - 1, cache_read: 0, cache_write: 0, output: 1, total });
  const services = {
    storage: {
      getAllConcepts: async () => concepts,
      getChronologicalCheckpoint: async () => null,
      setChronologicalCheckpoint: async () => {},
      getLastProcessedPR: async () => 0,
      getLastProcessedCommit: async () => null,
      linkConceptsToFiles: async () => ({ conceptsProcessed: 0, edgesLinked: 0 }),
      linkPRsToFiles: async () => ({ prsProcessed: 0, edgesLinked: 0 }),
      addToTotalUsage: async (repo: string, u: { total: number }) => {
        seen.usage.push({ repo, total: u.total });
      },
    },
    github: (token?: string) => {
      seen.github.push(token);
      return octokit;
    },
    clone: async (_owner: string, _repo: string, token?: string) => {
      seen.clone.push(token);
      return "/tmp/not-a-clone";
    },
    bootstrap: async () => ({ concepts: [], usage: usage(15) }),
    exploreConcept: async () => {},
    summarizer: { summarizeModifiedConcepts: async () => usage(2) },
    llm: {},
    prContent: async () => "",
    commitContent: async () => "",
    codeNodes: { withoutDescription: async () => [], contained: async () => ({}), writeDescriptions: async () => {} },
  };
  return { services, seen };
}

describe("the GitHub token in a concepts run", () => {
  it("reaches the steps through the actor's secret and never the run log", async () => {
    const prevKey = process.env.STRUT_API_KEY;
    delete process.env.STRUT_API_KEY;
    const root = await mkdtemp(join(MCP_DIR, ".lab-token-secret-"));
    const { services, seen } = fakeServices();
    const ws = new WorkspaceManager(root);
    await seedConceptWorkflows(ws);
    await seedConceptSteps(ws);
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    const strut = await createStrut({
      workspace: ws,
      registry,
      store: new MemoryRunStore(),
      secretStore: new MemorySecretStore(),
      actorSecretStore: new MemorySecretStore(),
      services,
      serveUi: false,
      enableChat: false,
      scheduler: false,
      autoResume: false,
    });
    try {
      // What mcp's proxy does before launching as the actor.
      const put = await strut.app.fetch(
        new Request("http://lab/actors/alice/secrets/GITHUB_TOKEN", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ value: TOKEN }),
        }),
      );
      assert.equal(put.status, 200, await put.text());

      const result = await strut.run("bootstrap-then-process", { owner: "o", repo: "r" }, { actor: "alice" });
      assert.equal(result.status, "success", JSON.stringify(result.error));

      // The steps that reach GitHub got the actor's token…
      assert.deepEqual(seen.clone, [TOKEN]);
      assert.ok(seen.github.includes(TOKEN), "fetch-changes read the secret");
      // …and both sweeps' spend landed on the repo's total.
      assert.deepEqual(
        seen.usage.sort((a, b) => a.total - b.total),
        [
          { repo: "o/r", total: 2 },
          { repo: "o/r", total: 15 },
        ],
      );

      // The log names the actor and carries the token nowhere.
      const events = await strut.store.getRunEvents("bootstrap-then-process", result.runId);
      const start = events.find((e) => e.type === "run.start");
      assert.equal(start?.actor, "alice");
      assert.equal(start?.principal, "alice");
      assert.ok(events.some((e) => e.type === "step.start" && e.path.endsWith("/clone")));
      assert.ok(!JSON.stringify(events).includes(TOKEN), "the token is in the run log");
    } finally {
      await strut.close();
      await rm(root, { recursive: true, force: true });
      if (prevKey !== undefined) process.env.STRUT_API_KEY = prevKey;
    }
  });
  it("runs the sweep alone on a repo that already has concepts", async () => {
    const prevKey = process.env.STRUT_API_KEY;
    delete process.env.STRUT_API_KEY;
    const root = await mkdtemp(join(MCP_DIR, ".lab-token-secret-"));
    const { services, seen } = fakeServices([EXISTING]);
    const ws = new WorkspaceManager(root);
    await seedConceptWorkflows(ws);
    await seedConceptSteps(ws);
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    const strut = await createStrut({
      workspace: ws,
      registry,
      store: new MemoryRunStore(),
      secretStore: new MemorySecretStore(),
      actorSecretStore: new MemorySecretStore(),
      services,
      serveUi: false,
      enableChat: false,
      scheduler: false,
      autoResume: false,
    });
    try {
      await strut.app.fetch(
        new Request("http://lab/actors/alice/secrets/GITHUB_TOKEN", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ value: TOKEN }),
        }),
      );

      // `check` is false: clone, bootstrap and record are skipped, and the
      // subflow reads `clone?.repoPath` off a step that never ran.
      const result = await strut.run("bootstrap-then-process", { owner: "o", repo: "r" }, { actor: "alice" });
      assert.equal(result.status, "success", JSON.stringify(result.error));
      assert.deepEqual(seen.clone, [], "no clone for an existing repo");
      assert.ok(seen.github.includes(TOKEN), "fetch-changes read the secret");
      assert.deepEqual(seen.usage, [{ repo: "o/r", total: 2 }], "only the sweep's spend");
      assert.equal((result.output as { count: number }).count, 1);

      const events = await strut.store.getRunEvents("bootstrap-then-process", result.runId);
      const skipped = events.filter((e) => e.type === "step.skipped").map((e) => e.path.split("/").pop());
      assert.deepEqual(skipped.sort(), ["bootstrap", "clone", "record"]);
      assert.ok(!events.some((e) => e.type === "step.error"), JSON.stringify(events.filter((e) => e.type === "step.error")));
    } finally {
      await strut.close();
      await rm(root, { recursive: true, force: true });
      if (prevKey !== undefined) process.env.STRUT_API_KEY = prevKey;
    }
  });
});
