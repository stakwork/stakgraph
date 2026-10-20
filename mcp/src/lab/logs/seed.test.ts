import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, createStrut, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedLogsSteps, seedLogsWorkflows } from "./seed.js";
import { buildLogsServices, type LogsServiceDeps } from "./service.js";
import {
  FETCH_CLOUDWATCH_DESCRIPTION,
  FETCH_QUICKWIT_DESCRIPTION,
  LIST_CLOUDWATCH_GROUPS_DESCRIPTION,
  fetchCloudwatchInput,
  fetchQuickwitInput,
  listCloudwatchGroupsInput,
} from "../../log/tools.js";

/** The logs agent's whole toolset: the committed list, so a built-in, a
 *  shell or another namespace slipping in fails here. */
const TOOLS = [
  "logs/fetch-cloudwatch",
  "logs/list-cloudwatch-groups",
  "logs/fetch-quickwit",
  "logs/search",
  "logs/read",
  "logs/insights",
];

function findRg(): LogsServiceDeps["rg"] | null {
  if (!spawnSync("rg", ["--version"]).error) return { command: "rg" };
  const cc = process.env["CLAUDE_CODE_EXECPATH"];
  if (cc && spawnSync(cc, ["--version"], { argv0: "rg", encoding: "utf8" }).stdout?.startsWith("ripgrep")) return { command: cc, argv0: "rg" };
  return null;
}
const RG = findRg();

describe("logs workflow (offline: fake agent that calls the real logs/* steps)", () => {
  const agentCalls: Record<string, any>[] = [];
  const toolResults: string[] = [];
  /** Stands in for the model: fetches, searches and answers through the run's own registry. */
  const fakeAgent = defineStep({
    type: "agent",
    input: z.object({}).passthrough(),
    output: z.any(),
    async run(cfg: any, ctx: any) {
      agentCalls.push(cfg);
      toolResults.length = 0;
      toolResults.push(await ctx.registry["logs/fetch-cloudwatch"].run({ log_group: "/swarms/38", filter_pattern: "ERROR" }, ctx));
      if (RG) toolResults.push(await ctx.registry["logs/search"].run({ pattern: "ingest failed" }, ctx));
      return { result: "The ingest failed at 10:02 (disk full).", steps: 3, usage: {}, cost: 0.02 };
    },
  });
  let root: string;
  let ws: WorkspaceManager;
  let run: (input: Record<string, unknown>, opts?: { job?: string }) => ReturnType<typeof runWorkflow>;
  let registry: Awaited<ReturnType<typeof buildRegistry>>["registry"];

  before(async () => {
    root = await mkdtemp(join(tmpdir(), "logs-wf-"));
    ws = new WorkspaceManager(root);
    await seedLogsSteps(ws);
    await seedLogsWorkflows(ws);
    const flow = await ws.getWorkflow("logs");
    ({ registry } = await buildRegistry(await ws.materializeCustomSteps()));
    const dataDir = join(root, "data");
    const logs = buildLogsServices({
      ...(RG ? { rg: RG } : {}),
      listStreams: async (group) => (group === "/swarms/38" ? [{ name: "stakgraph.sphinx", lastEventTime: "2026-10-20T10:00:00.000Z" }] : []),
      fetchCloudwatch: async (p) => {
        await writeFile(join(p.logsDir, "cw-swarms-38-x.log"), "[2026-10-20T10:02:00.000Z] [stakgraph.sphinx] ERROR ingest failed: disk full\n");
        return { file: "cw-swarms-38-x.log", lineCount: 1, logGroup: p.logGroupName, timeRange: { startTime: "s", endTime: "e" }, truncated: false, hydratedLines: 0 };
      },
    });
    // What createStrut + createLabStrut put on the bag: the artifacts dir
    // (job/dir without a job), the deployment's secrets, services.logs.
    const services = {
      ...standardServices({ secretsSource: { LOGS_SWARM_NAME: "swarm38" }, dataDir }),
      artifacts: fileArtifactsCapability(join(dataDir, "artifacts")),
      logs,
    };
    run = (input, opts = {}) => runWorkflow(flow, input, { ...registry, agent: fakeAgent } as typeof registry, { services, ...opts });
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("is seeded under `logs`, open to peers, with the explore skeleton and the six logs/* tools only", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "logs");
    assert.equal(entry?.category, "logs");
    assert.match(entry?.description ?? "", /strut\/run-workflow \{ peer, workflow: "logs"/);
    const yaml = await readFile(new URL("./workflows/logs.yaml", import.meta.url), "utf-8");
    assert.match(yaml, /^peer: true$/m, "strut runs a peer's launch only of a workflow that says so");
    const flow = (await ws.getWorkflow("logs")) as any;
    assert.deepEqual(Object.keys(flow.inputBlock ?? {}).sort(), ["prompt", "session"]);
    assert.deepEqual(flow.steps.map((s: any) => s.type), ["job/dir", "logs/context", "agent", "pack"]);
    assert.deepEqual(flow.params.tools, TOOLS);
    const agent = flow.steps[2].config;
    assert.deepEqual(agent.toolFilter, ["none"], "no built-ins: no bash, no file tools, no web");
    assert.equal(agent.mcp, false, "none of a caller's MCP servers");
    for (const t of TOOLS) assert.ok(registry[t], `registry missing ${t}`);
  });

  it("validates statically against the real registry (what meta/validate-workflow runs)", async () => {
    const real = await createStrut({ workspace: ws, serveUi: false, enableChat: false, scheduler: false });
    const authoring = (real.services as any).authoring;
    const entry = (await ws.listWorkflows()).find((w) => w.name === "logs")!;
    const v = await authoring.validateWorkflow(await ws.getWorkflowSource("logs", entry.activeVersion), "logs");
    assert.equal(v.ok, true, JSON.stringify(v.errors, null, 2));
    assert.equal(v.summary.steps, 4);
  });

  it("the three fetch steps say exactly what /logs/agent's tools say", () => {
    const json = (s: unknown) => JSON.stringify(z.toJSONSchema(s as z.ZodType));
    for (const [type, description, input] of [
      ["logs/fetch-cloudwatch", FETCH_CLOUDWATCH_DESCRIPTION, fetchCloudwatchInput],
      ["logs/list-cloudwatch-groups", LIST_CLOUDWATCH_GROUPS_DESCRIPTION, listCloudwatchGroupsInput],
      ["logs/fetch-quickwit", FETCH_QUICKWIT_DESCRIPTION, fetchQuickwitInput],
    ] as const) {
      assert.equal(registry[type]!.description, description, `${type}: description drifted from log/tools.ts`);
      assert.equal(json(registry[type]!.input), json(input), `${type}: input drifted from log/tools.ts`);
    }
  });

  it("a run: this deployment's swarm in the prompt, the endpoint's prompt without its end marker, tools that fetch and search, the answer packed", async () => {
    agentCalls.length = 0;
    const res = await run({ prompt: "Why did the last ingest fail?" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.deepEqual(res.output, { answer: "The ingest failed at 10:02 (disk full).", cost: 0.02 });

    const cfg = agentCalls.at(-1)!;
    assert.match(cfg.prompt, /^The CloudWatch log group for this swarm is "\/swarms\/38"\. Use this log group when fetching logs\./, "LOGS_SWARM_NAME, from the deployment's secrets");
    assert.match(cfg.prompt, /"stakgraph\.sphinx" \(service: stakgraph, last event: 2026-10-20T10:00:00\.000Z\)/);
    assert.match(cfg.prompt, /Why did the last ingest fail\?$/);
    assert.equal(cfg.session, undefined, "cold without a session");
    assert.ok(cfg.cwd.startsWith(join(root, "data", "artifacts")), "a one-shot keeps its files in its artifact dir");
    assert.match(cfg.finalAnswer, /complete analysis/);
    assert.match(cfg.system, /^You are a log analysis agent\./);
    assert.match(cfg.system, /You have roughly 8 minutes of wall-clock time total\./);
    assert.match(cfg.system, /Every tool result ends with a "\[time budget: \.\.\.\]" line/);
    assert.match(cfg.system, /logs_search.*logs_read.*logs_insights/s);
    assert.match(cfg.system, /Before reporting logs_insights counts as errors, read a few matching lines/);
    assert.doesNotMatch(cfg.system, /END_OF_ANSWER|bash|fetch_agent_log|Stakwork/);

    assert.match(toolResults[0]!, /^Fetched 1 log lines from \/swarms\/38 \(s to e\)\. Saved to file: cw-swarms-38-x\.log\. Search it with logs_search/);
    assert.match(toolResults[0]!, /\n\n\[time budget: 0\.\d min elapsed of 8 min; ~\d\.\d min left\]$/);
    if (RG) assert.match(toolResults[1]!, /^cw-swarms-38-x\.log:1:\[2026-10-20T10:02:00\.000Z\] \[stakgraph\.sphinx\] ERROR ingest failed: disk full\n\n\[time budget: /);
  });

  it("under a job the files stay in the job's directory, and `session` is the thread", async () => {
    agentCalls.length = 0;
    const res = await run({ prompt: "And before that?", session: "thread-1" }, { job: "job-1" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const cfg = agentCalls.at(-1)!;
    assert.equal(cfg.session, "thread-1");
    assert.equal(cfg.cwd, join(root, "data", "jobs", "job-1"));
    await readFile(join(root, "data", "jobs", "job-1", "logs", "cw-swarms-38-x.log"), "utf-8");
  });

  it("refuses a launch without a prompt", async () => {
    const res = await run({});
    assert.equal(res.status, "error");
    assert.match(res.error?.message ?? "", /prompt/);
  });
});
