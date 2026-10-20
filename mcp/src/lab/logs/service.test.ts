import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildLogsServices, inside, LOGS_SEARCH_HINT, OUTPUT_MAX_CHARS, type LogsServiceDeps } from "./service.js";

/** A ripgrep to run: the one on PATH (the mcp image installs it), else the
 *  copy Claude Code carries (argv0 `rg`), else the search cases skip. */
function findRg(): LogsServiceDeps["rg"] | null {
  if (!spawnSync("rg", ["--version"]).error) return { command: "rg" };
  const cc = process.env["CLAUDE_CODE_EXECPATH"];
  if (cc && spawnSync(cc, ["--version"], { argv0: "rg", encoding: "utf8" }).stdout?.startsWith("ripgrep")) return { command: cc, argv0: "rg" };
  return null;
}
const RG = findRg();

/** A Logs Insights client that answers one query with `rows` (log/__tests__'s mock, smaller). */
function insightsClient(rows: Array<Record<string, string>>, status = "Complete") {
  const sent: any[] = [];
  return {
    sent,
    client: {
      send: async (cmd: { constructor: { name: string }; input: any }) => {
        sent.push({ name: cmd.constructor.name, input: cmd.input });
        if (cmd.constructor.name === "StartQueryCommand") return { queryId: "q1" };
        if (cmd.constructor.name === "GetQueryResultsCommand") {
          return { status, results: rows.map((r) => Object.entries(r).map(([field, value]) => ({ field, value }))) };
        }
        throw new Error(`unexpected ${cmd.constructor.name}`);
      },
    } as any,
  };
}

describe("services.logs (lab/logs/service.ts)", () => {
  let root: string;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "logs-service-"));
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("a tool before logs/context says so; after it, every result carries the endpoint's time-budget note", async () => {
    const fetched: any[] = [];
    const logs = buildLogsServices({
      fetchCloudwatch: async (p) => {
        fetched.push(p);
        return { file: "cw-swarms-38-x.log", lineCount: 3, logGroup: p.logGroupName, timeRange: { startTime: "s", endTime: "e" }, truncated: false, hydratedLines: 0 };
      },
      listStreams: async () => [],
    });
    assert.match(await logs.fetchCloudwatch("r0", { log_group: "/swarms/38" }), /^No logs folder for this run/);
    assert.equal(fetched.length, 0);

    const { logsDir } = await logs.begin("r1", { dir: join(root, "r1"), prompt: "Why?", budgetMinutes: 5 });
    assert.equal(logsDir, join(root, "r1", "logs"));
    const text = await logs.fetchCloudwatch("r1", { log_group: "/swarms/38", filter_pattern: "ERROR" });
    assert.equal(
      text.replace(/\[time budget: [^\]]*\]$/, "[time budget]"),
      `Fetched 3 log lines from /swarms/38 (s to e). Saved to file: cw-swarms-38-x.log. ${LOGS_SEARCH_HINT}\n\n[time budget]`,
    );
    assert.match(text, /\[time budget: 0\.0 min elapsed of 5 min; ~5\.0 min left\]$/);
    assert.equal(fetched[0].logsDir, logsDir, "fetches land in the run's logs folder");
    assert.equal(fetched[0].filterPattern, "ERROR");
  });

  it("begin: the endpoint's prompt, naming the workflow's tools; this deployment's swarm", async () => {
    const logs = buildLogsServices({ listStreams: async (g) => (g.includes("pools") ? [] : [{ name: "stakgraph.sphinx" }]) });
    const { prompt } = await logs.begin("r2", { dir: join(root, "r2"), prompt: "Why did the pod fail?", swarmName: "swarm38", poolName: "p1", workspaceSlug: "acme" });
    assert.match(prompt, /^If the user asks about "pod", "pool", or "sandbox" logs.*"\/workspaces\/workspace-cluster\/pools\/p1\/workspaces"\. Use the logs_fetch_cloudwatch tool with this log group, then use logs_search to search the saved file\./);
    assert.match(prompt, /use the logs_fetch_quickwit tool to fetch logs from the actual application/);
    assert.match(prompt, /The CloudWatch log group for this swarm is "\/swarms\/38"/);
    assert.match(prompt, /"stakgraph\.sphinx" \(service: stakgraph\)/);
    assert.match(prompt, /Why did the pod fail\?$/);
    assert.doesNotMatch(prompt, /bash/);
  });

  it("search: ripgrep with fixed arguments, inside the run's logs folder only, redacted and capped", { skip: RG ? false : "no ripgrep here" }, async () => {
    const logs = buildLogsServices({ rg: RG!, listStreams: async () => [] });
    const { logsDir } = await logs.begin("r3", { dir: join(root, "r3"), prompt: "x" });
    await writeFile(join(logsDir, "cw-a.log"), "[t1] [s] ok\n[t2] [s] ERROR boom\n[t3] [s] error lower\n[t4] [s] Authorization: Bearer sk-live-123456\n");
    await writeFile(join(logsDir, "qw-b.log"), "[t] [error] [api] ERROR in api\n");
    const strip = (s: string) => s.replace(/\n\n\[time budget: [^\]]*\]$/, "");

    assert.equal(strip(await logs.search("r3", { pattern: "ERROR", file: "cw-a.log" })), "2:[t2] [s] ERROR boom");
    const both = strip(await logs.search("r3", { pattern: "error", ignore_case: true }));
    assert.match(both, /cw-a\.log:2:\[t2\] \[s\] ERROR boom/);
    assert.match(both, /cw-a\.log:3:/);
    assert.match(both, /qw-b\.log:1:/);
    assert.equal(strip(await logs.search("r3", { pattern: "error", ignore_case: true, count: true, file: "cw-a.log" })), "2");
    assert.equal(strip(await logs.search("r3", { pattern: "boom", file: "cw-a.log", context: 1 })), "1-[t1] [s] ok\n2:[t2] [s] ERROR boom\n3-[t3] [s] error lower");
    assert.equal(strip(await logs.search("r3", { pattern: "nothing-here" })), 'No matches for "nothing-here".');
    // The secret in the line is not read back.
    const auth = strip(await logs.search("r3", { pattern: "Authorization", file: "cw-a.log" }));
    assert.doesNotMatch(auth, /sk-live-123456/);
    // A pattern that looks like a flag is a pattern; a path out of the folder is refused.
    assert.equal(strip(await logs.search("r3", { pattern: "--pre=sh" })), 'No matches for "--pre=sh".');
    assert.equal(strip(await logs.search("r3", { pattern: "x", file: "../../etc/passwd" })), '"../../etc/passwd" is not a file in this run\'s logs folder.');
    assert.equal(strip(await logs.search("r3", { pattern: "x", file: "/etc/passwd" })), '"/etc/passwd" is not a file in this run\'s logs folder.');
    // Capped: the agent cannot `| head` here.
    await writeFile(join(logsDir, "big.log"), `${"match this line ".repeat(20)}\n`.repeat(200));
    const big = strip(await logs.search("r3", { pattern: "match", file: "big.log" }));
    assert.ok(big.length < OUTPUT_MAX_CHARS + 200, String(big.length));
    assert.match(big, /more characters not shown — narrow the pattern, name a file, or set max_count\]$/);
  });

  it("search: a missing ripgrep is an answer, not a crash", async () => {
    const logs = buildLogsServices({ rg: { command: "rg-not-installed-here" }, listStreams: async () => [] });
    await logs.begin("r4", { dir: join(root, "r4"), prompt: "x" });
    assert.match(await logs.search("r4", { pattern: "x" }), /^Search failed: rg \(ripgrep\) is not installed on this server/);
  });

  it("read: numbered lines, 200 by default, at most 500, inside the folder, redacted", async () => {
    const logs = buildLogsServices({ listStreams: async () => [] });
    const { logsDir } = await logs.begin("r5", { dir: join(root, "r5"), prompt: "x" });
    await writeFile(join(logsDir, "cw-a.log"), Array.from({ length: 1000 }, (_, i) => `line ${i + 1}`).join("\n"));
    await writeFile(join(logsDir, "s.log"), 'a\n{"api_key": "abc123xyz"}\nc');
    const strip = (s: string) => s.replace(/\n\n\[time budget: [^\]]*\]$/, "");
    const first = strip(await logs.read("r5", { file: "cw-a.log" }));
    assert.equal(first.split("\n")[0], "cw-a.log, lines 1-200 of 1000:");
    assert.equal(first.split("\n")[1], "1: line 1");
    assert.equal(first.split("\n").at(-1), "200: line 200");
    assert.equal(strip(await logs.read("r5", { file: "cw-a.log", from: 10, to: 12 })), "cw-a.log, lines 10-12 of 1000:\n10: line 10\n11: line 11\n12: line 12");
    assert.match(strip(await logs.read("r5", { file: "cw-a.log", from: 1, to: 1000 })), /^cw-a\.log, lines 1-500 of 1000:/);
    assert.match(strip(await logs.read("r5", { file: "cw-a.log", from: 999, to: 5000 })), /^cw-a\.log, lines 999-1000 of 1000:/);
    assert.doesNotMatch(await logs.read("r5", { file: "s.log" }), /abc123xyz/);
    assert.equal(strip(await logs.read("r5", { file: "../r5/logs/../../x" })), '"../r5/logs/../../x" is not a file in this run\'s logs folder.');
    assert.match(strip(await logs.read("r5", { file: "nope.log" })), /^Failed to read nope\.log: /);
  });

  it("insights: the agent's query on its group, rows inline and in a file, @ptr dropped", async () => {
    const rows = Array.from({ length: 250 }, (_, i) => ({ "bin(5m)": `2026-10-20 10:${String(i % 60).padStart(2, "0")}`, "count(*)": String(i), "@ptr": "p" }));
    const { client, sent } = insightsClient(rows);
    const logs = buildLogsServices({ cloudwatchClient: client, listStreams: async () => [] });
    const { logsDir } = await logs.begin("r6", { dir: join(root, "r6"), prompt: "x" });
    const text = await logs.insights("r6", { log_group: "/swarms/38", query: "stats count(*) by bin(5m)", minutes: 60 });
    const start = sent.find((s) => s.name === "StartQueryCommand")!.input;
    assert.equal(start.logGroupName, "/swarms/38");
    assert.equal(start.queryString, "stats count(*) by bin(5m)");
    assert.equal(start.endTime - start.startTime, 3600);
    assert.match(text, /^Insights query on \/swarms\/38 \(.* to .*\) returned 250 row\(s\)\. Saved to file: insights-swarms-38-.*\.log\.\n\nbin\(5m\)=2026-10-20 10:00 \| count\(\*\)=0\n/);
    assert.doesNotMatch(text, /@ptr/);
    assert.match(text, /\n\[\.\.\. 50 more row\(s\) in the file\]\n\n\[time budget: /);
    const [file] = (await readdir(logsDir)).filter((f) => f.startsWith("insights-"));
    assert.equal((await readFile(join(logsDir, file!), "utf-8")).split("\n").length, 250);

    const partial = insightsClient([{ "count(*)": "3" }], "Timeout");
    const cut = buildLogsServices({ cloudwatchClient: partial.client, listStreams: async () => [] });
    await cut.begin("r7", { dir: join(root, "r7"), prompt: "x" });
    assert.match(await cut.insights("r7", { log_group: "/g", query: "stats count(*)" }), /returned 1 row\(s\)\. NOTE: the query did not finish/);

    const failing = buildLogsServices({ cloudwatchClient: { send: async () => { throw new Error("AccessDeniedException: not your group"); } } as any, listStreams: async () => [] });
    await failing.begin("r8", { dir: join(root, "r8"), prompt: "x" });
    assert.match(await failing.insights("r8", { log_group: "/swarms/99", query: "stats count(*)" }), /^Failed to run the Insights query: AccessDeniedException: not your group/);
  });

  it("inside: a path relative to the folder, or null when it leaves it", () => {
    assert.equal(inside("/a/logs", "x.log"), "x.log");
    assert.equal(inside("/a/logs", "./sub/../x.log"), "x.log");
    assert.equal(inside("/a/logs", "."), ".");
    assert.equal(inside("/a/logs", "../logs2/x"), null);
    assert.equal(inside("/a/logs", "/etc/passwd"), null);
  });
});
