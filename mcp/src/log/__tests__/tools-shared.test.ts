import { test, expect } from "../../testkit.js";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  BASH_SEARCH_HINT,
  fetchCloudwatchText,
  fetchQuickwitText,
  get_log_tools,
  listCloudwatchGroupsText,
  timeBudgetNote,
} from "../tools.js";

// The three tools `/logs/agent` shares with the lab's `logs` workflow, as they
// answered before their bodies moved out of get_log_tools. Exact strings.

const cw = (over: Record<string, unknown> = {}) => async (p: any) => ({
  file: "cw-swarms-38-x.log",
  lineCount: 5,
  logGroup: p.logGroupName,
  timeRange: { startTime: "2026-10-20T09:30:00.000Z", endTime: "2026-10-20T10:00:00.000Z" },
  truncated: false,
  hydratedLines: 0,
  ...over,
});

test("fetchCloudwatchText: the result says where the file is and how to read it", async () => {
  expect(await fetchCloudwatchText({ log_group: "/swarms/38" }, { logsDir: "/tmp", fetch: cw() })).toBe(
    "Fetched 5 log lines from /swarms/38 (2026-10-20T09:30:00.000Z to 2026-10-20T10:00:00.000Z). Saved to file: cw-swarms-38-x.log. Use bash to search through it (e.g. rg, grep, head, tail, awk).",
  );
  expect(
    await fetchCloudwatchText({ log_group: "/swarms/38" }, { logsDir: "/tmp", fetch: cw({ truncated: true, hydratedLines: 3 }), searchHint: "Search it with logs_search." }),
  ).toBe(
    "Fetched 5 log lines from /swarms/38 (2026-10-20T09:30:00.000Z to 2026-10-20T10:00:00.000Z). Saved to file: cw-swarms-38-x.log. Search it with logs_search. 3 long line(s) were recovered to full length via GetLogRecord (Logs Insights truncates long messages in query results). NOTE: results may be incomplete (fetch hit the wall-clock cap or the query timed out on AWS) — narrow the filter_pattern, log_stream_names, or minutes to get complete results.",
  );
  const failing = async () => { throw new Error("AccessDenied"); };
  expect(await fetchCloudwatchText({ log_group: "/x" }, { logsDir: "/tmp", fetch: failing as any })).toBe("Failed to fetch CloudWatch logs: AccessDenied");
});

test("listCloudwatchGroupsText: one line per group, size and retention when known", async () => {
  const list = (logGroups: any[]) => async () => ({ logGroups });
  expect(await listCloudwatchGroupsText({}, { list: list([]) })).toBe("No log groups found");
  expect(await listCloudwatchGroupsText({ prefix: "/ecs/" }, { list: list([]) })).toBe('No log groups found with prefix "/ecs/"');
  expect(
    await listCloudwatchGroupsText({}, { list: list([{ name: "/swarms/38", storedBytes: 5 * 1024 * 1024, retentionDays: 7 }, { name: "/b" }]) }),
  ).toBe("/swarms/38 (5MB, 7d retention)\n/b (?)");
});

test("fetchQuickwitText: writes the hits one per line and says where", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "qw-shared-"));
  const search = async () => ({
    num_hits: 2,
    hits: [{ timestamp: 1700000000, level: "error", service: "api", message: "boom  " }, { message: "x" }],
  });
  const text = await fetchQuickwitText({ query: "level:error" }, { logsDir: dir, search: search as any });
  const [file] = fs.readdirSync(dir);
  expect(text).toBe(`Fetched 2 log hits from Quickwit (query: "level:error"). Saved 2 lines to file: ${file}. ${BASH_SEARCH_HINT}`);
  expect(file).toMatch(/^qw-level_error-.*\.log$/);
  expect(fs.readFileSync(path.join(dir, file!), "utf8")).toBe("[2023-11-14T22:13:20.000Z] [error] [api] boom\n[unknown] [] [] x");
});

test("timeBudgetNote: elapsed, budget and what is left; a warning in the last two minutes", () => {
  const now = Date.now();
  expect(timeBudgetNote(now - 3 * 60_000, 8 * 60_000)).toBe("\n\n[time budget: 3.0 min elapsed of 8 min; ~5.0 min left]");
  expect(timeBudgetNote(now - 7 * 60_000, 8 * 60_000)).toBe(
    "\n\n[time budget: 7.0 min elapsed of 8 min; ~1.0 min left — wrap up and write your final answer now]",
  );
});

test("get_log_tools: the same tools, the shared ones with the shared descriptions", () => {
  const tools = get_log_tools({ logsDir: os.tmpdir() }) as Record<string, { description?: string }>;
  expect(Object.keys(tools)).toEqual(["fetch_cloudwatch", "list_cloudwatch_groups", "fetch_quickwit", "bash"]);
  expect(tools["list_cloudwatch_groups"]!.description).toBe(
    "List available CloudWatch log groups. Use this to discover which log groups exist before fetching logs.",
  );
});
