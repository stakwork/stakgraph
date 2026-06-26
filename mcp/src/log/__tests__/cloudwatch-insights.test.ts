import { test, expect } from "@playwright/test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { fileURLToPath } from "url";

import {
  buildInsightsQuery,
  fetchCloudwatchLogsInsights,
  type FetchCloudwatchParams,
} from "../cloudwatch.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "cw-insights-test-"));
}

type SendResult = Record<string, unknown>;

/** Minimal CloudWatchLogsClient mock that uses a per-command handler map. */
function makeMockClient(handlers: {
  StartQueryCommand?: (input: Record<string, unknown>) => SendResult;
  GetQueryResultsCommand?: (input: Record<string, unknown>) => SendResult;
}) {
  return {
    send: async (command: { constructor: { name: string }; input: Record<string, unknown> }) => {
      const name = command.constructor.name as keyof typeof handlers;
      const handler = handlers[name];
      if (!handler) throw new Error(`No mock handler for ${name}`);
      return handler(command.input);
    },
  } as unknown as import("@aws-sdk/client-cloudwatch-logs").CloudWatchLogsClient;
}

// ---------------------------------------------------------------------------
// 1. Query string — no filter, no streams
// ---------------------------------------------------------------------------
test("buildInsightsQuery: no filter and no streams produces correct query", () => {
  const qs = buildInsightsQuery({});
  expect(qs).toContain("fields @timestamp, @message, @logStream");
  expect(qs).toContain("| sort @timestamp desc");
  expect(qs).toContain("| limit 10000");
  expect(qs).not.toContain("filter @message");
  expect(qs).not.toContain("filter @logStream");
});

// ---------------------------------------------------------------------------
// 2. Query string — string filter
// ---------------------------------------------------------------------------
test("buildInsightsQuery: string filter inserts message like clause", () => {
  const qs = buildInsightsQuery({ filterPattern: "147426952" });
  expect(qs).toContain("| filter @message like /147426952/");
  expect(qs).not.toContain("filter @logStream");
});

// ---------------------------------------------------------------------------
// 3. Query string — stream filter
// ---------------------------------------------------------------------------
test("buildInsightsQuery: stream filter inserts logStream in clause", () => {
  const qs = buildInsightsQuery({ logStreamNames: ["stream1"] });
  expect(qs).toContain('| filter @logStream in ["stream1"]');
  expect(qs).not.toContain("filter @message");
});

// ---------------------------------------------------------------------------
// 4. Query string — combined filter + streams
// ---------------------------------------------------------------------------
test("buildInsightsQuery: combined filter and streams inserts both clauses", () => {
  const qs = buildInsightsQuery({
    filterPattern: "ERROR",
    logStreamNames: ["stream1", "stream2"],
  });
  expect(qs).toContain("| filter @message like /ERROR/");
  expect(qs).toContain('| filter @logStream in ["stream1", "stream2"]');
});

// ---------------------------------------------------------------------------
// 5. JSON pattern rejection
// ---------------------------------------------------------------------------
test("buildInsightsQuery: throws on JSON-style filterPattern", () => {
  expect(() =>
    buildInsightsQuery({ filterPattern: "{ $.level = ERROR }" })
  ).toThrow("JSON-style filter patterns are not supported by Logs Insights");
});

// Also verify the throw surfaces correctly from the full function
test("fetchCloudwatchLogsInsights: throws on JSON-style filterPattern", async () => {
  const logsDir = makeTmpDir();
  const params: FetchCloudwatchParams = {
    logGroupName: "/test/group",
    filterPattern: "{ $.statusCode = 500 }",
    minutes: 5,
    logsDir,
  };
  await expect(fetchCloudwatchLogsInsights(params)).rejects.toThrow(
    "JSON-style filter patterns are not supported by Logs Insights"
  );
});

// ---------------------------------------------------------------------------
// 6. Truncation on timeout
// ---------------------------------------------------------------------------
test("fetchCloudwatchLogsInsights: truncated=true when deadline is hit before Complete", async () => {
  const logsDir = makeTmpDir();

  // Mock client: StartQuery returns a queryId; GetQueryResults always returns Running
  let getResultsCallCount = 0;
  const mockClient = makeMockClient({
    StartQueryCommand: () => ({ queryId: "qid-truncation-test" }),
    GetQueryResultsCommand: () => {
      getResultsCallCount++;
      return { status: "Running", results: [] };
    },
  });

  // Fast-forward Date.now so the deadline is immediately exceeded after the
  // first poll attempt (StartQuery + 1 GetQueryResults call).
  const realDateNow = Date.now.bind(Date);
  let callCount = 0;
  const patchedNow = () => {
    callCount++;
    // First two calls (compute endTimeMs + deadline = Date.now() + 60_000)
    // return real time so StartQuery is dispatched normally.
    if (callCount <= 2) return realDateNow();
    // Every call after that returns a time well past the deadline.
    return realDateNow() + 70_000;
  };
  (Date as unknown as { now: () => number }).now = patchedNow;

  try {
    const result = await fetchCloudwatchLogsInsights(
      {
        logGroupName: "/test/group",
        minutes: 5,
        logsDir,
      },
      mockClient
    );
    expect(result.truncated).toBe(true);
    expect(result.lineCount).toBe(0);
  } finally {
    (Date as unknown as { now: () => number }).now = realDateNow;
  }
});

// ---------------------------------------------------------------------------
// 7. Time conversion — epoch seconds (not ms)
// ---------------------------------------------------------------------------
test("fetchCloudwatchLogsInsights: passes startTime/endTime in epoch seconds", async () => {
  const logsDir = makeTmpDir();
  let capturedInput: Record<string, unknown> = {};

  const mockClient = makeMockClient({
    StartQueryCommand: (input) => {
      capturedInput = input;
      return { queryId: "qid-time-test" };
    },
    GetQueryResultsCommand: () => ({ status: "Complete", results: [] }),
  });

  const beforeMs = Date.now();

  await fetchCloudwatchLogsInsights(
    {
      logGroupName: "/test/group",
      minutes: 30,
      logsDir,
    },
    mockClient
  );

  const afterMs = Date.now();

  const startTime = capturedInput.startTime as number;
  const endTime = capturedInput.endTime as number;

  // Epoch seconds are in the ~1.7 billion range, far below 10^12 (ms range)
  expect(startTime).toBeLessThan(1e12);
  expect(endTime).toBeLessThan(1e12);

  // endTime should correspond to roughly now (within 5 seconds)
  expect(endTime).toBeGreaterThanOrEqual(Math.floor(beforeMs / 1000) - 1);
  expect(endTime).toBeLessThanOrEqual(Math.ceil(afterMs / 1000) + 1);

  // startTime should be 30 minutes (1800 seconds) before endTime
  const delta = endTime - startTime;
  expect(delta).toBeGreaterThanOrEqual(1798);
  expect(delta).toBeLessThanOrEqual(1802);
});

// ---------------------------------------------------------------------------
// Timeout with partial rows
// ---------------------------------------------------------------------------
test("fetchCloudwatchLogsInsights: Timeout status returns partial results immediately", async () => {
  const logsDir = makeTmpDir();
  let getQueryCallCount = 0;

  const mockClient = makeMockClient({
    StartQueryCommand: () => ({ queryId: "qid-timeout-partial" }),
    GetQueryResultsCommand: () => {
      getQueryCallCount++;
      return {
        status: "Timeout",
        results: [
          [
            { field: "@timestamp", value: "2024-01-01T00:00:00.000Z" },
            { field: "@logStream", value: "stream1" },
            { field: "@message", value: "partial log line" },
          ],
        ],
      };
    },
  });

  const result = await fetchCloudwatchLogsInsights(
    { logGroupName: "/test/group", minutes: 30, logsDir },
    mockClient
  );

  expect(result.truncated).toBe(true);
  expect(result.lineCount).toBe(1);
  expect(getQueryCallCount).toBe(1);
});

// ---------------------------------------------------------------------------
// Timeout with empty / undefined results
// ---------------------------------------------------------------------------
test("fetchCloudwatchLogsInsights: Timeout status with no results returns lineCount 0 without throwing", async () => {
  const logsDir = makeTmpDir();

  // Case A: results key missing entirely
  const mockClientUndefined = makeMockClient({
    StartQueryCommand: () => ({ queryId: "qid-timeout-empty-a" }),
    GetQueryResultsCommand: () => ({ status: "Timeout" }),
  });

  const resultA = await fetchCloudwatchLogsInsights(
    { logGroupName: "/test/group", minutes: 30, logsDir },
    mockClientUndefined
  );

  expect(resultA.truncated).toBe(true);
  expect(resultA.lineCount).toBe(0);

  // Case B: results is an empty array
  const mockClientEmpty = makeMockClient({
    StartQueryCommand: () => ({ queryId: "qid-timeout-empty-b" }),
    GetQueryResultsCommand: () => ({ status: "Timeout", results: [] }),
  });

  const resultB = await fetchCloudwatchLogsInsights(
    { logGroupName: "/test/group", minutes: 30, logsDir },
    mockClientEmpty
  );

  expect(resultB.truncated).toBe(true);
  expect(resultB.lineCount).toBe(0);
});
