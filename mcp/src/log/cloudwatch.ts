import {
  CloudWatchLogsClient,
  FilterLogEventsCommand,
  FilteredLogEvent,
  DescribeLogGroupsCommand,
  DescribeLogStreamsCommand,
  StartQueryCommand,
  GetQueryResultsCommand,
  GetLogRecordCommand,
} from "@aws-sdk/client-cloudwatch-logs";
import { fromIni } from "@aws-sdk/credential-providers";
import * as fs from "fs";
import * as path from "path";

function getClient(): CloudWatchLogsClient {
  const opts: ConstructorParameters<typeof CloudWatchLogsClient>[0] = {
    region: process.env.AWS_REGION || "us-east-1",
    // Bound every request so a stalled socket can't hang the agent forever.
    maxAttempts: 3,
    requestHandler: {
      connectionTimeout: 5_000,
      requestTimeout: 30_000,
    },
  };
  if (process.env.AWS_PROFILE) {
    opts.credentials = fromIni({ profile: process.env.AWS_PROFILE });
  }
  return new CloudWatchLogsClient(opts);
}

// Hard caps so FilterLogEvents can't paginate forever when a filter pattern
// matches sparsely across a high-volume log group (each page returns 0 events
// but a nextToken, so an `allEvents.length < limit` check never terminates).
const MAX_FETCH_PAGES = 200;
const MAX_FETCH_WALL_MS = 60_000;

// GetQueryResults truncates long field values (~1000 chars for @message).
// Each result row carries a @ptr whose GetLogRecord response contains the
// complete event, so any message at/over this length is suspect and gets
// re-fetched ("hydrated") individually.
export const HYDRATE_MIN_MSG_LEN = 950;
const MAX_HYDRATE_RECORDS = 500;
const HYDRATE_BATCH_SIZE = 5;
const HYDRATE_BATCH_INTERVAL_MS = 500; // 5 calls / 500ms keeps GetLogRecord under ~10 TPS
const MAX_HYDRATE_WALL_MS = 30_000;

/** Sanitize a log group name into a safe filename */
function safeFilename(logGroup: string): string {
  return logGroup.replace(/^\/+/, "").replace(/\//g, "-");
}

export interface FetchCloudwatchParams {
  logGroupName: string;
  logStreamNames?: string[];
  filterPattern?: string;
  minutes?: number;
  limit?: number;
  logsDir: string;
  abortSignal?: AbortSignal;
}

export interface FetchCloudwatchResult {
  file: string;
  lineCount: number;
  logGroup: string;
  timeRange: { startTime: string; endTime: string };
  truncated: boolean;
  /** Lines whose full text was recovered via GetLogRecord (Insights path only). */
  hydratedLines?: number;
}

/**
 * Fetch CloudWatch logs and write them to a file on disk.
 * Paginates through all results up to `limit` events.
 */
export async function fetchCloudwatchLogs(
  params: FetchCloudwatchParams
): Promise<FetchCloudwatchResult> {
  const { logGroupName, logStreamNames, filterPattern, minutes = 30, limit = 10000, logsDir, abortSignal } = params;
  const client = getClient();

  const endTime = Date.now();
  const startTime = endTime - minutes * 60 * 1000;

  const allEvents: FilteredLogEvent[] = [];
  let nextToken: string | undefined;
  let pages = 0;
  const deadline = Date.now() + MAX_FETCH_WALL_MS;
  let truncated = false;

  do {
    if (abortSignal?.aborted) {
      throw new Error("CloudWatch fetch aborted");
    }
    if (Date.now() > deadline) {
      truncated = true;
      console.warn(
        `[cloudwatch] fetch for ${logGroupName} hit ${MAX_FETCH_WALL_MS}ms wall-clock cap after ${pages} pages (${allEvents.length} events)`
      );
      break;
    }
    if (pages >= MAX_FETCH_PAGES) {
      truncated = true;
      console.warn(
        `[cloudwatch] fetch for ${logGroupName} hit ${MAX_FETCH_PAGES}-page cap (${allEvents.length} events)`
      );
      break;
    }

    const command = new FilterLogEventsCommand({
      logGroupName,
      logStreamNames: logStreamNames?.length ? logStreamNames : undefined,
      filterPattern,
      startTime,
      endTime,
      limit: Math.min(limit - allEvents.length, 10000),
      nextToken,
    });

    const response = await client.send(command, {
      abortSignal: abortSignal as any,
    });
    if (response.events) {
      allEvents.push(...response.events);
    }
    nextToken = response.nextToken;
    pages++;
  } while (nextToken && allEvents.length < limit);

  // Write to file
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `cw-${safeFilename(logGroupName)}-${ts}.log`;
  const filepath = path.join(logsDir, filename);

  const lines = allEvents.map((e) => {
    const time = e.timestamp
      ? new Date(e.timestamp).toISOString()
      : "unknown";
    const stream = e.logStreamName || "";
    const msg = (e.message || "").trimEnd();
    return `[${time}] [${stream}] ${msg}`;
  });

  fs.writeFileSync(filepath, lines.join("\n"), "utf-8");

  return {
    file: filename,
    lineCount: lines.length,
    logGroup: logGroupName,
    timeRange: {
      startTime: new Date(startTime).toISOString(),
      endTime: new Date(endTime).toISOString(),
    },
    truncated,
  };
}

// ---------------------------------------------------------------------------
// Logs Insights helpers (exported for unit testing)
// ---------------------------------------------------------------------------

export interface InsightsQueryParams {
  filterPattern?: string;
  logStreamNames?: string[];
  limit?: number;
}

/**
 * Build the CloudWatch Logs Insights query string.
 * Exported so tests can verify the query without touching AWS.
 * Throws if `filterPattern` is a JSON-style metric filter pattern.
 */
export function buildInsightsQuery(params: InsightsQueryParams): string {
  const { filterPattern, logStreamNames, limit } = params;

  if (filterPattern && filterPattern.trimStart().startsWith("{")) {
    throw new Error(
      "JSON-style filter patterns are not supported by Logs Insights — use a plain string pattern instead."
    );
  }

  const insightsLimit = Math.min(limit ?? 10000, 10000);
  const parts: string[] = ["fields @timestamp, @message, @logStream"];

  if (filterPattern) {
    const escaped = filterPattern.replace(/\//g, "\\/");
    parts.push(`| filter @message like /${escaped}/`);
  }

  if (logStreamNames && logStreamNames.length > 0) {
    const list = logStreamNames.map((s) => `"${s}"`).join(", ");
    parts.push(`| filter @logStream in [${list}]`);
  }

  parts.push("| sort @timestamp desc");
  parts.push(`| limit ${insightsLimit}`);

  return parts.join("\n");
}

/**
 * Fetch CloudWatch logs using Logs Insights (newest-first via `sort @timestamp desc`).
 * Replaces FilterLogEvents for high-volume log groups where oldest-first pagination
 * would exhaust page/time caps before reaching recent logs.
 *
 * @param params   Standard fetch params (same interface as fetchCloudwatchLogs).
 * @param _client  Optional CloudWatchLogsClient for dependency injection in tests.
 */
export async function fetchCloudwatchLogsInsights(
  params: FetchCloudwatchParams,
  _client?: CloudWatchLogsClient
): Promise<FetchCloudwatchResult> {
  const {
    logGroupName,
    logStreamNames,
    filterPattern,
    minutes = 30,
    limit = 10000,
    logsDir,
    abortSignal,
  } = params;

  const queryString = buildInsightsQuery({ filterPattern, logStreamNames, limit });
  const client = _client ?? getClient();

  const endTimeMs = Date.now();
  const startTimeMs = endTimeMs - minutes * 60 * 1000;

  // StartQueryCommand uses epoch seconds (not ms like FilterLogEvents)
  const startQueryResp = await client.send(
    new StartQueryCommand({
      logGroupName,
      startTime: Math.floor(startTimeMs / 1000),
      endTime: Math.floor(endTimeMs / 1000),
      queryString,
    }),
    { abortSignal: abortSignal as any }
  );

  const queryId = startQueryResp.queryId;
  if (!queryId) {
    throw new Error("CloudWatch Logs Insights did not return a queryId");
  }

  // Poll until Complete / Failed / Cancelled or the wall-clock deadline
  const deadline = Date.now() + MAX_FETCH_WALL_MS;
  let truncated = false;
  let results: import("@aws-sdk/client-cloudwatch-logs").ResultField[][] = [];

  const sleep = (ms: number) =>
    new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, ms);
      if (abortSignal) {
        abortSignal.addEventListener("abort", () => {
          clearTimeout(t);
          reject(new Error("CloudWatch fetch aborted"));
        });
      }
    });

  while (true) {
    if (abortSignal?.aborted) {
      throw new Error("CloudWatch fetch aborted");
    }
    if (Date.now() > deadline) {
      truncated = true;
      console.warn(
        `[cloudwatch] Insights query for ${logGroupName} hit ${MAX_FETCH_WALL_MS}ms wall-clock cap`
      );
      break;
    }

    const pollResp = await client.send(
      new GetQueryResultsCommand({ queryId }),
      { abortSignal: abortSignal as any }
    );

    const status = pollResp.status;
    if (status === "Complete") {
      results = pollResp.results ?? [];
      break;
    } else if (status === "Timeout") {
      results = pollResp.results ?? [];
      truncated = true;
      console.warn(
        `[cloudwatch] Insights query for ${logGroupName} returned terminal Timeout — returning ${results.length} partial result(s)`
      );
      break;
    } else if (status === "Failed" || status === "Cancelled") {
      throw new Error(`CloudWatch Logs Insights query ${status.toLowerCase()}`);
    }

    // Still Running / Scheduled — wait 1 s before the next poll
    await sleep(1000);
  }

  // Map result rows, keeping each row's @ptr so truncated messages can be
  // hydrated to their full text below.
  const rows = results.map((row) => {
    let timestamp = "";
    let logStream = "";
    let message = "";
    let ptr = "";
    for (const field of row) {
      if (field.field === "@timestamp") timestamp = field.value ?? "";
      else if (field.field === "@logStream") logStream = field.value ?? "";
      else if (field.field === "@message") message = (field.value ?? "").trimEnd();
      else if (field.field === "@ptr") ptr = field.value ?? "";
    }
    const isoTime = timestamp ? new Date(timestamp).toISOString() : "unknown";
    return { isoTime, logStream, message, ptr };
  });

  const hydratedLines = await hydrateTruncatedMessages(rows, client, abortSignal);

  const lines = rows.map((r) => `[${r.isoTime}] [${r.logStream}] ${r.message}`);

  // Write to file — same naming convention as fetchCloudwatchLogs
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `cw-${safeFilename(logGroupName)}-${ts}.log`;
  const filepath = path.join(logsDir, filename);
  fs.writeFileSync(filepath, lines.join("\n"), "utf-8");

  return {
    file: filename,
    lineCount: lines.length,
    logGroup: logGroupName,
    timeRange: {
      startTime: new Date(startTimeMs).toISOString(),
      endTime: new Date(endTimeMs).toISOString(),
    },
    truncated,
    hydratedLines,
  };
}

/**
 * Re-fetch suspected-truncated messages via GetLogRecord (the row's @ptr).
 * Mutates `rows` in place; returns how many messages were replaced with a
 * longer full-text version. Failures leave the truncated message untouched.
 */
async function hydrateTruncatedMessages(
  rows: { message: string; ptr: string }[],
  client: CloudWatchLogsClient,
  abortSignal?: AbortSignal
): Promise<number> {
  const suspects = rows.filter(
    (r) => r.ptr && r.message.length >= HYDRATE_MIN_MSG_LEN
  );
  if (suspects.length === 0) return 0;

  const toHydrate = suspects.slice(0, MAX_HYDRATE_RECORDS);
  if (suspects.length > toHydrate.length) {
    console.warn(
      `[cloudwatch] ${suspects.length} suspected-truncated lines; hydrating only the first ${MAX_HYDRATE_RECORDS}`
    );
  }

  const deadline = Date.now() + MAX_HYDRATE_WALL_MS;
  let hydrated = 0;
  let failed = 0;

  for (let i = 0; i < toHydrate.length; i += HYDRATE_BATCH_SIZE) {
    if (abortSignal?.aborted || Date.now() > deadline) break;
    const batch = toHydrate.slice(i, i + HYDRATE_BATCH_SIZE);
    await Promise.all(
      batch.map(async (row) => {
        try {
          const resp = await client.send(
            new GetLogRecordCommand({ logRecordPointer: row.ptr }),
            { abortSignal: abortSignal as any }
          );
          const full = (resp.logRecord?.["@message"] ?? "").trimEnd();
          if (full.length > row.message.length) {
            row.message = full;
            hydrated++;
          }
        } catch {
          failed++;
        }
      })
    );
    if (i + HYDRATE_BATCH_SIZE < toHydrate.length) {
      await new Promise<void>((resolve) => {
        const t = setTimeout(resolve, HYDRATE_BATCH_INTERVAL_MS);
        abortSignal?.addEventListener(
          "abort",
          () => {
            clearTimeout(t);
            resolve();
          },
          { once: true }
        );
      });
    }
  }

  if (failed > 0) {
    console.warn(
      `[cloudwatch] failed to hydrate ${failed} truncated line(s) via GetLogRecord`
    );
  }
  return hydrated;
}

export interface ListLogGroupsResult {
  logGroups: { name: string; storedBytes?: number; retentionDays?: number }[];
}

/** List available CloudWatch log groups (for discovery) */
export async function listCloudwatchLogGroups(
  prefix?: string
): Promise<ListLogGroupsResult> {
  const client = getClient();
  const command = new DescribeLogGroupsCommand({
    logGroupNamePrefix: prefix,
    limit: 50,
  });
  const response = await client.send(command);
  return {
    logGroups: (response.logGroups || []).map((g) => ({
      name: g.logGroupName || "",
      storedBytes: g.storedBytes,
      retentionDays: g.retentionInDays,
    })),
  };
}

export interface LogStreamInfo {
  name: string;
  lastEventTime?: string;
}

/** List log streams for a given log group */
export async function listCloudwatchLogStreams(
  logGroupName: string
): Promise<LogStreamInfo[]> {
  const client = getClient();
  const command = new DescribeLogStreamsCommand({
    logGroupName,
    orderBy: "LastEventTime",
    descending: true,
    limit: 50,
  });
  const response = await client.send(command);
  return (response.logStreams || []).map((s) => ({
    name: s.logStreamName || "",
    lastEventTime: s.lastEventTimestamp
      ? new Date(s.lastEventTimestamp).toISOString()
      : undefined,
  }));
}
