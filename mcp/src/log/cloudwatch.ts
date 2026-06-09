import {
  CloudWatchLogsClient,
  FilterLogEventsCommand,
  FilteredLogEvent,
  DescribeLogGroupsCommand,
  DescribeLogStreamsCommand,
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
