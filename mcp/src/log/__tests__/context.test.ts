import { test, expect } from "../../testkit.js";
import { buildLogsPrompt, swarmNameToLogGroup } from "../context.js";

// The `/logs/agent` prompt, as the handler built it before the assembly moved
// to context.ts (lab/logs builds the same one). Exact strings: a change here
// is a change to what the agent reads.

const deps = {
  listCloudwatchLogStreams: async (group: string) =>
    group.includes("pools")
      ? [{ name: "ws-1/code-server", lastEventTime: "2026-10-20T10:00:00.000Z" }]
      : [{ name: "stakgraph.sphinx", lastEventTime: "2026-10-20T10:00:00.000Z" }, { name: "boltwall.sphinx" }],
  getCurrentDateSnippet: () => "Today is Monday, October 20, 2026.",
};

const POOL =
  'If the user asks about "pod", "pool", or "sandbox" logs, they mean the workspace pod logs in the CloudWatch log group "/workspaces/workspace-cluster/pools/pool1/workspaces". Use the fetch_cloudwatch tool with this log group, then use bash to search the saved file.\n\n' +
  'Each workspace pod has its own set of log streams named "{workspace-id}/{container}". The main workspace container is "code-server" — for most questions about a running workspace/sandbox, fetch the code-server stream first. Other containers: port-detector (port detection sidecar), build-base-image / build-wrapper-image (init image builders), init-workspace (init: workspace setup), docker-auth (init: registry auth).\n\n' +
  'Available log streams in this log group:\n  - "ws-1/code-server" (last event: 2026-10-20T10:00:00.000Z)\n\n' +
  'Use the log_stream_names parameter to filter to a specific workspace/container (e.g. the "{workspace-id}/code-server" stream).';
const QUICKWIT = 'If the user mentions their actual application name (e.g. "acme"), use the fetch_quickwit tool to fetch logs from the actual application.';
const SWARM =
  'The CloudWatch log group for this swarm is "/swarms/38". Use this log group when fetching logs.\n\n' +
  'Available log streams (services) in this log group:\n' +
  '  - "stakgraph.sphinx" (service: stakgraph, last event: 2026-10-20T10:00:00.000Z)\n' +
  '  - "boltwall.sphinx" (service: boltwall)\n\n' +
  "If the user mentions a specific service, use the log_stream_names parameter to filter to that stream. If no specific service is mentioned, fetch all streams.";

test("buildLogsPrompt: swarm, app and pool — each block prepended, the last added first", async () => {
  // With a swarm the date line is not there: the handler's swarm block was
  // built on the bare prompt before the move, and still is.
  expect(await buildLogsPrompt("Why?", { swarmName: "swarm38", workspaceSlug: "acme", poolName: "pool1" }, deps)).toBe(
    `${POOL}\n\n${QUICKWIT}\n\n${SWARM}\n\nWhy?`,
  );
});

test("buildLogsPrompt: no swarm — the date line leads the question", async () => {
  expect(await buildLogsPrompt("Why?", { workspaceSlug: "acme" }, deps)).toBe(
    `${QUICKWIT}\n\nToday is Monday, October 20, 2026.\n\nWhy?`,
  );
});

test("buildLogsPrompt: a stream listing that fails leaves its block's first line", async () => {
  const failing = { ...deps, listCloudwatchLogStreams: async () => { throw new Error("AccessDenied"); } };
  expect(await buildLogsPrompt("Why?", { swarmName: "swarm38" }, failing)).toBe(
    'The CloudWatch log group for this swarm is "/swarms/38". Use this log group when fetching logs.\n\nWhy?',
  );
});

test("buildLogsPrompt: the tools it names are the reader's own", async () => {
  const prompt = await buildLogsPrompt("Why?", { workspaceSlug: "acme", poolName: "pool1" }, deps, {
    fetchCloudwatch: "logs_fetch_cloudwatch",
    fetchQuickwit: "logs_fetch_quickwit",
    search: "logs_search",
  });
  expect(prompt).toContain("Use the logs_fetch_cloudwatch tool with this log group, then use logs_search to search the saved file.");
  expect(prompt).toContain("use the logs_fetch_quickwit tool to fetch logs");
  expect(prompt).not.toContain("bash");
});

test("swarmNameToLogGroup", () => {
  expect(swarmNameToLogGroup("swarm38")).toBe("/swarms/38");
  expect(swarmNameToLogGroup("swarmHDYF7D")).toBe("/swarms/HDYF7D");
  expect(swarmNameToLogGroup("acme")).toBeNull();
});
