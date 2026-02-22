import * as fs from "fs";
import * as path from "path";
import { StakworkRunSummary } from "./types.js";

export interface FetchAgentLogParams {
  runs: StakworkRunSummary[];
  projectId: number;
  agent?: string;
  logsDir: string;
}

export interface FetchAgentLogResult {
  file: string;
  lineCount: number;
  projectId: number;
  agent: string;
}

/**
 * Fetch the full log content for an agent from a Stakwork workflow run
 * and write it to a file on disk.
 */
export async function fetchAgentLog(
  params: FetchAgentLogParams
): Promise<FetchAgentLogResult> {
  const { runs, projectId, agent, logsDir } = params;

  const run = runs.find((r) => r.projectId === projectId);
  if (!run) {
    throw new Error(`No run found with projectId ${projectId}`);
  }
  if (!run.agentLogs || run.agentLogs.length === 0) {
    throw new Error(`Run ${projectId} has no agent logs`);
  }

  let logEntry;
  if (agent) {
    logEntry = run.agentLogs.find(
      (l) => l.agent.toLowerCase() === agent.toLowerCase()
    );
    if (!logEntry) {
      const available = run.agentLogs.map((l) => l.agent).join(", ");
      throw new Error(
        `No agent log found for "${agent}" in run ${projectId}. Available agents: ${available}`
      );
    }
  } else {
    logEntry = run.agentLogs[0];
  }

  const agentName = agent || logEntry.agent;

  const resp = await fetch(logEntry.url);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Failed to fetch agent log (${resp.status}): ${body}`);
  }

  const content = await resp.text();
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const safeAgent = agentName
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .substring(0, 40);
  const filename = `agent-log-${projectId}-${safeAgent}-${ts}.log`;
  const filepath = path.join(logsDir, filename);

  fs.writeFileSync(filepath, content, "utf-8");

  const lineCount = content.split("\n").length;
  return { file: filename, lineCount, projectId, agent: agentName };
}
