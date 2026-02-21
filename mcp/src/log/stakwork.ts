import * as fs from "fs";
import * as path from "path";
import { ensureLogsDir } from "./utils.js";

export interface FetchWorkflowRunParams {
  apiKey: string;
  projectId: string;
  step?: string;
  status?: "success" | "error" | "warning" | "failed";
  include_children?: boolean;
  limit?: number;
  page?: number;
}

export interface WorkflowLog {
  id: number;
  project_id: number;
  project_name: string;
  step_name: string | null;
  status: string;
  message: string;
  created_at: string;
  skill_id: number | null;
}

export interface FetchWorkflowRunResult {
  file: string;
  logCount: number;
  projectId: string;
  pagination: {
    page: number;
    items: number;
    count: number;
    pages: number;
  };
}

/**
 * Fetch logs for a Stakwork workflow run (project) and write them to a file on disk.
 */
export async function fetchWorkflowRunLogs(
  params: FetchWorkflowRunParams
): Promise<FetchWorkflowRunResult> {
  const { apiKey, projectId, step, status, include_children, limit, page } =
    params;

  const qs = new URLSearchParams();
  if (step) qs.set("step", step);
  if (status) qs.set("status", status);
  if (include_children) qs.set("include_children", "true");
  if (limit !== undefined) qs.set("limit", String(limit));
  if (page !== undefined) qs.set("page", String(page));

  const query = qs.toString();
  const url = `https://api.stakwork.com/api/v1/projects/${projectId}/logs${query ? `?${query}` : ""}`;

  const resp = await fetch(url, {
    headers: {
      Authorization: `Token token=${apiKey}`,
    },
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Stakwork API error (${resp.status}): ${body}`);
  }

  const json = (await resp.json()) as {
    success: boolean;
    data: {
      logs: WorkflowLog[];
      pagination: {
        page: number;
        items: number;
        count: number;
        pages: number;
      };
    };
  };

  if (!json.success) {
    throw new Error(
      `Stakwork API returned success=false: ${JSON.stringify(json)}`
    );
  }

  const { logs, pagination } = json.data;

  // Write to file
  const dir = ensureLogsDir();
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `stakwork-${projectId}-p${pagination.page}-${ts}.log`;
  const filepath = path.join(dir, filename);

  const lines = logs.map((l) => {
    return `[${l.created_at}] [${l.status}] [${l.step_name || "unknown_step"}] ${l.message}`;
  });

  fs.writeFileSync(filepath, lines.join("\n"), "utf-8");

  return {
    file: filename,
    logCount: logs.length,
    projectId,
    pagination,
  };
}
