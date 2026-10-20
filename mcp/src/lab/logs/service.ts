import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import type { CloudWatchLogsClient } from "@aws-sdk/client-cloudwatch-logs";
import { buildLogsPrompt, type LogsPromptDeps, type LogsPromptTools } from "../../log/context.js";
import {
  fetchCloudwatchText,
  fetchQuickwitText,
  listCloudwatchGroupsText,
  timeBudgetNote,
  type fetchCloudwatchInput,
  type fetchQuickwitInput,
  type listCloudwatchGroupsInput,
} from "../../log/tools.js";
import { fetchCloudwatchLogsInsights, listCloudwatchLogGroups, runInsightsQuery } from "../../log/cloudwatch.js";
import { searchLogs } from "../../handler/logs.js";
import { redactSecrets } from "../../log/redact.js";
import { env } from "../../env.js";
import type { z } from "zod";

/**
 * `services.logs` — the logs agent's tools for the lab's `logs` workflow
 * (lab/logs): the same fetches, texts and time budget as `/logs/agent`
 * (log/tools.ts, log/context.ts), and in place of its `bash` three tools
 * that cannot run anything — `search` (rg with fixed arguments), `read` (a
 * range of lines) and `insights` (an agent's own Logs Insights query, for the
 * counting and grouping awk | sort | uniq did). The workflow may be launched
 * by another strut (`peer: true`), so nothing here takes a command line.
 *
 * In code, like gaia's grader: CloudWatch needs the AWS SDK and this host's
 * credentials (each swarm's IAM role reads its own log groups), and a seeded
 * step imports `strut` alone. The logs/* steps are thin plumbing over this.
 *
 * A run's state is what `/logs/agent` kept per request: the folder its
 * fetches write to and the moment its budget started. `begin` (the
 * logs/context step, before the agent) sets it; every tool after it, called
 * with the same `runId`, appends the `[time budget: …]` note as the
 * endpoint's tools do. What the agent reads of the logs themselves — search,
 * read and Insights output — goes through `redactSecrets`, as the endpoint's
 * bash output did; nothing else carries log content.
 */

/** The logs/* steps as the agent names them (strut turns `logs/search` into `logs_search`). */
export const LOGS_TOOLS: LogsPromptTools = {
  fetchCloudwatch: "logs_fetch_cloudwatch",
  fetchQuickwit: "logs_fetch_quickwit",
  search: "logs_search",
};

/** What a fetch result says about reading the file it saved (log/tools.ts `searchHint`). */
export const LOGS_SEARCH_HINT =
  "Search it with logs_search and read lines with logs_read; to count or group, query CloudWatch with logs_insights.";

const DEFAULT_BUDGET_MIN = 8;
/** One tool result at most — the endpoint's bash let an agent `| head`; these cannot. */
export const OUTPUT_MAX_CHARS = 30_000;
const READ_DEFAULT_LINES = 200;
const READ_MAX_LINES = 500;
const INSIGHTS_ROWS_SHOWN = 200;
const SEARCH_TIMEOUT_MS = 15_000;
const SEARCH_MAX_BYTES = 4 * 1024 * 1024;
/** A run's state outlives no run by much: dropped this long after `begin`. */
const RUN_STATE_TTL_MS = 6 * 60 * 60 * 1000;

const NO_RUN =
  "No logs folder for this run: the `logs` workflow's logs/context step sets it, and this tool is only for that workflow.";

export interface SearchArgs {
  pattern: string;
  file?: string;
  ignore_case?: boolean;
  fixed_strings?: boolean;
  context?: number;
  max_count?: number;
  count?: boolean;
}
export interface ReadArgs {
  file: string;
  from?: number;
  to?: number;
}
export interface InsightsArgs {
  log_group: string;
  query: string;
  minutes?: number;
}

export interface LogsServices {
  /** logs/context: start the run's budget and folder; the prompt the agent gets. */
  begin(
    runId: string,
    opts: { dir: string; prompt: string; swarmName?: string; poolName?: string; workspaceSlug?: string; budgetMinutes?: number },
  ): Promise<{ prompt: string; logsDir: string }>;
  fetchCloudwatch(runId: string, args: z.infer<typeof fetchCloudwatchInput>, signal?: AbortSignal): Promise<string>;
  listGroups(runId: string, args: z.infer<typeof listCloudwatchGroupsInput>): Promise<string>;
  fetchQuickwit(runId: string, args: z.infer<typeof fetchQuickwitInput>): Promise<string>;
  search(runId: string, args: SearchArgs): Promise<string>;
  read(runId: string, args: ReadArgs): Promise<string>;
  insights(runId: string, args: InsightsArgs, signal?: AbortSignal): Promise<string>;
}

/** What talks to AWS and Quickwit, injectable for tests. */
export interface LogsServiceDeps {
  fetchCloudwatch?: typeof fetchCloudwatchLogsInsights;
  listGroups?: typeof listCloudwatchLogGroups;
  searchQuickwit?: typeof searchLogs;
  listStreams?: LogsPromptDeps["listCloudwatchLogStreams"];
  cloudwatchClient?: CloudWatchLogsClient;
  /** The ripgrep to run: `rg` on PATH (the mcp image installs ripgrep). */
  rg?: { command: string; argv0?: string };
}

interface RunState {
  logsDir: string;
  startTime: number;
  budgetMs: number;
}

export function buildLogsServices(deps: LogsServiceDeps = {}): LogsServices {
  const runs = new Map<string, RunState>();

  const withRun = async (runId: string, work: (run: RunState) => Promise<string>): Promise<string> => {
    const run = runs.get(runId);
    if (!run) return NO_RUN;
    return (await work(run)) + timeBudgetNote(run.startTime, run.budgetMs);
  };

  return {
    async begin(runId, { dir, prompt, swarmName, poolName, workspaceSlug, budgetMinutes }) {
      const now = Date.now();
      for (const [id, r] of runs) if (now - r.startTime > RUN_STATE_TTL_MS) runs.delete(id);
      const logsDir = join(dir, "logs");
      mkdirSync(logsDir, { recursive: true });
      runs.set(runId, { logsDir, startTime: now, budgetMs: (budgetMinutes ?? DEFAULT_BUDGET_MIN) * 60_000 });
      const text = await buildLogsPrompt(
        prompt,
        { swarmName, poolName, workspaceSlug },
        deps.listStreams ? { listCloudwatchLogStreams: deps.listStreams } : {},
        LOGS_TOOLS,
      );
      return { prompt: text, logsDir };
    },

    fetchCloudwatch: (runId, args, signal) =>
      withRun(runId, (run) =>
        fetchCloudwatchText(args, {
          logsDir: run.logsDir,
          abortSignal: signal,
          searchHint: LOGS_SEARCH_HINT,
          ...(deps.fetchCloudwatch ? { fetch: deps.fetchCloudwatch } : {}),
        }),
      ),

    listGroups: (runId, args) =>
      withRun(runId, () => listCloudwatchGroupsText(args, deps.listGroups ? { list: deps.listGroups } : {})),

    fetchQuickwit: (runId, args) =>
      withRun(runId, (run) =>
        fetchQuickwitText(args, {
          logsDir: run.logsDir,
          searchHint: LOGS_SEARCH_HINT,
          ...(deps.searchQuickwit ? { search: deps.searchQuickwit } : {}),
        }),
      ),

    search: (runId, a) =>
      withRun(runId, async (run) => {
        const target = a.file ? inside(run.logsDir, a.file) : ".";
        if (target === null) return `"${a.file}" is not a file in this run's logs folder.`;
        const args = ["--line-number", "--no-heading", "--color", "never"];
        if (a.ignore_case) args.push("--ignore-case");
        if (a.fixed_strings) args.push("--fixed-strings");
        if (a.context) args.push("--context", String(clamp(a.context, 0, 20)));
        if (a.max_count) args.push("--max-count", String(clamp(a.max_count, 1, 100_000)));
        if (a.count) args.push("--count");
        // `--`: the pattern is never read as a flag, whatever it starts with.
        args.push("--", a.pattern, target);
        const r = await rg(args, run.logsDir, deps.rg ?? { command: "rg" });
        if (r.error) return `Search failed: ${r.error}`;
        if (r.code === 1) return `No matches for ${JSON.stringify(a.pattern)}${a.file ? ` in ${a.file}` : ""}.`;
        if (r.code !== 0) return `Search failed: ${r.stderr.trim() || `rg exited ${r.code}`}`;
        // Names as the fetch results gave them: rg prefixes `./` when it searches the folder.
        const out = r.stdout.replace(/\n$/, "").replace(/^\.\//gm, "");
        return redactSecrets(cap(out, "narrow the pattern, name a file, or set max_count"));
      }),

    read: (runId, a) =>
      withRun(runId, async (run) => {
        const rel = inside(run.logsDir, a.file);
        if (rel === null || rel === ".") return `"${a.file}" is not a file in this run's logs folder.`;
        let text: string;
        try {
          text = readFileSync(join(run.logsDir, rel), "utf-8");
        } catch (e: any) {
          return `Failed to read ${rel}: ${e.message}`;
        }
        const lines = text.split("\n");
        const from = clamp(Math.floor(a.from ?? 1), 1, lines.length);
        const to = clamp(Math.floor(a.to ?? from + READ_DEFAULT_LINES - 1), from, Math.min(lines.length, from + READ_MAX_LINES - 1));
        const body = lines.slice(from - 1, to).map((l, i) => `${from + i}: ${l}`).join("\n");
        return redactSecrets(cap(`${rel}, lines ${from}-${to} of ${lines.length}:\n${body}`, "read fewer lines"));
      }),

    insights: (runId, a, signal) =>
      withRun(runId, async (run) => {
        const endTimeMs = Date.now();
        const startTimeMs = endTimeMs - (a.minutes ?? 30) * 60_000;
        try {
          const { results, truncated } = await runInsightsQuery(
            { logGroupName: a.log_group, queryString: a.query, startTimeMs, endTimeMs, abortSignal: signal },
            deps.cloudwatchClient,
          );
          const rows = results.map((row) =>
            row
              .filter((f) => f.field !== "@ptr")
              .map((f) => `${f.field}=${f.value ?? ""}`)
              .join(" | "),
          );
          const ts = new Date().toISOString().replace(/[:.]/g, "-");
          const file = `insights-${a.log_group.replace(/^\/+/, "").replace(/\//g, "-")}-${ts}.log`;
          writeFileSync(join(run.logsDir, file), rows.join("\n"), "utf-8");
          const shown = rows.slice(0, INSIGHTS_ROWS_SHOWN);
          const more = rows.length > shown.length ? `\n[... ${rows.length - shown.length} more row(s) in the file]` : "";
          const truncNote = truncated
            ? " NOTE: the query did not finish (it hit the wall-clock cap or timed out on AWS) — these are the rows so far; narrow the query or minutes."
            : "";
          return redactSecrets(
            cap(
              `Insights query on ${a.log_group} (${new Date(startTimeMs).toISOString()} to ${new Date(endTimeMs).toISOString()}) returned ${rows.length} row(s).${truncNote} Saved to file: ${file}.` +
                (shown.length ? `\n\n${shown.join("\n")}${more}` : ""),
              "add a limit or a stats clause to the query",
            ),
          );
        } catch (e: any) {
          return `Failed to run the Insights query: ${e.message}`;
        }
      }),
  };
}

/** `rel` inside `base` as a path relative to it ("." for `base` itself), or null when it leaves it. */
export function inside(base: string, rel: string): string | null {
  const root = resolve(base);
  const abs = resolve(root, rel);
  if (abs !== root && !abs.startsWith(root + sep)) return null;
  return relative(root, abs) || ".";
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function cap(text: string, hint: string): string {
  if (text.length <= OUTPUT_MAX_CHARS) return text;
  return `${text.slice(0, OUTPUT_MAX_CHARS)}\n\n[... ${text.length - OUTPUT_MAX_CHARS} more characters not shown — ${hint}]`;
}

/** rg with an argument list and no shell, in `cwd`, bounded in time and output. */
function rg(
  args: string[],
  cwd: string,
  bin: { command: string; argv0?: string },
): Promise<{ code: number | null; stdout: string; stderr: string; error?: string }> {
  return new Promise((done) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (r: { code: number | null; error?: string }) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      done({ ...r, stdout, stderr });
    };
    const child = spawn(bin.command, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
      env: { PATH: env.PATH ?? "/usr/local/bin:/usr/bin:/bin" },
      ...(bin.argv0 ? { argv0: bin.argv0 } : {}),
    });
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({ code: null, error: `timed out after ${SEARCH_TIMEOUT_MS / 1000}s — narrow the pattern or name a file` });
    }, SEARCH_TIMEOUT_MS);
    child.stdout.on("data", (d) => {
      stdout += d.toString();
      // Enough for any cap above; the rest would only be cut.
      if (stdout.length > SEARCH_MAX_BYTES) {
        child.kill("SIGKILL");
        finish({ code: 0 });
      }
    });
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("error", (e: NodeJS.ErrnoException) =>
      finish({ code: null, error: e.code === "ENOENT" ? "rg (ripgrep) is not installed on this server" : e.message }),
    );
    child.on("close", (code) => finish({ code }));
  });
}
