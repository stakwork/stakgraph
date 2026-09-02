import { tool } from "ai";
import { z } from "zod";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import { fetch } from "undici";
import { AuditorContext } from "./types.js";
import { VerdictSchema } from "./schema.js";

const execAsync = promisify(exec);

export const END_OF_AUDIT = "[END_OF_AUDIT]";

const MUTATING: Array<[RegExp, string]> = [
  [/\bgit\b[^\n|;&]*\b(commit|push|reset|checkout|merge|rebase|add|clean|stash)\b/, "git write command"],
  [/\brm\b/, "rm"],
  [/\bmv\b/, "mv"],
  [/\bcp\b/, "cp"],
  [/\b(mkdir|rmdir|touch|truncate|tee|chmod|chown|ln)\b/, "filesystem mutation"],
  [/\bsed\b[^\n|;&]*-i\b/, "in-place sed"],
  [/\b(npm|yarn|pnpm|pip|pip3|cargo|go|apt|apt-get|brew)\b[^\n|;&]*\b(install|add|i|remove|uninstall|update|upgrade|get)\b/, "package install"],
  [/>>?/, "output redirection to a file"],
  [/\bkill\b|\bpkill\b/, "process kill"],
];

function commandRejection(cmd: string): string | undefined {
  for (const [re, label] of MUTATING) {
    if (re.test(cmd)) {
      return `run_command rejected: ${label} is not allowed — the Auditor only inspects, it never mutates. Use read-only commands (ls, cat, curl, timing).`;
    }
  }
  return undefined;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

export function getAuditorTools(ctx: AuditorContext) {
  const { deck, collector, browser } = ctx;

  const read_task = tool({
    description:
      "Read the TASK you are auditing: its prompt, description, and the diff of the change. The diff bounds the scope — judge only what this task changed.",
    inputSchema: z.object({}),
    execute: async () => ({
      prompt: deck.task.prompt,
      description: deck.task.description,
      diff: deck.diff,
    }),
  });

  const read_feature_context = tool({
    description:
      "Read the broader feature context. This is BACKGROUND ONLY — other tasks own other parts of the feature. Judge the TASK's responsibility, not the whole feature.",
    inputSchema: z.object({}),
    execute: async () => ({ featureContext: deck.featureContext }),
  });

  const read_map = tool({
    description:
      "Read the map to the running app: its base URL and any notes. This is where you exercise the app.",
    inputSchema: z.object({}),
    execute: async () => ({ appUrl: deck.map.appUrl, notes: deck.map.notes }),
  });

  const browser_open = tool({
    description:
      "Open a URL (absolute, or a path relative to the app base URL) in a real browser. Returns the load result.",
    inputSchema: z.object({
      url: z.string().describe("Absolute URL or a path relative to the app base URL."),
    }),
    execute: async ({ url }: { url: string }) => browser.open(url),
  });

  const browser_snapshot = tool({
    description:
      "Snapshot the current page: interactive elements as @eN refs plus visible text. Refs reset on navigation — re-snapshot after clicks that navigate.",
    inputSchema: z.object({}),
    execute: async () => browser.snapshot(),
  });

  const browser_click = tool({
    description: "Click an interactive element by its @eN ref from the latest snapshot.",
    inputSchema: z.object({ ref: z.string().describe("An @eN ref, e.g. 'e3'.") }),
    execute: async ({ ref }: { ref: string }) => browser.click(ref),
  });

  const browser_fill = tool({
    description: "Fill an input/textarea (by @eN ref) with a value.",
    inputSchema: z.object({
      ref: z.string().describe("An @eN ref from the latest snapshot."),
      value: z.string().describe("The text to type into the field."),
    }),
    execute: async ({ ref, value }: { ref: string; value: string }) => browser.fill(ref, value),
  });

  const browser_press = tool({
    description: "Press a keyboard key on the page (e.g. 'Enter', 'Tab', 'Escape').",
    inputSchema: z.object({ key: z.string().describe("Key name, e.g. 'Enter'.") }),
    execute: async ({ key }: { key: string }) => browser.press(key),
  });

  const browser_observe = tool({
    description:
      "Observe what actually happened on the page: the current snapshot plus any console errors, page errors, failed requests, and HTTP 4xx/5xx responses seen since the last observation. Use this to catch runtime failures.",
    inputSchema: z.object({
      instruction: z
        .string()
        .describe("What you are trying to observe or verify right now."),
    }),
    execute: async ({ instruction }: { instruction: string }) => {
      const snapshot = await browser.snapshot();
      const events = browser.drainSummary();
      return { instruction, snapshot, events };
    },
  });

  const browser_screenshot = tool({
    description:
      "Take a full-page screenshot of the current page. Returns the saved file path (usable as proof when captured).",
    inputSchema: z.object({}),
    execute: async () => {
      const path = await browser.screenshot(`audit-${Date.now()}.png`);
      return { path: path ?? "screenshot unavailable" };
    },
  });

  const http_request = tool({
    description:
      "Make a timed HTTP request against the running app or its API. Returns status, elapsed ms, response headers, and a snippet of the body.",
    inputSchema: z.object({
      url: z.string().describe("Absolute URL to request."),
      method: z.string().optional().describe("HTTP method (default GET)."),
      headers: z.record(z.string(), z.string()).optional(),
      body: z.string().optional().describe("Raw request body, if any."),
    }),
    execute: async ({
      url,
      method,
      headers,
      body,
    }: {
      url: string;
      method?: string;
      headers?: Record<string, string>;
      body?: string;
    }) => {
      const start = Date.now();
      try {
        const resp = await fetch(url, {
          method: method ?? "GET",
          headers,
          body: body ?? undefined,
        });
        const ms = Date.now() - start;
        const text = await resp.text();
        const respHeaders: Record<string, string> = {};
        resp.headers.forEach((v, k) => {
          respHeaders[k] = v;
        });
        return {
          status: resp.status,
          ms,
          headers: respHeaders,
          bodySnippet: text.slice(0, 2000),
        };
      } catch (err: any) {
        return {
          status: 0,
          ms: Date.now() - start,
          headers: {},
          bodySnippet: `request failed: ${err?.message ?? String(err)}`,
        };
      }
    },
  });

  const read_logs = tool({
    description:
      "Fetch recent application logs if a log source is configured for this environment. Returns the logs, or a note that no log source is available.",
    inputSchema: z.object({}),
    execute: async () => {
      const cmd = process.env.AUDIT_LOGS_CMD;
      if (!cmd) return { logs: "no log source" };
      try {
        const { stdout, stderr } = await execAsync(cmd, {
          timeout: 30000,
          maxBuffer: 1024 * 1024,
        });
        return { logs: (stdout || "") + (stderr ? `\n[stderr]\n${stderr}` : "") };
      } catch (err: any) {
        return { logs: `log fetch failed: ${err?.message ?? String(err)}` };
      }
    },
  });

  const run_command = tool({
    description:
      "Run a READ-ONLY shell command to inspect the running system (ls, cat, curl, timing, etc.). Mutating commands are rejected. Returns stdout and stderr.",
    inputSchema: z.object({
      cmd: z.string().describe("The read-only shell command to run."),
    }),
    execute: async ({ cmd }: { cmd: string }) => {
      const rejection = commandRejection(cmd);
      if (rejection) return { rejected: true, message: rejection };
      try {
        const { stdout, stderr } = await execAsync(cmd, {
          timeout: 60000,
          maxBuffer: 1024 * 1024,
        });
        return { stdout: stdout ?? "", stderr: stderr ?? "" };
      } catch (err: any) {
        return {
          stdout: err?.stdout ?? "",
          stderr: err?.stderr ?? (err?.message ?? String(err)),
        };
      }
    },
  });

  const sample = tool({
    description:
      "Call a URL n times and measure timing. Returns count, median ms, p95 ms, and the individual samples. Use for performance/timing claims.",
    inputSchema: z.object({
      url: z.string().describe("Absolute URL to sample."),
      n: z.number().describe("Number of requests to make."),
    }),
    execute: async ({ url, n }: { url: string; n: number }) => {
      const count = Math.max(1, Math.min(50, Math.floor(n)));
      const samples: number[] = [];
      for (let i = 0; i < count; i++) {
        const start = Date.now();
        try {
          const resp = await fetch(url, { method: "GET" });
          await resp.arrayBuffer();
        } catch {
          /* still record the elapsed time of the failed attempt */
        }
        samples.push(Date.now() - start);
      }
      const sorted = [...samples].sort((a, b) => a - b);
      const medianMs = percentile(sorted, 50);
      const p95Ms = percentile(sorted, 95);
      return { count, medianMs, p95Ms, samples };
    },
  });

  const capture = tool({
    description:
      "Capture an evidence record (a screenshot path, an HTTP response, a log line, a number, an error). Returns an evidence id you cite in a claim's proof[]. Only captured evidence can justify a works verdict.",
    inputSchema: z.object({
      kind: z
        .string()
        .describe("Kind of evidence, e.g. 'screenshot', 'http', 'log', 'timing', 'error'."),
      summary: z.string().describe("A short human-readable description of what this proves."),
      data: z.any().optional().describe("The underlying evidence data."),
    }),
    execute: async ({
      kind,
      summary,
      data,
    }: {
      kind: string;
      summary: string;
      data?: unknown;
    }) => {
      const id = collector.push(kind, summary, data);
      return { id };
    },
  });

  const submit_verdict = tool({
    description:
      "Submit the final audit verdict and END the audit. Each claim's verdict must be backed by captured evidence ids in proof[]. This is the terminal tool.",
    inputSchema: VerdictSchema,
    execute: async (input: z.infer<typeof VerdictSchema>) => {
      collector.verdict = {
        overall: input.overall,
        claims: input.claims,
        observations: input.observations,
        summary: input.summary,
      };
      return `Verdict recorded. ${END_OF_AUDIT}`;
    },
  });

  return {
    read_task,
    read_feature_context,
    read_map,
    browser_open,
    browser_snapshot,
    browser_click,
    browser_fill,
    browser_press,
    browser_observe,
    browser_screenshot,
    http_request,
    read_logs,
    run_command,
    sample,
    capture,
    submit_verdict,
  };
}

export type AuditorTools = ReturnType<typeof getAuditorTools>;
