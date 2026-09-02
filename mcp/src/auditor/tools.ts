import { tool } from "ai";
import { z } from "zod";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import { fetch } from "undici";
import { AuditorContext, ClaimVerdict } from "./types.js";
import { VerdictSchema } from "./schema.js";

const execAsync = promisify(exec);

export const END_OF_AUDIT = "[END_OF_AUDIT]";

function resolveUrl(appUrl: string, url: string): string {
  try {
    return new URL(url).toString();
  } catch {
    try {
      return new URL(url, appUrl).toString();
    } catch {
      return url;
    }
  }
}

function browserError(op: string, err: any) {
  return {
    error: `${op} failed: ${err?.message ?? String(err)}`,
    hint: "Wait briefly and retry, or use browser_observe to locate elements and browser_act to interact before concluding. Do not give up after one failure.",
  };
}

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
    execute: async ({ url }: { url: string }) => {
      try {
        return await browser.open(resolveUrl(deck.map.appUrl, url));
      } catch (err: any) {
        return browserError("browser_open", err);
      }
    },
  });

  const browser_act = tool({
    description:
      "Perform a natural-language action on the current page, e.g. 'Click the sign in button' or 'Type hello into the search input'. Keep actions atomic and specific.",
    inputSchema: z.object({
      action: z.string().describe("The atomic action to perform in natural language."),
    }),
    execute: async ({ action }: { action: string }) => {
      try {
        return await browser.act(action);
      } catch (err: any) {
        return browserError("browser_act", err);
      }
    },
  });

  const browser_observe = tool({
    description:
      "Observe the current page in natural language to find candidate actionable elements (e.g. 'find the login button'). Use before acting when you are unsure what is on the page.",
    inputSchema: z.object({
      instruction: z
        .string()
        .describe("What you are looking for on the page right now."),
    }),
    execute: async ({ instruction }: { instruction: string }) => {
      try {
        return await browser.observe(instruction);
      } catch (err: any) {
        return browserError("browser_observe", err);
      }
    },
  });

  const browser_extract = tool({
    description:
      "Extract a structured/text observation from the current page in natural language (e.g. 'the visible error message' or 'the list of rows'). Captures a dom evidence record and returns its id you can cite in proof[].",
    inputSchema: z.object({
      instruction: z
        .string()
        .describe("What to read/extract from the page."),
    }),
    execute: async ({ instruction }: { instruction: string }) => {
      try {
        const { extraction } = await browser.extract(instruction);
        const data =
          typeof extraction === "string"
            ? extraction
            : JSON.stringify(extraction);
        const id = collector.push("dom", instruction, data, true);
        return { id, extraction };
      } catch (err: any) {
        return browserError("browser_extract", err);
      }
    },
  });

  const browser_screenshot = tool({
    description:
      "Take a screenshot of the current page. Captures a screenshot evidence record and returns its id — cite that id in a claim's proof[]. The raw image is stored as evidence, not returned here.",
    inputSchema: z.object({}),
    execute: async () => {
      try {
        const base64 = await browser.screenshot();
        const id = collector.push("screenshot", "screenshot of current page", base64, true);
        return { id, note: `Screenshot captured as evidence ${id}. Cite ${id} in proof[].` };
      } catch (err: any) {
        return browserError("browser_screenshot", err);
      }
    },
  });

  const browser_current_url = tool({
    description:
      "Return the current page URL of the browser. Useful to confirm redirects and navigation outcomes. Captures a dom evidence record and returns its id plus the url.",
    inputSchema: z.object({}),
    execute: async () => {
      try {
        const url = await browser.currentUrl();
        const id = collector.push("dom", `current url: ${url}`, url, true);
        return { id, url };
      } catch (err: any) {
        return browserError("browser_current_url", err);
      }
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
        const bodySnippet = text.slice(0, 2000);
        const id = collector.push(
          "http",
          `HTTP ${method ?? "GET"} ${url} -> ${resp.status} in ${ms}ms`,
          JSON.stringify({ status: resp.status, ms, bodySnippet }),
          true,
        );
        return {
          id,
          status: resp.status,
          ms,
          headers: respHeaders,
          bodySnippet,
        };
      } catch (err: any) {
        const ms = Date.now() - start;
        const message = err?.message ?? String(err);
        const id = collector.push(
          "http",
          `HTTP ${method ?? "GET"} ${url} -> request failed in ${ms}ms`,
          JSON.stringify({ status: 0, ms, bodySnippet: `request failed: ${message}` }),
          true,
        );
        return {
          id,
          status: 0,
          ms,
          headers: {},
          bodySnippet: `request failed: ${message}`,
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
        const logs = (stdout || "") + (stderr ? `\n[stderr]\n${stderr}` : "");
        const id = collector.push("log", "recent application logs", logs, true);
        return { id, logs };
      } catch (err: any) {
        const logs = `log fetch failed: ${err?.message ?? String(err)}`;
        const id = collector.push("log", "log fetch failed", logs, true);
        return { id, logs };
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
      const id = collector.push(
        "timing",
        `sampled ${url} n=${count} median=${medianMs}ms p95=${p95Ms}ms`,
        JSON.stringify({ count, medianMs, p95Ms, samples }),
        true,
      );
      return { id, count, medianMs, p95Ms, samples };
    },
  });

  const capture = tool({
    description:
      "Record a free-form NOTE for the trail. A note is NOT proof and cannot back a works verdict — only the probe tools (http_request, sample, read_logs, browser_extract, browser_screenshot, browser_current_url) produce evidence that backs works. Use this for context you observed, not to justify a verdict.",
    inputSchema: z.object({
      summary: z.string().describe("A short human-readable description of what you observed."),
      data: z.string().optional().describe("The underlying note text."),
    }),
    execute: async ({
      summary,
      data,
    }: {
      summary: string;
      data?: string;
    }) => {
      const id = collector.push("note", summary, data);
      return { id, note: "Recorded as a NOTE — not proof; cannot back a works verdict." };
    },
  });

  const submit_verdict = tool({
    description:
      "Submit the final audit verdict and END the audit. A claim may be marked works ONLY if its proof[] cites at least one probe-captured evidence id (from http_request, sample, read_logs, browser_extract, browser_screenshot, or browser_current_url); notes do not count. A works claim without such proof is downgraded to unknown, and overall is downgraded to match. This is the terminal tool.",
    inputSchema: VerdictSchema,
    execute: async (input: z.infer<typeof VerdictSchema>) => {
      const strong = collector.strongIds;
      const notes: string[] = [];

      const claims: ClaimVerdict[] = input.claims.map((c): ClaimVerdict => {
        if (c.verdict !== "works") return c;
        const backed = c.proof.filter((id) => strong.has(id));
        if (backed.length === 0) {
          notes.push(
            `Guard: claim "${c.claim}" was submitted as works with no probe-captured proof; downgraded to unknown.`,
          );
          return {
            ...c,
            verdict: "unknown",
            proof: backed,
            reasoning: `${c.reasoning} [auditor guard: no captured proof backed this works claim]`,
          };
        }
        return { ...c, proof: backed };
      });

      const hasBroken = claims.some((c) => c.verdict === "broken");
      const allWorks = claims.length > 0 && claims.every((c) => c.verdict === "works");

      let overall = input.overall;
      if (overall === "works" && !allWorks) {
        overall = hasBroken ? "broken" : "unknown";
        notes.push(
          `Guard: overall downgraded from works to ${overall} because not every claim is backed as works.`,
        );
      }

      collector.verdict = {
        overall,
        claims,
        observations: notes.length > 0 ? [...input.observations, ...notes] : input.observations,
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
    browser_act,
    browser_observe,
    browser_extract,
    browser_screenshot,
    browser_current_url,
    http_request,
    read_logs,
    run_command,
    sample,
    capture,
    submit_verdict,
  };
}

export type AuditorTools = ReturnType<typeof getAuditorTools>;
