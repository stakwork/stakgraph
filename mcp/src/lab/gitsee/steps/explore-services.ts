import { z, defineStep } from "vein";
import { generateText, tool, hasToolCall, stepCountIs, type Tool } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { spawn } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { join, basename } from "node:path";

/**
 * SELF-CONTAINED port of `mcp/src/gitsee` "services" mode (the setup-profiler
 * agent). The goal of this lab experiment is to delete `src/gitsee` entirely,
 * so this step deliberately imports NOTHING from the existing codebase — only
 * `vein`, the third-party AI SDK (`ai` + `@ai-sdk/anthropic`), and Node
 * builtins. Every tool (repo map, file summary, fulltext search) and the whole
 * agent loop is inlined here.
 *
 * It runs an agentic tool loop over a LOCAL clone (`repoPath`) and emits a
 * pm2.config.js + docker-compose.yml pair via the `final_answer` tool. The
 * prompts (`system`, `finalAnswer`) are the experiment surface — supplied by
 * the calling workflow's `params` block, NOT baked in — so they can be swept
 * by a future `gitsee-optimize` loop (mirroring `concepts-optimize`).
 *
 * LLM wiring is intentionally minimal & provider-direct: Anthropic via
 * `ANTHROPIC_API_KEY` from env, model name from config. This drops aieo's
 * gateway routing / multi-provider / cost-usage normalization in exchange for
 * full self-containment.
 *
 * Output: { result, steps } — `result` is the raw two-file string (same shape
 * the old `explore(..., "services")` returned).
 */

// ── inline tool helpers (ported from gitsee/agent/tools.ts, no internal deps) ──

/** Run a command, capturing stdout, with a timeout + output cap. Resolves with
 *  whatever was captured (exit 1 → "No matches found", like ripgrep). */
function runCmd(
  cmd: string,
  args: string[],
  cwd: string,
  timeoutMs = 10000,
  maxBytes = 10000,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let done = false;
    const finish = (fn: () => void) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      fn();
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(() => reject(new Error(`Command timed out after ${timeoutMs}ms`)));
    }, timeoutMs);
    child.stdout.on("data", (d) => {
      stdout += d.toString();
      if (stdout.length > maxBytes) {
        child.kill("SIGKILL");
        finish(() =>
          resolve(stdout.slice(0, maxBytes) + "\n\n[... output truncated ...]"),
        );
      }
    });
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => {
      finish(() => {
        if (code === 0) resolve(stdout);
        else if (code === 1) resolve("No matches found");
        else reject(new Error(`Command failed (${code}): ${stderr}`));
      });
    });
    child.on("error", (err) => finish(() => reject(err)));
  });
}

/** A high-level repo map: list tracked files (`git ls-files`) and render an
 *  indented tree truncated to `maxDepth`. Built in JS so it needs no `tree`
 *  binary (more portable than the original shell pipeline). */
async function getRepoMap(repoPath: string, maxDepth = 3): Promise<string> {
  if (!repoPath || !existsSync(repoPath)) return "Repository not cloned yet";
  let listing: string;
  try {
    listing = await runCmd("git", ["ls-files"], repoPath, 10000, 200000);
  } catch (e) {
    return `Error getting repo map: ${(e as Error).message}`;
  }
  const files = listing.split("\n").filter(Boolean);
  if (!files.length) return "No tracked files found";

  // Build a nested dir tree, pruning anything deeper than maxDepth.
  type Node = { dirs: Map<string, Node>; files: Set<string> };
  const root: Node = { dirs: new Map(), files: new Set() };
  for (const f of files) {
    const parts = f.split("/");
    let node = root;
    for (let i = 0; i < parts.length - 1 && i < maxDepth; i++) {
      const seg = parts[i];
      if (!node.dirs.has(seg)) node.dirs.set(seg, { dirs: new Map(), files: new Set() });
      node = node.dirs.get(seg)!;
    }
    if (parts.length - 1 < maxDepth) node.files.add(parts[parts.length - 1]);
  }
  const lines: string[] = [];
  const render = (node: Node, depth: number) => {
    for (const name of [...node.dirs.keys()].sort()) {
      lines.push(`${"  ".repeat(depth)}${name}/`);
      render(node.dirs.get(name)!, depth + 1);
    }
    for (const name of [...node.files].sort()) {
      lines.push(`${"  ".repeat(depth)}${name}`);
    }
  };
  render(root, 0);
  const out = lines.join("\n");
  return out.length > 10000
    ? out.slice(0, 10000) + "\n\n[... output truncated ...]"
    : out;
}

/** First N lines of a file, each line capped at 200 chars (minified-file safe). */
function getFileSummary(filePath: string, repoPath: string, linesLimit: number): string {
  if (!repoPath) return "No repository path provided";
  const full = join(repoPath, filePath);
  if (!existsSync(full)) return "File not found";
  try {
    return readFileSync(full, "utf-8")
      .split("\n")
      .slice(0, linesLimit || 40)
      .map((l) => (l.length > 200 ? l.slice(0, 200) + "..." : l))
      .join("\n");
  } catch (e) {
    return `Error reading file: ${(e as Error).message}`;
  }
}

/** Ripgrep across the repo, grouped by file with hit counts + line numbers. */
async function fulltextSearch(query: string, repoPath: string): Promise<string> {
  if (!repoPath || !existsSync(repoPath)) return "Repository not cloned yet";
  let raw: string;
  try {
    raw = await runCmd(
      "rg",
      ["--glob", "!dist", "--ignore-file", ".gitignore", "-n", query, "./"],
      repoPath,
      5000,
    );
  } catch (e) {
    return `Error searching: ${(e as Error).message}`;
  }
  if (raw === "No matches found") return `No matches found for "${query}"`;
  const byFile: Record<string, number[]> = {};
  for (const line of raw.split("\n").filter(Boolean)) {
    const m = line.match(/^([^:]+):(\d+):/);
    if (m) (byFile[m[1]] ??= []).push(parseInt(m[2], 10));
  }
  const out = Object.entries(byFile)
    .sort((a, b) => b[1].length - a[1].length)
    .map(([file, ls]) => `${ls.length}\t${file} (lines: ${ls.join(", ")})`)
    .join("\n");
  return out.length > 10000 ? out.slice(0, 10000) + "\n\n[... truncated ...]" : out;
}

export default defineStep({
  type: "gitsee/explore-services",
  description:
    "Self-contained setup-profiler agent (ported from gitsee 'services' mode, no internal deps). Agentically explores a local repo clone (repo_overview, file_summary, fulltext_search) and emits a pm2.config.js + docker-compose.yml pair via final_answer. Config: repoPath, prompt (user task), system + finalAnswer (the prompt experiment surface, from params), fileLines (default 100), model (anthropic model id, default claude-sonnet-4-6), maxSteps (loop cap, default 40). Needs ANTHROPIC_API_KEY in env and `git`/`rg` on PATH. Output: { result, steps }.",
  input: z.object({
    repoPath: z.string().describe("local path to the cloned repo"),
    prompt: z.string().describe("the user task / question driving exploration"),
    system: z.string().describe("system prompt (the EXPLORER persona) — from params"),
    finalAnswer: z
      .string()
      .describe("final_answer tool description / output spec — from params; MY_REPO_NAME is substituted with the repo dir name"),
    fileLines: z.number().int().positive().default(100),
    model: z.string().default("claude-sonnet-4-6"),
    maxSteps: z.number().int().positive().default(40),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error("gitsee/explore-services requires ANTHROPIC_API_KEY in env.");

    const repoName = basename(cfg.repoPath) || "my-repo";
    const fad = cfg.finalAnswer.replaceAll("MY_REPO_NAME", repoName);
    const model = createAnthropic({ apiKey })(cfg.model);

    // `inputSchema` is cast to `any` to stop `ai`'s `tool()` from deeply
    // inferring the zod type (TS2589 under mcp's tsconfig); `execute` params
    // are explicitly annotated below instead.
    const tools: Record<string, Tool> = {
      repo_overview: tool({
        description:
          "Get a high-level view of the codebase architecture and structure. Use this to understand the project layout and identify where specific functionality might be located.",
        inputSchema: z.object({}) as any,
        execute: async () => {
          try {
            return await getRepoMap(cfg.repoPath);
          } catch {
            return "Could not retrieve repository map";
          }
        },
      }),
      file_summary: tool({
        description:
          "Get a summary of what a specific file contains and its role in the codebase. Only the first N lines are returned. Call with a hypothesis about the file's purpose.",
        inputSchema: z.object({
          file_path: z.string().describe("Path to the file to summarize"),
          hypothesis: z.string().describe("What you think this file might contain, based on its name/location"),
        }) as any,
        execute: async ({ file_path }: { file_path: string }) => {
          try {
            return getFileSummary(file_path, cfg.repoPath, cfg.fileLines);
          } catch {
            return "Bad file path";
          }
        },
      }),
      fulltext_search: tool({
        description:
          'Search the entire codebase for a specific term (e.g. "process.env."). Returns files where the term is found, occurrence counts, and line numbers. Use this to find env vars, integrations, and how central a symbol is.',
        inputSchema: z.object({
          query: z.string().describe("The term to search for"),
        }) as any,
        execute: async ({ query }: { query: string }) => {
          try {
            return await fulltextSearch(query, cfg.repoPath);
          } catch (e) {
            return `Search failed: ${e}`;
          }
        },
      }),
      final_answer: tool({
        description: fad,
        inputSchema: z.object({ answer: z.string() }) as any,
        execute: async ({ answer }: { answer: string }) => answer,
      }),
    };

    const startTime = Date.now();
    const { steps } = await generateText({
      model,
      tools,
      prompt: cfg.prompt,
      system: cfg.system,
      stopWhen: [hasToolCall("final_answer"), stepCountIs(cfg.maxSteps)],
      onStepFinish: (sf) => {
        if (!Array.isArray(sf.content)) return;
        for (const c of sf.content) {
          if (c.type === "tool-call" && c.toolName !== "final_answer") {
            console.log("[gitsee] TOOL CALL:", c.toolName, ":", JSON.stringify(c.input));
          }
        }
      },
    });

    // Extract the final_answer tool output; fall back to the last reasoning text.
    let final = "";
    let lastText = "";
    for (const step of steps) {
      for (const item of step.content) {
        if (item.type === "text" && item.text?.trim()) lastText = item.text.trim();
      }
    }
    for (const step of [...steps].reverse()) {
      const fa = step.content.find(
        (c) => c.type === "tool-result" && (c as { toolName?: string }).toolName === "final_answer",
      );
      if (fa) {
        final = String((fa as { output?: unknown }).output ?? "");
        break;
      }
    }
    if (!final && lastText) {
      console.warn("[gitsee] No final_answer tool call; using last reasoning text.");
      final = `${lastText}\n\n(Note: model did not invoke final_answer; using last reasoning text.)`;
    }

    console.log(`[gitsee] explore-services completed in ${Date.now() - startTime}ms (${steps.length} steps)`);
    return { result: final, steps: steps.length };
  },
});
