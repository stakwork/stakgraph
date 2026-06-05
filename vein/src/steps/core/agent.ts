import { z } from "zod";
import { defineStep } from "../../core.js";
import { usageFromResult, computeCost } from "../../pricing.js";
import { spawn } from "node:child_process";
import { existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import { join, resolve, dirname, isAbsolute, sep } from "node:path";

/**
 * Core AGENT step: a general tool-using agent loop (Vercel AI SDK
 * `ToolLoopAgent`). It explores a working directory (`cwd`) with a built-in set
 * of general-purpose tools and either:
 *   - calls a terminal `final_answer` tool and returns its text (`finalAnswer`
 *     mode — a free-form output contract the caller defines), or
 *   - produces a STRUCTURED object matching a JSON Schema (`schema` mode, via the
 *     SDK's `Output.object`), or
 *   - just returns the final assistant text (neither set).
 *
 * The tools are general (work on any codebase / working dir), so the step is
 * domain-agnostic: point it at a `cwd`, give it a `system` + `prompt`, and
 * optionally restrict the toolset with `toolFilter`. Anything domain-specific
 * (e.g. how to frame a particular workspace) lives in the CALLER's prompts.
 *
 * Built-in tools: repo_overview, file_summary, fulltext_search, bash,
 * str_replace_based_edit_tool (view/create/str_replace/insert files, sandboxed
 * to cwd), and (anthropic only) web_search. `final_answer` is added
 * automatically in finalAnswer mode and is always available regardless of
 * `toolFilter`.
 *
 * Provider-direct via the AI SDK (anthropic | openai), lazy-loaded. Needs the
 * provider's key in env (ANTHROPIC_API_KEY / OPENAI_API_KEY) and `git` + `rg` on
 * PATH for the repo tools. Output: { result, object?, steps, usage, cost }
 * (+ `messages`, the full session, only when `returnMessages` is set — it's huge
 * and persisted per step, so it's off by default). `usage` is the aggregated
 * token counts across the whole agent loop
 * and `cost` is its dollar cost at the provider's rates (see ../../pricing.ts).
 */

// ── tool helpers (pure: take cwd as an argument) ───────────────────────────────

/** Spawn a child, capture stdout with a timeout + output cap. Exit 1 with no
 *  stderr → "No matches found" (grep/rg/find idiom). */
function capture(
  child: ReturnType<typeof spawn>,
  timeoutMs: number,
  maxBytes: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let done = false;
    const cap = (s: string) =>
      s.length > maxBytes ? s.slice(0, maxBytes) + "\n\n[... output truncated ...]" : s;
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
    child.stdout?.on("data", (d) => {
      stdout += d.toString();
      if (stdout.length > maxBytes) {
        child.kill("SIGKILL");
        finish(() => resolve(cap(stdout)));
      }
    });
    child.stderr?.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) =>
      finish(() => {
        if (code === 0) resolve(cap(stdout));
        else if (code === 1 && !stderr) resolve(cap(stdout) || "No matches found");
        else reject(new Error(`Command failed (${code}): ${stderr || stdout || "Unknown error"}`));
      }),
    );
    child.on("error", (err) => finish(() => reject(err)));
  });
}

/** Run a program with explicit args (NO shell) — safe for untrusted args like a
 *  search query (no quoting/escaping/injection). */
const runCmd = (cmd: string, args: string[], cwd: string, timeoutMs = 10000, maxBytes = 10000) =>
  capture(spawn(cmd, args, { cwd, stdio: ["ignore", "pipe", "pipe"] }), timeoutMs, maxBytes);

/** Run an arbitrary shell command string — the `bash` tool needs a full shell. */
const runShell = (command: string, cwd: string, timeoutMs = 15000, maxBytes = 10000) =>
  capture(spawn(command, { cwd, shell: true, stdio: ["ignore", "pipe", "pipe"] }), timeoutMs, maxBytes);

/** Immediate subdirs of `cwd` that are git repos. */
function listRepos(cwd: string): string[] {
  if (!existsSync(cwd)) return [];
  return readdirSync(cwd, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(join(cwd, e.name, ".git")))
    .map((e) => e.name)
    .sort();
}

/** Directories that are build output / dependencies / generated noise: shown in
 *  the tree (so the agent knows they exist) but NEVER expanded — their contents
 *  would be high-token, low-signal. */
const NOISE_DIRS = new Set([
  "node_modules", ".git", "dist", "build", "out", "target", "vendor",
  "coverage", "__pycache__", ".venv", "venv", ".next", ".nuxt", ".svelte-kit",
  ".turbo", ".cache", "migrations", ".gradle", "Pods", ".terraform", "__snapshots__",
]);

/** Default budgets for the adaptive repo map (see `repoTree`). */
const REPO_MAP_MAX_LINES = 200; // deepen until a depth busts this, then step back
const REPO_MAP_MAX_DEPTH = 8; // never go deeper than this regardless of budget
const REPO_MAP_MAX_CHARS = 12000; // final hard backstop

interface TreeNode {
  dirs: Map<string, TreeNode>;
  files: Set<string>;
}

/** Build a full directory tree from a flat path list. */
function buildTree(files: string[]): TreeNode {
  const root: TreeNode = { dirs: new Map(), files: new Set() };
  for (const f of files) {
    const parts = f.split("/").filter(Boolean);
    if (!parts.length) continue;
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const seg = parts[i];
      let next = node.dirs.get(seg);
      if (!next) {
        next = { dirs: new Map(), files: new Set() };
        node.dirs.set(seg, next);
      }
      node = next;
    }
    node.files.add(parts[parts.length - 1]);
  }
  return root;
}

/** Render the tree to `maxDepth` levels (root entries = level 1). A `NOISE_DIRS`
 *  directory is shown but never expanded; a directory beyond `maxDepth` is shown
 *  collapsed (name only). */
function renderTreeAtDepth(root: TreeNode, maxDepth: number): string {
  const lines: string[] = [];
  const walk = (node: TreeNode, level: number) => {
    for (const name of [...node.dirs.keys()].sort()) {
      lines.push(`${"  ".repeat(level - 1)}${name}/`);
      if (NOISE_DIRS.has(name)) continue; // collapse noise dirs
      if (level < maxDepth) walk(node.dirs.get(name)!, level + 1);
    }
    for (const name of [...node.files].sort()) {
      lines.push(`${"  ".repeat(level - 1)}${name}`);
    }
  };
  walk(root, 1);
  return lines.join("\n");
}

/**
 * Adaptive directory tree: always show the root (depth 1, every top-level dir +
 * file), then iteratively deepen — try depth 2, 3, … — keeping the deepest
 * rendering that stays under `maxLines`, and stepping back one when a depth busts
 * the budget. Noise dirs (build/deps/generated) are collapsed at every depth.
 * Pure + git-free so it's unit-testable. Returns the chosen text + depth.
 */
export function repoTree(
  files: string[],
  opts: { maxLines?: number; maxDepth?: number } = {},
): { text: string; depth: number } {
  const maxLines = opts.maxLines ?? REPO_MAP_MAX_LINES;
  const maxDepth = opts.maxDepth ?? REPO_MAP_MAX_DEPTH;
  const root = buildTree(files);

  let best = renderTreeAtDepth(root, 1); // root always shown, even if over budget
  let depth = 1;
  for (let d = 2; d <= maxDepth; d++) {
    const rendered = renderTreeAtDepth(root, d);
    if (rendered.split("\n").length > maxLines) break; // too many → keep previous depth
    best = rendered;
    depth = d;
  }
  return { text: best, depth };
}

/** A high-level map of the working dir: `git ls-files` across every git-repo
 *  subdir (prefixed), or — if `cwd` is itself a single repo / plain dir — its
 *  own tracked files. Rendered as an adaptive-depth tree (see `repoTree`), so
 *  even a huge monorepo stays within a token budget. No `tree` binary needed. */
async function getRepoMap(cwd: string): Promise<string> {
  const repos = listRepos(cwd);
  const files: string[] = [];
  if (repos.length) {
    for (const repo of repos) {
      try {
        const listing = await runCmd("git", ["ls-files"], join(cwd, repo), 10000, 500000);
        for (const f of listing.split("\n").filter(Boolean)) files.push(`${repo}/${f}`);
      } catch {
        /* skip a repo that fails to list */
      }
    }
  } else {
    try {
      const listing = await runCmd("git", ["ls-files"], cwd, 10000, 500000);
      for (const f of listing.split("\n").filter(Boolean)) files.push(f);
    } catch {
      /* not a git repo */
    }
  }
  if (!files.length) return "No tracked files found";

  const { text, depth } = repoTree(files);
  const header = `(${files.length} tracked files; tree shown to depth ${depth}; build/dependency/migration dirs collapsed)\n`;
  const out = header + text;
  return out.length > REPO_MAP_MAX_CHARS ? out.slice(0, REPO_MAP_MAX_CHARS) + "\n\n[... output truncated ...]" : out;
}

/** Max chars returned for a file summary before truncation. */
const FILE_SUMMARY_MAX_CHARS = 12000;

/** Is an executable named `bin` on PATH? Unix-style — the agent's tools already
 *  assume a unix env (git/rg/bash). Used to skip a tool whose CLI isn't present
 *  (e.g. `file_summary` when `stakgraph` isn't installed). */
function isOnPath(bin: string): boolean {
  const dirs = (process.env.PATH ?? "").split(":");
  return dirs.some((d) => d && existsSync(join(d, bin)));
}

/** Structural summary of a file via the `stakgraph` AST CLI: `stakgraph "<file>"`
 *  in `cwd`. For code it returns imports + every function/class signature with
 *  line ranges + call edges; for config/data files it returns the content. Only
 *  used when `stakgraph` is on PATH (see `isOnPath`). */
async function stakgraphSummary(filePath: string, cwd: string): Promise<string> {
  if (!existsSync(join(cwd, filePath))) return "File not found";
  try {
    return await runShell(`stakgraph "${filePath}"`, cwd, 15000, FILE_SUMMARY_MAX_CHARS);
  } catch (e) {
    return `Error summarizing file: ${(e as Error).message}`;
  }
}

/** Ripgrep across cwd, grouped by file with hit counts + line numbers. */
async function fulltextSearch(query: string, cwd: string): Promise<string> {
  if (!existsSync(cwd)) return "Working directory does not exist";
  let raw: string;
  try {
    raw = await runCmd("rg", ["--glob", "!dist", "--ignore-file", ".gitignore", "-n", query, "./"], cwd, 5000);
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

/** A neutral listing of the working dir's immediate entries, prepended to the
 *  prompt so the agent knows the layout without a first tool call. Any
 *  interpretation of that layout belongs in the caller's `system`/`prompt`. */
function buildPreamble(cwd: string): string {
  if (!existsSync(cwd)) return "";
  const entries = readdirSync(cwd, { withFileTypes: true })
    .filter((e) => !e.name.startsWith("."))
    .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
    .sort();
  if (!entries.length) return "";
  return `Working directory (${cwd}) contains:\n` + entries.map((e) => `- ${e}`).join("\n");
}

// ── file editing tool (str_replace_based_edit_tool) ────────────────────────────

/** Max chars returned by a `view` before truncation. */
const FILE_VIEW_MAX_CHARS = 200_000;

/** The Anthropic text-editor tool's input shape (also used by the generic
 *  fallback for non-anthropic providers). All commands operate on a path that
 *  MUST resolve inside `cwd`. */
export interface TextEditInput {
  command: "view" | "create" | "str_replace" | "insert";
  path: string;
  file_text?: string;
  insert_line?: number;
  new_str?: string;
  insert_text?: string;
  old_str?: string;
  view_range?: number[];
}

/** Resolve a tool-supplied path against `cwd` and refuse anything that escapes
 *  it (directory-traversal / absolute-path guard). */
function resolveInCwd(p: string, cwd: string): string {
  const target = resolve(isAbsolute(p) ? p : join(cwd, p));
  const root = resolve(cwd);
  if (target !== root && !target.startsWith(root + sep)) {
    throw new Error(`path "${p}" escapes the working directory`);
  }
  return target;
}

/**
 * Pure handler for the str_replace-based text editor tool: view / create /
 * str_replace / insert, sandboxed to `cwd`. Mirrors Anthropic's tool contract
 * (1-indexed line numbers, exactly-one-match str_replace, `insert_line` 0 =
 * top-of-file) so it backs both the provider-defined anthropic tool and the
 * generic fallback. Returns a human-readable string (errors as `Error: …`).
 */
export function textEdit(input: TextEditInput, cwd: string): string {
  let target: string;
  try {
    target = resolveInCwd(input.path, cwd);
  } catch (e) {
    return `Error: ${(e as Error).message}`;
  }

  switch (input.command) {
    case "view": {
      if (!existsSync(target)) return "Error: File not found";
      if (statSync(target).isDirectory()) {
        const entries = readdirSync(target, { withFileTypes: true })
          .filter((e) => !e.name.startsWith("."))
          .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
          .sort();
        return entries.length ? entries.join("\n") : "(empty directory)";
      }
      const lines = readFileSync(target, "utf-8").split("\n");
      let start = 1;
      let end = lines.length;
      if (Array.isArray(input.view_range) && input.view_range.length === 2) {
        start = Math.max(1, input.view_range[0]);
        end = input.view_range[1] === -1 ? lines.length : input.view_range[1];
      }
      const out = lines
        .slice(start - 1, end)
        .map((l, i) => `${start + i}: ${l}`)
        .join("\n");
      return out.length > FILE_VIEW_MAX_CHARS
        ? out.slice(0, FILE_VIEW_MAX_CHARS) + "\n\n[... output truncated ...]"
        : out;
    }

    case "create": {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, input.file_text ?? "");
      return `Successfully created ${input.path}`;
    }

    case "str_replace": {
      if (!existsSync(target)) return "Error: File not found";
      const content = readFileSync(target, "utf-8");
      const old = input.old_str ?? "";
      const count = old ? content.split(old).length - 1 : 0;
      if (count === 0)
        return "Error: No match found for replacement. Please check your text and try again.";
      if (count > 1)
        return `Error: Found ${count} matches for replacement text. Please provide more context to make a unique match.`;
      writeFileSync(target, content.replace(old, input.new_str ?? ""));
      return "Successfully replaced text at exactly one location.";
    }

    case "insert": {
      if (!existsSync(target)) return "Error: File not found";
      const lines = readFileSync(target, "utf-8").split("\n");
      const at = input.insert_line ?? 0;
      if (at < 0 || at > lines.length)
        return `Error: insert_line ${at} is out of range (0-${lines.length})`;
      lines.splice(at, 0, input.insert_text ?? "");
      writeFileSync(target, lines.join("\n"));
      return `Successfully inserted text after line ${at}`;
    }

    default:
      return `Error: unknown command "${(input as { command?: string }).command}"`;
  }
}

export default defineStep({
  type: "agent",
  description:
    "General tool-using agent (AI SDK ToolLoopAgent). Explores AND edits a working dir (cwd) with built-in tools (repo_overview, fulltext_search, bash, str_replace_based_edit_tool for viewing/creating/editing files, + anthropic web_search, + file_summary when the `stakgraph` AST CLI is on PATH) and returns either a final_answer (set `finalAnswer` to its tool description), a STRUCTURED object (set `schema` to a JSON Schema → Output.object), or the final text. Config: cwd, system, prompt, finalAnswer?, schema?, toolFilter? (subset of tool names; empty = all), model?, provider? (anthropic|openai), maxSteps (default 40), returnMessages? (default false — the full session is huge + persisted per step). Needs the provider key in env + git/rg on PATH. Output: { result, object?, steps, usage, cost } (+ messages when returnMessages).",
  input: z.object({
    cwd: z.string().describe("working directory the tools operate in"),
    system: z.string().describe("system prompt / agent persona"),
    prompt: z.string().describe("the user task driving the agent"),
    finalAnswer: z
      .string()
      .optional()
      .describe("if set, a `final_answer` tool is added with this description; its output is the result. Omit when using `schema`."),
    schema: z
      .any()
      .optional()
      .describe("if set, a JSON Schema for STRUCTURED output (Output.object); the step returns the object. Mutually exclusive with finalAnswer."),
    toolFilter: z
      .array(z.string())
      .default([])
      .describe("subset of built-in tool names to enable; empty = all. (final_answer is always available in finalAnswer mode.)"),
    model: z.string().optional(),
    provider: z.string().optional(),
    maxSteps: z.number().int().positive().default(40),
    returnMessages: z
      .boolean()
      .default(false)
      .describe(
        "include the FULL tool-loop session (`messages`) in the output. Off by default — the session is huge and the runner persists every step's output, so returning it bloats events.jsonl/run.json and buries `result`. Turn on only for a fork/sub-agent that needs the transcript.",
      ),
  }),
  output: z.any(),
  async run(cfg) {
    const { ToolLoopAgent, Output, tool, stepCountIs, hasToolCall, jsonSchema } = await import("ai");

    const provider = cfg.provider ?? process.env["VEIN_LLM_PROVIDER"] ?? "anthropic";
    const modelName = cfg.model ?? process.env["VEIN_LLM_MODEL"];

    // Resolve the model + (anthropic-only) provider-defined web_search tool +
    // provider options. For anthropic we enable EPHEMERAL PROMPT CACHING at the
    // call level (the provider auto-inserts the cache breakpoints across the
    // static prefix — system + tool schemas — that the loop resends every step).
    // Big win for multi-step agents (and the future fork: the shared prefix is a
    // cache hit across all forks). Same pattern as aieo's getProviderOptions.
    let model: any;
    let webSearchTool: any;
    let textEditorTool: any;
    let providerOptions: any;
    switch (provider) {
      case "anthropic": {
        const { anthropic } = await import("@ai-sdk/anthropic");
        model = anthropic(modelName ?? "claude-sonnet-4-6");
        webSearchTool = anthropic.tools.webSearch_20260209({ maxUses: 3 });
        // Provider-defined text editor (the model is specially trained on its
        // schema); we supply the execute that performs the edit inside cfg.cwd.
        textEditorTool = anthropic.tools.textEditor_20250728({
          execute: async (input: TextEditInput) => textEdit(input, cfg.cwd),
        });
        providerOptions = { anthropic: { cacheControl: { type: "ephemeral" } } };
        break;
      }
      case "openai": {
        const { openai } = await import("@ai-sdk/openai");
        model = openai(modelName ?? "gpt-4o");
        break;
      }
      default:
        throw new Error(`Unknown LLM provider: "${provider}". Supported: anthropic, openai`);
    }

    // Built-in tools (operate on cfg.cwd). `inputSchema` cast to any to stop the
    // SDK's tool() from deeply inferring the zod type (TS2589 in strict builds).
    const allTools: Record<string, any> = {
      repo_overview: tool({
        description:
          "Get a high-level map of the codebase under the working dir (tracked files across any sub-repos). Use it to understand the layout and find where functionality lives.",
        inputSchema: z.object({}) as any,
        execute: async () => {
          try {
            return await getRepoMap(cfg.cwd);
          } catch {
            return "Could not retrieve repository map";
          }
        },
      }),
      fulltext_search: tool({
        description:
          'Search the codebase for a term (e.g. "process.env."). Returns files with the term, occurrence counts, and line numbers. Use it to find env vars, integrations, and how central a symbol is.',
        inputSchema: z.object({ query: z.string().describe("The term to search for") }) as any,
        execute: async ({ query }: { query: string }) => {
          try {
            return await fulltextSearch(query, cfg.cwd);
          } catch (e) {
            return `Search failed: ${e}`;
          }
        },
      }),
      bash: tool({
        description:
          "Execute a bash command inside the working dir. Use for listing dirs, reading files (cat/head), inspecting manifests/lockfiles/docker files, and anything the other tools don't cover.",
        inputSchema: z.object({ command: z.string().describe("The bash command to execute") }) as any,
        execute: async ({ command }: { command: string }) => {
          try {
            if (!existsSync(cfg.cwd)) return "Working directory does not exist";
            return await runShell(command, cfg.cwd);
          } catch (e) {
            return `Command execution failed: ${e}`;
          }
        },
      }),
    };
    // file_summary is the stakgraph AST CLI — only offer it when stakgraph is on
    // PATH; otherwise the agent reads files via `bash` (cat/head).
    if (isOnPath("stakgraph")) {
      allTools.file_summary = tool({
        description:
          "Get a STRUCTURAL summary of a file (relative to the working dir): for code, its imports + every function/class signature with line ranges + call edges; for config/data files, the content. Backed by the stakgraph AST parser — prefer it over cat for understanding code files.",
        inputSchema: z.object({
          file_path: z.string().describe("Path to the file, relative to the working dir"),
        }) as any,
        execute: async ({ file_path }: { file_path: string }) => {
          try {
            return await stakgraphSummary(file_path, cfg.cwd);
          } catch {
            return "Bad file path";
          }
        },
      });
    }
    if (webSearchTool) allTools.web_search = webSearchTool;

    // File editing (str_replace_based_edit_tool): view/create/str_replace/insert,
    // sandboxed to cfg.cwd. For anthropic use the provider-defined tool (the
    // model is specially trained on it); other providers get an identical generic
    // tool. The key MUST be `str_replace_based_edit_tool` for the anthropic case.
    allTools.str_replace_based_edit_tool = textEditorTool
      ? textEditorTool
      : tool({
          description:
            "View and edit text files (sandboxed to the working dir). Commands: " +
            '`view` (path, optional view_range [start,end]), `create` (path, file_text), ' +
            "`str_replace` (path, old_str must match EXACTLY once, new_str), " +
            "`insert` (path, insert_line — 0 = top of file, insert_text).",
          inputSchema: z.object({
            command: z.enum(["view", "create", "str_replace", "insert"]),
            path: z.string(),
            file_text: z.string().optional(),
            insert_line: z.number().int().optional(),
            new_str: z.string().optional(),
            insert_text: z.string().optional(),
            old_str: z.string().optional(),
            view_range: z.array(z.number().int()).optional(),
          }) as any,
          execute: async (input: TextEditInput) => textEdit(input, cfg.cwd),
        });

    // Apply toolFilter (empty = all). Unknown names are ignored.
    const filter = cfg.toolFilter ?? [];
    const tools: Record<string, any> = {};
    for (const [name, t] of Object.entries(allTools)) {
      if (!filter.length || filter.includes(name)) tools[name] = t;
    }

    // Output mode: schema (structured) vs finalAnswer (terminal tool) vs text.
    const useSchema = cfg.schema != null;
    if (useSchema && cfg.finalAnswer) {
      throw new Error("agent: set EITHER `schema` (structured output) OR `finalAnswer` (terminal tool), not both.");
    }
    if (!useSchema && cfg.finalAnswer) {
      tools.final_answer = tool({
        description: cfg.finalAnswer,
        inputSchema: z.object({ answer: z.string() }) as any,
        execute: async ({ answer }: { answer: string }) => answer,
      });
    }

    const stopWhen = !useSchema && cfg.finalAnswer
      ? [hasToolCall("final_answer"), stepCountIs(cfg.maxSteps)]
      : [stepCountIs(cfg.maxSteps)];

    const agent = new ToolLoopAgent({
      model,
      instructions: cfg.system,
      tools,
      stopWhen,
      ...(providerOptions ? { providerOptions } : {}),
      ...(useSchema ? { output: Output.object({ schema: jsonSchema(cfg.schema) }) } : {}),
      onStepFinish: (sf: any) => {
        if (!Array.isArray(sf.content)) return;
        for (const c of sf.content) {
          if (c.type === "tool-call" && c.toolName !== "final_answer") {
            console.log("[agent] TOOL CALL:", c.toolName, ":", JSON.stringify(c.input));
          }
        }
      },
    });

    const preamble = buildPreamble(cfg.cwd);
    const startTime = Date.now();
    const res = await agent.generate({
      prompt: preamble ? `${preamble}\n\n${cfg.prompt}` : cfg.prompt,
    });

    const steps = res.steps ?? [];
    // The full session is HUGE and the runner persists every step's output, so we
    // only include it when explicitly asked (a future fork/sub-agent). Off by
    // default keeps the explore step's persisted output to `{ result, steps, … }`.
    const messages = res.response?.messages ?? [];
    const maybeMessages = cfg.returnMessages ? { messages } : {};

    // Token usage + cost across the WHOLE agent loop (totalUsage aggregates every
    // step; fall back to the final-step usage). `provider` drives the rate table.
    const usage = usageFromResult(res.totalUsage ?? res.usage);
    const cost = computeCost(provider, usage);
    console.log(
      `[agent] tokens in:${usage.inputTokens} cacheRead:${usage.cacheReadTokens} cacheWrite:${usage.cacheWriteTokens} out:${usage.outputTokens} → $${cost.toFixed(4)}`,
    );

    // Structured mode: return the typed object.
    if (useSchema) {
      console.log(`[agent] completed in ${Date.now() - startTime}ms (${steps.length} steps, structured)`);
      return { result: res.text, object: res.output, steps: steps.length, ...maybeMessages, usage, cost };
    }

    // finalAnswer / text mode: extract the final_answer tool output, else last text.
    let final = "";
    let lastText = "";
    for (const step of steps) {
      for (const item of step.content) {
        if (item.type === "text" && item.text?.trim()) lastText = item.text.trim();
      }
    }
    if (cfg.finalAnswer) {
      for (const step of [...steps].reverse()) {
        const fa = step.content.find(
          (c: any) => c.type === "tool-result" && c.toolName === "final_answer",
        );
        if (fa) {
          final = String((fa as { output?: unknown }).output ?? "");
          break;
        }
      }
      if (!final && lastText) {
        console.warn("[agent] No final_answer tool call; using last reasoning text.");
        final = `${lastText}\n\n(Note: model did not invoke final_answer; using last reasoning text.)`;
      }
    } else {
      final = res.text || lastText;
    }

    console.log(`[agent] completed in ${Date.now() - startTime}ms (${steps.length} steps)`);
    return { result: final, steps: steps.length, ...maybeMessages, usage, cost };
  },
});
