import { Request, Response } from "express";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import path from "path";
import { db } from "../graph/neo4j.js";
import { loadStepMeta } from "../repo/session.js";
import {
  getProviderForModel,
  computeSessionCost,
  type Provider,
} from "../aieo/src/provider.js";


const SESSIONS_DIR = process.env.SESSIONS_DIR || ".sessions";

function sessionsDir(): string {
  return path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
}

function getText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    for (const item of content) {
      if (item && typeof item === "object" && (item as any).type === "text") {
        return String((item as any).text ?? "");
      }
    }
  }
  return "";
}

function parseSessionMessages(filePath: string): {
  userPromptPreview: string;
  answerPreview: string;
  toolSequence: string[];
  toolCallCount: number;
} {
  let userPromptPreview = "";
  let answerPreview = "";
  const toolSequence: string[] = [];

  try {
    const content = readFileSync(filePath, "utf-8");
    const lines = content.split("\n").filter((l) => l.trim());
    for (const line of lines) {
      try {
        const msg = JSON.parse(line) as {
          role?: string;
          content?: unknown;
          [key: string]: unknown;
        };
        const role = msg.role ?? "";
        const msgContent = msg.content;

        if (!userPromptPreview && role === "user") {
          userPromptPreview = getText(msgContent).slice(0, 200);
        }
        if (role === "assistant") {
          const t = getText(msgContent);
          if (t) answerPreview = t.slice(0, 200);
          if (Array.isArray(msgContent)) {
            for (const item of msgContent) {
              if (
                item &&
                typeof item === "object" &&
                (item as any).type === "tool-call"
              ) {
                toolSequence.push(String((item as any).toolName ?? "?"));
              }
            }
          }
        }
      } catch {
        // skip malformed lines
      }
    }
  } catch {
    // skip unreadable files
  }

  return {
    userPromptPreview,
    answerPreview,
    toolSequence,
    toolCallCount: toolSequence.length,
  };
}

function toNum(v: any): number {
  if (v == null) return 0;
  if (typeof v === "object" && typeof v.toNumber === "function")
    return v.toNumber();
  return Number(v) || 0;
}

function calcCost(
  model: string,
  providerHint: string,
  input: number,
  cacheRead: number,
  cacheWrite: number,
  output: number,
): number {
  if (!model && !providerHint) return 0;
  if (input === 0 && cacheRead === 0 && cacheWrite === 0 && output === 0)
    return 0;
  try {
    const provider = (providerHint || getProviderForModel(model)) as Provider;
    return computeSessionCost(provider, {
      input,
      cache_read: cacheRead,
      cache_write: cacheWrite,
      output,
    });
  } catch {
    return 0;
  }
}

export async function list_sessions(_req: Request, res: Response) {
  const dir = sessionsDir();

  // Try Neo4j first
  if (db) {
    try {
      const sessions = await db.list_agent_sessions();
      const runs = sessions.map((s) => {
        const id = String(s.node_key ?? s.name ?? "");
        const filePath = path.join(dir, `${id}.jsonl`);
        const {
          userPromptPreview,
          answerPreview,
          toolSequence,
          toolCallCount,
        } = existsSync(filePath)
          ? parseSessionMessages(filePath)
          : {
              userPromptPreview: "",
              answerPreview: "",
              toolSequence: [],
              toolCallCount: 0,
            };
        const startTimeMs = toNum(s.start_time);
        const input = toNum(s.input_tokens);
        const cache_read = toNum(s.cache_read_tokens);
        const cache_write = toNum(s.cache_write_tokens);
        const output = toNum(s.output_tokens);
        const total = toNum(s.total_tokens);
        const prov = String(s.provider ?? "");
        const mod = String(s.model ?? "");
        return {
          id,
          source: String(s.source ?? "unknown"),
          provider: prov,
          model: mod,
          timestamp: startTimeMs
            ? new Date(startTimeMs).toISOString()
            : new Date().toISOString(),
          duration_ms: toNum(s.duration_ms),
          token_usage: { input, cache_read, cache_write, output, total },
          cost_usd: calcCost(mod, prov, input, cache_read, cache_write, output),
          tool_sequence: toolSequence,
          tool_call_count: toolCallCount,
          user_prompt_preview: userPromptPreview,
          answer_preview: answerPreview,
        };
      });
      runs.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
      res.json(runs);
      return;
    } catch (e) {
      console.error("[sessions] Neo4j query failed, falling back to JSONL:", e);
    }
  }

  // Fallback: JSONL-only (no Neo4j)
  if (!existsSync(dir)) {
    res.json([]);
    return;
  }

  const files = readdirSync(dir).filter((f) => f.endsWith(".jsonl"));

  const runs = files.map((file) => {
    const id = file.replace(/\.jsonl$/, "");
    const fullPath = path.join(dir, file);
    const stat = statSync(fullPath);
    const { userPromptPreview, answerPreview, toolSequence, toolCallCount } =
      parseSessionMessages(fullPath);

    return {
      id,
      source: "unknown",
      provider: "",
      model: "",
      timestamp: stat.mtime.toISOString(),
      duration_ms: 0,
      token_usage: {
        input: 0,
        cache_read: 0,
        cache_write: 0,
        output: 0,
        total: 0,
      },
      cost_usd: 0,
      tool_sequence: toolSequence,
      tool_call_count: toolCallCount,
      user_prompt_preview: userPromptPreview,
      answer_preview: answerPreview,
    };
  });

  runs.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  res.json(runs);
}

export async function get_session(req: Request, res: Response) {
  const id = String(req.params.id);
  if (!id || id.includes("..") || id.includes("/")) {
    res.status(400).json({ error: "Invalid session id" });
    return;
  }

  const dir = sessionsDir();
  const filePath = path.join(dir, `${id}.jsonl`);
  if (!existsSync(filePath)) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  // Read trace from JSONL
  const content = readFileSync(filePath, "utf-8");
  const trace = content
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => {
      try {
        return JSON.parse(l);
      } catch {
        return null;
      }
    })
    .filter(Boolean);

  const { userPromptPreview, answerPreview, toolSequence, toolCallCount } =
    parseSessionMessages(filePath);

  const step_meta = loadStepMeta(id);

  if (db) {
    try {
      const s = await db.get_agent_session(id);
      if (s) {
        const startTimeMs = toNum(s.start_time);
        const input = toNum(s.input_tokens);
        const cache_read = toNum(s.cache_read_tokens);
        const cache_write = toNum(s.cache_write_tokens);
        const output = toNum(s.output_tokens);
        const total = toNum(s.total_tokens);
        const prov = String(s.provider ?? "");
        const mod = String(s.model ?? "");
        res.json({
          id,
          source: String(s.source ?? "unknown"),
          provider: prov,
          model: mod,
          timestamp: startTimeMs
            ? new Date(startTimeMs).toISOString()
            : new Date().toISOString(),
          duration_ms: toNum(s.duration_ms),
          token_usage: { input, cache_read, cache_write, output, total },
          cost_usd: calcCost(mod, prov, input, cache_read, cache_write, output),
          tool_sequence: toolSequence,
          tool_call_count: toolCallCount,
          user_prompt_preview: userPromptPreview,
          answer_preview: answerPreview,
          step_meta,
          trace,
        });
        return;
      }
    } catch (e) {
      console.error("[sessions] Neo4j get_session failed, falling back:", e);
    }
  }

  const stat = statSync(filePath);
  res.json({
    id,
    source: "unknown",
    provider: "",
    model: "",
    timestamp: stat.mtime.toISOString(),
    duration_ms: 0,
    token_usage: {
      input: 0,
      cache_read: 0,
      cache_write: 0,
      output: 0,
      total: 0,
    },
    cost_usd: 0,
    tool_sequence: toolSequence,
    tool_call_count: toolCallCount,
    user_prompt_preview: userPromptPreview,
    answer_preview: answerPreview,
    step_meta,
    trace,
  });
}

export async function session_stats(req: Request, res: Response) {
  const window = (req.query.window as string) || "all";
  const sourceFilter = (req.query.source as string) || null;
  const providerFilter = (req.query.provider as string) || null;
  const modelFilter = (req.query.model as string) || null;

  let since: number | null = null;
  if (window === "24h") since = Date.now() - 24 * 60 * 60 * 1000;
  else if (window === "7d") since = Date.now() - 7 * 24 * 60 * 60 * 1000;
  else if (window === "30d") since = Date.now() - 30 * 24 * 60 * 60 * 1000;

  if (!db) {
    res.json({
      window,
      filters: {
        source: sourceFilter,
        provider: providerFilter,
        model: modelFilter,
      },
      total_sessions: 0,
      total_cost_usd: 0,
      total_tokens: {
        input: 0,
        cache_read: 0,
        cache_write: 0,
        output: 0,
        total: 0,
      },
      by_model: [],
    });
    return;
  }

  try {
    const rows = await db.get_session_stats({
      since,
      source: sourceFilter,
      provider: providerFilter,
      model: modelFilter,
    });

    let total_cost_usd = 0;
    let total_input = 0;
    let total_cache_read = 0;
    let total_cache_write = 0;
    let total_output = 0;
    let total_all = 0;
    const byModelMap = new Map<
      string,
      {
        model: string;
        provider: string;
        sessions: number;
        cost_usd: number;
        input_tokens: number;
        cache_read_tokens: number;
        cache_write_tokens: number;
        output_tokens: number;
      }
    >();

    for (const s of rows) {
      const input = toNum(s.input_tokens);
      const cacheRead = toNum(s.cache_read_tokens);
      const cacheWrite = toNum(s.cache_write_tokens);
      const output = toNum(s.output_tokens);
      const prov = String(s.provider ?? "");
      const mod = String(s.model ?? "");
      const sessionCost = calcCost(
        mod,
        prov,
        input,
        cacheRead,
        cacheWrite,
        output,
      );
      total_cost_usd += sessionCost;
      total_input += input;
      total_cache_read += cacheRead;
      total_cache_write += cacheWrite;
      total_output += output;
      total_all += toNum(s.total_tokens);

      const key = `${prov}::${mod}`;
      const existing = byModelMap.get(key);
      if (existing) {
        existing.sessions += 1;
        existing.cost_usd += sessionCost;
        existing.input_tokens += input;
        existing.cache_read_tokens += cacheRead;
        existing.cache_write_tokens += cacheWrite;
        existing.output_tokens += output;
      } else {
        byModelMap.set(key, {
          model: mod,
          provider: prov,
          sessions: 1,
          cost_usd: sessionCost,
          input_tokens: input,
          cache_read_tokens: cacheRead,
          cache_write_tokens: cacheWrite,
          output_tokens: output,
        });
      }
    }

    res.json({
      window,
      filters: {
        source: sourceFilter,
        provider: providerFilter,
        model: modelFilter,
      },
      total_sessions: rows.length,
      total_cost_usd: parseFloat(total_cost_usd.toFixed(6)),
      total_tokens: {
        input: total_input,
        cache_read: total_cache_read,
        cache_write: total_cache_write,
        output: total_output,
        total: total_all,
      },
      by_model: Array.from(byModelMap.values()).sort(
        (a, b) => b.cost_usd - a.cost_usd,
      ),
    });
  } catch (e) {
    console.error("[sessions] stats query failed:", e);
    res.status(500).json({ error: "Failed to fetch stats" });
  }
}
