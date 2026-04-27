import { Request, Response } from "express";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import path from "path";
import { db } from "../graph/neo4j.js";
import { loadStepMeta } from "../repo/session.js";

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
        return {
          id,
          source: String(s.source ?? "unknown"),
          repo: "",
          model: String(s.model ?? ""),
          timestamp: startTimeMs
            ? new Date(startTimeMs).toISOString()
            : new Date().toISOString(),
          duration_ms: toNum(s.duration_ms),
          token_usage: {
            input: toNum(s.input_tokens),
            output: toNum(s.output_tokens),
            total: toNum(s.total_tokens),
          },
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
      repo: "",
      model: "",
      timestamp: stat.mtime.toISOString(),
      duration_ms: 0,
      token_usage: { input: 0, output: 0, total: 0 },
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
        res.json({
          id,
          source: String(s.source ?? "unknown"),
          repo: "",
          model: String(s.model ?? ""),
          timestamp: startTimeMs
            ? new Date(startTimeMs).toISOString()
            : new Date().toISOString(),
          duration_ms: toNum(s.duration_ms),
          token_usage: {
            input: toNum(s.input_tokens),
            output: toNum(s.output_tokens),
            total: toNum(s.total_tokens),
          },
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
    repo: "",
    model: "",
    timestamp: stat.mtime.toISOString(),
    duration_ms: 0,
    token_usage: { input: 0, output: 0, total: 0 },
    tool_sequence: toolSequence,
    tool_call_count: toolCallCount,
    user_prompt_preview: userPromptPreview,
    answer_preview: answerPreview,
    step_meta,
    trace,
  });
}
