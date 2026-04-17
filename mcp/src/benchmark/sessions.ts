import { Request, Response } from "express";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import path from "path";

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

function parseSessionFile(filePath: string): {
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
        const msg = JSON.parse(line) as { role?: string; content?: unknown };
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

export async function list_sessions(_req: Request, res: Response) {
  const dir = sessionsDir();
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
      parseSessionFile(fullPath);

    return {
      id,
      source: "session" as const,
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

  const stat = statSync(filePath);
  const { userPromptPreview, answerPreview, toolSequence, toolCallCount } =
    parseSessionFile(filePath);

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

  res.json({
    id,
    source: "session" as const,
    repo: "",
    model: "",
    timestamp: stat.mtime.toISOString(),
    duration_ms: 0,
    token_usage: { input: 0, output: 0, total: 0 },
    tool_sequence: toolSequence,
    tool_call_count: toolCallCount,
    user_prompt_preview: userPromptPreview,
    answer_preview: answerPreview,
    trace,
  });
}
