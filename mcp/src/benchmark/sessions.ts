import { Request, Response } from "express";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import path from "path";
import { db } from "../graph/neo4j.js";
import {
  loadStepMeta,
  loadSearchProvenance,
  loadAnnotations,
  appendAnnotation,
  type Annotation,
  type AnnotationMarker,
} from "../repo/session.js";
import {
  getProviderForModel,
  computeSessionCost,
  type Provider,
} from "../aieo/src/provider.js";
import { addUsage, emptyUsage, normalizeUsage } from "../aieo/src/usage.js";

const SESSIONS_DIR = process.env.SESSIONS_DIR || ".sessions";

/**
 * Usage/timing recovered from the `.meta.jsonl` sidecar, for sessions with no
 * readable Neo4j node. The sidecar holds provider-reported per-step usage, so
 * this is real data — not an estimate. Shared by the list and detail endpoints
 * so the two can't drift (they did: the detail endpoint used to hardcode zeros
 * here, silently zeroing every session whose node lookup missed).
 */
function deriveFromStepMeta(id: string, mtime: Date) {
  const steps = loadStepMeta(id);
  if (steps.length === 0) {
    return {
      usage: emptyUsage(),
      duration_ms: 0,
      timestamp: mtime.toISOString(),
    };
  }
  return {
    usage: addUsage(...steps.map((step) => normalizeUsage(step.usage))),
    duration_ms:
      new Date(steps[steps.length - 1].timestamp).getTime() -
      new Date(steps[0].timestamp).getTime(),
    timestamp: steps[0].timestamp,
  };
}

function buildOrphanRun(dir: string, file: string) {
  const id = file.replace(/\.jsonl$/, "");
  const fullPath = path.join(dir, file);
  const stat = statSync(fullPath);
  const { userPromptPreview, answerPreview, toolSequence, toolCallCount, messageCount } =
    parseSessionMessages(fullPath);
  const { usage, duration_ms, timestamp } = deriveFromStepMeta(id, stat.mtime);
  return {
    id,
    // Sub-agent sessions are named `<parent>-sub-<hex>`; recover the link
    // even when the Neo4j node is missing.
    parent_session_id: id.match(/^(.+)-sub-[0-9a-f]{8}$/)?.[1] ?? "",
    source: "unknown",
    provider: "",
    model: "",
    repo: "",
    timestamp,
    duration_ms,
    token_usage: {
      input: usage.input,
      cache_read: usage.cache_read,
      cache_write: usage.cache_write,
      output: usage.output,
      total: usage.total,
    },
    cost_usd: 0,
    status: "success",
    error_message: "",
    tool_sequence: toolSequence,
    tool_call_count: toolCallCount,
    message_count: messageCount,
    estimated_tokens: 0,
    user_prompt_preview: userPromptPreview,
    answer_preview: answerPreview,
  };
}
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
  messageCount: number;
} {
  let userPromptPreview = "";
  let answerPreview = "";
  const toolSequence: string[] = [];
  let messageCount = 0;

  try {
    const content = readFileSync(filePath, "utf-8");
    const lines = content.split("\n").filter((l) => l.trim());
    messageCount = lines.length;
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
    messageCount,
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
    }, model);
  } catch {
    return 0;
  }
}

// Summary row for an AgentSession Neo4j node — the shape used by the
// /sessions list and by the `children` field on the detail endpoint.
function buildRunFromNode(s: any, dir: string) {
  const id = String(s.node_key ?? s.name ?? "");
  const filePath = path.join(dir, `${id}.jsonl`);
  const {
    userPromptPreview,
    answerPreview,
    toolSequence,
    toolCallCount,
    messageCount,
  } = existsSync(filePath)
    ? parseSessionMessages(filePath)
    : {
        userPromptPreview: "",
        answerPreview: "",
        toolSequence: [],
        // No local transcript file for this node (e.g. sessions written by
        // hive's agent-logs webhook) — fall back to the counts it sent
        // directly on the AgentSession node.
        toolCallCount: toNum(s.tool_call_count),
        messageCount: toNum(s.message_count),
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
    parent_session_id: String(s.parent_session_id ?? ""),
    source: String(s.source ?? "unknown"),
    repo: String(s.repo ?? ""),
    provider: prov,
    model: mod,
    timestamp: startTimeMs
      ? new Date(startTimeMs).toISOString()
      : new Date().toISOString(),
    duration_ms: toNum(s.duration_ms),
    token_usage: { input, cache_read, cache_write, output, total },
    // Heuristic (text-length) estimate from hive's agent-logs webhook —
    // kept out of token_usage/cost_usd since it isn't provider-reported.
    estimated_tokens: toNum(s.estimated_tokens),
    cost_usd: calcCost(mod, prov, input, cache_read, cache_write, output),
    status: String(s.status ?? "success"),
    error_message: String(s.error_message ?? ""),
    tool_sequence: toolSequence,
    tool_call_count: toolCallCount,
    message_count: messageCount,
    user_prompt_preview: userPromptPreview,
    answer_preview: answerPreview,
  };
}

/** True for primary conversation JSONL files (skips every sidecar variant). */
function isSessionFile(file: string): boolean {
  return (
    file.endsWith(".jsonl") &&
    !file.endsWith(".meta.jsonl") &&
    !file.endsWith(".provenance.jsonl") &&
    !file.endsWith(".annotations.jsonl")
  );
}

/**
 * Summary rows for every descendant of a session, any depth, sorted oldest
 * first. Merges Neo4j nodes (name-prefix + parent_session_id match) with
 * orphan `<id>-sub-*` transcript files that have no node.
 */
async function collectDescendantRuns(dir: string, id: string) {
  const byId = new Map<string, ReturnType<typeof buildRunFromNode>>();
  if (db) {
    try {
      for (const s of await db.list_descendant_agent_sessions(id)) {
        const run = buildRunFromNode(s, dir);
        if (run.id) byId.set(run.id, run);
      }
    } catch (e) {
      console.error("[sessions] Neo4j descendant query failed:", e);
    }
  }
  if (existsSync(dir)) {
    for (const file of readdirSync(dir)) {
      if (!isSessionFile(file)) continue;
      const fid = file.replace(/\.jsonl$/, "");
      if (!fid.startsWith(`${id}-sub-`) || byId.has(fid)) continue;
      byId.set(fid, buildOrphanRun(dir, file));
    }
  }
  return Array.from(byId.values()).sort((a, b) =>
    a.timestamp.localeCompare(b.timestamp),
  );
}

/** Stamp each run with the number of direct children present in `runs`. */
function withChildCounts<T extends { id: string; parent_session_id: string }>(
  runs: T[],
): (T & { child_count: number })[] {
  const counts = new Map<string, number>();
  for (const r of runs) {
    if (r.parent_session_id) {
      counts.set(r.parent_session_id, (counts.get(r.parent_session_id) ?? 0) + 1);
    }
  }
  return runs.map((r) => ({ ...r, child_count: counts.get(r.id) ?? 0 }));
}

export async function list_sessions(_req: Request, res: Response) {
  const dir = sessionsDir();

  // Try Neo4j first
  if (db) {
    try {
      const sessions = await db.list_agent_sessions();
      const runs = sessions.map((s) => buildRunFromNode(s, dir));
      const neo4jIds = new Set(runs.map((r) => r.id));
      const isGhost = (r: (typeof runs)[number]) =>
        r.source === "unknown" &&
        r.token_usage.total === 0 &&
        r.duration_ms === 0;
      const liveRuns = runs.filter((r) => !isGhost(r));
      if (existsSync(dir)) {
        for (const file of readdirSync(dir)) {
          if (!isSessionFile(file)) continue;
          const id = file.replace(/\.jsonl$/, "");
          if (neo4jIds.has(id)) continue;
          liveRuns.push(buildOrphanRun(dir, file));
        }
      }
      liveRuns.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
      res.json(withChildCounts(liveRuns));
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

  const files = readdirSync(dir).filter(isSessionFile);

  const runs = files.map((file) => buildOrphanRun(dir, file));

  runs.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  res.json(withChildCounts(runs));
}

/**
 * Full session object (metadata + trace + sidecars) for the detail endpoint
 * and for `?recursive=true` descendants. Returns null when the session exists
 * nowhere (no transcript file AND no Neo4j node). A node without a local
 * transcript (e.g. hive's agent-logs webhook) resolves with an empty trace
 * instead of failing.
 */
async function buildFullSession(
  dir: string,
  id: string,
): Promise<Record<string, unknown> | null> {
  const filePath = path.join(dir, `${id}.jsonl`);
  const hasFile = existsSync(filePath);

  let trace: unknown[] = [];
  let userPromptPreview = "";
  let answerPreview = "";
  let toolSequence: string[] = [];
  let toolCallCount = 0;
  if (hasFile) {
    const content = readFileSync(filePath, "utf-8");
    trace = content
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
    ({ userPromptPreview, answerPreview, toolSequence, toolCallCount } =
      parseSessionMessages(filePath));
  }

  const step_meta = loadStepMeta(id);
  const search_provenance = loadSearchProvenance(id);
  const annotations = loadAnnotations(id);

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
        return {
          id,
          parent_session_id: String(s.parent_session_id ?? ""),
          source: String(s.source ?? "unknown"),
          repo: String(s.repo ?? ""),
          provider: prov,
          model: mod,
          timestamp: startTimeMs
            ? new Date(startTimeMs).toISOString()
            : new Date().toISOString(),
          duration_ms: toNum(s.duration_ms),
          token_usage: { input, cache_read, cache_write, output, total },
          cost_usd: calcCost(mod, prov, input, cache_read, cache_write, output),
          status: String(s.status ?? "success"),
          error_message: String(s.error_message ?? ""),
          tool_sequence: toolSequence,
          tool_call_count: toolCallCount,
          user_prompt_preview: userPromptPreview,
          answer_preview: answerPreview,
          step_meta,
          search_provenance,
          annotations,
          trace,
        };
      }
    } catch (e) {
      console.error("[sessions] Neo4j get_session failed, falling back:", e);
    }
  }

  if (!hasFile) return null;

  const stat = statSync(filePath);
  // No Neo4j node — recover usage/timing from the step-meta sidecar rather
  // than reporting zeros. cost_usd stays 0: pricing needs the model/provider,
  // which only the node carries.
  const { usage, duration_ms, timestamp } = deriveFromStepMeta(id, stat.mtime);
  return {
    id,
    parent_session_id: id.match(/^(.+)-sub-[0-9a-f]{8}$/)?.[1] ?? "",
    source: "unknown",
    repo: "",
    provider: "",
    model: "",
    timestamp,
    duration_ms,
    token_usage: {
      input: usage.input,
      cache_read: usage.cache_read,
      cache_write: usage.cache_write,
      output: usage.output,
      total: usage.total,
    },
    cost_usd: 0,
    status: "success",
    error_message: "",
    tool_sequence: toolSequence,
    tool_call_count: toolCallCount,
    user_prompt_preview: userPromptPreview,
    answer_preview: answerPreview,
    step_meta,
    search_provenance,
    annotations,
    trace,
  };
}

export async function get_session(req: Request, res: Response) {
  const id = String(req.params.id);
  if (!id || id.includes("..") || id.includes("/")) {
    res.status(400).json({ error: "Invalid session id" });
    return;
  }

  const dir = sessionsDir();
  const session = await buildFullSession(dir, id);
  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  // Sub-agent runs spawned by this session (see graph_sub_agent). `children`
  // is always present as summary rows; `?recursive=true` additionally inlines
  // every descendant (any depth) as a full session object, flat — each carries
  // its own parent_session_id so callers can rebuild the tree.
  const descendants = await collectDescendantRuns(dir, id);
  session.children = descendants.filter((r) => r.parent_session_id === id);

  const recursive = ["true", "1"].includes(String(req.query.recursive ?? ""));
  if (recursive && descendants.length > 0) {
    session.descendants = (
      await Promise.all(descendants.map((d) => buildFullSession(dir, d.id)))
    ).filter(Boolean);
  }

  res.json(session);
}

export async function add_annotation(req: Request, res: Response) {
  const id = String(req.params.id);
  if (!id || id.includes("..") || id.includes("/")) {
    res.status(400).json({ error: "Invalid session id" });
    return;
  }

  const VALID_MARKERS = new Set([
    "inefficient", "bad_search", "good_result", "loop", "wrong_tool", "wasted_tokens",
  ]);
  const VALID_TARGETS = new Set(["session", "tool_call"]);

  const body = req.body as Record<string, unknown>;
  const target = body.target ? String(body.target) : "";
  const marker = body.marker ? String(body.marker) : "";

  if (!VALID_TARGETS.has(target)) {
    res.status(400).json({ error: "Invalid target" });
    return;
  }
  if (!VALID_MARKERS.has(marker)) {
    res.status(400).json({ error: "Invalid marker" });
    return;
  }

  const annotation: Annotation = {
    ts: new Date().toISOString(),
    author: body.author ? String(body.author).slice(0, 64) : undefined,
    target: target as "session" | "tool_call",
    target_id: body.target_id ? String(body.target_id).slice(0, 256) : undefined,
    marker: marker as AnnotationMarker,
    note: body.note ? String(body.note).slice(0, 1000) : undefined,
  };

  appendAnnotation(id, annotation);
  res.status(201).json(annotation);
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
  else if (window === "3m") since = Date.now() - 90 * 24 * 60 * 60 * 1000;
  else if (window === "1y") since = Date.now() - 365 * 24 * 60 * 60 * 1000;

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
    let total_success = 0;
    let total_error = 0;
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
      if (String(s.status ?? "success") === "error") total_error++;
      else total_success++;

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
      by_status: { success: total_success, error: total_error },
      by_model: Array.from(byModelMap.values()).sort(
        (a, b) => b.cost_usd - a.cost_usd,
      ),
    });
  } catch (e) {
    console.error("[sessions] stats query failed:", e);
    res.status(500).json({ error: "Failed to fetch stats" });
  }
}
