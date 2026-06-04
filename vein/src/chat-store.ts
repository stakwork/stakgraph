import {
  mkdir,
  writeFile,
  appendFile,
  readFile,
  readdir,
  rm,
} from "node:fs/promises";
import { join } from "node:path";
import { tailJsonl } from "./store.js";

/**
 * Chat persistence — the server-side store that makes the AI builder a
 * detached "background job" (EVAL_SPEC §8) instead of a connection-bound
 * stream. A chat session is an append-only run with the SAME launch-detached
 * + tail-the-file lifecycle vein uses for workflow runs, PLUS a resumable
 * conversation log so a turn keeps running (and can be re-driven) after the
 * browser closes.
 *
 * Each chat lives in `<workspaceRoot>/chats/<chatId>/` with the deliberate
 * two-file split borrowed from `mcp/src/repo/session.ts`:
 *
 *   meta.json       — { id, title, status, model, createdAt, updatedAt,
 *                       currentTurn }. Cheap listing without parsing the logs.
 *   messages.jsonl  — append-only conversation (AI SDK ModelMessage objects).
 *                     The REPLAYABLE record: re-fed to the agent on the next
 *                     turn and rendered as the transcript. Whole messages only
 *                     — never deltas (keeps replay clean), lossless on disk.
 *   events.jsonl    — append-only fine-grained stream parts (text deltas, tool
 *                     calls/results, step/turn boundaries). The OBSERVABILITY
 *                     stream the SSE tail follows; never re-sent to the model.
 *
 * A chat is long-lived across many turns; the unit with launch+detach+tail
 * semantics is a TURN. Each turn's events carry `turn: N` and end with a
 * `chat.end`/`chat.error`, so the tail stops at the right boundary even when
 * replaying a multi-turn history (see `tailEvents`).
 */

export type ChatStatus = "live" | "done" | "error";

export interface ChatMeta {
  id: string;
  title?: string;
  status: ChatStatus;
  model?: string;
  createdAt: string;
  updatedAt: string;
  /** Index of the most recently launched turn (0-based). */
  currentTurn: number;
}

export type ChatEventType =
  | "text-delta"
  | "tool-input"
  | "tool-output"
  | "step.finish"
  | "chat.end"
  | "chat.error";

/** A single fine-grained event in a chat turn's observability stream. */
export interface ChatEvent {
  ts: string;
  chatId: string;
  /** Which user turn this event belongs to (0-based). */
  turn: number;
  type: ChatEventType;
  /** text-delta */
  delta?: string;
  /** tool-input / tool-output */
  toolName?: string;
  toolCallId?: string;
  input?: unknown;
  output?: unknown;
  /** chat.error */
  error?: { message: string };
}

/** A stored conversation message. Kept opaque (the AI SDK's `ModelMessage`
 *  shape) — the store only reads/writes JSON lines and never interprets it. */
export interface StoredMessage {
  role: string;
  content: unknown;
}

/** A turn is terminal once its log records a `chat.end` or `chat.error`. */
export function isChatTerminal(e: ChatEvent): boolean {
  return e.type === "chat.end" || e.type === "chat.error";
}

// ── Interface ──────────────────────────────────────────────────────────────

export interface ChatStore {
  createChat(init: { id: string; title?: string; model?: string }): Promise<ChatMeta>;
  getMeta(chatId: string): Promise<ChatMeta | null>;
  setMeta(chatId: string, patch: Partial<ChatMeta>): Promise<ChatMeta | null>;
  listChats(): Promise<ChatMeta[]>;
  appendMessages(chatId: string, messages: StoredMessage[]): Promise<void>;
  loadMessages(chatId: string): Promise<StoredMessage[]>;
  appendEvent(chatId: string, event: ChatEvent): Promise<void>;
  /** Tail a single turn's events (history → live) until that turn ends. */
  tailEvents(
    chatId: string,
    turn: number,
    opts?: { intervalMs?: number; signal?: AbortSignal },
  ): AsyncGenerator<ChatEvent>;
  deleteChat(chatId: string): Promise<void>;
}

// ── Tool-result truncation (token hygiene for long autonomous loops) ────────

/**
 * `messages.jsonl` stays lossless on disk (it's the transcript), but the copy
 * re-fed to the model each turn can balloon: a single `repo_overview` or eval
 * result is huge and the model already processed it. Truncate long strings
 * inside `role: "tool"` messages (tool RESULTS) before sending them back.
 * Conservative: only tool messages, only strings over `maxChars`, structure
 * preserved (we just shorten strings + add a marker). Mirrors
 * `session.ts:truncateToolResult`.
 */
export function truncateToolMessages(
  messages: StoredMessage[],
  maxChars = 4000,
): StoredMessage[] {
  const shorten = (s: string): string =>
    s.length > maxChars
      ? `${s.slice(0, maxChars)}\n\n[TRUNCATED: ${s.length} chars — full content is in the chat transcript]`
      : s;

  const walk = (v: unknown): unknown => {
    if (typeof v === "string") return shorten(v);
    if (Array.isArray(v)) return v.map(walk);
    if (v && typeof v === "object") {
      const out: Record<string, unknown> = {};
      for (const [k, val] of Object.entries(v)) out[k] = walk(val);
      return out;
    }
    return v;
  };

  return messages.map((m) =>
    m.role === "tool" ? { ...m, content: walk(m.content) } : m,
  );
}

// ── Filesystem implementation ──────────────────────────────────────────────

export class FileChatStore implements ChatStore {
  private workspaceRoot: string;

  constructor(workspaceRoot: string) {
    this.workspaceRoot = workspaceRoot;
  }

  private chatDir(chatId: string): string {
    return join(this.workspaceRoot, "chats", chatId);
  }

  private metaFile(chatId: string): string {
    return join(this.chatDir(chatId), "meta.json");
  }

  async createChat(init: { id: string; title?: string; model?: string }): Promise<ChatMeta> {
    const now = new Date().toISOString();
    const meta: ChatMeta = {
      id: init.id,
      ...(init.title ? { title: init.title } : {}),
      status: "live",
      ...(init.model ? { model: init.model } : {}),
      createdAt: now,
      updatedAt: now,
      currentTurn: -1,
    };
    await mkdir(this.chatDir(init.id), { recursive: true });
    await writeFile(this.metaFile(init.id), JSON.stringify(meta, null, 2), "utf-8");
    return meta;
  }

  async getMeta(chatId: string): Promise<ChatMeta | null> {
    try {
      return JSON.parse(await readFile(this.metaFile(chatId), "utf-8")) as ChatMeta;
    } catch {
      return null;
    }
  }

  async setMeta(chatId: string, patch: Partial<ChatMeta>): Promise<ChatMeta | null> {
    const current = await this.getMeta(chatId);
    if (!current) return null;
    const next: ChatMeta = { ...current, ...patch, updatedAt: new Date().toISOString() };
    await writeFile(this.metaFile(chatId), JSON.stringify(next, null, 2), "utf-8");
    return next;
  }

  async listChats(): Promise<ChatMeta[]> {
    const dir = join(this.workspaceRoot, "chats");
    let ids: string[];
    try {
      ids = await readdir(dir);
    } catch {
      return [];
    }
    const metas: ChatMeta[] = [];
    for (const id of ids) {
      const meta = await this.getMeta(id);
      if (meta) metas.push(meta);
    }
    // Newest first by updatedAt.
    return metas.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  async appendMessages(chatId: string, messages: StoredMessage[]): Promise<void> {
    if (messages.length === 0) return;
    await mkdir(this.chatDir(chatId), { recursive: true });
    const lines = messages.map((m) => JSON.stringify(m)).join("\n") + "\n";
    await appendFile(join(this.chatDir(chatId), "messages.jsonl"), lines, "utf-8");
  }

  async loadMessages(chatId: string): Promise<StoredMessage[]> {
    try {
      const raw = await readFile(join(this.chatDir(chatId), "messages.jsonl"), "utf-8");
      return raw
        .split("\n")
        .filter((l) => l.trim())
        .map((l) => JSON.parse(l) as StoredMessage);
    } catch {
      return [];
    }
  }

  async appendEvent(chatId: string, event: ChatEvent): Promise<void> {
    await mkdir(this.chatDir(chatId), { recursive: true });
    await appendFile(
      join(this.chatDir(chatId), "events.jsonl"),
      JSON.stringify(event) + "\n",
      "utf-8",
    );
  }

  async *tailEvents(
    chatId: string,
    turn: number,
    opts: { intervalMs?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<ChatEvent> {
    const file = join(this.chatDir(chatId), "events.jsonl");
    // Tail the whole file but only stop at THIS turn's terminal, and only
    // surface THIS turn's events — earlier turns' events (and terminals) are
    // replayed-through but filtered out, so a late reattach lands cleanly on
    // the requested turn regardless of how many turns precede it.
    for await (const e of tailJsonl<ChatEvent>(
      file,
      (e) => e.turn === turn && isChatTerminal(e),
      opts,
    )) {
      if (e.turn === turn) yield e;
    }
  }

  async deleteChat(chatId: string): Promise<void> {
    await rm(this.chatDir(chatId), { recursive: true, force: true });
  }
}

// ── In-memory implementation (for testing) ─────────────────────────────────

export class MemoryChatStore implements ChatStore {
  metas = new Map<string, ChatMeta>();
  messages = new Map<string, StoredMessage[]>();
  events = new Map<string, ChatEvent[]>();

  async createChat(init: { id: string; title?: string; model?: string }): Promise<ChatMeta> {
    const now = new Date().toISOString();
    const meta: ChatMeta = {
      id: init.id,
      ...(init.title ? { title: init.title } : {}),
      status: "live",
      ...(init.model ? { model: init.model } : {}),
      createdAt: now,
      updatedAt: now,
      currentTurn: -1,
    };
    this.metas.set(init.id, meta);
    return meta;
  }

  async getMeta(chatId: string): Promise<ChatMeta | null> {
    return this.metas.get(chatId) ?? null;
  }

  async setMeta(chatId: string, patch: Partial<ChatMeta>): Promise<ChatMeta | null> {
    const current = this.metas.get(chatId);
    if (!current) return null;
    const next: ChatMeta = { ...current, ...patch, updatedAt: new Date().toISOString() };
    this.metas.set(chatId, next);
    return next;
  }

  async listChats(): Promise<ChatMeta[]> {
    return [...this.metas.values()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  async appendMessages(chatId: string, messages: StoredMessage[]): Promise<void> {
    const arr = this.messages.get(chatId) ?? [];
    arr.push(...messages);
    this.messages.set(chatId, arr);
  }

  async loadMessages(chatId: string): Promise<StoredMessage[]> {
    return this.messages.get(chatId) ?? [];
  }

  async appendEvent(chatId: string, event: ChatEvent): Promise<void> {
    const arr = this.events.get(chatId) ?? [];
    arr.push(event);
    this.events.set(chatId, arr);
  }

  async *tailEvents(chatId: string, turn: number): AsyncGenerator<ChatEvent> {
    // Tests use this only for already-finished turns; replay what we have.
    for (const e of this.events.get(chatId) ?? []) {
      if (e.turn === turn) yield e;
      if (e.turn === turn && isChatTerminal(e)) return;
    }
  }

  async deleteChat(chatId: string): Promise<void> {
    this.metas.delete(chatId);
    this.messages.delete(chatId);
    this.events.delete(chatId);
  }
}

/** Generate a chat ID (timestamp + short random, sortable + collision-safe). */
export function generateChatId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
