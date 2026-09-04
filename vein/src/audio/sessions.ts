/**
 * Dictation sessions — the capture half of the dream cycle
 * (plans/local-desktop-and-stt.md §4.8). One JSONL file per session under
 * `<dataDir>/audio/sessions/`: every final the stream produced, plus the
 * user's corrections to those finals. A dream-cycle workflow reads these
 * through `GET /audio/sessions/:id`.
 */
import { appendFile, mkdir, readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import type { SttWord } from "./stt.js";

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

export type SessionEntry =
  | {
      type: "final";
      t: string;
      index: number;
      text: string;
      words: SttWord[];
      model: string;
      hotwords: string | null;
    }
  | { type: "correction"; t: string; index: number; text: string };

export interface SessionInfo {
  id: string;
  updatedAt: string;
  bytes: number;
}

export class SessionStore {
  readonly dir: string;
  constructor(dataDir: string) {
    this.dir = join(dataDir, "audio", "sessions");
  }

  private pathOf(id: string): string {
    if (!ID_RE.test(id)) throw new Error(`invalid session id: ${JSON.stringify(id)}`);
    return join(this.dir, `${id}.jsonl`);
  }

  async append(id: string, entry: SessionEntry): Promise<void> {
    await mkdir(this.dir, { recursive: true });
    await appendFile(this.pathOf(id), JSON.stringify(entry) + "\n");
  }

  async list(): Promise<SessionInfo[]> {
    let names: string[];
    try {
      names = (await readdir(this.dir)).filter((f) => f.endsWith(".jsonl"));
    } catch {
      return [];
    }
    const out: SessionInfo[] = [];
    for (const f of names.sort()) {
      const st = await stat(join(this.dir, f));
      out.push({ id: f.slice(0, -6), updatedAt: st.mtime.toISOString(), bytes: st.size });
    }
    return out;
  }

  /** The index the next final in this session should carry. */
  async nextIndex(id: string): Promise<number> {
    const entries = (await this.get(id)) ?? [];
    return entries.reduce((n, e) => (e.type === "final" ? Math.max(n, e.index + 1) : n), 0);
  }

  /** All entries in order, or null when the session doesn't exist. */
  async get(id: string): Promise<SessionEntry[] | null> {
    let text: string;
    try {
      text = await readFile(this.pathOf(id), "utf-8");
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw e;
    }
    return text
      .split("\n")
      .filter(Boolean)
      .map((l) => JSON.parse(l) as SessionEntry);
  }
}
