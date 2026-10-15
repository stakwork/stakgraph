import { createHmac, randomBytes } from "node:crypto";
import { brief, hiveApi, podCall, type PodCtx } from "./_shared.js";

/**
 * Shared by pod/config and pod/config-set (a seeded helper; the registry
 * skips a leading `_`). A workspace's POOL CONFIG is what hive keeps for it
 * and pushes to the pool: `services` (pm2.config.js is generated from them),
 * the workspace-wide env vars, `docker-compose.yml` and `Dockerfile`. Every
 * new pod is built from it; a pod is free to change its own copy, and what
 * it changes dies with it. pod/config reads both and diffs them; pod/config-set
 * writes a change back to hive, which recreates every unused pod of the
 * workspace. strut plans/pod-config.md.
 *
 * Hive serves env values MASKED to a key caller (`ab******yz`), and takes a
 * masked value written back as "keep the stored one" — so a list read here
 * and written back whole loses nothing. But the stored pm2.config.js and the
 * pod's own carry the values in clear: their `env` blocks are masked here
 * before anything leaves the step.
 */

// ── names ─────────────────────────────────────────────────────────────────

/** The files of a pool config this reads and writes, by hive's names. */
export const FILES = ["pm2.config.js", "docker-compose.yml", "Dockerfile"] as const;
export type FileName = (typeof FILES)[number];

/** A file's name as hive keys it: either compose name is `docker-compose.yml`,
 *  either pm2 name `pm2.config.js`. Null for anything else (devcontainer.json:
 *  nothing here edits it). */
export function poolName(name: string): FileName | null {
  if (name === "docker-compose.yml" || name === "docker-compose.yaml") return "docker-compose.yml";
  if (name === "pm2.config.js" || name === "pm2.config.cjs") return "pm2.config.js";
  if (name === "Dockerfile") return "Dockerfile";
  return null;
}

/** The pm2 env vars that are a service's CONFIG, not its env: hive generates
 *  them from the service's `port` and `scripts` (and shows them unmasked). */
export const CONFIG_VARS: Record<string, string> = {
  PORT: "port",
  INSTALL_COMMAND: "scripts.install",
  BUILD_COMMAND: "scripts.build",
  TEST_COMMAND: "scripts.test",
  E2E_TEST_COMMAND: "scripts.e2eTest",
  PRE_START_COMMAND: "scripts.preStart",
  POST_START_COMMAND: "scripts.postStart",
  REBUILD_COMMAND: "scripts.rebuild",
  RESET_COMMAND: "scripts.reset",
};

// ── masking ───────────────────────────────────────────────────────────────

/** Hive's mask (its src/lib/env-mask.ts): the first and last two characters
 *  around stars, four or fewer characters all stars. Idempotent: a masked
 *  value masks to itself, so masking what hive already masked changes nothing. */
export function mask(value: string): string {
  if (value.length <= 4) return "****";
  return `${value.slice(0, 2)}${"*".repeat(Math.max(value.length - 4, 4))}${value.slice(-2)}`;
}

/** What hive's mask can produce, and a file view's tagged form of it. */
export function looksMasked(value: string): boolean {
  return /^\*{4}$/.test(value) || /^[\s\S]{2}\*{4,}[\s\S]{2}(#[0-9a-f]{6})?$/.test(value);
}

/**
 * A tagger for one read: hive's mask plus a short keyed hash of the value,
 * `ab******yz#3fa9c1`. Two values that mask alike but differ — a URL whose
 * port changed — get different tags, so a diff shows the change; equal values
 * get equal tags. The key is random per read and never kept, so a tag says
 * nothing outside the one output it is in.
 */
export function tagger(): (value: string) => string {
  const key = randomBytes(32);
  return (value) => `${mask(value)}#${createHmac("sha256", key).update(value).digest("hex").slice(0, 6)}`;
}

function skipString(text: string, at: number): number {
  const quote = text[at];
  let i = at + 1;
  while (i < text.length && text[i] !== quote) i += text[i] === "\\" ? 2 : 1;
  return i + 1;
}

/** One `NAME: "value"` entry of an env block: the name bare or quoted, the
 *  value a string literal (escapes included). */
const ENTRY = /(["']?)([A-Za-z_$][\w$.-]*)\1(\s*:\s*)(["'`])((?:\\[\s\S]|(?!\4)[^\\])*)\4/g;

/**
 * A pm2 config with the values in its `env` blocks (`env`, `env_production`,
 * …) masked by `tag`, the service's config vars (CONFIG_VARS) left readable.
 * A block is found by scanning to its matching brace, strings skipped, so a
 * template literal or a nested object in it does not end it early.
 */
export function maskPm2(text: string, tag: (value: string) => string): string {
  const open = /\benv\w*\s*:\s*\{/g;
  let out = "";
  let from = 0;
  let m: RegExpExecArray | null;
  while ((m = open.exec(text))) {
    const start = m.index + m[0].length;
    let depth = 1;
    let i = start;
    while (i < text.length && depth > 0) {
      const c = text[i]!;
      if (c === '"' || c === "'" || c === "`") {
        i = skipString(text, i);
        continue;
      }
      if (c === "{") depth++;
      else if (c === "}") depth--;
      i++;
    }
    const end = depth === 0 ? i - 1 : text.length;
    const block = text
      .slice(start, end)
      .replace(ENTRY, (all, q: string, name: string, colon: string, quote: string, value: string) =>
        name in CONFIG_VARS ? all : `${q}${name}${q}${colon}${quote}${tag(value)}${quote}`,
      );
    out += text.slice(from, start) + block;
    from = end;
    open.lastIndex = end;
  }
  return out + text.slice(from);
}

// ── diff ──────────────────────────────────────────────────────────────────

/** A unified diff of two texts, by line, three lines of context; null when
 *  they are equal. Config files are small: past `maxCells` of work it says
 *  the files differ instead. */
export function unifiedDiff(a: string, b: string, aName: string, bName: string, maxCells = 4_000_000): string | null {
  if (a === b) return null;
  const lines = (s: string) => (s === "" ? [] : s.replace(/\n$/, "").split("\n"));
  const A = lines(a);
  const B = lines(b);
  const n = A.length;
  const m = B.length;
  const head = `--- ${aName}\n+++ ${bName}\n`;
  if (n * m > maxCells) return `${head}(the files differ; too long to diff here)`;
  const lcs = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i]![j] = A[i] === B[j] ? lcs[i + 1]![j + 1]! + 1 : Math.max(lcs[i + 1]![j]!, lcs[i]![j + 1]!);
    }
  }
  type Op = { t: " " | "-" | "+"; line: string; a: number; b: number };
  const ops: Op[] = [];
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) ops.push({ t: " ", line: A[i]!, a: i++, b: j++ });
    else if (j >= m || (i < n && lcs[i + 1]![j]! >= lcs[i]![j + 1]!)) ops.push({ t: "-", line: A[i]!, a: i++, b: j });
    else ops.push({ t: "+", line: B[j]!, a: i, b: j++ });
  }
  const CONTEXT = 3;
  const changed = ops.flatMap((op, k) => (op.t === " " ? [] : [k]));
  const hunks: string[] = [];
  for (let c = 0; c < changed.length; ) {
    let last = c;
    while (last + 1 < changed.length && changed[last + 1]! - changed[last]! <= 2 * CONTEXT) last++;
    const s = Math.max(0, changed[c]! - CONTEXT);
    const e = Math.min(ops.length - 1, changed[last]! + CONTEXT);
    const span = ops.slice(s, e + 1);
    const aLen = span.filter((op) => op.t !== "+").length;
    const bLen = span.filter((op) => op.t !== "-").length;
    const aStart = aLen ? span.find((op) => op.t !== "+")!.a + 1 : span[0]!.a;
    const bStart = bLen ? span.find((op) => op.t !== "-")!.b + 1 : span[0]!.b;
    hunks.push(`@@ -${aStart},${aLen} +${bStart},${bLen} @@\n${span.map((op) => op.t + op.line).join("\n")}`);
    c = last + 1;
  }
  return head + (hunks.length ? hunks.join("\n") : "(the files differ only in a final newline)");
}

// ── hive's stored config ──────────────────────────────────────────────────

/** A service as hive keeps it; `env` masked on read. */
export interface Service {
  name: string;
  port?: number | string;
  scripts?: Record<string, string>;
  env?: Record<string, string>;
  [key: string]: unknown;
}

export interface Stored {
  services: Service[];
  /** The workspace-wide env vars, masked. */
  env: Array<{ name: string; value: string }>;
  /** The files, decoded, by hive's names; pm2.config.js in clear. */
  files: Partial<Record<FileName, string>>;
  rollout: Record<string, unknown> | null;
}

const configUrl = (base: string, workspace: string) => `${base}/api/workspaces/${encodeURIComponent(workspace)}/stakgraph`;

/** What a failed hive call said, never a body that could hold config. */
export function hiveError(body: any): string {
  const b = body ?? {};
  const parts = [b.message, b.error, b.details ? JSON.stringify(b.details) : undefined].filter(Boolean);
  return parts.length ? brief(parts.join(" — "), 600) : "no error message";
}

/** Hive's stored pool config for a workspace, as a key caller reads it. */
export async function readStored(ctx: PodCtx, workspace: string): Promise<Stored> {
  const { base, headers } = await hiveApi(ctx);
  const url = configUrl(base, workspace);
  const res = await ctx.services.http(url, { method: "GET", headers });
  const b: any = res.body ?? {};
  if (!res.ok || b.success === false) {
    throw new Error(
      `hive pool config for workspace "${workspace}": HTTP ${res.status} — ${hiveError(b)} (the workspace by its slug, and HIVE_API_KEY an org key for its org)`,
    );
  }
  const d = b.data ?? {};
  const files: Partial<Record<FileName, string>> = {};
  for (const [name, b64] of Object.entries((d.containerFiles ?? {}) as Record<string, string>)) {
    const n = poolName(name);
    if (n && typeof b64 === "string") files[n] = Buffer.from(b64, "base64").toString("utf-8");
  }
  return {
    services: Array.isArray(d.services) ? d.services : [],
    env: Array.isArray(d.environmentVariables) ? d.environmentVariables : [],
    files,
    rollout: d.rollout ?? null,
  };
}

/** Write a change to hive's stored config: hive saves it and pushes the whole
 *  stored state to the pool. Returns the PUT's change record, never its body
 *  (which carries the stored files in clear). */
export async function writeStored(
  ctx: PodCtx,
  workspace: string,
  body: Record<string, unknown>,
): Promise<{ changeId: string | null; requestedAt: string | null; infraChanged: boolean; state: string; error: string | null }> {
  const { base, headers } = await hiveApi(ctx);
  const res = await ctx.services.http(configUrl(base, workspace), {
    method: "PUT",
    headers: { ...headers, "content-type": "application/json" },
    body,
  });
  const b: any = res.body ?? {};
  if (!res.ok || b.success === false) {
    throw new Error(`hive refused the pool config change for workspace "${workspace}": HTTP ${res.status} — ${hiveError(b)}`);
  }
  return {
    changeId: typeof b.changeId === "string" ? b.changeId : null,
    requestedAt: typeof b.requestedAt === "string" ? b.requestedAt : null,
    infraChanged: b.infraChanged === true,
    state: String(b.rollout?.state ?? (b.changeId ? "rolling" : "idle")),
    error: b.rollout?.error ? String(b.rollout.error) : null,
  };
}

// ── the pod's files ───────────────────────────────────────────────────────

export interface PodFile {
  name: FileName;
  path: string;
  content: string;
}

/** The files the pod runs from (staklink `GET /pod-config`), by hive's names. */
export async function readPod(ctx: PodCtx, control: string, sealed: string): Promise<PodFile[]> {
  const res = await podCall(ctx, control, sealed, "/pod-config");
  if (res.status === 404) {
    throw new Error("GET /pod-config: 404 — this pod's staklink predates /pod-config; read hive's config alone (leave out control and sealed)");
  }
  const b: any = res.body ?? {};
  if (!res.ok || b.error) throw new Error(`GET /pod-config: ${res.status} — ${brief(b.message ?? b.error ?? b)}`);
  if (!Array.isArray(b.files)) throw new Error(`GET /pod-config returned no file list: ${brief(b)}`);
  const out: PodFile[] = [];
  for (const f of b.files) {
    const name = poolName(String(f?.name ?? ""));
    if (name && !out.some((o) => o.name === name)) out.push({ name, path: String(f.path ?? ""), content: String(f.content ?? "") });
  }
  return out;
}

// ── edits by name ─────────────────────────────────────────────────────────

/**
 * A list of named things with edits applied by name: a name set to a value is
 * written (in place, or appended when new), `null` removes it, a name left
 * out stays as it is. Removing a name the list does not have is an error — a
 * typo would otherwise pass silently.
 */
export function applyEdits<T extends { name: string }>(
  list: T[],
  edits: Record<string, T | null>,
  what: string,
): { list: T[]; written: string[]; removed: string[] } {
  const out = list.map((x) => ({ ...x }));
  const written: string[] = [];
  const removed: string[] = [];
  for (const [name, value] of Object.entries(edits)) {
    const at = out.findIndex((x) => x.name === name);
    if (value === null) {
      if (at < 0) throw new Error(`there is no ${what} "${name}" to remove (stored: ${list.map((x) => x.name).join(", ") || "none"})`);
      out.splice(at, 1);
      removed.push(name);
    } else {
      const next = { ...value, name };
      if (at < 0) out.push(next);
      else out[at] = next;
      written.push(name);
    }
  }
  return { list: out, written, removed };
}

/**
 * Refuse what would corrupt the stored config: a masked value that is not
 * the stored one's own mask (hive would store the stars as the value), and a
 * service's config var written as an env var (hive generates those from the
 * service's `port` and `scripts`).
 */
export function checkEnvValues(env: Record<string, string>, storedMasked: Record<string, string>, where: string): void {
  for (const [name, value] of Object.entries(env)) {
    if (name in CONFIG_VARS) {
      throw new Error(`${where}: ${name} is generated from the service's ${CONFIG_VARS[name]} — set it there, not as an env var`);
    }
    if (looksMasked(value) && value !== storedMasked[name]) {
      throw new Error(
        `${where}: ${name} is a masked value, not a value — write the real value, or leave ${name} out to keep the stored one`,
      );
    }
  }
}
