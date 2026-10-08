/**
 * Shared by every `browser/*` step. A leading `_` makes this a helper the
 * registry skips but siblings import (`./_shared.js`). Like the steps it is
 * seeded into the workspace verbatim, so it imports `strut`, node's own
 * modules and, TYPE-ONLY, the lab's service and shot types — nothing that
 * must exist at runtime beside it.
 */
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z, jobRoot, type StepContext, type StrutCapabilities } from "strut";
import type { BrowserService } from "../service.js";
import type { ShotSink } from "../shot.js";

export type { BrowserService, ShotSink };

export type Ctx = StepContext<StrutCapabilities & { browser: BrowserService }>;

/** Exactly one of these: a ref for a model, a selector for an author. */
export const target = {
  ref: z.string().optional().describe("an element ref from the last browser/snapshot, e.g. e12 — reset by any navigation"),
  selector: z
    .string()
    .optional()
    .describe("a Playwright selector for an author who knows the page: text=Annual, role=button[name=Save], or CSS"),
};

export const oneOf =
  (...keys: string[]) =>
  (v: Record<string, unknown>) =>
    keys.filter((k) => v[k] !== undefined && v[k] !== "").length === 1;

export const atMostOneOf =
  (...keys: string[]) =>
  (v: Record<string, unknown>) =>
    keys.filter((k) => v[k] !== undefined && v[k] !== "").length <= 1;

export const ONE_TARGET = { message: "give exactly one of ref or selector" };

export const timeoutMs = z.number().int().positive().optional().describe("per-call cap in ms (default BROWSER_STEP_TIMEOUT_MS, 10 s)");

export const viewport = z
  .object({ width: z.number().int().positive(), height: z.number().int().positive() })
  .optional()
  .describe("viewport in CSS pixels; default 1280×800");

export const navigated = z.object({
  url: z.string(),
  title: z.string(),
  status: z.number().describe("HTTP status of the document, 0 when nothing was fetched"),
});

export const ok = z.object({ ok: z.literal(true) });

/** The verbs a `capture` list may name — must match the service's CAPTURE_VERBS. */
export const CAPTURE_VERBS = ["goto", "click", "fill", "type", "press", "select", "hover", "scroll", "wait", "text", "evaluate", "back"] as const;

export function browserOf(ctx: Ctx): BrowserService {
  const browser = ctx.services?.browser;
  if (!browser) throw new Error("browser/*: no `browser` service on ctx.services — the host did not build one (its boot log says why)");
  return browser;
}

export function artifactsOf(ctx: Ctx): NonNullable<StrutCapabilities["artifacts"]> {
  const artifacts = ctx.services?.artifacts;
  if (!artifacts) throw new Error("browser/*: no `artifacts` capability on ctx.services");
  return artifacts;
}

/**
 * Where this run's screenshots go (shot.ts): the run's artifacts — or, on a
 * run launched under a job, the job's directory, the one `job/dir` hands
 * out (`<dataDir>/jobs/<job>`, with job/dir's fallback when the host set no
 * `dataDir`). A job's host reads a file off the swarm by the job or by a run
 * id it launched; a child run (meta/run-workflow from a job turn) has a run
 * id the host never saw, so its shots must be job files to reach it.
 */
export function shotSinkOf(ctx: Ctx): ShotSink {
  const sink: ShotSink = { artifacts: artifactsOf(ctx), runId: ctx.runId };
  if (ctx.job) {
    const dataDir = ctx.services?.dataDir ?? join(tmpdir(), "strut");
    sink.job = { name: ctx.job, root: jobRoot(dataDir, ctx.job) };
  }
  return sink;
}
