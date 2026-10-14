import { z, defineStep } from "strut";
import type { StepContext } from "strut";
import type { ConceptServices } from "../services.js";

/**
 * Write a short `description` on code nodes that have a body and none yet
 * (Class, Endpoint, Request, Function, Datamodel, Page, Trait, Var), batch
 * by batch, until none are left or the cost cap is reached.
 *
 * Each description is one call of the core `llm` step, run through the
 * registry with this step's context — so the call is billed and capped like
 * any other step's (the Mothership, when routed) and none of them lands in
 * the run log as an event of its own. Nothing here embeds: strut's embedding
 * pass composes each code type's search text from its Schema `index`, which
 * includes `description`, and re-embeds a node once that text changes.
 *
 * Output: { processed, failed, cost, stopped } — `stopped` is "done" (no node
 * left), "cost_limit", or "no_progress" (a whole batch failed).
 */

const CODE_TYPES = ["Class", "Endpoint", "Request", "Function", "Datamodel", "Page", "Trait", "Var"];

// Errors no retry will fix: the key or the account, not the node.
const FATAL = ["key limit exceeded", "quota", "insufficient", "401", "invalid api key", "invalid_api_key", "unauthorized"];

/** An LLM call's error in full. A run log records an error's `message`
 *  only, and the AI SDK keeps the part that says what went wrong elsewhere:
 *  "Invalid JSON response" carries the parse failure as `cause` and the
 *  provider's status and body as properties (of it, or of the error it
 *  wraps — `cause`, a RetryError's `lastError`). */
export function said(err: Error): string {
  const parts = [err.message];
  let http = "";
  const seen = new Set<unknown>();
  for (let e: any = err; e && typeof e === "object" && !seen.has(e); e = e.cause ?? e.lastError) {
    seen.add(e);
    if (e !== err && e instanceof Error && e.message && !parts.includes(e.message)) parts.push(e.message);
    if (!http && (e.statusCode != null || typeof e.responseBody === "string")) {
      const body = typeof e.responseBody === "string" ? e.responseBody.trim() : "";
      const shown = body.length > 1000 ? `${body.slice(0, 1000)}… [${body.length} chars]` : body;
      http = ` (HTTP ${e.statusCode ?? "?"}${shown ? `: ${shown}` : ""})`;
    }
  }
  return parts.join(": ") + http;
}

type LlmDef = { input: z.ZodTypeAny; run: (cfg: unknown, ctx: unknown) => Promise<{ text?: string; cost?: number }> };

function repoPaths(repoUrl: string | undefined): string[] {
  if (!repoUrl || !repoUrl.trim()) return [];
  return repoUrl
    .split(",")
    .map((u) => u.trim())
    .filter(Boolean)
    .map((url) => {
      const match = url.match(/github\.com\/([^/]+\/[^/]+?)(\.git)?$/);
      if (!match) throw new Error(`Invalid repo URL format: ${url}`);
      return match[1]!;
    });
}

function fill(template: string, values: Record<string, string>): string {
  return template.replace(/\{(type|name|docs|code)\}/g, (_, key: string) => values[key] ?? "");
}

export default defineStep({
  type: "concepts/describe-code",
  description:
    "Write a 1-3 sentence `description` on code nodes (Class, Endpoint, Request, Function, Datamodel, Page, Trait, Var) that have a body and none yet, until none are left or `costLimit` is spent. One `llm` call per node; the embedding pass picks the description up. Output: { processed, failed, cost, stopped }.",
  input: z.object({
    repoUrl: z
      .string()
      .optional()
      .describe("GitHub URLs, comma-separated, to limit it to those repos (https://github.com/owner/repo); omit for every repo"),
    filePaths: z
      .array(z.string())
      .default([])
      .describe("file path suffixes to limit it to; empty for every file"),
    model: z.string().describe("the model for each description (an aieo name)"),
    prompt: z
      .string()
      .describe("the prompt for one node; {type}, {name}, {docs} and {code} are filled in"),
    maxCodeChars: z.number().int().positive().default(2000).describe("how much of the body {code} carries"),
    batchSize: z.number().int().positive().default(50),
    concurrency: z.number().int().positive().default(5),
    costLimit: z.number().positive().default(0.5).describe("dollars; no new call starts once this much is spent"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { codeNodes } = ctx.services as ConceptServices;
    const llm = (ctx as StepContext & { registry?: Record<string, LlmDef> }).registry?.["llm"];
    if (!llm) throw new Error("concepts/describe-code: core `llm` step not found — requires the runner-populated ctx.registry");
    const repos = repoPaths(cfg.repoUrl);

    let cost = 0;
    let processed = 0;
    const failed = new Set<string>();
    // A ref, so what the workers set is not narrowed away at the read below.
    const fatal: { error: Error | null } = { error: null };
    let stopped: "done" | "cost_limit" | "no_progress" = "done";

    while (true) {
      if (cost >= cfg.costLimit) {
        stopped = "cost_limit";
        break;
      }
      await ctx.control?.checkpoint();

      // A node whose call failed still has no description, so it comes back
      // first in every later read: read past the ones already failed.
      const nodes = (await codeNodes.withoutDescription(cfg.batchSize + failed.size, repos, cfg.filePaths))
        .filter((n) => !failed.has(n.ref_id))
        .slice(0, cfg.batchSize);
      if (nodes.length === 0) break;

      const written: Array<{ ref_id: string; description: string }> = [];
      let next = 0;
      const worker = async () => {
        while (next < nodes.length && !fatal.error && cost < cfg.costLimit) {
          const node = nodes[next++]!;
          const type = node.labels.find((l) => CODE_TYPES.includes(l)) ?? "Node";
          const prompt = fill(cfg.prompt, {
            type,
            name: node.name || "Unknown",
            docs: node.docs,
            code: node.body.slice(0, cfg.maxCodeChars),
          });
          try {
            const out = await llm.run(llm.input.parse({ prompt, model: cfg.model }), ctx);
            cost += out.cost ?? 0;
            const description = (out.text ?? "").trim();
            if (description) written.push({ ref_id: node.ref_id, description });
            else failed.add(node.ref_id);
          } catch (e) {
            failed.add(node.ref_id);
            const err = e instanceof Error ? e : new Error(String(e));
            const full = said(err);
            if (FATAL.some((needle) => full.toLowerCase().includes(needle))) fatal.error ??= new Error(full, { cause: err });
            else console.warn(`[concepts/describe-code] ${node.name}: ${full}`);
          }
        }
      };
      await Promise.all(Array.from({ length: Math.min(cfg.concurrency, nodes.length) }, worker));

      if (written.length > 0) {
        await codeNodes.writeDescriptions(written);
        processed += written.length;
      }
      if (fatal.error) {
        throw new Error(`concepts/describe-code: stopped after ${processed} description(s): ${fatal.error.message}`);
      }
      if (written.length === 0) {
        stopped = "no_progress";
        break;
      }
    }

    return { processed, failed: failed.size, cost, stopped };
  },
});
