import { z, defineStep, usageFromResult, computeCost, addUsage, coerceUsage } from "vein";
import vm from "node:vm";
import yaml from "js-yaml";

/**
 * STRUCTURED, HYBRID scorer for the gitsee setup-profiler (replaces the pure-LLM
 * `eval/score` for gitsee). Both `actual` and `expected` are a pm2.config.js +
 * docker-compose.yml pair in the `FILENAME: <name>` + fenced-block format. The
 * gold is the canonical known-good pair, so it's the ANSWER KEY — which is what
 * makes deterministic scoring possible WITHOUT understanding any dependency: we
 * never interpret what an env var or image means, we only set-diff NAMES against
 * the gold.
 *
 * Two tiers (see plans/gitsee-structured-scorer-and-subagents.md):
 *
 *  1. DETERMINISTIC (dominant) — name set-diffs vs the gold:
 *     - env-key completeness — RECALL-ONLY: keys(produced pm2 env) vs keys(gold
 *       pm2 env), scored on recall (did we produce the gold's keys?) but NOT
 *       precision. Extra keys are NOT penalized: the gold is a minimal curated
 *       set, so the agent's complete real set (every `process.env.X` the code
 *       reads) shows up as "extra" — but those are legitimate and an extra env
 *       var never breaks a boot. Penalizing them just teaches the optimizer to
 *       drop real keys. Recall stays exact-name + repo-agnostic because the key
 *       NAME is dictated by the code. (Build/run directives like INSTALL_COMMAND
 *       live in this env block too, so "key commands" are covered.)
 *     - service set — recall AND precision: compose service IDENTITY (image base
 *       name, tag stripped, or the service name for build-only services) produced
 *       vs gold. Unlike env keys, an extra service IS real harm (an invented
 *       `redis`/`soketi` spins up an unneeded container), so service
 *       over-provisioning is a precision hit.
 *
 *  2. LLM SEMANTIC RESIDUE (optional, capped) — only what needs interpretation
 *     and therefore can't be a name set-diff: is each pm2 `script` the right
 *     start command, is a host-binding flag present when the framework needs one,
 *     do the DB/connection creds in the pm2 env line up with the compose service
 *     (naming-agnostic — the LLM reads both), is an added service appropriate.
 *     Folded in as a bounded MULTIPLIER so the deterministic tier dominates.
 *
 * Combine: one recall-weighted F-beta (β=2) over the UNIFIED expected item set
 * (env keys ∪ services). Recall counts every gold item; precision counts only
 * spurious SERVICES (extra env keys are ignored — see tier 1). Then
 * `score = base * (SEM_FLOOR + (1-SEM_FLOOR)*semantic)`.
 *
 * HARD CONTRACT (eval/optimize + eval/reflect depend on this shape — don't break
 * it): output `{ score, recall, precision, matched, missing, spurious, reason,
 * insight, markdown }`.
 *
 * Self-contained: imports only `vein`, `js-yaml`, `node:vm`, and (lazily, only
 * when the LLM tier is on) `ai` + `@ai-sdk/anthropic`. No imports from `src/`.
 */

const BETA = 2; // weight recall this many times more than precision
const SEM_FLOOR = 0.7; // the LLM tier can dock at most (1 - SEM_FLOOR) of the score

function fBeta(recall: number, precision: number): number {
  const b2 = BETA * BETA;
  const denom = b2 * precision + recall;
  return denom === 0 ? 0 : ((1 + b2) * precision * recall) / denom;
}

// ── parse the two-file "FILENAME: …" + fenced-block string ────────────────────

interface TwoFiles {
  pm2?: string;
  compose?: string;
}

/** Split a "FILENAME: <name>\n```lang\n…\n```" doc into its named fenced blocks,
 *  then bucket them into the pm2 config and the compose file by filename. */
function splitFiles(doc: string): TwoFiles {
  const out: TwoFiles = {};
  const re = /FILENAME:\s*([^\n]+?)\s*\n+```[a-zA-Z0-9]*\n([\s\S]*?)\n```/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(doc)) !== null) {
    const name = m[1].trim().toLowerCase();
    const body = m[2];
    if (name.includes("pm2")) out.pm2 ??= body;
    else if (name.includes("compose")) out.compose ??= body;
  }
  return out;
}

/** A callable, infinitely-chainable no-op — lets sandboxed `require(...)` chains
 *  (e.g. `require("dotenv").config()`) resolve without throwing. */
function callableStub(): unknown {
  const fn = function () {
    return stub;
  };
  const stub: unknown = new Proxy(fn, { get: () => stub, apply: () => stub });
  return stub;
}

interface Pm2App {
  name?: string;
  script?: string;
  cwd?: string;
  env?: Record<string, unknown>;
}

/** Eval `module.exports = {...}` in a locked-down vm (no real require/process,
 *  1s timeout). Returns the apps array, or null if it doesn't parse/execute. */
function parsePm2(code: string | undefined): Pm2App[] | null {
  if (!code) return null;
  const sandbox: Record<string, unknown> = {
    module: { exports: {} },
    require: () => callableStub(),
    console: { log() {}, error() {}, warn() {}, info() {} },
    process: { env: {} },
  };
  sandbox.exports = (sandbox.module as { exports: unknown }).exports;
  try {
    vm.createContext(sandbox);
    vm.runInContext(code, sandbox, { timeout: 1000 });
    const exported = (sandbox.module as { exports: unknown }).exports as { apps?: unknown };
    const apps = exported?.apps;
    if (!Array.isArray(apps)) return null;
    return apps as Pm2App[];
  } catch {
    return null;
  }
}

interface ComposeService {
  id: string; // identity used for set-matching (image base name, or service name)
  name: string;
}

/** Parse compose YAML → service identities. Identity = image base name (tag
 *  stripped) when present, else the service key (covers build-only services like
 *  the base "app"). Returns null if the YAML doesn't parse. */
function parseCompose(text: string | undefined): ComposeService[] | null {
  if (!text) return null;
  let doc: unknown;
  try {
    doc = yaml.load(text);
  } catch {
    return null;
  }
  const services = (doc as { services?: Record<string, unknown> })?.services;
  if (!services || typeof services !== "object") return [];
  return Object.entries(services).map(([name, svc]) => {
    const image = (svc as { image?: unknown })?.image;
    const id =
      typeof image === "string" && image.trim()
        ? image.split(":")[0].trim().toLowerCase()
        : name.toLowerCase();
    return { id, name };
  });
}

// ── set-diff helpers ──────────────────────────────────────────────────────────

interface SetDiff {
  matched: string[];
  missing: string[]; // expected but not produced
  spurious: string[]; // produced but not expected
}

/** Order-stable set diff of `produced` against `expected` (the gold). */
function diffSets(expected: Iterable<string>, produced: Iterable<string>): SetDiff {
  const exp = new Set(expected);
  const prod = new Set(produced);
  const matched: string[] = [];
  const missing: string[] = [];
  for (const e of exp) (prod.has(e) ? matched : missing).push(e);
  const spurious: string[] = [];
  for (const p of prod) if (!exp.has(p)) spurious.push(p);
  return { matched, missing, spurious };
}

/** Union of env-var key names across all pm2 apps (minus an ignore list). */
function envKeys(apps: Pm2App[] | null, ignore: Set<string>): string[] {
  if (!apps) return [];
  const keys = new Set<string>();
  for (const app of apps) {
    if (app?.env && typeof app.env === "object") {
      for (const k of Object.keys(app.env)) if (!ignore.has(k)) keys.add(k);
    }
  }
  return [...keys];
}

// ── optional LLM semantic residue ─────────────────────────────────────────────

interface Semantic {
  semanticScore: number; // 0..1
  issues: string[];
  insight: string;
}

/** The judge's semantic verdict + what its own LLM call cost. */
interface JudgeResult {
  semantic: Semantic;
  usage: ReturnType<typeof usageFromResult>;
  cost: number;
}

async function judgeSemantics(
  cfg: { actual: string; expected: string; rubric?: string; provider?: string; model?: string },
): Promise<JudgeResult> {
  const provider = cfg.provider ?? process.env["VEIN_LLM_PROVIDER"] ?? "anthropic";
  const modelName = cfg.model ?? process.env["VEIN_LLM_MODEL"];
  const { generateObject } = await import("ai");
  let model: unknown;
  switch (provider) {
    case "anthropic": {
      const { anthropic } = await import("@ai-sdk/anthropic");
      model = anthropic(modelName ?? "claude-sonnet-4-20250514");
      break;
    }
    default:
      throw new Error(`Unknown LLM provider: "${provider}". Supported: anthropic`);
  }

  const schema = z.object({
    semanticScore: z
      .number()
      .min(0)
      .max(1)
      .describe(
        "0..1: how well the produced files satisfy the SEMANTIC (non-mechanical) requirements only — right start command(s), host-binding flag where the framework needs one, DB/connection creds in the pm2 env consistent with the compose service, and added services being appropriate. 1 = all good; do NOT re-penalize missing env keys / services here (those are scored separately).",
      ),
    issues: z
      .array(z.string())
      .describe("Concrete SEMANTIC problems only (wrong start command, missing host flag, mismatched DB creds, an inappropriate extra service). Empty if none."),
    insight: z
      .string()
      .describe(
        "1 sentence: a GENERAL, transferable lesson about the GENERATOR (never naming this input's specifics) — e.g. 'the generator forgets the host-binding flag for frameworks that default to localhost'.",
      ),
  });

  const rubric =
    cfg.rubric ??
    `Judge ONLY the semantic (non-mechanical) quality of the PRODUCED setup vs the GOLD. The env-var KEY SET and the SERVICE SET are scored separately and deterministically — IGNORE them here. Focus exclusively on: (a) does each pm2 app's \`script\` actually start the repo's real server/entrypoint; (b) is a host-binding flag present when the framework needs one (Next.js/Vite) and absent when it doesn't; (c) are DB/connection credentials in the pm2 env consistent with the corresponding compose service (same db/user/password/host:port), regardless of how they're named; (d) is any service the produced file adds beyond the gold actually appropriate (not bloat/over-provisioning). Reward functional equivalence, not textual mimicry.`;

  const prompt = `${rubric}

# GOLD (canonical known-good)
${cfg.expected}

# PRODUCED
${cfg.actual}`;

  const { object, usage: rawUsage } = await generateObject({ model: model as any, prompt, schema: schema as any });
  const usage = usageFromResult(rawUsage);
  return { semantic: object as Semantic, usage, cost: computeCost(provider, usage) };
}

// ── step ──────────────────────────────────────────────────────────────────────

export default defineStep({
  type: "gitsee/score-setup",
  description:
    "Structured + hybrid scorer for the gitsee setup-profiler. Deterministically parses a pm2.config.js (node:vm) + docker-compose.yml (js-yaml) pair out of both `actual` and `expected`, then scores by NAME set-diffs vs the gold: env-key completeness (RECALL-ONLY — extra env keys are not penalized) + compose service/image set (recall AND precision — extra services are over-provisioning) combined as a recall-weighted F-beta, β=2. An optional LLM tier judges only the semantic residue (start command, host-binding flag, cross-file cred consistency, service appropriateness) and is folded in as a bounded multiplier. Config: actual, expected, rubric? (semantic-judge rubric), useLLM? (default true), ignoreEnvKeys? (default []), provider?, model?. Output: { score, recall, precision, matched, missing, spurious, reason, insight, markdown }.",
  input: z.object({
    actual: z.string(),
    expected: z.string(),
    rubric: z.string().optional(),
    useLLM: z.boolean().default(true),
    ignoreEnvKeys: z.array(z.string()).default([]),
    provider: z.string().optional(),
    model: z.string().optional(),
    // Upstream (explorer agent) token usage + cost to fold the judge's own
    // tokens+$ into, so the output `usage`/`cost` is the full per-eval total.
    priorUsage: z.any().optional(),
    priorCost: z.number().optional(),
  }),
  output: z.any(),
  async run(cfg) {
    const ignore = new Set(cfg.ignoreEnvKeys);

    const goldFiles = splitFiles(cfg.expected);
    const prodFiles = splitFiles(cfg.actual);

    const goldApps = parsePm2(goldFiles.pm2);
    const prodApps = parsePm2(prodFiles.pm2);
    const goldServices = parseCompose(goldFiles.compose);
    const prodServices = parseCompose(prodFiles.compose);

    // unparseable flags (surface as a concrete defect, drive score toward 0)
    const unparseable: string[] = [];
    if (goldFiles.pm2 && !goldApps) unparseable.push("gold pm2.config.js did not parse");
    if (goldFiles.compose && !goldServices) unparseable.push("gold docker-compose.yml did not parse");
    if (prodFiles.pm2 && !prodApps) unparseable.push("produced pm2.config.js did not parse");
    if (prodFiles.compose && !prodServices) unparseable.push("produced docker-compose.yml did not parse");

    // ── deterministic sub-scores ──────────────────────────────────────────────
    const envDiff = diffSets(envKeys(goldApps, ignore), envKeys(prodApps, ignore));
    const svcDiff = diffSets(
      (goldServices ?? []).map((s) => s.id),
      (prodServices ?? []).map((s) => s.id),
    );

    const envExpected = envDiff.matched.length + envDiff.missing.length;
    const envRecall = envExpected === 0 ? 1 : envDiff.matched.length / envExpected;
    const svcExpected = svcDiff.matched.length + svcDiff.missing.length;
    const svcRecall = svcExpected === 0 ? 1 : svcDiff.matched.length / svcExpected;

    // unified item set (prefixed so env/service items can't collide)
    const matched = [
      ...envDiff.matched.map((k) => `env:${k}`),
      ...svcDiff.matched.map((s) => `service:${s}`),
    ];
    const missingDet = [
      ...envDiff.missing.map((k) => `env:${k}`),
      ...svcDiff.missing.map((s) => `service:${s}`),
    ];
    // ENV IS RECALL-ONLY: extra env keys are NOT penalized. They're almost always
    // REAL keys the app reads (the gold is a minimal curated set, not the complete
    // one), and an extra env var doesn't break a boot — so it must not drag the
    // score or the reflect feedback. Only SERVICE over-provisioning is real harm
    // (an unneeded redis/soketi container), so only spurious services count against
    // precision. `envDiff.spurious` is kept for the markdown/detail (informational).
    const spurious = [...svcDiff.spurious.map((s) => `service:${s}`)];

    const expectedTotal = matched.length + missingDet.length;
    const producedTotal = matched.length + spurious.length;
    const recall = expectedTotal === 0 ? 1 : matched.length / expectedTotal;
    const precision = producedTotal === 0 ? 1 : matched.length / producedTotal;
    const base = fBeta(recall, precision);

    // ── optional LLM semantic residue ─────────────────────────────────────────
    // Token usage + cost: start from the upstream explorer agent's, then add the
    // judge's own LLM call (when run) → the output carries the full per-eval cost.
    let usage = coerceUsage(cfg.priorUsage);
    let cost = typeof cfg.priorCost === "number" ? cfg.priorCost : 0;
    let sem: Semantic | undefined;
    if (cfg.useLLM) {
      try {
        const judged = await judgeSemantics(cfg);
        sem = judged.semantic;
        usage = addUsage(usage, judged.usage);
        cost += judged.cost;
      } catch (e) {
        console.warn("[gitsee/score-setup] semantic judge failed, deterministic-only:", e instanceof Error ? e.message : e);
      }
    }
    const semanticScore = sem ? Math.max(0, Math.min(1, sem.semanticScore)) : 1;
    const semIssues = sem?.issues ?? [];

    const score = Math.round(base * (SEM_FLOOR + (1 - SEM_FLOOR) * semanticScore) * 100) / 100;

    // missing (for reflect): deterministic gaps + unparseable + semantic issues
    const missing = [
      ...unparseable.map((u) => `parse:${u}`),
      ...missingDet,
      ...semIssues.map((i) => `semantic:${i}`),
    ];

    // ── insight + reason ──────────────────────────────────────────────────────
    let insight = sem?.insight ?? "";
    if (!insight) {
      if (envDiff.missing.length)
        insight = "the generator omits environment variables the app reads at runtime — enumerate every process.env key the code references before writing the config.";
      else if (svcDiff.spurious.length)
        insight = "the generator over-provisions services the app doesn't actually depend on — only declare backing services the code connects to.";
      else if (svcDiff.missing.length)
        insight = "the generator misses a required backing service — trace the app's data/connection dependencies to a compose service.";
      else insight = "the generator produces a functionally complete setup; refine only the semantic details.";
    }

    const reasonParts: string[] = [];
    if (unparseable.length) reasonParts.push(`Unparseable: ${unparseable.join("; ")}.`);
    if (envDiff.missing.length) reasonParts.push(`Missing env keys: ${envDiff.missing.join(", ")}.`);
    if (svcDiff.missing.length) reasonParts.push(`Missing services: ${svcDiff.missing.join(", ")}.`);
    if (svcDiff.spurious.length) reasonParts.push(`Extra services: ${svcDiff.spurious.join(", ")}.`);
    if (semIssues.length) reasonParts.push(`Semantic: ${semIssues.join("; ")}.`);
    const reason = reasonParts.join(" ") || "Produced setup matches the gold's functional requirements.";

    // ── markdown breakdown ────────────────────────────────────────────────────
    const pct = (n: number) => `${Math.round(n * 100)}%`;
    const markdown = [
      `**Score: ${score}**  (recall ${pct(recall)}, precision ${pct(precision)}${cfg.useLLM ? `, semantic ${pct(semanticScore)}` : ""})`,
      ``,
      `- **env keys**: ${envDiff.matched.length}/${envExpected} matched (recall ${pct(envRecall)})` +
        (envDiff.missing.length ? ` — missing: ${envDiff.missing.join(", ")}` : ""),
      `- **services**: ${svcDiff.matched.length}/${svcExpected} matched (recall ${pct(svcRecall)})` +
        (svcDiff.missing.length ? ` — missing: ${svcDiff.missing.join(", ")}` : "") +
        (svcDiff.spurious.length ? ` — extra: ${svcDiff.spurious.join(", ")}` : ""),
      envDiff.spurious.length ? `- **extra env keys** (not penalized — env is recall-only): ${envDiff.spurious.join(", ")}` : "",
      unparseable.length ? `- **parse errors**: ${unparseable.join("; ")}` : "",
      cfg.useLLM && semIssues.length ? `- **semantic issues**: ${semIssues.join("; ")}` : "",
      ``,
      `${reason}`,
      insight ? `\n_Insight: ${insight}_` : "",
    ]
      .filter((l) => l !== "")
      .join("\n");

    return {
      score,
      recall,
      precision,
      matched,
      missing,
      spurious,
      reason,
      insight,
      markdown,
      // explorer agent + judge token usage and dollar cost for this eval
      usage,
      cost,
      // extra structured detail (ignored by the contract consumers, handy for debugging)
      detail: {
        env: { ...envDiff, recall: envRecall },
        services: { ...svcDiff, recall: svcRecall },
        semanticScore: cfg.useLLM ? semanticScore : null,
        unparseable,
      },
    };
  },
});
