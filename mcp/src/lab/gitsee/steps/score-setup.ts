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
// BOOT + RENDER (the screenshot) is the dominant signal — the real proof the
// setup works. The file-shape set-diff is only a small residual hint: a setup
// that booted AND rendered scores at least RENDER_FLOOR regardless of how its
// files compare to the gold (so a non-boot-critical missing key or an extra
// service barely moves it). Booted-but-broken-UI and non-boot are scaled down.
const RENDER_FLOOR = 0.85; // a booted+rendered setup scores in [RENDER_FLOOR, 1]
const RENDER_FAIL_MULT = 0.45; // booted but the page didn't render
const RENDER_UNKNOWN_MULT = 0.6; // booted but render couldn't be judged (no browser)
const NO_BOOT_MULT = 0.1; // did not boot at all

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
    "Scorer for the gitsee setup-profiler where BOOT + RENDER dominate. The booted+rendered screenshot verdict (from gitsee/verify-setup, via `booted`/`rendered`) is the headline signal: a booted+rendered setup scores in [0.85,1]; booted-but-broken and non-boot are scaled down hard. The file-shape set-diff is only a small residual hint — it deterministically parses the pm2.config.js (node:vm) + docker-compose.yml (js-yaml) pair from `actual`/`expected` and scores by NAME set-diffs vs the gold, RECALL-ONLY for BOTH env keys and services (extra keys AND extra services are NOT penalized — the boot gate is the arbiter), recall-weighted F-beta β=2, times an optional bounded LLM semantic tier. Config: actual, expected, booted?, rendered?, bootReason?, rubric?, useLLM? (default true), ignoreEnvKeys? (default []), provider?, model?. Output: { score, recall, precision, matched, missing, spurious, reason, insight, markdown }.",
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
    // DOMINANT BOOT GATE (from gitsee/verify-setup). A setup that doesn't boot is
    // a failure regardless of file shape, so these clamp the score: !booted →
    // heavy penalty, booted-but-not-rendered → medium. null/undefined (verify
    // skipped) leaves the score untouched (back-compat). `rendered:null` means the
    // browser couldn't run (no penalty) — boot alone gates.
    booted: z.boolean().nullable().optional(),
    rendered: z.boolean().nullable().optional(),
    bootReason: z.string().optional(),
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
    // BOTH ENV AND SERVICES ARE RECALL-ONLY: extra env keys AND extra services are
    // NOT penalized. The gold is a minimal curated set, not the complete one, and
    // the BOOT GATE is the real arbiter of correctness — if the produced setup
    // boots and renders, an extra service (e.g. a redis the gold mocks) or extra
    // env key did no harm. So precision is informational only; the spurious lists
    // (env + service) are kept for the markdown/detail, not scored.
    const spurious: string[] = [];

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

    // The file-shape score (recall-weighted F-beta × semantic) — now only a small
    // RESIDUAL that nudges within the boot/render band.
    const setupScore = base * (SEM_FLOOR + (1 - SEM_FLOOR) * semanticScore);

    // ── BOOT + RENDER dominate ─────────────────────────────────────────────────
    // The screenshot (did the real UI render?) is the headline signal. A
    // booted+rendered setup scores in [RENDER_FLOOR, 1], with setupScore moving
    // only the top (1 - RENDER_FLOOR). Booted-but-broken and non-boot are scaled
    // down hard. `booted == null` (no verify in this run) → pure setupScore.
    const renderedBand = RENDER_FLOOR + (1 - RENDER_FLOOR) * setupScore;
    const bootGaps: string[] = [];
    let raw: number;
    if (cfg.booted == null) {
      raw = setupScore; // no verify step — fall back to file-shape only
    } else if (cfg.booted && cfg.rendered === true) {
      raw = renderedBand; // ✅ the real thing
    } else if (cfg.booted && (cfg.rendered === null || cfg.rendered === undefined)) {
      raw = RENDER_UNKNOWN_MULT * renderedBand; // booted, render unjudged (no browser)
    } else if (cfg.booted && cfg.rendered === false) {
      raw = RENDER_FAIL_MULT * renderedBand;
      bootGaps.push(`boot:booted but the page did not render${cfg.bootReason ? ` (${cfg.bootReason})` : ""}`);
    } else {
      raw = NO_BOOT_MULT * setupScore;
      bootGaps.push(`boot:setup did not boot${cfg.bootReason ? ` (${cfg.bootReason})` : ""}`);
    }

    const score = Math.round(raw * 100) / 100;

    // missing (for reflect): boot failures FIRST (dominant) + deterministic gaps +
    // unparseable + semantic issues
    const missing = [
      ...bootGaps,
      ...unparseable.map((u) => `parse:${u}`),
      ...missingDet,
      ...semIssues.map((i) => `semantic:${i}`),
    ];

    // ── insight + reason ──────────────────────────────────────────────────────
    // Boot failure is the dominant signal, so it owns the insight when present.
    let insight = "";
    if (cfg.booted === false)
      insight =
        "the generated setup does not actually boot — prioritize a runnable config (correct install/start commands, real backing services, mock flags) over file completeness.";
    else if (cfg.booted === true && cfg.rendered === false)
      insight =
        "the setup boots but the frontend does not render — check the start command, host-binding flag, and that required services/env are wired so the page actually loads.";
    if (!insight) insight = sem?.insight ?? "";
    if (!insight) {
      if (svcDiff.missing.length)
        insight = "the generator misses a required backing service the app needs to boot — trace the app's data/connection dependencies to a compose service.";
      else if (envDiff.missing.length)
        insight = "the generator omits an environment variable the gold sets — check whether the app reads it at boot (extra/missing non-critical keys don't matter if it boots+renders).";
      else insight = "the generator produces a setup that boots and renders; refine only the semantic details.";
    }

    const reasonParts: string[] = [];
    if (cfg.booted === false) reasonParts.push(`Did not boot${cfg.bootReason ? `: ${cfg.bootReason}` : ""}.`);
    else if (cfg.booted === true && cfg.rendered === false)
      reasonParts.push(`Booted but did not render${cfg.bootReason ? `: ${cfg.bootReason}` : ""}.`);
    else if (cfg.booted === true && cfg.rendered === true) reasonParts.push("Booted and rendered. ✅");
    if (unparseable.length) reasonParts.push(`Unparseable: ${unparseable.join("; ")}.`);
    if (svcDiff.missing.length) reasonParts.push(`Missing services: ${svcDiff.missing.join(", ")}.`);
    if (envDiff.missing.length) reasonParts.push(`Missing env keys (not boot-critical if it rendered): ${envDiff.missing.join(", ")}.`);
    // Extra env keys + extra services are NOT defects (recall-only) — noted in the
    // markdown only, never in the reason fed to reflect.
    if (semIssues.length) reasonParts.push(`Semantic: ${semIssues.join("; ")}.`);
    const reason = reasonParts.join(" ") || "Produced setup matches the gold's functional requirements.";

    // ── markdown breakdown ────────────────────────────────────────────────────
    const pct = (n: number) => `${Math.round(n * 100)}%`;
    const bootBadge =
      cfg.booted === false
        ? ` — ⛔ DID NOT BOOT (×${NO_BOOT_MULT})`
        : cfg.booted === true && cfg.rendered === false
          ? ` — ⚠️ booted, no render (×${RENDER_FAIL_MULT})`
          : cfg.booted === true && cfg.rendered === true
            ? ` — ✅ booted & rendered`
            : "";
    const markdown = [
      `**Score: ${score}**${bootBadge}  (file-shape residual ${Math.round(setupScore * 100) / 100}: recall ${pct(recall)}${cfg.useLLM ? `, semantic ${pct(semanticScore)}` : ""} — boot+render dominates)`,
      ``,
      `- **env keys**: ${envDiff.matched.length}/${envExpected} matched (recall ${pct(envRecall)}, recall-only)` +
        (envDiff.missing.length ? ` — missing: ${envDiff.missing.join(", ")}` : ""),
      `- **services**: ${svcDiff.matched.length}/${svcExpected} matched (recall ${pct(svcRecall)}, recall-only)` +
        (svcDiff.missing.length ? ` — missing: ${svcDiff.missing.join(", ")}` : "") +
        (svcDiff.spurious.length ? ` — extra (not penalized): ${svcDiff.spurious.join(", ")}` : ""),
      envDiff.spurious.length ? `- **extra env keys** (not penalized — recall-only): ${envDiff.spurious.join(", ")}` : "",
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
