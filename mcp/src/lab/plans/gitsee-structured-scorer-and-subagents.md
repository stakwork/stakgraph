# gitsee: structured scorer + sub-agent exploration

Direction doc for the next phase of the `gitsee` lab experiment (the
self-contained setup-profiler that emits `pm2.config.js` + `docker-compose.yml`
to boot a workspace's frontend). Two linked goals, in deliberate order:

1. **A much more sophisticated, trustworthy scorer** — especially: verify the
   required env vars are actually present in the produced pm2 config.
2. **Sub-agents** that explore specific areas of the code (services, env vars,
   commands, cross-repo deps), then synthesize the config.

**Sequence matters: scorer FIRST, sub-agents SECOND.** You can't tell whether
sub-agents improve anything if the scorer is noisy — you'd be optimizing against
measurement error. A granular, deterministic scorer is also exactly what makes
per-area sub-agents independently optimizable.

---

## Background / current state

- `gitsee-explore-services` clones a WORKSPACE (repos as siblings under
  `/workspaces/<repo>`) and runs ONE flat agent loop
  (`mcp/src/lab/gitsee/steps/explore-services.ts`) with tools `repo_overview`,
  `file_summary`, `fulltext_search`, `bash`, `web_search`, `final_answer`. It
  emits the two files as a string in `FILENAME: <name>\n```...```` format.
- Eval stack (`mcp/src/lab/gitsee/workflows/`): `gitsee-eval` (produce → score)
  → `gitsee-eval-score` wraps the **generic** `eval/score`
  (`mcp/src/lab/eval/steps/score.ts`) — an LLM-as-judge that decomposes the GOLD
  into requirements and matches the produced files against them
  (functional-equivalence rubric). `gitsee-optimize` tunes the `system` prompt
  via `eval/optimize` over a dataset of workspaces.
- Validated baselines: `heroku-node` 0.95, `hive` 0.94 (live runs).

### The problem (why the scorer is the bottleneck)

The hive gold has **~15 env vars** (DATABASE_URL, NEXTAUTH_SECRET,
TOKEN_ENCRYPTION_KEY, JWT_SECRET, POD_URL, …). The LLM judge collapsed them into
roughly ONE "env present" matched item — so the score barely moves if a
**boot-critical** key is missing. Env-var completeness is mechanical and
boot-critical; an LLM is the wrong tool for it. The judge also gave hive 0.94
while it *over-provisioned* (added a `redis` service + `REDIS_URL` + many mock
env vars not in the gold) — the kind of precision error the loop should chase,
but the signal is currently mushy.

Until the scorer is sharp and low-variance, `gitsee-optimize` has nothing
reliable to optimize against.

---

## Phase 1 — structured / hybrid scorer

### Decision

Replace the pure-LLM `gitsee-eval-score` with a **structured scorer** that:

- **Deterministically extracts** structure from both files (they're parseable):
  - `pm2.config.js` is JS → eval `module.exports = {...}` in a Node `vm` with a
    stub `module`/`exports`/`require`, read `apps[].{ name, script, cwd, env(keys) }`.
  - `docker-compose.yml` is YAML → `js-yaml` → `services{}` (names, images,
    ports, `environment` keys).
  - Split the produced two-file string with a regex on the `FILENAME:` markers +
    following fenced block. (Same for the gold — it's the same format.)
- **Hard deterministic sub-scores**:
  - **env-key completeness** (the headline ask): recall/precision over the pm2
    `env` KEY SET vs the gold's (values stay cosmetic). A missing required key is
    a heavy penalty — this is the boot signal.
  - **service set**: required compose services present (by role/image), extra
    services flagged (over-provisioning → precision hit).
  - (optional) **cross-file consistency**: DB creds in the pm2 `DATABASE_URL`
    match the compose postgres service env — boot-critical and deterministic.
- **LLM judge keeps only the semantic residue**: is the `script` the right
  start command, host-binding flag present, is an added service actually
  appropriate, the one-line `insight`.
- **Combine** into a single recall-weighted score, env-completeness dominant.

This is *more sophisticated* yet *simpler/more honest* — a set diff beats a
paragraph of rubric prose hoping the judge enumerates 15 keys, and it kills the
variance that makes optimization unreliable.

### Hard contract to preserve

The scorer MUST keep emitting `{ score, recall, precision, missing[],
spurious[], insight, markdown }` — `eval/optimize`
(`mcp/src/lab/eval/steps/optimize.ts`) reads `score`/`missing`/`spurious`/
`insight` off each eval run, and `eval/reflect` digests `missing`/`spurious`/
`insight`. Don't break that shape or the loop breaks.

### Self-containment rule (do not violate)

gitsee steps import ONLY `vein`, third-party npm (`ai`, `@ai-sdk/anthropic`,
`js-yaml`, `zod`), and Node builtins (`node:vm`, etc.) — NO imports from
existing `src/` code. (Workspace-seeded `.ts` steps resolve bare specifiers
against `mcp/node_modules`, so `js-yaml`/`ai` are available; `node:vm` is fine.)
The goal is still to delete `src/gitsee` eventually.

### Files (proposed)

- **New** `mcp/src/lab/gitsee/steps/score-setup.ts` (`gitsee/score-setup`) — the
  structured scorer step. Inputs `{ actual, expected, rubric? }`; parses both,
  computes deterministic sub-scores, optionally calls the LLM (via
  `ai` + `@ai-sdk/anthropic`, like `eval/score`) for the semantic residue, emits
  the contract shape above.
- **Edit** `mcp/src/lab/gitsee/workflows/gitsee-eval-score.yaml` — point `judge`
  at `gitsee/score-setup` instead of `eval/score`; move the matching criteria
  into its `params` (what counts as a required env var, optional-key policy,
  semantic-judge rubric).
- **Edit** `mcp/src/lab/gitsee/seed.ts` — add `score-setup.ts` to `SEED_STEPS`.
- **Edit** `mcp/src/lab/AGENTS.md` — document the structured scorer.
- `gitsee-eval.yaml` / `gitsee-optimize.yaml` unchanged (same contract).

### Validation

- Unit-ish: feed the scorer the hive `confs.md` gold as BOTH `actual` and
  `expected` → expect score ≈ 1.0, empty missing.
- Feed a gold with one env key deleted from `actual` → that key shows in
  `missing`, score drops materially.
- Re-run `smoke-eval.ts hive` / `heroku-node`; confirm scores are sane and that
  hive's over-provisioning (redis + extra mocks) now shows as concrete spurious
  items. (`smoke-eval.ts` builds a real lab vein; needs Neo4j + ANTHROPIC_API_KEY
  + GITHUB_TOKEN; hive eval is ~3 min.)

---

## Phase 2 — sub-agent exploration (after the scorer)

### Sketch

Decompose the single flat `explore-services` loop into **specialized
sub-agents**, each with focused context + its own tool budget:

- `gitsee/explore-services-needed` — required databases/caches/queues to boot.
- `gitsee/explore-env` — every env var the frontend needs (the highest-value
  area, pairs directly with the env-completeness sub-score).
- `gitsee/explore-commands` — package manager + install/build/dev/start +
  host-binding + migrations.
- (maybe) `gitsee/explore-deps` — cross-repo `file:`/workspace links among the
  cloned siblings.

An orchestrator workflow runs them in parallel (vein `depends: []`), each emits
**structured findings**, and a **synthesis step** assembles the final pm2/compose
from the findings (the pod contract / `finalAnswer` logic moves here).

### Why it pairs with Phase 1

A per-area deterministic scorer lets you eval each sub-agent in isolation (e.g.
"did `explore-env` recover every required env key?") — a much tighter feedback
loop than scoring the whole config at once, and each sub-agent's prompt becomes
an independent experiment surface for `eval/optimize`.

### Prior art in the repo

`mcp/src/repo` already has a sub-agent mechanism (`callRemoteAgent`,
`subAgentRepoNames`, `SubAgent` in `tools.ts`) and the "Repo Agent" is a
documented stakgraph capability. Worth studying for the pattern (but remember
the self-containment rule — don't import it; the lab version reimplements the
loop inline, as `explore-services` already does).

---

## Open questions (decide before/while building)

1. **Scorer location**: gitsee-specific step (`gitsee/score-setup`) vs a reusable
   generic `eval/score-structured` primitive that takes pluggable extractors?
   Recommendation: **gitsee-specific** — pm2/compose extraction is domain logic;
   keep `eval/*` clean. Revisit if a 2nd experiment needs structural scoring.

2. **Pure-deterministic vs hybrid**: keep the LLM judge for the semantic residue,
   or go fully deterministic (env keys + service set + a regex/AST check on the
   start command) and drop the LLM entirely for gitsee? Fully deterministic =
   free, instant, zero variance (great for many optimize iterations) but can't
   judge "is this the right start command." Recommendation: **hybrid**, with the
   deterministic env/service sub-scores dominant; measure whether the LLM part
   even moves the needle and consider dropping it.

3. **Which gold env keys are REQUIRED?** Some gold keys may be non-essential
   (e.g. `STAKWORK_WORKFLOW_ID`). Options: (a) treat *every* gold key as required
   (strict — simplest, penalizes any omission); (b) let the gold mark optional
   keys (e.g. a sidecar `optionalEnv: [...]` or a `# optional` convention);
   (c) infer essentiality from the repo (out of scope, too fuzzy). Which?

4. **Score weighting**: how to combine env-completeness recall, service-set
   recall/precision, cross-file consistency, and the LLM semantic score into one
   number? Keep the recall-weighted F-beta (β=2) shape but over what item set?
   Propose: treat env keys + services + key commands as the unified "expected
   item" set for one F-beta, and fold the LLM semantic verdict as a
   multiplier/penalty. Needs a concrete formula — propose one in the PR.

5. **pm2 JS eval safety**: eval'ing LLM-generated `module.exports` in a `node:vm`
   sandbox (no network, stub `require`) — acceptable? It's our own generated
   content. Alternative: regex/Acorn-parse the `env: { ... }` block without
   executing (more brittle, no exec). Recommendation: **vm sandbox** with a hard
   timeout + stubbed globals; fall back to "score 0 / unparseable" on throw.

6. **Cross-file consistency check** (DB creds match between pm2 env and compose
   service) — include in Phase 1, or defer? It's deterministic and boot-critical.
   Lean: include if cheap.

7. **Sub-agent decomposition** (Phase 2): is services / env / commands / deps the
   right split? Do sub-agents share one clone (read-only) or get scoped to a
   sub-tree? Do we want **per-area golds + per-area eval**, or keep whole-config
   eval and only split the *exploration*? (Per-area eval needs authoring per-area
   golds — more work, but the tightest loop.)

8. **Cost/latency budget**: the eval is in the hot loop of `gitsee-optimize`
   (gens × workspaces × ~3 min). The structured scorer should make scoring
   *cheaper* (less/no LLM). Sub-agents in parallel could make *exploration*
   faster or slower — keep an eye on wall-clock per eval; it caps how many
   optimize generations are practical.

---

## Out of scope (for this phase)

- Top-level **dataset entity** (separate versioned store + reference resolution)
  — deferred; datasets stay in `params.dataset` for now. (The Phase 1 YAML param
  editor already makes them editable.)
- Actually *running* the generated pm2/compose to verify boot (executable eval) —
  the real gold standard, but needs the pod/docker; out of scope here.
- Multi-provider/gateway LLM routing — gitsee stays Anthropic-direct.

---

## Key references

- Scorer to replace: `mcp/src/lab/eval/steps/score.ts` (the generic LLM judge +
  the F-beta math to reuse).
- Optimize/reflect contract: `mcp/src/lab/eval/steps/optimize.ts`,
  `mcp/src/lab/eval/steps/reflect.ts`.
- Producer + tools to (later) decompose: `mcp/src/lab/gitsee/steps/explore-services.ts`.
- Workflows: `mcp/src/lab/gitsee/workflows/gitsee-eval*.yaml`, `gitsee-optimize.yaml`.
- Gold format / dataset: `gitsee-optimize.yaml` `params.dataset` (hive +
  heroku-node); the canonical hive gold is `mcp/src/lab/gitsee/confs.md` (gitignored).
- Seeding + self-containment + workspace gotcha: `mcp/src/lab/AGENTS.md`.
- Dev harnesses: `mcp/src/lab/gitsee/smoke.ts` (steps direct),
  `smoke-eval.ts <label>` (full `gitsee-eval` via a real lab vein).
