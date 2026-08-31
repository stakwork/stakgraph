# lab

A sandbox for **vein workflow experiments**. The goal: take pipelines that
are normally hardcoded TS and express them as editable vein workflows
(YAML + step config) so we can iterate on flows and prompts without code
changes. See `../../../vein/AGENTS.md` for the engine itself.

## Model

One vein instance for the whole lab (`createLabVein.ts`), mounted into the
Express app at **`/lab`** (`mount.ts`). Experiments are just groups of
workflows/steps inside it — **not** separate servers. Adding an experiment
= register its steps + merge its services + seed its workflows in
`createLabVein`.

- `createLabVein.ts` — the single instance: registry (vein core+lib + all
  experiment steps), merged `services` bag, seeded workflow templates.
- `mount.ts` — bridges the vein (Hono) app into Express under `/lab`
  (API + run-streaming SSE). Registered before `express.json()` to keep
  raw request streams. Lazy-initialized so mcp boot isn't coupled to
  Neo4j / LLM keys.

## Experiments

### `concepts/` — GitHub history → "Concept" knowledge graph

A clue-free, Graph-only port of `mcp/src/gitree` (renamed `Feature` →
`Concept`, new `Concept` Neo4j label; reuses `PullRequest`/`Commit`/`File`).
Walks a repo's PRs + commits chronologically, LLM-classifies each into
concepts, summarizes, and links to code files.

- `concepts/steps/` — the in-code vein steps (one unit of work each).
- `concepts/workflows/` — `process-change`, `process-repo-chronological`,
  `bootstrap-then-process` (the entry workflow).
- `concepts/services.ts` — `ConceptServices` bag `{ storage, octokit, llm,
  summarizer }` injected via `ctx.services`.
- `concepts/pipeline.ts` — pure helpers ported from gitree's builder
  (fetch changes, checkpoints, decision prompt, apply decision).
- `concepts/store/` — Graph (Neo4j) store; `llm.ts`, `bootstrap.ts`,
  `summarizer.ts`, `pr.ts`, `commit.ts` — ported domain logic.

Experimentation seams (edit without touching code):
- **prompts** → `process-change` workflow's `params` block
  (`systemPrompt`, `guidelines`); the `concepts/decide` step consumes
  them via `{{ params.* }}` config. Sweep by running `process-change`
  with a per-run `params` override; promote a winner by editing the
  `params` default and publishing a new workflow version.
- **ordering** → `concepts/prioritize-changes` strategy (swap the step)
- **orchestration** → fork the top-level workflow (chronological vs
  bootstrap vs future adaptive/loop variants)

### `gitsee/` — setup profiler (port of `mcp/src/gitsee`)

A port of `mcp/src/gitsee`'s "services" mode (the agent that emits a
`pm2.config.js` + `docker-compose.yml` to set up a project). No import from
existing code (`src/gitsee`), so that dir can eventually be deleted. The agent
loop itself is **not** gitsee code anymore — it's the **vein-core `agent`
step** (see `vein/AGENTS.md`); gitsee supplies the clone + the prompts, and (for
the product loop) a `gitsee` **`services` bag** (`gitsee/services/`: per-run
browser + stack managers + a vision judge) that its thin tool-steps reach via
`ctx.services.gitsee.*` — see "The product loop" below.

**WORKSPACE-oriented** (not single-repo): a workspace is a set of repos cloned
as **siblings** under `/workspaces/<repo>` — typically one runnable **frontend**
plus local-dependency repos it builds against (file: deps). The goal is to *get
the frontend running*, so the gold is the frontend's pm2 + the shared services.

- `gitsee/steps/clone-workspace.ts` (`gitsee/clone-workspace`) — clones N repos
  as siblings under one workspace dir (idempotent, per-rev). Each repo may pin a
  `rev`; `token` falls back to `GITHUB_TOKEN` env (private repos). `clean`
  (default true) resets a REUSED clone to a pristine working tree
  (`git reset --hard && git clean -fd`) so each run / optimizer generation starts
  fresh — discarding the prior explore agent's edits + created files (keeps
  gitignored `node_modules` for speed). Output `{ workspacePath, repos }`. **The
  only gitsee-specific producer step.**
- Exploration is the **core `agent` step** (`vein/src/steps/core/agent.ts`),
  pointed at `cwd = clone.workspacePath`. Its general tools (`repo_overview`,
  `fulltext_search`, `bash`, `str_replace_based_edit_tool` — view/create/edit
  files in the cloned workspace (lets the agent make a repo local-first, e.g.
  flip a `USE_MOCKS` default or patch a hardcoded cloud URL), anthropic
  `web_search`, + `file_summary` — the `stakgraph` AST CLI, only offered when
  `stakgraph` is on PATH) + the agent loop
  live in vein core now — what was the old inlined `gitsee/explore-services` step
  (deleted). gitsee runs it in **`finalAnswer` (FILENAME text) mode**; the
  structured-`schema` mode is intentionally unused here for now. (For the
  `file_summary` tool, `stakgraph` must be on PATH; the agent falls back to
  `bash`/`cat` otherwise.)
- `gitsee/workflows/gitsee-explore-services.yaml` — `clone → agent → capture`;
  input `{ workspace, repos: [{owner,repo,rev?}], token? }`; the `system`/`prompt`/
  `finalAnswer` prompts live in `params` (the experiment surface, frontend-
  focused). The agent injects a neutral working-dir listing, so the prompts
  stay repo-agnostic (workspace framing moved into `params.system`). The agent
  MAY edit the cloned repos (via the core `str_replace_based_edit_tool`) to make
  them boot local-first (move a misplaced migration, flip a mock flag, patch a
  cloud-only URL); those edits are captured by the final `gitsee/capture-edits`
  step as a replayable `git diff` and SHIPPED as part of the deliverable (output
  `diff` + `changedRepos`, alongside the passed-through `result`/`usage`/`cost`).
  The split: FILE changes go in the repo (the diff re-applies on a fresh pod
  clone), RUNTIME steps (db reset/migrate/seed) go in `PRE_START_COMMAND` — so a
  Supabase-style "move the migration + reset the db" becomes a clean diff hunk +
  a clean PRE_START, not a file-shuffling shell hack. The agent also narrates its
  edits in a `## CHANGES` section of the final answer (human-readable companion to
  the diff). (`capture-edits` is the LAST step because a workflow's output is its
  last step's output; it passes the agent's fields through so `produce.result`
  etc. still resolve.)

**The boot gate (`gitsee/verify-setup`) — the dominant eval signal.** A setup
that doesn't actually *run* is a failure no matter how well its files match the
gold, so the eval now RUNS the produced pair the pod way and proves the frontend
loads. `gitsee/verify-setup.ts` (`gitsee/verify-setup`): stages the produced
`pm2.config.js` + `docker-compose.yml` into the cloned workspace exactly where
**staklink** looks (`<root>/pm2.config.js` + `<root>/.pod-config/.user-dockerfile/
pm2.config.js`), rewriting the pod-absolute `cwd: /workspaces/<repo>` to the local
clone root; `docker compose up -d --wait` for the backing services; boots the apps
via **staklink** (`npx staklink start` → REBUILD→INSTALL→PRE_START→`pm2 start`→
POST_START — pod-faithful; it does NOT run BUILD_COMMAND, so dev-mode boot is the
target) or a pm2-free inline fallback (`useStaklink:false`); polls the frontend
`PORT`; then loads `http://localhost:<port><checkPath>` in headless chromium
(`@playwright/test`), screenshots to `<root>/.verify/render.png`, and **judges that
screenshot with a VISION model** (`useVision`, anthropic, default on) — the real
"did it render" signal, since an HTTP 200 + non-empty DOM still passes for a white
screen or a styled error page. It asks the model whether the intended app UI
rendered vs a blank/error/404-500 page; an HTTP-status + error-overlay heuristic is
the fallback when vision is off or unavailable. Output `{ booted, rendered, port,
httpStatus, title, reason, logs, screenshotPath, cost, usage }` (cost = the
vision-judge tokens, folded into the eval total via `produce.cost + verify.cost`).
Missing browsers degrade to a boot-only gate (`rendered:null`); `enabled:false`
makes it a no-op (skip the gate in cheap sweeps). Needs `docker` + `git` on PATH, `npx playwright install chromium`
for the render check, and (for `useStaklink`) network for `npx staklink`. **The
agent can also EDIT the cloned repos** (via the core agent's
`str_replace_based_edit_tool`) to make a repo local-first before this gate runs.

**Teardown.** On a normal/errored finish the step's `finally` removes everything:
`pm2 delete all`, `staklink stop`, `docker compose down -v`, AND — via a container
SNAPSHOT taken just before boot — `docker rm -fv` of every container that appeared
during the run. That snapshot-diff is what catches **app-spawned** stacks our
compose file never declared: a `supabase start` CLI project (~12 `supabase_*`
containers), a minio, etc. **But teardown only runs if the process finishes** —
if you KILL the run (Ctrl-C / kill the optimize), the `finally` never fires and the
booted stack is left up. Clean it with `npx tsx src/lab/gitsee/cleanup.ts`
(removes the stale pm2 procs, supabase CLI stacks, and gitsee-lab compose
projects; leaves Neo4j etc. alone). `keepUp:true` intentionally skips teardown for
debugging.

**The product loop (`gitsee-setup-and-run`) — NOT an eval signal.** Where
`verify-setup` is a READ-ONLY boot GATE (boot once → one screenshot → one vision
verdict → score), `gitsee-setup-and-run` is the autonomous "set up a repo until it
actually runs" loop: an agent that BOOTS the app, DRIVES the live frontend in a
real headless browser, OBSERVES failures like a QA engineer, FIXES the cause,
REBOOTS, and repeats until functional. Because it WRITES/FIXES, it must **NOT** be
wired into the scored `gitsee-optimize` loop (fixing in place erases the gradient
that teaches the explorer); the deliverable is a known-good `setup` + `diff` +
`report`, not a grade.

**Architecture: this is now DECOMPOSED onto the vein-core `agent` step** (it used
to be one ~1050-line `gitsee/boot-and-exercise` step that forked the whole agent
loop). See `vein/plans/agentic-loop-as-workflow.md` for the full design. The
pieces:

- **The QA harness = a gitsee `services` bag** (`gitsee/services/`, in-code,
  merged into `LabServices` by `createLabVein`; NOT seeded): `BrowserManager` +
  `StackManager` (per-run sessions keyed by `runId`) + a stateless `vision` judge.
  `_infra.ts` holds the shell/pm2/compose/pod-url/port/log helpers; the
  per-run state (browser page, booted stack, last vision verdict) lives on the
  session. **Teardown is automatic**: `LabServices.onRunEnd(runId)` (the generic
  vein hook the runner calls in a `finally`, success OR error) disposes the run's
  browser + booted stack — no teardown code in any step. (`cleanup.ts` is still
  the rescue for a hard `SIGKILL`, which skips the in-process `finally`.)
- **The capabilities = thin seeded tool-steps** (`gitsee/steps/`) reaching the
  harness via `ctx.services.gitsee.*`, keyed by `ctx.runId`: `gitsee/stage-setup`
  (stage the produced files + create the per-run stack session), `gitsee/boot`
  (re-stage + compose up + staklink/pm2 + wait for port), `gitsee/browser-open` /
  `-snapshot` / `-click` / `-fill` / `-press`, **`gitsee/browser-observe`** (drains
  console errors + failed requests + **4xx/5xx API responses** — the "renders but
  is broken" signal a screenshot can't show), `gitsee/assess-ui` (the "eyes":
  screenshot + errors + logs → vision verdict, recorded on the stack session),
  `gitsee/read-logs`, and `gitsee/finalize-setup` (the deliverable: pod-portable
  `setup` + per-repo `git diff` + the booted/working verdict). Each works BOTH as
  a workflow step AND as an `agentTools` tool.
- **The loop = the core `agent` step** with those steps as `agentTools` (+
  built-in `bash`/`str_replace_based_edit_tool` for edits). `gitsee-setup-and-run`
  is `clone → produce (gitsee-explore-services) → stage → qa(agent) → finalize`.
  Every tool call emits a nested run event (the agent step's `wrapToolsWithEmit`),
  so **each iteration is visible in the UI events panel / run drill-down**. The QA
  `system` prompt + `agentTools` list + `model` + `maxSteps` live in the
  workflow's `params` (the harness/policy split — a future `gitsee-setup-optimize`
  can sweep the prompt with the harness fixed).

  *Why agent-orchestrated, not a deterministic `loop` step:* vein's `loop` THROWS
  on `maxIterations`-without-convergence (`runner.ts`), which would error the run
  and yield NO deliverable when an app can't be fixed; and QA control flow is
  inherently dynamic. The agent's own tool loop still gives per-iteration
  visibility, preserves autonomy, and always finalizes.

**Pod URLs (`$POD_ID`/`$POD_URL`)** are kept in the deliverable (the pod contract)
but **localized only in the staged-for-boot copy** (`podSubstituteLocal` in
`services/_infra.ts`: `https://$POD_ID-<port>.<domain>` → `http://localhost:<port>`,
`$POD_URL` → `http://localhost:<frontendPort>`) — on the real sandbox the platform
expands them + proxies `<podid>-<port>.<domain>` to `localhost:<port>`; locally
there's no proxy, so we emulate it. The QA system prompt tells the agent these are
auto-substituted and to KEEP them.

Output `{ booted, working, reason, port, setup, report, diff, changedRepos,
changed, screenshotPath }`. Needs `docker` + `git` + `npx playwright install
chromium`.

**`gitsee/boot-and-exercise` (the old monolith) is still seeded but UNUSED by the
workflow** — kept as the A/B reference to validate the decomposed loop reaches
parity before deletion (plan Step E). Its dev smoke `src/lab/gitsee/smoke-boot.ts`
still drives that step directly; `src/lab/gitsee/services/smoke-services.ts` is a
no-docker lifecycle smoke for the new harness (per-run keying + `onRunEnd`).

**Eval/optimize stack** (mirrors `concepts-*`; reuses the generic `eval/*`
steps EXCEPT scoring, which is gitsee-specific — see below). The gold is the
**actual canonical pm2.config.js + docker-compose.yml pair** (produced vs gold
is apples-to-apples), but the **boot result dominates** (see below).

Scoring is a **structured + hybrid** scorer (`gitsee/score-setup`), NOT the
generic LLM `eval/score`. The **dominant tier is now the boot gate**:
`score-setup` takes `booted`/`rendered` from `gitsee/verify-setup` and clamps the
file-shape score — `!booted` → ×0.15 (it didn't even run), booted-but-not-rendered
→ ×0.5, booted+rendered → full. (Null/absent leaves the score untouched, so a
verify-free run still works.) BELOW that gate, both files are parseable and the
gold is the ANSWER KEY — so the file-shape score is **deterministic name set-diffs
vs the gold**, which is why it stays repo-agnostic without understanding any
dependency:
- **env-key completeness** — `keys(produced pm2 env)` vs `keys(gold pm2 env)`,
  recall-weighted. Robust + general because the key NAME is dictated by the
  repo's code (`process.env.X`); a different name simply isn't read, so the
  gold's names are canonical. (Build/run directives like `INSTALL_COMMAND` live
  in this env block too, so "key commands" come for free.) This is the headline
  fix: the old LLM judge collapsed hive's ~15 env vars into ~1 "env present"
  item, so a missing boot-critical key barely moved the score.
- **service set** — compose service IDENTITY (image base name, tag stripped, or
  the service name for build-only services) produced vs gold. An extra image the
  gold lacks (e.g. an invented `redis`) is a precision hit → catches
  over-provisioning.
- **LLM semantic residue** (optional, `useLLM`, capped multiplier so the
  deterministic tier dominates) — only what needs interpretation and therefore
  can't be a name set-diff: is each `script` the right start command, is a
  host-binding flag present when the framework needs one, do the pm2 DB creds
  line up with the compose service (naming-agnostic), is an added service
  appropriate. (Cross-file cred consistency lives HERE, not in the deterministic
  tier — matching a `DATABASE_URL` to `POSTGRES_*` across every datastore's env
  conventions doesn't generalize deterministically; the LLM reads both files
  regardless of naming.)
- pm2 is eval'd in a locked-down `node:vm` (stub `require`/`process`, 1s timeout
  → "unparseable / score 0" on throw); compose via `js-yaml`. Combine into one
  recall-weighted F-beta (β=2) over env keys ∪ services, then apply the bounded
  semantic multiplier.

`gitsee/score-setup` PRESERVES the scorer contract `{ score, recall, precision,
matched, missing, spurious, reason, insight, markdown }` that `eval/optimize` +
`eval/reflect` depend on.

- `gitsee-eval` — harness: clone (`gitsee/clone-workspace`) → produce (subflow →
  `gitsee-explore-services`, re-clones the same idempotent path) → **verify**
  (`gitsee/verify-setup` — the boot gate, using `clone.workspacePath`) → score
  (subflow → `gitsee-eval-score`, threaded `booted`/`rendered`). No reset step
  (gitsee is stateless). Input `{ label, repos, token?, expected? }` — `label` is
  the workspace name (and the `eval: <label>` link); `expected` gold falls back to
  `params.expected`. Boot-gate knobs in `params`: `verify` (default true — set
  false for cheap docker-free sweeps), `checkPath`, `bootTimeoutMs`, `useStaklink`.
- `gitsee-eval-score` — `gitsee/score-setup` + the matching policy in `params`
  (`useLLM`, `ignoreEnvKeys`, the semantic-residue `rubric`); passes through the
  `booted`/`rendered`/`bootReason` boot gate. Strict env policy: every gold env key
  is required (exempt noise keys via `ignoreEnvKeys`).
- `gitsee-eval-reflect` — `eval/reflect` + the setup task/guidance.
- `gitsee-optimize` — `eval/optimize` loop. Tunes **`system`** (the explorer
  prompt), NOT `finalAnswer` (the hard pod contract). Cohort in `params.dataset`,
  one entry per WORKSPACE: `{ label, repos: [{owner,repo,rev?}], expected }`.

**Cost accounting.** Every LLM call in the loop reports its token usage + dollar
cost, summed into the optimize output's `{ totalCost, totalUsage }` (and each
`generations[]` entry's own `{ cost, usage }`). The chain: the core `agent` step
returns `{ usage, cost }` (aggregated across its whole tool loop, priced via
`vein/src/pricing.ts` — table copied from `aieo/src/provider.ts`); gitsee-eval
threads that into `gitsee/score-setup`, which folds in its OWN semantic-judge
tokens+$ so each eval's `cost` is explorer + judge; `eval/reflect` returns its
reflection's cost; `eval/optimize` sums eval runs + reflections per generation
and run-wide. So a detached optimize job records exactly what it burned. (Set
`gitsee-eval-score`'s `useLLM:false` to drop the judge LLM cost entirely.)

Dataset: `heroku-node` (1-repo Express, verified 0.95) + `hive` (Next.js +
Postgres + Prisma; `hive` pinned, with sibling dep repos sphinx-voice /
system-canvas / staklink). Add more workspaces for a stronger multi-example
optimize (EVAL_SPEC §11.2).

Needs `ANTHROPIC_API_KEY` + `git` + `rg` on PATH (Neo4j only for booting the
lab, not for gitsee itself). The **boot gate** additionally needs `docker` on
PATH, `npx playwright install chromium` (browsers; otherwise the gate is
boot-only), and network for `npx staklink` — set `params.verify=false` to skip it
entirely. Trigger:
`POST /lab/workflows/gitsee-explore-services/run` with
`{ input: { workspace, repos: [{owner,repo,rev?}], token? } }`, or launch
`gitsee-optimize` detached with `{ input: {} }`. Dev smoke harnesses (not
seeded/built): `src/lab/gitsee/smoke.ts` (steps direct, no server) and
`smoke-eval.ts` (full `gitsee-eval` via a real lab vein).

### `jarvis/` — knowledge-graph steps (NOT an experiment)

Self-contained ports of the mcp repo-agent's Jarvis tools
(`mcp/src/repo/toolsJarvis.ts`) as seeded vein steps — same endpoints, same
schemas, same LLM-facing descriptions — so workflows (and agent steps) can
read/write the Jarvis knowledge graph. **Concepts are Jarvis nodes** (filter
`type: "Concept"` on search/neighbors), so no concept-specific steps exist.

- **Reads:** `jarvis/get-ontology`, `jarvis/get-ontology-type`,
  `jarvis/graph-search` (hybrid + field-scoped vector search),
  `jarvis/graph-get`, `jarvis/graph-get-batched`, `jarvis/graph-neighbors`.
- **Writes:** `jarvis/create-node`, `jarvis/edit-node`,
  `jarvis/create-triplet`, `jarvis/create-batch-triplet`. The ontology CRUD
  family is deliberately NOT ported (schema editing stays a human/setup
  activity).
- **Config is automatic:** each step resolves `JARVIS_URL` + `API_TOKEN`
  (+ optional `JARVIS_HTTP_TIMEOUT_MS`) through `ctx.services.secrets`
  (secret store → env fallback) and calls through `ctx.services.http` — so
  runs are cassette-recordable and credentials are scrubbed from fixtures.
  Steps are ALWAYS seeded; without `JARVIS_URL` they fail loudly per run
  rather than silently missing.
- **Granting to agents:** `agentTools: ["jarvis/*"]` (glob, vein-core
  `expandAgentTools`) for everything, or list the read steps explicitly for a
  read-only child. Sub-agents = grant `"agent"` itself and pass the child a
  narrower `agentTools` list (recursion depth is whether the child gets
  `"agent"` again).
- **Self-contained duplication is deliberate:** each step file inlines its
  small `jarvisCtx` preamble (seeded steps may only value-import `"vein"`);
  the contract is documented once in `jarvis/steps/_shared.ts` — change it
  there AND in every step.
- **Smoke:** `npx tsx src/lab/jarvis/smoke.ts` — offline; seeds into a temp
  workspace, verifies registry discovery, and runs every step against a fake
  `ctx.services.http` Jarvis.

### `sheets/` — Google Sheets steps (NOT an experiment)

Self-contained ports of the mcp repo-agent's Google Sheets tools
(`mcp/src/repo/toolsGoogleSheets.ts` — untouched; it stays the production
repo-agent implementation) as seeded vein steps — same Sheets/Drive REST
endpoints, same schemas, same LLM-facing descriptions — so workflows (and
agent steps) can create spreadsheets and read/write cell values and live
formulas.

- **Steps:** `sheets/create-spreadsheet`, `sheets/update-values`,
  `sheets/batch-update-values`, `sheets/get-values`, `sheets/add-sheet`,
  `sheets/import-spreadsheet` (best-effort per-sheet import of an .xlsx or
  native Sheet into a destination spreadsheet, with auto-conversion,
  collision-suffixed tab names, and cross-sheet-formula warnings).
- **Config resolution:** `cfg.serviceAccount` (explicit step config — parsed
  JSON, JSON string, or base64 JSON) wins, else the
  `GOOGLE_SERVICE_ACCOUNT_JSON` secret (secret store → env fallback); same
  for `cfg.driveFolderId` / `GOOGLE_DRIVE_FOLDER_ID` (spreadsheets are
  created inside that folder — share it with the service account's
  client_email so humans can see agent-created sheets). Auth is a plain
  service-account JWT flow: an RS256 assertion built with `node:crypto` (no
  jsonwebtoken/axios deps), exchanged at the SA's `token_uri` for a bearer
  token (scopes `spreadsheets` + `drive`), cached per step module with 60s
  expiry slack. Everything goes through `ctx.services.http` +
  `ctx.services.secrets`, so runs are cassette-recordable and credentials
  are scrubbed from fixtures. Steps are ALWAYS seeded; without
  `GOOGLE_SERVICE_ACCOUNT_JSON` they fail loudly per run rather than
  silently missing. API errors come back as teaching strings (e.g. a 403 on
  create names the folder and client_email to share it with), never throws
  at the LLM.
- **Granting to agents:** `agentTools: ["sheets/*"]` (glob, vein-core
  `expandAgentTools`) for everything, or an explicit subset — e.g. a
  read-only child gets just `sheets/get-values`.
- **Self-contained duplication is deliberate:** each step file inlines its
  `sheetsCtx` auth/request preamble (seeded steps may only value-import
  `"vein"` + node builtins); the contract is documented once in
  `sheets/steps/_shared.ts` — change it there AND in every step.
- **Smoke:** `npx tsx src/lab/sheets/smoke.ts` — offline; seeds into a temp
  workspace, verifies registry discovery, then runs every step against a
  fake `ctx.services.http` Google (real RSA keypair so the JWT signature is
  verified; covers token caching, bearer headers, round-trip shapes, the
  loud missing-credentials error, and a teaching-error case). No live
  Google call has been made yet — end-to-end verification with a real
  service account is pending.

### `harvey/` — Harvey LAB verification (the hardcoded grader)

Runs the **actual** Harvey LAB legal-benchmark eval (the
`/Users/…/harvey-labs` checkout's `uv run python -m evaluation.run_eval`) as
a subprocess. Nothing is ported and nothing is editable: the grader lives in
the in-code `harvey` **service** (`harvey/service.ts`, on the `LabServices`
bag), NOT in a seeded step — the workflow-authoring agent must never be able
to edit its own grader. The two seeded `harvey/*` steps are thin plumbing
over `ctx.services.harvey.*`; editing them can only break plumbing.

- **Integrity invariant** (enforced per grade, not per boot): the checkout
  must be a CLEAN git tree, and when `HARVEY_LABS_REV` is set, HEAD must
  match it — otherwise `evaluate` refuses. Untracked `results/` entries (our
  own staged runs) are tolerated. Every result carries `benchmarkRev` (the
  exact SHA) so scores are attributable to a benchmark version.
- `harvey/get-task` — a task's title/instructions/deliverable names + input
  documents listing, with the grading rubric (`criteria`) **stripped in the
  service** — a producing agent must never see how it will be graded. Safe to
  grant to producers.
- `harvey/evaluate` — stages this run's artifact deliverables (subdir `from`,
  default `output`, of `ctx.services.artifacts.dir(ctx.runId)`) into the
  checkout's `results/vein-<runId>/output/`, runs the real eval (single judge
  or `dual`), and returns the harness's own `scores.json` (all-pass scoring,
  `criteria_results`, …) + `benchmarkRev` + `reportPath` (the harness's
  `report.html`, kept in the checkout's `results/` as the run's record).
  **GUARDRAIL: grant only to harness workflows — NEVER to the producing
  agent's `agentTools`** (an agent that can query its own grader mid-task
  trains against the rubric).
- **Config (env):** `HARVEY_LABS_DIR` (checkout path; loud per-run error when
  unset — same posture as jarvis), optional `HARVEY_LABS_REV` pin. The eval
  subprocess inherits mcp's env (`ANTHROPIC_API_KEY`; plus `OPENAI_API_KEY`
  for `dual`) and needs `uv` + `git` on PATH (+ pandoc per harvey-labs docs).
- **Smoke:** `npx tsx src/lab/harvey/smoke.ts` — offline (real throwaway git
  repo as a fake checkout, fake `uv` exec): integrity enforcement (dirty
  tree / rev pin / missing dir), rubric stripping, artifact staging, CLI
  args, step wiring.

**The DELIVER pipeline (`harvey-deliver`) — the production-style port, NOT
part of the benchmark harness.** A lab-shaped recreation of the stakwork
Harvey LAB workflow (normalize → register namespace → EvalSet/requirements →
per-doc graph ingestion → checklist/fact base/case law/tailor → drafters ×N →
4 parallel verifiers → aggregator → per-criterion LLM judge → dispute →
persist eval chain → webhook → recursion gate). STANDALONE: the rubric
(`[{ id, title, match_criteria, deliverables }]`) is a RUN INPUT and the
pipeline self-scores against it — so it must NEVER be wired as a harvey-run
produce candidate (candidates may not see a rubric). Documents come from the
local harvey-labs checkout via `harvey/get-task`; the graph namespace AND
EvalSet id are the slugified task id, so reruns reuse the namespace, skip
completed ingestions, and skip re-merging requirements.

- Workflows: `harvey-deliver` (orchestrator) → subflows `harvey-ingest-doc`
  (per document; Document node keyed on `source_link`, dedupe via a
  `status: "ingested"` COMPLETION MARKER written only after a successful
  agent run — a partially-failed ingestion is retried, not skipped),
  `harvey-knowledge` (checklist writer → cross-check fact base [GRAPH-ONLY
  reads — no bash, deliberately blind to source docs; unregistered findings
  land as ScratchpadEntry via `allow_scratchpad`] → case-law research
  [SerpAPI/CourtListener keys via the agent step's `secretsEnv` —
  GOOGLE_SERPA + COURTLISTENER_API_KEY reach bash as env only] →
  tailor [ADDITIVE, then checklist.md freezes]), `harvey-draft` (drafter
  foreach → 4 explicit fall-soft verifier agents → aggregator into
  `./output/`), `harvey-score` (validate exact deliverable names [hard
  gate] → filter contested [fails open] → judge foreach → aggregate →
  RECORD the eval chain EvalSet→EvalTrigger→EvalTriggerOutput→
  CriterionResult per the jarvis schema_library ontology → dispute foreach
  [root-cause audit: flagged=verdict wrong, contested=criterion defective;
  annotates + writes Cause triplets onto the recorded CriterionResult refs;
  never flips verdicts] → merge [fail-soft] → write annotations back onto
  the CriterionResult nodes + contested onto EvalRequirements → webhook →
  all-pass ⇒ `EvalSet.recursion=false`), and `harvey-judge-criterion` /
  `harvey-dispute-criterion` (per-criterion agent-in-schema-mode subflows —
  they exist because foreach bodies get no onError; a judge crash scores as
  an honest FAIL via `harvey/aggregate-scores`' zip, never a pass).
- Steps: `harvey/normalize-documents`, `harvey/ingest-state`,
  `harvey/drafter-plan`, `harvey/validate-deliverables`,
  `harvey/filter-contested`, `harvey/aggregate-scores`,
  `harvey/merge-disputes`, `harvey/build-eval-chain`,
  `harvey/criterion-refs` (pure/plumbing), `harvey/generate-docx` +
  `harvey/generate-xlsx` (pandoc / openpyxl deliverable generators,
  granted as agent tools so the production prompts' generate calls work),
  `harvey/graph-sub-agent` — the PINNED read-only graph research sub-agent
  (wraps the core `agent` step via ctx.registry with fixed system frame +
  read-only jarvis grants; parents supply only the question, so a role agent
  can never widen the child's tools) — and `jarvis/register-namespace`
  (+ `allow_scratchpad` passthrough added to `jarvis/create-triplet` /
  `create-batch-triplet`).
- **Prompts are the VERBATIM production texts**, kept as markdown files in
  `harvey/prompts/` and spliced into step configs at seed time via
  `@@include(FILE.md)` markers (`expandIncludes` in `harvey/seed.ts`,
  indentation-aware; the content hash covers the expanded YAML, so editing
  a prompt file re-seeds its workflows). Stakwork `[$(step).output.*]`
  interpolation tokens were translated to vein `{{ … }}` templates (which
  is why prompt bodies live in step CONFIG, not params — template
  resolution is single-pass, so a `{{ }}` inside a params value never
  resolves), tool names to the lab step names (`jarvis_graph_search`,
  `harvey_graph_sub_agent`, `harvey_generate_docx`, `sheets_*`…), and
  container paths to `./`. The raw exports + transform live in git history
  (`notes/harvey_prompts`, scratchpad transform). EvalRequirement ids
  follow the production convention `<task_slug>-<criterion_id>`.
- Judge verdict identity is ORDER-BASED (foreach preserves input order; the
  aggregate zips rubric×results and refuses on length mismatch) — the judge
  LLM never echoes criterion ids.
- Needs `JARVIS_URL` + `API_TOKEN` + `HARVEY_LABS_DIR` + `ANTHROPIC_API_KEY`
  (+ pandoc, python3+openpyxl; optional GOOGLE_SERPA + COURTLISTENER_API_KEY
  for case-law research, GOOGLE_SERVICE_ACCOUNT_JSON for the shared FACTS
  spreadsheet). Smoke (offline, no LLM/graph):
  `npx tsx src/lab/harvey/deliver-smoke.ts`.

### `gaia/` — GAIA benchmark scoring (the hardcoded grader)

Scores answers with the **actual** GAIA leaderboard scorer (`scorer.py`,
quasi-exact match with type-aware normalization) as a `python3 -c` subprocess
against the validation split's gold answers. Same discipline as harvey: the
grader AND the gold live in the in-code `gaia` service (`gaia/service.ts`,
on the `LabServices` bag), never in a seeded/authored step. `gaia/*` steps
are thin plumbing over `ctx.services.gaia.*`.

- **Committed harness** (`gaia/seed.ts`, seeded at boot like harvey's):
  steps `gaia/list-tasks`, `gaia/get-task` (stages a task's attached file
  into the run's artifacts dir), `gaia/evaluate` (HARNESS-ONLY), and the
  combiners `gaia/pack-result` + `gaia/summarize-batch`; workflows
  `gaia-produce` (agent step; the produce system prompt, model, maxSteps: 50
  and agentTools live in `params`; an `onError` fallback scores a blown-up
  agent as an empty wrong answer instead of killing the batch), `gaia-run`
  (single task: produce → score; `input.produceWorkflow` swaps in a seeded
  produce variant) and `gaia-batch` ({ level, limit }: one
  score call for the whole batch). This is the harness that went 1/5 → 5/5
  on the level-1 batch (EVOLVE_SPEC §1), promoted from the workspace where
  the assistant authored it. Seeding is content-hash reconciled — the
  committed copy is authoritative at boot, so a workspace-side evolution
  survives restarts only once it's ported back here. The from-scratch
  authoring recipe is kept in `notes/GAIA.md` as an authoring eval; it is no
  longer the path to a working harness.

- **Evolve harness** (mirrors harvey's, on the generic `eval/evolve-loop`):
  `gaia/digest-results` (verdict-channel digest — accuracy as `fitness`,
  misses tagged wrong-answer / empty-answer / produce-error per EVOLVE_SPEC
  §8's taxonomy, candidate answers + question excerpts, never gold),
  `gaia-candidate-run` (runs an ai-stamped candidate on one task via
  `meta/run-workflow`, scores its reported answer via `gaia/evaluate`'s
  `fromRun` unpack — a failed run is an honest zero), `gaia-evolve-gen`
  (one generation: meta/* author → pinned candidate over the task set,
  `produceConcurrency`-wide → digest) and `gaia-evolve` (capture →
  hill-climb → report). The capture stage runs the baseline
  `baselineSamples` times (nested foreach, numeric outer items) and folds
  the samples with `eval/matrix`: the matrix object IS the loop's baseline
  (`fitness` = the MAX sample, the conservative bar) and its MEASURED
  produce-sampling floor (`noise.suggestedMargin`, the max fitness delta
  between identical-YAML re-runs) becomes the loop's `improveMargin` —
  `params.improveMargin: 0` is only the fallback when the floor is
  unmeasured (`baselineSamples: 1`, the pre-Phase-1 behavior).
  Candidate contract: input `{ taskId }`, last step outputs `taskId`,
  `answer` (bare string), `cost`, `steps`; candidates may use
  `gaia/get-task` / `gaia/pack-result` as steps but NEVER `gaia/evaluate`
  (produce-time oracle) and never gaia/*, eval/*, meta/* as agentTools.
  Scores are TRAIN scores — validate the best version on a held-out
  `gaia-batch` slice before promoting. Offline checks:
  `npx tsx src/lab/gaia/evolve-smoke.ts`.

- **Setup**: automatic (`gaia/bootstrap.ts`) — the one required env var is
  **`HF_TOKEN`**. First use materialises the dataset into `<cache>/vein/gaia`, installs the
  leaderboard Space's `scorer.py` (verified against the in-repo
  `SCORER_SHA256`), and resolves a numpy-capable python: `python3` in the prod
  image (the agent venv is on PATH), else a cached venv built on demand.
- **`DATASET_REV` is pinned, and the pin is load-bearing.** The repo's default
  branch NO LONGER CARRIES THE BENCHMARK — as of `682dd723` (main) the
  `metadata.jsonl` files (questions AND gold) are gone. A plain `git clone`
  yields a checkout the grader cannot read. Bootstrap therefore does
  init → `fetch --depth 1 <pinned sha>` → `checkout FETCH_HEAD` against
  `897f2dfb`, the last revision with the full benchmark (165 validation +
  301 test rows), which is also what every score so far was graded against.
- **`HF_TOKEN` is REQUIRED for a cold bootstrap.** The dataset is gated:
  anonymous `ls-remote` answers (which makes the repo look open), but the
  actual `git-upload-pack` fetch is refused without credentials. Bootstrap
  fails fast — before any subprocess — when no token is set. No username is
  needed (HF takes the token as the password with any username); the token
  reaches git via an env-reading credential helper, never `.git/config` or
  argv. An already-populated checkout needs no token.
- **git-lfs is required**: GAIA's attachments are LFS-backed and a checkout
  without it silently yields ~130-byte pointer stubs. Checked before fetching
  and detected after.
- Overrides, all optional: `GAIA_DIR`, `VEIN_CACHE_DIR`, `GAIA_PYTHON`, `GAIA_SCORER_SHA256`, `GAIA_AUTO_SETUP=0`. The dataset is NOT
  baked into the image (the terms forbid resharing outside a gated/private
  repo); mount a volume at the cache dir in prod.
- **Integrity invariant** (per grade): dataset checkout must be a CLEAN git
  tree (a doctored `metadata.jsonl` is doctored gold; untracked `scorer.py`
  is tolerated), and the scorer hash must match the pin — which is now
  **always enforced**, defaulting to the in-repo `SCORER_SHA256`. An
  operator-supplied pin is self-certifying; a repo constant is reviewable in
  a diff, the same class of object as the graders themselves (EVOLVE_SPEC §6).
  A fresh auto-clone is clean by construction. Results carry `benchmarkRev` +
  `scorerSha256`.
- **Gold isolation**: `getTask`/`listTasks` strip `Final answer`; only the
  `score()` subprocess reads it. Never grant a scoring step to the
  producing agent's `agentTools`.
- `smoke.ts` — offline integrity paths + the real python driver against a
  stub scorer, plus the bootstrap paths (missing-token fail-fast / no
  git-lfs / gated 403 /
  scorer-hash mismatch / LFS pointer stubs / idempotent re-entry / half-written
  checkout) with `exec` and `fetchText` faked (`npx tsx src/lab/gaia/smoke.ts`).

### `eval/` — generic, reusable eval primitives (NOT an experiment)

Domain-agnostic eval substrate, shared by every experiment. See
`vein/EVAL_SPEC.md`. **Steps only** — no domain config baked in:

- `eval/steps/score.ts` (`eval/score`) — match a produced set vs an expected
  gold set by a `rubric`; recall-weighted F-beta score.
- `eval/steps/reflect.ts` (`eval/reflect`) — propose a better prompt from the
  AGGREGATED results across a dataset (multi-example → avoids overfitting).
- `eval/steps/optimize.ts` (`eval/optimize`) — the `eval → keep best → reflect`
  loop, run as a single detached "background job" (EVAL_SPEC §8). Runs
  sub-workflows via an injected `services.optimizer` (closure over `vein.run`).
  Multi-example: takes a dataset (`evalInputs[]`), evals the candidate over
  every entry per generation and AVERAGES the scores (the overfitting fix,
  §11.2) — the per-example results array is fed to reflect. Each entry carries
  its own gold (e.g. `{ owner, repo, expected }`), read by the eval workflow
  from `input`. (A single example is just a 1-entry `evalInputs`.)
- `eval/steps/evolve-loop.ts` (`eval/evolve-loop`) — the generic hill-climb
  over WORKFLOW VERSIONS (EVOLVE_SPEC §5.3.3 generalized from the harvey
  instance): runs a domain's one-generation workflow (author → run candidate
  over tasks → digest) up to N generations, briefing each author with every
  prior attempt anchored to the best-so-far, flipping exploit→explore after
  `exploreAfter` non-improving attempts. Fitness is the generation digest's
  `fitness` (fallback `meanPassRate`), named in briefings by `fitnessName`;
  improvements must clear `improveMargin` (judge noise for LLM-judged
  domains — harvey 0.02; produce-sampling noise for deterministic scorers —
  gaia 0). Needs `services.optimizer`. Wired by `harvey-evolve`
  (pass-rate) and `gaia-evolve` (accuracy).
  Three guards keep the climb honest, all learned from live runs:
  (a) **no-op generations.** An author can burn its budget and publish
  nothing; the version fallback in the `*-evolve-gen` workflows then
  resolves to the PREVIOUS generation's publish. The `published` gate
  (`vbefore` vs the resolved version) catches that and skips `candeval`
  entirely, so the generation reports `noop: true` and costs one author
  instead of a whole task set. The loop records it with no fitness — a 0
  there would libel an approach that was never tried.
  (b) **re-score guard.** A version this run already graded cannot become
  the best on a second, luckier sample (`isNewBest` + the `scored` ledger).
  Fitness is resampled, so without this, produce-sampling noise gets
  written into the lineage as a hill-climb step.
  (c) **budget caps.** `maxCost` / `maxMinutes`, checked BETWEEN generations
  (never a mid-generation kill), both null by default. Generation count is
  a poor budget on its own: authors reliably evolve toward more expensive
  architectures, so per-generation cost and wall-clock GROW over a run.
- `eval/steps/matrix.ts` (`eval/matrix`) — the task×version MATRIX across
  measurements (plans/evolve-scoreboard-and-task-matrix.md, Phase 1): folds
  every `{ version, results }` measurement into per-task bands
  (floor/movable/ceiling), an EMPIRICAL noise floor from same-version
  re-measurements (identical-YAML fitness deltas + task flips; UNKNOWN, not
  0, when no version has n≥2), and bias-vs-variance tags on never-correct
  tasks (byte-identical wrong answer ×≥3 = bias — immune to redundancy and
  prompt nudges; distinct wrong answers = variance). Verdict channel only —
  gold never enters. Smoke: `npx tsx src/lab/eval/matrix-smoke.ts`.

**Naming rule:** `eval/*` = generic. The eval *workflows* that wire these with
a rubric/task/dataset belong to the experiment and are named `<experiment>-…`.
The concepts experiment's live in `concepts/workflows/`: `concepts-eval`
(harness), `concepts-eval-score` (rubric), `concepts-eval-reflect`
(task+guidance), `concepts-optimize` (the wired loop). A new experiment `foo`
adds `foo-eval`, `foo-eval-score`, … reusing the same `eval/*` steps.

## Running / gotchas

- **The `/lab` AI chat can now run experiments autonomously.** vein threads the
  lab `services` bag into the chat agent's `run_workflow` tool, so the builder
  can launch `gitsee-explore-services`, `gitsee-eval`, or the `gitsee-optimize`
  loop (which needs `services.optimizer`) — not just service-free core/lab
  workflows. And chat is a detached background job (see `vein/AGENTS.md`):
  describe an eval, tell it to "try it and report back", close the browser, and
  the turn keeps running server-side (persisted to `chats/<id>/`). Reopen to
  reattach. Long runs DISPATCH: a `run_workflow` still executing after
  `VEIN_CHAT_RUN_WAIT_MS` (default 60s) auto-detaches — the tool returns
  `{ status: "running", runId }`, the agent ends its turn, and when the run
  finishes a `[run-notification]` message wakes the chat with the result
  (capped at `VEIN_CHAT_MAX_AUTO_TURNS` consecutive machine-triggered turns;
  a human reply resets the cap). See `vein/plans/dispatch-run-notifications.md`.
- Needs **Neo4j** + `GITHUB_TOKEN` + an LLM key (e.g. `ANTHROPIC_API_KEY`).
- Workflow YAML templates are seeded into the workspace
  (`VEIN_LAB_WORKSPACE`, default `./lab-workspace`) on first boot, then
  edited/versioned via the vein UI.
- vein is consumed as a `file:` dep, which **yarn copies** (not symlinks):
  changes to `../../../vein` (engine or `web/`) only reach `/lab` after a
  rebuild + reinstall. `yarn dev` runs `refresh-vein` automatically before
  starting (**skipped when `$CI` is set** — CI has no `web/` deps, so `vite`
  would fail), so a plain local `yarn dev` picks up vein changes; run
  `yarn refresh-vein` by hand to refresh without a restart. CI builds vein
  before `mcp` install for the same reason.
- The vein UI is path-agnostic (relative assets + runtime API base), so it
  works under `/lab` (with the `/lab` → `/lab/` redirect in `mount.ts`).
- Trigger a run: `POST /lab/workflows/bootstrap-then-process/run` with
  `{ input: { owner, repo, token } }`, or use the UI at `/lab/`.

## Run it end-to-end (manual)

Nothing is automated yet — no CI job exercises `/lab`. Manual steps:

1. **Neo4j**: `cd mcp && docker compose -f neo4j.yaml up -d` (wait healthy).
2. **Env**: `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` (and `NEO4J_HOST`/`NEO4J_USER`/
   `NEO4J_PASSWORD` if not default).
3. **Start mcp**: `cd mcp && yarn dev` (serves on `:3355`). Locally, `dev`
   runs `refresh-vein` first, so vein (engine + `web/`) is rebuilt and
   reinstalled automatically — no separate build step needed. (Skipped when
   `$CI` is set.)
4. **Init + seed** (lazy on first hit): `curl localhost:3355/lab/health`,
   then `curl localhost:3355/lab/workflows` to confirm the 3 workflows
   seeded.
5. **Run** (detached launch + reattach — see `vein/EVAL_SPEC.md` §8). The
   `POST …/run` returns `{ runId }` immediately (the run executes server-side);
   reattach to its SSE event tail to watch it:
   ```
   RUN=$(curl -s -X POST localhost:3355/lab/workflows/bootstrap-then-process/run \
     -H 'content-type: application/json' \
     -d '{"input":{"owner":"OWNER","repo":"REPO","token":"<gh token>"}}' \
     | jq -r .runId)
   curl -N localhost:3355/lab/workflows/bootstrap-then-process/runs/$RUN/stream
   ```
   Use a **tiny repo** first (LLM cost/time per PR+commit).
6. **Verify**: query Neo4j directly — `MATCH (c:Concept) RETURN c.name,
   c.description` — or watch the reattached SSE `step.*` events. (There is no
   concept-listing HTTP endpoint yet; vein only exposes `/workflows`.)

**Prerequisite gap for file linking:** `concepts/link-files` connects
concepts to `File` nodes, which only exist if the repo's **code graph has
been ingested** (stakgraph parse → Neo4j). Without ingestion the run still
succeeds, but produces 0 `MODIFIES` edges. To exercise linking, ingest the
same repo first (e.g. via the standalone `/ingest` or mcp's upload flow).

**Build assets:** `seed.ts` locates its templates relative to its own
compiled module (`import.meta.url`), but `tsc` only emits `.js` — so the
workflow `*.yaml` templates and the `steps/*.ts` sources (read as text) are
copied into `build/lab/` by `scripts/copy-lab-assets.mjs`, run after `tsc`
in the `build` script. Add new lab assets under a `workflows/` (`.yaml`) or
`steps/` (`.ts`) dir and they're picked up automatically.

**Prod runs with a TS loader.** Seeded steps are published as `.ts` source
into the workspace and vein loads them via dynamic `import()`. Plain `node`
can't import `.ts`, so the prod server runs as `node --import tsx
build/index.js` (`start` script + Docker `CMD`); `tsx` is a runtime
dependency. This is what lets agents/users author steps in TypeScript and
have them run in prod without a compile step.

**Known follow-ups** (not blockers for a basic run): `/lab` runs bypass mcp
auth (mounted before auth middleware).
