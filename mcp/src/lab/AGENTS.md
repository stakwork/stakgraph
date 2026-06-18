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
step** (see `vein/AGENTS.md`); gitsee only supplies the clone + the prompts.

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

**The product loop (`gitsee/boot-and-exercise`) — NOT an eval signal.** Where
`verify-setup` is a READ-ONLY boot GATE (boot once → one screenshot → one vision
verdict → score), `boot-and-exercise.ts` (`gitsee/boot-and-exercise`) is the
autonomous "set up a repo until it actually runs" loop. It stages the produced
setup, then runs a **tool-using agent** that BOOTS the app, DRIVES the live
frontend in a real headless browser, OBSERVES failures like a QA engineer, FIXES
the cause, REBOOTS, and repeats until the app is functional. Tools: `boot`
(re-stage + compose up + staklink/pm2 + wait for port; call after every edit),
`browser_open` / `browser_snapshot` (visible interactive elements with `@eN`
refs) / `browser_click` / `browser_fill` / `browser_press`, **`browser_observe`**
(drains console errors + failed requests + **4xx/5xx API responses** since the
last call — the key "renders but is broken" signal a screenshot can't show),
`assess_ui` (the "eyes": fresh screenshot + errors + server logs → an anthropic
vision verdict), `read_logs`, `bash`, `str_replace_based_edit_tool` (edit
pm2.config.js env / docker-compose.yml / repo source, sandboxed to the
workspace), and `final_answer` (a `## SUMMARY` / `## WORKING` / `## MISSING`
markdown report). The agent works in LOCAL path terms; the final `setup` output
is rewritten back to pod-portable `/workspaces/<repo>`. A working-dir **preamble**
(the real local workspace path + sibling repos) is prepended to the prompt so the
agent doesn't burn steps guessing pod paths. **Pod URLs (`$POD_ID`/`$POD_URL`)**
are kept in the deliverable (the pod contract) but **localized only in the
staged-for-boot copy** (`podSubstituteLocal`: `https://$POD_ID-<port>.<domain>` →
`http://localhost:<port>`, `$POD_URL` → `http://localhost:<frontendPort>`) — on
the real sandbox the platform expands them + proxies `<podid>-<port>.<domain>` to
`localhost:<port>`; locally there's no proxy, so we emulate it (NOT a staklink
concern). The agent is told these are auto-substituted and to KEEP them rather
than rewrite to localhost. (verify-setup could adopt the same helper.) Because it is allowed to
**WRITE/FIX**, it must **NOT** be wired into the scored `gitsee-optimize` loop —
fixing in place would erase the gradient that teaches the explorer. Its home is
the standalone **`gitsee-setup-and-run`** workflow (`clone → produce
(gitsee-explore-services) → boot-and-exercise`); the deliverable is a known-good
`setup` + the `diff` + the `report`, not a grade. Like the explore path it also
captures the agent's repo edits as a replayable per-repo `git diff` (inlined here
rather than reusing `gitsee/capture-edits` — a workflow's output is its last
step's, and capture-edits would drop the richer setup/report/booted/working
fields). Same boot/teardown machinery + caveats as verify-setup (snapshot-diff
container teardown; `keepUp:true` to inspect; needs `docker` + `git` + `npx
playwright install chromium`). Output `{ booted, working, port, setup, report,
diff, changedRepos, changed, screenshotPath, iterations, logsTail, usage, cost }`.
Dev smoke (not seeded): `src/lab/gitsee/smoke-boot.ts` (`clone → core agent →
boot-and-exercise`, real calls; `KEEP_UP=1` leaves the stack up).

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
  reattach. (Long optimizes still run inline inside one `run_workflow` call —
  detached *workflow*-run mode for the tool is a later add.)
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
