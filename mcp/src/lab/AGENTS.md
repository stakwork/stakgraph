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
  `rev`; `token` falls back to `GITHUB_TOKEN` env (private repos). Output
  `{ workspacePath, repos }`. **The only gitsee-specific producer step.**
- Exploration is the **core `agent` step** (`vein/src/steps/core/agent.ts`),
  pointed at `cwd = clone.workspacePath`. Its general tools (`repo_overview`,
  `fulltext_search`, `bash`, anthropic `web_search`, + `file_summary` — the
  `stakgraph` AST CLI, only offered when `stakgraph` is on PATH) + the agent loop
  live in vein core now — what was the old inlined `gitsee/explore-services` step
  (deleted). gitsee runs it in **`finalAnswer` (FILENAME text) mode**; the
  structured-`schema` mode is intentionally unused here for now. (For the
  `file_summary` tool, `stakgraph` must be on PATH; the agent falls back to
  `bash`/`cat` otherwise.)
- `gitsee/workflows/gitsee-explore-services.yaml` — `clone → agent`; input
  `{ workspace, repos: [{owner,repo,rev?}], token? }`; the `system`/`prompt`/
  `finalAnswer` prompts live in `params` (the experiment surface, frontend-
  focused). The agent injects a neutral working-dir listing, so the prompts
  stay repo-agnostic (workspace framing moved into `params.system`).

**Eval/optimize stack** (mirrors `concepts-*`; reuses the generic `eval/*`
steps EXCEPT scoring, which is gitsee-specific — see below). The gold is the
**actual canonical pm2.config.js + docker-compose.yml pair** (produced vs gold
is apples-to-apples).

Scoring is a **structured + hybrid** scorer (`gitsee/score-setup`), NOT the
generic LLM `eval/score`. Both files are parseable, and the gold is the ANSWER
KEY — so the dominant tier is **deterministic name set-diffs vs the gold**, which
is why it stays repo-agnostic without understanding any dependency:
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

- `gitsee-eval` — harness: produce (subflow → `gitsee-explore-services`) →
  score (subflow → `gitsee-eval-score`). No reset step (gitsee is stateless).
  Input `{ label, repos, token?, expected? }` — `label` is the workspace name
  (and the `eval: <label>` link); `expected` gold falls back to `params.expected`.
- `gitsee-eval-score` — `gitsee/score-setup` + the matching policy in `params`
  (`useLLM`, `ignoreEnvKeys`, the semantic-residue `rubric`). Strict env policy:
  every gold env key is required (exempt noise keys via `ignoreEnvKeys`).
- `gitsee-eval-reflect` — `eval/reflect` + the setup task/guidance.
- `gitsee-optimize` — `eval/optimize` loop. Tunes **`system`** (the explorer
  prompt), NOT `finalAnswer` (the hard pod contract). Cohort in `params.dataset`,
  one entry per WORKSPACE: `{ label, repos: [{owner,repo,rev?}], expected }`.

Dataset: `heroku-node` (1-repo Express, verified 0.95) + `hive` (Next.js +
Postgres + Prisma; `hive` pinned, with sibling dep repos sphinx-voice /
system-canvas / staklink). Add more workspaces for a stronger multi-example
optimize (EVAL_SPEC §11.2).

Needs `ANTHROPIC_API_KEY` + `git` + `rg` on PATH (Neo4j only for booting the
lab, not for gitsee itself). Trigger:
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
