# lab

A sandbox for **strut workflow experiments**. The goal: take pipelines that
are normally hardcoded TS and express them as editable strut workflows
(YAML + step config) so we can iterate on flows and prompts without code
changes. See `AGENTS.md` in the strut repo (github.com/stakwork/strut) for the engine itself.

## Model

One strut instance for the whole lab (`createLabStrut.ts`), mounted into the
Express app at **`/lab`** (`mount.ts`). Experiments are just groups of
workflows/steps inside it — **not** separate servers. Adding an experiment
= register its steps + merge its services + seed its workflows in
`createLabStrut`.

- `createLabStrut.ts` — the single instance: registry (strut core+lib + all
  experiment steps), merged `services` bag, seeded workflow templates.
  **Seeding is additive**: dropping a step from a seeder's `SEED_STEPS`, or
  a workflow from its `SEED_WORKFLOWS` (deleting the YAML from git), does
  NOT remove it from existing workspaces (graph-backed ones persist; the
  author agent keeps discovering the step, and the workflow stays listed,
  runnable and schedulable). When you remove or rename a seeded step or
  workflow, add the old name to that seeder's `RETIRED_STEPS` /
  `RETIRED_WORKFLOWS` list — `retireSteps` / `retireWorkflows`
  (`seed-opts.ts`) soft-delete it at boot.
- `mount.ts` — bridges the strut (Hono) app into Express under `/lab`
  (API + run-streaming SSE). Registered before `express.json()` to keep
  raw request streams. Lazy-initialized so mcp boot isn't coupled to
  Neo4j / LLM keys. Exports `labStrut()`, the same instance, for in-process
  launches.
  **The env boundary** (strut src/env.ts, stakwork/strut#228): when the
  lab boots, `labStrut()` calls strut's `scrubProcessEnv()` BEFORE
  `createLabStrut` — `process.env` is emptied down to what a shell needs
  (`KEEP_ENV`, plus mcp's own `GIT_TERMINAL_PROMPT` / `GIT_ASKPASS`), so a
  seeded step, or a child it spawns with `process.env`, sees no deployment
  key, while `ctx.services.secrets` still resolves one through strut's boot
  snapshot. Before, not after, the strut is built: strut's `standardServices`
  holds that snapshot by reference, and until the scrub the snapshot IS
  `process.env` (see `scrubEnv` in mount.ts). mcp's own reads — at
  construction (the workspace path, `NEO4J_*`, `GITHUB_TOKEN`, the provider
  key, `GAIA_*`) and per request (`API_TOKEN`, aieo's provider keys, …) —
  go through `env` (`mcp/src/env.ts`), the copy frozen right before the
  scrub; host code never reads `process.env` after boot (a module-level
  `const` read at import is fine). `STRUT_ENV_SCRUB=0` turns the scrub off;
  `STRUT_ENV_KEEP=A,B` keeps named variables. **A seeded step reaches the shell, files, the
  network and keys only through `ctx.services`** (`shell`, `artifacts`,
  `http`, `secrets`): strut's registry warns at boot about a workspace step
  whose source imports `node:fs` / `node:child_process` / `http`, reads
  `process.env` or calls the global `fetch` ("[steps] <type> reaches past
  ctx.services"), and a later strut refuses to load it. The one deliberate
  exception today is `pod-setup/repo-facts`, which walks a checkout with
  `node:fs` (see its header) and needs a strut-side door.
- `proxy.ts` — `POST /gitree/process` and `POST /repo/describe` as lab
  launches (docs/plans/gitree-to-workflows.md). A request naming an actor
  (`x-strut-actor` beside a matching `x-api-token` — `labAuth`'s rule;
  hive and standalone) runs `bootstrap-then-process` / `describe-code` as
  that actor through `strut.run`, wired to the old `request_id` /
  `/progress` shape; one without (mcp's SPA, dev mode, an older caller)
  goes to the old handler unchanged, so the deploys are order-independent.
  A `token` in the query is stored as the actor's `GITHUB_TOKEN` secret
  (in-process `PUT /actors/:actor/secrets/…`), never put in the input.
  One sweep per repo: an in-flight map, seeded on a miss from strut's run
  list so a run auto-resumed after a restart is attached to, not relaunched;
  `isRepoInFlight` is what `GET /gitree/concepts` reports as `processing`.

## Experiments

### `concepts/` — GitHub history → "Concept" knowledge graph

A clue-free, Graph-only port of `mcp/src/gitree` (renamed `Feature` →
`Concept`, new `Concept` Neo4j label; reuses `PullRequest`/`Commit`/`File`).
Walks a repo's PRs + commits chronologically, LLM-classifies each into
concepts, summarizes, and links to code files.

- `concepts/steps/` — the in-code strut steps (one unit of work each).
- `concepts/workflows/` — `process-change`, `process-repo-chronological`,
  `bootstrap-then-process` (the entry workflow), `describe-code`
  (a `description` on code nodes, then a `summary` on Files — File's Schema
  indexes [name, summary]; `params.model` defaults to `haiku`).
- `concepts/services.ts` — `ConceptServices` bag `{ storage, github(token),
  llm, summarizer, … }` injected via `ctx.services`. **No token in a
  workflow input**: a run's input and every step's config are written to
  the run log. The steps that reach GitHub (`clone-repo`, `fetch-changes`,
  `fetch-content`, `apply-decision`) read `ctx.services.secrets.get("GITHUB_TOKEN")`
  — strut binds `secrets` to the run's principal, so that is the actor's
  token (hive, or mcp's proxy, stores it before launching), else the
  deployment's — and hand it to `github(token)` / `clone` / `prContent` /
  `commitContent`. `__tests__/token-secret.test.ts` runs the seeded
  workflow through a real strut and asserts the token is in no event.
- `concepts/steps/record-usage.ts` — adds a sweep's spend to the repo's
  running total (`cumulativeUsage` on `GET /gitree/concepts`): once in
  `process-repo-chronological` (the collect rollup + summaries), once in
  `bootstrap-then-process` beside the bootstrap.
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

### `jarvis/` — knowledge-graph steps (NOT an experiment)

Self-contained ports of the mcp repo-agent's Jarvis tools
(`mcp/src/repo/toolsJarvis.ts`) as seeded strut steps — same endpoints, same
schemas, same LLM-facing descriptions — so workflows (and agent steps) can
read/write the Jarvis knowledge graph. **Concepts are Jarvis nodes** (filter
`type: "Concept"` on search/neighbors), so no concept-specific steps exist.

- **Reads:** `jarvis/get-ontology`, `jarvis/get-ontology-type`,
  `jarvis/graph-search` (hybrid + field-scoped vector search),
  `jarvis/graph-get`, `jarvis/graph-get-batched`, `jarvis/graph-neighbors`.
- **Writes:** `jarvis/create-node`, `jarvis/edit-node`,
  `jarvis/create-triplet`, `jarvis/create-batch-triplet`. The ontology CRUD
  family is deliberately NOT ported (schema editing stays a human/setup
  activity). Strut's own `graph/*` twins go one step further: strut-only
  `graph/create-schema` (register/extend a node type) and `graph/edit-edge`
  (patch an edge's properties) exist there, with no jarvis/* counterpart.
- **Config is automatic:** each step resolves `JARVIS_URL` + `API_TOKEN`
  (+ optional `JARVIS_HTTP_TIMEOUT_MS`) through `ctx.services.secrets`
  (secret store → env fallback) and calls through `ctx.services.http` — so
  runs are cassette-recordable and credentials are scrubbed from fixtures.
  Steps are ALWAYS seeded; without `JARVIS_URL` they fail loudly per run
  rather than silently missing.
- **Granting to agents:** `agentTools: ["jarvis/*"]` (glob, strut-core
  `expandAgentTools`) for everything, or list the read steps explicitly for a
  read-only child. Sub-agents = grant `"agent"` itself and pass the child a
  narrower `agentTools` list (recursion depth is whether the child gets
  `"agent"` again).
- **Self-contained duplication is deliberate:** each step file inlines its
  small `jarvisCtx` preamble (seeded steps may only value-import `"strut"`);
  the contract is documented once in `jarvis/steps/_shared.ts` — change it
  there AND in every step.
- **Smoke:** `npx tsx src/lab/jarvis/smoke.ts` — offline; seeds into a temp
  workspace, verifies registry discovery, and runs every step against a fake
  `ctx.services.http` Jarvis.

The strut-native twins of these steps — `graph/*`, same names, inputs, and
output shapes, backed by strut's own Neo4j-over-bolt graph layer with no
jarvis in the loop — are NOT lab steps: they ship inside strut as lib steps
(`strut/src/steps/lib/graph/`, auto-discovered by the registry) and only
need `NEO4J_URI` (+ `NEO4J_USER`/`NEO4J_PASSWORD`) in the env or secret
store. A workflow swaps backends by step type (`jarvis/graph-search` ↔
`graph/graph-search`); grant with `agentTools: ["graph/*"]`.

### `sheets/*` and `browser/*` — strut's lib steps now (no lab code)

The Google Sheets steps (`sheets/*`) and the browser steps (`browser/*`,
`ctx.services.browser`, and the `browser-capture` / `browser-explore` /
`browser-watch` workflows) moved into strut (stakwork/strut#200, #201); see
strut's AGENTS.md for both. So did captchas a person solves in Hive's
captcha card (stakwork/strut#206): a page the job keeps across turns
(`browser/keep`, `browser/open { session }`), the manifest written in code
(`browser/captcha-ask`) and the `browser-captcha` workflow, wired by
strut's job prompt — no pod, no `pod/file`. The lab keeps no copy:

- **The seeded steps are retired at boot** (`MOVED_TO_STRUT` in
  `createLabStrut.ts`, through `retireSteps`): a workspace custom step
  shadows a lib step of the same name, so a swarm ran the lab's copies until
  they were gone. The list stays as long as a swarm might still hold one;
  missing names are a no-op.
- **The browser workflows are NOT retired**: strut seeds them under the same
  names (the same YAML), so a retire here would delete strut's own and the
  two seeders would churn every boot.
- **The browser** is strut's: `createStrut` builds a `BrowserService` from
  `BROWSER_*` (sphinx-swarm sets `BROWSER_WS_URL` + `BROWSER_WS_PATH` to the
  `ghcr.io/stakwork/strut-browser` node) unless the host passes its own
  `services.browser`, which the lab does not. `playwright-core` is strut's
  dependency; mcp's own copy (1.58.1, under `@playwright/test`) is the e2e
  tests', and the `sharp` resolution keeps one copy for strut and
  transformers.
- Granted as before: `agentTools: ["sheets/*"]`, `["browser/*"]`;
  `browser/verify` + `browser/walkthrough` are in strut's own `job.yaml`
  grants.

### `pods/` — hive sandboxes as steps (NOT an experiment)

A pod is a container hive's pool manager hands out: the workspace's
repositories, the app's dev server and an IDE a person can open, and
staklink — a control server on the pod that runs a coding agent (goose),
resets repositories, diffs, pushes and tests. Every staklink call is one
seeded step, `pod/*`, and the same steps are the job agent's tools
(`agentTools: ["pod/*"]`): the way to make a change that must build, run or
be seen to be judged (the `Pod` Concept under `Code Change`, `pods/concepts`,
says when and how; the job agent reads it off the graph, never the prompt).
Replaces the `hive/*` pod steps and the `pod-pr` pipeline tom prototyped on
swarm38 — a one-shot that decided everything the agent should decide and
released the pod at the end.

- **The pod outlives a run.** `pod/claim` on a run with a job registers a
  HOLD on the job (strut `ctx.services.jobs`, plans/jobs.md §6) naming
  `pod/release` as the way to let it go, with the pod's `frontend` as the
  hold's `url` — strut folds it into the job's `urls.pod`, so a claim's
  check says `{{ input.urls.pod }}/settings` and survives the pod being
  claimed again (strut plans/job-claims.md §3); strut runs it when the job is
  deleted (`DELETE /lab/jobs/:id`) or idle past `STRUT_WORKDIR_TTL_DAYS`.
  So the agent in the pod keeps its session — its memory lives on the pod's
  disk — and the dev server stays up between turns for the person to watch
  (hive renders a `url` artifact as a live frame). Without a hold a claimed
  pod leaks: hive's stale sweep releases pods by `Task.podId`, and strut
  claims with no task.
- **The password never leaves the step.** `pod/claim` returns it SEALED
  (AES-GCM under a key derived from HIVE_API_KEY — whoever holds that key
  can claim pods anyway, so no second secret) and every other step takes
  `sealed` and opens it in-process for the one request. The run log, the
  output, the callback and the agent's context carry `sealed` only; the
  smoke proves it.
- **Steps** (`pods/steps/`, one file each; `_shared.ts` is a seeded helper
  the registry skips): `claim` (→ `{ podId, frontend, ide, control, sealed }`,
  the hold) and `release`; `latest` (reset repositories to a branch, the
  run's GitHub token on the fetch); the agent in the pod three ways —
  `agent-start` (returns a `request_id` at once, for a workflow that polls;
  nothing wakes a job when the task finishes, so `Pod.md` has a job turn
  use `agent`), `agent-status` (one look, never fails on a failed task),
  `agent` (start + wait, up to 2 h, a cancelled run stops waiting between
  polls; past the timeout the task keeps running and the error names its
  `request_id`);
  `diff` (the working trees), `branch-diff` (the whole branch against its
  base; file contents capped, as every tool result here is); `push`
  (commit + push as the run's GitHub identity; a new branch for a new
  change, `stay_on_branch: true` for a REVISION — the same branch, and
  `pr_url` is the PR it already has, which is what a follow-up turn needs;
  nothing to commit still pushes what is unpushed);
  `run-tests` (the pod's own test commands, never fails on red; not named `test` — the graph keys a step by its name with separators stripped, so it would collide with a workspace's custom `pod_test`; when tests ran it also answers a claim's check contract, `supports` + `content`, and gives no verdict when none ran — a cannot-run, never a refutation);
  `config` and `config-set`, the workspace's POOL CONFIG (below). The agent's
  `session` defaults to the run's job, so on a job the goose thread
  continues across turns and the model never picks the id; its model calls
  go through this run's LLM gateway grant (`ctx.services.llmAuth`) — never
  the pod's own key, never a caller-supplied one. The grant's macaroon rides
  INSIDE the key handed to staklink, `<vk>.<macaroon>` (the gateway wrapper
  splits it; `gateway/wrapper/authsplit.go`): goose's anthropic provider
  sends no custom headers, so as an `x-macaroon` header it never arrived.
- **The pool config** (strut plans/pod-config.md). What a pod changes in
  itself dies with it; hive's pool config for the workspace (`services`,
  the workspace-wide env, `docker-compose.yml`, `Dockerfile` — `GET`/`PUT
  /api/workspaces/{slug}/stakgraph`) is what every new pod is built from, and
  a write recreates every unused pod of the workspace. `pod/config` reads it,
  and with a claimed pod the files the pod runs from (staklink `GET
  /pod-config`), with a diff per file. Hive masks the env lists for a key
  caller and takes a mask written back as "keep the stored value"; the pm2
  files carry values in clear, so their `env` blocks are masked here
  (`_pool.ts`) with a per-read tag that tells apart two values masking alike.
  `pod/config-set` takes edits BY NAME (`services` / `env` maps, `null`
  removes; whole compose / Dockerfile texts), reads hive right before its
  PUT and sends both lists whenever either changes (hive regenerates
  pm2.config.js from them in the write; an env-only PUT would be overridden
  by the old file's values at push), refuses what would corrupt the config
  (a mask that is not the stored one's, a `*_COMMAND` as an env var,
  pm2.config.js, devcontainer.json), then polls the rollout until its change
  — or a later one, whose push carries it — is `ready`. `pod/claim { fresh:
  true }` sends `?generation=new` (a pod created after the last push; 409
  until one is ready) and gives back a pod a hive without it hands out; its
  hold carries no `url`, so the job's `urls.pod` stays its own pod. The
  `Pod Config` Concept (under `Pod`) is the convention: fix in the pod,
  promote only what every job needs and only after a person says yes (and
  after a needed PR merges), check on a fresh pod, report a `pod-config`
  deliverable a later job finds through `job/list`; no credential is
  promoted (secrets are the plan's V2).
- **Secrets:** `HIVE_URL` + `HIVE_API_KEY` (deployment; an org-scoped key —
  the hub claims for every workspace of its org, the id arrives on the job
  launch), `GITHUB_TOKEN` (the run's: hive pushes it as an actor secret on
  every job turn).
- **Smoke** (offline — seeds and discovers, then a stand-in hive + staklink
  on localhost: claim → latest → agent → branch-diff → push with `job` on
  the launch, the revision turn with agent-start/status and a push onto
  the same branch, the hold on the record, the password nowhere strut
  keeps it, `DELETE /jobs/:id` releasing the pod; then the pool config —
  read, diffed, a promotion written by name and waited for, the refusals,
  a fresh claim — with no stored env value in the run's record): `npx tsx
  src/lab/pods/smoke.ts`. The masking, diff and edit rules are unit-tested
  in `pods/pool.test.ts` (`yarn test:node`).

### `gaia/` — GAIA benchmark scoring (the hardcoded grader)

Scores answers with the **actual** GAIA leaderboard scorer (`scorer.py`,
quasi-exact match with type-aware normalization) as a `python3 -c` subprocess
against the validation split's gold answers. The discipline: the
grader AND the gold live in the in-code `gaia` service (`gaia/service.ts`,
on the `LabServices` bag), never in a seeded/authored step. `gaia/*` steps
are thin plumbing over `ctx.services.gaia.*`.

- **Committed harness** (`gaia/seed.ts`, seeded at boot):
  steps `gaia/list-tasks`, `gaia/get-task` (stages a task's attached file
  into the run's artifacts dir), `gaia/evaluate` (HARNESS-ONLY), and the
  combiner `gaia/summarize-batch` (+ strut's core `pack`); workflows
  `gaia-produce` (agent step; the produce system prompt, model, maxSteps: 50
  and agentTools live in `params`; an `onError` fallback scores a blown-up
  agent as an empty wrong answer instead of killing the batch), `gaia-run`
  (single task: produce → score; `input.produceWorkflow` swaps in a seeded
  produce variant) and `gaia-batch` ({ level, limit }: one
  score call for the whole batch). This is the harness that went 1/5 → 5/5
  on the level-1 batch (EVOLVE_SPEC §1), promoted from the workspace where
  the assistant authored it. Seeding is content-hash reconciled (`SEED_OPTS`,
  `seed-opts.ts`) — a CHANGED committed copy wins at boot, an unchanged one
  leaves a workspace-side edit active, so a workspace-side evolution
  survives restarts but propagates to other instances only once it's ported
  back here. The from-scratch
  authoring recipe is kept in `notes/GAIA.md` as an authoring eval; it is no
  longer the path to a working harness.

- **Evolve harness** (on the generic `eval/evolve-loop`):
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
  `gaia/get-task` / `pack` as steps but NEVER `gaia/evaluate`
  (produce-time oracle) and never gaia/*, eval/*, meta/* as agentTools.
  Scores are TRAIN scores — validate the best version on a held-out
  `gaia-batch` slice before promoting. Offline checks:
  `npx tsx src/lab/gaia/evolve-smoke.ts`.

- **Setup**: automatic (`gaia/bootstrap.ts`) — the one required env var is
  **`HF_TOKEN`**. First use materialises the dataset into `<cache>/strut/gaia`, installs the
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
- Overrides, all optional: `GAIA_DIR`, `STRUT_CACHE_DIR`, `GAIA_PYTHON`, `GAIA_SCORER_SHA256`, `GAIA_AUTO_SETUP=0`. The dataset is NOT
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

### `wfbench/` — Workflow Editor Agent Benchmark (stakwork 58313's twin)

The benchmark harness for the workflow-AUTHORING agent, rebuilt as a strut
workflow (design + step map: `plans/wfbench-harness.md`). One task in
(`{ task_slug, task_title?, instructions, criteria, workflow_input_json?,
rerun_expected_output?, webhook_url? }` — Hive's payload shape), one callback
out. `wfbench-run`: graph roster (EvalSet → EvalRequirement×N, EvalTrigger,
HAS_BASELINE_TRIGGER on the EvalSet's first trigger else HAS_TRIGGER — written
with strut's `graph/*` steps under `input.namespace || params.namespace`,
default `default`. EvalSet id = the task slug VERBATIM and EvalRequirement id
= `<slug>::<criterion_id>` — Hive's `eval-nodes.ts` ids, because Hive upserts
this roster itself before dispatch and its rubrics reader looks it up by
those; ours merge onto its nodes by node_key. Trigger/output/criterion ids
follow 58313/58312: `<slug>-<runId>`, `<slug>-<runId>-<criterion_id>`) ‖ the
author (core `agent` + `agentTools: ["meta/*"]`, editor tool only — 54419's
twin) builds `wfbench-<slug>` → resolve the version it actually shipped
(`vpin || vactive`, `published` vs `vbefore` — the gaia-evolve-gen guard) →
input-key gate (`wfbench/check-input-keys`: the `input.<key>` references in
the produced YAML vs the launch payload; a mismatch or empty body is a harness
error, never launched) → rerun via `meta/run-workflow` (57425's twin; own runId
= 58313's project_id) → `wfbench/classify-run` (a runtime-FAILED rerun is still
judged; no runId is a harness error) → `meta/validate-workflow` on the produced YAML
(judge EVIDENCE for structural criteria, never a gate) → `wfbench/build-materials`
(workflow YAML + custom step sources + validation result + launch payload + run
output + expected output, inlined as one markdown block) → per-criterion judge (`wfbench-judge-criterion`: agent
schema mode, nothing to read — a crash packs `{ error }` = honest FAIL) →
`eval/aggregate-scores` → record `EvalTrigger -HAS_OUTPUT-> EvalTriggerOutput
-HAS_CRITERION_RESULT-> CriterionResult <-HAS_CRITERION_RESULT- EvalRequirement`
(`wfbench/build-eval-output`, 58312's ids: `<slug>-<runId>`,
`<slug>-<runId>-<criterion_id>`) → `wfbench/webhook-body` (58313's 4-way
`resolve_webhook_payload`: success `{ task_slug, task_title, n_passed, n_total,
all_pass, pass_rate, judge_model, criteria_results }` or `{ harness_error:
true, error_type, error }` — no fake 0/N) → POST `webhook_url` → the run's
output IS that body (+ diagnostics), fixing 58313's set_output divergence.

- Graph writes use only ontology-declared attributes (strut's backend rejects
  the rest): 58313's `EvalSet.project_id` (an int there) and the `name` on
  EvalTrigger / EvalTriggerOutput are omitted; 58312's `CriterionResult
  -HAS_CAUSE-> Workflow_version` is not written (no such relationship in
  the ontology, and a strut workflow is not a Workflow_version node).
- Grant discipline: `wfbench/*` is never an agentTool. The author gets
  `meta/*` (+ `meta/validate-workflow`) and the editor; the judge gets
  nothing useful; the produced workflow is necessarily publisher `ai`.
- The author's `params.authorSystem` carries a stakwork→strut translation
  table, so PORTING a stakwork workflow is just a task whose instructions
  are the stakwork body (plus a real project's input/output as
  `workflow_input_json` / `rerun_expected_output`).
- v1 is create-new only (58313 v1 too). Needs `ANTHROPIC_API_KEY` + the
  graph; no stakwork credentials.
- Smoke (offline — seeds, discovers, static-validates both workflows
  through the authoring capability, drives every pure step, and checks the
  graph payloads against `JARVIS_ONTOLOGY`): `npx tsx src/lab/wfbench/smoke.ts`.

### `code/` — code changes as strut workflows (hive's `propose_code_change` + its approval)

Hive's canvas agent proposes small code changes; the preview used to be
mcp's `/repo/agent` polled from inside a Vercel function (and timing out).
It is now strut workflows — design + the hive half in strut
`plans/code-change.md` — whose only model is strut's OWN core `agent` step:

- `code-change-propose` (§4): `git/checkout` (strut lib: a credential-free
  bare cache per remote + a detached worktree per run, removed at run end)
  → `agent` (cwd = the worktree; `params.system` / `finalAnswer` are the
  experiment surface) → `git/diff` (stage all, one unified diff, caps,
  gitleaks when on PATH) → `pack` `{ diff, diffSha256, filesChanged, files,
  scanned, baseBranch, baseSha, summary, cost, usage, steps }`. Hive
  launches it with `POST /lab/workflows/code-change-propose/run { input:
  { repo, prompt }, callback }` and reads that pack from the callback's
  `output`; `filesChanged: 0` is a success run with nothing to propose.
- `code-change-land` (§6) — the approval, NO agent step: `git/checkout` at
  `input.baseBranch` → `git/apply` (exactly the approved bytes, `sha256`
  checked before the tree is touched, `--index`) → `git/push` (commit as
  the token's GitHub identity, push `HEAD:refs/heads/<branch>`, never
  `--force`) → `github/create-pr` (idempotent by head branch; base =
  `checkout.ref`, the branch checked out) → `pack` `{ url, number, branch,
  base, headSha, filesChanged, files, diffSha256 }`. Input `{ repo,
  baseBranch, diff, diffSha256, branch, title, body }` — a propose run's
  diff once the user approved it. Deterministic on purpose: a base that
  moved fails `git apply` honestly and hive re-proposes. The codes hive
  classifies on, as prefixes of `error.message`: `patch_conflict:` |
  `push_rejected:` | `no_push_permission:` | `pr_create_failed: <status>
  <message>`; anything else (a checkout failure) carries no code.
- `code-change-pr` — propose + land in ONE run, for a caller that reviews
  on the pull request itself: the `job` agent (strut `plans/jobs.md` §4;
  the `job` workflow grants `meta/run-workflow`). `git/checkout` (at
  `input.branch` on a follow-up, else the default branch) → `agent` →
  `git/diff` (stage + scan) → `git/push` (branch = `input.branch`, else
  `params.branch_prefix` + the run id) → `github/create-pr` (base =
  `input.base`, else the remote's default; idempotent by head, so a
  follow-up returns the open PR with `created: false`) → `pack` `{ url,
  number, repo (owner/name), branch, base, headSha, created, filesChanged,
  files, summary, cost }`. Input `{ repo, prompt, title, branch?, base?,
  body? }`. The PR is the proposal: to revise it, run again with the same
  `branch`. An agent that changed nothing fails at `git/push` (its plain
  nothing-staged error — nothing pushed, no PR). The `Code Change`
  Concept (`code/concepts/Code Change.md`, parent `Job`,
  seeded as `CODE_CONCEPTS` beside the builder's and janitor's pages) is
  how the `job` agent is routed here: the workflow and its input, the
  brief, the follow-up rule, the `pull_request` artifact, when a sandbox
  is warranted instead, and how a workspace adds a child (another way to
  deliver a change, or a repository's own rules).

- **No custom steps** — every step is strut's, so this seeder ships YAML
  only (`seedCodeWorkflows`, category `code`, unstamped).
- **No credential in the checkout, none in the input.** The clone token —
  and, for land, the push and pull-request one — is the run principal's
  `GITHUB_TOKEN` actor secret (hive pushes it to `PUT
  /lab/actors/:actor/secrets/GITHUB_TOKEN` before dispatching as that
  actor; strut binds the run's `secrets` to its principal), else the
  deployment's; it reaches git through the child env only. The agent's
  bash cannot push: HTTPS push needs a credential and the worktree has none.
- Needs `git`, `rg` on PATH (the image has them, plus gitleaks and the
  stakgraph CLI for `file_summary`) and a model key or a Mothership
  delegation for the actor.
- Smoke (offline — seeds, discovers, static-validates both, then RUNS them
  against a local BARE origin: propose with the `agent` step swapped for a
  fake that edits a file; land with that diff — real apply, commit and
  push into the bare (no token → git/push's fallback author), a fake
  `github/create-pr` that records what it was asked to open — then land
  again on a moved main for `patch_conflict:`; then `code-change-pr` three
  times: a first turn that makes the branch and the PR, a follow-up on that
  `branch` that adds a commit and gets the SAME PR back, and a no-change
  turn that fails at git/push with nothing pushed; `CODE_SMOKE_LIVE=1` adds
  a real propose run against `CODE_SMOKE_REPO`): `npx tsx src/lab/code/smoke.ts`.

### `job/` — the job agent as a workflow (hive's `start_job` / `continue_job`)

Jobs V1 (strut `plans/jobs.md`; the hive half is `start_job` /
`continue_job` + the `job_turn` handler + one artifact reader route, a
CLOSED contract). Hive's Jamie chat hands strut a *job*: an id it mints,
passed on the LAUNCH — `POST /lab/workflows/job/run { job, input: { prompt,
workspace, session? }, callback }` — never inside `input`. `workspace` is
the hive workspace id the job belongs to (the hub strut serves every
workspace of its org): the YAML declares it (the runner strips an
undeclared key), the agent's message starts with `Workspace: @<slug>, the
one this strut belongs to.`, and that is what a pod is claimed for.
`session` is the agent's thread, the job id unless the launch names one
(hive passes none today). One workflow, `job` (category `job`, unstamped,
YAML only — every step is strut's) — and since 2026-10-13 the YAML is
STRUT's (`strut/src/workflows/job.yaml`, seeded at boot by `createStrut`,
strut plans/jobs.md §5): `job/seed.ts` ships only the lab's DELTA,
`JOB_SEED`, passed as `createStrut({ job })` in `createLabStrut.ts` — the
`pod/*` grants appended to `params.tools`, and appended to `params.system`
what only this deployment knows: that the strut belongs to one Hive
workspace (`@<slug>`, every other a peer reached with
`strut/run-workflow { peer }`) — merged into strut's template
before it is hashed and published, so the lab's `job` is one version of the
same workflow every strut has. Never seed `job` from here as well: one
seeder per name, or the two leave whichever template changed last active.

- `job/dir` → `checkouts` (`foreach` over `input.repos`, one
  `git/checkout { workdir: "{{ $job }}" }` each, INTO the job's directory —
  plans/jobs.md §2 as written; `repos?` on the launch is the generic way a
  repository reaches the agent's directory, kept there turn to turn, a fresh
  copy in a one-shot's own dir; `|| 0` is "none") → `agent` (cwd = the
  job's directory, `session: "{{ input.session || $job }}"`, a JSON
  `schema` for the answer) → `pack` `{ text, artifacts, ask?, cost,
  session }`. Hive passes no `repos` today (the URL rides in the prompt and
  the agent delegates to `code-change-pr` or a pod); `explore/` below takes
  the same launch. The directory
  (`<dataDir>/jobs/<job>/`) is the SAME for every run launched with that
  job, and by default so is the agent thread, so "revise step 2" is
  another launch with the same id: the agent opens the `plan.md` it wrote
  last turn with a transcript that remembers writing it. The two are tied
  by that one YAML line, not by strut (a job and a session are unrelated
  stores keyed alike — strut `plans/jobs.md` §2): a launch that names a
  `session` works on the job's files with that thread instead — a fresh
  one, or a second beside the first. Without a job (strut's Run button)
  the same YAML is a one-shot in the run's artifact directory — `$job` is
  undefined, so the agent is cold unless a `session` is given.
- **Deliverables are files.** The agent's `artifacts` list names what it
  produced or revised THIS turn — `{ id, kind?, title, label?, summary?,
  path | url | content }`, `id` stable across turns ("plan", not
  "plan-v3") — and strut's `run.end` callback carries the list RESOLVED to
  links (`/lab/jobs/<job>/files/plan.md`, `kind` from the extension), the
  same list at `GET …/runs/<runId>/artifacts`. An EARLIER job's
  deliverable the turn points the person to goes in the same list by the
  `url` `job/get` gave it (`/jobs/<thatJob>/files/…`): strut passes a
  strut-relative `url` through, and hive's reader serves any job it
  launched on that swarm, so it is a card like the turn's own. A path
  named in `text` is nothing the person can open (2026-10-10: a turn that
  answered from an earlier job's note named the note's path in its reply
  and returned no artifact). `ask: { message }` is the
  agent stopping for a decision; the run still ends `success` and the
  answer is the next turn's prompt.
- **`params` are the evolvable surface, and the whole point** (the owner's
  ruling in plans/jobs.md §2): `system`, `model`, `maxSteps` and `tools` —
  registry steps granted as `agentTools` ON TOP of the agent's built-ins
  (files, `bash`, `web_search`, `web_fetch`; never a `toolFilter`). Today:
  the `graph/*` reads (the Concept tree) and the `meta/*` READ + RUN tools —
  `meta/list-workflows`, `meta/get-workflow`, `meta/run-workflow`,
  `meta/get-run` — so a turn runs another workflow as a CHILD run under the
  same job (strut stamps the job on it and the child shares the job's
  directory, plans/jobs.md §4); the `job/*` reads — `job/list` (earlier
  jobs by words: a title, or what one delivered), `job/get` (what one
  produced, its files, its threads), `job/read` (one of its files; an image
  the agent then sees) — strut's job index (plans/job-index.md), so a
  turn builds on earlier work instead of starting over; the job's CLAIMS
  (strut plans/job-claims.md) — `meta/list-claims`, `meta/add-claim`,
  `meta/add-check`, `meta/edit-check`, `meta/verify-job`, `http/verify`
  and the page instruments `browser/verify` and `browser/walkthrough` (a
  frame per step, for a flow), all strut's — and the `Verify` page under `Job` saying when to claim, which instrument
  fits which surface and that a turn which says "done" verifies first; and
  the `pod/*` tools (`pods/`, below) — a
  hive sandbox the job HOLDS between turns. WHICH workflow or way for which
  kind of work is never in the prompt: strut's prompt opens the `Job`
  Concept (strut's page, `strut/src/concepts/Job.md`) first and follows the
  kind's page — the lab's kinds hang under it (`job/concepts`,
  `JOB_CONCEPTS`, and the sets below) —
  `Code Change` (`code/concepts`) sizes the request by ONE test — a change
  judged from its diff alone runs `code-change-pr` and the PR comes back
  as a `pull_request` artifact; anything else is built in a pod, after a
  plan (unasked) when it holds choices the person has not made; how
  detailed the request is never makes it smaller — and its child `Pod`
  (`pods/concepts`) says when a change needs the app running and how to
  make it in a pod;
  `Plan Mode`, when the person asks for a plan first, is ONE static
  `plan.html` (overview up front, the technical part in a collapsed
  `<details>`, the page style) revised until they approve. Strut's prompt
  also carries the one rule every kind shares: to SEE a page the agent
  runs `browser-explore` as a child run (its shots land in the job
  directory), never a browser it installs with `bash` — the repo2graph
  image ships Playwright's browsers, so `npx playwright` there just
  works, and Sonnet 5.5 reached for it (2026-10-08); the
  `code-change-pr` coder's prompt says the same (`browser/verify`, granted,
  is a check — it opens a URL and answers whether the expectations held —
  not a way to look around). Held back until
  running works: publishing / authoring (`meta/publish-workflow`,
  `meta/create-step`), the rest of `browser/*`. A job learns to do
  something new by a new VERSION of this workflow on the swarm — a line in
  `params.tools`, a paragraph in `params.system` — never by a hive PR.
- Smoke (offline — seeds, discovers, static-validates against the real
  registry, then runs it the way hive does: over HTTP with `job` + a local
  callback server, the `agent` step swapped for a fake that writes
  `plan.md` — two turns of one job behind one link, the sandboxed file
  route, and a launch without a job resolving to `/artifacts/<runId>/…`;
  `JOB_SMOKE_LIVE=1` adds one real turn): `npx tsx src/lab/job/smoke.ts`.

### `explore/` — the explorer (what another strut asks this one to run)

Strut federation's first use case (strut `plans/federation.md` §2.2): a
strut that has this swarm on file as a PEER runs `explore` here —
`strut/run-workflow { peer, workflow: "explore", input: { prompt, repos? } }`,
or the builder's `run_workflow` with `peer` — to ask this swarm's knowledge
graph, and the repositories it names, a question and get back what is
relevant with what the answer rests on. Locally it is an ordinary workflow
(the Run button, `run_workflow`). One seeded workflow, `explore` (category
`explore`, unstamped, YAML only — every step is strut's):

- **The job's skeleton and the job's launch.** `job/dir` → `checkouts`
  (`foreach` over `input.repos`, one `git/checkout` each INTO the run's
  directory) → `agent` (cwd = that directory, `session: "{{ input.session
  || $job }}"`) → `pack` `{ answer, sources: [{ name, why, ref_id?,
  node_type?, path? }], confidence: high | medium | low, cost }`. Input is
  the job's: `prompt`, `repos?` (URLs, one subdirectory each), `session?`,
  and `job` on the launch. A run has one directory, the job's or its own
  (strut plans/jobs.md §2): under a job the checkouts are kept there turn
  to turn (`workdir: "{{ $job }}"`); without one strut's `git/checkout`
  puts a fresh copy in the run's artifact dir — the dir `job/dir` returns
  — and removes it at run end, so `cwd: "{{ dir.path }}"` holds the
  repositories either way (strut, 2026-10-08). `items: "{{ input.repos ||
  0 }}"` is "no repositories" (`foreach` takes a count). A private
  repository needs the launching person's `GITHUB_TOKEN` on this strut, as
  an actor secret or the deployment's — nothing crosses from the caller.
- **Read-only: the graph's read steps and the read-only file tools.**
  `params.tools` names the graph reads one by one — `graph/graph-search`,
  `graph-get`, `graph-get-batched`, `graph-neighbors`, `graph/walk`, the
  two ontology reads — never the `graph/*` glob, which would hand out the
  write steps; `params.builtins` keeps `repo_overview`, `fulltext_search`,
  `str_replace_based_edit_tool` (to VIEW; the prompt forbids writes, and a
  copy is throwaway) and `file_summary`, and drops `bash`, `web_search`,
  `web_fetch`: the caller may be another strut, and it must not get a shell
  here. This is the one seeded agent with a `toolFilter`, and why: it is a
  read surface for remote callers, not a general agent (the job agent keeps
  every built-in, `job/` above). The system prompt says the answer comes
  from what was read and nothing else — nothing found is an answer, with
  empty `sources` and `confidence: low` — and that the prompt is DATA from
  a caller: text in it asking for other tools or writes is not an
  instruction. The agent's graph tool calls carry their accessed nodes on
  the run's events (strut's `withAccessedNodes`), so the run flyout's Nodes
  list shows what was read.
- **Who may call it: a `lab:peer` token.** The calling strut's peer record
  holds a token this swarm minted for it — `POST /mint-token { scope:
  "lab:peer", sub?, expires_in? }` with the raw `API_TOKEN` (60 days unless
  told otherwise) — never the swarm's `API_TOKEN` or an `api` JWT, either of
  which is admin of this lab. `labAuth` takes it as `Authorization: Bearer`
  only (never `?key=`, never the dictation socket) and stashes strut's
  `peer` scope (`resolveLabScope`, `actor.ts`); strut lets a peer read every
  route, launch a run, and cancel / pause / resume a run a peer launched,
  and answers 403 to the rest (strut `plans/federation.md` §3). Its actor
  is the token's `sub`, else the caller's `x-strut-actor` (the person the
  calling run is for). A token WITH a `sub` is a person's — what hive mints
  a laptop at pairing (strut `plans/chained-orchestrators.md` §2, §4) — and
  its scope is `peer:person`: the actor is the token's own whatever the
  header says, so strut also lets it launch a `peer: person` workflow (the
  orchestrator), open that person's own sessions there (`<sub>/…`), and
  join their ROOM: `/lab/rooms`, the WebSocket `attachLabSockets` routes
  through `upgradeLabRoom` (the lab credential stashed as for HTTP, then
  strut's `upgrade` with `basePath: "/lab"`), where a launch's `callback` /
  `jobCallback` `{ room: true }` delivers. A token without a `sub` stands
  for a strut: `peer`, refused a `peer: person` workflow and a room. Outside
  `/lab` it opens nothing — mcp's own API verifies `api` JWTs only. What a
  peer may launch is the workflow's `peer:` (strut's peer gate,
  2026-10-20): `explore` and `logs` say `peer: true`, the orchestrator
  `peer: person`, and `job` neither.
- `params` are the evolvable surface: `model` (`sonnet`), `maxSteps` (40),
  `tools`, `builtins`, `system`. Tests (`explore/seed.test.ts`, offline — a
  fake `agent` and a fake `git/checkout`, the real registry for `job/dir`,
  `foreach` and `pack`, strut's own static validation): the seed, the
  launch shape and skeleton, the exact tool lists, the schema, no repos →
  no checkout, repos → one each (fresh on a one-shot, `workdir` = the job
  under one) and named in the prompt, `session` precedence, a launch
  without a prompt refused.

### `orchestrator/` — Jamie's turn as a workflow (hive's canvas chat + its proposals)

Strut's second shipped workflow, `orchestrator` (strut
`src/workflows/orchestrator.yaml`, seeded at boot by `createStrut` as
`job` is): one turn of a long conversation — an agent on the caller's
`session` with no built-ins (`toolFilter: [none]`), strut's memory and
job steps as tools, jobs started DETACHED. Hive launches one per Jamie
message (strut `plans/proposals.md` §9). As with `job`, the lab ships only
its DELTA, `ORCHESTRATOR_SEED` (`orchestrator/seed.ts`), passed as
`createStrut({ orchestrator })` beside `job: JOB_SEED`; never seed
`orchestrator` from here as well.

- **`tools`**: strut's `graph/*` steps on this swarm's graph — the reads
  (`graph-search`, `graph-get`, `graph-get-batched`, `graph-neighbors`,
  the two ontology reads) and the seven writes hive's `propose_*` tools
  were (`create-node`, `edit-node`, `create-triplet`,
  `create-batch-triplet`, `move-node`, `delete-edge`, `delete-node`) —
  plus `meta/list-workflows`, `meta/get-workflow`, `meta/run-workflow`, so
  a turn runs `explore` inside it. BY NAME, never `graph/*`: the glob
  would hand out `graph/create-schema` and `graph/edit-edge` ungated.
- **`approve`**: exactly the seven writes. PROPOSALS (strut
  `plans/proposals.md`, `src/proposals.ts`): a call to a gated tool does
  not run the step — strut records a proposal, the model gets `{ proposal,
  status: "pending" }` and goes on, hive draws the card from `GET
  /lab/proposals/:id`, and on accept (`POST /lab/proposals/:id/decide`, the
  clicker as `x-strut-actor`) strut runs the step in code with the recorded
  input. The proposing session hears the decision on its next turn as
  `[proposal-result]` lines before the prompt. The reads stay immediate.
  `seed.test.ts` fails on a write leaving the gate or an eighth joining it.
- **`system`**: what only this deployment knows, after strut's prompt —
  which Hive workspace this strut belongs to (`@<slug>` on the message's
  first line, as `JOB_SEED` says); that every graph write is proposed,
  never made, and what the result line looks like; what a Concept is on
  this graph — `name` (the key), `description`, the body in `docs`
  (jarvis's `documentation` is deprecated and unindexed) — and how a NEW
  one is proposed: ONE `graph/create-batch-triplet` with the Concept
  inline as the target of both `<parent> -PARENT_OF->` and `<HiveWorkspace>
  -PROCESS->` (batch-triplet dedupes identical inline sides). The anchor
  is the proposal's own because nothing else writes it for a Concept made
  this way: hive's `handleApproval` wrote it for `conceptCreate` and
  `concept-anchor.ts` writes it for seeded pages at boot, and strut has no
  after-apply hook. One card, one accept, placed and reachable from the
  workspace node.
- Hive-owned kinds (initiatives, milestones, prompts) are later: the
  `hive/*` steps of plan §9 do not exist yet, so `handleApproval` keeps
  those branches. Tests: `orchestrator/seed.test.ts` (pure: the merged
  YAML's `params.approve`, every grant a step of the real registry, the
  agent step reading both from params, the section after strut's).

### Concept pages — what the builder and the job agent know about this deployment, read off the graph

Strut's builder prompt knows how to build workflows, and its job prompt how
to work a job, never WHAT a deployment builds or which way does which kind
of work. That knowledge is two trees of `Concept` nodes in the knowledge
graph. Strut ships the two roots (`strut/src/concepts/`, seeded inside
`createStrut`); the lab ships pages under them:

```
Workflow Builder            what this workspace builds       (strut)
├─ App                      a kind: an app over workflows    (strut)
│  └─ Benchmark App         an app over a benchmark          (eval/concepts)
└─ Janitor                  a kind: its convention           (janitor/concepts)
   └─ <a mandate>           an instance                      (the workspace's own)

Job                         what a job agent can do here     (strut)
├─ Code Change              a kind: which workflow, how      (code/concepts)
│  ├─ Pod                   when a change runs in a sandbox  (pods/concepts)
│  │  └─ Pod Config > Pod Setup > …                          (pods/, pod-setup/concepts)
│  └─ <a way | a repo>      a sandbox workflow / repo rules  (the workspace's own)
├─ Explore                  a way of working: a note kept    (job/concepts)
├─ Logs                     what the logs say: a note kept   (logs/concepts)
├─ Onboarding               a kind: a workspace's first pod  (job/concepts)
├─ Plan Mode                a way of working: plan.html first (job/concepts)
└─ Verify                   claims + checks, verify before done (job/concepts)
```

Both readers are strut's. The builder's prompt carries `Workflow Builder`'s
page and its kinds (strut's default `chatSystem`, `src/ai/builder-system.ts`
— the lab passes no hook; the builder opens a kind with `graph_get`, which
strut offers because `createLabStrut` passes the workspace's graph). The
`job` agent opens `Job` every turn as a tool result (strut's `job.yaml`
says so; `JOB_SEED` adds nothing about it). So WHICH workflow does which
kind of work is graph data a workspace edits, never a line in a prompt.

**What ships here is seeded into EVERY workspace.** A Concept that only
makes sense for one kind of workspace — a mandate about eval rubrics, a
legal topic — is that workspace's to add (the builder, hive's Learn page),
never a file in a seeded `concepts/` dir. `seed.test.ts` fails on a seeded
file, or an engine description, that names such a domain. A benchmark is
not one: `Benchmark App` ships to every workspace, generic, because any
workspace may measure its workflows; what one benchmark scores, and in
which field, stays that workspace's.

- **The seeder is strut's** (`seedConcepts(graph, sets)`, `ConceptSet`,
  `strut/src/concept-seed.ts`, on Neo4j and Ladybug alike). A SET is one
  directory of `<Name>.md` files and the prefix its nodes are stamped with;
  a file is one Concept (front matter `description`, optional `parent` —
  another Concept's name, from any set in the call or already in the graph
  — body = `docs`). Reconciled per node by `unique_source_id =
  <prefix><file>@<sha256[0..12]>`: unseen → create; same stamp → keep
  (nothing written: an edit made in the graph sticks until the file
  changes); our path, older hash → update; our path, deleted → restored;
  no stamp or another's → never touched. `PARENT_OF` follows the files
  between the Concepts a call owns; an edge with anyone else's Concept at
  either end is never touched. A set's `retired` files have their nodes
  soft-deleted while they still carry that file's stamp (`Janitor`'s set
  retires `Overfit Concept Janitor.md`).
- **Strut owns the root pages' names.** Before strut shipped them the lab
  seeded `Workflow Builder` (`lab/builder/concepts/`) and `Job`
  (`lab/job/concepts/`); strut counts those stamps as its own
  (`legacyPrefixes`). `Workflow Builder.md` was byte-identical, so a
  deployed swarm keeps its node and stamp; the lab's `Job.md` was replaced
  by strut's generic page on the first boot after the pin bump, and its
  Hive-workspace paragraph moved into `JOB_SEED` (its browser paragraph
  into strut's `job.yaml`). The lab
  never ships a file under one of strut's names (`seed.test.ts`). Every
  lab set keeps its own prefix, so its stamps stay its own: a file that
  moves to another set reads as someone else's.
- **Order.** `createLabStrut` seeds the lab's sets AFTER `createStrut`
  returned (strut seeds its pages inside it; the lab's hang under them by
  name), in ONE call with `STRUT_CONCEPTS` riding along — all kept, nothing
  written — so strut's pages are owned in that call too: their `PARENT_OF`
  edges to the lab's follow the files, and they are anchored with the
  lab's.
- **The anchor** (`concept-anchor.ts`, `anchorConcepts(graph, outcomes)`,
  the lab's: strut knows no `HiveWorkspace`): every Concept the call owns
  (create / update / keep — strut's pages and the lab's) gets
  `HiveWorkspace -PROCESS-> <Concept>` to the one workspace node hive's
  mirror writes into this graph — what hive writes for every Concept a
  person approves (jarvis migration 111), so its graph walker reaches a
  Concept from the workspace node. Nothing in strut or the lab reads it.
  No workspace node (a standalone strut) or more than one (never guess) →
  no edge, one log line. Neo4j only.
- **Trust.** The builder's section has system authority over a builder
  that can publish steps and run bash, and it comes from a node the
  workspace can edit: strut inlines only the entry page, caps its docs and
  frames them as the workspace's notes. An agent granted graph WRITE steps
  can rewrite these pages: grant writes by name, as the janitor engine
  grants reads.
- Needs the `Concept` schema (a jarvis-seeded swarm, or
  `STRUT_GRAPH_SEED_ONTOLOGY=1`); without it nothing is seeded, one log
  line. No graph (the fs workspace): nothing seeded, no section.

### `janitor/` — janitors: the engine workflows, their four steps + the `Janitor` Concept's mandates

A janitor is an automated agent that cleans something up — code, a Concept
tree, other data. Its MANDATE (what "dirty" means, what to flag, what to
propose) lives in the knowledge graph as a `Concept` under `Janitor`, one
child per janitor; the ENGINE is one seeded workflow, `graph-janitor`, and
a janitor IS an automation on it: `input.concept` names the mandate,
`input.start` the Concept whose subtree to sweep, the automation's schedule
is the janitor's schedule and its enabled switch turns it off. One
automation per janitor, so a workspace grows its own by adding a child
Concept (hive's Learn page, Jamie, strut's `graph/*` steps through the
builder's `run_step`) and scheduling it — nothing to publish. Automations
are not seeded. The builder learns HOW from `Janitor`'s docs (see
Concept pages above); the workflow's description says what the workflow does.

- **The engine** (`workflows/graph-janitor.yaml`, category
  `graph-maintenance`, unstamped, SEED_OPTS): three
  `graph/graph-get` reads BY NAME — `params.parent` (`Janitor`, with its
  children), the mandate, the start — each an exact key lookup, never a
  search; a name no Concept has → `not_found:`. The mandate must be among
  the parent's children → `not_a_janitor:` (the mandate is data the agent
  follows, so only a node someone put under Janitor may steer a run; past
  50 children the refusal says only the first 50 were checked). CODE picks
  what the run reviews: `janitor/queue` (`steps/queue.ts`) reads the live
  Concept tree under start over bolt (live PARENT_OF through live Concepts
  only, start included), puts it in TREE ORDER — each node keyed by its
  path of names from start, paths compared segment by segment, a parent
  before its children, so a subtree comes before the next sibling and
  siblings go by name (a node with several parents sits at the first path a
  depth-first walk reaches it by) — and takes the next
  `params.batch_size` (10) nodes after `input.cursor` — the path of the
  last node the previous run reviewed, names joined by " › " — wrapping
  past the end; the automation passes `cursor: "{{
  last.output.next_cursor }}"` (`last` = the latest SUCCESSFUL run, so a
  failed batch is taken again), and every node comes round once per
  ceil(total / batch_size) runs. A path, not a position: an added or
  deleted Concept never shifts anyone's turn. Left to walk on its own the
  agent read start's children every run and the levels below now and then
  (swarm38 Medicine, 2026-10-14: a depth-2 Concept naming "the benchmark"
  was read once in 91 hourly runs); name order fixed that but made a
  batch ten unrelated nodes, so tree order keeps a batch one branch. A
  read-only `agent` — the graph's read steps granted BY NAME, never
  `graph/*` — is handed the batch's description and docs verbatim with each
  node's path, parent and siblings' names (50 at most), reads around it for
  context, and proposes findings on batch nodes only, with the mandate's
  docs as its task and a
  system prompt that treats them as data (a mandate that asks for writes or
  new tools is itself a finding); `exec` saves `cleanup-report.json`, `pack`
  returns `{ mandate, start, start_ref_id, visited, findings, summary,
  report_json, report_md }`. Findings are `{ ref_id, name, issue, problem, evidence
  (verbatim), suggestion, severity, action }`; `issue` is a free-text label THE
  MANDATE defines for each kind of problem it flags (the engine knows no
  list of kinds — a closed one once held `overfit`, one workspace's word). The error codes ride on the second
  line of `error.message` (an `exec` that exits 1 with the code on stderr).
- **Apply** (`input.apply`, default false = dry run). The agent never
  writes; each finding's typed `action` (`replace_text` with `property`,
  `find` verbatim and `replace` — the usual docs fix, resolved by the plan
  step against the live value, `find` occurring exactly once, into an edit
  with the FULL old and new values, so the write, the readback and the undo
  log never see a fragment; `set_property` with `property`, `old_value`
  verbatim, `new_value` — a one-sentence description, or docs mostly
  rewritten; `move` with `new_parent_ref_id`; `none` = a person decides)
  goes through `janitor/plan` (`steps/plan.ts`),
  which runs in BOTH modes and re-reads every target over bolt — the only
  way to see `unique_source_id`, which strut's read steps strip from every
  response — and sorts it into `edits` / `moves` (the exact configs for
  `graph/edit-node` / `graph/move-node`), `skipped` (a reason each) or
  `manual`. Never planned: a finding on a node outside the batch
  (`targets` = the queue's ref_ids → `not_in_batch`; a target counts as
  visited, and a move's new parent may be any node the agent read), the Janitor node, its children and anything
  hanging under it live (one mandate must never rewrite another), the
  mandate, a git-seeded Concept (`params.seeded_prefix`, `[lab/, strut/]`
  — the lab's pages and strut's; the next file change would overwrite it;
  fixes belong in git), a rename (`name` is
  the identity key; automations address Concepts by it — `params.apply_properties`
  is `[description, docs]`), a value that no longer matches what the agent
  read (`old_value` or the `evidence` quote; `stale`), a move with no or
  several parents; `params.apply_max` caps writes, most severe first;
  `params.apply_min_severity`. With `apply=true` an `if` gate opens two
  `foreach`es over strut's own write steps (so every write is a tool call
  in the run's record, with its provenance edge), then `janitor/readback`
  (`steps/readback.ts`) reads each result (graph steps return error
  strings) and every node back, writes `applied.json` — the UNDO LOG:
  every applied change with the value or parent it replaced — appends the
  outcome and the revert instruction to `report.md`, and fails
  `apply_failed:` (first line of `error.message`) unless every write read
  back. `plan.json` is written in both modes. `result` depends on both
  branches: `{ …, apply, plan: { edits, moves, skipped, manual, counts },
  applied, applied_json, report_plan }`, plus `visited` (the batch),
  `queue: { from, size, total, wrapped }` and `next_cursor`.
- **Revert** (`workflows/graph-janitor-revert.yaml`): `{ run_id, changes?
  }` → `janitor/revert-plan` (`steps/revert-plan.ts`) reads that run's
  `applied.json` through the artifacts capability (kept forever; one run
  may read another's by id) and emits the inverse as the same plan shape —
  an edit puts the replaced value back (deletes the property when there
  was none), a move goes back under the previous parent — ONLY where the
  node still holds what the janitor wrote (`moved_on` otherwise, never
  overwritten; `unknown_change` for an id the run never applied;
  `no_applied_log:` when the run has none). Then the same foreach +
  readback: the revert is logged with `revert_of`, so it can be reverted.
  The graph itself keeps no history (no before-image on a node or an
  `ACCESSED` edge), so the log under `/artifacts/<runId>/` is the only
  copy. The four steps are seeded like explore's (`SEED_STEPS`, one
  self-contained file each importing only "strut": `graphCtx` and the bolt
  read are inlined per file).
- **The tree** (`concepts/Janitor.md`, seeded by strut's `seedConcepts`):
  `Janitor` (parent `Workflow Builder`), whose docs STATE THE CONVENTION —
  what a child is, how to run one, and the three steps to add one
  (`graph/create-node`, `graph/create-triplet` `PARENT_OF` from `Janitor`,
  an automation). NO MANDATE SHIPS: what counts as dirty is each
  workspace's to say. `Overfit Concept Janitor` (Concepts overfit to the
  rubric / eval that spawned them: one eval workspace's focus) shipped to
  every workspace for a day; it is now `retired`, and lives on as
  `fixtures/Overfit Concept Janitor.md`, which the tests and the smokes
  plant themselves (`JANITOR_FIXTURES`; not under a `concepts` dir, so not
  in the build).
- The engine needs a model key (or a Mothership delegation for the
  automation's owner — automations run as the workflow's owner, so claim
  the workflow before scheduling).
- Tests: `src/lab/janitor/seed.test.ts` — pure (parse + reconcile rule, the
  committed tree, the builder's section) and the engine offline (real
  if/pack/exec/artifacts-dir over the seeded YAML, fake `graph/graph-get` +
  agent: the three reads by key, the guard, what the agent is handed) in
  `test:node`; `apply.test.ts` — the plan / readback / revert rules pure,
  and both workflows' wiring offline (the gate, the foreach bodies, what
  readback gets; plan, readback, the write steps faked). Live suites (the
  seeder: reconcile, the anchor on every owned Concept, `PARENT_OF`
  following the files, someone else's Concepts left alone; then the
  committed tree, the builder's section and the engine over the REAL graph
  steps with a fake agent; then apply + revert for real: dry run, apply,
  revert, a second revert finds everything moved on, a dry run has no undo
  log — all in `seed.test.ts`, one process: test files run in parallel and
  these share the Janitor tree) run when
  `STRUT_TEST_NEO4J_URI` points at a throwaway Neo4j. Nothing else may use
  that database while they run: strut's `test:graph` wipes it, and two
  suites at once fail with deadlocks and missing nodes. The first run right
  after a wipe can also fail in strut's boot ("insanely frequent schema
  changes": Neo4j is still dropping the wiped indexes) — run it again. The
  live smoke
  (`src/lab/janitor/smoke.ts`: a REAL model through `createStrut` over a
  planted `Law` subtree — two overfit Concepts, three clean — asserting
  both are flagged `overfit`, the clean doctrine is not, `visited` is
  Concepts only, no graph write tool was called, and `report.md` +
  `cleanup-report.json` are in the run's artifacts; then the
  `not_a_janitor:` refusal; ~70 s, ~$0.20): `ANTHROPIC_API_KEY=…
  NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=struttest
  STRUT_GRAPH_SEED_ONTOLOGY=1 STRUT_GRAPH_EMBEDDINGS=off npx tsx
  src/lab/janitor/smoke.ts`. The agent's working dir is empty when it
  starts, so the step gives it no cwd preamble: the prompt names
  `{{ dir.path }}`, or the model guesses a directory for `report.md`.
  The builder smoke (`src/lab/janitor/builder-smoke.ts`: the real
  `createLabStrut`, one chat message — "make me a janitor that flags
  near-duplicate Concepts under Law, daily at 6am" — then a report of the
  builder's tool calls and what it left behind: the mandate under
  `Janitor`, the automation on `graph-janitor`; nothing asserted, a
  model's route varies): same env, `npx tsx
  src/lab/janitor/builder-smoke.ts`.

### `eval/` — generic, reusable eval primitives (NOT an experiment)

Domain-agnostic eval substrate, shared by every experiment. See
strut's `specs/EVAL_SPEC.md`. **Steps only** — no domain config baked in:

- `eval/steps/score.ts` (`eval/score`) — match a produced set vs an expected
  gold set by a `rubric`; recall-weighted F-beta score.
- `eval/steps/reflect.ts` (`eval/reflect`) — propose a better prompt from the
  AGGREGATED results across a dataset (multi-example → avoids overfitting).
- `eval/steps/optimize.ts` (`eval/optimize`) — the `eval → keep best → reflect`
  loop, run as a single detached "background job" (EVAL_SPEC §8). Runs
  sub-workflows via an injected `services.optimizer` (closure over `strut.run`).
  Multi-example: takes a dataset (`evalInputs[]`), evals the candidate over
  every entry per generation and AVERAGES the scores (the overfitting fix,
  §11.2) — the per-example results array is fed to reflect. Each entry carries
  its own gold (e.g. `{ owner, repo, expected }`), read by the eval workflow
  from `input`. (A single example is just a 1-entry `evalInputs`.)
- `eval/steps/evolve-loop.ts` (`eval/evolve-loop`) — the generic hill-climb
  over WORKFLOW VERSIONS (EVOLVE_SPEC §5.3.3, generalized from its first
  instance): runs a domain's one-generation workflow (author → run candidate
  over tasks → digest) up to N generations, briefing each author with every
  prior attempt anchored to the best-so-far, flipping exploit→explore after
  `exploreAfter` non-improving attempts. Fitness is the generation digest's
  `fitness` (fallback `meanPassRate`), named in briefings by `fitnessName`;
  improvements must clear `improveMargin` (judge noise for LLM-judged
  domains — 0.02 is a sane start; produce-sampling noise for deterministic
  scorers — gaia 0). Needs `services.optimizer`. Wired by `gaia-evolve`
  (accuracy).
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
- `eval/aggregate-scores`, `eval/build-eval-chain`, `eval/criterion-refs` —
  rubric-judged SCORING plumbing (exercised by wfbench): zip a rubric with per-criterion judge verdicts
  (same order; null/error = honest FAIL; length mismatch refuses) into
  scores_json; build the idempotent EvalSet→EvalTrigger→EvalTriggerOutput→
  CriterionResult(+EvalRequirement) batch-triplet payload for
  `graph/create-batch-triplet` (`workflow` names the scored workflow);
  recover the persisted CriterionResult ref_ids from the batch results.

**Naming rule:** `eval/*` = generic. The eval *workflows* that wire these with
a rubric/task/dataset belong to the experiment and are named `<experiment>-…`.
The concepts experiment's live in `concepts/workflows/`: `concepts-eval`
(harness), `concepts-eval-score` (rubric), `concepts-eval-reflect`
(task+guidance), `concepts-optimize` (the wired loop). A new experiment `foo`
adds `foo-eval`, `foo-eval-score`, … reusing the same `eval/*` steps.

## Running / gotchas

- **The `/lab` AI chat can now run experiments autonomously.** strut threads the
  lab `services` bag into the chat agent's `run_workflow` tool, so the builder
  can launch an eval or an `eval/optimize` loop (which needs
  `services.optimizer`) — not just service-free core/lab workflows. And chat is a detached background job (see `strut/AGENTS.md`):
  describe an eval, tell it to "try it and report back", close the browser, and
  the turn keeps running server-side (persisted to `chats/<id>/`). Reopen to
  reattach. Long runs DISPATCH: a `run_workflow` still executing after
  `STRUT_CHAT_RUN_WAIT_MS` (default 60s) auto-detaches — the tool returns
  `{ status: "running", runId }`, the agent ends its turn, and when the run
  finishes a `[run-notification]` message wakes the chat with the result
  (capped at `STRUT_CHAT_MAX_AUTO_TURNS` consecutive machine-triggered turns;
  a human reply resets the cap). See `strut/plans/dispatch-run-notifications.md`.
- Needs **Neo4j** + `GITHUB_TOKEN` + an LLM key (e.g. `ANTHROPIC_API_KEY`).
- Workflow YAML templates are seeded into the workspace
  (`STRUT_LAB_WORKSPACE`, default `./lab-workspace`) on first boot, then
  edited/versioned via the strut UI.
- strut is a **git dependency**: `package.json` pins
  `git+https://github.com/stakwork/strut.git#<commit>`, and `yarn install`
  clones it and runs its `prepare` (engine `tsc` + `web/` vite build), so no
  strut checkout is needed to boot. To bump: edit the pinned commit, `yarn
  install`, commit `yarn.lock`. The GitHub shorthand (`stakwork/strut#sha`)
  does NOT work — yarn fetches those as a source tarball and skips `prepare`.
  `mcp/.yarnrc` sets `network-concurrency 1` and must stay: yarn 1 runs a
  nested install for the `prepare`, and with parallel fetches on a cold cache
  the two installs corrupt each other's cache entries (see the comment there).
- To hack on strut and mcp together: `yarn link` in a strut checkout, then
  `yarn link strut` here (yarn keeps the symlink across installs). Rebuild
  strut (`npm run build`, `npm run build:web`) and restart mcp to pick up
  changes; `yarn unlink strut && yarn install --force` returns to the pin.
- The strut UI is path-agnostic (relative assets + runtime API base), so it
  works under `/lab` (with the `/lab` → `/lab/` redirect in `mount.ts`).
- Trigger a run: `POST /lab/workflows/bootstrap-then-process/run` with
  `{ input: { owner, repo, token } }`, or use the UI at `/lab/`.

## Run it end-to-end (manual)

Nothing is automated yet — no CI job exercises `/lab`. Manual steps:

1. **Neo4j**: `cd mcp && docker compose -f neo4j.yaml up -d` (wait healthy).
2. **Env**: `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` (and `NEO4J_HOST`/`NEO4J_USER`/
   `NEO4J_PASSWORD` if not default).
3. **Start mcp**: `cd mcp && yarn dev` (serves on `:3355`).
4. **Init + seed** (lazy on first hit): `curl localhost:3355/lab/health`,
   then `curl localhost:3355/lab/workflows` to confirm the 3 workflows
   seeded.
5. **Run** (detached launch + reattach — see strut's `specs/EVAL_SPEC.md` §8). The
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
   concept-listing HTTP endpoint yet; strut only exposes `/workflows`.)

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
into the workspace and strut loads them via dynamic `import()`. Plain `node`
can't import `.ts`, so the prod server runs as `node --import tsx
build/index.js` (`start` script + Docker `CMD`); `tsx` is a runtime
dependency. This is what lets agents/users author steps in TypeScript and
have them run in prod without a compile step.

**Known follow-ups** (not blockers for a basic run): `/lab` runs bypass mcp
auth (mounted before auth middleware).
