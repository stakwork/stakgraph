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
5. **Run** (SSE stream):
   ```
   curl -N -X POST localhost:3355/lab/workflows/bootstrap-then-process/run \
     -H 'content-type: application/json' \
     -d '{"input":{"owner":"OWNER","repo":"REPO","token":"<gh token>"}}'
   ```
   Use a **tiny repo** first (LLM cost/time per PR+commit).
6. **Verify**: query Neo4j directly — `MATCH (c:Concept) RETURN c.name,
   c.description` — or watch the SSE `step.*` events. (There is no
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

**Known follow-ups** (not blockers for a basic run): `/lab` runs bypass mcp
auth (mounted before auth middleware).
