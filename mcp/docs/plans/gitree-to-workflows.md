# `/gitree/process` and `/repo/describe` → lab workflows

Status: proposed (2026-10-13), revised the same day after a cross-repo read
(stakgraph, strut e42a2d9, hive, sphinx-swarm). Follows stakgraph#1774 (lab
concept storage at gitree parity) and stakgraph#1775 (`describe-code` + strut
pin with the code Schema `index` change).

## Goal

The two endpoints keep their URL and response, so their callers barely
change. Behind them, mcp launches the lab's strut workflows for any request
that names an actor, and keeps the old code for the ones that don't:

| Endpoint | Workflow |
| --- | --- |
| `POST /gitree/process` | `bootstrap-then-process` |
| `POST /repo/describe` | `describe-code` |

Each run is then a tracked strut run: visible in `/lab`, cancellable,
resumed after a crash, and billed to a person. Embedding needs no work here:
the writers write fields, and strut's embedding pass turns them into
`text_embeddings`.

Hive's change is to say who each call is for: `x-strut-actor` on the two
gitree calls, and a `strut_actor` field on the sync bodies standalone
forwards, plus the delegation it already ensures on its other strut launches.

## Today

**`POST /gitree/process?owner&repo|repo_url&token&summarize&link&analyze_clues`**
(`gitree/routes.ts`, `gitree_process`)
- **Response:** returns `{ request_id, status: "pending" }` immediately and runs in the background.
- **On completion:** `asyncReqs.finishReq(request_id, { status, message, usage, cumulativeUsage, sessionId, linkResult? })`.
- **Busy flag:** holds `startTracking("gitree/processRepo")` for `/busy`.
- **Side channel:** `GET /gitree/concepts` (`gitree_list_concepts`) reports
  `processing: isProcessing` (the module flag `gitree_process` sets) and
  `cumulativeUsage` (`storage.getTotalUsage(repo)`, which `gitree_process`
  grows with `addToTotalUsage`). Hive's `learnings/concepts/route.ts` passes
  both through, and `LearnSidebar.tsx` disables the Learn/seed buttons on
  `processing` and renders `UsageDisplay` from `cumulativeUsage`. Mcp's own
  `web/src/components/Sidebar.tsx` polls `processing` to know when a run ended.
- **Callers:**
  - hive `src/services/swarm/auto-learn.ts` → `triggerAutoLearn`: after the stakgraph sync's completion webhook, with the workspace **owner's** PAT. Fire-and-forget.
  - hive `src/app/api/learnings/route.ts`: the "learn" button, with the **requesting user's** PAT. Fire-and-forget.
  - Both always send `summarize=true&link=true` and never `analyze_clues`; neither reads the response.
  - mcp's web SPA, `Sidebar.tsx` `handleGenerateConcepts`: `repo_url` + the token from Settings, authenticated by the SPA JWT (minted with no `sub`, so no actor). **Stays on the old code** — see "Who the run is for".

**`POST /repo/describe { cost_limit, batch_size, concurrency, repo_url, file_paths, embed, apiKey, model }`**
(`repo/descriptions.ts`, `describe_nodes_agent`)
- **Response and progress:** returns `{ request_id, status: "pending" }`, then reports `updateReq` progress (`processed`, `total_cost`, `total_tokens`, `current_batch_size`) and a final `{ success, processed, total_cost, total_tokens, usage }`.
- **Model:** the OpenRouter qwen model when an OpenRouter key exists, else the env default provider's light model.
- **Callers:**
  - standalone Rust `call_mcp_embed` (`standalone/src/utils/mod.rs`): fire-and-forget, `x-api-token`, no actor. Standalone calls it when the sync body's `embeddings` names the repo. Hive's per-workspace push webhook (`webhook/[workspaceId]/route.ts`) sends `embeddings: true` whenever `Repository.embeddingsEnabled` is set, and that defaults to `true`, so this runs after nearly every push; a manual ingest sends the same per-repo flag. Incremental syncs pass the modified files as `file_paths`, with standalone's `embeddings_limit` (default $5) as the cost cap.
  - mcp's web `EnrichButton.tsx`, which sends the user's Settings `apiKey` when set, polls `/progress`, and shows `processed` / `total_cost`. **Stays on the old code.**

**`POST /repo/embed`** writes BGE `embeddings`. It has no callers (nothing in
mcp's web, hive, standalone or sphinx-swarm posts to it). BGE is still a live
read path today — `VECTOR_SEARCH_QUERY` (the index is `ON n.embeddings`)
behind the `vector_search` agent tool, and `gitree_search_concepts` via
`gds.similarity.cosine(c.embeddings, …)` — so this plan leaves it alone and
does not stop the old describe path from writing BGE. The workflow path writes
no BGE: code-node BGE comes from body vectorization anyway (describe only
overwrote it), and gitree's `GraphStorage.initialize()` backfills BGE on any
Concept that lacks it, on every list/search call. Dropping BGE is its own plan.

## The proxies (mcp)

One module, `src/lab/proxy.ts`, mounted where the old routes are in `index.ts`,
behind the same `authMiddleware`.

1. **Dispatch on the actor.** The proxy reads `x-strut-actor` the way
   `labAuthorized` (`lab/mount.ts`) does: honored only beside a matching
   `x-api-token` (the header proves the caller is hive or standalone). The
   SPA's Bearer JWT, `?token=`, Basic, and dev mode with no `API_TOKEN` all
   yield no actor. **With an actor** the request launches the workflow.
   **Without one** it calls the old handler (`gitree_process`,
   `describe_nodes_agent`) unchanged. This is not a rollout shim: it is the
   SPA's path (and any older hive/standalone build's), and it stays until the
   SPA is reworked. The two deploys are therefore order-independent.
2. **Reach the lab strut.** Export `labStrut()` from `lab/mount.ts`. It is the
   same lazily built `Strut<LabServices>` instance `/lab` serves, so the runs
   land in the lab's run store and UI.
3. **Launch through `strut.run`.**
   `strut.run(workflow, input, { actor, runId: generateRunId(), params })`
   is the tracked launch: a controller (cancel, live status), `run.start`
   with `actor` and `principal`, and auto-resume. Don't await it in the
   request; wire its promise to `asyncReqs`:
   - `startReq()` before the launch, and return `{ request_id, status: "pending", run_id }`;
   - `finishReq(request_id, mapped(result))` on success, `failReq` on error.

   No `startTracking`: see "`/busy`" below.
4. **Map the inputs.**
   - `/gitree/process`: `owner` / `repo` (or parsed from `repo_url`) →
     `{ owner, repo }`. `summarize` and `link` are always on in
     `process-repo-chronological`; `analyze_clues` is ignored (clues are gone).
     A `token` in the query goes into the actor's secrets (below); with none,
     the run reads whatever the actor or the deployment already holds, so a
     missing token is no longer a 400 (hive always sends one).
   - `/repo/describe`: `repo_url`, `file_paths`, `cost_limit` → the workflow
     input. `batch_size`, `concurrency` → per-run `params`. **Model:** the
     YAML default becomes `haiku`; the proxy passes
     `params.model = openrouter/qwen/qwen3-coder-30b-a3b-instruct` when
     `hasApiKeyForProvider("openrouter")`, mirroring `descriptions.ts`
     (today the YAML hardcodes OpenRouter, which fails outright on a swarm
     without that key). A `model` in the body wins. `embed` is ignored (the
     pass embeds). `apiKey` is refused on this path: a key in a request has
     no place in a billed run, and the one caller that sends it (the Enrich
     button) has no actor and never reaches here.
5. **Map the result.**
   - `/gitree/process`: `{ status, message: "Processed owner/repo", run_id, output }`.
   - `/repo/describe`: `{ success: true, processed, total_cost: cost, failed, stopped, run_id }`.
     `describe-code` is one step, so there is no mid-run `updateReq`
     progress on this path; nothing with an actor polls for it.
6. **One run per repo, surviving restarts.** This replaces gitree's
   process-wide `isProcessing` flag. Two concurrent sweeps of one repo would
   race on its checkpoint. The proxy keeps `inFlight: Map<"owner/repo", { request_id, run_id }>`,
   set at launch and cleared when the run promise settles; a second call for
   a repo in the map answers with that entry. An in-memory map alone is not
   enough: after a restart strut auto-resumes the run while the map is empty.
   So before launching, the proxy also asks strut — in-process
   `strut.app.fetch("/workflows/bootstrap-then-process/runs")` (no key is
   configured on the lab strut, so no header) — for summary-less runs whose
   `status` is not `stale`, reads each one's `run.start.input` from
   `strut.store.getRunEvents`, and treats a match on `owner/repo` as in
   flight: it answers with that `run_id` and a fresh `request_id` that it
   finishes by following `strut.store.tailEvents` to `run.end`.
   `describe-code` needs none of this: it re-reads "nodes without a
   description" on every batch.
7. **`processing` and `cumulativeUsage` keep working.** Export
   `isRepoInFlight(repo?: string)` from the proxy (any repo when unscoped);
   `gitree_list_concepts` reports `processing: isProcessing || isRepoInFlight(repo)`.
   Usage is recorded by the workflow itself with a new step,
   `concepts/record-usage { owner, repo, usage: any[] }`, which sums its
   inputs the way `collect-results` does and calls
   `storage.addToTotalUsage(repoId, usage)` (the lab's `GraphStorage` already
   implements it). It runs twice: in `process-repo-chronological` after
   `summarize` with `["{{ collect.usage }}", "{{ summarize.usage }}"]`, and
   in `bootstrap-then-process` beside `bootstrap` (`depends: [check, bootstrap]`,
   `when: true`) with `["{{ bootstrap.usage }}"]`. That matches what
   `gitree_process` summed (bootstrap + process + summarize). Eval runs of
   these workflows record usage too, which is fine: it is what was spent on
   that repo.
8. **`/repo/embed`** is untouched (see "Today").

### The GitHub token never enters the run

A run's `input` is written to its log (`run.start`), and a step's resolved
config is written on `step.start` — including the `process` subflow's `input`,
which carries `token` today. Today's workflows thread `token` through both
(`bootstrap-then-process` → `clone-repo`, `process-repo-chronological`). And
the concept steps' Octokit is built **once at boot** from mcp's env
`GITHUB_TOKEN` (`services.ts`), not from the request's token. So private
repos fail on a swarm without that env var.

The fix uses strut's per-actor secrets (strut AGENTS.md "Actor secrets"):
- **Proxy:** before launching, store the request's `token` as the actor's
  `GITHUB_TOKEN`: in-process `strut.app.fetch(PUT /actors/:actor/secrets/GITHUB_TOKEN { value })`.
  The lab strut sets no `STRUT_API_KEY` (mcp only derives
  `STRUT_SECRET_KEY`), so the route takes no header in-process. A typed
  `actorSecrets` accessor on the `Strut` instance would be nicer but means a
  strut change and a pin bump; the fetch is enough.
- **Steps:** `clone-repo`, `fetch-changes`, `fetch-content` and
  `apply-decision` read `await ctx.services.secrets.get("GITHUB_TOKEN")`.
  Strut binds `secrets` to the run's principal once per run
  (`SecretsCapability.forPrincipal`), so this returns the actor's token
  first, then the deployment's, then env. They pass it to the services:
  - `services.github(token)` returns an Octokit for that token, built per
    run and cached by token;
  - `clone(owner, repo, token)`, `prContent` and `commitContent` take the
    token, or the Octokit.

  The boot-time Octokit goes away.
- **Workflows:** remove the `token` input from `bootstrap-then-process` and
  `process-repo-chronological`.
- **Persistence:** the lab's deployment secrets are a `FileSecretStore` under
  `STRUT_LAB_WORKSPACE`, so strut keeps actor secrets in `actor-secrets.json`
  beside them. A resumed run reads the token at step time and finds it.
- **Precedent:** hive already pushes `GITHUB_TOKEN` as an actor secret
  before every code-change and job dispatch (`services/strut-actor-secret.ts`,
  called from `strut-runs.ts`). The proxy writes the same name with the
  same PAT, so a later run by the same actor sees nothing it would not have
  seen already. Once no old mcp build remains, hive can push it itself with
  `ensureStrutActorSecrets` and drop the token from the query.

### Who the run is for

- **`x-strut-actor`** on the request names the actor (item 1 above).
- **With an actor:** run as that actor. The token goes into their secrets,
  and LLM calls made through strut's `llm` / `agent` steps are billed to
  their delegation.
- **Without an actor:** the old code, as today, for the SPA and for any
  caller that predates the header.
- **What `STRUT_MOTHERSHIP_REQUIRED` means today:** `describe-code`'s calls
  go through strut's `llm` step, so they need a principal with a
  delegation. The concept steps (`concepts/decide`, `concepts/summarize`,
  `concepts/bootstrap-explore`, `exploreConcept`) still call aieo directly
  with mcp's Anthropic key, as gitree does. They work without a delegation
  and bill nobody. Moving them onto the registry `llm` / `agent` steps is
  the follow-up that bills them (below).
- **The standalone Rust caller gets an actor from hive.** Hive already
  decides whose PAT a sync carries (the requesting user on an ingest, the
  workspace owner on the push webhooks). It sends that user's actor as
  `strut_actor` in the same `/sync_async` / `/ingest_async` body, after
  `ensureStrutDelegation` for them. Standalone adds
  `strut_actor: Option<String>` to `ProcessBody`, keeps it with the request,
  and `call_mcp_embed` sends it as `x-strut-actor`. Nothing else changes:
  standalone still calls only when `embeddings` names the repo, and
  incremental syncs still pass the modified files. What changes is who
  pays: today's per-push describe spends mcp's env key; after this it is a
  `describe-code` run on the owner's delegation, capped by the same
  `embeddings_limit`. A standalone build without the field sends no actor
  and lands on the old describe path.

### `/busy`

The proxies hold no `startTracking`. The only reader that acts on the value
is sphinx-swarm's `fast_service_update.rs`, which skips a container update
while mcp says it is busy (hive's `checkStakgraphAvailability` only checks
the endpoint answers). A workflow run does not need that protection: the
lab's runs are in a `FileRunStore`, so strut's boot-time auto-resume (up to
5 resumes within 7 days) restarts a cut-off run a few seconds after boot,
skipping every journaled step and foreach iteration. The concept steps
tolerate redoing one change (MERGE saves, a per-change checkpoint), and
`describe-code` re-reads nodes without a description. The old no-actor path
keeps its own tracking unchanged.

## Hive

One PR:
- `auto-learn.ts` `triggerAutoLearn` and `learnings/route.ts` send
  `x-strut-actor: await resolveStrutActor(userId)`, after
  `ensureStrutDelegation(...)` for that user. Use the same calls
  `strutTools.ts` and the benchmark routes make. The user is:
  - the **workspace owner** for auto-learn, whose PAT is already the one sent
    (the webhook path has `swarm.workspaceId`, `workspace.ownerId` and `slug`,
    which is the `WorkspaceAuth` the delegation call takes);
  - the **requesting user** for the learn button.
- `stakgraph-actions.ts` `triggerAsyncSync` / `triggerIngestAsync` take a
  `strutActor` and send it as `data.strut_actor`. The three callers
  (`ingest/route.ts`, both webhook routes) pass the actor of the user whose
  PAT they send, after `ensureStrutDelegation`.
- The URL, query and fire-and-forget stay as they are. The token still
  travels in the query: the proxy moves it into the actor's secrets, so it
  never reaches a run log.

## Order

Order-independent: a request with no actor, or to an mcp that ignores the
header, takes the old path.

1. **stakgraph (mcp):**
   - the proxies: actor → workflow, no actor → old handler;
   - the token through actor secrets in the concept steps and services;
   - `token` removed from the two workflows;
   - `concepts/record-usage` and its two uses; `processing` from the
     in-flight map; the dedup backed by strut's run list;
   - `describe-code.yaml` `params.model: haiku`.
2. **stakgraph (standalone):** `strut_actor` on `ProcessBody` →
   `x-strut-actor` on `call_mcp_embed`.
3. **hive:** the actor header on the two gitree calls, the `strut_actor`
   field on the sync bodies, delegations for both.

Nothing is deleted. `gitree_process`, `descriptions.ts`, `gitree/service.js`
and `GraphStorage` stay: the first two are the SPA's path, and
`repo/concepts.ts`, `repo/tools.ts` (`learn_concept`) and `handler/tools.ts`
still import the others. Moving those readers onto the graph, and the SPA
onto the workflows, are their own changes.

4. **Follow-up, billing:** move `concepts/decide`, `concepts/summarize` and
   the bootstrap / explore agents onto the registry `llm` / `agent` steps
   (as `describe-code` does). Their spend then goes through the Mothership
   to the actor, and the lab services lose `LLMClient` / `Summarizer`.

## Verification

- **Unit:** the proxies' input/output mapping; the actor rule (header beside
  `x-api-token` only; SPA JWT, Basic, dev mode → old handler); the per-repo
  de-duplication, including a seeded summary-less running run in the store;
  `record-usage` summing; `processing` reflecting the map; and the token
  reaching steps only through `secrets` (a test asserting it is absent from
  `run.start` and every `step.start`).
- **Live, throwaway Neo4j, `API_TOKEN` set:** `POST /gitree/process` with
  `x-strut-actor` and a PAT for a small repo.
  - **Expected:**
    - a `bootstrap-then-process` run in `/lab` with that actor;
    - Concepts with `node_key`, linked to `File`s, and embedded by the pass;
    - `PullRequest -[:MODIFIES]-> File` edges;
    - `/progress` finishing with the mapped result;
    - `GET /gitree/concepts?repo=` showing `processing: true` during the run
      and a grown `cumulativeUsage` after;
    - the PAT nowhere in `events.jsonl`, present in `actor-secrets.json`;
    - a second `POST` during the run answering with the same `run_id`;
    - kill mcp mid-run, restart: the run resumes, and a `POST` for the same
      repo attaches to it instead of launching another.
  - Then `POST /repo/describe` with the actor against the same repo:
    descriptions written, re-embedded on the next sweep; with no OpenRouter
    key in the env the run's `llm` calls name `haiku`.
  - The SPA's "generate concepts" and "Enrich" still work (old path).
- **swarm38, after deploy:**
  - a push triggers auto-learn and the run appears under the owner;
  - the learn button runs under the clicking user;
  - the same push's sync, for a repo with `embeddingsEnabled`, produces a
    `describe-code` run under the owner, with the modified files as
    `file_paths`.

## Open questions

None. Strut is not changed: the proxy stores the token with an in-process
`strut.app.fetch` of the same `PUT /actors/:actor/secrets/GITHUB_TOKEN` hive
already calls over HTTP.
