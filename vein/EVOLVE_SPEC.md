# Self-Evolving Workflows — Spec

What a workflow can change about itself, and how each kind of change gets
proposed, measured, and kept.

Builds on `EVAL_SPEC.md` (the **measurement** substrate: run target → collect
→ score → a number). This doc is the layer above it: **what varies** between
one measured run and the next. The guiding idea:

> A workflow's behavior is determined by three things: the **prompts** it
> runs, the **environment** those prompts run inside, and the **structure**
> that wires them together. Each is a candidate the system can improve, each
> evolves on its own cadence, and each promotes to a **versioned, pinned,
> reviewable artifact** — never a runtime mutation. The eval is the fitness
> function for all three.

---

## 1. The three layers

| Layer | Candidate | Promotion artifact | Cadence | Status |
| --- | --- | --- | --- | --- |
| **1. Prompt** | a `params` value | param default + new workflow version | minutes; fully autonomous | **built** (`eval/optimize`) |
| **2. Environment** | an `env.manifest` diff | rebuilt image | days; human-reviewed | **todo** (§4) |
| **3. Structure** | a workflow version + step sources | published version | hours; agent-interactive | **built** (§5 — `meta/*` + `services.authoring`; first authoring harness `harvey-evolve` §9.4; optimize generalization §5.3.3 todo) |

The layers are ordered by how cheap they are to try and how safely they can
be automated. Prompt tuning is mechanical and reversible. Environment changes
outlive the run that motivated them and touch every other experiment.
Structural changes author *code*.

### Why all three, empirically

The first GAIA batch (level 1, limit 5) scored **1/5**. The second, after
changes across two layers, scored **5/5** — and the per-question attribution
separates the layers cleanly:

| Task | Run 1 → Run 2 | Layer responsible |
| --- | --- | --- |
| Kipchoge (`17000` → `17`) | ❌ → ✅, cheaper | **1** — unit re-read rule |
| Sosa (`2` → `3`) | ❌ → ✅, cheaper | **1** — persistence rule (404'd both runs; recovered only in run 2) |
| Fish-bag PDF | ✅ → ✅, 19 → 3 steps, $1.33 → $0.74 | **2** — `pdftotext`/`pdfplumber` existed |
| Bird video (`2` → `3`) | ❌ → ✅, 2 → 50 steps | **2** — `yt-dlp`/`ffmpeg` existed |
| Ping-pong (`55` → `3`) | ❌ → ✅ | unattributed — likely variance |

Two observations that motivate this whole doc:

- **The environment was the largest single lever**, and its two wins have
  *opposite* cost signatures: the PDF task got 5× cheaper (stopped fighting
  its tools), the video task got 40× more expensive (could finally attempt
  the work instead of guessing). A layer that only prompt-tuning can reach
  would have found neither.
- **A layer-3 fix was load-bearing and nobody asked for it.** The batch
  initially crashed: one task exhausting `maxSteps: 30` threw
  `AI_NoObjectGeneratedError` and killed the whole `foreach`. The assistant
  raised the cap to 50 and added an `onError` fallback. The bird task then
  used *exactly* 50 steps — at the old cap, 4/5.

---

## 2. The shared loop

Every layer runs the same four beats. Only the artifact differs.

```
   capture ──► propose ──► evaluate ──► promote
      │                        │            │
   signal from            EVAL_SPEC       a pinned,
   a real run             (a number)      versioned thing
```

- **Capture** — turn an observed failure into structured data. This is the
  beat most often skipped, and skipping it is why a gap stays an anecdote in
  a chat transcript instead of becoming an input.
- **Propose** — an LLM (or plain code) reads the aggregate and emits a
  candidate. Must see the **aggregate across the dataset**, never one
  example, or it overfits (EVAL_SPEC §4/§7 — the wrong-insight problem).
- **Evaluate** — run the candidate through the harness. One number, plus the
  per-example detail the next proposal needs.
- **Promote** — write the winner somewhere durable and diffable. Never leave
  an improvement living only in a run's memory.

**Promotion is always to a reviewable artifact.** A param default in
published YAML, a manifest line consumed at image build, a new workflow
version. Nothing self-modifies in place at runtime — that is the invariant
that keeps a score attributable to a configuration (§6).

---

## 3. Layer 1 — prompts

**Built.** The candidate is a `params` value; `paramOverrides` (`runner.ts`)
reaches a knob at any nesting depth by workflow name, so a generation can
sweep a prompt buried two subflows down without republishing anything.

- `eval/score` — match produced ↔ expected by a `rubric`, F-beta recall-weighted.
- `eval/reflect` — propose the next candidate from the aggregate, anchored to
  the best-so-far and handed the full trajectory of prior generations.
- `eval/optimize` — the loop, as one detached run.

**What this session added to the picture:** the two winning rules were both
*meta-instructions about method*, not domain knowledge —

1. re-read the question before finalizing and check units/phrasing;
2. if an approach fails, try a genuinely different one before answering.

Both bought a question, and both made the run **cheaper**. That's worth
noting because it cuts against the assumption that accuracy is bought with
effort: these were failures of care, not of budget.

**Known limitation.** `eval/optimize` tunes a single `promptParam` string and
tracks `bestPrompt`. Layer 3 needs the candidate generalized to a workflow
ref + version (§5).

---

## 4. Layer 2 — environment

**Todo.** The tools available to `bash` inside an agent step: interpreters,
CLIs, libraries. Today this is `mcp/Dockerfile` — edited by hand, discovered
by an agent failing at something.

### 4.1 Why not just let the agent install what it needs

It technically could: the container runs as root, `bash` allows up to a
10-minute timeout, and `/usr/src/agent-venv/bin` is first on `PATH`, so
`pip install` would land in the right place. It is disallowed inside graded
runs for three reasons:

1. **Attribution.** A score where one run had pandas 2.1 and the next had 2.3
   isn't comparable to itself. This is the same discipline the graders
   already enforce with `benchmarkRev` / `scorerSha256` / clean-tree checks
   (§6); a self-mutating environment silently breaks it.
2. **It moves the tax, it doesn't remove it.** The PDF task's 19 steps of
   *fighting* tooling would have become 19 steps of *installing* tooling —
   still inside the run, still on the cost line, still against a step budget
   meant for research. And the container filesystem is ephemeral in prod, so
   every run re-pays it forever. Nothing accumulates.
3. **Injection surface.** Benchmark tasks send agents to fetch arbitrary web
   pages. An agent that installs packages it decided it needs, right after
   reading attacker-controlled text, is a short path from prompt injection to
   arbitrary code execution.

**Authoring ≠ measuring.** The chat *builder* may install freely — that's
exploration, and it already can. The obligation is that a discovery there
becomes a manifest proposal, not an assumption that the install persists.

### 4.2 The mechanism

**Capture — `meta/search-runs` + `env/missing-tools`.** Every bash call
already emits a nested run event (`wrapToolsWithEmit`, `agent.ts`), so the
evidence is on disk; the question is how it gets read. The general
instrument is **`meta/search-runs`** (done — `searchRunEvents`,
`authoring.ts`): grep a regex across a workflow's recent run event logs →
matching (runId, event path, snippet) tuples plus a per-run frequency
summary, behind the same agent-authored gate as `meta/get-run` (raw
filesystem grep over the workspace would read grader logs — §6). That's
what lets a propose agent hunt *flexibly* — signatures nobody enumerated,
a tool that exists but is too old — instead of being limited to a fixed
parser. One caveat: events hold ~1500-char output previews
(`summarizeForEvent`), not full transcripts, so a signature buried deep in
long output can be missed. On top of it, a deterministic `env/missing-tools`
scan for the known signatures (`command not found: yt-dlp`,
`ModuleNotFoundError: No module named 'fitz'`) is the free always-on
tripwire: zero tokens per batch, and its misses just mean a gap surfaces a
batch later through the agent's own search.

**Propose.** Aggregate across a batch → a frequency-ranked list
(`pdftotext: 12 runs, yt-dlp: 3, tesseract: 2`). This is `eval/reflect`
pointed at the environment, with `meta/search-runs` as its instrument.

**Promote — `env.manifest`.** A versioned file (apt packages + pip packages,
**pinned**) that the Dockerfile installs from at build time. The agent
proposes a diff; a human reviews two lines instead of auditing a Dockerfile;
CI rebuilds. The manifest becomes the same class of object as a workflow
version: agent-authored, human-reviewed, diffable, pinned — and the mutation
happens at **build** time, where it can be attributed.

### 4.3 Credentials — `secretsEnv` (built)

How an in-workflow agent uses an authenticated API without ever seeing a
credential. The `agent` step takes `secretsEnv: [<secret name>, …]`; the
step resolves the names via `ctx.services.secrets` **in code** and injects
the values into the bash tool's subprocess env only. The model writes
`curl -H "Authorization: Token $COURTLISTENER_API_KEY" …`; the shell
expands it at exec time. Two guarantees and one accepted residual:

- **Values never enter context or logs.** The prompt and event log carry
  `$NAME` literally, and every tool output is masked
  (`wrapToolsWithMask`, applied inside the emit wrapper) before the model
  or `events.jsonl` sees it — covering `echo $KEY`, `env`, curl errors
  echoing the URL, and files the shell wrote that another tool later reads.
- **Grant discipline.** `secretsEnv` goes on a dedicated, narrow research
  sub-agent (an `agent` step whose whole job is e.g. case-law lookup) —
  never on a drafting/producing agent, never on the meta/* author.
- **Accepted residual: egress.** A prompt-injected agent can still *send*
  `$KEY` somewhere — masking stops leakage into context/logs, not
  exfiltration by the shell itself. This is the trade for bash-native
  exploration (agents iterate curl+jq far faster than any structured http
  tool); bound it by keeping the sub-agent's inputs narrow (legal APIs,
  not arbitrary pages) and its grant list minimal.

Why bash-env beats a per-API wrapper step: the first live evolve run showed
the author routing around the value firewall by authoring brittle TS
wrappers (the only place values could be injected), one of which shipped
hardcoded "fallback data" — the exact staleness it was built to fix. With
`secretsEnv`, a research capability is one `agent` step whose entire
content is a *prompt* (tunable by layer 1 forever after), and the author
can iterate it live via `meta/run-step type=agent`.

### 4.4 Current baseline (what `mcp/Dockerfile` provides)

`/usr/src/agent-venv` first on `PATH` — numpy, pandas, openpyxl, pypdf,
pdfplumber, pillow, requests, beautifulsoup4, yt-dlp — plus `pdftotext`
(poppler), `ffmpeg`, `pandoc`, `gh`, `ripgrep`, `git`. Deliberately a
separate venv from the system python, which carries the pinned
`docx-mcp` / `mcp<2.0.0` resolution.

---

## 5. Layer 3 — structure

**Partial.** The candidate is the *shape* of the pipeline: new steps, new
tools, new data sources, different wiring, different budgets. EVAL_SPEC §4
already frames a structural variant as "just a new workflow version," which
versioning captures and the eval scores. What's missing is the ability for a
workflow to author one **from inside a run**.

### 5.1 The gap (now closed)

`agentTools` grants **registry step types** as LLM tools. The authoring
tools — `create_step`, `edit_step`, `create_workflow`, `edit_workflow`,
`run_workflow` — were bespoke AI SDK tools in `vein/src/ai/tools.ts`, bound to
the workspace. They are not registry steps, so an in-workflow agent could not
reach them. That's why EVAL_SPEC §7 says structural evolution "stays
agent-interactive for now." §5.2 closes this.

### 5.2 The fix: `meta/*` steps — **built**

Implemented in vein core, benchmark-agnostic: `vein/src/authoring.ts` is the
shared authoring core (the chat tools now sit on the same helpers), the steps
live in `vein/src/steps/lib/meta/`, and `createVein` auto-provides
`services.authoring` the way it already provides `http`/`secrets`/
`artifacts` — so every vein deployment gets the surface, not just the lab.

The roster mirrors the chat assistant's real authoring surface
(`vein/src/ai/tools.ts`). The loop the assistant actually runs is
create → test → edit → test; a roster missing the edit/test half strands an
in-workflow author on iteration one:

| Meta step | Wraps | Why |
| --- | --- | --- |
| `meta/list-steps`, `meta/search-steps`, `meta/get-step` | step discovery | read before authoring |
| `meta/create-step`, `meta/edit-step` | `workspace.publishStep` | `create_step` refuses existing names — without edit, every iteration pollutes the registry (`my-fetcher-2`, `-3`, …) |
| `meta/run-step` | `runSingleStep` + cassettes | the inner loop: test ONE step, offline via record/replay, without paying for a full candidate run |
| `meta/list-workflows`, `meta/get-workflow` | workflow discovery | read current source before editing |
| `meta/publish-workflow` | create/edit as an explicit upsert | the candidate artifact |
| `meta/run-workflow` | `runWorkflow` | test the candidate end to end |
| `meta/list-runs`, `meta/get-run` | run-store reads | debug a failed candidate — **provenance-scoped, see §6** |
| `meta/list-secrets` | secret NAMES only | author steps that reference auth by name |

Deliberately absent: `bash` and `web_search` — the agent step grants those
itself, and §5.3.2 forbids `meta/*` + `bash` on one agent anyway — and
`set_workflow_category` (UI cosmetics; and keeping it out means an in-run
author cannot re-file a workflow into the candidate namespace, §6).

The chat tools carry real logic worth not duplicating: name-conflict checks,
JSON-string arg coercion (`coerceJsonArg`), and the load-verify-and-report
behavior §5.3.4 demands. Both surfaces sit on the one shared core
(`authoring.ts`); the capability adds the meta POLICY on top — everything it
publishes is stamped `publisher: "ai"`, and its publish, run, and
run-history operations are **closed over that stamped set** (§6).

Then `agentTools: ["meta/*"]` closes the loop:

```
author  (agent, agentTools: ["meta/*"])   → candidate workflow name
run     (meta/run-workflow, per dataset example)
grade   (harvey/evaluate | gaia/evaluate | eval/score)
reflect (eval/reflect)                    → next candidate
```

### 5.3 Five things that will bite

1. **The registry is a per-RUN snapshot, not just per-step.**
   `buildRegistryTools` resolves `agentTools` once when the agent step
   begins, so a step authored mid-turn is not in that agent's own toolset —
   but it's worse than that: `runWorkflow` threads ONE registry through the
   entire run, so a step published by `meta/create-step` at step N is
   invisible to step N+1 too. Splitting authoring and execution into
   separate steps is necessary but not sufficient: `meta/run-workflow` and
   `meta/run-step` must build a fresh registry from the workspace at call
   time (`services.authoring.getRegistry()`), never use the run's
   `ctx.registry`. The chat tools already do the equivalent —
   `deps.registry = await deps.getRegistry()` after every publish.
2. **The authoring agent must never be the producing agent.** An agent with
   `meta/*` *and* `bash` can author a step that reads a rubric or shells into
   a benchmark checkout. Separate runs, disjoint grants (§6).
3. **`eval/optimize` optimizes a string.** Generalizing the candidate from
   "prompt" to "workflow ref + version" is a real change to that step's
   shape, not a new step beside it.
4. **Broken generated steps fail silently.** `loadStepFile` catches, warns,
   and returns `null` — a step that doesn't compile simply doesn't exist. An
   authoring agent will flail against that; `meta/create-step` must
   typecheck-and-import before returning and hand the error back.
5. **The template evaluator does not short-circuit.** `expr.ts` is a direct
   evaluator: ternary and `&&`/`||` still evaluate BOTH sides, so
   `{{ run.output ? run.output.outputDir : undefined }}` throws
   ("Cannot access property of undefined") exactly when the guard was
   needed. Harness YAML must never deep-access a possibly-undefined object:
   pass the WHOLE object into a step and unpack in code
   (`harvey/evaluate`'s `fromRun`), or keep error-path packs shape-stable
   (`usage: {}` in every `onError` fallback so `x.usage.inputTokens` stays
   safe). This bites generated workflows too — candidates should copy the
   base workflow's patterns, not invent guards.

---

## 6. Fixed points — what must never evolve

The counterpart to all of the above. These stay outside the agent-editable
surface, permanently:

- **Graders and gold.** `harvey/service.ts` and `gaia/service.ts` live in code
  on the services bag, never as seeded steps. Seeded `harvey/*` / `gaia/*`
  steps are thin plumbing; editing them can only break plumbing, never change
  how grading works.
- **Rubrics and answers, at read time.** `harvey/get-task` strips `criteria`;
  `gaia/getTask` strips `Final answer`. A producing agent that can read its
  own rubric trains against it.
- **Grader grants.** `harvey/evaluate` / `gaia/evaluate` are harness-only,
  never in a producing agent's `agentTools`.
- **The meta surface is closed over its own namespace** (*built* —
  `authoring.ts`). A harness workflow's run log records what its grader
  steps were handed — potentially gold and rubric text. So the capability
  scopes by **workflow provenance**, not by actor: `meta/publish-workflow`
  stamps everything it publishes `publisher: "ai"` (the capability sets the
  stamp; the calling agent cannot override it), and `meta/run-workflow`,
  `meta/list-runs`, and `meta/get-run` act on stamped workflows ONLY,
  failing closed on everything else. Publishing over an existing unstamped
  workflow is refused outright (so the loop also can't rewire the harness
  that grades it), an identical-content republish never re-stamps (no
  claiming a workflow by republishing its own YAML verbatim), and
  `meta/edit-step` edits only steps published as `"ai"` (seeded harness
  steps carry seeder publishers). Actor scoping ("runs this agent
  started") would have broken iteration — generation N must inspect
  generation N−1's candidate runs, which a different run started; the
  stamp lives on the workflow record, so it survives sessions. This is
  defense in depth, not the sole barrier: graders must also resolve gold
  internally (by task id, from the services bag) and emit only verdicts,
  so that NO run's event log carries answers even where scoping is
  misconfigured — including a laundered candidate that embeds a grader
  step directly. The residual channel — hill-climbing answers against
  pass/fail verdicts as an oracle — is the train-set problem, and §7's
  train/val split is what answers it, not scoping.
- **Benchmark provenance.** Clean-tree checks, `benchmarkRev`,
  `scorerSha256`. A dirty checkout refuses to grade.
- **Infra constants.** Values a workflow author has no basis for choosing and
  where a wrong value is only ever a bug — e.g. `maxOutputTokensFor(provider)`
  (`pricing.ts`). These are derived, not configured, and never exposed as
  step config for a model to pick.

---

## 7. Autonomy and measurement discipline

**Autonomy ladder.** Layer 1 is fully autonomous (a background optimize job
sweeps params for hours). Layer 2 is agent-proposed, human-approved — the
blast radius extends past the run that motivated it. Layer 3 is
agent-interactive today; a headless authoring job is possible once §5.2
lands, but it authors code and deserves review before it is trusted unattended.

**Cost is a constraint, not telemetry.** A structural evolver will reliably
discover that adding agents raises scores. `eval/optimize` already sums
`totalUsage` / `totalCost` per generation — put it in the fitness function or
the loop just buys accuracy.

**A tuned-on set is a train set.** The five GAIA tasks that went 1/5 → 5/5
were the tasks the prompt rules were written against. 5/5 there is the
expected outcome, not evidence of 100% on level 1. Every layer needs a
train/val split before its number means anything, and n=5 is an anecdote:
one flip is ±20 points.

**No silent caps.** If a harness bounds coverage (top-N, no retry, sampling),
log what was dropped. Silent truncation reads as "covered everything."

---

## 8. How the layers interact

They are not independent, and the coupling is the interesting part. The
observed chain from this session:

```
layer 1: "try a different approach before answering"
   └─► agents persist instead of guessing
        └─► runs get longer, hit maxSteps: 30
             └─► forced final turn throws, foreach dies
                  └─► layer 3 fix: maxSteps 50 + onError fallback
```

**A layer-1 change created a layer-3 requirement.** The bird task then used
exactly 50 of 50 steps — the fix was load-bearing for the result. Expect
this: prompt changes shift resource envelopes, environment changes shift what
prompts are worth writing, structural changes change what the environment
needs. An optimizer that can only reach one layer will hill-climb into
another layer's wall and stop, with no signal saying why.

Practical consequence: the per-miss taxonomy — **formatting / persistence /
tooling / capability** — should be a *required output* of every batch run.
It routes each failure to the layer that owns it, and it is the cheapest
possible version of "capture."

---

## 9. What's next

1. **Environment capture** (§4.2) — ~~`meta/search-runs`~~ **done**
   (cross-run grep over event logs, gated like `meta/get-run`; also a
   `search_runs` chat tool). Next: the deterministic `env/missing-tools`
   tripwire on top.
2. **Full level-1 GAIA sweep** (53 tasks, ~$25) — a real baseline on 48
   unseen tasks, and the first dataset big enough for layer-1 optimize to
   have honest signal.
3. **`env.manifest` + Dockerfile consumption** (§4.2) — makes layer 2 a
   proposable artifact.
4. ~~**`meta/*` steps + `services.authoring`** (§5.2) — closes layer 3.~~
   **Done** — `authoring.ts` + `steps/lib/meta/*`, auto-wired by
   `createVein`. ~~Next is a first authoring-harness workflow that uses
   them.~~ **Done** — `harvey-evolve` in the lab
   (`mcp/src/lab/harvey/workflows/`), built on the Harvey produce/grade
   split: `harvey-produce` is the swappable candidate unit (per-task
   `workdir` for batch isolation), `harvey-run` grades any produce workflow
   by ref (`input.produceWorkflow`), and `harvey-candidate-run` runs an
   ai-stamped candidate under its own runId via `meta/run-workflow` (fresh
   registry — §5.3.1) and grades the `outputDir` it reports. The evolve run
   is one full generation: baseline over the task set →
   `harvey/digest-results` (verdict-channel digest, §6) → authoring agent
   (`agentTools: ["meta/*"]`, no bash — §5.3.2) publishes + smoke-tests a
   candidate → harness re-runs it pinned over the same tasks → report with
   baseline/candidate deltas and any `missingSecrets` the author captured
   (the credential-request beat). Scores are TRAIN scores (§7); promotion
   stays human. Offline checks: `mcp/src/lab/harvey/evolve-smoke.ts`.
5. **Generalize `eval/optimize`'s candidate** from prompt string to workflow
   ref + version (§5.3.3) — the change that lets one loop drive all three
   layers. **Harvey instance built** — `harvey/evolve-loop` (lab): up to N
   generations of `harvey-evolve-gen` (author → run pinned candidate →
   digest), each briefed with every prior attempt's version/pass-rate/
   approach/failures, anchored to the best-so-far (never the latest), with
   the directive flipping from exploit to "try a GENUINELY DIFFERENT
   approach" after `exploreAfter` non-improving attempts. Fitness is
   criteria pass-rate (binary all-pass has no gradient); improvements must
   clear a judge-noise margin (default 0.02 ≈ one criterion at n=50). The
   GENERIC step remains open — this is the shape it should generalize.
