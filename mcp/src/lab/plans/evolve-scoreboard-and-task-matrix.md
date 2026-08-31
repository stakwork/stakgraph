# evolve: version scoreboard + task matrix

Direction doc for the next phase of the evolve harness (`eval/evolve-loop` and
its gaia/harvey instances), written from the post-mortem of prod run
`gaia-evolve/1788061734710` (2026-08-30, swarm38 — 25 tasks, 8 generations,
17h, $214). Relation to what's in flight: PR #1621 (no-op generation gate,
re-score guard, budget caps) is the tactical layer and should land as-is —
this doc is the restructure that makes that class of guard unnecessary, and
it deliberately does NOT propose more guards.

**The thesis: every failure in that run is one disease — the harness trusts
narration where it already owns ground truth.** It trusted the author's echoed
version string (the version registry knew the truth), the author's approach
summary (the YAML diff between versions is computable and cannot lie), a
single accuracy number (the per-task correctness vector it computed and threw
away knew better), and the run to finish before writing anything down (the
event log had everything; the post-mortem reconstructed $214 of results from
it after the process died mid-generation-8). Each patch so far — the summary
filler-guard (#1620), the no-op gate (#1621) — hardens one narration channel.
The simplification is to stop asking agents to report what the harness can
observe.

---

## The evidence (run 1788061734710, compressed)

- Reported: baseline 0.68 → best 0.80 at gen 5. **Gen 5 published nothing** —
  its author degenerated into a filler loop, the `vpin || vactive` fallback
  resolved to gen 4's v11, and the harness re-graded identical YAML 0.04
  higher. Gen 7 repeated the accident with `version: "pending"` on v12.
  Together: $68.56 (33% of budget) measuring nothing new. (#1621's gate now
  stops the re-grade.)
- The accidents were a free noise measurement: **v11 scored 0.76 and 0.80;
  v12 scored 0.72 and 0.76.** Identical bytes, ±0.04 — with
  `improveMargin: 0`, so every 1-task delta (1/25 = 0.04) registered as
  signal. Every architectural verdict in the run rests on 1–3 task
  differences at or under that floor.
- Nine full measurements split the 25 tasks into bands that never mixed:
  **12 correct every time** (paid for every generation, never at risk),
  **5 wrong every time** (unreachable by any prompt/structure change tried),
  **8 that flip on re-runs of identical YAML** — the entire dynamic range.
  Gen 5's 0.80 was a clean sweep of all eight coin-flips: the maximum
  reachable score, drawn by luck, recorded as the best architecture.
- Two of the five never-solved tasks returned **byte-identical wrong answers
  in all nine measurements** across four architectures (`"36"`, `"193"`).
  The flagship dual-attempt + reconciler architecture (~3× produce cost) is a
  *variance* reducer and structurally cannot touch a *bias* failure — both
  attempts agree on the wrong answer, the reconciler rubber-stamps. Nothing
  in the digest could reveal the repetition: it shows one generation's answer
  at a time.
- The baseline's miss on the Freon task was **`webbook.nist.gov` returning
  HTTP 400**; the agent fell back to a textbook approximation. The digest
  reported `WRONG-ANSWER: "193"`, the authors reasoned about unit conversion
  for six generations. Tool-level `step.error` events never reach
  `digest-results`, and candidate runs scatter theirs across ~225 child
  event logs.
- 225 produce runs strictly serial (`foreach` has no concurrency), 17h for 8
  of 10 generations, process died mid-gen-8, **no summary written** —
  `/runs/:id` 404s; results exist only as events.

---

## The shape: measurements are version-keyed samples over a task matrix

Today the unit of record is *"generation N scored X"* — a chain of prose
entries, each one number deep, each trusted once. Replace it with two owned
structures:

**The scoreboard** — measurements attach to *versions*, not generations:

```
scoreboard = {
  candidate: "gaia-produce-ai",
  tasks: [taskId, …],                      // the train set, fixed per run
  versions: {
    "v11": {
      measurements: [                       // each a FULL per-task vector
        { vector: {taskId → {correct, answer, missTag, cost, toolErrors[]}},
          fitness, runIds, at },
      ],
      meanFitness, n,
    }, …
  },
}
```

A generation becomes *an attempt to mint a new version*. It has no score of
its own — only the version it minted (or didn't) does. This dissolves the
worst bug instead of patching it: a generation that publishes nothing has no
version, so there is nothing to attribute a fitness to — the stale-version
accident becomes **unrepresentable**, not guarded-against. And where #1621
discards a resample of an already-scored version, the scoreboard *banks it*:
re-measuring the incumbent is exactly what a noise-aware loop should spend on.

**The matrix** — the per-task view across all measurements of all versions
(what the post-mortem built by hand from event logs). Everything intelligent
falls out of it as queries, not features:

- **band(task)**: floor (correct in every measurement) / movable / ceiling
  (never correct). Drives budget: floor tasks run once per new version as a
  regression check; movable tasks get the repeats (they carry all the
  signal); ceiling tasks leave the fitness denominator and get routed to a
  debug directive.
- **flipRate(task)** over same-version re-measurements: the *empirical* noise
  floor. Replaces `improveMargin` as a hand-set param.
- **distinctAnswers(task)**: 1 distinct wrong answer × 9 measurements = bias
  (structural; needs a different data path or method, provably immune to
  prompt evolution). Many distinct wrong answers = variance (redundancy and
  reconciliation help). The run spent six generations unable to make this
  distinction.
- **toolErrors(task)**: aggregated from the produce run's tool-level
  `step.error` events into the vector at measurement time — `NIST 400 ×2`
  is the single most actionable string the harness can hand an author, and
  today it is the one thing it hides.

Gold discipline (§6) is unchanged: the matrix carries verdicts, the
candidate's own answers, and tool errors — never the gold.

---

## Phase 1 — measurement first

Mirrors the gitsee plan's ordering argument: until measurement is sharp,
every other change optimizes against noise, and you can't even *verify* the
later phases helped.

1. **`foreach` concurrency** (vein `runner.ts`): a `concurrency` config on
   the step, bounded worker pool. Iteration paths are already `#i`-keyed and
   journal-replayed individually, so durable resume is unaffected; candidate
   runs already execute under their own runIds with their own artifacts dirs
   (`meta/run-workflow` (c)), so parallel tasks don't collide. Default it
   low (4–8): the observed constraint is target-site rate limits
   (metmuseum 429s), not CPU. This turns a 17h run into ~2.5h and makes
   repeated measurement affordable in wall-clock.
2. **Repeated measurement**: `gaia-evolve` grows `params.baselineSamples`
   (default 2–3) and measures every *new* version at n≥2 on the movable
   band before comparison (floor tasks 1×, ceiling tasks 0× — matrix-driven,
   which is why the matrix comes first even here). Money note: a repeat
   restricted to the movable band costs ~⅓ of a full sweep.
3. **`eval/matrix` step** (or grow `digest-results`): fold each measurement's
   graded results + the produce runs' tool-error events into the scoreboard
   shape above; emit both the object and a compact `text` rendering (bands,
   flip rates, identical-answer flags) for briefings.
4. **Incremental persistence**: write the run summary after every top-level
   step / generation, not at `run.end`. A dead process should cost one
   generation, not the report. (vein-level fix; the `liveStatus` fallback on
   the list endpoint already proves the store can serve partials.)

## Phase 2 — the scoreboard loop

Rewrite `eval/evolve-loop` around the scoreboard (it keeps its name and its
`services.optimizer` seam; the gen workflows keep their author → eval →
digest shape):

- **Registry diff is the only truth channel.** Snapshot the candidate's
  active version before the author runs (#1621's `vbefore` — keep it);
  afterwards, the version that gets measured is *the registry's* new active
  version, full stop. The author's echoed `version` field becomes advisory
  color. Deletes `vpin`, `vactive`, the `||` fallback, and the garbage-echo
  failure class. No new version ⇒ the attempt failed ⇒ retry/stop; never
  grade.
- **Promotion is a paired comparison, not a threshold.** Challenger vs
  incumbent on their shared task vectors: promote when net task wins on the
  movable band exceed what same-version re-measurements flip (the empirical
  floor from the matrix). Delete `improveMargin` and `stopFitness`-on-one-
  sample; a "no verdict" outcome is legal and correct — on this run's task
  set, most generations honestly were one.
- **Cost enters the fitness** (§7: "cost is a constraint, not telemetry" —
  the spec predicted exactly what happened: authors reliably evolve toward
  expensive shapes; v10+ tripled per-task produce cost for gains at the
  noise floor). Minimum viable: promotion requires fitness up AND
  cost/task ≤ incumbent × k (k ≈ 1.5), with the cap stated in the briefing
  so authors design under it.
- Keep #1621's `maxCost`/`maxMinutes` gates and the explore/exploit
  directive, but drive `sinceImprove` from paired verdicts, not raw deltas.

## Phase 3 — briefings from ground truth

- **Show diffs, not summaries.** The briefing renders the computed YAML diff
  between each version and its parent, alongside its scoreboard row. The
  author's `summary` demotes to optional color (and the #1620 filler-guard
  becomes dead code — a diff cannot be `"placeholder"`).
- **Route by failure mode.** Matrix-tagged *bias* tasks (identical wrong
  answer, n≥3) get a `debug` directive: one task, root-cause it, tool-error
  evidence attached — the mode gen 6 proved works when it binary-searched
  the Caesar-cipher zero-token failure. *Variance* tasks get the statistical
  treatment and an explicit "do not chase" marking below the noise floor.
  Ceiling tasks stop costing produce budget in the main loop.
- **Answer history in the briefing.** Nine identical `"36"`s is a different
  diagnosis from nine different wrong answers; the matrix has it, the author
  should see it.

## Phase 4 — dataset honesty

- **Grow the train set.** 8 movable tasks cannot support fine distinctions —
  one flip is 0.04 and the whole dynamic range is 0.32. At ~60–80 L2/L3
  tasks the flip quantum drops to ~0.013 and the movable band widens to
  where paired stats have power. (§7's "n=5 is an anecdote" applies at 25
  too, just more quietly.)
- **Automate the holdout beat.** `gaia-evolve` takes `input.holdoutTasks`;
  the final report runs the promoted version once over the holdout via
  `gaia-candidate-run` and prints train vs holdout side by side. Today the
  train-set caveat is a `note:` string asking a human to do this; nobody
  did, and the reported 0.80 shipped without it.

---

## What gets deleted (the receipts)

| Deleted | Replaced by |
| --- | --- |
| `vpin` / `vactive` steps + `\|\|` fallback | registry diff (measure what the registry says is new) |
| summary filler-guard (#1620's `usableSummary`) | briefings render computed YAML diffs; summary is color |
| `improveMargin` param | empirical flip-rate floor from same-version re-measurements |
| `stopFitness` on a single sample | paired verdict on n≥2 |
| most of `composeBriefing`'s prose assembly | matrix `text` rendering + diffs |
| the no-op guard's *discard* of resamples (#1621) | resamples are banked as incumbent measurements |

The system gets smaller and stops lying to itself. Sequence matters:
Phase 1 before 2 (paired promotion needs repeats and the matrix), 2 before 3
(briefings render scoreboard state), 4 whenever budget allows — but nothing
after Phase 1 should be evaluated except through Phase 1's own measurements.
