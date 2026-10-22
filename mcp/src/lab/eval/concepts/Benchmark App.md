---
description: An app over a BENCHMARK, a set of tasks with known answers — the steps and workflows it needs underneath (catalogue, task, run, scorer, improve, climb), where the answers stay, and how the app drives them.
parent: App
---
An app over a BENCHMARK, a set of tasks with known answers — the steps and workflows it needs underneath (catalogue, task, run, scorer, improve, climb), where the answers stay, and how the app drives them.

A benchmark measures a workflow against a set of tasks whose answers are
known. Every benchmark is different (its tasks, its material, how an
answer is scored), but the app sits on the same few pieces, and each piece
is a step or a workflow. Build the pieces first, then the app over them
(`App` says how to build any app). Nothing new is stored: tasks come from a
step, attempts are runs, scores are run outputs, and what the climb learns
is graph data.

## The pieces

| Piece | What it is | Its shape |
| --- | --- | --- |
| Catalogue | The tasks, filterable (a split, a type) | A deterministic step: `{ tasks: [{ id, title, … }] }`, fast, no model. Wrap it in a one-step workflow for a host that lists by workflow |
| Task | One task as the producer may see it | A step `{ id }` that returns the inputs and NEVER the answer, and puts the task's files in the run's directory |
| Run | One attempt at one task, scored | A workflow `{ taskId, … }`: load the task → prepare → produce → score → pack. One run, one score |
| Scorer | Compares what was produced with the answer | A step. It reads the answer itself, by task id; nothing hands it in |
| Batch | Many tasks | `foreach` over the catalogue's ids, each a `subflow` of the run, then an aggregate |
| Improve | What the misses teach | A workflow over finished runs that changes what the producer READS |
| Climb | Run, improve, run again | One workflow whose steps are subflows; it stops at the target, at the budget, or when an improve proposed nothing |

Build them in that order. Each one is useful before the next exists, and a
host can list tasks and launch runs as soon as the first three are
published.

## The app: reads are steps, writes are workflows

The reads are steps the app may call on a render, the writes are workflows
it launches on a click, and runs are the history it reads. It needs no
table of results of its own.

| The person | The piece | In the app |
| --- | --- | --- |
| Browses tasks | Catalogue | `useStep` on the catalogue step |
| Opens a task | Task | `useStep` on the task step |
| Runs one | Run | `useLaunch(run)` with the task id as the `key`: one run in flight per task |
| Watches it, reads it | Runs | `useRuns` / `useRun` with `live`; `RunTimeline`, `RunGraph`, `Transcript`, `ArtifactView` |
| Sees the score over time | Runs | a chart over `useRuns`' outputs |
| Improves, climbs | Improve, Climb | `useLaunch`, keyed |
| Disputes an answer | A contest workflow | `useLaunch`; the contests read back with the graph steps |
| Stops one | Cancel | `useCancel` |

`grants.workflows` lists the run, improve, climb and contest workflows;
`grants.steps` the catalogue, the task step and the graph reads. Never the
scorer.

## What a run returns

The run's output leads with `score` (a number from 0 to 1, the headline),
`metric` (its name), `officialScore` (when the official number differs
from the one you climb on), `summary` (markdown) and `artifacts` (the
run's files). Then the per-item detail: what matched, what was missed,
what was extra. Numbers are numbers, never `"32/36"`. The app charts these
fields, and a host that lists benchmarks without an app draws the same
ones.

## The answers stay out of the producer's reach

- The task step strips whatever the scorer checks against. A producing
  agent never has the scorer, or the step that reads answers, in its
  `agentTools`.
- The scorer reads the answer by task id and returns its verdict: the
  score and the per-item detail. The answer text is never in a step's
  input, so no run log hands it to whoever reads the run next.
- A missed item names part of an answer. That is the point on the tasks
  you climb on, and it is why a held-out split never reaches an improve.
- Once the scorer is settled, a PERSON may seal it (`sealed: true` on the
  step). The builder cannot publish a sealed step, or a workflow that
  reaches one.

## Producing: agents in series, knowledge in the graph

- **Prepare once, read many times.** When a task comes with material
  (documents, sections, records), split it and give each piece its own
  agent in a `foreach`. Each one writes what it found into the graph, in a
  namespace of the task's own; later stages read the graph instead of the
  raw material. This is usually where the money goes, so look each piece
  up before launching its agent and skip what is already there.
- **Every agent that writes says where, first.** The namespace goes at the
  top of its prompt: an agent's first write follows its first lines.
- **Stages in series, each narrow.** Give each agent one job and a
  `schema`, so the next stage reads `{{ stage.object }}`, not prose. Hand
  bigger things over as files in the run's directory. Independent stages
  run side by side (`depends`), and several checkers can each take a slice
  before one stage decides.
- **What the producer knows is Concepts.** Its knowledge of the field is a
  Concept tree it reads (`context/concept`, `graph/graph-get`), not text in
  its prompt. Improving it means writing graph data: reviewable, undoable,
  and visible to every other agent in the workspace.
- **Models are `params`**, not input, so a launch can pick them and the
  workflow's defaults apply otherwise.

## Improve and climb

- **The improve reads misses, proposes, checks, writes.** It reads the
  misses (across runs where it can, never one run alone, or it learns that
  one task) and proposes changes to what the producer reads. Code checks
  every proposal against something the model cannot fake: a quote must be
  in the material word for word, a code must exist in its table. Then the
  improve writes what passed, and the next proposal sees why the rest
  failed.
- **The answers can be wrong.** A contest is DATA: the agent disputes one
  item with evidence, code checks the evidence, the contest is stored in
  the graph, and the scorer applies it (a replacement, or an exclusion). A
  model's say-so never changes a score.
- **The climb is one run.** It stops at the target, at the budget, or when
  an improve proposed nothing at all. A rejected attempt does not stop it:
  the next proposal sees why it was rejected. Put the cost in what it
  optimizes, or it will buy accuracy with more agents.
- **A win is promoted, never left in a run.** It becomes a Concept, or a
  param default in a new version.

## Measuring honestly

- Tasks you climbed on are a training set. Report on tasks you did not
  tune on. A handful of tasks is an anecdote: one flip moves the number a
  lot.
- Each task's material lives in its own namespace, so one run cannot feed
  another. What was learned lives in the shared tree, on purpose.
- No silent caps. If a run samples, truncates or skips, its output says
  so.
- Give every miss a cause (format, persistence, tooling, capability). It
  is the cheapest way to send each failure to the layer that owns it.
- A real run can cost tens of dollars and take an hour. Test the pieces
  one step at a time (`run_step`, or `meta/run-step` in a builder job) on
  the smallest task, and test a step that writes on a scratch namespace.
  Launch a whole run only when the person asks for one.
