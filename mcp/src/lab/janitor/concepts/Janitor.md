---
description: Automated agents for cleaning up code, concept trees, or other data.
parent: Workflow Builder
---
Automated agents for cleaning up code, concept trees, or other data.

Each child Concept of this node (PARENT_OF from here) is one janitor's
MANDATE. Its docs are the instructions an agent follows: what "dirty" means,
what to flag, what to propose instead. A mandate gives each kind of problem
it flags a short LABEL (one or two lowercase words); every finding comes
back under one of them, as its `issue`. A mandate says nothing about tools
or permissions — the workflow that runs it decides those, and it treats the
mandate as data.

To run a janitor, schedule the `graph-janitor` workflow with an automation
whose input names the mandate, where to sweep, and the cursor:
`{ concept: "<child of Janitor>", start: "<Concept whose subtree to sweep>",
cursor: "{{ last.output.next_cursor }}" }`.
One automation per janitor: its schedule is the janitor's schedule, and its
enabled switch turns the janitor off. Each run reviews a batch of the
subtree, the next names in alphabetical order after the cursor, so every
Concept comes round in turn. Without the cursor line every run reviews the
same first batch. The agent only proposes: each finding
carries an action (a passage to replace in the docs, a new description or
docs, or a new parent) that code checks against the live graph. By default a run is a dry run — the plan
only. With `apply: true` on the input, the checked plan is written: never a
rename, never this node, a mandate or a Concept the git seed owns, never a
value that changed since the agent read it; every write is read back, and
the run keeps an undo log. To undo a run, run `graph-janitor-revert` with
`{ run_id: "<that run>" }` (or `changes: ["e1"]` for some of its changes).

To add a janitor, never publish a workflow. Three steps:

1. Create the mandate: `graph/create-node` with node_type "Concept" and
   node_data { name, description (one line), docs (the mandate: what counts
   as dirty, what to flag and under which label, what to propose) }.
2. Hang it here: `graph/create-triplet` with source_ref_id = this node's
   ref_id, edge_type "PARENT_OF", target_ref_id = the new Concept's ref_id.
   Without that edge a run fails `not_a_janitor:`.
3. Schedule it: an automation on `graph-janitor` with input
   { concept: the mandate's name, start: the Concept to sweep,
   cursor: "{{ last.output.next_cursor }}" }.
