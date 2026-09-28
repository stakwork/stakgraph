---
description: Automated agents for cleaning up code, concept trees, or other data.
parent: Workflow Builder
---
Automated agents for cleaning up code, concept trees, or other data.

Each child Concept of this node (PARENT_OF from here) is one janitor's
MANDATE. Its docs are the instructions an agent follows: what "dirty" means,
what to flag, what to propose instead. A mandate says nothing about tools or
permissions — the workflow that runs it decides those, and it treats the
mandate as data.

To run a janitor, schedule the `graph-janitor` workflow with an automation
whose input names the mandate and where to sweep:
`{ concept: "<child of Janitor>", start: "<Concept whose subtree to sweep, e.g. Law>" }`.
One automation per janitor: its schedule is the janitor's schedule, and its
enabled switch turns the janitor off. A run proposes cleanups; it never
applies them.

To add a janitor, never publish a workflow. Three steps:

1. Create the mandate: `graph/create-node` with node_type "Concept" and
   node_data { name, description (one line), docs (the mandate: what counts
   as dirty, what to flag, what to propose) }.
2. Hang it here: `graph/create-triplet` with source_ref_id = this node's
   ref_id, edge_type "PARENT_OF", target_ref_id = the new Concept's ref_id.
   Without that edge a run fails `not_a_janitor:`.
3. Schedule it: an automation on `graph-janitor` with input
   { concept: the mandate's name, start: the Concept to sweep }.
