---
description: Automated agents for cleaning up code, concept trees, or other data.
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

To add a janitor, add a child Concept here with the mandate as its docs,
then add the automation.
