---
description: A change to a repository's source code, delivered as a pull request.
parent: Workflow Builder
---
A change to a repository's source code, delivered as a pull request.

The pull request IS the proposal: review happens there, on GitHub. A
follow-up ("also rename the helper", "the test you added fails") is the
SAME pull request — its branch, with the earlier commits on it — never a
second one.

To make a change, run the `code-change-pr` workflow (read it first with
meta/get-workflow): `{ repo, prompt, title }` — the repository URL, the
brief for the coding agent, the pull request's title. It checks the
repository out, lets a coding agent make the change in an isolated working
copy, commits and pushes it as the person, and opens the pull request; the
output carries `url`, `number`, `repo` (owner/name), `branch`, `created`,
the files changed and the agent's summary. To revise that pull request on
a later turn, run it again with the same `branch`. Nothing here is done by
hand: never clone, commit or push yourself.

The credential is the run's own — the person's GitHub token, bound to the
run — so there is nothing to pass or to look up. A `no_push_permission:`
error means that token cannot push to the repository: stop and ask.

The brief (`prompt`) must stand alone: the coding agent sees nothing of
this conversation. Name the repository, the files or area, the exact
change, what to check, and the conventions to match. When a plan for the
change already exists in your directory, hand over its relevant section
rather than a paraphrase. If the person has not named the repository and
nothing here does, ask before running anything.

Report the pull request as an artifact of kind `pull_request`, under a
stable id (`pr`, or one per repository when a job touches several), with
`content: { url, repo, number, state: "open" }` taken from the output.

A change you can judge from the diff alone — text, a constant, a small
local edit — takes this path. A change that must build, run or be tested
to be judged needs a sandbox with the application running; when this
workspace has a workflow for that, it is a child of this Concept and its
page says when to prefer it and what it takes.

Each child Concept of this node (PARENT_OF from here) is either another
WAY to deliver a change — a sandbox workflow, named with its input — or a
REPOSITORY's own rules: test commands, branch conventions, a preferred
workflow. Its first line says which. Read the children before choosing and
open the one that matches the request. To record one: `graph/create-node`
with node_type "Concept" and node_data { name, description (one line),
docs }, then `graph/create-triplet` with source_ref_id = this node's
ref_id, edge_type "PARENT_OF", target_ref_id = the new Concept's ref_id.
