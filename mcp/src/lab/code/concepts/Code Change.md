---
description: A change to a repository's source code, delivered as a pull request.
parent: Job
---
A change to a repository's source code, delivered as a pull request.

The pull request IS the proposal: review happens there, on GitHub. A
follow-up ("also rename the helper", "the test you added fails") is the
SAME pull request — its branch, with the earlier commits on it — never a
second one.

Choose the way by ONE test, before anything else: can the change be
judged from its diff alone?

- Yes — text, a constant, a small edit in one place, right or wrong on
  sight: run `code-change-pr`, below.
- Anything else — new behaviour, a change across files or areas, anything
  its tests, type check, build or running app have to judge: a POD, this
  page's child `Pod`. When the change holds choices the person would want
  to make (what a screen shows, which way a design goes) and they have not
  made them, a plan comes first without being asked (`Plan Mode`, under
  `Job`), and the pod once they say go.

How detailed the request is, or how much you already know about the code,
does not change the answer: a fully specified change across several files
is still a pod. When you cannot tell yet, find out first (`Explore`, under
`Job`, in this same turn) and answer the test from the note. Never
install, build or test in the repositories in your directory, which are
for reading, and never clone, commit or push yourself.

`code-change-pr` (read it first with meta/get-workflow) takes `{ repo,
prompt, title }` — the repository URL, the brief for the coding agent, the
pull request's title. It checks the repository out, lets a coding agent
make the change in an isolated working copy, commits and pushes it as the
person, and opens the pull request; the output carries `url`, `number`,
`repo` (owner/name), `branch`, `created`, the files changed and the
agent's summary. To revise that pull request on a later turn, run it again
with the same `branch`.

The credential is the run's own — the person's GitHub token, bound to the
run — so there is nothing to pass or to look up. A `no_push_permission:`
error means that token cannot push to the repository: stop and ask.

The brief (`prompt`) must stand alone: the coding agent sees nothing of
this conversation. Name the repository, the files or area, the exact
change, what to check, and the conventions to match. The coding agent
cannot read your directory: when a note there bears on the change, put
what it found in the brief (the files and lines, the conventions, the
checks), quoted rather than paraphrased, and only what this change needs.
When a plan for the change already exists in your directory, hand over its
relevant section the same way. If the person has not named the
repository and nothing here does, ask before running anything.

Report the pull request as an artifact of kind `pull_request`, under a
stable id (`pr`, or one per repository when a job touches several), with
`content: { url, repo, number, state: "open" }` taken from the output.

Each child Concept of this node (PARENT_OF from here) is either another
WAY to deliver a change — a sandbox workflow, named with its input — or a
REPOSITORY's own rules: test commands, branch conventions, a preferred
workflow. Its first line says which. Read the children before choosing and
open the one that matches the request. To record one: `graph/create-node`
with node_type "Concept" and node_data { name, description (one line),
docs }, then `graph/create-triplet` with source_ref_id = this node's
ref_id, edge_type "PARENT_OF", target_ref_id = the new Concept's ref_id.
