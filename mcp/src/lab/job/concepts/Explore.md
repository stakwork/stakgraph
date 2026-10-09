---
description: Finding out how something works in a workspace's code or graph, kept as a note in your directory that later turns and the coding agent start from.
parent: Job
---
Finding out how something works in a workspace's code or graph, kept as a note in your directory that later turns and the coding agent start from.

Explore when the person asks how something works, where it lives or what
a change would touch, and before a change you cannot already place to the
file. The `explore` workflow does the reading as its own agent, with its
own budget, and hands back only what it found. Prefer it to reading a
repository yourself: your thread keeps the answer, not the files.

Look before you explore. Your directory may already hold a note from an
earlier turn: they live under `notes/`. Work in another job may have one
too (`job/list { q }`, then `job/read`). When a note answers the question,
use it. When it answers part, explore only the rest and revise that note.

Which workspace. Your message's first line names the workspace this strut
belongs to, `@<slug>`. Any other `@<slug>` in the request is another
workspace of this org, with its own graph and repositories on its own
strut: a peer. Explore where the code lives; a request that names no
workspace means this one.

- This workspace: `meta/run-workflow { name: "explore", input: { prompt, repos } }`.
  `repos` are the repository URLs to read, checked out into your directory
  and kept for the job. Without them it answers from this graph alone.
- A peer: `strut/run-workflow { peer: "<slug>", workflow: "explore", input: { prompt } }`.
  It runs there, against that workspace's graph, and the answer is under
  `output`. Its `ref_id`s belong to that graph: never look them up in
  this one. Add `repos` only for detail the graph lacks; if a checkout
  fails there, ask again without them.

Several workspaces or areas are several calls, one question each.

The question must stand alone: the explorer sees nothing of this
conversation. Say what to find, where to look when you know, and what it
is for, so it knows what matters: "the files a change to X would touch,
the conventions they follow, and the tests that cover them".

Write the note. `explore` is read-only and answers
`{ answer, sources, confidence }`; you keep it as `notes/<topic>.md`, the
topic short and stable (`notes/billing-webhooks.md`), revised in place when
you explore the same thing again. In it:

- The question, in one line.
- What was found: the answer, edited for a reader who will build on it.
  Keep file paths, function names and line ranges exact.
- Sources: each file as `<repo>/<path>:<lines>`, each graph node by name,
  and the workspace it came from when it is not this one.
- How sure it is, and what is still open or was not found. Never fill a
  gap from general knowledge.

Report it as `{ id: "notes-<topic>", kind: "markdown", title, label: "Notes", path: "notes/<topic>.md", summary }`,
the same id each time it is revised. In `text`, give the gist in a few
sentences; the note carries the detail.

A note is how the work goes on. A change built on it passes its path to
the workflow that makes the change, and the next turn reads it instead of
exploring again.
