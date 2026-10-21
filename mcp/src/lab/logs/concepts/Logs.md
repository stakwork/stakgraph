---
description: Finding out what a swarm's logs say about something that happened, kept as a note in your directory that later turns start from.
parent: Job
---
Finding out what a swarm's logs say about something that happened, kept as a note in your directory that later turns start from.

Read the logs when the person asks why something failed, what the app or
its infrastructure did at a time, about an error they saw or for a
request's trace, and before fixing a failure you cannot already place. The
`logs` workflow does the reading as its own agent, with its own time
budget: it fetches the swarm's logs into files, searches them, counts and
groups with queries, and hands back only its analysis. Prefer it to
fetching logs yourself: your thread keeps the answer, not the lines.

Look before you ask. Your directory may already hold a note from an
earlier turn: they live under `notes/`. Work in another job may have one
too (`job/list { q }`, then `job/get` and `job/read`). When a note answers
the question — the same failure, the same window — use it. When it answers
part, ask only about the rest and revise that note — another job's in a
copy under your `notes/`, since its directory is not yours to write.

Which swarm. Your message's first line names the workspace this strut
belongs to, `@<slug>`; `logs` here reads its swarm. Any other `@<slug>` in
the request is another workspace of this org, with its own swarm and logs
on its own strut: a peer. Read the swarm where it happened; a request that
names no workspace means this one.

- This swarm: `meta/run-workflow { name: "logs", input: { prompt, session } }`.
  It runs under your job: what it fetched stays in your directory, under
  `logs/`.
- A peer's: `strut/run-workflow { peer: "<slug>", workflow: "logs", job, input: { prompt, session } }`.
  It runs there, with that swarm's access, and the answer is under
  `output`. What it fetched stays there, under the `job` you name — your
  job id will do; a peer keeps it apart from its own jobs.

Which log group, service or index to read is the workflow's to find: it
knows where its swarm logs. Say what happened, not where it was written.
Several swarms are several calls, one question each.

The question must stand alone: the logs agent sees nothing of this
conversation. Give the time window, made absolute with its time zone
("2026-10-20 14:00–14:30 UTC", never "this morning"); what to look for —
the error text exactly as the person saw it, a request id, a path, a
service or pod; and what the answer is for, so it knows what matters: "why
the upload failed, and whether it failed the same way before".

A follow-up. `logs` answers `{ answer, cost }`: `answer` is its analysis
in markdown — what it found, what in the logs it rests on, what it could
not confirm. It starts cold unless you give it a `session`, an id you
choose; the same `session` again asks about what it already fetched
instead of fetching it again. Ids are shared by every run, so make it
yours alone — `<your job id>/logs-<topic>` — and ask the follow-up under
the same job (on a peer, the same `job` too). A question about something
else is a new session.

Write the note. `logs` reads and answers; you keep what it found as
`notes/logs-<topic>.md`, the topic short and stable
(`notes/logs-upload-timeouts.md`), revised in place when you ask about the
same thing again. In it:

- The question, in one line, with its window, and the workspace when it is
  not this one.
- What was found: the answer, edited for a reader who will act on it. Keep
  timestamps, error text, request ids, services and log groups exact.
- What it rests on: the lines or counts as the logs gave them, each with
  its time and where it was logged.
- How sure it is, and what is still open or was not found. Never fill a
  gap from general knowledge.
- The `session` it ran in (and the peer's `job`), so a later turn can
  follow up.

Report it as `{ id: "notes-logs-<topic>", kind: "markdown", title, label: "Notes", path: "notes/logs-<topic>.md", summary }`,
the same id each time it is revised. Another job's note that you used
unchanged is reported the same way, with the `url` job/get gave it in
place of `path` — a path named in `text` is nothing the person can open.
In `text`, give the gist in a few sentences; the note carries the detail.

A note is how the work goes on. The next turn reads it instead of asking
the logs again. A fix built on it carries what it found in the brief: the
agent making the change, here, on a peer or in a pod, cannot read your
directory.
