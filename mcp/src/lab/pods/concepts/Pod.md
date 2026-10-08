---
description: A change made and run in a pod — a sandbox with the app running — when the diff alone cannot be judged.
parent: Code Change
---
A change made and run in a pod — a sandbox with the app running — when the diff alone cannot be judged.

Take this way instead of `code-change-pr` when the change must build, run
or be tested to be judged, when it follows a plan the person approved,
when they want to click through it, or when they ask for it. A pod is hive's sandbox: the workspace's repositories,
the app's dev server (`frontend`), an IDE (`ide`), and a coding agent of its
own that works in there. The `pod/*` tools drive it; the job's messages
start with `Hive workspace: <id>` — the id to claim for.

Claim ONCE per job: `pod/claim { workspace }`. Its `sealed` and `control`
go to every other pod tool. The pod stays with the job between turns — the
agent in it remembers, the dev server keeps running — so report it as soon
as you have it, with stable ids, in every turn you work in it:

- `{ id: "pod", kind: "url", title: "Pod", url: <frontend> }`
- `{ id: "ide", kind: "url", title: "IDE", url: <ide> }`

Then `pod/latest { control, sealed, repos: [{ url: "https://github.com/<owner>/<name>.git" }] }`
once, so the repositories start at the base branch. To revise a pull
request on a pod that was released in between, claim again and give
`latest` the pull request's branch as `base_branch`.

Hand the work to the agent in the pod with the task standing alone (the
repository, the files or area, the exact change, what to check, the
conventions to match) — it sees nothing of this conversation. On the first
turn in a pod, `pod/agent-start` and END YOUR TURN: say what it is doing,
list the pod artifacts, so the person can watch it in the frontend; on the
next turn `pod/agent-status { request_id }` tells you how it went. When the
cards are already up, `pod/agent` waits for the task and returns its
output. The agent's session is this job, so it remembers its earlier turns
in the pod: a follow-up can be short.

Judge the work: `pod/branch-diff` for the change as it stands, `pod/run-tests`
to run the app's tests, and when the change is something the app SHOWS, see
it. Run `browser-explore` as a child run — `meta/run-workflow { name:
"browser-explore", input: { goal, url } }` — with `url` the pod's `frontend`
plus the path of the page that changed, and `goal` saying what the person
asked for and what the page should now show, in words that stand alone (the
explorer sees nothing of this conversation). It drives a real browser, looks
for itself and answers `{ answer, evidence: [{ path, url }], pages }`; its
screenshots never enter this thread, and on a job they are written to your
directory as `shots/NNN.png`. Report the last evidence as
`{ id: "after", kind: "image", title: "After", label: "Screenshot", url: <its url, verbatim>, summary: <the answer, one line> }`;
a shot taken the same way before `pod/agent-start` ("show the page as it
is"), reported as `before`, gives the person the pair. When the answer says
the goal was not met, hand it to the agent in the pod as a follow-up before
pushing. The dev server reloads source edits on its own; a new dependency or
a crashed server does not, which the explorer's `browser_observe` reports,
and `pod/run-tests` stays the judge for a change the app does not show.

Then `pod/push { repo_url, branch_name, commit_message }`
opens the pull request onto the repository's default branch — the pod reads
that from the remote, so leave `base_branch` out for a new change; a guess
(`main` on a `master` repository) is refused with the real default named.
Report it as `{ id: "pr", kind: "pull_request",
title, content: { url, repo: "<owner>/<name>", number, state: "open" } }`.
If `pod/push` fails, its error says why — a base the remote does not have,
nothing to commit — so fix that and call `pod/push` again. Never have the
agent in the pod commit, push or open the pull request itself: it has no
GitHub identity of the person's, and a pull request it opens is somebody
else's.
A later turn that revises the same change pushes with
`stay_on_branch: true` — the same branch, and `pr_url` is the same pull
request, the same `pr` id. Never open a second pull request for one change.
That holds only while the pod is still on the pull request's branch: after
`pod/latest`, or on a freshly claimed pod, it sits on the base, and a push
with `stay_on_branch` there would land on the base branch itself.

What happens to a pull request you reported reaches you as a turn whose
message begins `[artifact-event] pull_request <url> <what happened>`, the
specifics on the lines below — hive forwards it; nobody wrote it. You
decide, with the tools you have:

- **`… merged` or `… closed`:** look at the pod that pushed it. Release it
  (`pod/release`) when nothing else you pushed from it is still open and no
  work is pending in it; otherwise keep it and say what it is still for.
  Report the pull request's card with its new `state`. One line of `text`;
  no other artifacts.
- **`… checks failed`** (the head commit and the failing checks follow, each
  with a link): the pod is usually still on the pull request's branch.
  Reproduce (`pod/run-tests`, or the check's own command), hand the fix to
  the agent in the pod with the failure standing alone, judge it, then
  `pod/push { stay_on_branch: true }` so the same pull request gains the
  commit — never a second pull request. Report the same `pr` id. If the pod
  is gone (password rejected, or released after a merge), claim again and
  give `pod/latest` the pull request's branch as `base_branch`, as above.
- **An event about an artifact you do not know:** say so and do nothing —
  never release a pod on a guess.

Release the pod with `pod/release { workspace, podId }` when the job is done
with it — every pull request from it is merged or closed, or the person
says so. Otherwise keep it: strut releases it itself when the job is
deleted or sits idle. If a pod tool says the pod rejected its password,
the pod is gone: claim again. A `no pod available` error means the pool is
empty: tell the person and try later.
