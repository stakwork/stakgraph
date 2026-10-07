---
description: A change made and run in a pod — a sandbox with the app running — when the diff alone cannot be judged.
parent: Code Change
---
A change made and run in a pod — a sandbox with the app running — when the diff alone cannot be judged.

Take this way instead of `code-change-pr` when the change must build, run
or be tested to be judged, when the person wants to click through it, or
when they ask for it. A pod is hive's sandbox: the workspace's repositories,
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

Judge the work: `pod/branch-diff` for the change as it stands, `pod/test`
to run the app's tests. Then `pod/push { repo_url, branch_name, commit_message, base_branch }`
opens the pull request; report it as `{ id: "pr", kind: "pull_request",
title, content: { url, repo: "<owner>/<name>", number, state: "open" } }`.
A later turn that revises the same change pushes with
`stay_on_branch: true, create_pr: false` — the same branch, the same pull
request, the same `pr` id. Never open a second pull request for one change.

Release the pod with `pod/release { workspace, podId }` when the job is done
with it — the pull request is merged, or the person says so. Otherwise keep
it: strut releases it itself when the job is deleted or sits idle. If a
pod tool says the pod rejected its password, the pod is gone: claim again.
A `no pod available` error means the pool is empty: tell the person and
try later.
