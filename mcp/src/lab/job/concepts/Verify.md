---
description: Stating what the work must deliver as claims about the job, each with a check that can tell, and verifying before saying it is done.
parent: Job
---
Stating what the work must deliver as claims about the job, each with a check that can tell, and verifying before saying it is done.

The work is done when it is verified, not when your last tool call
returned. A claim is about THIS job — `meta/add-claim` with subject
`{ kind: "job", name: <your job id> }` — and is one plain sentence about
what the work must deliver, as the person would check it: "the settings
page shows the notification preferences", "the export file has one row per
order", "GET /api/settings answers 200". Not how you will do it, and not
your output restated.

When. Once you understand what is asked, before the work, so the person
sees the contract early and can add their own. Look first:
`meta/list-claims` (or `claims` on `job/get` for your job) shows what is
already claimed, by you in an earlier turn or by the person from their
side. When the person stated a claim, hang your instrument on it with
`meta/add-check` rather than writing a second claim that says the same.

Each claim gets a check that can tell, the instrument that fits what the
work is:

- A page an app serves: `browser/verify` with `url: "{{ input.urls.pod }}/<path>"`
  — the address of the pod the job holds, so the check survives the pod
  being claimed again — the steps to get there (browser/capture's list)
  and `expect`: `text` that must be visible, the `url` the page should end
  at, `selector`s that must be present, `noErrors`. It answers with a
  screenshot of where the page got to. Run it yourself first, as a tool:
  you see what the check will see, then hang the same config on the claim.
- An endpoint: `http/verify` with the request and the answer expected —
  `status`, `contains`, `json` paths — at `{{ input.urls.pod }}/<path>`.
  A credential is named, never written: `secrets: { Authorization: "Bearer {NAME}" }`.
- A file you wrote, or tests over code checked out in your directory:
  `exec` with `cwd: "{{ input.dir }}"` (a repository is a subdirectory of
  it), a command that exits 0 when the claim holds.
- Tests on a pod: `pod/run-tests` with the pod's `control` and `sealed`.
  It names that pod: after you claim another, edit the check (`meta/edit-check`).
- What only a person can judge — does it look right, is it what they
  meant: an external check, `{ description }` saying what to look at and
  why code cannot. The person answers it on their side.

A check on a job is `manual`: it runs when the job is verified, not on
every turn.

Before you say the work is done, call `meta/verify-job`. It runs every
check against the job as it is now and returns the ledger. A claim
`refuted` means not done: fix the work (or a check that was wrong,
`meta/edit-check`) and verify again. `unknown` with a check that could not
run (its `lastVerify` says why — a pod that is gone, a page that does not
open) is not done either: fix what kept it from running. A claim waiting
on a person is not yours to answer: say in `text` what they should look
at. Say in `text` what was verified, and never call the work done while a
claim is refuted. If you cannot fix it this turn, say so plainly.

At the start of a turn, read the ledger: `claims` on `job/get` for your
job. A later turn makes every verdict `stale` until the job is verified
again. When a person refutes a claim — on their side, or in a
`[claim-event]` line that opens your turn — their word stands on the turn
they looked at until the turn moves: verifying that turn again does not
lift it. Treat it as the request: fix what they said, then verify; the
new turn's verdict is what counts.

A check's screenshot is a file in your directory (`shots/NNN.png`) and the
verdict's locator: the person sees it beside the claim.
