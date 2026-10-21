---
description: Stating what the work must deliver as a claim about the job — one per change — with a check that uses what was delivered the way its user would, or the person's look when nothing here runs it, and verifying before saying it is done.
parent: Job
---
Stating what the work must deliver as a claim about the job — one per change — with a check that uses what was delivered the way its user would, or the person's look when nothing here runs it, and verifying before saying it is done.

The work is done when it is verified, not when your last tool call
returned. A claim is about THIS job — `meta/add-claim` with subject
`{ kind: "job", name: <your job id> }` — and is one plain sentence about
what the person gets: "the settings page shows the notification
preferences", "GET /api/settings answers 200", "`acme export --csv` writes
one row per order". Not how you will do it, and not your output restated.

How many. ONE claim for a change. Several only when the ask has separate
parts a person would check separately — a new page AND a new email — one
per part. A one-line fix is one claim.

When. Once you understand what is asked, before the work, so the person
sees the contract early. Look first: `meta/list-claims` (or `claims` on
`job/get` for your job) shows what is already claimed, by you in an
earlier turn or by the person from their side. When the person stated a
claim, hang your check on it with `meta/add-check` rather than writing a
second claim that says the same.

The check uses what you delivered the way its user would, where it runs —
the instrument that fits what the work is:

- A page an app serves, in the pod the job holds: `browser/verify` with
  `url: "{{ input.urls.pod }}/<path>"` — the address of the pod, so the
  check survives the pod being claimed again — the steps to get there
  (browser/capture's list; each check starts signed out, so a page behind
  a sign-in signs in there) and `expect`: `text` that must be visible, the
  `url` the page should end at, `selector`s that must be present,
  `noErrors`. It answers with a screenshot of where the page got to, which
  is what the person looks at. Run it yourself first, as a tool: you see
  what the check will see, then hang the same config on the claim.
- A feature that is a flow — several views a person clicks through, where
  the one view at the end does not show it works: `browser/walkthrough`,
  the same config plus a `title` and a `caption` on each step a person
  should see. It records a frame after each captioned step, and the person
  scrubs through them beside the claim. Most changes are one page: use
  `browser/verify` for those. Keep a walkthrough to a few frames; a longer
  flow is several claims.
- An endpoint: `http/verify` with the request and the answer expected —
  `status`, `contains`, `json` paths — at `{{ input.urls.pod }}/<path>`.
  A credential is named, never written: `secrets: { Authorization: "Bearer {NAME}" }`.
- A command, a CLI, a script or a file you deliver: `exec` with
  `cwd: "{{ input.dir }}"` (a repository is a subdirectory of it), running
  it as its user would — a command that exits 0 when the claim holds.
  Never an install, a build, a type-check or a test suite there.
- Tests you were asked to add or fix, on a pod: `pod/run-tests` with the
  pod's `control` and `sealed`. It names that pod: after you claim
  another, edit the check (`meta/edit-check`).
- What cannot run here — no pod holds the app the change is in — or what
  only a person can judge (does it look right, is it what they meant): an
  external check, `{ description }` saying what the person should open and
  look at. They accept or reject it on their side.

Never a check that reads your own diff or code to confirm you wrote what
you meant to — a grep of your diff, a script asserting your code's shape.
It only says your work agrees with itself; the person learns nothing from
it. Use the thing, not its source.

A check on a job is `manual`: it runs when the job is verified, not on
every turn.

Before you say the work is done, call `meta/verify-job`. It runs the
checks against the job as it is now and returns the ledger. A claim
`refuted` means not done: fix the work (or a check that was wrong,
`meta/edit-check`) and verify again. `unknown` with a check that could not
run (its `lastVerify` says why — a pod that is gone, a page that does not
open) is not done either: fix what kept it from running. A claim waiting
on the person is not yours to answer: say in `text` what they should look
at. Never call the work done while a claim is refuted. If you cannot fix
it this turn, say so plainly.

At the start of a turn, read the ledger: `claims` on `job/get` for your
job. A later turn makes every verdict `stale` until the job is verified
again. When the person rejects a claim — on their side, or in a
`[claim-event]` line that opens your turn — their word stands on the turn
they looked at until the turn moves: verifying that turn again does not
lift it. Treat it as the request: fix what they said, then verify; the
new turn's verdict is what counts.

A check's screenshot is a file in your directory (`shots/NNN.png`), a
walkthrough a directory (`walkthroughs/NNN/`, its `walkthrough.json` the
locator): the person sees it beside the claim. To hand either to the person
as well, list it in `artifacts` by the path the tool gave you — the same
path, never a copy — with `kind: "walkthrough"` for a recording.
