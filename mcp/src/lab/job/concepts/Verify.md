---
description: Stating what the work must deliver as a claim about the job — one per change — checked by a screenshot when the job holds a pod and by the person's look when it does not, and verifying before saying it is done.
parent: Job
---
Stating what the work must deliver as a claim about the job — one per change — checked by a screenshot when the job holds a pod and by the person's look when it does not, and verifying before saying it is done.

The work is done when it is verified, not when your last tool call
returned. A claim is about THIS job — `meta/add-claim` with subject
`{ kind: "job", name: <your job id> }` — and is one plain sentence about
what the person will see: "the settings page shows the notification
preferences", "the strut page fills the window edge to edge". Not how you
will do it, and not your output restated.

How many. ONE claim for a change. Several only when the ask has separate
parts a person would check separately — a new page AND a new email — one
per part. A one-line fix is one claim.

When. Once you understand what is asked, before the work, so the person
sees the contract early. Look first: `meta/list-claims` (or `claims` on
`job/get` for your job) shows what is already claimed, by you in an
earlier turn or by the person from their side. When the person stated a
claim, hang your check on it with `meta/add-check` rather than writing a
second claim that says the same.

The check is one of two, and nothing else:

- The job holds a pod (you claimed one, so `input.urls.pod` is its
  address): `browser/verify` with `url: "{{ input.urls.pod }}/<path>"` —
  the page that shows the change, the address surviving the pod being
  claimed again — the steps to get there (browser/capture's list) and
  `expect`: `text` that must be visible, the `url` the page should end at,
  `selector`s that must be present. It answers with a screenshot, which
  is what the person looks at. Run it yourself first, as a tool: you see
  what the check will see, then hang the same config on the claim.
- The job holds no pod: an external check, `{ description }` saying what
  the person should open and look at once the change is live — the page,
  what should be different. The person accepts or rejects it on their
  side.

Never a script, a command or a request over your own work as a check — no
`exec`, no `http/verify`, no grep of your own diff. It only says your work
agrees with itself; the person cannot see it and learns nothing from it.
Run whatever you like while you work; it is not a claim's check.

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

A check's screenshot is a file in your directory (`shots/NNN.png`) and the
verdict's locator: the person sees it beside the claim.
