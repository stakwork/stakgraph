---
description: "Onboarding a workspace: getting its app running in a pod for the first time — draft the setup, run it in a pod, check it, and fix it there."
parent: Job
---
Onboarding a workspace: getting its app running in a pod for the first time — draft the setup, run it in a pod, check it, and fix it there.

The request names the workspace (`@<slug>`) and its repositories. A
workspace not set up yet runs a placeholder in its pods — a `frontend` that
answers "hello world" — and the work is to give them the real setup, and
to see it run. Work in this order; nothing here waits for a new pod to be
built.

1. **Claim a pod**: `pod/claim { workspace }`. It runs what the pool holds,
   and once you hold it the pool leaves it alone. Report it at once, as `Pod`
   says (`pod` and `ide`, kind `url`), so the person can watch.
2. **Draft the setup**: `meta/run-workflow { name: "identify-services",
   input: { repos: [<the repository URLs>] } }`. It reads the repositories
   and the `Pod Setup` pages and returns `services` and `containerFiles`
   (the compose file, and a Dockerfile only when the platform needs one),
   with `notes` on what it fixed in code. Use its output as it is.
3. **Write it to the pool.** Read the pool first:
   `pod/config { workspace, control, sealed }`. When it holds the
   placeholder, a lone `frontend` running `npx -y hell0-w0rld`, this is the
   workspace's first setup and the person asked to be onboarded.
   Do not ask first: `pod/config-set { workspace, services,
   containerFiles, wait: false }`, with every service in `services` and
   the files as given. Any other pool config is not a first setup: it is
   one the workspace already runs, and replacing it asks first, as `Pod
   Config` says, with the draft beside what the pool holds. Go on from
   here on a yes. Every later change asks too.
4. **Run it in your pod**: `pod/config-apply { workspace, control, sealed }`.
   The pod gets the files hive now stores, brings up the compose services
   and installs, builds and starts the apps. Read `success`, `compose`,
   `reload` and `processes`.
5. **Check it.** Author the claim before you look: "the app answers at the
   pod's address" — `meta/add-claim` about this job, with an `http/verify`
   check on `{{ input.urls.pod }}` (a page that must render gets a
   `browser/verify` check instead), as `Verify` says. Then look:
   `browser-explore` on the pod's `frontend` with a goal that says what
   the app is, and verify with `meta/verify-job`.
6. **It works**: report a `pod-config` deliverable (`Pod Config` says its
   shape) with what the setup is and which `Pod Setup` pages it rests on,
   and a screenshot. Done.
7. **It does not**: fix it in this pod and promote the fix, as `Pod
   Config` says. A fix made by hand is what a fresh pod catches, so the
   `pod/claim { fresh: true }` check comes after you have reported the pod
   that works, never before.

When the draft itself fails (a repository cannot be checked out, a
`Pod Setup` page is missing), say what failed and stop: nothing was
written, and the pool's config keeps the pods booting until it is fixed.
