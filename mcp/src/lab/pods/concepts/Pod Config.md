---
description: Making a pod run the app the way a request needs — fixed in the pod first, then promoted to every pod of the workspace when everyone needs the fix.
parent: Pod
---
Making a pod run the app the way a request needs — fixed in the pod first, then promoted to every pod of the workspace when everyone needs the fix.

Open this when the app does not run in the pod, or must run differently for
this request: with mocks, with an env var or a flag changed, with an extra
service beside it (a database, an object store).

**Look first.** Before claiming a pod for a workspace whose app you have not
run yet, look for an earlier repair: `job/list { q: "pod config <workspace>" }`,
then `job/get` on a hit. Its `pod-config` deliverable is the diff that was
promoted and what a fresh pod still lacked; its thread says how it got there.

**What a pod runs from.** Hive builds every pod of a workspace from its pool
config: `services` (one pm2 app each; `frontend` is the one people open),
workspace-wide env vars, `docker-compose.yml` (the containers beside the
apps) and `Dockerfile`. In the pod they are files under
`/workspaces/.pod-config/.user-dockerfile/`. A service's commands are env vars
of its pm2 app: INSTALL_COMMAND, BUILD_COMMAND, PRE_START_COMMAND (before the
apps start), POST_START_COMMAND (after), REBUILD_COMMAND, TEST_COMMAND,
E2E_TEST_COMMAND, RESET_COMMAND, and PORT.

**Fix it in the pod.** Hand the agent in the pod the fix as a task that
stands alone, with what it needs to know about the pod:

- edit those files; `npx -y staklink reload` restarts the apps and runs the
  install, build and start commands again;
- never `pm2 stop all`, `pm2 delete all`, `pm2 restart all` or `pm2 kill`: the
  agent runs inside pm2 and would stop itself. One app: `pm2 restart <name>`;
  logs one app at a time: `pm2 logs <name> --lines 100 --nostream`; the
  containers: `docker ps`, `docker logs <name>`;
- the start command is the development one (with hot reload where the app
  has it), bound to `0.0.0.0` so the app is reachable from outside the pod;
- there is always a `frontend` app; when the code has no web app or API to
  open, a placeholder (`npx -y hell0-w0rld`);
- with several repositories, each app's `cwd` is its own `/workspaces/<repo>`;
- an Android app: base image `ghcr.io/stakwork/staklink-android:latest`, PORT
  8000 (the screen stream), `adb wait-for-device` before start, the APK built
  by BUILD_COMMAND and REBUILD_COMMAND, installed and launched after start;
- mock code, fixtures and seed scripts belong in the repository: they ship
  as a pull request (`pod/push`).

Ask it to say what it changed and why. Then judge the app yourself, as `Pod`
says (`browser-explore` on the frontend, `pod/run-tests`).

**Promote or keep.** Promote what every job on the workspace needs: the app
does not run, a service is missing, a setting is wrong for everyone. Keep
what only this request needs in the pod — mocks for this case, a flag flipped
to try something — and report it without promoting it. A fix that needs
repository changes (a start script, a seed, mock code) is promoted only once
its pull request merges: until then a fresh pod lacks the code the config
relies on. The merge reaches you as a turn (`[artifact-event] pull_request
<url> merged`); promote from there.

**Promotion asks first.** `pod/config { workspace, control, sealed }` returns
hive's config, the pod's files and a diff of each. End the turn with an
`ask`: the diff, what you verified, and whether to apply it to every pod of
the workspace. A pool config change recreates every unused pod of the
workspace — everyone's — so the person decides. With a pull request pending,
the ask goes with it ("once this merges, apply it to every pod?"), and the
merge's turn promotes on that yes.

**On yes: `pod/config-set`**, with only what changes, by name:
`services: { <name>: <the whole service> }` — copied from pod/config's
`services` with the fix applied; its masked env values, written back as they
are, keep their stored values — `env: { NAME: "value" }` for the
workspace-wide env, `null` to remove a name, and `containerFiles` with the
whole text of `docker-compose.yml` or `Dockerfile`. The pm2 file is generated
from `services`: a command goes in the service's `scripts` (`postStart` is
POST_START_COMMAND) and its port in `port`. `devcontainer.json` is never
changed. No credential is promoted: when the app needs a key, say which, and
a person sets it in hive. pod/config-set waits until a pod built from the
change is ready.

**Run the pool's config in your pod.** `pod/config-apply { workspace,
control, sealed }` writes hive's stored files into the pod you hold and
reloads it, with no new pod: after a promotion it brings your pod in line
with every other, and after trying things in the pod it puts the pool's
config back. It is not the check below: what was made by hand in your pod
is still there.

**Check it on a fresh pod.** `pod/claim { workspace, fresh: true }` claims a
second pod, built after the push (`no fresh pod ready yet`: try again
shortly). Run the same check there. It catches what was made by hand: a
bucket or a table created in the first pod exists only there, and the fix is
not done until creating it is in the config (the service's `postStart`, or a
service in the compose file that sets it up). When the check fails, fix
forward (another promotion, asked again) or put back what you changed, and
say which. A Dockerfile change cannot be judged in a pod built before it:
promote it, asking first, and judge it here.

**Report it.** A turn that promoted a fix — or set a pod up in a way worth
repeating, promoted or not — reports
`{ id: "pod-config", kind: "diff", title: "Pod config — <workspace>", summary: "<what broke, what fixed it, what a fresh pod still lacks>", content: "<the promoted files' diffs, as one text>" }`
so a later job finds it. Then `pod/release` the fresh pod.
