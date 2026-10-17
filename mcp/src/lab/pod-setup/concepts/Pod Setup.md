---
description: "How a workspace's repositories run in a pod — the files hive builds every pod from, and what they must say."
parent: Pod Config
---
How a workspace's repositories run in a pod — the files hive builds every pod from, and what they must say.

A pod is a container with the workspace's repositories at
`/workspaces/<repo>` and pm2 running its apps. Hive builds every pod of a
workspace from its pool config:

- **services**: one pm2 app each — the repository it runs in (and a
  directory in it), its port, its commands (install, and when the
  repository has them build, test, pre-start, post-start, rebuild, reset)
  and its env vars. `pm2.config.js` is generated from them.
- **docker-compose.yml**: only what runs BESIDE the apps — a database, a
  cache, an object store. The pod is the container the apps run in: there
  is never a service for the app itself.
- **Dockerfile**: the pod's base image, `ghcr.io/stakwork/staklink-js:latest`
  unless a platform's page says otherwise.

Every setup follows these rules:

1. **One service is the one a person opens**: the web app, or for a
   backend-only project its API. When nothing in the workspace serves a
   page or an API, that service is a placeholder, `npx -y hell0-w0rld` on
   port 3000, a hello-world server that shows the pod is up.
2. **It binds 0.0.0.0**, never localhost or 127.0.0.1, so it is reachable
   from outside the pod.
3. **Development, not production**: the dev script, with hot reload where
   the app has it; `start` only when there is no dev script.
4. **Commands match the package manager** the repository uses — its
   lockfile says which (`yarn.lock` yarn, `pnpm-lock.yaml` pnpm,
   `package-lock.json` npm, `bun.lock` or `bun.lockb` bun). Never `npm run
   …` in a yarn or pnpm repository.
5. **Every service has an install command.** Add build, test, pre-start (a
   migration, a schema) and post-start (a seed) when the repository has
   them.
6. **Local only.** The setup boots with no external account and no real
   credential: every dependency is a service beside the app, the
   repository's own mock or offline mode, or a deterministic placeholder.
   `Pod Setup - Local Services` says which.
7. **Every env var the code reads gets a working value.** A secret gets a
   dev value (`dev-secret-key-change-in-production-12345678901234567890`), a
   hex key a hex one (`bb54aa41a75298418586c5443f264338013520b3bad612fce9ac2fc32ed19882`),
   a URL to a service beside the app `127.0.0.1` and the port that service
   publishes, and a credential the same value the compose file gives the
   service.
8. **With several repositories**, each service runs in its own; usually one
   is the app and the others are local dependencies it builds against,
   installed and built first (its install or pre-start command does it).

Done for you, in code: the person-facing service's name; each service's
directory, `/workspaces/<repo>/<dir>`, from the repository and directory
you name; the compose file's top-level volumes; a compose service's name
used as a host in an env value (it becomes `127.0.0.1`); and the default
Dockerfile.

The pages under this one (PARENT_OF) are the platforms — how one kind of
app runs in a pod — and `Pod Setup - Local Services`, what runs beside it.
