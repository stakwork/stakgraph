---
description: "A web app or an API in a pod — served over HTTP from Node, Python, Go, Ruby or anything else: the dev command, the host flag, the port."
parent: Pod Setup
---
A web app or an API in a pod — served over HTTP from Node, Python, Go, Ruby or anything else: the dev command, the host flag, the port.

**The start command** is the repository's own dev command, bound to all
interfaces:

- **Next.js**: the dev script with the host flag, usually `--hostname
  0.0.0.0` (`npm run dev -- --hostname 0.0.0.0`; some apps take `-H
  0.0.0.0`). Prefer dev over start when a dev script exists.
- **Vite**: `--host 0.0.0.0` (`npm run dev -- --host 0.0.0.0`). Vite
  projects use dev, never start.
- **`--` is npm's alone**: npm needs it to pass a flag on to the script.
  yarn, pnpm and bun pass flags on by themselves, and pnpm hands a `--` to
  the script as it is, where the dev server takes the flag after it for
  something else or ignores it. So, with the framework's flag: `yarn dev
  --hostname 0.0.0.0`, `pnpm
  dev --host 0.0.0.0`, `bun run dev --host 0.0.0.0` — never `yarn dev --
  …` or `pnpm run dev -- …`.
- **Other servers** take their own flag or env var: `uvicorn app:app --host
  0.0.0.0`, `python manage.py runserver 0.0.0.0:8000`, `bin/rails server
  -b 0.0.0.0`, `flask run --host 0.0.0.0`, a `HOST=0.0.0.0` the code reads.
  A Go app runs with `go run .` (or its `cmd/<name>`), not a binary that
  may not exist yet.

**The port** is the one the dev server listens on. When the code reads
`PORT`, the service's port is what it gets.

**Install and build**: the package manager's install (`npm install`,
`yarn install`, `pnpm install`, `bun install`, `pip install -r
requirements.txt`, `uv sync`, `bundle install`, `go mod download`); a
build only when the dev command needs one first. In a monorepo the app's
own directory, with the workspace tool's command for that package.

**A backend beside the frontend** is a second service in its own directory
(or repository), on its own port; the frontend reaches it at `127.0.0.1`
and that port.
