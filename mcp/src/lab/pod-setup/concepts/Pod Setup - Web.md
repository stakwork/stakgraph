---
description: "A web app or an API in a pod — served over HTTP from Node, Python, Go, Ruby or anything else: the dev command, the host flag, the port."
parent: Pod Setup
---
A web app or an API in a pod — served over HTTP from Node, Python, Go, Ruby or anything else: the dev command, the host flag, the port.

**The start command** is the repository's own dev command, bound to all
interfaces:

- **Next.js**: the dev script with the host flag, usually
  `-- --hostname 0.0.0.0` (`npm run dev -- --hostname 0.0.0.0`; some apps
  take `-H 0.0.0.0`). Prefer dev over start when a dev script exists.
- **Vite**: `-- --host 0.0.0.0`. Vite projects use dev, never start.
- **With yarn or pnpm**, the same flags through that manager:
  `yarn dev -- --host 0.0.0.0`, `pnpm run dev -- --hostname 0.0.0.0`.
- **Other servers** take their own flag or env var: `uvicorn app:app --host
  0.0.0.0`, `python manage.py runserver 0.0.0.0:8000`, `bin/rails server
  -b 0.0.0.0`, `flask run --host 0.0.0.0`, a `HOST=0.0.0.0` the code reads.
  A Go app runs with `go run .` (or its `cmd/<name>`), not a binary that
  may not exist yet.

**The port** is the one the dev server listens on. When the code reads
`PORT`, the service's port is what it gets.

**Install and build**: the package manager's install (`npm install`,
`yarn install`, `pnpm install`, `pip install -r requirements.txt`, `uv
sync`, `bundle install`, `go mod download`); a build only when the dev
command needs one first. In a monorepo the app's own directory, with the
workspace tool's command for that package.

**A backend beside the frontend** is a second service in its own directory
(or repository), on its own port; the frontend reaches it at `localhost`
and that port.
