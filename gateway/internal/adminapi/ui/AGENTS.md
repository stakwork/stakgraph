# Admin UI

Preact + Vite SPA that ships embedded in the gateway plugin's `.so`
via `//go:embed all:ui/dist`. Served at `/_plugin/ui/*` behind the
session cookie issued by `POST /_plugin/login`.

Designed for desktop operators (≥ 1280px). Dark-only, no theme
toggle. The whole bundle is ~134 KB / 50 KB gzipped.

## Stack

| Concern | Choice |
|---|---|
| Framework | **Preact** via `@preact/preset-vite` (React-compatible API, ~3 KB runtime) |
| Build | **Vite** with `base: '/_plugin/ui/'` |
| Routing | **`wouter-preact`** (hooks-based, flat routes, ~1.5 KB) |
| Data | **`@tanstack/react-query`** aliased through `preact/compat` |
| Charts | **`uPlot`** + a thin wrapper in `components/charts/UplotChart.tsx` |
| Styling | Vanilla CSS + custom properties; two files only (`styles/base.css`, `styles/components.css`) |
| Types | Hand-maintained in `src/api/types.ts` (will be tygo-generated once `make tygo` runs in CI) |

Explicit non-choices: no Tailwind, no styled-components, no SSR,
no react-router, no global state library beyond Tanstack Query, no
icon library (icons are inline SVGs in `components/icons.tsx`).

## Layout

```
ui/
├── package.json          # pinned deps; lockfile committed for reproducible CI
├── vite.config.ts        # base path, react→preact alias, dev proxy
├── tsconfig.json         # strict + preact/compat react paths
├── index.html            # SPA shell; "Agent Gateway" title
└── src/
    ├── main.tsx          # entry; renders <App/> into #root
    ├── app.tsx           # Router + QueryClient + route table
    ├── api/
    │   ├── client.ts     # typed fetch wrapper, 401 → UnauthorizedError
    │   ├── queries.ts    # one hook per endpoint (useMe, useSpendByAgent, …)
    │   ├── types.ts      # mirrors Go response structs
    │   └── window.ts     # Window label → seconds (shared by pages + chart)
    ├── components/
    │   ├── layout/       # Shell / Sidebar / Topbar
    │   ├── charts/       # UplotChart + CostHistogram
    │   ├── tables/       # DataTable (sortable)
    │   ├── controls/     # WindowPicker
    │   ├── icons.tsx     # UserIcon, BotIcon (inline SVG)
    │   ├── EmptyState.tsx
    │   └── ErrorBoundary.tsx
    ├── pages/
    │   ├── Login.tsx          # Basic auth → session cookie
    │   ├── Dashboard.tsx      # KPIs + cost-by-agent chart + top-5 tables
    │   ├── People.tsx         # users in the window
    │   ├── UserDetail.tsx     # one user's KPIs + chart + agents-used + runs
    │   ├── Agents.tsx         # agents in the window, with budget meter
    │   ├── AgentDetail.tsx    # one agent's chart + budget card + runs
    │   ├── RunDetail.tsx      # Provenance card + paginated call log
    │   └── NotFound.tsx
    └── styles/
        ├── base.css       # palette + reset (CSS variables on :root)
        └── components.css # utility classes for every component
```

## Local development

```bash
# From repo root:
make -C gateway docker-up   # bring up the backend (port 8181)

# Then, in this directory:
npm install
npm run dev                  # Vite dev server on :5173 with HMR
                              # /_plugin/* proxies to localhost:8181
```

Open <http://localhost:5173/_plugin/ui/>. The Vite proxy means
cookies set by `POST /_plugin/login` flow through to the SPA without
CORS.

## Building for the Docker image

The Dockerfile's `plugin-ui-builder` stage runs:

```dockerfile
COPY internal/adminapi/ui/package.json internal/adminapi/ui/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY internal/adminapi/ui/ ./
RUN npm run build
```

then the Go build stage drops the compiled bundle in place:

```dockerfile
COPY --from=plugin-ui-builder /pui/dist /plugin/internal/adminapi/ui/dist
```

so that `//go:embed all:ui/dist` in `gateway/internal/adminapi/ui.go`
picks it up at compile time. The runtime image has no Node.

For a local non-Docker build:

```bash
npm run build      # writes dist/ in place
```

The `.gitkeep` placeholder + a stub `index.html` live in `dist/`
in the repo so `go build` works offline without needing Vite to
have run first. `emptyOutDir: false` in `vite.config.ts` preserves
the placeholder across local builds. Hashed filenames (`assets/
index-<hash>.js`) bust browser cache automatically on every redeploy.

## Conventions

- **One hook per endpoint** in `api/queries.ts`. Polling cadence
  lives at the hook level, not the page — adding a new page that
  reuses an existing hook gets the right refetch interval for free.
- **All API calls go through `apiFetch`** in `api/client.ts`.
  Anything else loses the 401 → /login redirect.
- **401 = unauthenticated**, NOT just "you don't have permission."
  The QueryCache `onError` in `app.tsx` redirects to /login on any
  `UnauthorizedError` so individual pages never branch on auth.
- **Tables → `DataTable`**. Sortable, click-row navigation, ~80
  lines. Replace if/when feature creep makes it the bottleneck.
- **Icons are inline SVG** in `components/icons.tsx`. Match the
  existing stroke weight (1.8) and 24×24 viewbox; size with `1em`
  so they scale with surrounding text.
- **Currency formatting** scales digits with magnitude — values
  under 1¢ render with 6 decimals, otherwise 2. Helper duplicated
  across pages (cheap; not worth a util module yet).

## Auth model the SPA expects

| Endpoint | What the SPA sends | What the server does |
|---|---|---|
| `GET /_plugin/ui/*` | nothing | serves index.html (anonymous) |
| `POST /_plugin/login` | `Authorization: Basic <…>` | sets `bifrost_session` cookie, returns `{user}` |
| Everything else | `Cookie: bifrost_session=…` | resolves session, attaches user to req context |
| 401 anywhere | n/a | SPA's QueryCache catches → `setLocation('/login?next=…')` |

The SPA never sends a bearer token. Hive uses bearer; the dashboard
uses the cookie. Some endpoints accept either (the read-only ones);
the trust mutations (POST/DELETE on `/_plugin/trust/*`) and Hive's
bootstrap (`/_plugin/admin-credentials`) are bearer-only by design.

## When adding a new page

1. Define the Go response struct in `gateway/internal/adminapi/`
   (or extend an existing one).
2. Mirror it in `src/api/types.ts` (eventually tygo will generate
   this from the Go struct — until then keep them in lockstep).
3. Add a `useFoo` hook in `src/api/queries.ts` with the right
   polling cadence.
4. Create `src/pages/Foo.tsx` and wire a `<Route>` into `app.tsx`.
5. Add a sidebar entry to `components/layout/Sidebar.tsx` if it's
   a top-level destination.
6. Run `npx tsc -b --noEmit` and `npm run build` before pushing.

## When adding a new icon

Drop it into `components/icons.tsx` alongside `UserIcon` / `BotIcon`.
Use `currentColor` for stroke so it inherits parent color (e.g.
`text-dim` desaturates it along with the label).
