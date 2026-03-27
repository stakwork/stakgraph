# mcp/web

Vite + React + Three.js frontend for stakgraph. Serves on `/` from the Express server.

## Stack

- **Vite** — build tool
- **React 19** — UI framework
- **Tailwind CSS v4** — styling (with `@tailwindcss/vite` plugin)
- **shadcn/ui** — Button, Badge components (`src/components/ui/`)
- **Three.js / React Three Fiber** — 3D graph visualization
- **drei** — R3F helpers (Instances, Billboard, Text, Html, CameraControls)
- **Zustand** — state management
- **react-markdown** — markdown rendering for docs/concepts viewer
- **lucide-react** — icons
- **framer-motion** — sidebar animations

## Structure

```
src/
  graph/
    config.ts          # All graph tuning constants (edit this first)
    GraphScene.tsx      # Canvas + data fetching + camera
    types.ts            # GraphNode, NodeExtended, Link, etc.
    components/
      Graph.tsx              # Scene graph wiring (positions instances per frame)
      NodePoints.tsx         # Instanced spheres per node (colored by type)
      Edges.tsx              # LineSegments for all edges
      LayerLabels.tsx        # Billboard text labels per layer
      LayerHoverHighlight.tsx # Raycasted layer hover (plane + border + label)
      NodeDetailsPanel.tsx   # Html overlay for selected node details
  stores/
    useGraphData.ts    # Nodes, links, selection, feature highlight
    useSimulation.ts   # Deterministic grid layout (calculateGridMap)
    useIngestion.ts    # Ingestion phase, repo URL, credentials (persisted to localStorage)
    useChat.ts         # Chat messages, tool call events, streaming state
  components/
    Sidebar.tsx        # Docs + Concepts sidebar (fetches /docs and /gitree/features)
    DocViewer.tsx       # Markdown viewer for docs/concepts
    MarkdownRenderer.tsx
    chat/
      Chat.tsx           # Floating chat overlay on graph (pointer-events-none, centered max-w-2xl)
      ChatInput.tsx      # Auto-growing textarea with send button
      ChatMessage.tsx    # User/assistant message bubbles with markdown
      ToolCallFlow.tsx   # Real-time tool call display with icons + input preview
    ui/                # shadcn components (Badge, Button)
  hooks/
    useApi.ts          # Generic fetch hook
    useAgentChat.ts    # Calls POST /repo/agent with stream=true, parses AI SDK SSE stream
  lib/
    utils.ts           # cn() utility
  types.ts             # Doc, FeatureSummary, FeaturesResponse
  App.tsx              # Main layout (header + graph/doc view + sidebar + chat overlay)
  index.css            # Tailwind + dark theme CSS variables
```

## Graph config

All visual tuning lives in `src/graph/config.ts`:

- **LAYER_ORDER** — which node types to show and in what order (top to bottom)
- **INITIAL_CAMERA_POSITION** — `[x, y, z]` where z controls zoom (higher = more zoomed out)
- **LAYER_SPACING** — vertical distance between layers
- **GRID_SPACING** — horizontal distance between nodes within a layer
- **EDGE_OPACITY** — base edge transparency
- **NODE_SIZE** — sphere radius
- **AUTO_ROTATE_SPEED** — initial spin speed (stops permanently on user interaction)

## Data flow

1. `GraphScene` fetches `GET /graph?edges=true&no_body=true&node_types=...` and `GET /gitree/all-features-graph?no_body=true&node_types=...`
2. Merged + deduped nodes/edges go into `useGraphData.setData()`
3. `useSimulation.layoutNodes()` computes a deterministic square grid per layer (no physics)
4. `Graph.tsx` reads positions from `nodePositions` map in `useFrame` and sets them on drei `Instance` children
5. Sidebar concept hover calls `setHighlightedFeature()` which dims non-connected nodes/edges

## Chat overlay

A floating chat UI overlaid on the 3D graph (like hive's DashboardChat). Input anchored to bottom-center, messages bubble up.

1. `useAgentChat` hook sends `POST /repo/agent` with `stream: true`
2. Server calls `agent.stream()` on the same `ToolLoopAgent`, pipes `toUIMessageStreamResponse()` directly
3. Client reads the SSE stream (`data: {json}\n\n` format) via `fetch` + `ReadableStream`
4. `text-delta` events stream text into `useChat.appendText()`, shown in real-time
5. `tool-input-available` events add tool calls to `useChat.addToolCall()`, shown in `ToolCallFlow`
6. Repo URL is derived from graph Repository nodes (`source_link` or `https://github.com/{name}`), not dependent on ingestion store
7. Credentials (username/pat) for private repos are persisted to localStorage via `useIngestion.setRepo()`
8. The chat overlay uses `pointer-events-none` on wrapper divs so the 3D graph remains interactive; only message bubbles, input, and buttons capture events

## Building

```sh
cd mcp/web
npm install
npm run build    # outputs to dist/
npm run dev      # vite dev server on :5173
```

The Express server in `mcp/src/index.ts` serves `web/dist/` as static files on `/`. Run `npm run build-web` from `mcp/` to install + build in one step.

## API endpoints used

- `GET /graph?edges=true&no_body=true&node_types=...&limit=500&limit_mode=per_type` — code graph nodes + edges
- `GET /gitree/all-features-graph?no_body=true&node_types=...` — feature nodes + edges
- `GET /docs` — repository documentation
- `GET /gitree/features` — feature summaries for sidebar
- `GET /gitree/features/:id` — full feature detail (on concept click)
- `POST /repo/agent` with `stream: true` — streaming agent chat (AI SDK SSE format)
- `POST /repo/agent` without `stream` — async agent (returns `request_id`, poll via `/progress`)
