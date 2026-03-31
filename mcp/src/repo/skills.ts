export const SKILLS: Record<string, string> = {};

export const mermaid = `
## Mermaid Diagram Style Guide

When generating mermaid diagrams for workspaces, follow these rules:

### Structure
- Use \`graph TD\` (top-down) for system/architecture diagrams
- Group related nodes with \`subgraph Name["Label"]\`
- Use cylinder syntax \`[("label")]\` for databases/stores
- Use descriptive edge labels: \`-->|"verb phrase"|\`
- Add a \`%% ── SECTION NAME ──\` comment before each subgraph block

### Color Classes
Always end the diagram with classDef definitions and class assignments.
Use this palette (dark-mode optimized, muted fills, bright borders):

\`\`\`mermaid
classDef client    fill:#1e3a5f,stroke:#5b9cf6,color:#c7e2ff
classDef gateway   fill:#431c0d,stroke:#fb923c,color:#ffe4cc
classDef service   fill:#2e1a4a,stroke:#a78bfa,color:#ede9fe
classDef data      fill:#2a1040,stroke:#c084fc,color:#f3e8ff
classDef external  fill:#3b1030,stroke:#f472b6,color:#fce7f3
classDef observe   fill:#0d3328,stroke:#34d399,color:#d1fae5
\`\`\`

Assign every node to a class. No unstyled nodes.

### Grouping Rules
- Adapt the number of subgraphs to the actual content — could be 1 node with no subgraphs, could be 30 nodes across 6 layers. Don't force structure that isn't there.
- Only create a subgraph when there are 2+ related nodes that benefit from grouping.
- A workspace with a single repo is just a single node. That's fine. Don't pad it.
- As complexity grows, common layers emerge naturally:
  - Clients (apps, web, admin)
  - Gateway / Ingress (LB, API gateway, auth)
  - Services (core business logic)
  - Data (DBs, caches, queues, object stores)
  - External (3rd party APIs, email, payments)
  - Observability (logging, metrics, tracing)
- Use these as inspiration, not a checklist. If a workspace is just 3 backend services and a database, that's 4 nodes and maybe 1 subgraph or none. Let the content dictate the shape.

### Edge Labels
- Keep labels short: 2-3 words (\`read/write\`, \`publish event\`, \`HTTPS\`)
- Use \`-->\` for sync, \`-.->\` for async/optional if relevant

### Syntax Rules
- Every \`subgraph\` MUST have a matching \`end\` keyword on its own line
- Every edge must have exactly one source and one target: \`A -->|"label"| B\`. Never chain multiple labels or arrows in a single edge.
- Every node referenced in an edge or \`class\` assignment must be defined (e.g. \`NODE_ID["Label"]\`) inside a subgraph first
- Do not nest subgraphs — close each subgraph with \`end\` before opening the next

### Avoid
- Don't use \`style\` on individual nodes — only \`classDef\` + \`class\`
- Don't exceed ~30 nodes (split into multiple diagrams instead)
- Don't leave subgraphs with only 1 node
`

SKILLS["mermaid"] = mermaid;