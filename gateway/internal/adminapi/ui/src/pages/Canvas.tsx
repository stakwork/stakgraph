// Canvas — the swarm's flowchart page.
//
// Layout: four vertical columns rendered with system-canvas.
//
//   ┌─ agents ─┐   ┌─ users ─┐   ┌─ gateway ─┐   ┌─ providers ─┐
//   │ small    │   │ medium  │   │ singleton │   │ medium      │
//   │ N per    │   │ 1 per   │   │ swarm hub │   │ 1 per       │
//   │ (agent,  │   │ user    │   │           │   │ provider    │
//   │  user)   │   │         │   │           │   │             │
//   └──────────┘   └─────────┘   └───────────┘   └─────────────┘
//
// Per-column data:
//   - Agents: one node per (agent × user) pair from
//     `/_plugin/spend/by-agent-user`. customData.cost = that
//     pairing's total spend in the window.
//   - Users: one node per distinct user. customData.totalCost +
//     requestCount are summed over all that user's pairings.
//   - Gateway: singleton. customData.totalCost = swarm-wide total
//     (sum of every row's cost).
//   - Providers: hardcoded 4 — anthropic / openai / openrouter /
//     google — pulled from gateway/docker-compose.yml env. Real
//     per-provider spend lands when the matrix endpoint grows a
//     provider dimension.
//
// The canvas takes over the full shell-main area; the WindowPicker
// floats in the top-right corner as the only chrome. No numeric
// table — pop open devtools if you need to verify raw rows.

import { useMemo, useState } from "preact/hooks";
import type { CanvasData, CanvasEdge, CanvasNode } from "system-canvas";
import { SystemCanvas } from "system-canvas-react";

import { useSpendByAgentUser } from "../api/queries";
import type { AgentUserSpend, Window } from "../api/types";
import { WindowPicker } from "../components/controls/WindowPicker";

import { canvasTheme, PROVIDER_DISPLAY, providerIcon } from "./canvasTheme";

// Currency formatter — duplicated from Dashboard.tsx per AGENTS.md
// "Currency formatting"; cheap helper, not worth a util module.
const fmtUSD = (v: number) => {
  if (v === 0) return "$0.00";
  const digits = Math.abs(v) < 0.01 ? 6 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
};

const fmtInt = (v: number) =>
  new Intl.NumberFormat("en-US").format(Math.round(v));

// Hardcoded provider list — these ids match Bifrost's `provider`
// column on every log row (`gateway/data/config.json`'s `providers`
// map), so the per-provider rollup from the matrix endpoint
// dispatches against `customData.icon` cleanly. The order here is
// the vertical render order in the providers column.
const PROVIDERS = ["anthropic", "openai", "openrouter", "gemini"];

// Column x-coordinates (canvas-space, centered around 0). Spacing
// picked so even the widest columns (gateway 220) don't touch their
// neighbors at the lib's default zoom. Centers chosen so the gaps
// between right-edge of one column and left-edge of the next are
// ~80px.
const COL_X = {
  agent: -540,
  user: -280,
  gateway: 40,
  provider: 360,
} as const;

// Per-column node dimensions — kept in sync with the demo
// `gateway.ts` (system-diagram/demo/src/gateway.ts). Don't drift.
const SIZE = {
  agent: { w: 160, h: 52, gap: 10 },
  user: { w: 180, h: 64, gap: 16 },
  gateway: { w: 220, h: 100 },
  provider: { w: 180, h: 64, gap: 16 },
} as const;

// Stack a column of nodes vertically and center the whole stack
// around y=0. Returns the y for the i-th item.
function stackY(count: number, i: number, h: number, gap: number): number {
  const totalH = count * h + (count - 1) * gap;
  const startY = -totalH / 2;
  return startY + i * (h + gap) + h / 2; // node centers at this y
}

// Build the four-column canvas from the live matrix. Pure — easy
// to unit-test later if we want.
function buildCanvas(rows: AgentUserSpend[]): CanvasData {
  // Roll up by user (for the User column) and collect all
  // (agent, user) pairs (for the Agent column). Walking the rows
  // once produces both views.
  const userTotals = new Map<
    string,
    { totalCost: number; requestCount: number; userName: string }
  >();
  let swarmTotal = 0;
  // Per-provider roll-up across all (agent, user) pairs. The matrix
  // endpoint hands us one `providers[]` per row already aggregated by
  // (agent, user, provider); we just fan them up to the gateway-wide
  // band so we can size the gateway→provider edges and feed each
  // provider card its real spend.
  const providerTotals = new Map<
    string,
    { totalCost: number; requestCount: number }
  >();
  for (const r of rows) {
    swarmTotal += r.total_cost;
    const u = userTotals.get(r.user_id);
    if (u) {
      u.totalCost += r.total_cost;
      u.requestCount += r.request_count;
    } else {
      userTotals.set(r.user_id, {
        totalCost: r.total_cost,
        requestCount: r.request_count,
        userName: r.user_name || r.user_id,
      });
    }
    for (const p of r.providers ?? []) {
      const existing = providerTotals.get(p.provider);
      if (existing) {
        existing.totalCost += p.total_cost;
        existing.requestCount += p.request_count;
      } else {
        providerTotals.set(p.provider, {
          totalCost: p.total_cost,
          requestCount: p.request_count,
        });
      }
    }
  }

  const users = Array.from(userTotals.entries());
  // Sort: highest spend at the top. The matrix endpoint already
  // sorts pairs by cost desc, so iterating `rows` in order gives
  // us agent boxes top-to-bottom by spend.
  users.sort((a, b) => b[1].totalCost - a[1].totalCost);

  const nodes: CanvasNode[] = [];

  // ─── Agents column (one per pairing) ──────────────────────────
  // Name flows through the header text slot (driven by
  // customData.name), so node.text is left empty — that suppresses
  // the default label render which would otherwise double up.
  rows.forEach((r, i) => {
    nodes.push({
      id: `agent:${r.agent_name}:${r.user_id}`,
      type: "text",
      category: "agent",
      text: "",
      x: COL_X.agent,
      y: stackY(rows.length, i, SIZE.agent.h, SIZE.agent.gap),
      width: SIZE.agent.w,
      height: SIZE.agent.h,
      customData: {
        name: r.agent_name,
        cost: r.total_cost,
        calls: r.request_count,
      },
    });
  });

  // ─── Users column ─────────────────────────────────────────────
  // Username flows through the header text slot (driven by
  // customData.name); node.text empty so the default label
  // doesn't double-render.
  users.forEach(([userID, agg], i) => {
    nodes.push({
      id: `user:${userID}`,
      type: "text",
      category: "user",
      text: "",
      x: COL_X.user,
      y: stackY(users.length, i, SIZE.user.h, SIZE.user.gap),
      width: SIZE.user.w,
      height: SIZE.user.h,
      customData: {
        name: agg.userName,
        totalCost: agg.totalCost,
        requestCount: agg.requestCount,
      },
    });
  });

  // ─── Gateway singleton ────────────────────────────────────────
  nodes.push({
    id: "gateway",
    type: "text",
    category: "gateway",
    text: "Agent Mothership", // suppressed by the `body` slot, but
                           // kept so screen readers + DOM
                           // inspectors still see a label.
    x: COL_X.gateway,
    y: 0,
    width: SIZE.gateway.w,
    height: SIZE.gateway.h,
    customData: { totalCost: swarmTotal },
  });

  // ─── Providers column ─────────────────────────────────────────
  // - `node.color` = the per-provider brand-ish hex (drives stroke
  //   + rightEdge stripe inheritance).
  // - `node.text = ""` because the provider name flows through the
  //   header text slot (customData.name).
  // - `customData.totalCost` / `requestCount` come from the
  //   provider-rollup we computed above; consumed by the provider
  //   drawer and any future provider-card slot.
  PROVIDERS.forEach((id, i) => {
    const display = PROVIDER_DISPLAY[id];
    const totals = providerTotals.get(id);
    nodes.push({
      id: `provider:${id}`,
      type: "text",
      category: "provider",
      text: "",
      x: COL_X.provider,
      y: stackY(PROVIDERS.length, i, SIZE.provider.h, SIZE.provider.gap),
      width: SIZE.provider.w,
      height: SIZE.provider.h,
      color: display?.color,
      customData: {
        name: display?.label ?? id,
        icon: providerIcon(id),
        totalCost: totals?.totalCost ?? 0,
        requestCount: totals?.requestCount ?? 0,
      },
    });
  });

  // ─── Edges ────────────────────────────────────────────────────
  // Three bands of connectivity, all running left→right since that
  // matches the columns:
  //   1. agent → user (one per pairing row)
  //   2. user  → gateway (one per distinct user)
  //   3. gateway → provider (one per provider, weighted by that
  //      provider's swarm-wide spend)
  // fromSide/toSide pin the endpoints to the column-facing edges so
  // the routing doesn't loop around the node bodies.
  //
  // Stroke width encodes proportional cost within each band. We
  // normalize per-band (so the thickest agent→user edge and the
  // thickest user→gateway edge both render at EDGE_W_MAX), which
  // keeps the agent column visible — without per-band normalization,
  // the user→gateway lines (each carrying a sum of many pairs) would
  // dwarf every individual pair line and squash them flat.
  const edges: CanvasEdge[] = [];
  const maxRowCost = rows.reduce((m, r) => Math.max(m, r.total_cost), 0);
  const maxUserCost = Array.from(userTotals.values()).reduce(
    (m, u) => Math.max(m, u.totalCost),
    0,
  );
  const maxProviderCost = Array.from(providerTotals.values()).reduce(
    (m, p) => Math.max(m, p.totalCost),
    0,
  );
  rows.forEach((r) => {
    edges.push({
      id: `e:agent-user:${r.agent_name}:${r.user_id}`,
      fromNode: `agent:${r.agent_name}:${r.user_id}`,
      fromSide: "right",
      toNode: `user:${r.user_id}`,
      toSide: "left",
      toEnd: "none",
      strokeWidth: weightToWidth(r.total_cost, maxRowCost),
    });
  });
  users.forEach(([userID, agg]) => {
    edges.push({
      id: `e:user-gateway:${userID}`,
      fromNode: `user:${userID}`,
      fromSide: "right",
      toNode: "gateway",
      toSide: "left",
      toEnd: "none",
      strokeWidth: weightToWidth(agg.totalCost, maxUserCost),
    });
  });
  PROVIDERS.forEach((id) => {
    const cost = providerTotals.get(id)?.totalCost ?? 0;
    edges.push({
      id: `e:gateway-provider:${id}`,
      fromNode: "gateway",
      fromSide: "right",
      toNode: `provider:${id}`,
      toSide: "left",
      toEnd: "none",
      strokeWidth: weightToWidth(cost, maxProviderCost),
    });
  });

  return { nodes, edges };
}

// Map a per-band weight to a stroke width in canvas-space px.
// Linear in (weight/max). Zero-cost edges render at EDGE_W_MIN so
// they stay visible — operators want to see "there's a link here
// that happens to be quiet" rather than nothing at all.
const EDGE_W_MIN = 1;
const EDGE_W_MAX = 8;
function weightToWidth(weight: number, max: number): number {
  if (max <= 0) return EDGE_W_MIN;
  const t = Math.max(0, Math.min(1, weight / max));
  return EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * t;
}

export function Canvas() {
  const [window, setWindow] = useState<Window>("24h");
  const matrix = useSpendByAgentUser(window);
  // Selected node id drives the right-drawer. We store the id (not
  // the node object) so a data refetch — which produces fresh node
  // objects — doesn't strand the drawer on a stale reference.
  const [selectedID, setSelectedID] = useState<string | null>(null);

  const rows = matrix.data?.results ?? [];
  const canvas = useMemo(() => buildCanvas(rows), [rows]);

  const selectedNode = selectedID
    ? canvas.nodes?.find((n) => n.id === selectedID) ?? null
    : null;

  // Negative margins cancel `.shell-main`'s padding so the canvas
  // bleeds edge-to-edge inside the shell. `position: relative` here
  // anchors the absolutely-positioned WindowPicker overlay.
  return (
    <div
      style="position: relative; height: 100%; margin: calc(-1 * var(--sp-5)) calc(-1 * var(--sp-6));"
    >
      <SystemCanvas
        canvas={canvas}
        theme={canvasTheme}
        onNodeClick={(n) => setSelectedID(n.id)}
      />
      <div style="position: absolute; top: var(--sp-4); right: var(--sp-5); z-index: 1;">
        <WindowPicker value={window} onChange={setWindow} />
      </div>
      {selectedNode ? (
        <NodeDrawer
          node={selectedNode}
          rows={rows}
          onClose={() => setSelectedID(null)}
        />
      ) : null}
    </div>
  );
}

// ─── NodeDrawer ──────────────────────────────────────────────────
// Right-side overlay surfaced by clicking a canvas node. Reuses the
// `.drawer*` styles RunDetail already defined — same visual language
// for "you clicked a thing, here's its detail panel".
//
// Body shape branches on `node.category`:
//   - agent    → per-pairing spend + revoke action
//   - user     → roll-up across all their agents + suspend action
//   - gateway  → swarm totals (no actions — it's not a target)
//   - provider → placeholder pending real per-provider spend
//
// All actions today are placeholders: they log and toast. Real
// endpoints (POST /_plugin/users/:id/disable, etc) land separately
// — see AGENTS.md for the auth model these will plug into.
function NodeDrawer({
  node,
  rows,
  onClose,
}: {
  node: CanvasNode;
  rows: AgentUserSpend[];
  onClose: () => void;
}) {
  return (
    <>
      <div class="drawer-backdrop" onClick={onClose} />
      <aside class="drawer" role="dialog" aria-label="Node detail">
        <header class="drawer-header">
          <div>
            <div class="drawer-eyebrow">{drawerEyebrow(node)}</div>
            <div class="drawer-title">{drawerTitle(node)}</div>
          </div>
          <button class="drawer-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div class="drawer-body">
          <NodeDrawerBody node={node} rows={rows} />
        </div>
      </aside>
    </>
  );
}

function drawerEyebrow(node: CanvasNode): string {
  switch (node.category) {
    case "agent":
      return "Agent × User";
    case "user":
      return "User";
    case "gateway":
      return "Gateway";
    case "provider":
      return "Provider";
    default:
      return "Node";
  }
}

function drawerTitle(node: CanvasNode): string {
  const cd = (node.customData ?? {}) as { name?: string };
  return cd.name ?? "Agent Mothership";
}

function NodeDrawerBody({
  node,
  rows,
}: {
  node: CanvasNode;
  rows: AgentUserSpend[];
}) {
  switch (node.category) {
    case "agent":
      return <AgentBody node={node} />;
    case "user":
      return <UserBody node={node} rows={rows} />;
    case "gateway":
      return <GatewayBody node={node} rows={rows} />;
    case "provider":
      return <ProviderBody node={node} rows={rows} />;
    default:
      return null;
  }
}

// Stat ribbon used at the top of each body. Mirrors the
// `.drawer-meta-grid` pattern used by RunDetail's call drawer.
function MetaGrid({ items }: { items: { label: string; value: string }[] }) {
  return (
    <div class="drawer-meta-grid">
      {items.map((it) => (
        <div key={it.label}>
          <div class="drawer-meta-label">{it.label}</div>
          <div class="drawer-meta-value">{it.value}</div>
        </div>
      ))}
    </div>
  );
}

// Placeholder action button row. Each button logs to console and
// renders a transient confirmation. When the real endpoints land,
// swap `onClick` for a mutation hook (see queries.ts patterns).
function ActionRow({
  actions,
}: {
  actions: { label: string; danger?: boolean; onClick: () => void }[];
}) {
  return (
    <div style="display: flex; flex-wrap: wrap; gap: var(--sp-2);">
      {actions.map((a) => (
        <button
          key={a.label}
          type="button"
          class="btn"
          style={a.danger ? "border-color: var(--danger); color: var(--danger);" : undefined}
          onClick={a.onClick}
        >
          {a.label}
        </button>
      ))}
    </div>
  );
}

// Placeholder action handler. Centralized so the swap-out (when
// real endpoints land) is a one-line edit.
function todoAction(label: string, target: string) {
  // eslint-disable-next-line no-console
  console.warn(`[canvas] TODO: ${label} on ${target}`);
  // window.alert keeps the UX honest until a real toast lands —
  // operators see immediately that nothing actually happened yet.
  window.alert(`TODO: wire ${label} → ${target}`);
}

function AgentBody({ node }: { node: CanvasNode }) {
  const cd = (node.customData ?? {}) as {
    name?: string;
    cost?: number;
    calls?: number;
  };
  // Agent nodes are scoped per (agent, user) pair; the id encodes
  // that — parse it back out so the action row can describe the
  // exact target.
  // id shape: `agent:<agent_name>:<user_id>`
  const [, agentName, userID] = node.id.split(":");
  const target = `agent=${agentName} user=${userID}`;
  return (
    <>
      <MetaGrid
        items={[
          { label: "Agent", value: cd.name ?? agentName ?? "—" },
          { label: "User", value: userID ?? "—" },
          { label: "Spend", value: fmtUSD(cd.cost ?? 0) },
          { label: "Calls", value: fmtInt(cd.calls ?? 0) },
        ]}
      />
      <div class="drawer-section">
        <div class="drawer-section-title">Actions</div>
        <ActionRow
          actions={[
            {
              label: "Revoke agent access for this user",
              danger: true,
              onClick: () => todoAction("revoke-agent-user", target),
            },
            {
              label: "Set per-pair budget",
              onClick: () => todoAction("set-pair-budget", target),
            },
          ]}
        />
      </div>
    </>
  );
}

function UserBody({
  node,
  rows,
}: {
  node: CanvasNode;
  rows: AgentUserSpend[];
}) {
  const cd = (node.customData ?? {}) as {
    name?: string;
    totalCost?: number;
    requestCount?: number;
  };
  const userID = node.id.slice("user:".length);
  const agents = rows
    .filter((r) => r.user_id === userID)
    .sort((a, b) => b.total_cost - a.total_cost);

  return (
    <>
      <MetaGrid
        items={[
          { label: "User", value: cd.name ?? userID },
          { label: "User ID", value: userID },
          { label: "Total spend", value: fmtUSD(cd.totalCost ?? 0) },
          { label: "Calls", value: fmtInt(cd.requestCount ?? 0) },
        ]}
      />
      <div class="drawer-section">
        <div class="drawer-section-title">Agents used</div>
        {agents.length === 0 ? (
          <div class="text-dim">No activity in this window.</div>
        ) : (
          <ul style="margin: 0; padding: 0; list-style: none;">
            {agents.map((r) => (
              <li
                key={r.agent_name}
                style="display: flex; justify-content: space-between; padding: var(--sp-2) 0; border-bottom: 1px solid var(--border);"
              >
                <span>{r.agent_name}</span>
                <span class="mono text-dim">{fmtUSD(r.total_cost)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div class="drawer-section">
        <div class="drawer-section-title">Actions</div>
        <ActionRow
          actions={[
            {
              label: "Suspend user",
              danger: true,
              onClick: () => todoAction("suspend-user", userID),
            },
            {
              label: "Revoke all agent access",
              danger: true,
              onClick: () => todoAction("revoke-all-agents", userID),
            },
            {
              label: "Set user budget",
              onClick: () => todoAction("set-user-budget", userID),
            },
          ]}
        />
      </div>
    </>
  );
}

function GatewayBody({
  node,
  rows,
}: {
  node: CanvasNode;
  rows: AgentUserSpend[];
}) {
  const cd = (node.customData ?? {}) as { totalCost?: number };
  const totalCalls = rows.reduce((s, r) => s + r.request_count, 0);
  const userCount = new Set(rows.map((r) => r.user_id)).size;
  const agentCount = new Set(rows.map((r) => r.agent_name)).size;
  return (
    <>
      <MetaGrid
        items={[
          { label: "Total spend", value: fmtUSD(cd.totalCost ?? 0) },
          { label: "Total calls", value: fmtInt(totalCalls) },
          { label: "Active users", value: fmtInt(userCount) },
          { label: "Active agents", value: fmtInt(agentCount) },
        ]}
      />
    </>
  );
}

function ProviderBody({
  node,
  rows,
}: {
  node: CanvasNode;
  rows: AgentUserSpend[];
}) {
  const cd = (node.customData ?? {}) as {
    name?: string;
    totalCost?: number;
    requestCount?: number;
  };
  const providerID = node.id.slice("provider:".length);

  // Top spenders for this provider — flatten the matrix to
  // (agent, user, providerSlice) tuples, keep only ones touching
  // this provider, sort by cost. Bounded to the matrix's 200k-row
  // ceiling, so this is cheap even for busy swarms.
  type TopRow = { agent: string; user: string; cost: number; calls: number };
  const topPairs: TopRow[] = [];
  for (const r of rows) {
    for (const p of r.providers ?? []) {
      if (p.provider !== providerID) continue;
      topPairs.push({
        agent: r.agent_name,
        user: r.user_name || r.user_id,
        cost: p.total_cost,
        calls: p.request_count,
      });
    }
  }
  topPairs.sort((a, b) => b.cost - a.cost);

  return (
    <>
      <MetaGrid
        items={[
          { label: "Provider", value: cd.name ?? providerID },
          { label: "Provider ID", value: providerID },
          { label: "Total spend", value: fmtUSD(cd.totalCost ?? 0) },
          { label: "Calls", value: fmtInt(cd.requestCount ?? 0) },
        ]}
      />
      <div class="drawer-section">
        <div class="drawer-section-title">Top agent × user pairs</div>
        {topPairs.length === 0 ? (
          <div class="text-dim">No activity for this provider in this window.</div>
        ) : (
          <ul style="margin: 0; padding: 0; list-style: none;">
            {topPairs.slice(0, 8).map((p) => (
              <li
                key={`${p.agent}\x00${p.user}`}
                style="display: flex; justify-content: space-between; padding: var(--sp-2) 0; border-bottom: 1px solid var(--border);"
              >
                <span>
                  {p.agent}
                  <span class="text-dim"> · {p.user}</span>
                </span>
                <span class="mono text-dim">{fmtUSD(p.cost)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div class="drawer-section">
        <div class="drawer-section-title">Actions</div>
        <ActionRow
          actions={[
            {
              label: "Disable provider",
              danger: true,
              onClick: () => todoAction("disable-provider", providerID),
            },
            {
              label: "Rotate API key",
              onClick: () => todoAction("rotate-provider-key", providerID),
            },
          ]}
        />
      </div>
    </>
  );
}
