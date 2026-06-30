// Agents — full table of agents: the union of the neo4j registry
// (every catalog agent, traffic or not) and spend-derived agents
// (anything that produced calls in the window). A seeded agent with
// no traffic still shows; an agent with traffic but no catalog entry
// still shows (tagged "traffic only"). Each row links to AgentDetail.

import { useMemo, useState } from "preact/hooks";
import { Link, useLocation } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { WindowPicker } from "../components/controls/WindowPicker";
import { getErrorMessage } from "../api/client";
import {
  useAgentBudgets,
  useAgentCatalogList,
  useSpendByAgent,
} from "../api/queries";
import type { Window } from "../api/types";

// A merged agent row: spend metrics (zeroed when registry-only) plus
// catalog identity/counts (absent when traffic-only).
interface AgentRow {
  agent_name: string;
  total_cost: number;
  total_tokens: number;
  request_count: number;
  has_traffic: boolean;
  in_catalog: boolean;
  display_name?: string;
  default_model?: string;
  prompts: number;
  tools: number;
  skills: number;
}

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

export function Agents() {
  const [window, setWindow] = useState<Window>("24h");
  const [, setLocation] = useLocation();
  const q = useSpendByAgent(window);
  const catalog = useAgentCatalogList();

  // Union the registry with spend, keyed by name. Start from the
  // catalog (so every seeded agent always renders, even at $0), then
  // overlay spend; spend-only agents with no catalog entry get
  // appended and tagged `in_catalog: false`.
  const rows = useMemo<AgentRow[]>(() => {
    const byName = new Map<string, AgentRow>();
    for (const a of catalog.data?.agents ?? []) {
      byName.set(a.name, {
        agent_name: a.name,
        total_cost: 0,
        total_tokens: 0,
        request_count: 0,
        has_traffic: false,
        in_catalog: true,
        display_name: a.display_name,
        default_model: a.default_model,
        prompts: a.prompts,
        tools: a.tools,
        skills: a.skills,
      });
    }
    for (const s of q.data?.results ?? []) {
      const existing = byName.get(s.agent_name);
      if (existing) {
        existing.total_cost = s.total_cost;
        existing.total_tokens = s.total_tokens;
        existing.request_count = s.request_count;
        existing.has_traffic = true;
      } else {
        byName.set(s.agent_name, {
          agent_name: s.agent_name,
          total_cost: s.total_cost,
          total_tokens: s.total_tokens,
          request_count: s.request_count,
          has_traffic: true,
          in_catalog: false,
          prompts: 0,
          tools: 0,
          skills: 0,
        });
      }
    }
    return [...byName.values()];
  }, [catalog.data, q.data]);

  // Fan out per-agent budget queries. The list of names comes from
  // the merged rows; useMemo so the fan-out doesn't churn on every
  // render (which would resubscribe the per-row Tanstack queries and
  // reset their polling clocks).
  const agentNames = useMemo(() => rows.map((r) => r.agent_name), [rows]);
  const budgets = useAgentBudgets(agentNames);

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">Observability</div>
          <h1>Agents</h1>
        </div>
        <WindowPicker value={window} onChange={setWindow} />
      </div>

      {q.isError ? (
        <div class="error-banner">{getErrorMessage(q.error)}</div>
      ) : (
        <DataTable<AgentRow>
          rows={rows}
          emptyMessage="No agents in the registry, and none have produced calls in this window."
          onRowClick={(r) =>
            setLocation(`/agents/${encodeURIComponent(r.agent_name)}`)
          }
          columns={[
            {
              key: "agent",
              header: "Agent",
              cell: (r) => (
                <div class="agent-cell">
                  <Link href={`/agents/${encodeURIComponent(r.agent_name)}`}>
                    {r.agent_name}
                  </Link>
                  {!r.in_catalog ? (
                    <span class="pill" title="Has traffic but no catalog entry">
                      traffic only
                    </span>
                  ) : r.prompts + r.tools + r.skills > 0 ? (
                    <span class="cap-badges mono text-dim">
                      {r.prompts}p · {r.tools}t · {r.skills}s
                    </span>
                  ) : null}
                </div>
              ),
              sort: (r) => r.agent_name,
            },
            {
              key: "model",
              header: "Model",
              cell: (r) =>
                r.default_model ? (
                  <span class="mono text-dim">{r.default_model}</span>
                ) : (
                  <span class="text-dim">—</span>
                ),
              sort: (r) => r.default_model ?? "",
            },
            {
              key: "cost",
              header: "Spend",
              align: "num",
              cell: (r) =>
                r.has_traffic ? (
                  fmtUSD(r.total_cost)
                ) : (
                  <span class="text-dim">—</span>
                ),
              sort: (r) => r.total_cost,
            },
            {
              key: "budget",
              header: "Budget",
              align: "num",
              cell: (r) => {
                const b = budgets[r.agent_name];
                if (!b || b.cap_usd == null) {
                  return <span class="text-dim">—</span>;
                }
                return (
                  <span class="mono">
                    {fmtUSD(b.cap_usd)}
                    <span class="text-dim"> / {b.window}</span>
                  </span>
                );
              },
              // Sort by cap, treating "no budget" as -1 so they sink
              // to the bottom of a desc sort.
              sort: (r) => budgets[r.agent_name]?.cap_usd ?? -1,
            },
            {
              key: "usage",
              header: "Budget used",
              align: "num",
              cell: (r) => {
                const b = budgets[r.agent_name];
                if (!b || b.cap_usd == null || b.ratio == null) {
                  return <span class="text-dim">—</span>;
                }
                return <BudgetMeter ratio={b.ratio} />;
              },
              sort: (r) => budgets[r.agent_name]?.ratio ?? -1,
            },
            {
              key: "calls",
              header: "Calls",
              align: "num",
              cell: (r) =>
                r.has_traffic ? (
                  fmtInt(r.request_count)
                ) : (
                  <span class="text-dim">—</span>
                ),
              sort: (r) => r.request_count,
            },
          ]}
          defaultSortKey="agent"
          defaultSortOrder="asc"
        />
      )}
    </>
  );
}

// Tiny inline meter for the Agents list. The richer card-style
// progress bar lives on AgentDetail.
function BudgetMeter({ ratio }: { ratio: number }) {
  const pct = Math.min(100, Math.max(0, ratio * 100));
  const tone =
    ratio >= 1 ? "danger" : ratio >= 0.8 ? "warning" : "ok";
  return (
    <div class="budget-meter" title={`${(ratio * 100).toFixed(1)}%`}>
      <div class={`budget-meter-bar tone-${tone}`} style={`width:${pct}%`} />
      <span class="budget-meter-label mono">{Math.round(pct)}%</span>
    </div>
  );
}
