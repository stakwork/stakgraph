// Agents — full table of observed agents in the window.
//
// Phase 8 doesn't render kill switches, budget editors, or current-
// bucket spend (all phase 9). Just name + spend + tokens + calls,
// each row linking to AgentDetail.

import { useMemo, useState } from "preact/hooks";
import { Link, useLocation } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { WindowPicker } from "../components/controls/WindowPicker";
import { getErrorMessage } from "../api/client";
import { useAgentBudgets, useSpendByAgent } from "../api/queries";
import type { AgentSpend, Window } from "../api/types";

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

  // Fan out per-agent budget queries. The list of names comes from
  // the by-agent query; useMemo so the fan-out doesn't churn on
  // every render (which would resubscribe the per-row Tanstack
  // queries and reset their polling clocks).
  const agentNames = useMemo(
    () => (q.data?.results ?? []).map((r) => r.agent_name),
    [q.data]
  );
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
        <DataTable<AgentSpend>
          rows={q.data?.results ?? []}
          emptyMessage="No agents have produced calls in this window."
          onRowClick={(r) =>
            setLocation(`/agents/${encodeURIComponent(r.agent_name)}`)
          }
          columns={[
            {
              key: "agent",
              header: "Agent",
              cell: (r) => (
                <Link href={`/agents/${encodeURIComponent(r.agent_name)}`}>
                  {r.agent_name}
                </Link>
              ),
              sort: (r) => r.agent_name,
            },
            {
              key: "cost",
              header: "Spend",
              align: "num",
              cell: (r) => fmtUSD(r.total_cost),
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
              cell: (r) => fmtInt(r.request_count),
              sort: (r) => r.request_count,
            },
          ]}
          defaultSortKey="cost"
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
