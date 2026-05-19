// Dashboard — the "what is this swarm doing right now" page.
//
// Composition: three KPI cards, one cost histogram chart, and two
// top-5 ranking tables (agents, users). Window picker drives all
// queries through the same `Window` state.

import { useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { CostHistogram } from "../components/charts/CostHistogram";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { DataTable } from "../components/tables/DataTable";
import { WindowPicker } from "../components/controls/WindowPicker";
import { getErrorMessage } from "../api/client";
import {
  useHistogramCost,
  useSpendByAgent,
  useSpendByUser,
} from "../api/queries";
import type { AgentSpend, UserSpend, Window } from "../api/types";
import { windowToSeconds } from "../api/window";

// LLM call costs can be fractions of a cent; clamping to 2 decimals
// turns the entire dashboard into a wall of $0.00 during dev. Render
// down to 6 decimals when the value is below a cent, otherwise the
// usual two — matches what an operator wants to see in either regime.
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

export function Dashboard() {
  // The dashboard's headline KPIs use the same window as the
  // rankings; the chart bucket scales with window.
  const [window, setWindow] = useState<Window>("24h");
  const bucket = window === "1h" ? "5m" : window === "6h" ? "10m" : "1h";
  const windowSeconds = windowToSeconds(window);

  const agents = useSpendByAgent(window);
  const users = useSpendByUser(window);
  const histogram = useHistogramCost({
    window,
    bucket,
    dimension: "agent-name",
  });

  const totalCost =
    agents.data?.results.reduce((s, r) => s + r.total_cost, 0) ?? 0;
  const totalReq =
    agents.data?.results.reduce((s, r) => s + r.request_count, 0) ?? 0;

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">Overview</div>
          <h1>Dashboard</h1>
        </div>
        <WindowPicker value={window} onChange={setWindow} />
      </div>

      <div class="kpi">
        <Kpi label={`Spend (${window})`} value={fmtUSD(totalCost)} loading={agents.isLoading} />
        <Kpi label={`Requests (${window})`} value={fmtInt(totalReq)} loading={agents.isLoading} />
        <Kpi
          label="Distinct agents"
          value={fmtInt(agents.data?.results.length ?? 0)}
          loading={agents.isLoading}
        />
      </div>

      <section class="chart-frame">
        <div class="card-header">
          <div class="card-title">Cost by agent</div>
          <div class="text-dim mono" style="font-size: 11px">
            bucket: {bucket}
          </div>
        </div>
        {histogram.isError ? (
          <div class="error-banner">{getErrorMessage(histogram.error)}</div>
        ) : histogram.isLoading || !histogram.data ? (
          <div class="loading">Loading…</div>
        ) : (
          <ErrorBoundary>
            <CostHistogram data={histogram.data} windowSeconds={windowSeconds} />
          </ErrorBoundary>
        )}
      </section>

      <div class="grid-2">
        <section>
          <h2 style="margin-bottom: 16px">Top agents</h2>
          {agents.isError ? (
            <div class="error-banner">{getErrorMessage(agents.error)}</div>
          ) : (
            <DataTable<AgentSpend>
              rows={(agents.data?.results ?? []).slice(0, 5)}
              emptyMessage="No agent activity in this window."
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
        </section>

        <section>
          <h2 style="margin-bottom: 16px">Top users</h2>
          {users.isError ? (
            <div class="error-banner">{getErrorMessage(users.error)}</div>
          ) : (
            <DataTable<UserSpend>
              rows={(users.data?.results ?? []).slice(0, 5)}
              emptyMessage="No user activity in this window."
              columns={[
                {
                  key: "user",
                  header: "User",
                  cell: (r) => <span class="mono">{r.user_name || r.user_id}</span>,
                  sort: (r) => r.user_id,
                },
                {
                  key: "cost",
                  header: "Spend",
                  align: "num",
                  cell: (r) => fmtUSD(r.total_cost),
                  sort: (r) => r.total_cost,
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
        </section>
      </div>
    </>
  );
}

function Kpi({
  label,
  value,
  loading,
}: {
  label: string;
  value: string;
  loading: boolean;
}) {
  return (
    <div class="card kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{loading ? "…" : value}</div>
    </div>
  );
}
