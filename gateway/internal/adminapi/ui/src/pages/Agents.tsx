// Agents — full table of observed agents in the window.
//
// Phase 8 doesn't render kill switches, budget editors, or current-
// bucket spend (all phase 9). Just name + spend + tokens + calls,
// each row linking to AgentDetail.

import { useState } from "preact/hooks";
import { Link, useLocation } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { WindowPicker } from "../components/controls/WindowPicker";
import { getErrorMessage } from "../api/client";
import { useSpendByAgent } from "../api/queries";
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
              key: "tokens",
              header: "Tokens",
              align: "num",
              cell: (r) => fmtInt(r.total_tokens),
              sort: (r) => r.total_tokens,
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
