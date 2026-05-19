// RunDetail — summary card + paginated call log for one run.
//
// Read-only; phase 9 grows the kill button and live-state panel
// (both of which depend on Redis hot state that phase 8 doesn't
// surface).

import { Link } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { getErrorMessage } from "../api/client";
import { useRunDetail } from "../api/queries";
import type { RunLogEntry } from "../api/types";

interface Props {
  runID: string;
}

const fmtUSD = (v: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  }).format(v);

const fmtInt = (v: number) =>
  new Intl.NumberFormat("en-US").format(Math.round(v));

const fmtTs = (s: string) => {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
};

export function RunDetail({ runID }: Props) {
  const q = useRunDetail(runID);

  const firstLog = q.data?.logs[0];
  const agent = firstLog?.metadata["agent-name"] ?? "—";
  const user = firstLog?.metadata["user-id"] ?? "—";

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">
            <Link href={`/agents/${encodeURIComponent(agent)}`}>{agent}</Link> /
            run
          </div>
          <h1 class="mono">{runID}</h1>
        </div>
      </div>

      {q.isError ? (
        <div class="error-banner">{getErrorMessage(q.error)}</div>
      ) : q.isLoading ? (
        <div class="loading">Loading…</div>
      ) : (
        <>
          <div class="kpi">
            <Kpi
              label="Total cost"
              value={fmtUSD(q.data?.stats.total_cost ?? 0)}
            />
            <Kpi
              label="Calls"
              value={fmtInt(q.data?.stats.total_requests ?? 0)}
            />
            <Kpi label="User" value={user} mono />
          </div>

          <section>
            <h2 style="margin-bottom: 16px">Call log</h2>
            <DataTable<RunLogEntry>
              rows={q.data?.logs ?? []}
              emptyMessage="No calls recorded for this run."
              columns={[
                {
                  key: "ts",
                  header: "Timestamp",
                  cell: (r) => <span class="mono">{fmtTs(r.timestamp)}</span>,
                  sort: (r) => r.timestamp,
                },
                {
                  key: "provider",
                  header: "Provider",
                  cell: (r) => r.provider,
                  sort: (r) => r.provider,
                },
                {
                  key: "model",
                  header: "Model",
                  cell: (r) => <span class="mono">{r.model}</span>,
                  sort: (r) => r.model,
                },
                {
                  key: "status",
                  header: "Status",
                  cell: (r) => (
                    <span
                      class={
                        r.status === "success" ? "text-ok" : "text-danger"
                      }
                    >
                      {r.status}
                    </span>
                  ),
                  sort: (r) => r.status,
                },
                {
                  key: "cost",
                  header: "Cost",
                  align: "num",
                  cell: (r) => fmtUSD(r.cost),
                  sort: (r) => r.cost,
                },
                {
                  key: "latency",
                  header: "Latency (ms)",
                  align: "num",
                  cell: (r) => fmtInt(r.latency),
                  sort: (r) => r.latency,
                },
              ]}
              defaultSortKey="ts"
            />
          </section>
        </>
      )}
    </>
  );
}

function Kpi({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div class="card kpi-card">
      <div class="kpi-label">{label}</div>
      <div class={"kpi-value" + (mono ? " mono" : "")}>{value}</div>
    </div>
  );
}
