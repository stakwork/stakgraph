// RunDetail — provenance + summary + paginated call log for one run.
//
// Read-only; phase 9 grows the kill button and live-state panel
// (both of which depend on Redis hot state that phase 8 doesn't
// surface). Phase-8.5 adds the Provenance card, which is a pure
// frontend rearrangement of fields already on logs.metadata.

import { Link } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { getErrorMessage } from "../api/client";
import { useRunDetail } from "../api/queries";
import type { RunLogEntry } from "../api/types";

interface Props {
  runID: string;
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

const fmtTs = (s: string) => {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
};

// Relative-time helper for "issued 12 minutes ago". Used on the
// Provenance card to give the operator a feel for run age without
// having to do clock math.
function fmtRelative(absISO: string): string {
  try {
    const then = new Date(absISO).getTime();
    const now = Date.now();
    const sec = Math.round((now - then) / 1000);
    if (sec < 60) return `${sec}s ago`;
    const min = Math.round(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.round(min / 60);
    if (hr < 48) return `${hr}h ago`;
    const day = Math.round(hr / 24);
    return `${day}d ago`;
  } catch {
    return "";
  }
}

export function RunDetail({ runID }: Props) {
  const q = useRunDetail(runID);

  // The logs come back desc by timestamp (Bifrost default); first /
  // last give us the span of the run. All metadata fields are
  // populated identically across calls in a run (they come from the
  // same caller env / macaroon), so we read provenance from any row.
  const logs = q.data?.logs ?? [];
  const firstLog = logs[0]; // newest
  const lastLog = logs[logs.length - 1]; // oldest
  const md = firstLog?.metadata ?? {};

  const agent = md["agent-name"] ?? "—";
  const user = md["user-id"] ?? "—";
  const realm = md["realm-id"] ?? "—";
  const session = md["session-id"] ?? "";
  const deployment = md["deployment"] ?? "";

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

          {/* Provenance: the dim values stamped on every call in
              this run. Reads directly from logs.metadata — no
              extra backend round-trip. Once phase 6 lands, these
              values become cryptographically attested without any
              UI change. */}
          {firstLog ? (
            <section class="card provenance">
              <div class="card-header">
                <div class="card-title">Provenance</div>
                <div class="text-dim mono" style="font-size: 11px">
                  from logs.metadata
                </div>
              </div>
              <dl class="kvgrid">
                <ProvField label="Agent">
                  <Link href={`/agents/${encodeURIComponent(agent)}`}>
                    <span class="mono">{agent}</span>
                  </Link>
                </ProvField>
                <ProvField label="User">
                  <span class="mono">{user}</span>
                </ProvField>
                <ProvField label="Workspace">
                  <span class="mono">{realm}</span>
                </ProvField>
                {session ? (
                  <ProvField label="Session">
                    <span class="mono">{session}</span>
                  </ProvField>
                ) : null}
                {deployment ? (
                  <ProvField label="Deployment">
                    <span class="mono">{deployment}</span>
                  </ProvField>
                ) : null}
                {lastLog ? (
                  <ProvField label="First seen">
                    <span class="mono">{fmtTs(lastLog.timestamp)}</span>{" "}
                    <span class="text-dim">
                      ({fmtRelative(lastLog.timestamp)})
                    </span>
                  </ProvField>
                ) : null}
                {firstLog ? (
                  <ProvField label="Last seen">
                    <span class="mono">{fmtTs(firstLog.timestamp)}</span>{" "}
                    <span class="text-dim">
                      ({fmtRelative(firstLog.timestamp)})
                    </span>
                  </ProvField>
                ) : null}
              </dl>
            </section>
          ) : null}

          <section>
            <h2 style="margin-bottom: 16px">Call log</h2>
            <DataTable<RunLogEntry>
              rows={logs}
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

function ProvField({
  label,
  children,
}: {
  label: string;
  children: any;
}) {
  return (
    <>
      <dt class="kvgrid-key">{label}</dt>
      <dd class="kvgrid-val">{children}</dd>
    </>
  );
}
