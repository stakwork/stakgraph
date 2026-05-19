// AgentDetail — per-agent cost histogram + recent runs.
//
// Phase-8 limitation
// ------------------
// The backend's /_plugin/histogram/cost endpoint doesn't yet accept
// a per-agent metadata filter; it returns one series per agent
// across the whole window. We work around that here by filtering
// the response client-side to just this agent's series. A future
// backend revision can add a server-side filter without touching
// this page.

import { useMemo, useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { CostHistogram } from "../components/charts/CostHistogram";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { WindowPicker } from "../components/controls/WindowPicker";
import { getErrorMessage } from "../api/client";
import { useHistogramCost } from "../api/queries";
import type { HistogramCostResponse, Window } from "../api/types";
import { windowToSeconds } from "../api/window";

interface Props {
  name: string;
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

export function AgentDetail({ name }: Props) {
  const [window, setWindow] = useState<Window>("24h");
  const bucket = window === "1h" ? "5m" : window === "6h" ? "10m" : "1h";
  const windowSeconds = windowToSeconds(window);

  const histogram = useHistogramCost({
    window,
    bucket,
    dimension: "agent-name",
  });

  // Filter the histogram down to just this agent's series.
  const filtered = useMemo<HistogramCostResponse | undefined>(() => {
    if (!histogram.data) return undefined;
    return {
      ...histogram.data,
      series: histogram.data.series.filter((s) => s.dimension_value === name),
    };
  }, [histogram.data, name]);

  // Pull run-ids from the same window to derive the "recent runs"
  // table. We don't have a dedicated `/spend/by-run` endpoint yet,
  // so we use the agent dimension histogram to surface activity and
  // a separate dimension histogram by run-id for the same window.
  const runsHistogram = useHistogramCost({
    window,
    bucket,
    dimension: "run-id",
  });
  const recentRuns = useMemo(() => {
    if (!runsHistogram.data) return [];
    // Sort by total cost desc and cap to top 50 — the dashboard plan
    // commits to "first 100 runs in window"; 50 keeps render cheap.
    return [...runsHistogram.data.series]
      .map((s) => ({
        run_id: s.dimension_value,
        cost: s.points.reduce((acc, p) => acc + p.cost, 0),
        calls: s.points.length, // approximate; chart points are by bucket
      }))
      .sort((a, b) => b.cost - a.cost)
      .slice(0, 50);
  }, [runsHistogram.data]);

  const totalCost = filtered?.series[0]?.points.reduce((s, p) => s + p.cost, 0) ?? 0;

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">
            <Link href="/agents">Agents</Link> / {name}
          </div>
          <h1 class="mono">{name}</h1>
        </div>
        <WindowPicker value={window} onChange={setWindow} />
      </div>

      <div class="kpi">
        <div class="card kpi-card">
          <div class="kpi-label">Spend ({window})</div>
          <div class="kpi-value">{fmtUSD(totalCost)}</div>
        </div>
      </div>

      <section class="chart-frame">
        <div class="card-header">
          <div class="card-title">Cost over time</div>
          <div class="text-dim mono" style="font-size: 11px">
            bucket: {bucket}
          </div>
        </div>
        {histogram.isError ? (
          <div class="error-banner">{getErrorMessage(histogram.error)}</div>
        ) : filtered ? (
          <ErrorBoundary>
            <CostHistogram data={filtered} windowSeconds={windowSeconds} />
          </ErrorBoundary>
        ) : (
          <div class="loading">Loading…</div>
        )}
      </section>

      <section>
        <h2 style="margin-bottom: 16px">Recent runs</h2>
        {recentRuns.length === 0 ? (
          <div class="empty">No runs in this window.</div>
        ) : (
          <div class="table-wrap">
            <table class="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th class="num">Spend</th>
                  <th class="num">Buckets seen</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((r) => (
                  <tr key={r.run_id} class="row-link">
                    <td>
                      <Link href={`/runs/${encodeURIComponent(r.run_id)}`}>
                        <span class="mono">{r.run_id}</span>
                      </Link>
                    </td>
                    <td class="num">{fmtUSD(r.cost)}</td>
                    <td class="num">{fmtInt(r.calls)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
