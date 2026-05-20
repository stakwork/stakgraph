// UserDetail — KPIs + cost histogram + agents used + recent runs,
// all scoped to one user's traffic in the window.
//
// Data flow
// ---------
// One query (`useUserDetail`) drives the KPI cards and the two
// tables. A separate `useHistogramCost` call with `userID` filters
// the time-series to the same window — that runs server-side on
// the backend so no client-side filtering is needed.
//
// Order on the page is intentional: KPIs (the "is this person a
// heavy user?" answer), Provenance-y card with first/last seen,
// chart, then the two drill-down tables. Same shape as AgentDetail
// so an operator who learns one learns the other.

import { useMemo, useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { CostHistogram } from "../components/charts/CostHistogram";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { WindowPicker } from "../components/controls/WindowPicker";
import { BotIcon, UserIcon } from "../components/icons";
import { getErrorMessage } from "../api/client";
import { useHistogramCost, useUserDetail } from "../api/queries";
import type { UserAgentUsage, UserRunSummary, Window } from "../api/types";
import { windowToSeconds } from "../api/window";

interface Props {
  userID: string;
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

const fmtTs = (s?: string) => {
  if (!s) return "";
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
};

function fmtRelative(absISO?: string): string {
  if (!absISO) return "";
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

export function UserDetail({ userID }: Props) {
  const [window, setWindow] = useState<Window>("24h");
  const bucket = window === "1h" ? "5m" : window === "6h" ? "10m" : "1h";
  const windowSeconds = windowToSeconds(window);

  const q = useUserDetail(userID, window);
  // Per-user histogram, scoped server-side. `dimension=agent-name`
  // so the chart breaks the user's spend down by which agent they
  // used — most informative slice for the "what is this person
  // doing?" question.
  const histogram = useHistogramCost({
    window,
    bucket,
    dimension: "agent-name",
    userID,
  });

  const agentsUsed = q.data?.agents_used ?? [];
  const recentRuns = q.data?.recent_runs ?? [];

  // Stable identity to render in the page header — fall back to
  // the URL slug if the response hasn't arrived yet.
  const displayID = q.data?.user_id ?? userID;
  const distinctAgents = useMemo(
    () => new Set(agentsUsed.map((a) => a.agent_name)).size,
    [agentsUsed]
  );

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">
            <Link href="/people">People</Link> / {displayID}
          </div>
          <h1 class="mono">
            <span class="prov-with-icon">
              <UserIcon class="prov-icon" />
              {displayID}
            </span>
          </h1>
        </div>
        <WindowPicker value={window} onChange={setWindow} />
      </div>

      {q.isError ? (
        <div class="error-banner">{getErrorMessage(q.error)}</div>
      ) : (
        <>
          <div class="kpi">
            <div class="card kpi-card">
              <div class="kpi-label">Spend ({window})</div>
              <div class="kpi-value">{fmtUSD(q.data?.total_cost ?? 0)}</div>
            </div>
            <div class="card kpi-card">
              <div class="kpi-label">Calls</div>
              <div class="kpi-value">{fmtInt(q.data?.request_count ?? 0)}</div>
            </div>
            <div class="card kpi-card">
              <div class="kpi-label">Distinct agents</div>
              <div class="kpi-value">{fmtInt(distinctAgents)}</div>
            </div>
          </div>

          {/* Lightweight activity card — first/last seen plus
              identity. Same role as the Provenance card on
              RunDetail but scoped to the user, not one run. */}
          {q.data ? (
            <section class="card provenance">
              <div class="card-header">
                <div class="card-title">Activity</div>
                <div class="text-dim mono" style="font-size: 11px">
                  scoped to {window}
                </div>
              </div>
              <dl class="kvgrid">
                <dt class="kvgrid-key">First seen</dt>
                <dd class="kvgrid-val">
                  <span class="mono">{fmtTs(q.data.first_seen)}</span>{" "}
                  <span class="text-dim">
                    {q.data.first_seen
                      ? `(${fmtRelative(q.data.first_seen)})`
                      : ""}
                  </span>
                </dd>
                <dt class="kvgrid-key">Last seen</dt>
                <dd class="kvgrid-val">
                  <span class="mono">{fmtTs(q.data.last_seen)}</span>{" "}
                  <span class="text-dim">
                    {q.data.last_seen
                      ? `(${fmtRelative(q.data.last_seen)})`
                      : ""}
                  </span>
                </dd>
              </dl>
            </section>
          ) : null}

          <section class="chart-frame">
            <div class="card-header">
              <div class="card-title">Cost by agent (this user)</div>
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
                <CostHistogram
                  data={histogram.data}
                  windowSeconds={windowSeconds}
                />
              </ErrorBoundary>
            )}
          </section>

          <section style="margin-bottom: 24px">
            <h2 style="margin-bottom: 16px">Agents used</h2>
            {agentsUsed.length === 0 ? (
              <div class="empty">No agent activity in this window.</div>
            ) : (
              <div class="table-wrap">
                <table class="table">
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th class="num">Spend</th>
                      <th class="num">Calls</th>
                      <th>Last call</th>
                    </tr>
                  </thead>
                  <tbody>
                    {agentsUsed.map((a: UserAgentUsage) => (
                      <tr key={a.agent_name}>
                        <td>
                          <span class="prov-with-icon">
                            <BotIcon class="prov-icon" />
                            <Link
                              href={`/agents/${encodeURIComponent(a.agent_name)}`}
                            >
                              <span class="mono">{a.agent_name}</span>
                            </Link>
                          </span>
                        </td>
                        <td class="num">{fmtUSD(a.total_cost)}</td>
                        <td class="num">{fmtInt(a.request_count)}</td>
                        <td class="text-dim">
                          {fmtRelative(a.last_seen) || fmtTs(a.last_seen)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
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
                      <th>Agent</th>
                      <th class="num">Spend</th>
                      <th class="num">Calls</th>
                      <th>Last call</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentRuns.map((r: UserRunSummary) => (
                      <tr key={r.run_id} class="row-link">
                        <td>
                          <Link href={`/runs/${encodeURIComponent(r.run_id)}`}>
                            <span class="mono">{r.run_id}</span>
                          </Link>
                        </td>
                        <td>
                          <span class="prov-with-icon">
                            <BotIcon class="prov-icon" />
                            <Link
                              href={`/agents/${encodeURIComponent(r.agent_name)}`}
                            >
                              <span class="mono">{r.agent_name}</span>
                            </Link>
                          </span>
                        </td>
                        <td class="num">{fmtUSD(r.total_cost)}</td>
                        <td class="num">{fmtInt(r.request_count)}</td>
                        <td class="text-dim">
                          {fmtRelative(r.last_seen) || fmtTs(r.last_seen)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </>
  );
}
