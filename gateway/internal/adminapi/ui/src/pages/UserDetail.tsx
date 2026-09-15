// UserDetail — KPIs + cost histogram + agents used + recent runs,
// all scoped to one user's traffic in the window, plus the user
// revoke switch (the third kill axis after run and agent).
//
// Data flow
// ---------
// One query (`useUserDetail`) drives the KPI cards and the two
// tables. A separate `useHistogramCost` call with `userID` filters
// the time-series to the same window — that runs server-side on
// the backend so no client-side filtering is needed. `useUserRevoke`
// reads the Redis cutoff and feeds both the header badge and the
// Authorization card; it is independent of the logstore, so the
// switch keeps working when the analytics half of the page errors.
//
// Order on the page is intentional: the Authorization card first
// (a set cutoff is the one thing an operator must not miss), KPIs
// (the "is this person a heavy user?" answer), the activity card
// with first/last seen, chart, then the two drill-down tables. Same
// shape as AgentDetail so an operator who learns one learns the
// other.

import { useMemo, useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { CostHistogram } from "../components/charts/CostHistogram";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { WindowPicker } from "../components/controls/WindowPicker";
import { BotIcon, StopIcon, UserIcon } from "../components/icons";
import { KillConfirmModal } from "../components/KillConfirmModal";
import { StatusBadge } from "../components/StatusBadge";
import { getErrorMessage } from "../api/client";
import {
  useClearUserRevoke,
  useHistogramCost,
  useRevokeUser,
  useUserDetail,
  useUserQuota,
  useUserRevoke,
  type UserRevokeState,
} from "../api/queries";
import type { UserAgentUsage, UserRunSummary } from "../api/types";
import type { Window } from "../api/manual";
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

  // Redis hot state: the revoke cutoff. `null` when the swarm has no
  // Redis (badge hidden, switch disabled), `{ before: null }` when no
  // cutoff is set.
  const revoke = useUserRevoke(userID);
  const revoked = !!revoke.data?.before;

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
            {revoked ? (
              <StatusBadge
                status="killed"
                label="revoked"
                title={REVOKED_TITLE}
              />
            ) : null}
          </h1>
        </div>
        <div class="page-actions">
          <WindowPicker value={window} onChange={setWindow} />
          <RevokeUserSwitch userID={userID} state={revoke.data} />
        </div>
      </div>

      <AuthorizationCard state={revoke.data} />

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

// ─── user revocation ───────────────────────────────────────────────
//
// The cutoff is `revoke_user_before:<user_id>` in Redis. Once set,
// the hot path rejects every macaroon whose user authorization was
// issued before it — every in-flight run of this user, whatever agent
// carries it, and every new spawn under the current authorization —
// until Hive issues a fresh one. Per swarm, no TTL. The card states
// that precisely ("issued before <t> are rejected") rather than
// "revoked", because after Hive re-issues, the cutoff is still set
// and still true, but the user is working again.

const REVOKED_TITLE =
  "Revoke cutoff set. Macaroons issued before it are rejected on the next LLM call when enforce_macaroons=true; logged only in shadow mode.";

function AuthorizationCard({
  state,
}: {
  state: UserRevokeState | null | undefined;
}) {
  const unavailable = state === null;
  const before = state?.before ?? null;
  return (
    <section class="card provenance">
      <div class="card-header">
        <div class="card-title">Authorization</div>
        {unavailable ? (
          <span class="text-dim" style="font-size: 11px">
            hot state unavailable on this swarm (no Redis)
          </span>
        ) : before ? (
          <StatusBadge status="killed" label="revoked" title={REVOKED_TITLE} />
        ) : state ? (
          <StatusBadge
            status="done"
            label="accepted"
            title="No cutoff set. Every valid authorization this user holds is accepted."
          />
        ) : (
          <span class="text-dim">…</span>
        )}
      </div>
      {before ? (
        <>
          <dl class="kvgrid" style="margin-bottom: var(--sp-3)">
            <dt class="kvgrid-key">Cutoff</dt>
            <dd class="kvgrid-val">
              <span class="mono">{fmtTs(before)}</span>{" "}
              <span class="text-dim">({fmtRelative(before)})</span>
            </dd>
          </dl>
          {/* Prose lives outside the kvgrid: `.kvgrid-val` breaks
              words anywhere (it is sized for opaque ids), which
              splits sentences mid-word. */}
          <p class="text-dim" style="margin: 0">
            Authorizations issued before the cutoff are rejected on this
            swarm. In-flight runs stop on their next LLM call; new spawns
            need Hive to issue a fresh authorization. No expiry — clear it
            to lift.
          </p>
        </>
      ) : (
        <p class="text-dim" style="margin: 0">
          {unavailable
            ? "Revocation state lives in Redis; this swarm has none configured, so the switch is off."
            : "No cutoff set. Every valid authorization this user holds is accepted on this swarm."}
        </p>
      )}
    </section>
  );
}

// Revoke / Clear for one user, swarm-wide. `state` is the
// /revoke/user/:id snapshot: undefined while loading, null when the
// swarm has no Redis (switch disabled with a tooltip). Typed
// confirmation in the modal because the blast radius is every run
// of this user, not just the ones on this page. The in-flight list
// for the modal comes from /users/:id/quota and is fetched only
// while the modal is open.
function RevokeUserSwitch({
  userID,
  state,
}: {
  userID: string;
  state: UserRevokeState | null | undefined;
}) {
  const revoke = useRevokeUser(userID);
  const clear = useClearUserRevoke(userID);
  const [modal, setModal] = useState<"kill" | "unkill" | null>(null);
  const active = modal === "kill" ? revoke : clear;
  const unavailable = state === null;
  const quota = useUserQuota(userID, modal === "kill");

  const open = (which: "kill" | "unkill") => {
    revoke.reset();
    clear.reset();
    setModal(which);
  };

  return (
    <>
      {state?.before ? (
        <button
          type="button"
          class="btn"
          title="Remove the cutoff; macaroons issued before it are accepted again on their next LLM call"
          onClick={() => open("unkill")}
        >
          Clear revoke
        </button>
      ) : (
        <button
          type="button"
          class="btn btn-icon is-danger-solid"
          disabled={!state}
          title={
            unavailable
              ? "Hot state unavailable on this swarm (no Redis) — revocation is off"
              : "Reject every authorization this user holds on this swarm until Hive re-issues"
          }
          onClick={() => open("kill")}
        >
          <StopIcon />
          Revoke user
        </button>
      )}
      {modal ? (
        <KillConfirmModal
          target={{
            kind: "user",
            id: userID,
            inflight: quota.isError
              ? null
              : quota.data
                ? quota.data.inflight_runs ?? null
                : undefined,
          }}
          action={modal}
          pending={active.isPending}
          error={active.isError ? getErrorMessage(active.error) : null}
          onConfirm={() => {
            const done = { onSuccess: () => setModal(null) };
            if (modal === "kill") revoke.mutate(undefined, done);
            else clear.mutate(undefined, done);
          }}
          onClose={() => setModal(null)}
        />
      ) : null}
    </>
  );
}
