// RunDetail — live state + kill switch, provenance, summary, and the
// paginated call log for one run.
//
// Phase 9 adds the LiveStateCard (Redis hot state via /runs/:id/state,
// polled at 2s while the run is in flight) and the Kill / Unkill
// switch behind KillConfirmModal. The call log and Provenance card
// are unchanged and still read-only. Phase-8.5 added the Provenance
// card, which is a pure frontend rearrangement of fields already on
// logs.metadata.

import { useEffect, useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { BotIcon, StopIcon, UserIcon } from "../components/icons";
import { KillConfirmModal } from "../components/KillConfirmModal";
import { StatusBadge, deriveRunStatus } from "../components/StatusBadge";
import { getErrorMessage } from "../api/client";
import {
  useKillRun,
  useRunCall,
  useRunDetail,
  useRunState,
  useTrustOrg,
  useTrustStatus,
  useUnkillRun,
} from "../api/queries";
import type {
  CallDetailResponse,
  RunLogEntry,
  RunStateResponse,
} from "../api/types";
import type {
  CacheDebug,
  ChatContentBlock,
  ChatMessage,
  ChatToolCall,
  TokenUsage,
  TrustOrg,
} from "../api/manual";

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

  // `selectedCallID` drives the right-side drawer. Set on row
  // click, cleared on close / ESC. The drawer itself owns the
  // fetch via useRunCall — keeping that hook inside the drawer
  // means closing the drawer fully releases the in-flight request
  // (Tanstack cancels disabled queries).
  const [selectedCallID, setSelectedCallID] = useState<string | null>(null);

  // ESC closes the drawer. Bound only while the drawer is open so
  // we don't intercept keys for the rest of the page (some operators
  // use ESC to dismiss other browser-level UI).
  useEffect(() => {
    if (!selectedCallID) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedCallID(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedCallID]);

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
  const session = md["session-id"] ?? "";
  const deployment = md["deployment"] ?? "";
  const orgID = md["org-id"] ?? "";

  // Lookup the org in the trust registry. `data === null` is a
  // meaningful state (org_id present but not in registry → render
  // "not in registry" badge); `undefined` is in-flight.
  const trust = useTrustOrg(orgID || undefined);

  // Trust status carries the swarm's own realm_id (phase 11). Used
  // by the AuthorizedBy banner to surface "processed by swarm w1"
  // — the realm dropped off per-row metadata, so the registry
  // status is the only place it lives now.
  const trustStatus = useTrustStatus();

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

      <LiveStateCard runID={runID} lastCallISO={firstLog?.timestamp} />

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

              {/* Top section: which org's signature authorized
                  this run. Cross-references the trust registry
                  to surface pubkey + issuer + a verification
                  badge. Hidden when the run carries no org-id
                  dim (pre-phase-6 traffic without macaroon). */}
              {orgID ? (
                <AuthorizedBy
                  orgID={orgID}
                  trust={trust.data}
                  loading={trust.isLoading}
                  swarmRealmID={trustStatus.data?.realm_id}
                />
              ) : null}

              <dl class="kvgrid">
                {/* Order: User → Agent → Workspace. Human first,
                    then the tool the human pointed at it, then
                    the workspace context. The icons are
                    decorative + a quick visual anchor for skimming. */}
                <ProvField label="User">
                  <span class="prov-with-icon">
                    <UserIcon class="prov-icon" />
                    <Link href={`/people/${encodeURIComponent(user)}`}>
                      <span class="mono">{user}</span>
                    </Link>
                  </span>
                </ProvField>
                <ProvField label="Agent">
                  <span class="prov-with-icon">
                    <BotIcon class="prov-icon" />
                    <Link href={`/agents/${encodeURIComponent(agent)}`}>
                      <span class="mono">{agent}</span>
                    </Link>
                  </span>
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
            <div class="text-dim" style="margin-bottom: 8px; font-size: 12px">
              Click a row to view the request &amp; response payload.
            </div>
            <DataTable<RunLogEntry>
              rows={logs}
              emptyMessage="No calls recorded for this run."
              onRowClick={(r) => setSelectedCallID(r.id)}
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

      {selectedCallID ? (
        <CallDetailDrawer
          runID={runID}
          callID={selectedCallID}
          onClose={() => setSelectedCallID(null)}
        />
      ) : null}
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

// ─── LiveStateCard ───────────────────────────────────────────────────
//
// The "stop this thing" card. Reads the run's Redis hot state
// (/runs/:id/state) and owns the Kill / Unkill switch. Three
// non-error shapes to render:
//
//   data === null       swarm has no Redis ⇒ inline "unavailable"
//                       note, switch disabled. Not an error page —
//                       the rest of RunDetail is log-backed and fine.
//   ttl_seconds === -2  Redis is up but this run has no state yet
//                       (never made a macaroon-bearing call, or the
//                       accumulator expired). Kill still allowed:
//                       KillRun doesn't need prior state.
//   otherwise           figures + last tools + TTL countdown.
//
// Status (StatusBadge.deriveRunStatus) needs "when did this run last
// do anything". Two sources, whichever is newer: the newest call-log
// row (fetched once — useRunDetail doesn't poll) and the /state step
// counter moving between polls (useStepActivity). The result also
// drives the poll cadence: 2s while running, 30s once done, 500ms
// for 30s after a kill (see useRunState).
function LiveStateCard({
  runID,
  lastCallISO,
}: {
  runID: string;
  lastCallISO?: string;
}) {
  const [inFlight, setInFlight] = useState(false);
  const stateQ = useRunState(runID, { inFlight });
  const kill = useKillRun(runID);
  const unkill = useUnkillRun(runID);
  const [modal, setModal] = useState<"kill" | "unkill" | null>(null);
  const now = useNow(15_000);

  const st = stateQ.data; // undefined: loading · null: redis off
  const stepActivityMs = useStepActivity(st?.steps);
  const lastCallMs = lastCallISO ? Date.parse(lastCallISO) : undefined;
  const evidence = [lastCallMs, stepActivityMs].filter(
    (v): v is number => v !== undefined && !Number.isNaN(v),
  );
  const lastActivityMs = evidence.length ? Math.max(...evidence) : undefined;
  // Over cap on this run or any ancestor: the cap walk rejects the
  // child for a parent's exhausted budget too.
  const exceeded =
    !!st && (overCap(st) || (st.ancestors ?? []).some((a) => overCap(a)));
  const status = deriveRunStatus({
    killed: !!st?.killed,
    exceeded,
    lastActivityMs,
    now,
  });
  useEffect(() => setInFlight(status === "running"), [status]);

  const unavailable = st === null;
  const canAct = !!st && !stateQ.isError;
  const active = modal === "kill" ? kill : unkill;
  const tools = st?.tools ?? [];
  const noState = !!st && st.ttl_seconds === -2;

  const open = (which: "kill" | "unkill") => {
    // Clear any stale error from a previous attempt so the modal
    // opens clean.
    kill.reset();
    unkill.reset();
    setModal(which);
  };
  // Two explicit branches (not `(a ? kill : unkill).mutate(...)`):
  // the two mutations have different result types, so TS can't type
  // `mutate` on their union.
  const confirm = () => {
    const done = { onSuccess: () => setModal(null) };
    if (modal === "kill") kill.mutate(undefined, done);
    else unkill.mutate(undefined, done);
  };

  return (
    <section class="card hotstate">
      <div class="card-header">
        <div class="hotstate-title">
          <div class="card-title">Live state</div>
          {st ? (
            <StatusBadge
              status={status}
              title={
                status === "running" && lastActivityMs !== undefined
                  ? "Last activity " +
                    fmtRelative(new Date(lastActivityMs).toISOString())
                  : undefined
              }
            />
          ) : null}
        </div>
        <div class="hotstate-actions">
          <div class="text-dim mono" style="font-size: 11px">
            {unavailable ? "redis off" : "redis hot state"}
          </div>
          {st?.killed ? (
            <button
              type="button"
              class="btn"
              disabled={!canAct}
              title="Clear the kill flag; the run resumes on its next LLM call"
              onClick={() => open("unkill")}
            >
              Unkill
            </button>
          ) : (
            <button
              type="button"
              class="btn btn-icon is-danger-solid"
              disabled={!canAct}
              title={
                unavailable
                  ? "Hot state unavailable on this swarm (no Redis) — kill switches are off"
                  : "Stop this run and every sub-agent it spawned"
              }
              onClick={() => open("kill")}
            >
              <StopIcon />
              Kill run
            </button>
          )}
        </div>
      </div>

      {unavailable ? (
        <div class="hotstate-note">
          Hot state unavailable on this swarm — the gateway has no Redis
          configured (<span class="mono">BIFROST_PLUGIN_REDIS_URL</span>),
          so there are no live accumulators and no kill switch here. The
          call log below is unaffected.
        </div>
      ) : stateQ.isError ? (
        <div class="error-banner" style="margin-bottom: 0">
          {getErrorMessage(stateQ.error)}
        </div>
      ) : !st ? (
        <div class="loading">Loading…</div>
      ) : (
        <>
          <div class="hotstate-row">
            <Figure
              label="Cost (live)"
              value={noState ? "—" : fmtUSD(st.cost_usd)}
            />
            <Figure label="Steps" value={noState ? "—" : fmtInt(st.steps)} />
            <Figure label="State expires in" value={fmtTTL(st.ttl_seconds)} />
            <Figure
              label="Kill flag"
              value={st.killed ? "set" : "clear"}
              tone={st.killed ? "danger" : undefined}
            />
          </div>
          {noState ? null : <CapMeters st={st} />}
          <div class="hotstate-tools">
            <span class="text-dim" style="font-size: 12px">
              Last tools:
            </span>
            {tools.length === 0 ? (
              <span class="text-dim" style="font-size: 12px">
                none recorded
              </span>
            ) : (
              tools.map((t, i) => (
                <span key={i} class="pill mono" title={i === 0 ? "most recent" : undefined}>
                  {t}
                </span>
              ))
            )}
          </div>
          {noState ? (
            <div class="hotstate-note">
              No hot state for this run yet — it hasn't made a
              macaroon-bearing call through this gateway, or its
              accumulator has expired. Killing is still allowed; the flag
              catches the run's next call.
            </div>
          ) : null}
          {exceeded && !st.killed ? (
            <div class="hotstate-note tone-danger">
              Over cap. The run's <strong>next LLM call</strong> is rejected
              with 402 when the swarm has{" "}
              <span class="mono">enforce_budgets=true</span>; in shadow mode
              the overrun is logged and the call goes through.
            </div>
          ) : null}
          {st.killed ? (
            <div class="hotstate-note tone-danger">
              Kill flag set
              {kill.data ? ` at ${fmtTs(kill.data.killed_at)}` : ""}. It
              takes effect on the run's <strong>next LLM call</strong>;
              sub-agents this run spawned are killed with it. If calls
              keep landing, the swarm is most likely in shadow mode (
              <span class="mono">enforce_macaroons=false</span>), where
              kills are logged but not enforced. The flag expires on its
              own after 1 hour.
            </div>
          ) : null}
        </>
      )}

      {modal ? (
        <KillConfirmModal
          target={{ kind: "run", id: runID }}
          action={modal}
          pending={active.isPending}
          error={active.isError ? getErrorMessage(active.error) : null}
          onConfirm={confirm}
          onClose={() => setModal(null)}
        />
      ) : null}
    </section>
  );
}

// ─── CapMeters ───────────────────────────────────────────────────────
//
// "Cost vs max_cost_usd, steps vs max_steps, one meter per ancestor"
// (phase 9 "Run detail"). The caps come from the macaroon chain: the
// accumulator records each layer's caveats in meta:run:<id> and
// /state serves them back as max_cost_usd / max_steps plus the
// ancestors walk. A null cap means the layer declared none — the
// meter cell says so rather than drawing an empty bar. Ancestors are
// nearest-parent first; each links to its own run page.

type Capped = Pick<
  RunStateResponse,
  "run_id" | "cost_usd" | "steps" | "max_cost_usd" | "max_steps"
>;

function overCap(c: Capped): boolean {
  return (
    (c.max_cost_usd != null && c.cost_usd >= c.max_cost_usd) ||
    (c.max_steps != null && c.steps >= c.max_steps)
  );
}

function meterTone(ratio: number): "ok" | "warning" | "danger" {
  return ratio >= 1 ? "danger" : ratio >= 0.8 ? "warning" : "ok";
}

function CapMeters({ st }: { st: RunStateResponse }) {
  const ancestors = st.ancestors ?? [];
  const leafHasCap = st.max_cost_usd != null || st.max_steps != null;
  if (!leafHasCap && ancestors.length === 0) {
    return (
      <div class="capmeter-none" style="margin-top: var(--sp-4)">
        No caps recorded for this run — its macaroon declared neither
        max_cost_usd nor max_steps.
      </div>
    );
  }
  return (
    <div class="capmeters">
      <CapRow who="this run" run={st} />
      {ancestors.map((a, i) => (
        <CapRow
          key={a.run_id}
          who={i === 0 ? "parent" : `ancestor ${i + 1}`}
          run={a}
          link
          killed={a.killed}
        />
      ))}
    </div>
  );
}

function CapRow({
  who,
  run,
  link,
  killed,
}: {
  who: string;
  run: Capped;
  link?: boolean;
  killed?: boolean;
}) {
  return (
    <div class={"capmeter-run" + (link ? " is-ancestor" : "")}>
      <div class="capmeter-who">
        <span>
          {who}
          {killed ? (
            <>
              {" "}
              <span class="text-danger">· killed</span>
            </>
          ) : null}
        </span>
        {link ? (
          <Link href={`/runs/${encodeURIComponent(run.run_id)}`}>
            <span class="mono" title={run.run_id}>
              {run.run_id}
            </span>
          </Link>
        ) : (
          <span class="mono" title={run.run_id}>
            {run.run_id}
          </span>
        )}
      </div>
      <CapMeter
        label="Cost"
        value={run.cost_usd}
        cap={run.max_cost_usd}
        fmt={fmtUSD}
      />
      <CapMeter
        label="Steps"
        value={run.steps}
        cap={run.max_steps}
        fmt={fmtInt}
      />
    </div>
  );
}

function CapMeter({
  label,
  value,
  cap,
  fmt,
}: {
  label: string;
  value: number;
  cap: number | null | undefined;
  fmt: (v: number) => string;
}) {
  if (cap == null) {
    return (
      <div class="capmeter">
        <div class="capmeter-figures">
          <span>{label}</span>
          <span>
            <span class="mono">{fmt(value)}</span> · no cap
          </span>
        </div>
        <div class="budget-meter budget-meter-lg" aria-hidden="true" />
      </div>
    );
  }
  const ratio = cap > 0 ? value / cap : 0;
  const pct = Math.min(100, Math.max(0, ratio * 100));
  const tone = meterTone(ratio);
  return (
    <div class="capmeter">
      <div class="capmeter-figures">
        <span>{label}</span>
        <span class={"tone-" + tone}>
          <span class="mono">{fmt(value)}</span> of{" "}
          <span class="mono">{fmt(cap)}</span> ({(ratio * 100).toFixed(0)}%)
        </span>
      </div>
      <div
        class="budget-meter budget-meter-lg"
        role="meter"
        aria-label={`${label}: ${fmt(value)} of ${fmt(cap)}`}
        aria-valuemin={0}
        aria-valuemax={cap}
        aria-valuenow={value}
      >
        <div class={`budget-meter-bar tone-${tone}`} style={`width:${pct}%`} />
      </div>
    </div>
  );
}

function Figure({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "danger";
}) {
  return (
    <div class="budget-figure">
      <div class="budget-figure-label">{label}</div>
      <div class={"budget-figure-val" + (tone ? " tone-" + tone : "")}>
        {value}
      </div>
    </div>
  );
}

// Re-render tick so time-relative derivations (running → done after
// RUN_ACTIVE_WINDOW_MS of silence) move even when the polled data
// itself doesn't change between polls.
function useNow(everyMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(t);
  }, [everyMs]);
  return now;
}

// Turns the /state step counter into "when did it last move". The
// first observation is a baseline, not evidence of activity; only a
// change between polls counts.
function useStepActivity(steps: number | undefined): number | undefined {
  const [seen, setSeen] = useState<{ steps: number; at: number } | undefined>();
  useEffect(() => {
    if (steps === undefined) return;
    setSeen((prev) => {
      if (!prev) return { steps, at: 0 };
      if (prev.steps === steps) return prev;
      return { steps, at: Date.now() };
    });
  }, [steps]);
  return seen && seen.at > 0 ? seen.at : undefined;
}

// TTL of the cost accumulator (auth/accumulator.go: clamp(exp − now +
// 1h, 1h, 7d), refreshed on every write). -2 = no key, -1 = no expiry.
function fmtTTL(s: number): string {
  if (s === -2) return "—";
  if (s === -1) return "never";
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h}h ${rm}m` : `${h}h`;
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

// AuthorizedBy is the "this run was signed off by org X" banner at
// the top of the Provenance card. Three visual states:
//
//   - loading: skeleton placeholder
//   - found:   green ✓ badge + pubkey + issuer URL + swarm realm_id
//   - missing: amber ⚠ badge "Not in trust registry"
//
// The badge color is the operator's signal that the verifier
// would (or wouldn't) accept this org's signatures. Pre-phase-6
// the dim is caller-stamped, so "in registry" doesn't yet mean
// "cryptographically verified" — it just means "the operator
// has added this org to the local trust set." When phase 6 lands,
// the same UI accurately reflects the verification chain.
//
// Phase 11: when the swarm has a `realm_id` configured (multi-swarm
// deployments), render it on the same card. Per-row realm-id
// metadata is gone (every row in this swarm's logs is by definition
// for this swarm's realm), so the trust card is the right home for
// "what's this swarm's identity?".
function AuthorizedBy({
  orgID,
  trust,
  loading,
  swarmRealmID,
}: {
  orgID: string;
  trust: TrustOrg | null | undefined;
  loading: boolean;
  swarmRealmID?: string;
}) {
  const fmtKey = (k: string) =>
    k.length > 16 ? k.slice(0, 8) + "…" + k.slice(-6) : k;

  if (loading) {
    return (
      <div class="authorized-by is-loading">
        <div class="authorized-by-head">
          <span class="authorized-by-label">Authorized by</span>
          <span class="mono">{orgID}</span>
        </div>
        <div class="text-dim">Checking trust registry…</div>
      </div>
    );
  }
  if (!trust) {
    return (
      <div class="authorized-by tone-warning">
        <div class="authorized-by-head">
          <span class="authorized-by-label">Authorized by</span>
          <span class="mono">{orgID}</span>
          <span class="badge badge-warning">⚠ Not in trust registry</span>
        </div>
        <div class="text-dim" style="font-size: 12px">
          The plugin has no public key on file for this org.
          Signatures from it would fail verification.
        </div>
      </div>
    );
  }
  return (
    <div class="authorized-by tone-ok">
      <div class="authorized-by-head">
        <span class="authorized-by-label">Authorized by</span>
        <span class="mono">{trust.org_id}</span>
        <span class="badge badge-ok">✓ Trusted</span>
      </div>
      <dl class="authorized-by-grid">
        <dt>Pubkey</dt>
        <dd class="mono" title={trust.pubkey}>
          {fmtKey(trust.pubkey)}
        </dd>
        {trust.issuer_url ? (
          <>
            <dt>Issuer</dt>
            <dd class="mono">
              <a href={trust.issuer_url} target="_blank" rel="noreferrer">
                {trust.issuer_url}
              </a>
            </dd>
          </>
        ) : null}
        {swarmRealmID ? (
          <>
            <dt>Realm</dt>
            <dd class="mono" title="This swarm's realm_id (set via PUT /_plugin/trust/realm_id)">
              {swarmRealmID}
            </dd>
          </>
        ) : null}
        {trust.grace_pubkeys && trust.grace_pubkeys.length > 0 ? (
          <>
            <dt>Grace</dt>
            <dd class="mono text-dim">
              {trust.grace_pubkeys.length} key(s){" "}
              {trust.grace_until ? "until " + trust.grace_until : ""}
            </dd>
          </>
        ) : null}
      </dl>
    </div>
  );
}

// ─── CallDetailDrawer ────────────────────────────────────────────────
//
// Right-side overlay that fetches /runs/:id/calls/:id and renders
// the request / response bodies (plus tools, params, error) as
// pretty-printed JSON.
//
// Why a drawer (not an expandable row): the call log already wears
// six numeric columns. Expanding inline would push the table into
// horizontal scroll on a 1280px viewport, and the JSON bodies are
// commonly hundreds of lines tall — they want their own scroll
// container. A right-side panel keeps the row context visible.
//
// We render the surrounding overlay (backdrop + close button +
// header) only here; the JSON bodies are delegated to JsonBlock
// which handles the "pretty-print or fall back to .toString()"
// dance once for every field.
function CallDetailDrawer({
  runID,
  callID,
  onClose,
}: {
  runID: string;
  callID: string;
  onClose: () => void;
}) {
  const q = useRunCall(runID, callID);

  return (
    <>
      <div class="drawer-backdrop" onClick={onClose} />
      <aside class="drawer" role="dialog" aria-label="Call detail">
        <header class="drawer-header">
          <div>
            <div class="drawer-eyebrow">Call</div>
            <div class="mono drawer-title" title={callID}>
              {callID}
            </div>
          </div>
          <button class="drawer-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div class="drawer-body">
          {q.isLoading ? (
            <div class="loading">Loading call…</div>
          ) : q.isError ? (
            <div class="error-banner">{getErrorMessage(q.error)}</div>
          ) : q.data ? (
            <CallDetailContent call={q.data} />
          ) : null}
        </div>
      </aside>
    </>
  );
}

// useDemoHash computes a SHA-256 over the call's user-facing content
// (input_history + output_message) and returns the hex digest. Used
// for the "chain" pill in the drawer's status row — it's a real hash
// of real bytes the SPA already received, but it is NOT yet a chain:
// each call hashes in isolation. Phase-? will tie this to the prior
// turn's hash, stored in Redis, and verify continuity on the way in.
//
// Async because window.crypto.subtle.digest is async; the hook
// returns `undefined` while in flight (typically <1ms) so the pill
// can show a placeholder without blocking render.
function useDemoHash(call: CallDetailResponse): string | undefined {
  const [hex, setHex] = useState<string | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    const payload = JSON.stringify({
      input_history: call.input_history ?? null,
      output_message: call.output_message ?? null,
    });
    const bytes = new TextEncoder().encode(payload);
    crypto.subtle
      .digest("SHA-256", bytes)
      .then((buf) => {
        if (cancelled) return;
        const arr = new Uint8Array(buf);
        let s = "";
        for (let i = 0; i < arr.length; i++) {
          s += arr[i].toString(16).padStart(2, "0");
        }
        setHex(s);
      })
      .catch(() => {
        // crypto.subtle requires a secure context (https or
        // localhost). On a non-secure origin we just suppress the
        // pill — better than throwing in a render path.
        if (!cancelled) setHex(undefined);
      });
    return () => {
      cancelled = true;
    };
  }, [call.id]);
  return hex;
}

function CallDetailContent({ call }: { call: CallDetailResponse }) {
  // The call log row above the drawer already shows provider /
  // model / timestamp / cost; repeating them here just pushes the
  // chat down. We keep the status strip ultra-compact and surface
  // only what the row *doesn't* carry: stop_reason, stream flag,
  // retries, prompt-cache badge.
  const cachedRead =
    call.token_usage?.prompt_tokens_details?.cached_read_tokens ?? 0;
  const cachedWrite =
    call.token_usage?.prompt_tokens_details?.cached_write_tokens ?? 0;
  const chainHash = useDemoHash(call);

  // Assemble the full conversation as a single ordered list. The
  // tail item is the assistant's output_message, separated visually
  // from input_history by a "response" divider (input_history is
  // what the model *saw*; output is what it produced). Both render
  // through the same ChatBubble component.
  const inputMessages = Array.isArray(call.input_history)
    ? call.input_history
    : [];
  const outputMessage = call.output_message ?? null;

  const hasConversation = inputMessages.length > 0 || outputMessage;

  return (
    <>
      <section class="drawer-status">
        <StatusPill
          label={call.status}
          tone={call.status === "success" ? "ok" : "danger"}
        />
        <Pill>
          <span class="text-dim">latency:</span>{" "}
          <span class="mono">{fmtInt(call.latency)} ms</span>
        </Pill>
        {chainHash ? (
          <Pill
            tone="accent"
            title={`SHA-256 of input_history + output_message — full digest: ${chainHash}`}
          >
            <span class="text-dim">chain:</span>{" "}
            <span class="mono">
              {chainHash.slice(0, 8)}…{chainHash.slice(-4)}
            </span>
          </Pill>
        ) : null}
        {call.stream ? <Pill tone="accent">streamed</Pill> : null}
        {call.number_of_retries > 0 ? (
          <Pill tone="warning">
            retries: {fmtInt(call.number_of_retries)}
          </Pill>
        ) : null}
        {call.fallback_index > 0 ? (
          <Pill tone="warning">
            fallback #{fmtInt(call.fallback_index)}
          </Pill>
        ) : null}
        {cachedRead > 0 || cachedWrite > 0 ? (
          <Pill tone="accent" title="Prompt-cache tokens this call hit/wrote">
            cache: {cachedRead > 0 ? `${fmtInt(cachedRead)} read` : ""}
            {cachedRead > 0 && cachedWrite > 0 ? " · " : ""}
            {cachedWrite > 0 ? `${fmtInt(cachedWrite)} write` : ""}
          </Pill>
        ) : null}
        {call.cache_debug?.cache_hit ? (
          <Pill tone="ok">semantic cache hit</Pill>
        ) : null}
      </section>

      {/* Error first — if the call failed we want it to be the
          first thing the operator sees. */}
      <JsonBlock title="Error" value={call.error_details} tone="danger" />

      {hasConversation ? (
        <section class="drawer-section">
          <div class="drawer-section-title">Conversation</div>
          <div class="chat-thread">
            {inputMessages.map((m, i) => (
              <ChatBubble key={"in-" + i} msg={m} />
            ))}
            {outputMessage ? (
              <>
                <div class="chat-divider">
                  <span>response</span>
                </div>
                <ChatBubble msg={outputMessage} />
              </>
            ) : null}
          </div>
        </section>
      ) : null}

      <TokenUsageCard usage={call.token_usage} />
      <CacheDebugCard debug={call.cache_debug} />

      {/* Auxiliary JSON blocks. Kept verbatim for power users who
          need to inspect provider-specific fields the chat
          renderer abstracts away. */}
      <JsonBlock title="Params" value={call.params} />
      <JsonBlock title="Tools" value={call.tools} />
      {call.content_summary ? (
        <section class="drawer-section">
          <div class="drawer-section-title">Content summary</div>
          <pre class="json-block">{call.content_summary}</pre>
        </section>
      ) : null}
      {call.raw_request ? (
        <JsonBlock title="Raw request" value={call.raw_request} raw />
      ) : null}
      {call.raw_response ? (
        <JsonBlock title="Raw response" value={call.raw_response} raw />
      ) : null}
    </>
  );
}

// ─── chat bubbles ────────────────────────────────────────────────────
//
// ChatBubble renders one message as a bubble whose alignment and
// color reflect its role:
//
//   user      → right-aligned, accent-tinted
//   assistant → left-aligned, neutral surface
//   tool      → left-aligned, dim (function-return color)
//   system    → centered, dim italic (system prompts are
//               infrastructure, not content)
//
// We don't print the "role" label — the bubble shape carries it.
// A small role tag sits above the bubble for screen readers and
// dense threads where alignment alone might be ambiguous.
//
// Cache-control markers (Anthropic prompt-cache breakpoints) appear
// as a tiny chip on the relevant bubble; assistant tool_calls render
// in their own sub-block below the textual content.
function ChatBubble({ msg }: { msg: ChatMessage }) {
  const role = (msg.role || "").toLowerCase();
  const sideClass =
    role === "user"
      ? "chat-bubble-row align-right"
      : role === "system"
        ? "chat-bubble-row align-center"
        : "chat-bubble-row align-left";
  const toneClass =
    role === "user"
      ? "tone-user"
      : role === "assistant"
        ? "tone-assistant"
        : role === "tool"
          ? "tone-tool"
          : role === "system"
            ? "tone-system"
            : "tone-other";

  // Detect cache_control on any content block. Operators care that
  // a message was cached, not which block — surface a single chip
  // on the bubble.
  const cached = hasCacheControl(msg);

  return (
    <div class={sideClass}>
      <div class="chat-bubble-stack">
        <div class="chat-role">
          {role || "—"}
          {msg.name ? <span class="text-dim"> · {msg.name}</span> : null}
          {cached ? <span class="chat-chip">cache_control</span> : null}
        </div>
        <div class={"chat-bubble " + toneClass}>
          <ChatContent content={msg.content} />
          {msg.reasoning ? (
            <div class="chat-reasoning">
              <div class="chat-sub-label">reasoning</div>
              <div>{msg.reasoning}</div>
            </div>
          ) : null}
          {msg.refusal ? (
            <div class="chat-refusal">
              <div class="chat-sub-label">refusal</div>
              <div>{msg.refusal}</div>
            </div>
          ) : null}
          {msg.tool_calls && msg.tool_calls.length > 0 ? (
            <div class="chat-tool-calls">
              {msg.tool_calls.map((tc, i) => (
                <ToolCallBlock key={tc.id ?? i} call={tc} />
              ))}
            </div>
          ) : null}
          {msg.tool_call_id ? (
            <div class="chat-sub-label" style="margin-top: 6px">
              tool_call_id: <span class="mono">{msg.tool_call_id}</span>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ChatContent({ content }: { content: ChatMessage["content"] }) {
  if (content == null) return null;
  if (typeof content === "string") {
    return <div class="chat-text">{content}</div>;
  }
  if (Array.isArray(content)) {
    return (
      <>
        {content.map((block, i) => (
          <ContentBlock key={i} block={block} />
        ))}
      </>
    );
  }
  return null;
}

function ContentBlock({ block }: { block: ChatContentBlock }) {
  // Most common: a text block. Provider-specific block types
  // (image_url, input_audio, file) get a compact placeholder rather
  // than a full render — the operator can drop to "Raw request" /
  // "Raw response" for the bytes.
  if (block.type === "text" && block.text) {
    return <div class="chat-text">{block.text}</div>;
  }
  if (block.type === "refusal" && block.refusal) {
    return (
      <div class="chat-refusal">
        <div class="chat-sub-label">refusal</div>
        <div>{block.refusal}</div>
      </div>
    );
  }
  if (block.type === "image_url") {
    return (
      <div class="chat-block-placeholder">[image_url block]</div>
    );
  }
  if (block.type === "input_audio") {
    return (
      <div class="chat-block-placeholder">[input_audio block]</div>
    );
  }
  if (block.type === "file") {
    return <div class="chat-block-placeholder">[file block]</div>;
  }
  // Unknown / future block types: fall back to a single-line summary
  // rather than dumping JSON inside the chat bubble.
  return (
    <div class="chat-block-placeholder">[{block.type || "block"}]</div>
  );
}

function ToolCallBlock({ call }: { call: ChatToolCall }) {
  const name = call.function?.name ?? "(no name)";
  const args = call.function?.arguments ?? "";
  // Arguments are often a stringified JSON. Try to pretty-print;
  // fall back to the verbatim string if it doesn't parse.
  let prettyArgs = args;
  if (args) {
    try {
      prettyArgs = JSON.stringify(JSON.parse(args), null, 2);
    } catch {
      prettyArgs = args;
    }
  }
  return (
    <div class="chat-tool-call">
      <div class="chat-sub-label">
        tool_call <span class="mono">{name}</span>
      </div>
      {prettyArgs ? (
        <pre class="chat-tool-args">{prettyArgs}</pre>
      ) : null}
    </div>
  );
}

function hasCacheControl(msg: ChatMessage): boolean {
  if (!Array.isArray(msg.content)) return false;
  return msg.content.some(
    (b) => b && (b.cache_control != null || b.cachePoint != null),
  );
}

// ─── token + cache cards ─────────────────────────────────────────────

function TokenUsageCard({ usage }: { usage?: TokenUsage }) {
  if (!usage) return null;
  const prompt = usage.prompt_tokens ?? 0;
  const completion = usage.completion_tokens ?? 0;
  const total =
    usage.total_tokens ?? (prompt + completion > 0 ? prompt + completion : 0);
  if (total === 0 && prompt === 0 && completion === 0) return null;

  const det = usage.prompt_tokens_details;
  const cachedRead = det?.cached_read_tokens ?? 0;
  const cachedWrite = det?.cached_write_tokens ?? 0;
  // Anthropic emits per-TTL cached_write splits (5m vs 1h
  // breakpoints). Surface both when present — they price
  // differently.
  const w5m = det?.cached_write_token_details?.cached_write_tokens_5m ?? 0;
  const w1h = det?.cached_write_token_details?.cached_write_tokens_1h ?? 0;

  return (
    <section class="drawer-section">
      <div class="drawer-section-title">Tokens</div>
      <div class="token-grid">
        <TokenBox label="prompt" value={prompt} />
        <TokenBox label="completion" value={completion} />
        <TokenBox label="total" value={total} accent />
      </div>
      {cachedRead > 0 || cachedWrite > 0 ? (
        <div class="token-cache">
          <div class="chat-sub-label">Prompt cache</div>
          <div class="token-cache-row">
            {cachedRead > 0 ? (
              <span>
                <span class="text-dim">read:</span>{" "}
                <span class="mono">{fmtInt(cachedRead)}</span>
              </span>
            ) : null}
            {cachedWrite > 0 ? (
              <span>
                <span class="text-dim">write:</span>{" "}
                <span class="mono">{fmtInt(cachedWrite)}</span>
                {w5m > 0 || w1h > 0 ? (
                  <span class="text-dim">
                    {" "}
                    ({w5m > 0 ? `${fmtInt(w5m)} @ 5m` : ""}
                    {w5m > 0 && w1h > 0 ? ", " : ""}
                    {w1h > 0 ? `${fmtInt(w1h)} @ 1h` : ""})
                  </span>
                ) : null}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function TokenBox({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
  accent?: boolean;
}) {
  return (
    <div class={"token-box" + (accent ? " is-accent" : "")}>
      <div class="token-box-label">{label}</div>
      <div class="token-box-value mono">{fmtInt(value)}</div>
    </div>
  );
}

function CacheDebugCard({ debug }: { debug?: CacheDebug }) {
  if (!debug) return null;
  const hit = debug.cache_hit;
  return (
    <section class={"drawer-section" + (hit ? " tone-ok" : "")}>
      <div class="drawer-section-title">Semantic cache</div>
      <div class="drawer-meta-grid">
        <Meta label="Verdict" valueClass={hit ? "text-ok" : "text-dim"}>
          {hit ? "hit" : "miss"}
        </Meta>
        {debug.hit_type ? (
          <Meta label="Type" mono>
            {debug.hit_type}
          </Meta>
        ) : null}
        {debug.similarity !== undefined ? (
          <Meta label="Similarity" mono>
            {debug.similarity.toFixed(3)}
            {debug.threshold !== undefined ? (
              <span class="text-dim"> / {debug.threshold.toFixed(3)}</span>
            ) : null}
          </Meta>
        ) : null}
        {debug.input_tokens !== undefined ? (
          <Meta label="Input tokens" mono>
            {fmtInt(debug.input_tokens)}
          </Meta>
        ) : null}
        {debug.provider_used && debug.requested_provider &&
        debug.provider_used !== debug.requested_provider ? (
          <Meta label="Provider used">
            <span class="mono">{debug.provider_used}</span>{" "}
            <span class="text-dim">
              (req: <span class="mono">{debug.requested_provider}</span>)
            </span>
          </Meta>
        ) : null}
      </div>
    </section>
  );
}

// ─── small primitives ────────────────────────────────────────────────

function Pill({
  children,
  tone,
  title,
}: {
  children: any;
  tone?: "ok" | "danger" | "warning" | "accent";
  title?: string;
}) {
  return (
    <span class={"pill" + (tone ? " pill-" + tone : "")} title={title}>
      {children}
    </span>
  );
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "ok" | "danger";
}) {
  return <Pill tone={tone}>{label}</Pill>;
}

function Meta({
  label,
  children,
  mono,
  valueClass,
}: {
  label: string;
  children: any;
  mono?: boolean;
  valueClass?: string;
}) {
  return (
    <div class="drawer-meta">
      <div class="drawer-meta-label">{label}</div>
      <div
        class={
          "drawer-meta-value" +
          (mono ? " mono" : "") +
          (valueClass ? " " + valueClass : "")
        }
      >
        {children}
      </div>
    </div>
  );
}

// JsonBlock pretty-prints `value` as JSON. Accepts:
//   - parsed JSON (from the json.RawMessage handed back by the
//     plugin, already JSON.parse'd into the response shape)
//   - a string (Bifrost emits `raw_response` as a string — set
//     `raw` to skip the parse attempt and just render verbatim)
//   - null / undefined → renders nothing (the field is empty)
//
// `tone="danger"` tints the surround red for the error block,
// matching the call-log status colors.
function JsonBlock({
  title,
  value,
  tone,
  raw,
}: {
  title: string;
  value: unknown;
  tone?: "danger";
  raw?: boolean;
}) {
  if (value === null || value === undefined || value === "") return null;

  let body: string;
  if (raw && typeof value === "string") {
    // Try to pretty-print it if it parses as JSON; otherwise the
    // string is most likely a non-JSON upstream payload (HTML
    // error page, plain text). Render it as-is.
    try {
      body = JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      body = value;
    }
  } else if (typeof value === "string") {
    body = value;
  } else {
    try {
      body = JSON.stringify(value, null, 2);
    } catch {
      body = String(value);
    }
  }

  return (
    <section class={"drawer-section" + (tone ? " tone-" + tone : "")}>
      <div class="drawer-section-title">{title}</div>
      <pre class="json-block">{body}</pre>
    </section>
  );
}
