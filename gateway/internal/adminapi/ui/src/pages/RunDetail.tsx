// RunDetail — provenance + summary + paginated call log for one run.
//
// Read-only; phase 9 grows the kill button and live-state panel
// (both of which depend on Redis hot state that phase 8 doesn't
// surface). Phase-8.5 adds the Provenance card, which is a pure
// frontend rearrangement of fields already on logs.metadata.

import { Link } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { BotIcon, UserIcon } from "../components/icons";
import { getErrorMessage } from "../api/client";
import { useRunDetail, useTrustOrg } from "../api/queries";
import type { RunLogEntry, TrustOrg } from "../api/types";

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
  const orgID = md["org-id"] ?? "";

  // Lookup the org in the trust registry. `data === null` is a
  // meaningful state (org_id present but not in registry → render
  // "not in registry" badge); `undefined` is in-flight.
  const trust = useTrustOrg(orgID || undefined);

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

              {/* Top section: which org's signature authorized
                  this run. Cross-references the trust registry
                  to surface pubkey + issuer + a verification
                  badge. Hidden when the run carries no org-id
                  dim (pre-phase-6 traffic without macaroon). */}
              {orgID ? (
                <AuthorizedBy orgID={orgID} trust={trust.data} loading={trust.isLoading} />
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

// AuthorizedBy is the "this run was signed off by org X" banner at
// the top of the Provenance card. Three visual states:
//
//   - loading: skeleton placeholder
//   - found:   green ✓ badge + pubkey + issuer URL
//   - missing: amber ⚠ badge "Not in trust registry"
//
// The badge color is the operator's signal that the verifier
// would (or wouldn't) accept this org's signatures. Pre-phase-6
// the dim is caller-stamped, so "in registry" doesn't yet mean
// "cryptographically verified" — it just means "the operator
// has added this org to the local trust set." When phase 6 lands,
// the same UI accurately reflects the verification chain.
function AuthorizedBy({
  orgID,
  trust,
  loading,
}: {
  orgID: string;
  trust: TrustOrg | null | undefined;
  loading: boolean;
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
