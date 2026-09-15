// People — list of observed users in the selected window.
//
// Mirror of the Agents list visually, but the per-row identifier is
// `user_id` and the click-through goes to /people/:id (UserDetail).
// Reuses the existing `useSpendByUser` query — that's the canonical
// per-dim rollup endpoint, and "people" is just the user-facing
// label for what the backend calls "user". The "Revoked" column is
// the user-axis twin of the Agents list's "Kill state": one
// /revoke/users read for the whole table rather than a per-row
// fan-out, since the list is a single ZSET on the server.

import { useMemo, useState } from "preact/hooks";
import { Link, useLocation } from "wouter-preact";

import { DataTable } from "../components/tables/DataTable";
import { WindowPicker } from "../components/controls/WindowPicker";
import { UserIcon } from "../components/icons";
import { StatusBadge } from "../components/StatusBadge";
import { getErrorMessage } from "../api/client";
import { useRevokedUsers, useSpendByUser } from "../api/queries";
import type { UserSpend } from "../api/types";
import type { Window } from "../api/manual";

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

export function People() {
  const [window, setWindow] = useState<Window>("24h");
  const [, setLocation] = useLocation();
  const q = useSpendByUser(window);
  const revoked = useRevokedUsers();
  // user_id → cutoff. `null` ⇒ redis off on this swarm (every cell
  // renders "—" with a reason), `undefined` ⇒ still loading. The
  // explicit return type and local binding keep the narrowing stable
  // across TS / react-query versions (the CI image build failed to
  // narrow `revoked.data` through the truthiness check).
  const cutoffs = useMemo((): Map<string, string> | null | undefined => {
    const d = revoked.data;
    if (d === undefined) return undefined;
    if (d === null) return null;
    const m = new Map<string, string>();
    for (const u of d.users) m.set(u.user_id, u.before);
    return m;
  }, [revoked.data]);

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">Observability</div>
          <h1>People</h1>
        </div>
        <WindowPicker value={window} onChange={setWindow} />
      </div>

      {q.isError ? (
        <div class="error-banner">{getErrorMessage(q.error)}</div>
      ) : (
        <DataTable<UserSpend>
          rows={q.data?.results ?? []}
          emptyMessage="No users have produced calls in this window."
          onRowClick={(r) =>
            setLocation(`/people/${encodeURIComponent(r.user_id)}`)
          }
          columns={[
            {
              key: "user",
              header: "User",
              cell: (r) => (
                <span class="prov-with-icon">
                  <UserIcon class="prov-icon" />
                  <Link href={`/people/${encodeURIComponent(r.user_id)}`}>
                    <span class="mono">{r.user_name || r.user_id}</span>
                  </Link>
                </span>
              ),
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
            {
              key: "revoked",
              header: "Revoked",
              cell: (r) => {
                if (cutoffs === undefined) {
                  return <span class="text-dim">…</span>;
                }
                if (cutoffs === null) {
                  return (
                    <span
                      class="text-dim"
                      title="Hot state unavailable on this swarm (no Redis)"
                    >
                      —
                    </span>
                  );
                }
                const before = cutoffs.get(r.user_id);
                return before !== undefined ? (
                  <StatusBadge
                    status="killed"
                    label="revoked"
                    title={
                      before
                        ? `Authorizations issued before ${new Date(before).toLocaleString()} are rejected`
                        : "Revoke cutoff set (stored value unreadable)"
                    }
                  />
                ) : (
                  <span class="text-dim" title="No revoke cutoff">
                    —
                  </span>
                );
              },
              // Revoked first on a desc sort, like the Agents list's
              // kill column.
              sort: (r) => (cutoffs?.has(r.user_id) ? 1 : 0),
            },
          ]}
          defaultSortKey="cost"
        />
      )}
    </>
  );
}
