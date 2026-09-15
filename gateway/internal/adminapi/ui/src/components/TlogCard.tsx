// TlogCard — the gateway's own view of its phase-12 transparency log.
//
// Everything on this card is a local fact the gateway can attest by
// itself: how many leaves it holds, the root they hash to, the
// per-boot key it signs heads with, and when the newest leaf landed.
// "Witnessed" (a head countersigned with the org key) is Hive's fact,
// not the gateway's, so it is deliberately absent — the "witnessed at
// head N" badge waits for the Hive witness (phase-12 spec, "Not in
// either part"). No leaves and no STH signature either: those reach
// the witness over the bearer-only /tlog/sth route, which a dashboard
// cookie cannot read.
//
// Header badge:
//   Logging      healthy — the log is open and appending
//   Disabled     the log refused to come up or stopped itself after
//                an unrecoverable write; the reason prints under it
//   Unavailable  the status fetch itself failed, which says nothing
//                about the log either way

import { useEffect, useState } from "preact/hooks";

import { getErrorMessage } from "../api/client";
import { useTlogStatus } from "../api/queries";
import { StatusBadge } from "./StatusBadge";

const fmtInt = (v: number) =>
  new Intl.NumberFormat("en-US").format(Math.round(v));

// Relative time, same shape as RunDetail's helper. Takes `now` from
// the ticker below so "12s ago" keeps counting between polls.
function fmtRelative(absISO: string, now: number): string {
  const then = new Date(absISO).getTime();
  if (Number.isNaN(then)) return absISO;
  const sec = Math.max(0, Math.round((now - then) / 1000));
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 48) return `${hr}h ago`;
  return `${Math.round(hr / 24)}d ago`;
}

// First 16 hex chars: enough to tell two roots apart at a glance.
// The full value rides on `title` for hover.
const fmtHex = (h: string) => (h.length > 16 ? h.slice(0, 16) + "…" : h);

function useNow(everyMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(t);
  }, [everyMs]);
  return now;
}

export function TlogCard() {
  const q = useTlogStatus();
  const now = useNow(5_000);
  const st = q.data;
  // tygo types the Go *string fields as optional; the wire says null.
  const reason = st?.error ?? null;
  const lastTS = st?.last_leaf_ts ?? null;

  return (
    <section class="card tlog">
      <div class="card-header">
        <div class="tlog-title">
          <div class="card-title">Transparency log</div>
          {q.isError ? (
            <StatusBadge
              status="done"
              label="Unavailable"
              title="The dashboard could not read /_plugin/tlog/status. This says nothing about the log itself."
            />
          ) : st ? (
            st.healthy ? (
              <StatusBadge
                status="running"
                label="Logging"
                title="The log is open. Every accounted call appends a leaf."
              />
            ) : (
              <StatusBadge
                status="killed"
                label="Disabled"
                title="The log is down. Calls are still served but leave no leaf until the cause is fixed and the plugin restarts."
              />
            )
          ) : null}
        </div>
        {st?.path ? (
          <div
            class="text-dim mono"
            style="font-size: 11px"
            title="Leaf file (BIFROST_PLUGIN_TLOG_PATH)"
          >
            {st.path}
          </div>
        ) : null}
      </div>

      {q.isError ? (
        <div class="error-banner" style="margin-bottom: 0">
          {getErrorMessage(q.error)}
        </div>
      ) : !st ? (
        <div class="loading">Loading…</div>
      ) : (
        <>
          {reason ? <div class="tlog-reason mono">{reason}</div> : null}
          <dl class="kvgrid">
            <dt class="kvgrid-key">Leaves</dt>
            <dd class="kvgrid-val mono">{fmtInt(st.tree_size)}</dd>

            <dt class="kvgrid-key">Last leaf</dt>
            <dd class="kvgrid-val mono" title={lastTS ?? undefined}>
              {lastTS ? fmtRelative(lastTS, now) : "—"}
            </dd>

            <dt class="kvgrid-key">Root</dt>
            <dd class="kvgrid-val mono" title={st.root_hash}>
              {fmtHex(st.root_hash)}
            </dd>

            <dt class="kvgrid-key">Log key</dt>
            <dd class="kvgrid-val mono" title={st.log_pubkey || undefined}>
              {st.log_pubkey ? fmtHex(st.log_pubkey) : "—"}{" "}
              <span
                class="text-dim tlog-hint"
                title="Generated at every plugin boot and held in memory. It attests that this gateway served a head, not who the gateway is."
              >
                per boot
              </span>
            </dd>
          </dl>
        </>
      )}
    </section>
  );
}
