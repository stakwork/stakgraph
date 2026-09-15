// StatusBadge — the run / agent kill-state pill, plus the two tiny
// derivations that decide which one to show.
//
// The derivation is deliberately shallow and lives here so every
// page agrees on what "running" means. Inputs are things the SPA
// already has (the /state snapshot and the newest call timestamp
// from the call log) — no new backend.
//
//   killed    state.killed. The kill key is set; the run's (or the
//             agent's runs') next LLM call is rejected when the swarm
//             has enforce_macaroons=true, logged otherwise.
//   exceeded  agents: current_spend_usd >= configured_cap_usd.
//             runs: cost or steps at/over the macaroon layer's cap
//             (/state surfaces max_cost_usd / max_steps from the
//             accumulator's meta:run record), or any ancestor over
//             its own cap — the cap walk rejects the child for that
//             too. Only enforced when enforce_budgets=true; in shadow
//             it is the operator's cue, not a hard stop.
//   running   a call landed within RUN_ACTIVE_WINDOW_MS (either the
//             newest call-log row or the /state step counter moving
//             between polls). This is a heuristic: a run idling in a
//             long tool call reads as "done" until its next LLM call.
//   done      none of the above.

import type { AgentStateResponse } from "../api/types";

export type Status = "running" | "killed" | "exceeded" | "done";

/** How recent the last LLM call must be for a run to count as
 *  in-flight. Five minutes covers the long tool calls we see in
 *  practice without keeping a finished run "running" all afternoon. */
export const RUN_ACTIVE_WINDOW_MS = 5 * 60_000;

export function deriveRunStatus(args: {
  killed: boolean;
  /** Cost or steps at/over a cap on this run or an ancestor. */
  exceeded?: boolean;
  /** Epoch ms of the most recent evidence of activity, if any. */
  lastActivityMs?: number;
  now?: number;
}): Status {
  const { killed, exceeded = false, lastActivityMs, now = Date.now() } = args;
  if (killed) return "killed";
  if (exceeded) return "exceeded";
  if (
    lastActivityMs !== undefined &&
    now - lastActivityMs < RUN_ACTIVE_WINDOW_MS
  ) {
    return "running";
  }
  return "done";
}

/** Agents have no "running"/"done" — an agent is a name, not a
 *  process — so only the two blocking states get a badge. `null`
 *  means "nothing to flag". */
export function deriveAgentStatus(
  state: AgentStateResponse,
): Extract<Status, "killed" | "exceeded"> | null {
  if (state.killed) return "killed";
  if (
    state.configured_cap_usd != null &&
    state.current_spend_usd >= state.configured_cap_usd
  ) {
    return "exceeded";
  }
  return null;
}

const TONE: Record<Status, string> = {
  running: "badge-accent",
  killed: "badge-danger",
  exceeded: "badge-warning",
  done: "badge-dim",
};

const DEFAULT_TITLE: Record<Status, string> = {
  running: "A call landed within the last few minutes.",
  killed:
    "Kill flag set. Rejected on the next LLM call when enforce_macaroons=true; logged only in shadow mode.",
  exceeded: "Current-bucket spend is at or over the configured cap.",
  done: "No recent calls.",
};

export function StatusBadge({
  status,
  title,
  label,
}: {
  status: Status;
  title?: string;
  /** Override the rendered text; `status` then only picks the tone
   *  (running = live accent + pulse, killed = danger, exceeded =
   *  warning, done = dim). For cards whose states are not run states
   *  — the transparency-log card's Logging / Disabled / Unavailable.
   *  Pass `title` too: the default titles describe run states. */
  label?: string;
}) {
  return (
    <span class={"badge " + TONE[status]} title={title ?? DEFAULT_TITLE[status]}>
      {status === "running" ? <span class="badge-pulse" /> : null}
      {label ?? status}
    </span>
  );
}
