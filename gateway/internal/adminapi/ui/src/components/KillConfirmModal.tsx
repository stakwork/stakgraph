// KillConfirmModal — the one confirmation dialog for every kill and
// unkill in the SPA. Two modes, picked by the target:
//
//   run    plain confirm. Blast radius is one run plus the sub-agents
//          it spawned.
//   agent  typed confirm — the operator must type the agent name.
//          Blast radius is every run of that agent, swarm-wide, so
//          the friction is deliberately higher (phase 9 "Destructive").
//
// Unkill uses the same modal with the same friction: clearing a
// swarm-wide agent kill is as consequential as setting it.
//
// A custom element rather than window.confirm(): Hive embeds this SPA
// in a sandboxed iframe without `allow-modals`, where confirm() is
// silently suppressed (EvalsView's ConfirmButton has the same note).
//
// No optimistic update. The caller keeps the modal open on error and
// leaves the button live for retry — the whole point of clicking Kill
// is to watch the kill actually land, so the modal only closes once
// the server said 200.

import { useEffect, useRef, useState } from "preact/hooks";

import { StopIcon } from "./icons";

export type KillTarget =
  | { kind: "run"; id: string }
  | { kind: "agent"; name: string };

export type KillAction = "kill" | "unkill";

interface Props {
  target: KillTarget;
  action: KillAction;
  /** Mutation in flight — disables the buttons and swaps the label. */
  pending: boolean;
  /** Last mutation error, rendered inline; null when none. */
  error: string | null;
  onConfirm: () => void;
  onClose: () => void;
}

export function KillConfirmModal({
  target,
  action,
  pending,
  error,
  onConfirm,
  onClose,
}: Props) {
  const typed = target.kind === "agent";
  const targetLabel = target.kind === "run" ? target.id : target.name;
  const [input, setInput] = useState("");
  const ready = !typed || input.trim() === target.name;

  // Focus the input (agent) or the confirm button (run) on open so
  // keyboard operators can drive the whole flow without a mouse.
  const inputRef = useRef<HTMLInputElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    (typed ? inputRef.current : confirmRef.current)?.focus();
  }, [typed]);

  // ESC closes — but not mid-request, so a stray keypress can't
  // leave the operator unsure whether the kill went out.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !pending) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pending, onClose]);

  const submit = () => {
    if (pending || !ready) return;
    onConfirm();
  };

  const title =
    action === "kill"
      ? target.kind === "run"
        ? "Kill run"
        : "Kill agent"
      : target.kind === "run"
        ? "Clear kill on run"
        : "Clear kill on agent";

  const confirmLabel = pending
    ? action === "kill"
      ? "Killing…"
      : "Clearing…"
    : action === "kill"
      ? target.kind === "run"
        ? "Kill run"
        : "Kill agent"
      : "Clear kill";

  return (
    <div
      class="modal-backdrop"
      onClick={() => {
        if (!pending) onClose();
      }}
    >
      <div
        class="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="kill-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header class="modal-header">
          {action === "kill" ? (
            <StopIcon class="modal-icon text-danger" />
          ) : null}
          <div id="kill-modal-title" class="modal-title">
            {title}
          </div>
        </header>

        <div class="modal-body">
          <div class="modal-target" title={targetLabel}>
            {targetLabel}
          </div>

          <Scope target={target} action={action} />

          <div class="modal-callout">
            {action === "kill" ? (
              <>
                Takes effect on the <strong>next LLM call</strong>, not
                immediately. It is only enforced when the swarm runs with{" "}
                <span class="mono">enforce_macaroons=true</span>; in shadow
                mode the kill is logged and calls continue. If the run keeps
                making calls after this succeeds, that is the likely reason.
              </>
            ) : (
              <>
                Affected runs resume on their <strong>next LLM call</strong>.
                Nothing is restarted — a run that already exited stays exited.
              </>
            )}
          </div>

          {typed ? (
            <label class="modal-field">
              <span class="modal-label">
                Type <span class="mono">{target.name}</span> to confirm
              </span>
              <input
                ref={inputRef}
                class="input mono"
                value={input}
                disabled={pending}
                autocomplete="off"
                spellcheck={false}
                onInput={(e) => setInput((e.target as HTMLInputElement).value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submit();
                }}
              />
            </label>
          ) : null}

          {error ? <div class="error-banner">{error}</div> : null}
        </div>

        <footer class="modal-footer">
          <button
            type="button"
            class="btn"
            disabled={pending}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            ref={confirmRef}
            type="button"
            class={
              "btn btn-icon " +
              (action === "kill" ? "is-danger-solid" : "is-primary")
            }
            disabled={pending || !ready}
            onClick={submit}
          >
            {action === "kill" ? <StopIcon /> : null}
            {confirmLabel}
          </button>
        </footer>
      </div>
    </div>
  );
}

// Scope spells out the blast radius and the TTL — the two things an
// operator most often gets wrong about these switches (a run kill
// cascades to sub-agents; an agent kill does NOT cascade to
// differently-named sub-agents; both expire on their own). TTLs
// mirror auth/kill.go: killRunTTL = 1h, killAgentTTL = 24h.
function Scope({ target, action }: { target: KillTarget; action: KillAction }) {
  if (target.kind === "run") {
    return action === "kill" ? (
      <p class="modal-note">
        Stops this run <strong>and every sub-agent it spawned</strong>.
        The kill flag expires on its own after <strong>1 hour</strong>;
        re-killing refreshes it.
      </p>
    ) : (
      <p class="modal-note">
        Clears the kill flag on this run and, with it, on the sub-agents
        it spawned.
      </p>
    );
  }
  return action === "kill" ? (
    <p class="modal-note">
      Stops <strong>every run whose leaf agent is this name</strong>,
      across the whole swarm — not just the ones you can see. Sub-agents
      running under other names are not affected. The kill flag expires
      on its own after <strong>24 hours</strong>; re-killing refreshes it.
    </p>
  ) : (
    <p class="modal-note">
      Clears the swarm-wide kill flag for this agent. Every run of this
      agent may resume.
    </p>
  );
}
