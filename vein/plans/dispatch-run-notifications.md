# Dispatch-mode `run_workflow` + run-completion chat notifications

## Problem

The AI builder's `run_workflow` tool awaits the run inline inside one tool
call. For long workflows (a 2-hour legal-benchmark run, an overnight
optimize) that's the wrong shape:

- The agent can do nothing else while it waits, and one severed connection
  wastes the whole wait.
- `VEIN_CHAT_MAX_STEPS` bounds the *turn*, so babysitting a run by polling
  burns tool-call iterations to repeatedly learn "still running".
- The autonomous improve-and-iterate loop (launch → inspect score → revise →
  relaunch) needs the agent to *come back* when a run finishes, without a
  human poking it.

## Design: dispatch + wake (webhook-style), not polling

The chat is already a detached background job whose unit of execution is a
**turn** (launch server-side, tail to reattach). A run-completion
notification is just *something other than a human* appending a message and
launching the next turn. No new execution model — a second trigger for an
existing mechanism.

### 1. `run_workflow` auto-upgrades to detached

The tool races the run against a wait window (`VEIN_CHAT_RUN_WAIT_MS`,
default 60s):

- Run finishes in time → return the result exactly as today (the quick
  inner-loop path stays synchronous — the LLM never has to predict run
  duration).
- Still running → the tool returns immediately with
  `{ status: "running", detached: true, runId, workflow, note }` as the tool
  RESULT (so `messages.jsonl` stays well-formed — no dangling tool calls;
  never retro-patch a result), and hands the still-pending promise to the
  host via `AiDeps.detach.onDetach`.

The tool generates the `runId` itself (so the stub can report it) and
passes it into `runWorkflow`. Detached-tool runs are tracked in
`activeRuns` like HTTP-launched detached runs, so `/runs` listings show
"running" rather than "stale".

When `AiDeps.detach` is absent (tests, embedders that don't wire chat),
behavior is unchanged: fully synchronous.

### 2. Completion wakes the chat: the notifier

`src/ai/notifier.ts` — `createChatNotifier({ chatStore, maxAutoTurns,
startTurn })`, owned by `createVein`'s chat block:

- `deliver(chatId, text)` — called when a detached run settles. If the chat
  has a live turn **in this process**, queue; else launch a notification
  turn now.
- `turnStarted(chatId)` / `turnEnded(chatId)` — liveness bookkeeping, hooked
  into `launchChatTurn`. `turnEnded` drains the queue: all notifications
  that arrived during the turn are delivered in ONE wake-up turn (two A/B
  arms finishing close together → one turn that sees both).
- Liveness is an in-process set (mirroring `activeRuns`), NOT
  `meta.status` — a crashed process leaves `status: "live"` stale, and
  pending notifications die with the process anyway (same crash posture as
  runs; no durable delivery in v1).

A notification turn: append a **user-role** message
`[run-notification] workflow "X" run <id> finished: success in 41m …` (slim:
status, duration, error message, output truncated to ~2000 chars — the agent
has `get_run` for full detail), bump `currentTurn`, set meta live, launch
via the same `launchChatTurn`.

### 3. Runaway guard: consecutive-auto-turn cap

`ChatMeta.autoTurns` counts notification-triggered turns since the last
human message; `POST /chat` resets it to 0. When a notification arrives and
`autoTurns >= maxAutoTurns` (`VEIN_CHAT_MAX_AUTO_TURNS`, default 10), the
notification message is still appended to the transcript (the next human
turn sees it) but NO turn is launched — a runaway loop parks instead of
burning tokens all night past its budget.

### 4. Prompt contract

System prompt + tool description teach the model: long runs auto-detach;
when you get the detached stub, finish your turn normally (say what you're
waiting on) — do NOT poll `get_run` in a loop; a `[run-notification]`
message will start your next turn.

### 5. Web UI follow-along

`ChatFlyout` currently tails one turn and stops. Server-initiated turns
mean new turns can appear while the flyout is open (or closed). Add:

- After a streamed turn finishes, and on a light poll while idle (~4s,
  only when the flyout is open), re-fetch `GET /chat/:id`; if
  `currentTurn` advanced past the last seen turn, re-render the transcript
  and attach to the live turn's stream.
- Render `[run-notification]` user messages as a distinct "notice" bubble.

## Non-goals (v1)

- Durable notification delivery across restarts (crash loses in-flight runs
  anyway).
- Detached mode for `run_step` (single steps are the fast inner loop).
- Cancel-run tooling for the agent.
- A separate `run_workflow_detached` tool — auto-upgrade keeps one tool.

## Touch list

- `src/ai/notifier.ts` (new): notifier + `formatRunNotification` + tests.
- `src/ai/tools.ts`: `run_workflow` race + detached stub.
- `src/ai/prompts.ts`: `AiDeps.detach` type; prompt updates.
- `src/chat-store.ts`: `ChatMeta.autoTurns`.
- `src/createVein.ts`: `chatRunWaitMs` / `chatMaxAutoTurns` options + env,
  notifier wiring into `launchChatTurn` + `POST /chat`, `onDetach` →
  `activeRuns` + `deliver`.
- `web/src/components/ChatFlyout.tsx` + `styles/components.css`: follow
  server-initiated turns; notice bubble.
- `AGENTS.md`: env table + key-concepts entry.
