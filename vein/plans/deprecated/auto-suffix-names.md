# Auto-suffix workflow & step names

THIS PLAN WAS WRONG! WE CREATED createWorkflow separately from "publish" flow (those are separate intentions)

## Problem

`publishWorkflow` and `publishStep` silently overwrite existing entries
when called with a duplicate name. The AI builder hits this whenever
the model picks a name that already exists.

## Decision

Always auto-suffix on collision. No `mode` param. A future
`edit_workflow` tool will handle "publish a new version of an existing
workflow" — that is a different intent and gets its own tool.

## Changes

### `src/workspace.ts`

- `publishWorkflow(name, version, content, description)` → returns
  `string` (the actual name used). If `<workspace>/workflows/<name>/`
  already exists, try `<name>-2`, `<name>-3`, ... until a free slot
  is found, then write there.
- `publishStep(name, code, description)` → same treatment. Returns
  the actual name used. Collision check is against
  `<workspace>/steps/custom/<name>.ts`.
- Collision check uses `fs.stat` (or `readdir` of the parent) — not
  the in-memory metadata, since the filesystem is the source of truth.

### `src/ai/tools.ts`

- `create_workflow` execute: capture the returned name and respond
  with `{ ok: true, name: finalName, version: "v1", renamed: finalName !== name, requested: name }`.
  The model will use `finalName` in its reply automatically.

### Callers to update

Anything calling `publishWorkflow` / `publishStep` that ignores the
return value is fine (string return is additive). Check:

- `src/server.ts` — POST /workflows, POST /steps. These should
  return the final name in the HTTP response so the UI can navigate
  to the right place.
- `web/src/api.ts` — update return types if the endpoints now return
  the resolved name.
- Tests in `*.test.ts` that publish workflows with known names —
  should still pass since they don't create collisions.

## Tool description tweak

Update `create_workflow` description to mention the auto-suffix
behavior briefly, so the model doesn't get confused if it sees a
`-2` suffix come back:

> Create and publish a new workflow from YAML. If the name already
> exists, a numeric suffix is appended (e.g. `send-email-2`). The
> response includes the final name used.

## Out of scope

- `edit_workflow` tool (new version of existing workflow) — separate task.
- UI affordance for "this was renamed from X" — the chat reply
  already conveys it; sidebar just shows the final name.

## Verification

- `npm test` (199 tests should still pass)
- Manual: ask the AI builder to create two workflows with the same
  name in one session, confirm the second comes back as `<name>-2`.
- Manual: `ls workspace/workflows/` should show both dirs.
