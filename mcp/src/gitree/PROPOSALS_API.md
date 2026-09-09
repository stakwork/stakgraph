# Concept Proposals API

Pending create/update/delete/merge changes to Concepts, stored in the swarm graph and reviewed by a human before they land. Base URL is the swarm (port 3355), same auth as other `/gitree/*` endpoints (`x-api-token` header).

## Proposal shape

```jsonc
{
  "id": "uuid",
  "action": "create | update | delete | merge",
  "status": "pending | accepted | rejected",
  "repo": "owner/repo",

  // targets (update/delete/merge). For merge: conceptId is absorbed into mergeIntoConceptId.
  "conceptId": "owner/repo/auth-system",
  "mergeIntoConceptId": "owner/repo/authentication",

  // proposed content
  "name": "…",              // create only
  "description": "…",       // optional description change
  "documentation": "…",     // full proposed markdown docs
  "parent": "…",            // create only, optional parent concept id

  // snapshots captured server-side at propose time — use for diff rendering
  "baseDocs": "…",          // docs of the edited concept when the proposal was made
  "absorbedDocs": "…",      // merge only: docs of the concept that will be deleted

  // review metadata
  "rationale": "…",         // why — show this to the reviewer
  "source": "…",            // producer, e.g. "pr-merge", "dedup-workflow", "agent"
  "prNumbers": [123],
  "sessionIds": ["…"],

  // decision (set once decided)
  "decidedBy": "…",
  "decisionReason": "…",
  "decidedAt": "ISO date",
  "createdConceptId": "…",  // accepted create: the minted concept id (deep-link target)

  "createdAt": "ISO date"
}
```

**Diff rendering:** for `update`/`merge`, diff `baseDocs` → `documentation`. For `create`, there is no before. For `delete`, show `baseDocs` as removed. For `merge`, also show `absorbedDocs` (the concept being folded in and deleted).

## Endpoints

### `GET /gitree/proposals?repo=owner/repo&status=pending`
Both params optional. → `{ proposals: Proposal[], count, repo }`, newest first. Use `status=pending` for the review queue / badge count.

### `GET /gitree/proposals/:id`
→ `{ proposal }` or 404.

### `POST /gitree/proposals`
Create a proposal (for producers; the UI likely doesn't call this). Body: `action` plus the relevant fields above (`baseDocs`/`absorbedDocs` are captured server-side — do not send). Validates targets exist (404) and, for create, that the name doesn't collide (409). → `{ status: "success", proposal }`.

### `POST /gitree/proposals/:id/accept`
Body: `{ decidedBy?: string, force?: boolean }`.
Applies the change to the Concept graph, then stamps the proposal accepted. → `{ status: "success", proposal }`.

Errors:
- `404` — proposal (or its target concept) no longer exists
- `409 { status }` — already accepted/rejected (idempotency guard)
- `409 { code: "stale_base", conceptId }` — the concept's docs changed since the proposal was made. Surface this in the UI ("concept has changed since this was proposed — re-review") and offer a re-fetch + explicit force retry (`force: true` overrides).

### `POST /gitree/proposals/:id/reject`
Body: `{ decidedBy?: string, reason?: string }`. No graph side effects. Same 404/409-already-decided errors. → `{ status: "success", proposal }`.

## Accept semantics (what the UI can promise the user)

- **create** → new Concept minted exactly like `POST /gitree/create-concept-direct` (id = `owner/repo/<slug-of-name>`, in `createdConceptId`).
- **update** → target's documentation (and description, if proposed) replaced.
- **delete** → target Concept deleted.
- **merge** → surviving concept gets the merged docs + unioned PR/commit provenance; absorbed concept is deleted.

Decided proposals are kept in the graph (audit trail) — the list endpoint with `status=accepted|rejected` is a per-concept change history via their `conceptId`/`createdConceptId`.

## Agent tools (producer side)

Swarm agents can file and inspect proposals without HTTP: `propose_concept_change` and `list_concept_proposals` in `repo/tools.ts` (opt-in via tools config, like `create_triplet`). They call the same `createProposal`/`listProposals` service functions as the endpoints, with `source: "agent"`. Agents never get accept/reject — deciding is human-only.

## Hive integration notes

- Follow the existing thin-proxy pattern of `src/app/api/learnings/concepts/*` (swarm URL + `x-api-token` via `getSwarmConfig`, `requireReadAccess` for GET, `requireMemberAccess` for accept/reject).
- Pass the Hive user's identifier as `decidedBy` on accept/reject.
- Diff UI: reuse `computeUnifiedDiff` (`src/lib/diff/unifiedLineDiff.ts`) and the `UnifiedDiffView` rendering from `ProposalCard.tsx` — same contract as the chat proposal cards (`oldStr` = `baseDocs`, `newStr` = `documentation`).
- After an accepted create, deep-link to `/w/{slug}/learn?concept={createdConceptId}`.
