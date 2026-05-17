# Phase 5 — Trust Registry: Sourcing and Admin Surface

> Concrete shape for how a Bifrost+plugin instance learns which orgs
> it trusts. Companion to `cryptographic-identity.md` ("Trust
> registration") and `phases/phase-1-reconciler.md` (which assumes a
> claimed swarm).
>
> Phase 1 stood up per-workspace Bifrosts with VKs and Customers. The
> identity doc introduced the trust registry as a concept. This doc
> says where the registry lives, who can write to it, and how it gets
> populated across the three deployment shapes the plugin supports.

## What the plugin needs at steady state

Per `cryptographic-identity.md`, the plugin verifies every incoming
macaroon against:

```
trust_registry := {
  org_id → {
    pubkey:                   <secp256k1 pubkey or multisig policy>,
    issuer_url:               <URL for revocation pulls>,
    revocation_poll_seconds:  <int>,
    grace_pubkeys:            [ <previous-root keys during rotation grace> ],
    grace_until:              <RFC3339 deadline for grace pubkeys>,
  }
}
```

That's the state the plugin must hold. **How it gets there is what
this doc decides.**

### Deliberately NOT in the trust registry

The trust registry's job is purely **"do I trust this org's
signatures?"** It does not carry per-org budgets, workspace
allow-lists, or any other policy that would amount to a "swarm-side
backstop" on what the org's users can do.

Why: the macaroon shape (`phase-4-macaroon-shape.md`) already
encodes those concerns at the right layer:

- `UserAuthorization.Permissions.Workspaces` — org grants user a set
  of workspaces (org-signed)
- `UserAuthorization.Permissions.Agents` — org grants user a set of
  agents (org-signed)
- `Invocation.MaxCostUSD` / `MaxSteps` — user picks per-invocation
  budget (user-signed)
- `AttenuationCaveats.MaxCostUSD` / `MaxSteps` — sub-agent narrowing
  (HMAC-chained)

If a swarm operator wanted a defense-in-depth cap on a _trusted_
org (cross-org scenario where Swarm B trusts Org A but doesn't want
a runaway Org A agent burning Swarm B's compute), that's a separate
swarm-config concern, not a trust-registry concern. We may add it
later as a clearly-named `swarm_limits` field; we are **not**
front-loading it in phase 5.

If an org ever needs to cap user-chosen invocation budgets (e.g. "this
user can sign invocations up to $X"), the right place is a new
field inside `UserAuthorization` (org-signed) — not the per-swarm
trust registry. That's an extension of phase 4, not phase 5.

## Three configuration sources

The plugin reads its trust registry from three sources, in this order
of authority at steady state:

1. **Persisted state** — sqlite (or json) file in the plugin's data
   volume. Survives restarts. **Canonical.**
2. **Admin HTTP API** — `/_plugin/trust/*`, authenticated with a
   per-swarm shared bearer secret. Writes update persisted state.
3. **Env-var seed** — `BIFROST_PLUGIN_TRUST` (inline JSON) or
   `BIFROST_PLUGIN_TRUST_FILE` (path to JSON). Read once at startup.
   Seeds persisted state only when persisted state is empty.

Each source maps to a deployment shape:

| Deployment                                | Primary source                                      | Why                                                                                             |
| ----------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Sphinx-swarm + Hive                       | Admin API                                           | Hive decides which orgs trust each swarm; reconciles at workspace creation / grant-access time. |
| Self-host (docker-compose, single-tenant) | Env-var seed                                        | Operator knows the trust set up front; wants declarative config.                                |
| Declarative infra (Terraform / Helm)      | Env-var seed                                        | Trust set lives in version control, matches the spec.                                           |
| Mixed                                     | Env seed for initial set, API for runtime additions | Sensible defaults plus live mutation.                                                           |

## Precedence: "persisted is canonical, env seeds only when empty"

The rule is short and unambiguous:

```
On plugin startup:
  1. Load persisted trust registry from data volume.
     If file doesn't exist → empty registry.

  2. Read env-var preset (BIFROST_PLUGIN_TRUST or _FILE), if set.

  3. If persisted state is empty AND env-var preset is present:
       seed persisted state from env preset, write to disk, log a
       "trust seeded from env" line.

  4. If persisted state is non-empty AND env-var preset is present:
       compare the two.
         - If they match → log "trust env matches persisted" and proceed.
         - If they differ → behavior depends on BIFROST_PLUGIN_TRUST_RECONCILE:
             "ignore" (default) → log a warning, use persisted as-is.
             "overwrite"        → replace persisted with env, log loudly.
             "refuse"           → exit non-zero with a clear error.

  5. Admin API mutations write to persisted state directly.
     They do NOT consult env-var presets.

  6. The verifier only reads from in-memory state loaded from persisted.
     Env vars are never consulted on the hot path.
```

The default reconcile mode is `ignore` because that's the safe choice
for the most common upgrade path: someone set an env var months ago,
the registry has evolved via the API since, and a restart shouldn't
silently revert the API changes. Operators who want strict env-driven
config flip to `overwrite` or `refuse`.

## Env-var preset shape

```
BIFROST_PLUGIN_TRUST='{"orgs":[ … ]}'
```

or, for non-trivial payloads:

```
BIFROST_PLUGIN_TRUST_FILE=/etc/bifrost/trust.json
```

with `/etc/bifrost/trust.json` containing:

```json
{
  "orgs": [
    {
      "org_id": "org_acme",
      "pubkey": "0204abcd…",
      "issuer_url": "https://hive.acme.example.com",
      "revocation_poll_seconds": 60
    }
  ]
}
```

Field semantics match the wire shape of `POST /_plugin/trust` (below).
The plugin parses the JSON once at startup, validates it
(pubkey decodes as a valid compressed secp256k1 point, URLs well-
formed), and either
seeds or compares per the precedence rules.

If parsing fails, the plugin **exits non-zero** with a parse error.
Better to fail to start than to come up with a silently-empty
registry.

## Admin HTTP API

All admin paths live under `/_plugin/trust/` and require
`Authorization: Bearer <shared_secret>`. The shared secret is read
from `BIFROST_PROVISIONING_TOKEN` at plugin startup — the same env
var the existing `/_plugin/admin-credentials` route uses (see
`phase-3-swarm-handoff.md`). One token, one admin surface.

### `GET /_plugin/trust/status`

Returns whether the plugin has any trusted orgs and a high-level
summary. Useful for Hive's reconciler to decide whether to claim.

```http
GET /_plugin/trust/status HTTP/1.1
Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>
```

```json
{
  "claimed":         true,
  "org_count":       1,
  "orgs":            ["org_acme"],
  "seed_source":     "env" | "api" | null,
  "last_modified":   "2026-05-14T12:00:00Z"
}
```

`claimed` is `true` iff `org_count > 0`. `seed_source` records whether
the _initial_ registry came from env-var seed or from an API call;
purely informational.

### `POST /_plugin/trust`

Register a new org or replace an existing one. Idempotent on
`(org_id, pubkey, issuer_url, revocation_poll_seconds)`; calling
twice with the same body is a no-op.

```http
POST /_plugin/trust HTTP/1.1
Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>
Content-Type: application/json

{
  "org_id":                  "org_acme",
  "pubkey":                  "0x04abcd…",
  "issuer_url":              "https://hive.acme.example.com",
  "revocation_poll_seconds": 60
}
```

```json
{ "ok": true, "org_id": "org_acme" }
```

Replacing an existing org's pubkey is allowed via this endpoint in
phase 1 (custodial). In phase 3, key rotation will require an
org-root-signed payload — see "Phase 3 hardening" below.

### `POST /_plugin/trust/:org_id/rotate`

Add a new pubkey for an org while keeping the previous one valid
during a grace window. After grace expires, the old key is dropped.

```http
POST /_plugin/trust/org_acme/rotate HTTP/1.1
Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>
Content-Type: application/json

{
  "new_pubkey":         "0x04ef01…",
  "grace_seconds":      86400
}
```

```json
{
  "ok": true,
  "active_pubkey": "0x04ef01…",
  "grace_until": "2026-05-15T12:00:00Z",
  "grace_pubkeys": ["0x04abcd…"]
}
```

During the grace window, the plugin accepts macaroons signed by either
the active or the grace pubkey. After `grace_until`, only the active
key verifies.

### `DELETE /_plugin/trust/:org_id`

Remove an org from the registry. All in-flight macaroons signed by
that org are rejected on the next call.

```http
DELETE /_plugin/trust/org_acme HTTP/1.1
Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>
```

```json
{ "ok": true, "removed": "org_acme" }
```

### `GET /_plugin/trust/:org_id`

Read a single org's trust entry. Returns the pubkey, issuer_url,
revocation_poll_seconds, and grace state. Useful for Hive to verify
a reconcile landed correctly.

### Failure modes

| Status | When                                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `401`  | Missing or wrong `Authorization` bearer                                                                                                          |
| `400`  | Invalid pubkey encoding, malformed JSON, unknown fields                                                                                          |
| `404`  | `GET`/`DELETE`/`rotate` on an unknown `org_id`                                                                                                   |
| `409`  | `POST` with an `org_id` already present and a non-matching pubkey, when `If-Match` semantics weren't satisfied (future; phase 1 just overwrites) |
| `500`  | Disk write failed or persisted-state corruption detected                                                                                         |

## Persisted state

The plugin writes the trust registry to a single file in its data
volume — same volume that holds `logs.db`. Suggested path:
`/app/data/trust.json` (mirrors Bifrost's existing data path).

Format: JSON with the same shape as the env-var preset, plus
bookkeeping:

```json
{
  "version":        1,
  "seed_source":    "api",
  "last_modified":  "2026-05-14T12:00:00Z",
  "orgs":           [ … same shape as env preset … ]
}
```

Writes are atomic: write-to-temp, fsync, rename. The plugin reads the
file once at startup and holds the registry in memory; subsequent
admin mutations update both memory and disk.

If the file is missing on startup, the plugin starts with an empty
registry (or seeds from env, per the precedence rules). If the file
is present but unparseable, the plugin exits non-zero — operator
must repair manually rather than risk verifying against an
inconsistent registry.

## Sphinx-swarm integration

Sphinx-swarm already wires `boltwall.stakwork_secret` into the gateway
container as `BIFROST_PROVISIONING_TOKEN` — see
`phase-3-swarm-handoff.md` for the existing `bifrost.rs` change that
powers `/_plugin/admin-credentials`. The trust API reuses the same
token; **no new env vars on the swarm path**.

For reference, the existing wiring in `bifrost.rs`:

```rust
// Already in place from phase 3:
if let Some(boltwall) = boltwall {
    if let Some(api_token) = &boltwall.stakwork_secret {
        env.push(format!("BIFROST_PROVISIONING_TOKEN={}", api_token));
    }
}
```

The plugin reads `BIFROST_PROVISIONING_TOKEN` once at startup and uses
it as the admin bearer for all `/_plugin/*` routes (admin-credentials
and trust alike). Sphinx-swarm emits **no** `BIFROST_PLUGIN_TRUST` env
var — trust is managed entirely via the API by Hive.

### Hive's reconciler addition

Phase 1 reconciled `(workspace × user) → (Customer, VK)`. Phase 5
adds a sibling reconciler:

```
reconcile_trust(workspace_id):
    1. Look up org_id    = workspaces[workspace_id].org_id (Hive's data)
    2. Look up bifrost   = workspaces[workspace_id].bifrost_url
    3. Look up secret    = workspaces[workspace_id].stakwork_secret
    4. GET <bifrost>/_plugin/trust/status
       - if 200 and org_id in orgs[] → done
       - if 200 and claimed=false   → POST trust
       - if 200 and other orgs → reconcile (add if missing, optionally remove stale)
       - if 401                    → operational error: secret mismatch
       - if connection refused     → bifrost not ready; retry next sweep
    5. POST <bifrost>/_plugin/trust
         body: { org_id, pubkey, issuer_url, revocation_poll_seconds }
       - 200 → done
       - 4xx → log + alert (operator action)
       - 5xx → retry next sweep
```

**Trigger model** (mirrors phase 1's lazy-only approach):

- On workspace creation, after Bifrost container is healthy, call
  `reconcile_trust(workspace_id)`.
- On user-grant-access, no trust change needed (trust is per-org, not
  per-user) — but the _existing_ phase-1 reconciler still runs for
  the new user's VK.
- Background sweep every N minutes (configurable; default 10m) walks
  every workspace and ensures trust is current. Catches drift after
  org key rotation, new partner orgs added, etc.

**Failure handling.** Same shape as phase 1: log loudly, surface to
ops, no retry queue. The next sweep is itself the retry. If a
workspace's Bifrost is unreachable, macaroon verification fails in
that workspace — which is correct, because the plugin can't verify
without a populated registry.

## Self-host / declarative integration

Drop the env var in your docker-compose / Helm chart / Terraform
spec:

```yaml
# docker-compose.yml
services:
  bifrost:
    image: maximhq/bifrost:latest
    environment:
      BIFROST_PROVISIONING_TOKEN: ${OPERATOR_ADMIN_BEARER}
      BIFROST_PLUGIN_TRUST_FILE: /etc/bifrost/trust.json
      BIFROST_PLUGIN_TRUST_RECONCILE: refuse # if you want strict declarative
      BIFROST_PLUGIN_TRUST_PATH: /app/data/trust.json # optional; this is the default
    volumes:
      - ./trust.json:/etc/bifrost/trust.json:ro
      - bifrost-data:/app/data
```

With `BIFROST_PLUGIN_TRUST_RECONCILE=refuse`, the plugin will refuse
to start if persisted state diverges from the env-supplied file —
exactly what you want when the spec is source of truth.

For pure stateless setups (no persisted data volume), the env-var
path is the only source. The plugin still writes a transient trust
file to its working directory; this is fine because it gets
recreated from env on every restart.

## Phase 3 hardening (preview, not in this phase)

When orgs adopt their own root keys (identity doc, phase 3):

- Trust-registry mutations that change an org's pubkey will require
  the request body to be signed by the _current_ org root key, in
  addition to bearing the swarm's admin token. The plugin verifies
  both. This is the rule "swarm decides who to trust, but the org
  must consent to changes to its own entry."
- `POST /_plugin/trust` for a brand-new `org_id` still works with
  admin-token-only auth — registering a _new_ org is the swarm
  operator's call, not the new org's call.
- `DELETE /_plugin/trust/:org_id` stays admin-token-only: the swarm
  operator can always un-trust an org without that org's consent.

The current phase-5 design accommodates this by treating the
`Authorization: Bearer` check as one layer and leaving room for an
optional `X-Org-Signature` header that the plugin will verify in a
later phase. Phase 5 ignores that header if present.

## Wire-up checklist for phase 5

### Plugin (Go) — DONE

- [x] Plugin reads `BIFROST_PLUGIN_TRUST` / `BIFROST_PLUGIN_TRUST_FILE`
      at startup and applies precedence rules
      → `gateway/internal/trust/persistence.go` (`LoadFromEnv`)
- [x] Plugin reads admin bearer from `BIFROST_PROVISIONING_TOKEN`
      (shared with `/_plugin/admin-credentials`)
      → `gateway/internal/env/env.go` + `adminapi/server.go`
- [x] Plugin persists trust state to the path in
      `BIFROST_PLUGIN_TRUST_PATH` (default `/app/data/trust.json`)
      atomically (write-temp → fsync → rename)
      → `gateway/internal/trust/persistence.go` (`persistLocked`)
- [x] Plugin exposes `/_plugin/trust/*` endpoints
      (status, upsert, get, delete, rotate)
      → `gateway/internal/adminapi/trust.go`
- [x] `bifrost.rs` emits `BIFROST_PROVISIONING_TOKEN` (already wired
      in phase 3 — no change for phase 5)
- [x] Plugin Init() loads trust registry before starting admin server;
      fatal-exits on parse error or `refuse` mode mismatch
      → `gateway/main.go`
- [x] Unit + HTTP integration tests covering precedence modes,
      idempotency, rotation, auth, validation
      → `gateway/internal/trust/registry_test.go`,
        `gateway/internal/adminapi/trust_test.go` (28 tests)

### Hive (TS/Node) — TODO

- [ ] Hive adds `reconcile_trust(workspace_id)` to its workspace
      reconciler, called on workspace-create and on background sweep
- [ ] Hive's data model has `workspaces[wid].org_id` (or whatever
      field names the org→workspace mapping)
- [ ] Hive's secret store already holds `(workspace_id) →
      stakwork_secret` from phase 3 — reuse the same value as the
      trust admin bearer

### Cross-cutting — TODO

- [ ] `enforce_auth_on_inference` and macaroon enforcement remain
      off through phase 5 — trust registry is wired but the plugin is
      still in observability mode for macaroons
- [ ] End-to-end integration test: boot plugin, drive full reconcile
      flow from a fake Hive client, verify persistence across restart

## What this design buys

- **Self-host works without orchestration.** Drop env var, run plugin,
  done. No bootstrap script, no claim RPC, no Hive.
- **Sphinx-swarm works without env vars.** Hive's reconciler claims
  every swarm via the API using the secret it already has. No new
  secret distribution.
- **Declarative infra works without state divergence.** `refuse` mode
  guarantees the spec wins on every restart.
- **Persisted state is canonical** — runtime mutations don't get
  silently reverted by stale env config.
- **One admin surface for everyone.** Sphinx-swarm uses it via Hive;
  self-hosters can use it too if they want to mutate at runtime.
  Same code path, same tests.
- **Future org-signed mutations fit without redesign.** The
  `X-Org-Signature` extension point is reserved now; the trust-
  mutation auth model evolves without breaking phase-5 callers.
