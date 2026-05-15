# Phase 3 — Swarm Handoff: Provisioning Bifrost Admin Credentials

> Concrete plan for how a fresh stakgraph-gateway container deployed
> by sphinx-swarm gets its Bifrost admin credentials into Hive's DB
> so Hive's phase-1 reconciler can call `/api/governance/*`.
>
> Companion to `phase-1-reconciler.md` (which assumes Hive already has
> the admin user + password) and the gateway README's "Authentication"
> section (which describes how the gateway image's auth is enforced).

## Status

The gateway image **already** ships with everything it needs for this
to work:

- `data/config.json` declares `auth_config` referencing
  `env.BIFROST_ADMIN_USER` / `env.BIFROST_ADMIN_PASS`. Bifrost
  resolves these on first boot, bcrypt-hashes the password, and
  persists the hash in `config.db`. The `env.*` reference is kept
  alongside the hash so a rebuild with a different value picks up
  the new password automatically (verified by Bifrost's own test
  `password_change_flushes_existing_sessions`).
- The plugin (`main.go`) runs an in-process HTTP server on
  `127.0.0.1:8189` exposing `GET /_plugin/admin-credentials`
  (Bearer auth, token = `$BIFROST_PROVISIONING_TOKEN`). The
  endpoint returns `{"admin_username":"…","admin_password":"…"}`.
- The wrapper binary (`wrapper/main.go`) fronts both bifrost-http
  (loopback:8181) and the plugin HTTP server (loopback:8189) on a
  single public port, so `/_plugin/*` is reachable via the same
  TLS-terminated host:port as the dashboard.
- The gateway repo's smoke tests confirm all of the above end-to-end
  including password rotation (a rebuild with a different
  `BIFROST_ADMIN_PASS` cleanly invalidates the old password and
  starts accepting the new one).

What's **not yet done** is the swarm-side wiring and the hive-side
consumption. This document specifies both.

## The handoff in one diagram

```
  sphinx-swarm provisions
  the gateway container
  ───────────────────────
                                   shared
                                  stakwork
                                   secret
                                     │
                  generates          │            stores encrypted
                  random pw          │              in DB once
                       │             │                  │
                       ▼             ▼                  ▼
                ┌──────────────────────────┐    ┌──────────────┐
swarm super  ─► │  stakgraph-gateway        │    │   Hive       │
admin API    ─► │  ─ BIFROST_ADMIN_USER     │    │              │
exposes the  ─► │  ─ BIFROST_ADMIN_PASS     │    │  resolveBifrost
admin user      │  ─ BIFROST_PROVISIONING_  │    │  (lazy on    │
+ pubkey URL    │      TOKEN (= stakwork    │    │   first call) │
                │      secret)              │    │      │       │
                │                           │    │      │ first │
                │  config.json auth_config  │    │      │ call  │
                │  → env.BIFROST_ADMIN_*    │    │      ▼       │
                │  (bifrost bcrypts + saves)│    │ GET /_plugin/│
                │                           │ ◄──┤ admin-       │
                │  plugin HTTP server       │    │ credentials  │
                │  → echoes the same values │    │ Bearer ─token│
                │    via /_plugin/admin-    │ ──►│              │
                │      credentials          │    │ saves +      │
                │                           │    │ encrypts to  │
                └──────────────────────────┘    │ swarm.bifrost*│
                                                 │ columns      │
                                                 └──────────────┘
                                                        │
                                                        ▼
                                                 phase-1 reconciler
                                                 calls /api/governance
                                                 with Basic auth
                                                 (admin / decrypted pw)
```

The shared `stakwork_secret` is the only pre-existing credential. It
already flows from swarm → boltwall → repo2graph (see
`sphinx-swarm/src/images/repo2graph.rs:97`) and is held by Hive's
backend. We pass the same value to the gateway as
`BIFROST_PROVISIONING_TOKEN`. Hive presents it as a Bearer token; the
plugin verifies in constant time and returns the admin pair.

**Why a separate provisioning token instead of just reusing the
admin password directly?** Hive cannot know the admin password until
it has been generated, which happens inside the swarm-provisioned
container. The stakwork secret IS pre-known to both sides. So:
- `stakwork_secret`: pre-shared, used **once** by Hive to bootstrap
  itself with the admin password.
- `admin_password`: generated per-swarm, retrieved by Hive, then used
  for all subsequent Bifrost admin API calls.

## Step B1 — sphinx-swarm changes

File: `sphinx-swarm/src/images/bifrost.rs`.

Today the image points at `maximhq/bifrost:latest` (statically linked,
cannot load Go plugins) and only forwards provider API keys. The
changes are:

### B1.1 Switch the image

```rust
impl DockerHubImage for BifrostImage {
    fn repo(&self) -> Repository {
        Repository {
            registry: Registry::DockerHub,
            org: "sphinxlightning".to_string(),     // ← change
            repo: "stakgraph-gateway".to_string(),  // ← change
            root_volume: "/app/data".to_string(),
        }
    }
}
```

(Whatever org/repo we publish to. The image is built from this repo's
`gateway/Dockerfile`.)

### B1.2 Add admin credential fields

Mirror `Neo4jImage`'s pattern (see `sphinx-swarm/src/images/neo4j.rs:48`
where `password: secrets::random_word(32)` is generated at struct
construction):

```rust
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub struct BifrostImage {
    pub name: String,
    pub version: String,
    pub port: String,
    pub host: Option<String>,
    pub links: Links,
    pub admin_user: String,           // ← new, hardcode "admin"
    pub admin_password: String,       // ← new, secrets::random_word(32)
}

impl BifrostImage {
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            // …existing fields…
            admin_user: "admin".to_string(),
            admin_password: secrets::random_word(32),
        }
    }
}
```

Because `BifrostImage` is `Serialize` + `Deserialize`, swarm's existing
state persistence layer automatically saves the generated password
across restarts — no extra code needed.

### B1.3 Inject env vars into the container

In `pub fn bifrost(img: &BifrostImage) -> Config<String>`, accept the
boltwall image as an optional argument (so we can read its
`stakwork_secret`) — mirror what `repo2graph.rs` does — and extend
`env`:

```rust
pub fn bifrost(img: &BifrostImage, boltwall: Option<&BoltwallImage>) -> Config<String> {
    // …existing env vec…

    env.push(format!("BIFROST_ADMIN_USER={}", img.admin_user));
    env.push(format!("BIFROST_ADMIN_PASS={}", img.admin_password));

    if let Some(boltwall) = boltwall {
        if let Some(api_token) = &boltwall.stakwork_secret {
            env.push(format!("BIFROST_PROVISIONING_TOKEN={}", api_token));
        }
    }

    // …rest unchanged…
}
```

The call site in `images/mod.rs` (or wherever `bifrost()` is invoked)
needs to be updated to pass `boltwall`. This is identical to what
`repo2graph()` already does.

### B1.4 Expose admin creds via super-admin details

Hive consumes admin creds via the existing
`GET /super/details?id=<swarm_id>` endpoint that already returns
`x_api_key` (see `sphinx-swarm/src/bin/super/cmd.rs:256` for the
response struct).

Add two fields to the response payload:

```rust
pub struct SuperSwarmData {
    pub address: String,
    pub ec2_id: String,
    pub x_api_key: String,
    pub bifrost_admin_user: String,      // ← new
    pub bifrost_admin_password: String,  // ← new
    // …
}
```

Populate them from the swarm's persisted `BifrostImage` state in
`get_swarm_details_by_id` (`sphinx-swarm/src/bin/super/util.rs:1574`).

**Optional alternative**: don't include them in the existing details
payload (it's already large and surfaces in places we may not want to
leak credentials). Instead, add a separate
`GET /super/bifrost-credentials?id=<swarm_id>` endpoint, also
`x-super-token`-gated. Cleaner separation, one extra route. Either
works; the cost is roughly the same on both sides.

## Step B2 — hive changes

File: `hive/src/services/bifrost/resolve.ts` (the one you wrote
earlier, currently throws if `bifrostAdminUser`/`bifrostAdminPassword`
are missing).

### B2.1 Add a bootstrap function

```ts
// hive/src/services/bifrost/bootstrap.ts
export async function bootstrapAdminCreds(
  workspaceId: string,
): Promise<{ adminUser: string; adminPassword: string }> {
  const swarm = await db.swarm.findUniqueOrThrow({
    where: { workspaceId },
    select: { id: true, swarmId: true, swarmUrl: true /* … */ },
  });
  if (!swarm.swarmUrl) throw new BifrostConfigError(...);

  // Pull the stakwork shared secret from swarm super-admin. This is
  // the same value swarm passes to the gateway container as
  // BIFROST_PROVISIONING_TOKEN, so we can present it as a Bearer
  // token to /_plugin/admin-credentials.
  const stakworkSecret = await getStakworkSecretForSwarm(swarm.id);

  // /_plugin/* lives on the gateway's public port (8181 in dev,
  // whatever swarm exposes in prod), reached via TLS at the same
  // hostname as the rest of bifrost.
  const baseUrl = deriveBifrostBaseUrl(swarm.swarmUrl);

  const res = await fetch(`${baseUrl}/_plugin/admin-credentials`, {
    method: 'GET',
    headers: { Authorization: `Bearer ${stakworkSecret}` },
  });
  if (!res.ok) {
    throw new BifrostConfigError(
      `bootstrap failed: ${res.status} ${await res.text()}`,
    );
  }
  const body = (await res.json()) as {
    admin_username: string;
    admin_password: string;
  };

  // Persist encrypted, same idiom as swarmApiKey
  // (hive/src/services/swarm/db.ts:117).
  const enc = EncryptionService.getInstance();
  await db.swarm.update({
    where: { workspaceId },
    data: {
      bifrostAdminUser: body.admin_username,
      bifrostAdminPassword: JSON.stringify(
        enc.encryptField('bifrostAdminPassword', body.admin_password),
      ),
    },
  });

  return { adminUser: body.admin_username, adminPassword: body.admin_password };
}
```

### B2.2 Lazy invocation in `resolveBifrost`

Today (`hive/src/services/bifrost/resolve.ts:70`):

```ts
if (!swarm.bifrostAdminUser || !swarm.bifrostAdminPassword) {
  throw new BifrostConfigError(...);
}
```

Becomes:

```ts
if (!swarm.bifrostAdminUser || !swarm.bifrostAdminPassword) {
  // Lazy bootstrap. Idempotent — calling it twice on a healthy
  // swarm just reads + re-encrypts the same values.
  const { adminUser, adminPassword } = await bootstrapAdminCreds(workspaceId);
  return { baseUrl, adminUser, adminPassword };
}
```

This is the only consumer change. The existing reconciler
(`reconciler.ts`) and `BifrostClient` need no edits.

### B2.3 Where does `stakworkSecret` come from?

In Hive today it's referenced from a few places (search
`stakworkService().createSecret` — see
`hive/src/app/api/swarm/route.ts:315`) and is the same shared key
swarm uses. Wire `getStakworkSecretForSwarm()` to whatever Hive
already does to access it for `createSecret`; the value doesn't need
to be fetched fresh, it can come from env or from the swarm's
super-admin details payload at swarm-creation time. (If it isn't
already persisted per-swarm, this is the right time to do so.)

## Failure modes and idempotency

The bootstrap is designed to be re-runnable. Each component handles
the "what if I'm called twice" case explicitly:

| Scenario | Behaviour |
|---|---|
| Hive calls `bootstrapAdminCreds` twice in parallel | Both return the same plaintext; second `db.swarm.update` overwrites with the same values. Mild waste, no data corruption. |
| `BIFROST_ADMIN_PASS` is rotated by swarm (new pw injected) | Bifrost re-hashes and flushes sessions on next boot. Hive's stored value is now wrong → all `/api/governance/*` calls 401 → next `resolveBifrost` should detect and re-bootstrap. (See open question below.) |
| `BIFROST_PROVISIONING_TOKEN` doesn't match | Plugin returns 401, `bootstrapAdminCreds` throws `BifrostConfigError`. Hive surfaces the error to the user / logs it. No bootstrap. |
| Gateway is unreachable | Bootstrap throws on fetch. `resolveBifrost` rethrows. Reconciler retries on next tick. |
| Plugin env vars (`BIFROST_ADMIN_*`) not set | Plugin's `startPluginServer` logs a warning and the route doesn't register. `bootstrapAdminCreds` gets a 404 (wrapper returns 503 when the plugin proxy is disabled — see wrapper main.go). Surface that to the operator. |

### Open question: how does Hive detect a password rotation?

A clean detection signal would be: on a 401 from any
`/api/governance/*` call in `BifrostClient`, treat it as
"creds-may-be-stale", re-bootstrap, and retry once. That's a
~10-line change in `BifrostClient.request` (catch 401 → call
bootstrap → retry). Worth doing as soon as we expect password
rotation to be a thing (which is "never on the auto-rollout path but
yes when humans rotate them manually").

For now, document the manual recovery path:
> If the swarm rotates `BIFROST_ADMIN_PASS`, clear the affected
> `swarm.bifrostAdminUser` / `swarm.bifrostAdminPassword` rows in
> Hive's DB. The next `resolveBifrost` call re-bootstraps.

## What this doesn't cover

- **The gateway image itself**. Already done; see the gateway README.
- **Phase 1 reconciler.** Phase 1 assumes Bifrost admin creds are
  present. Phase 3 makes that assumption true.
- **Hot enforcement** (macaroon verification, budget enforcement,
  tool-loop heuristic). Those are phases 4+, still gated on
  `llm-governance-v2.md`.
- **Per-tenant Bifrost provisioning at scale**. We're standing up
  one gateway per swarm. Multi-tenant single-gateway is a separate
  problem (see `agent-registry.md`).

## Rollout order

1. Publish a versioned `stakgraph-gateway:<tag>` image from this repo
   (one-time CI step; the Makefile already has `docker-build`).
2. Land B1 in sphinx-swarm. Wait for it to ship.
3. Land B2 in Hive. Behind a feature flag if desired (`if (env.USE_BIFROST_BOOTSTRAP) { ... }`).
4. New swarms work automatically. For existing swarms, force-re-pull
   the gateway image and restart — Hive will bootstrap on the first
   `resolveBifrost` after restart.
