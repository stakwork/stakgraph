# Phase 12 — Transparency Log

> Every accounted LLM call becomes a leaf in an append-only Merkle
> log inside the gateway plugin. Hive polls the signed tree head,
> checks it reproduces from the leaves it holds, and countersigns
> with the org key. The org key never touches the gateway. Once Hive
> has signed a head, nothing below it can be edited or dropped without
> the witnessed roots disagreeing. That is **Part 1**, and it needs
> nothing from any client: any model, SDK, or harness routed through
> the gateway is covered.
>
> **Part 2**, later and opt-in per client, hands the caller a signed
> receipt with an inclusion proof, so a client that keeps receipts can
> also prove a call was logged at all. aieo is the first such client;
> a verifying sidecar is the path for every other harness.
>
> Companion to `phase-6-plugin-enforcement.md` (the accounting sites
> this phase appends from), `phase-5-trust-registry.md` (the org key
> Hive signs with), and `cryptographic-identity.md` (why the plugin
> holds no identity keys). Extends `gateway/auth/` with a second
> cross-language verifier (Merkle proofs, tree heads, receipts)
> shipped in `gatekey`.

## The idea

The gateway is the party we are defending against. Anything it stores
locally — `logs.db`, Redis, a file — proves nothing on its own,
because the operator can edit it. What we can get instead is two
properties:

1. **The past is frozen.** Once a root hash has been signed by someone
   outside the swarm, every leaf below it is fixed. Rewriting history
   means producing a tree that no longer reproduces that root. This
   is Part 1, and it is entirely between the gateway and Hive.
2. **Every call leaves evidence in someone else's hands.** The agent
   holds a signed receipt per call. A receipt whose leaf is missing
   from a later witnessed tree is proof of tampering, not a dispute.
   This is Part 2, and it only holds for clients that keep and verify
   receipts.

A gateway can still lie in real time (fabricate a call, or not log
one). Neither part solves that; see "What this proves".

```
agent (any harness)            gateway plugin                          Hive
───────────────────            ──────────────                          ────
POST /anthropic/v1/messages ─► verify macaroon            (phase 6)
  x-macaroon                   forward to provider
                               account                    (phase 6)
                               append leaf ─► tree ─► leaves.jsonl
                                                  ◄── GET /_plugin/tlog/sth?since=N   (every minute)
                                                  ──► signed STH + leaves[N..)
                                                                                 verify sth sig
                                                                                 root reproduces from leaves
                                                                                 org-sign STH, store

Part 2 adds, for clients that opt in:
  x-tlog-call-id ────────────► stash receipt per stream
◄── x-tlog-receipt ─────────── leaf + inclusion proof + STH      (non-stream)
◄── GET /_plugin/tlog/receipts/{call_id}                         (stream)
verify against bytes sent
```

## What this proves

| Attack                                            | Outcome                                                                                                                                                                  | Part |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| Edit or delete a leaf Hive has already witnessed  | Detected: the served root no longer reproduces from Hive's leaves; Hive refuses to sign and alerts.                                                                      | 1    |
| Edit the stored transcript in `logs.db`           | Detected: `request_sha256` no longer matches the leaf.                                                                                                                   | 1    |
| Gateway stops logging while runs are active       | Flagged, not proven: Hive sees active runs and a tree that does not grow.                                                                                                | 1    |
| Serve a response without logging it               | Rejected by a verifying client in `enforce` mode: no receipt, call fails. Invisible to a non-verifying client.                                                           | 2    |
| Edit a logged call before Hive's next poll        | Detected by a verifying client: the receipt's inclusion proof is against a root the rewritten tree cannot extend. Otherwise the poll interval is the exposure.           | 2    |
| Drop a call the agent got a receipt for           | Detected: the receipt's leaf is absent from every later witnessed tree.                                                                                                  | 2    |
| Fabricate a call by replaying a real macaroon     | **Not covered.** Needs an agent request signature; the leaf reserves the field (see "Not in either part").                                                               | —    |
| Hive and the swarm operator collude               | **Not covered.** A second witness (org-run puller, OpenTimestamps) points at the same endpoint.                                                                          | —    |

Token counts say how much was said. The hashes say what was said. The
leaf carries both.

The Part 1 rows are the ones that matter for "the past cannot be
rewritten", and they hold for every caller. The Part 2 rows are about
completeness: whether the record has everything that happened. That
needs a party other than the gateway to hold evidence per call, so it
can only be as strong as the client that keeps it.

---

# Part 1 — Witnessed log

## What Part 1 decides

- A Merkle log lives in-process in the plugin (`internal/tlog/`).
  Leaves persist as JSONL on the named volume and the tree is rebuilt
  at boot. No Merkle library; RFC 6962 hashing pinned by test vectors.
- One signing key on the gateway, per boot, in memory: the **log
  key**. It attests "this gateway served this head". It is not an
  identity key; compromising it forges nothing that Hive's
  countersignature does not already have to corroborate.
- Leaves are appended from the two existing accounting sites and
  nowhere else, under the same gate as `auth.ApplyToLLMPost`.
- Nothing is read from or returned to the caller. No new request
  headers, no response headers, no client code.
- Hive is the first witness. It pulls, verifies, countersigns with the
  org key, and stores both the signed heads and the leaves.
- Verifiers (Merkle root, inclusion, consistency, STH) ship in
  `gatekey`, fixture-tested against the Go producer exactly like the
  macaroon code.

## The log (`gateway/internal/tlog/`)

### Tree

RFC 6962 SHA-256 Merkle tree:

```
leaf_hash(d)     = SHA256(0x00 || d)
node_hash(l, r)  = SHA256(0x01 || l || r)
```

Keep every level's hashes in memory (one slice per level). Append is
O(log n): push the leaf hash, fold complete pairs upward. The root of
a non-power-of-two size is computed per RFC 6962 §2.1. Inclusion and
consistency proofs are the standard §2.1.1 / §2.1.2 constructions.
Pin root, inclusion, consistency, and the empty tree against the RFC
6962 test vectors.

The tree and the current STH sit behind one package-level
`sync.Mutex`. `MarkAccounted` is only race-free within a request;
appends arrive from concurrent `LLMPost` / `StreamChunk` goroutines.

### Persistence

Every appended leaf is written as one line of canonical JSON to
`BIFROST_PLUGIN_TLOG_PATH` (default `/app/data/tlog/leaves.jsonl`, on
the named volume that already holds Bifrost's own state). Plain write
on the append path, no fsync; `Sync()` once before every STH is
served, so a head Hive signs is never ahead of the disk after a
process crash.

At `Init`, read the file, rehash every line, rebuild the tree. A
corrupt tail line (crash mid-write) is truncated and logged with the
dropped count. The log key is fresh per boot; Hive keys nothing on it
(see "Hive witness"), so a restart is invisible to the chain.

### Log key

secp256k1, generated with `crypto/rand` at plugin `Init`, held in
memory only. Generated **outside** the config block: `pluginlog.Init`
JSON-dumps the whole plugin config at boot, so a key in config would
land in `docker logs`. Never logged. Wire shapes carry only the
compressed-hex public key, as `log_pubkey`. Use the existing `auth/go`
helpers (`EcdsaSecp256k1Sign`, `EcdsaSecp256k1PublicKey`) so
signatures are byte-identical to what `gatekey` already verifies.

Part 1 signs the STH on demand when `/_plugin/tlog/sth` is served,
cached by tree size so repeated polls at the same size return the same
bytes. Part 2 moves to one signature per append so receipts can carry
it. Sub-millisecond either way.

## Leaf

Canonical JSON (JCS, via `auth/go/jcs.go`), every field present,
nulls explicit, so the same object hashes the same in Go and TS:

```json
{
  "v": 1,
  "leaf_id": "<128-bit hex, gateway-generated>",
  "ts": "<RFC 3339, gateway clock>",
  "org_id": "org_acme",
  "user_id": "u_alice",
  "run_id": "r_01H…",
  "agent": "coder",
  "macaroon_sha256": "<hex>",
  "request_sha256": "<hex>",
  "response_sha256": null,
  "model": "claude-sonnet-5",
  "provider": "anthropic",
  "prompt_tokens": 1234,
  "completion_tokens": 56,
  "cost_usd": 0.0123,
  "status": "ok",
  "agent_request_sig": null
}
```

- `leaf_id` is unique per leaf. It is the handle Part 2 receipts and
  the dashboard refer to; it carries no meaning beyond identity.
- `org_id`, `user_id`, `run_id`, `agent` come from `VerifiedClaims`
  (`OrgID`, `UserID`, `RunID`, `AgentName`), never from the caller's
  dim headers.
- `macaroon_sha256` is over the raw `x-macaroon` header string
  (`pluginctx.RawMacaroon`). It binds the leaf to the exact
  authorization chain; an auditor holding the macaroon can re-verify
  org → user → HMAC for any leaf offline.
- `request_sha256` is over the raw HTTP body bytes as received in
  `TransportPre` (`req.Body`). It commits the gateway to the
  transcript at logging time: an auditor with the leaf and the
  `logs.db` row can tell a doctored transcript, and a Part 2 client
  can check it against the bytes it sent. This is what makes the log
  a transcript commitment rather than a cost ledger. `req.Body` is
  empty when bifrost skips the body copy (`fasthttpToHTTPRequest`:
  large-payload mode, or Content-Length above the large-payload
  threshold or unknown). Neither key is ever set in the OSS
  bifrost-http we build, so in practice the body is always there;
  the rule for the empty case is `request_sha256: null`. The call is
  logged but not transcript-committed.
- `response_sha256` is reserved `null` in v1. Raw response bytes exist
  only in `TransportPost` (non-stream) and in no hook for streams;
  filling it is later (see "Not in either part").
- `status` is `ok` or `error`. Errored calls still get a leaf; phase 6
  accounts them as a step too.
- `agent_request_sig` is reserved `null` for the fabrication defence.

## STH

Signed tree head. Canonical JSON; `sig` is the log key's ECDSA
signature over `SHA256(JCS(sth \ sig))`, same construction as the
macaroon layers:

```json
{ "tree_size": 1042, "root_hash": "<hex>", "signed_at": "<RFC 3339>", "sig": "<hex>" }
```

The empty tree is a valid, signed STH: `tree_size: 0`, RFC 6962 empty
root.

## Hot path

`TransportPre` computes `SHA256(req.Body)` next to the existing
`x-macaroon` handling and stashes it via `pluginctx.SetRequestHash`,
skipped when `req.Body` is empty so the leaf carries `null` (see
"Leaf"). The body itself is not retained.

```
LLMPost   (existing gate: claims != nil && MarkAccounted; both the
           hadResp && !isStreamRequest path and the hadErr path)
  → auth.ApplyToLLMPost(...)              phase 6, unchanged
  → tlog.Append(leaf)                     new; synchronous, under the mutex

StreamChunk (existing gate: claims != nil && usage-bearing && MarkAccounted,
             plus the chunk.BifrostError path)
  → auth.ApplyToLLMPost(...)              phase 6, unchanged
  → tlog.Append(leaf)
```

That is the whole hot path. `TransportPost` is untouched in Part 1.
`ApplyToLLMPre` stamps claims in both `enforce_macaroons` modes
whenever the macaroon verifies, so a shadow-mode gateway logs every
valid-macaroon call; only a missing or invalid macaroon yields no
claims and no leaf.

## Witness API: `GET /_plugin/tlog/sth?since=N`

Registered exactly like `/_plugin/admin-credentials`:
`mux.HandleFunc("/_plugin/tlog/sth", bearer(h.sth))` with
`sessionGuard.bearerOnly`. Not `methodMuxedAuth`, not
`cookieOrBearer`: a dashboard cookie must not be able to pull material
Hive then org-signs.

A second, dashboard-facing route, `GET /_plugin/tlog/status`
(`cookieOrBearer`), landed with the status card in PR #1690. It
returns local facts only — enabled flag, tree size, root hash, the
per-boot `log_pubkey`, leaf path, last-leaf time — never a signed STH
or leaves, so a dashboard cookie still cannot pull anything Hive would
org-sign.

`since` is a base-10 integer in `[0, tree_size]`. Anything else → 400.
`since > tree_size` → 409 with the current size in the body (see
"Failure modes"). The handler `Sync()`s the leaf file, then responds:

```json
{
  "sth": { "tree_size": 1042, "root_hash": "<hex>", "signed_at": "…", "sig": "<hex>" },
  "log_pubkey": "<compressed hex>",
  "consistency_proof": ["<hex>", "…"],
  "leaves": [ { "…": "leaf" } ],
  "next": 1042
}
```

- `consistency_proof` is from `since` to `tree_size`; empty when
  `since` is `0` or equals `tree_size`. Hive does not need it (it
  holds every leaf); it exists for light witnesses that hold only
  heads.
- `leaves` are indices `[since, next)`, capped at 5000 per response.
  Hive pages with `since = next` until `next == tree_size`.
- The empty tree returns the signed empty STH, no proof, no leaves.

## Hive witness

All new. Nothing tlog-shaped exists in Hive today.

**Client.** `BifrostPluginClient.getTlogSth(since)` beside
`getTrustStatus`, same Bearer + provisioning-token transport.

**Reconciler.** `src/services/bifrost/tlog-reconciler.ts`. Per swarm:

1. Load the last stored head `(tree_size, root_hash)` for the swarm,
   or `0` if none.
2. `getTlogSth(since = tree_size)`; page until `next == tree_size`.
3. Verify `sth.sig` over `SHA256(JCS(sth \ sig))` with
   `log_pubkey` (`gatekey.verifySth`).
4. Recompute the root over stored leaves plus new leaves
   (`gatekey.rootFromLeaves`) and compare to `sth.root_hash`. Hive is
   a full replica, so this one check proves both that the new head
   extends the old (the old leaves are unchanged and still prefix
   the tree) and that the served leaves are the tree's real leaves.
   No separate consistency verification needed. Full recompute is
   O(n) SHA-256 per poll; fine to well past a million leaves, and a
   frontier can replace it later without a protocol change.
5. Only then: org-sign the STH bytes with the org key
   (`macaroonOrgPrivkey`, custodial phase 1) and store the head and
   the leaves in one transaction.

Any failure in 3 or 4: store nothing, log at error, surface on the
swarm's health. That is the witness refusing.

**Storage.** Two tables, one migration
(`prisma/migrations/<ts>_add_bifrost_tlog/`), same convention as
`20260911141500_add_fluentbit_stats_sample`:

```
BifrostTlogSth   swarmId, treeSize, rootHash, signedAt, logPubkey, sthSig, orgSig, witnessedAt
BifrostTlogLeaf  swarmId, leafIndex, leafHash, leaf (canonical JSON text)   unique (swarmId, leafIndex)
```

One row per witnessed head rather than columns on `Swarm`: the
sequence of heads is the audit timeline, and the leaves are the
history itself. Store the leaf as the canonical string exactly as
hashed, so recompute never depends on a JSON round-trip.

**Cadence.** `src/app/api/cron/tlog-poll/route.ts`, gated by
`CRON_SECRET` bearer or `x-vercel-cron` (same pattern as
`jarvis-pr-links`), behind `TLOG_POLL_CRON_ENABLED=true`, registered
in `vercel.json` at `* * * * *`. Uses only the stored swarm plugin
URL; never a URL from the request. Not `revocation_poll_seconds`;
that field is stored on trust upsert and never scheduled. The poll
interval is the unwitnessed window: anything the gateway does to
leaves younger than the last poll is invisible to Part 1. Tightening
it is Hive's knob and needs no client change.

**Staleness.** Two alerts, both on the swarm's health: the last
successful witness is older than a threshold (Hive unreachable, or
the gateway refusing), and the tree size is unchanged across N polls
while the swarm has active runs. The second is the only Part 1 signal
that a gateway has stopped logging. It flags; it does not prove.

**Org key.** Phase 1 custodial: the org key is already in Hive, so
Hive signs with it directly. Phase 3 (multisig org root) swaps in a
derived audit key the root delegates once, the same envelope pattern
as `user_authorization`. Protocol unchanged; only which key signs.

## `gatekey` additions (`gateway/auth/ts`)

```
tlog/merkle.ts    leafHash, nodeHash, rootFromLeaves, verifyInclusion, verifyConsistency
tlog/sth.ts       verifySth
```

Pure functions, no I/O. Fixtures in `gateway/auth/fixtures/tlog-*.json`
are **generated by the Go implementation** (`go test -update`, a new
flag; the macaroon fixture test has none today), since Go is the
producer here; the mirror of the macaroon fixtures, which TS produces
and Go checks. Both suites load the same files and assert
byte-for-byte: leaf JCS bytes, leaf hashes, roots at every size 0..8,
one inclusion proof per index, consistency proofs for every `(m, n)`
pair, STH signing input, signature.

## Failure modes

| Condition                                         | Behaviour                                                                                                                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Append fails (disk full, bad path)                | Log at error; the provider call still returns; the leaf is missing. Hive cannot tell a missing leaf from a call that never happened. Only a Part 2 client can.                                    |
| Body copy skipped by bifrost (large-payload mode) | Leaf carries `request_sha256: null`; the call is logged but not transcript-committed. Never set in the OSS bifrost-http we build.                                                                  |
| Power loss after STH served, before OS flushed    | Tree at boot is shorter than Hive's last head. Hive's `since > tree_size` → 409. Hive logs at error and stops polling that swarm until an operator resets its stored head. Never silently restart. |
| Hive unreachable                                  | Gateway keeps appending. The unwitnessed window grows. Hive alerts on staleness past a threshold, like revocation staleness.                                                                       |
| `log_pubkey` changes                              | Normal after a restart. Hive verifies by recompute regardless and records the new key on the head it signs.                                                                                       |
| Leaf file has a corrupt tail                      | Truncate to the last complete line at boot, log the dropped count. Hive's next poll 409s if its head was past the truncation; same operator path as power loss.                                   |

## Configuration

```
BIFROST_PLUGIN_TLOG_PATH   default /app/data/tlog/leaves.jsonl
```

Nothing else. The log key is not configurable, the poll cadence is
Hive's, and the page size is a constant.

## Testing

**Go** (`internal/tlog`, `internal/hooks`, `internal/adminapi`): RFC
6962 vectors for root / inclusion / consistency / empty tree;
append-once across `LLMPost` vs `StreamChunk` including both error
paths; concurrent appends under the mutex; rebuild-from-file equals
the in-memory tree; corrupt-tail truncation; `since` 400 and 409; STH
401 without bearer and 401 with a dashboard cookie; STH bytes stable
across polls at the same size and re-signed after an append;
`TransportPre` stashes the body hash and sets none for an empty body.

**TS** (`gatekey`): fixture parity as above.

**Hive**: reconciler stores nothing on a bad `sth.sig` or a root that
does not reproduce; stores head + leaves + org sig on success; pubkey
change with a reproducing root is accepted; 409 halts polling for that
swarm; the flat-tree alert fires only when the swarm has active runs.

No E2E in this part.

## Implementation order

1. `internal/tlog`: tree, persistence, key, STH. Fixture generator.
   Unit tests.
2. Hooks: `TransportPre` body hash; `LLMPost` / `StreamChunk` append.
3. `/_plugin/tlog/sth`.
4. `gatekey` merkle + STH verifiers + fixture parity tests.
5. Publish `gatekey` 0.2.0 to npm. Hive pins `^0.1.1` and consumes it
   from npm, so step 6 cannot start until this lands.
6. Hive: client, reconciler, tables, cron, staleness alerts. Ships
   dark behind `TLOG_POLL_CRON_ENABLED`.

Part 1 is complete and useful on its own. Nothing in it changes when
Part 2 lands; Part 2 only adds.

---

# Part 2 — Receipts (later, opt-in per client)

## Why this is separate

The gateway cannot enforce receipts on itself. It is the adversary,
and "enforce" means refusing a response that arrived without proof it
was logged; only the party receiving the response can refuse it. So
receipt verification is inherently per client, and the gateway side
has to stand alone, which Part 1 guarantees. Any model or harness
routed through the gateway gets Part 1 with no changes. Part 2 is what
a client adds when it wants completeness as well.

What a verifying client gains over Part 1 is the three Part 2 rows in
"What this proves": a call served without a leaf, a leaf rewritten
before Hive's poll, and a leaf dropped after the fact all become
detectable at the client. A non-verifying client keeps exactly what
Part 1 gives it.

## Receipt

Returned to the caller:

```json
{
  "leaf": { "…": "leaf object as above" },
  "leaf_index": 1041,
  "inclusion_proof": ["<hex>", "…"],
  "sth": { "…": "STH object as above" },
  "log_pubkey": "<compressed hex>"
}
```

One signature per receipt. The leaf is attested by its inclusion
proof against the signed root; a separate leaf signature would add
nothing. Because append is synchronous and in-process, the inclusion
proof is available immediately. There is no SCT-style promise and no
merge delay. STH signing moves from on-demand to once per append so
the receipt can carry the head that includes it.

## Hot path additions

`TransportPre` additionally reads `x-tlog-call-id`, caller-chosen
128-bit hex, via a new `pluginctx.SetCallID`. **Optional.** Used only
as the lookup key for the stream poll; trusted for nothing else. A
caller that omits it still gets a leaf, and a non-stream caller still
gets the header. Only the stream poll needs it.

The receipt map joins the tree and the STH behind the package mutex.

### Non-stream

```
LLMPost   (same gate as Part 1)
  → auth.ApplyToLLMPost(...)
  → receipt := tlog.Append(leaf)           Append now returns the receipt
  → pluginctx.SetReceipt(ctx, receipt)

TransportPost
  → if r := pluginctx.Receipt(ctx); r != nil {
        resp.Headers["x-tlog-receipt"] = json(r)
    }
```

`TransportPost` is the only hook with a mutable `*schemas.HTTPResponse`
whose headers reach the client — for non-stream calls. In transports
v1.6.2 it also fires for streams, deferred after the stream ends, on a
captured copy with `applyResponse=false`, so a header written there
goes nowhere. `SetReceipt` is called only from the non-stream `LLMPost`
branch, and `TransportPost` never consults the stream stash. The
comment in `hooks/transport_posthook.go` saying the hook does not fire
for streams is stale; fix it in the same PR.

### Stream

```
StreamChunk (same gate as Part 1)
  → auth.ApplyToLLMPost(...)
  → receipt := tlog.Append(leaf)
  → tlog.Stash(claims.UserID, callID, receipt)      only when a call id was sent
```

There is no in-band place for a stream receipt: stream headers are
flushed before the usage chunk exists, the deferred post hook does
not apply its response, the chunk hook is 1:1 and its converters drop
`ExtraFields`, and plugin responses have no trailers. Confirmed
against transports v1.6.2 (`handlers/inference.go` `runCompleter`,
`handlers/middlewares.go` `runTransportPostHooksCaptured`): the
deferred post hook runs before `[DONE]`, but only an *error* it
returns is emitted, as an `event: error` frame. Re-confirm on the
next `BIFROST_VERSION` bump; a way to append one SSE event would
delete the poll below. So the client polls for it after the body ends.

`Stash` keys an in-memory map by the struct `{user_id, call_id}`. The
`user_id` comes from the verified macaroon, so one caller cannot read
or overwrite another's entry; no HMAC and no extra key. Entries are
deleted on read and swept after 10 minutes.

### `GET /_plugin/tlog/receipts/{call_id}`

Registered in `registerRoutes` under the prefix
`/_plugin/tlog/receipts/`. Auth is the macaroon, not the bearer and
not the dashboard cookie:

1. Read `x-macaroon`; `auth.Evaluate`. Missing or invalid → 401, even
   when `enforce_macaroons` is off. Do not reuse `ApplyToLLMPre`'s
   shadow pass-through here.
2. Look up `{claims.UserID, call_id}`. Absent → 404. Never 403; don't
   enumerate.
3. Return the receipt and delete the entry.

No rate limiter in v1. `Evaluate` already costs the caller a full
macaroon verification per poll, and there is one poll per streamed
call.

## Clients

The gateway side stands alone. A client that wants receipts does three
things: sends a call id, hashes the body it sent, and verifies the
receipt with `gatekey`. Two ways to get that.

### aieo, the first verifying client

`TLOG_RECEIPTS=off | warn | enforce`, default `off`. Hive sets it per
spawn once the swarm runs a Part 2 gateway image. `enforce_macaroons`
is independent: a shadow-mode gateway returns receipts for every
valid macaroon. Only a missing or invalid macaroon yields no claims,
no leaf, and no receipt; `enforce` fails those calls by design, in
either gateway mode.

Enforcement runs only when the call is gateway-routed
(`LLM_GATEWAY_URL` set, or a caller `baseUrl`). Direct provider calls
never see any of this.

`buildTimeoutFetch` in `provider.ts` today returns the `Response` as
soon as headers arrive and `callModel` consumes the body from there.
It gains, for gateway-routed requests only:

1. Generate a 128-bit hex `x-tlog-call-id` per attempt (the wrapper
   re-sends on connect-phase timeouts; a first attempt may still have
   reached the gateway, and a reused id would collide in the stash),
   merge into `init.headers`, compute `sha256(init.body)`. The AI SDK
   always sends a JSON string body, which is what the gateway hashes.
2. **Non-stream** (`Content-Type` not `text/event-stream`): read
   `x-tlog-receipt`, then
   `gatekey.verifyReceipt(receipt, { requestSha256 })`. Missing,
   malformed, bad signature, bad inclusion proof, leaf
   `request_sha256` mismatch or `null` → `warn` logs, `enforce`
   throws.
3. **Stream:** wrap `response.body` so that on reader EOF the wrapper
   GETs `{gatewayRoot}/_plugin/tlog/receipts/{callId}` with the
   `x-macaroon` from `init.headers`, then verifies as in 2.
   `gatewayRoot` is the request URL's origin: the wrapper serves
   `/_plugin/*` on the same public port as the provider routes
   (`gateway/wrapper/main.go`, `pluginPathPrefix`). Poll or
   verify failure → `warn` logs, `enforce` errors the stream. Partial
   tokens may already have been yielded; that is the cost of flushed
   stream headers.
4. **Evidence.** On success log one line:
   `[tlog] receipt idx=<n> size=<s> root=<hex8> id=<leaf_id>`.
   Run logs are already collected, so receipts leave the agent
   process without new plumbing. Forwarding them to Hive is later.

The wrapper lives in `getModel`, which already hands it to every
provider client it builds (anthropic, openai, openrouter, xai, and
google via the OpenAI-compat route), so enforcement covers every
gateway-routed provider, not only Anthropic. The `createAnthropic`
instances in `tools.ts`, `search.ts`, and `fetch.ts` exist only to
reach `anthropic.tools.*`, which are static descriptors
(`provider.tools = anthropicTools` in `@ai-sdk/anthropic`); they
never make an HTTP call and need no wrapper.

Dependency: `gatekey`, which brings `@noble/curves`, `@noble/hashes`,
and `canonicalize`. aieo has no crypto deps today, and consumes
`gatekey` from npm like every other dependency.

### A verifying sidecar, the general path

A small proxy on the agent pod, in front of the gateway URL, that does
what the aieo wrapper does for any harness: adds the call id, hashes
the body, verifies the receipt, and fails the response on a bad or
missing one. Claude Code via `ANTHROPIC_BASE_URL`, or any SDK pointed
at the sidecar, then gets the same guarantee aieo gets. Same `gatekey`
verifier, or `auth/go` if it is written in Go. Not specced here; it
is what to build the first time a non-aieo harness needs enforcement.

## `gatekey` additions

```
tlog/receipt.ts   verifyReceipt
```

Fixtures extend the Part 1 set with one receipt per index.

## Failure modes

| Condition                                         | Behaviour                                                                                                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Append fails (disk full, bad path)                | As Part 1, plus: no receipt. aieo `enforce` fails the call. Fail-open on the gateway, fail-closed at the client: the client is the enforcer.                               |
| Body copy skipped by bifrost (large-payload mode) | As Part 1, plus: aieo `warn` logs, `enforce` rejects the `null` hash. Fail closed: a gateway that hands back a null hash has not committed to the transcript.              |
| Restart between usage chunk and stream poll       | Receipt map is in-memory; the poll 404s; aieo fails the call. Accepted for v1. The leaf itself is on disk and in the tree.                                                |

No new configuration. The receipt TTL is a constant.

## Testing

**Go**: receipts GET 401 without a macaroon even with enforce off; 404
on another user's `call_id`; delete-on-read; `TransportPost` sets
`x-tlog-receipt` only when `LLMPost` set a non-stream receipt on
context, never from a stream's stash; `TransportPre` stashes
`call_id`; a stream without a call id appends but stashes nothing;
STH re-signed per append.

**TS** (`gatekey`): receipt fixture parity.

**TS** (aieo): no-op when not gateway-routed; `warn` never throws;
`enforce` rejects missing / malformed / bad-sig / bad-proof / wrong
`request_sha256` or a `null` one; stream wrapper polls after EOF and
rejects on 404 or a bad receipt; every provider client `getModel`
builds receives the wrapper.

## Implementation order

1. `internal/tlog`: receipt, stash, per-append STH signing.
2. Hooks: `TransportPre` call id; `LLMPost` `SetReceipt`;
   `TransportPost` header; `StreamChunk` stash; receipts GET.
3. `gatekey` `verifyReceipt` + fixtures; publish.
4. aieo: wrapper, flag. Ships with `TLOG_RECEIPTS` defaulting to
   `off`; Hive flips swarms to `warn`, then `enforce`.
5. Sidecar, if and when a non-aieo harness needs enforcement.

---

## Not in either part

- **`response_sha256`.** Non-stream needs the append finalized in
  `TransportPost`, where the raw response bytes are. Streams need the
  client to hash what it received and the gateway to hash what it
  sent, which the chunk hook cannot see. Field reserved.
- **`agent_request_sig`.** Ephemeral per-invocation Ed25519 key,
  pubkey committed as an attenuation caveat, signature over
  `request_sha256`. Closes fabrication. Field reserved.
- **Second witness.** An org-run puller, or an OpenTimestamps
  submission of each org-signed head. Same endpoint, another caller;
  this is what the `consistency_proof` is served for.
- **Receipt forwarding** (after Part 2). aieo posting receipts to
  Hive so evidence leaves the agent process in real time rather than
  via run logs.
- **Redis-backed receipt map** (after Part 2), so a restart mid-stream
  does not 404 the poll.
- **Dashboard.** A provenance badge on `/runs/:id` ("witnessed at
  head N") and, after Part 2, a per-run receipt list. (The `TlogCard`
  on the Dashboard is not this: it renders the unsigned
  `/_plugin/tlog/status` facts, not witness state.)
- **Actions the gateway never sees** (git push, file writes). The
  leaf format generalizes; the submit path does not exist yet.
