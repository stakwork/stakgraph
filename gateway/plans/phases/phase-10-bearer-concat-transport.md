# Phase 10 — Bearer-Concat Macaroon Transport

> Concrete plan for accepting `Authorization: Bearer <vk>.<macaroon>`
> (and the equivalent on `x-api-key` / `x-goog-api-key`) as an
> alternate way to deliver both credentials for harnesses that don't
> let callers set custom HTTP headers.
>
> Companion to `phase-4-macaroon-shape.md` (which defines the macaroon
> wire format and currently states the credential is carried in
> `x-macaroon`) and `phase-6-plugin-enforcement.md` (which consumes
> `x-macaroon` after the wrapper has normalized it). This phase
> changes **only the inbound transport shape** — neither the macaroon
> bytes, the verifier, the plugin hooks, nor the agent libraries
> change. The split happens in `gateway/wrapper` before Bifrost or
> the plugin ever sees the request.

## Motivation

The two-header design (`Authorization: Bearer <vk>` plus
`x-macaroon: <macaroon>`) is the cleanest expression of the
two-credential model: a VK that identifies the customer to Bifrost and
a macaroon that authorizes the invocation to the plugin. It works
for any caller that controls its HTTP client at construction time —
the AI SDK, plain `fetch`, the OpenAI/Anthropic/Gemini SDKs all
support default headers on initialization.

It does **not** work for several agent harnesses we need to support.
Some only expose a single "API key" field and route it into
`Authorization: Bearer` (or `x-api-key`) with no escape hatch for
extra headers. Some forward the API key through layers of
configuration that strip unknown headers along the way. The list of
harnesses we want to enable is long enough, and changing each one is
intractable enough, that the platform should accept a wire shape
those harnesses already support.

The concat form fits inside the API-key field every harness already
exposes:

```
Authorization: Bearer sk-bf-abc123.<base64url-macaroon>
```

The wrapper splits on the first `.`, rewrites the originating header
back to just the VK, and injects `x-macaroon` from the right half.
Bifrost sees its expected `Authorization: Bearer sk-bf-…`. The plugin
sees its expected `x-macaroon`. End-to-end verification is unchanged.

## Why the wrapper, not the plugin

Bifrost-http authenticates the VK at HTTP ingress, **before** any of
our plugin's hooks fire. If the incoming `Authorization` header has a
macaroon glued onto the VK, Bifrost's VK lookup fails (no matching
key in the customer table) and the request 401s before
`HTTPTransportPreHook` ever runs. Splitting must happen at a layer
that owns the wire before bifrost-http parses it.

The wrapper at `gateway/wrapper/main.go` is that layer. It already:

- Runs as PID 1 of the container (under tini).
- Owns the single public port (`:8181` by default).
- Reverse-proxies `/_plugin/*` to the plugin's loopback admin server
  and everything else to bifrost-http at `127.0.0.1:8080`.
- Sees every inbound LLM request in raw HTTP form before any Bifrost
  middleware runs.

A `Rewrite` hook on the bifrost-bound `httputil.ReverseProxy` is
exactly where the split belongs. Zero other components change.

## Inbound header set

Bifrost-http accepts the VK on three inbound headers, mirroring what
the three major provider SDKs natively send. Confirmed against
`bifrost/core/schemas/plugin_test.go` (the `x-api-key` and
`x-goog-api-key` exact-match plus case-insensitive lookup tests):

| Header           | Used by                                          | Value shape       |
| ---------------- | ------------------------------------------------ | ----------------- |
| `Authorization`  | OpenAI-compatible SDKs (OpenAI, OpenRouter, AI SDK, …) | `Bearer <vk>` |
| `x-api-key`      | Anthropic SDK                                    | `<vk>` (raw)      |
| `x-goog-api-key` | Google Gemini SDK                                | `<vk>` (raw)      |

The wrapper handles all three. Lookup is case-insensitive (matching
Bifrost's own behavior). No other inbound header carries the VK, so
there is no need to scan further.

The wrapper does **not** need to know which provider the request
targets — path-based dispatch (`/anthropic/v1/...` vs
`/openai/v1/...` vs `/v1beta/models/...`) is bifrost-http's job once
the headers are clean.

## Wire format

```
<original-header-value> := <prefix?> <vk> "." <macaroon_b64url>
```

- `<prefix?>` — `Bearer ` (case-insensitive) for `Authorization`;
  empty for `x-api-key` and `x-goog-api-key`.
- `<vk>` — a Bifrost virtual key. Always starts with `sk-bf-` and
  uses the URL-safe alphabet `[A-Za-z0-9_-]` (no `.`).
- `<macaroon_b64url>` — the same base64url(JCS(macaroon)) value that
  `x-macaroon` already carries, per `phase-4-macaroon-shape.md`.
  Base64url has no `.` in its alphabet.

`.` is unambiguous as a separator because neither the VK alphabet nor
the base64url alphabet contains `.`. The first `.` from the left
splits the value cleanly.

## Wrapper behavior

For every request the wrapper proxies to bifrost-http (i.e. every
request that is **not** under `/_plugin/*`):

1. **Iterate the three known auth headers** in order: `Authorization`,
   `x-api-key`, `x-goog-api-key`. Lookup is case-insensitive.
2. **Find the first one with a non-empty value.** If none, pass
   through (Bifrost will return its own 401).
3. **Strip a leading case-insensitive `Bearer ` prefix** from the
   value (only for `Authorization`).
4. **Locate the first `.` in the stripped value.** If there is no
   `.`, pass through — this is a plain VK using the two-header form
   (or a malformed request; Bifrost will handle either).
5. **Validate the left side as a VK shape.** Must match
   `^sk-bf-[A-Za-z0-9_-]+$`. If not, pass through.
6. **Validate the right side is non-empty.** If empty, pass through.
7. **Check that no `x-macaroon` header is already set.** If one is
   set, pass through — the caller has explicitly used the two-header
   form and we don't overwrite it.
8. **Apply the rewrite:**
   - Set the originating header to just the VK, preserving the
     `Bearer ` prefix if applicable.
   - Set `x-macaroon` to the right side.
9. Stop. Only the first match is rewritten.

Pseudocode (lives in a new file `gateway/wrapper/authsplit.go`):

```go
var authHeaders = []string{"Authorization", "X-Api-Key", "X-Goog-Api-Key"}

// trySplit returns (vk, macaroon, ok) given a raw header value with
// any "Bearer " prefix already stripped.
func trySplit(raw string) (vk, mac string, ok bool) {
    i := strings.IndexByte(raw, '.')
    if i <= 0 || i == len(raw)-1 {
        return "", "", false
    }
    vk, mac = raw[:i], raw[i+1:]
    if !vkPattern.MatchString(vk) {
        return "", "", false
    }
    return vk, mac, true
}

// rewriteAuthHeaders is invoked from the bifrost-bound proxy's
// Rewrite hook. It mutates pr.Out.Header in place.
func rewriteAuthHeaders(pr *httputil.ProxyRequest) {
    if pr.Out.Header.Get("X-Macaroon") != "" {
        return // caller used two-header form; leave alone
    }
    for _, name := range authHeaders {
        v := pr.Out.Header.Get(name)
        if v == "" {
            continue
        }
        stripped, hadBearer := stripBearer(v) // case-insensitive
        vk, mac, ok := trySplit(stripped)
        if !ok {
            return // first present header wins; don't fall through
        }
        if hadBearer {
            pr.Out.Header.Set(name, "Bearer "+vk)
        } else {
            pr.Out.Header.Set(name, vk)
        }
        pr.Out.Header.Set("X-Macaroon", mac)
        return
    }
}
```

The "first present header wins, don't fall through on no-match" rule
matters: if a harness sends `Authorization: Bearer <vk>` with no `.`
AND also happens to send `x-api-key` for some reason, we should not
go hunting in `x-api-key` for a macaroon. The convention is "the
header carrying the credential is the one to rewrite," and we lock
that in by stopping at the first non-empty header.

## What the wrapper does NOT do

- **No macaroon validation.** Shape-checking the right side (e.g.
  attempting base64url decode) is the plugin's job via
  `gateway/internal/auth/verifier.go`. The wrapper is a transport
  rewriter, not a verifier. If the right side isn't a real macaroon,
  the plugin's `auth.Verify` produces the same 401 it does today for
  a malformed `x-macaroon`.
- **No customer lookup.** The wrapper has no Bifrost or trust-registry
  state. Bifrost still resolves the VK to a customer; the plugin
  still cross-checks `claims.user_id == ctx.customer_id`.
- **No new logging surface.** The existing redact rules in
  `gateway/internal/hooks/redact.go` already redact `x-macaroon`.
  The wrapper's own access log (today: `[wrapper]` prefix on stderr
  via `log.New`) does not log header bodies. We add a single counter
  log line "auth header rewritten" with no values, useful for
  observability but leak-free.
- **No effect on `/_plugin/*` traffic.** The split runs only on the
  bifrost-bound proxy branch. Admin endpoints continue to use their
  own bearer scheme (`BIFROST_PROVISIONING_TOKEN`).

## Backward compatibility

Strictly additive. Three regression cases to verify:

| Caller behavior                                    | Wrapper action     | Outcome                |
| -------------------------------------------------- | ------------------ | ---------------------- |
| `Authorization: Bearer <vk>` only                  | Pass through       | Works as today         |
| `Authorization: Bearer <vk>` + `x-macaroon: <mac>` | Pass through (rule 7) | Works as today      |
| `Authorization: Bearer <vk>.<mac>`                 | Split + inject     | New path under test    |
| `x-api-key: <vk>.<mac>` (Anthropic SDK)            | Split + inject     | New path under test    |
| `x-goog-api-key: <vk>.<mac>` (Gemini SDK)          | Split + inject     | New path under test    |
| Plain VK with a coincidental `.` in the suffix     | Pass through (rule 5: VK pattern fails) | No regression |

The "rule 5" case is the one to be most careful about. The VK pattern
match — `^sk-bf-[A-Za-z0-9_-]+$` — anchors at both ends and contains
no `.`, so any pre-existing VK Bifrost ever issued passes the rule-5
gate when used alone and trivially does not look like a concat. If
Bifrost ever changes its VK format to include `.`, this phase needs
revisiting; the test suite below pins down that assumption.

## Size considerations

A typical macaroon is ~1 KB base64url-encoded. After a few layers of
sub-agent attenuation (each link adds a caveat block + 64-char HMAC)
this grows to several KB. With a VK prefix that's still well under
Bifrost's `MaxHeaderBytes` (default 1 MB in `net/http`).

The concern lives at intermediate proxies: nginx, Traefik,
CloudFront, k8s ingress controllers often cap header lines at 4–8 KB
by default. The two-header form has the same total bytes but split
across two headers, which most defaults handle fine; the concat form
puts it all in one. If a deployment fronts the gateway with such a
proxy, that proxy's header-line limit must be raised. This is a
deployment concern to document in the gateway README, not a protocol
issue.

The wrapper itself imposes no additional limit beyond
`http.Server.ReadHeaderTimeout` (already set to 30s) and Go's default
max header bytes (which we explicitly do NOT lower).

## Test surface

`gateway/wrapper/authsplit_test.go`, pure unit tests on the
header-mutation function:

| Case                                              | Expected                                        |
| ------------------------------------------------- | ----------------------------------------------- |
| `Authorization: Bearer sk-bf-x.<mac>`             | Authorization → `Bearer sk-bf-x`; x-macaroon set |
| `Authorization: bearer sk-bf-x.<mac>` (lowercase) | Same; prefix detection case-insensitive         |
| `x-api-key: sk-bf-x.<mac>`                        | x-api-key → `sk-bf-x`; x-macaroon set            |
| `x-goog-api-key: sk-bf-x.<mac>`                   | x-goog-api-key → `sk-bf-x`; x-macaroon set       |
| `Authorization: Bearer sk-bf-x` (no dot)          | Unchanged; x-macaroon absent                    |
| `Authorization: Bearer sk-bf-x.<mac>` + existing `x-macaroon` | Unchanged; existing wins              |
| `Authorization: Bearer not-a-vk.<mac>`            | Unchanged; rule 5 rejects                        |
| `Authorization: Bearer sk-bf-x.`                  | Unchanged; right side empty                      |
| `Authorization: Bearer .<mac>`                    | Unchanged; left side fails VK pattern            |
| No auth header present                            | Unchanged                                        |
| Both `Authorization` and `x-api-key` set, only the first has a dot | First wins, second untouched      |

One integration test in `gateway/scripts/`: a smoke variant that
hits `/anthropic/v1/messages` with the concat form and confirms the
end-to-end LLM call succeeds. The existing
`smoke-test-enforcement.sh` flow continues to use the two-header
form so both paths get coverage on every run.

## Documentation deltas

Touch three files; no protocol document changes substantively:

1. **`phase-4-macaroon-shape.md` §"Encoding choices"** — add a note:
   "The macaroon is carried in `x-macaroon` on the inbound request.
   As a transport convenience for harnesses that don't allow custom
   headers, the macaroon may instead be concatenated into the VK
   credential as `<vk>.<macaroon>`; the gateway wrapper splits and
   normalizes this form into the canonical two-header shape before
   any downstream component sees the request. See
   `phase-10-bearer-concat-transport.md`."
2. **`llm-governance-v2.md` §"The wire protocol"** — add the concat
   form alongside the existing example, with a one-line pointer to
   this phase.
3. **`gateway/README.md`** — update the curl example in the
   "Quickstart" section to show both forms, and add a paragraph in
   the deployment notes about intermediate-proxy header-line limits.

## Rollout

This phase is independent of every other phase and can ship at any
time. No coordination with the issuer, the plugin, the auth library,
or callers is required. New callers that adopt the concat form
benefit immediately; existing callers using the two-header form are
unaffected.

Suggested order:

1. Land the wrapper change with unit tests (single PR).
2. Add the integration smoke variant.
3. Document in the three files listed above.
4. Notify harness owners that the concat form is now available; let
   them migrate at their own pace.

There is no flag gate. The split logic is always on. The cost of
"always on" is a `strings.IndexByte` per request on the bifrost-bound
branch — trivially cheaper than the proxy round-trip the wrapper is
already doing.

## What this design buys

- **Every agent harness can now carry both credentials.** Single API
  key field is sufficient; no custom headers required.
- **No protocol change.** Macaroon bytes, signatures, verifier, hooks,
  Redis keys — all unchanged. The concat form is a transport-only
  affordance, decoded back to the canonical two-header shape before
  any downstream component sees it.
- **No new failure surface.** A malformed concat falls through to the
  same errors a malformed two-header request would produce today.
- **Decoupled from the trust chain.** The wrapper has no
  cryptographic knowledge, no trust-registry access, no signing
  keys. It rewrites bytes by lexical pattern. The phase-4 verifier
  remains the sole judge of macaroon validity.
