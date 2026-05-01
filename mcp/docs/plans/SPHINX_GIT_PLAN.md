# sphinx-git CLI Implementation Plan

## Overview

Build a new Node.js CLI under `mcp/src/sphinx-git/` that wraps the system `git` binary and transparently signs commits/tags with a per-agent ed25519 key. The CLI is intended to replace `git` on Sphinx sandbox AI-agent servers so that every commit, tag, and push is cryptographically attributable to a specific agent (and, through Hive's record of `(user, child) → pubkey`, to the human that owns it) without the agent ever acting as the human.

This is the identity layer only. Budget/metering will come later in a separate effort.

## Goals

- Drop-in replacement for `git` on agent sandboxes (`sphinx-git <anything>` behaves like `git <anything>`).
- Each agent has its own ed25519 keypair plus a `child` index — both supplied externally (e.g. by Hive) at agent creation time.
- Commits and annotated tags are signed with the agent's key using stock SSH-signing so verification works with standard tooling (`ssh-keygen -Y verify`, GitHub's "Verified" badge, etc.).
- `push` refuses to send when `HEAD` is unsigned (v0); the full multi-commit walk is a documented follow-up.
- All other git subcommands pass through with zero overhead.
- No master seed, no networked key derivation, no custom signing crypto in this CLI: we hand a key file to stock `ssh-keygen` (via git's signing path) and let it produce the SSHSIG envelope.

## Non-Goals (for this phase)

- Budget tracking / LLM proxying / Anthropic key management.
- BIP32 / xpub-based offline derivation. Decided against in this CLI; ed25519 has no non-hardened public derivation. Hive (or another upstream registry) is the source of truth for `(user, child) → pubkey`. The CLI just stores and uses what Hive gives it.
- Master-seed handling on the sandbox. The sandbox only ever holds the per-agent privkey.
- A signing daemon / unix-socket sidecar. The agent process holds its privkey directly for MVP.
- Server-side push verification beyond what the receiving git host already does.
- Implementing the SSHSIG envelope ourselves: stock `ssh-keygen` does it.

## Design Decisions

### 1. Identity model

- Three pieces define an agent: `child` (opaque integer assigned by Hive), `privkey` (raw 32-byte ed25519), and the derived `pubkey` (32-byte ed25519).
- `child` is **opaque to this CLI**. It is not auto-incremented, not validated against any local notion of order, and not interpreted in any way. The CLI stores exactly what `new-agent` is given so that third-party verifiers can look up `(user, child) → pubkey` against Hive (or whatever registry binds child indices to users) independently.
- The agent is named `agent-<child>` in commit author/committer fields.
- `pubkey` is derivable from `privkey` on demand via `ed25519.getPublicKey(privkey)` — we do not store it separately.

### 2. Module location

```
mcp/src/sphinx-git/
├── bin.ts              # #!/usr/bin/env node entry; imports and invokes cli.run()
├── cli.ts              # arg routing / subcommand dispatch (exports run())
├── config.ts           # CONFIG_DIR, CONFIG_FILE, KEY_FILE; readConfig / writeConfig / writeSigningKey
├── keys.ts             # privHex → pubHex; build OpenSSH PEM; format identity strings
├── git.ts              # spawnGit helper with stdio inherit + exit-code/signal forwarding
├── commands/
│   ├── new_agent.ts
│   ├── whoami.ts
│   ├── commit.ts
│   ├── tag.ts
│   └── push.ts
└── README.md
```

(No `sign.ts` — stock `ssh-keygen` produces the SSHSIG envelope.)

### 3. `package.json` changes

Add to `mcp/package.json`:

```jsonc
"scripts": {
  // ...
  "sphinx-git": "tsx src/sphinx-git/bin.ts"
},
"bin": {
  // ...
  "sphinx-git": "./build/sphinx-git/bin.js"
},
"dependencies": {
  // ...
  "@noble/ed25519": "2.3.0",
  "@noble/hashes": "1.7.1"
}
```

Keep ESM + Node16 module resolution to match the rest of mcp. Match the existing gitree CLI shebang (`#!/usr/bin/env node`) and `build/<name>/bin.js` output path.

### 4. On-disk layout

```
~/.config/sphinx/                      mode 0700
├── git.json                           mode 0600
└── signing_key                        mode 0600   (OpenSSH private key file)
```

`git.json`:

```json
{
  "child": 7,
  "privkey": "<64-hex-char ed25519 32-byte privkey>"
}
```

- Written/overwritten by `new-agent`. Created with parent dir if missing.
- Both files always rewritten together so they cannot drift.
- If `git.json` is missing or `privkey` is unset, any signing-required subcommand errors with: `no agent configured; run \`sphinx-git new-agent <hex_privkey> <child>\``.
- If `git.json` is present but `signing_key` is missing (e.g. file deleted out-of-band), `readConfig()` regenerates it from `git.json`'s `privkey` before returning. This keeps the two files self-healing and avoids requiring the user to re-run `new-agent`.
- The privkey appears on disk twice (hex in `git.json`, OpenSSH PEM in `signing_key`). That's the tradeoff for letting `ssh-keygen` do the signing for us; both files are mode 0600 in a 0700 dir.

### 5. Subcommands

| Command | Behavior |
|---|---|
| `sphinx-git new-agent <hex_privkey> <child>` | Validate hex privkey decodes to exactly 32 bytes; validate `<child>` is a non-negative integer; derive pubkey; write both `git.json` and `signing_key`. Print `agent-<child>`, the pubkey hex, and the agent's git author string so the human can eyeball-compare against Hive. |
| `sphinx-git whoami` | Read config, derive pubkey, print `child`, pubkey hex, and the formatted git author string. Exit non-zero if no config. |
| `sphinx-git commit ...args` | Run `git commit` with signing config injected (see §6). |
| `sphinx-git tag ...args` | Run `git tag` with signing config injected when an annotated/signed tag is being created (argv contains `-a`/`--annotate`/`-s`/`--sign`/`-u`/`--local-user`/`-m`/`--message`). For lightweight `git tag <name>` invocations, plain passthrough (lightweight tags carry no signature; git silently produces unsigned ones even with `tag.gpgsign=true`). |
| `sphinx-git push ...args` | Pre-flight check (see §7); on success run `git push` with identity env + signing `-c` flags. |
| `sphinx-git <anything else>` | Plain `spawn` of real git with current argv, no env or `-c` modification. |

Routing is done by hand (not via commander) for `commit`/`tag`/`push`/passthrough so git's own flags aren't mis-parsed by commander. Only the sphinx-specific subs (`new-agent`, `whoami`) go through commander:

```ts
const SPHINX_SUBS = new Set(["new-agent", "whoami"]);

const argv = process.argv.slice(2);
const sub = argv[0];

if (SPHINX_SUBS.has(sub))  runSphinxSubcommand(argv);   // commander
else if (sub === "commit") runCommit(argv.slice(1));
else if (sub === "tag")    runTag(argv.slice(1));
else if (sub === "push")   runPush(argv.slice(1));
else                       passthroughGit(argv);
```

### 6. Signing — author identity and git config injection

Identity is set via **environment variables**, not `-c user.name=...`. Env vars take precedence over git config in the cases that matter (rebase, cherry-pick, `commit --amend`), and they don't show up in `ps` listings.

```
GIT_AUTHOR_NAME    = agent-<child>
GIT_AUTHOR_EMAIL   = <pubkey_hex>@agents.sphinx.chat
GIT_COMMITTER_NAME = agent-<child>
GIT_COMMITTER_EMAIL= <pubkey_hex>@agents.sphinx.chat
```

Signing config is injected via `-c key=value` flags on the git invocation (per-process, never written to `~/.gitconfig`):

```
git \
  -c gpg.format=ssh \
  -c user.signingkey=<KEY_FILE>            # absolute path to ~/.config/sphinx/signing_key
  -c commit.gpgsign=true \
  -c tag.gpgsign=true \
  <subcommand> ...args
```

(Note: git config keys are case-insensitive, but lowercase `user.signingkey` is the canonical form. `gpg.ssh.program` is left at its default — system `ssh-keygen` — and is what actually constructs the SSHSIG envelope from the OpenSSH key file.)

Spawn:

```ts
const child = spawn("git", gitArgs, {
  stdio: "inherit",
  env: { ...process.env, ...identityEnv },
});
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 1);
});
```

Exit code is **explicitly forwarded** — `stdio: 'inherit'` does not propagate it on its own.

### 7. Push pre-flight (`commands/push.ts`)

Goal: refuse to push if any commit being pushed lacks our SSH signature.

We check the commit object directly rather than relying on `git verify-commit`, because `git verify-commit` for SSH-signed commits requires `gpg.ssh.allowedSignersFile` to be configured and its exit-code semantics vary across git versions when the file is missing.

Algorithm (v0 — HEAD-only):

1. Read config; derive pubkey hex.
2. Resolve `HEAD` via `git rev-parse HEAD`. If that fails (e.g. empty repo), pass through to `git push` and let git's own error surface.
3. Run `git cat-file commit HEAD` and look in the commit headers for a `gpgsig` line introducing a `-----BEGIN SSH SIGNATURE-----` block. If absent → unsigned.
4. If unsigned, print the offending SHA with a clear message and exit `1` — do not invoke `git push`.
5. Otherwise spawn `git push` with the agent identity env vars and signing `-c` flags applied (so any push-side hooks observing `GIT_AUTHOR_*` see the agent identity; the signing flags are harmless for push but kept for symmetry with commit/tag spawns).

Deferred to a follow-up (leave a `TODO` referencing this section):
- Walk `@{u}..HEAD` (or `merge-base(<default>, HEAD)..HEAD` when no upstream) and check every commit, not just HEAD.
- Optional pubkey-binding check: parse out the SSHSIG, decode the embedded ssh-ed25519 pubkey, confirm it matches our `pubkey_hex`. (v0 only checks "is there a signature header"; this v1 check would refuse commits signed by a *different* agent.)
- Multiple refspecs in one push.
- Force-pushes / detached HEAD.

### 8. `keys.ts` — what we actually have to write

Three small functions, no crypto beyond `@noble/ed25519`:

```ts
// 32-byte hex → 32-byte hex
export function privHexToPubHex(hexPriv: string): string

// Format the agent's git identity. `full` is the standard "Name <email>"
// rendering; `<child>` and `<pubkey_hex>` below are placeholders for the
// actual values, not literal angle brackets.
export function authorString(child: number, pubkeyHex: string): {
  name: string;     // agent-<child>
  email: string;    // <pubkey_hex>@agents.sphinx.chat
  full: string;     // agent-<child> <<pubkey_hex>@agents.sphinx.chat>
}

// Build an OpenSSH unencrypted private key file (the only non-trivial bit)
export function buildOpenSSHPrivateKey(hexPriv: string, comment: string): string
```

`buildOpenSSHPrivateKey` produces the standard OpenSSH key format. Layout (all lengths big-endian uint32, all `string` fields are length-prefixed byte arrays):

```
"openssh-key-v1\0"                    (15 bytes magic, NUL-terminated)
string  ciphername                    "none"
string  kdfname                       "none"
string  kdfoptions                    ""    (empty string)
uint32  num_keys                      1
string  public_key_blob               (see below)
string  encrypted_section             (see below; not actually encrypted, cipher=none)

public_key_blob:
  string  "ssh-ed25519"
  string  pubkey_32                   (raw 32-byte pubkey)

encrypted_section (must be 8-byte aligned via PKCS#7-style 1,2,3,...,n padding):
  uint32  checkint1                   (random)
  uint32  checkint2                   (== checkint1)
  string  "ssh-ed25519"
  string  pubkey_32
  string  privkey_64                  (raw_priv_32 || pubkey_32 — OpenSSH's "expanded" ed25519 secret)
  string  comment                     ("agent-<child>")
  bytes   padding                     (1,2,3,... up to 8-byte alignment)
```

PEM-armor:

```
-----BEGIN OPENSSH PRIVATE KEY-----
<base64 of the whole blob, wrapped at 70 chars>
-----END OPENSSH PRIVATE KEY-----
```

Acceptance test: pipe the output through `ssh-keygen -y -f <file>` and confirm the public key it prints corresponds to `privHexToPubHex(privkey)`.

### 9. Verification (out of scope here)

Verification is performed by *third parties* against the registry that maps `(user, child) → pubkey`:

- A reviewer reads a commit's author email, extracts the pubkey hex, looks up `(user, child)` in Hive, confirms the pubkey matches, and runs the SSH signature check.
- This CLI does **not** ship a `verify` subcommand in v0; `git verify-commit` (with an `allowed_signers` file built from Hive's registry) and `ssh-keygen -Y verify` are sufficient. Construction of the `allowed_signers` file is the verifier's responsibility, not this CLI's; see the "Allowed-signers helper" item under Open Questions for a possible future subcommand.
- A future `sphinx-git verify <commit-ish>` could be added cheaply if useful for tests.

### 10. Error UX

- All errors print `sphinx-git: <message>` to stderr and exit non-zero.
- Missing config / privkey is the most common case — surface it on the first signing-required invocation with an actionable hint that names the exact command to run.
- Never silently fall back to unsigned commits.
- `new-agent` validates inputs strictly (privkey must be 64-char hex → 32 bytes; `child` must be a non-negative integer parseable by `Number.isSafeInteger`) and prints a clear error pointing at the offending arg.

## Implementation Steps

### Step 1 — package wiring
1. Add `@noble/ed25519@2.3.0` and `@noble/hashes@1.7.1` to `mcp/package.json` dependencies.
2. Add `"sphinx-git": "tsx src/sphinx-git/bin.ts"` script and the `bin` entry as in §3.
3. `yarn install` to verify resolution.

### Step 2 — config + keys
1. Create `config.ts` with `CONFIG_DIR`, `CONFIG_FILE`, `KEY_FILE`, `readConfig()`, `writeConfig(child, hexPriv)`, `writeSigningKey(hexPriv, child)`. Ensure `0700`/`0600` perms; `mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 })`. `writeSigningKey` calls `keys.buildOpenSSHPrivateKey(hexPriv, "agent-<child>")` to produce the PEM body, then writes it. `readConfig()` self-heals: if `git.json` exists but `signing_key` is missing, it regenerates the latter before returning.
2. Create `keys.ts` with `privHexToPubHex`, `authorString`, `buildOpenSSHPrivateKey(hexPriv, comment)`.
3. Verify `buildOpenSSHPrivateKey` against `ssh-keygen -y -f` (manual smoke step — see Step 8).

### Step 3 — git spawn helper
1. `git.ts` with `spawnGit(args: string[], extraEnv?: Record<string,string>): Promise<number>` using `stdio: 'inherit'` and explicit exit-code forwarding (also propagate signals).

### Step 4 — sphinx subcommands
1. `commands/new_agent.ts`: validate args, `writeConfig`, `writeSigningKey`, print summary.
2. `commands/whoami.ts`: read config, derive pubkey, print `child` / pubkey hex / author string.

### Step 5 — signed git subcommands
1. `commands/commit.ts`: build identity env + signing `-c` flags, `spawnGit(["commit", ...args])`.
2. `commands/tag.ts`: detect annotated/signed (argv inspection); if annotated, signing path; otherwise plain passthrough.
3. `commands/push.ts`: implement HEAD-only signature check via `git cat-file commit HEAD` (v0); on success, spawn `git push` with identity env + signing `-c` flags as in §6. Leave a TODO for the `@{u}..HEAD` rev-list path described in §7.

### Step 6 — bin + dispatch
1. `bin.ts` shebang `#!/usr/bin/env node` + import `cli`.
2. `cli.ts` routing as in §5. Sphinx-specific subs use `commander`; signed subs and passthrough do not.

### Step 7 — README
1. `mcp/src/sphinx-git/README.md` covering install, `new-agent <hex> <child>`, `whoami`, signed `commit`/`tag`/`push`, how to verify externally with `ssh-keygen -Y verify`, and the Hive integration boundary (Hive supplies `(privkey, child)`; this CLI stores and uses them).

### Step 8 — manual smoke test
1. `yarn build`.
2. `node build/sphinx-git/bin.js new-agent <random 64-hex> 7` → check `~/.config/sphinx/{git.json,signing_key}` exist with mode 0600.
3. `ssh-keygen -y -f ~/.config/sphinx/signing_key` → derive pubkey; confirm matches `node build/sphinx-git/bin.js whoami`.
4. In a scratch repo: `node build/sphinx-git/bin.js commit -m "test"` → confirm:
   - `git log --format='%an <%ae>' -1` shows `agent-7 <<pubkey_hex>@agents.sphinx.chat>` where the outer `< >` come from the literal characters in the format string and `<pubkey_hex>` is the placeholder for the actual hex. (Concretely: `agent-7 <a3f1...e2@agents.sphinx.chat>`.)
   - `git log --show-signature -1` shows a valid SSH ed25519 signature.
5. With an `allowed_signers` file containing the agent pubkey: `git verify-commit HEAD` succeeds.
6. `node build/sphinx-git/bin.js tag -a v1 -m "release"` produces an SSH-signed annotated tag.
7. `node build/sphinx-git/bin.js push` on an unsigned HEAD refuses with a clear error; on a signed HEAD succeeds.

## Open Questions / Future Work

- **Hive integration**: how Hive delivers `(hex_privkey, child)` to the sandbox is out of scope. Likely env vars or a one-shot bootstrap call; the CLI just consumes them via `new-agent`.
- **Key rotation cadence**: once per sandbox? per task? per commit? The CLI supports any cadence (each `new-agent` is a clean rotation); policy lives elsewhere.
- **Sidecar daemon**: moving privkey custody out of the agent process to a unix-socket signer is the obvious next hardening step but not part of this PR.
- **Allowed-signers helper**: a future `sphinx-git allowed-signers` subcommand could emit a properly-formatted `allowed_signers` file for the current agent, useful for CI verifying agent-authored commits.
- **Full rev-list push check**: replace v0's HEAD-only check with `@{u}..HEAD` walk plus merge-base fallback for new branches.
- **GitHub "Verified" badge**: requires the agent's ssh-ed25519 pubkey to be uploaded as a signing key to whichever GitHub account will be the push target. Document but don't automate.
- **Budget layer**: separate module, separate proxy, will reuse the same `(child, pubkey)` identity primitive.

## Acceptance Criteria

- `sphinx-git new-agent <hex> <child>` writes both `~/.config/sphinx/git.json` (mode 0600) and `~/.config/sphinx/signing_key` (mode 0600) with the directory at mode 0700.
- `sphinx-git whoami` prints `agent-<child>` and pubkey hex; exits non-zero with no config.
- `sphinx-git commit -m "..."` produces a commit whose:
  - Author and committer name is `agent-<child>`.
  - Author and committer email is `<pubkey_hex>@agents.sphinx.chat`.
  - `git log --show-signature` shows a valid SSH ed25519 signature using our pubkey.
- `sphinx-git tag -a v1 -m "..."` produces an SSH-signed annotated tag; `sphinx-git tag v1` produces an unsigned lightweight tag (passthrough).
- `sphinx-git push` refuses when `HEAD` lacks an SSH signature header (v0 scope per §7); succeeds otherwise. The full multi-commit walk is deferred.
- All other invocations (`status`, `log`, `diff`, `checkout`, `rebase -i`, etc.) behave identically to `git`, including interactive editors and TTY pagers, and propagate the real git exit code.
- No file is written outside `~/.config/sphinx/`.
- No network calls beyond what the underlying `git` binary makes.
