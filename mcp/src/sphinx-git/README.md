# sphinx-git

Drop-in `git` replacement for Sphinx sandbox AI-agent servers. Every commit,
annotated tag, and push is cryptographically attributable to a specific agent
via a per-agent ed25519 keypair, signed using stock OpenSSH SSHSIG so it
verifies with standard tooling.

## Install

From the `mcp` workspace:

```sh
yarn install
yarn build
```

After `yarn build`, the CLI is at `build/sphinx-git/bin.js`. The `bin` entry
in `package.json` exposes it as `sphinx-git`. During development you can also
run it via `yarn sphinx-git ...`.

## Usage

### Configure an agent

```sh
sphinx-git new-agent <hex_privkey> <child>
```

- `hex_privkey`: 64-char hex (32-byte ed25519 private key)
- `child`: non-negative integer assigned by Hive — opaque to this CLI

This writes:

```
~/.config/sphinx/                      mode 0700
├── git.json                           mode 0600   (child + privkey)
└── signing_key                        mode 0600   (OpenSSH-format key file)
```

If `signing_key` is later deleted but `git.json` remains, the next
signing-required invocation regenerates it from `git.json`.

### Inspect the current identity

```sh
sphinx-git whoami
```

Prints `child`, pubkey hex, and the git author string
`agent-<child> <<pubkey_hex>@agents.sphinx.chat>`.

### Signed git operations

```sh
sphinx-git commit -m "..."        # SSH-signed, agent-named
sphinx-git tag -a v1 -m "..."     # annotated tag → signed
sphinx-git tag v1                 # lightweight tag → unsigned passthrough
sphinx-git push                   # refuses unsigned HEAD (v0)
```

### Everything else

Anything that isn't `new-agent`, `whoami`, `commit`, `tag`, or `push` is
passed straight through to `git` with the user's environment intact —
including interactive editors, pagers, and TTY handling.

```sh
sphinx-git status
sphinx-git log --oneline
sphinx-git rebase -i main
```

## Verification

This CLI does **not** ship a `verify` subcommand. Verification is performed
by third parties against the registry that maps `(user, child) → pubkey`
(Hive). Typical flow:

1. Read the commit's author email, extract the pubkey hex (everything before
   `@agents.sphinx.chat`).
2. Look up `(user, child)` in Hive and confirm the pubkey matches.
3. Build an `allowed_signers` file containing that pubkey and run

   ```sh
   git -c gpg.ssh.allowedSignersFile=<file> verify-commit <sha>
   ```

   or `ssh-keygen -Y verify` for direct SSHSIG verification.

## Hive integration boundary

Hive is the source of truth for `(user, child) → pubkey`. This CLI does not
talk to Hive directly; it just stores and uses what Hive (or any equivalent
upstream) supplies via `new-agent`.

How Hive delivers `(hex_privkey, child)` to the sandbox (env vars, one-shot
bootstrap call, etc.) is intentionally out of scope.

## Push behavior (v0)

`sphinx-git push` currently checks **only `HEAD`** for an SSH signature
header. The full multi-commit walk over `@{u}..HEAD` and the pubkey-binding
check (does the embedded SSHSIG pubkey match _our_ configured pubkey?) are
documented as follow-ups in `SPHINX_GIT_PLAN.md` §7.

## Files written

`sphinx-git` only ever writes to `~/.config/sphinx/`. It makes no network
calls beyond what the underlying `git` binary makes.

<!-- signing smoketest: this commit should be SSH-signed by agent-1 -->

