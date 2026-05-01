import { readConfig } from "../config.js";
import { privHexToPubHex, authorString } from "../keys.js";
import { spawnGit } from "../git.js";
import { identityEnvFor, signingConfigFlags } from "./commit.js";

/**
 * `git tag` is split: lightweight tags carry no metadata or signature, so we
 * just pass them through. Anything that creates an annotated or signed tag
 * gets the agent identity + ssh-signing config.
 */
const ANNOTATED_FLAGS = new Set([
  "-a", "--annotate",
  "-s", "--sign",
  "-u", "--local-user",
  "-m", "--message",
  "-F", "--file",
]);

function isAnnotated(args: string[]): boolean {
  for (const a of args) {
    if (ANNOTATED_FLAGS.has(a)) return true;
    // long-form with value: --message=foo, --local-user=foo, --file=foo
    if (
      a.startsWith("--annotate=") ||
      a.startsWith("--sign") /* --sign or --sign=... */ ||
      a.startsWith("--local-user=") ||
      a.startsWith("--message=") ||
      a.startsWith("--file=")
    ) {
      return true;
    }
  }
  return false;
}

export async function runTag(args: string[]): Promise<void> {
  if (!isAnnotated(args)) {
    // lightweight tag (or `git tag` with no args, which just lists) — pass through
    await spawnGit(["tag", ...args]);
    return;
  }

  const cfg = readConfig();
  if (!cfg) {
    process.stderr.write(
      "sphinx-git: no agent configured; run `sphinx-git new-agent <hex_privkey> <child>`\n",
    );
    process.exit(1);
  }

  const pubHex = privHexToPubHex(cfg.privkey);
  const author = authorString(cfg.child, pubHex);

  const identityEnv = identityEnvFor(author.name, author.email);
  const signingFlags = signingConfigFlags();

  await spawnGit([...signingFlags, "tag", ...args], identityEnv);
}
