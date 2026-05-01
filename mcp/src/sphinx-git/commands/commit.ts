import { readConfig, KEY_FILE } from "../config.js";
import { privHexToPubHex, authorString } from "../keys.js";
import { spawnGit } from "../git.js";

/**
 * Run `git commit` with the agent's identity injected via env and SSH-signing
 * config injected via per-process `-c` flags.
 */
export async function runCommit(args: string[]): Promise<void> {
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

  await spawnGit([...signingFlags, "commit", ...args], identityEnv);
}

export function identityEnvFor(name: string, email: string): Record<string, string> {
  return {
    GIT_AUTHOR_NAME: name,
    GIT_AUTHOR_EMAIL: email,
    GIT_COMMITTER_NAME: name,
    GIT_COMMITTER_EMAIL: email,
  };
}

export function signingConfigFlags(): string[] {
  return [
    "-c", "gpg.format=ssh",
    "-c", `user.signingkey=${KEY_FILE}`,
    "-c", "commit.gpgsign=true",
    "-c", "tag.gpgsign=true",
  ];
}
