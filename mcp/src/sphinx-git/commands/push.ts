import { readConfig } from "../config.js";
import { privHexToPubHex, authorString } from "../keys.js";
import { spawnGit, gitCapture } from "../git.js";
import { identityEnvFor, signingConfigFlags } from "./commit.js";

/**
 * Pre-flight: refuse to push if HEAD lacks an SSH signature header.
 *
 * v0 only checks HEAD. The full @{u}..HEAD walk + pubkey-binding check is
 * documented as a follow-up in mcp/docs/plans/SPHINX_GIT_PLAN.md §7.
 *
 * TODO(sphinx-git): walk `@{u}..HEAD` (or `merge-base(<default>,HEAD)..HEAD`
 * when no upstream) and verify each commit, not just HEAD. Also verify the
 * embedded SSHSIG pubkey matches the agent's configured pubkey, not just
 * "some" signature.
 */
export async function runPush(args: string[]): Promise<void> {
  const cfg = readConfig();
  if (!cfg) {
    process.stderr.write(
      "sphinx-git: no agent configured; run `sphinx-git new-agent <hex_privkey> <child>`\n",
    );
    process.exit(1);
  }

  const head = gitCapture(["rev-parse", "HEAD"]);
  if (head === null) {
    // Empty repo, or not a git repo, or other rev-parse failure.
    // Let real git produce its native error.
    const pubHex = privHexToPubHex(cfg.privkey);
    const author = authorString(cfg.child, pubHex);
    await spawnGit(
      [...signingConfigFlags(), "push", ...args],
      identityEnvFor(author.name, author.email),
    );
    return;
  }
  const headSha = head.trim();

  const commitObj = gitCapture(["cat-file", "commit", headSha]);
  if (commitObj === null) {
    process.stderr.write(
      `sphinx-git: could not read commit object for HEAD (${headSha})\n`,
    );
    process.exit(1);
  }

  if (!hasSshSignature(commitObj)) {
    process.stderr.write(
      `sphinx-git: refusing to push: HEAD (${headSha}) is not SSH-signed.\n` +
        `  Re-create the commit with \`sphinx-git commit ...\` so it carries an agent signature.\n`,
    );
    process.exit(1);
  }

  const pubHex = privHexToPubHex(cfg.privkey);
  const author = authorString(cfg.child, pubHex);
  await spawnGit(
    [...signingConfigFlags(), "push", ...args],
    identityEnvFor(author.name, author.email),
  );
}

/**
 * Inspect a raw commit object. Returns true if the commit headers contain a
 * `gpgsig` line whose payload is an SSHSIG block. The headers end at the first
 * blank line; everything after that is the commit message and must be
 * ignored (the message body could contain a literal "BEGIN SSH SIGNATURE"
 * string and that should not count).
 */
export function hasSshSignature(commitObj: string): boolean {
  const blankIdx = commitObj.indexOf("\n\n");
  const headers = blankIdx >= 0 ? commitObj.slice(0, blankIdx) : commitObj;

  // Header continuation lines start with a single space. We're looking for a
  // header named `gpgsig` (or `gpgsig-sha256` for sha256 repos) whose value
  // contains the SSH signature armor.
  const lines = headers.split("\n");
  let inGpgSig = false;
  for (const line of lines) {
    if (line.startsWith("gpgsig ") || line.startsWith("gpgsig-sha256 ")) {
      inGpgSig = true;
      if (line.includes("-----BEGIN SSH SIGNATURE-----")) return true;
      continue;
    }
    if (inGpgSig) {
      if (line.startsWith(" ")) {
        if (line.includes("-----BEGIN SSH SIGNATURE-----")) return true;
        continue;
      }
      // a non-continuation header ended the gpgsig block
      inGpgSig = false;
    }
  }
  return false;
}
