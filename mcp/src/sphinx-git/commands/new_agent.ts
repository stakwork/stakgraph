import { writeConfig, writeSigningKey, CONFIG_FILE, KEY_FILE } from "../config.js";
import { privHexToPubHex, authorString } from "../keys.js";

/**
 * Validate the args, write both `git.json` and `signing_key`, then print a
 * human-readable summary of the new identity so the operator can eyeball-
 * compare against Hive's record.
 */
export function runNewAgent(args: string[]): void {
  if (args.length !== 2) {
    fail(
      "usage: sphinx-git new-agent <hex_privkey> <child>\n" +
        "  hex_privkey: 64 hex chars (32-byte ed25519 private key)\n" +
        "  child:       non-negative integer assigned by Hive",
    );
  }

  const [hexPriv, childRaw] = args;

  if (!/^[0-9a-fA-F]{64}$/.test(hexPriv)) {
    fail("hex_privkey must be exactly 64 hexadecimal characters");
  }

  const child = Number(childRaw);
  if (!Number.isSafeInteger(child) || child < 0) {
    fail(`child must be a non-negative integer (got ${JSON.stringify(childRaw)})`);
  }

  let pubHex: string;
  try {
    pubHex = privHexToPubHex(hexPriv);
  } catch (e) {
    fail(`failed to derive pubkey: ${(e as Error).message}`);
    return;
  }

  writeConfig(child, hexPriv.toLowerCase());
  writeSigningKey(hexPriv.toLowerCase(), child);

  const author = authorString(child, pubHex);

  process.stdout.write(
    `agent configured\n` +
      `  child:   ${child}\n` +
      `  pubkey:  ${pubHex}\n` +
      `  author:  ${author.full}\n` +
      `  config:  ${CONFIG_FILE}\n` +
      `  key:     ${KEY_FILE}\n`,
  );
}

function fail(msg: string): never {
  process.stderr.write(`sphinx-git: ${msg}\n`);
  process.exit(1);
}
