import { readConfig } from "../config.js";
import { privHexToPubHex, authorString } from "../keys.js";

export function runWhoami(): void {
  const cfg = readConfig();
  if (!cfg) {
    process.stderr.write(
      "sphinx-git: no agent configured; run `sphinx-git new-agent <hex_privkey> <child>`\n",
    );
    process.exit(1);
  }

  const pubHex = privHexToPubHex(cfg.privkey);
  const author = authorString(cfg.child, pubHex);

  process.stdout.write(
    `child:   ${cfg.child}\n` +
      `pubkey:  ${pubHex}\n` +
      `author:  ${author.full}\n`,
  );
}
