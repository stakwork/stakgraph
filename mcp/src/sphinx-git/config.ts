import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { buildOpenSSHPrivateKey } from "./keys.js";

export const CONFIG_DIR = join(homedir(), ".config", "sphinx");
export const CONFIG_FILE = join(CONFIG_DIR, "git.json");
export const KEY_FILE = join(CONFIG_DIR, "signing_key");

export interface AgentConfig {
  child: number;
  privkey: string; // 64-char hex (32 bytes ed25519)
}

function ensureConfigDir(): void {
  mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
}

/**
 * Read the agent config. Self-heals: if `git.json` is present but
 * `signing_key` is missing, regenerates the latter from `privkey` before
 * returning. Returns null if there is no config at all.
 */
export function readConfig(): AgentConfig | null {
  if (!existsSync(CONFIG_FILE)) return null;

  const raw = readFileSync(CONFIG_FILE, "utf8");
  let parsed: any;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`failed to parse ${CONFIG_FILE}`);
  }
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    typeof parsed.privkey !== "string" ||
    typeof parsed.child !== "number"
  ) {
    throw new Error(`malformed ${CONFIG_FILE}: missing child or privkey`);
  }

  const cfg: AgentConfig = { child: parsed.child, privkey: parsed.privkey };

  // Self-heal: regenerate the OpenSSH key file if it's gone missing.
  if (!existsSync(KEY_FILE)) {
    writeSigningKey(cfg.privkey, cfg.child);
  }

  return cfg;
}

export function writeConfig(child: number, hexPriv: string): void {
  ensureConfigDir();
  const data: AgentConfig = { child, privkey: hexPriv };
  writeFileSync(CONFIG_FILE, JSON.stringify(data, null, 2) + "\n", {
    mode: 0o600,
  });
}

export function writeSigningKey(hexPriv: string, child: number): void {
  ensureConfigDir();
  const pem = buildOpenSSHPrivateKey(hexPriv, `agent-${child}`);
  writeFileSync(KEY_FILE, pem, { mode: 0o600 });
}
