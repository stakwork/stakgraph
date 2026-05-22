/**
 * mint-macaroon.ts — mint a fresh wire-format macaroon for smoke tests.
 *
 * Companion to gateway/scripts/smoke-test-enforcement.sh. Run via:
 *
 *   cd gateway/auth/ts && node --import tsx ../../scripts/mint-macaroon.ts \
 *     --org-id org_smoke \
 *     --user-id u_alice \
 *     --agent coder \
 *     --run-id r_smoke_<unix> \
 *     --max-cost-usd 1.00 \
 *     --max-steps 50 \
 *     --ttl-seconds 600 \
 *     [--realm-budgets '{"w1":{"max_total_usd":50}}']
 *
 * Prints the base64url-encoded macaroon on stdout (one line, no
 * trailing newline cruft). Errors go to stderr.
 *
 * Phase 11 dropped the singular `--realm` flag. Single-swarm
 * deployments mint without any realm scoping. Multi-swarm deployments
 * pass `--realm-budgets` as a JSON map of realm-id → per-realm cap
 * (`{ max_total_usd: <number> }`), which lands on both the UA's
 * Budget and the invocation's Budget so the verifier can run its
 * symmetric narrowing checks end-to-end.
 *
 * Why this lives in gateway/scripts/ instead of gateway/auth/ts/scripts/
 * --------------------------------------------------------------------
 * The mint helper is smoke-test support code, not a piece of the
 * cryptographic package. Keeping it next to the bash script that
 * invokes it (gateway/scripts/smoke-test-enforcement.sh) means the
 * pair moves together when either changes. The auth/ts/scripts/
 * directory is reserved for fixture regeneration, which is the only
 * other auth-package-internal tool.
 *
 * Why not bake the values inline
 * ------------------------------
 * The shell script needs control over org_id, user_id, run_id, and
 * caveat values per-run (run_id MUST be fresh so cost accumulators
 * don't collide with prior smoke runs; exp MUST be in the future).
 * Passing them as CLI flags keeps the bash side declarative.
 *
 * Keys are NOT a flag
 * -------------------
 * We always sign with the fixture seed keys (priv hex 0x…01 for org,
 * 0x…11…11 for user) loaded from gateway/auth/fixtures/keys.json.
 * Those values are deliberately published test material — see the
 * fixture file's `description` field. The smoke test registers the
 * matching org pubkey against `org_id` via /_plugin/trust before
 * minting, so the gateway's verifier accepts the signatures.
 *
 * The signing cadence and caveats here mirror what Hive's
 * /macaroons/issue endpoint will eventually do in production
 * (cryptographic-identity.md "Issuer endpoints"); this script is its
 * stunt-double for smoke tests.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  bytesToBase64url,
  hexToBytes,
  jcs,
  signInvocation,
  signUserAuthorizationSingle,
  utf8Bytes,
} from "../auth/ts/src/index.js";
import type {
  Budget,
  InvocationUnsigned,
  Macaroon,
  RealmBudget,
  UserAuthorizationUnsigned,
} from "../auth/ts/src/index.js";

// ─── arg parsing ─────────────────────────────────────────────────────

interface Args {
  orgId: string;
  userId: string;
  agent: string;
  runId: string;
  maxCostUsd: number;
  maxSteps: number;
  ttlSeconds: number;
  /** Optional phase-11 multi-swarm scoping. JSON map of realm-id → RealmBudget. */
  realmBudgets?: Record<string, RealmBudget>;
}

function parseArgs(argv: string[]): Args {
  const required = new Set([
    "--org-id",
    "--user-id",
    "--agent",
    "--run-id",
  ]);
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i];
    const val = argv[i + 1];
    if (!flag || !flag.startsWith("--") || val === undefined) {
      die(`bad flag/value pair near argv[${i}]: ${flag} ${val}`);
    }
    out[flag] = val;
  }
  for (const r of required) {
    if (!out[r]) die(`missing required flag ${r}`);
  }
  // realm-budgets is optional JSON. Parsed once up-front so a bad
  // argument fails fast (rather than crashing later inside the
  // signing path with a confusing stack trace).
  let realmBudgets: Record<string, RealmBudget> | undefined;
  if (out["--realm-budgets"]) {
    try {
      const parsed = JSON.parse(out["--realm-budgets"]);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        realmBudgets = parsed as Record<string, RealmBudget>;
      } else {
        die("--realm-budgets must be a JSON object");
      }
    } catch (e) {
      die(`--realm-budgets is not valid JSON: ${String(e)}`);
    }
  }
  return {
    orgId: out["--org-id"]!,
    userId: out["--user-id"]!,
    agent: out["--agent"]!,
    runId: out["--run-id"]!,
    // Numeric flags default to values that match a typical
    // smoke-test invocation: a sub-dollar cap that's plenty for the
    // 20-token calls the bash script makes, 50 steps (we make ≤5),
    // and 10 minutes of liveness (script finishes in seconds).
    maxCostUsd: out["--max-cost-usd"] ? Number(out["--max-cost-usd"]) : 1.0,
    maxSteps: out["--max-steps"] ? Number(out["--max-steps"]) : 50,
    ttlSeconds: out["--ttl-seconds"] ? Number(out["--ttl-seconds"]) : 600,
    realmBudgets,
  };
}

function die(msg: string): never {
  process.stderr.write(`mint-macaroon: ${msg}\n`);
  process.exit(2);
}

// ─── key material ────────────────────────────────────────────────────

interface KeysFile {
  org_keys: Array<{ index: number; priv_hex: string; pub_hex: string }>;
  user_keys: Array<{ index: number; priv_hex: string; pub_hex: string }>;
}

function loadFixtureKeys(): {
  orgPriv: Uint8Array;
  userPriv: Uint8Array;
  userPubHex: string;
} {
  // Resolve relative to *this script's* location so the file works
  // regardless of the caller's cwd. The script lives at
  // gateway/scripts/mint-macaroon.ts; fixtures at gateway/auth/fixtures/.
  const here = dirname(fileURLToPath(import.meta.url));
  const fixturesPath = join(here, "..", "auth", "fixtures", "keys.json");
  const raw = readFileSync(fixturesPath, "utf8");
  const keys = JSON.parse(raw) as KeysFile;
  const org0 = keys.org_keys[0];
  const user0 = keys.user_keys[0];
  if (!org0 || !user0) die(`fixtures/keys.json missing expected entries`);
  return {
    orgPriv: hexToBytes(org0.priv_hex),
    userPriv: hexToBytes(user0.priv_hex),
    userPubHex: user0.pub_hex,
  };
}

// ─── nonce ───────────────────────────────────────────────────────────

function randomNonceHex(): string {
  // 16 random bytes → 32 hex chars, per phase-4-macaroon-shape.md
  // "Nonces, run_ids, and other identifiers".
  const buf = new Uint8Array(16);
  // Node ≥20 has globalThis.crypto.
  globalThis.crypto.getRandomValues(buf);
  let s = "";
  for (const b of buf) s += b.toString(16).padStart(2, "0");
  return s;
}

// ─── timestamps ──────────────────────────────────────────────────────

/** RFC3339 / ISO 8601 UTC at second precision (matches fixtures). */
function rfc3339(d: Date): string {
  return d.toISOString().replace(/\.\d{3}Z$/, "Z");
}

// ─── main ────────────────────────────────────────────────────────────

function main() {
  const args = parseArgs(process.argv.slice(2));
  const { orgPriv, userPriv, userPubHex } = loadFixtureKeys();

  const now = new Date();
  const iat = rfc3339(now);
  // UA exp comfortably outlives the invocation. We don't bother
  // making UA exp configurable: the smoke script only needs one
  // UA-worth of life, and a 1h ceiling is plenty.
  const uaExp = rfc3339(new Date(now.getTime() + 60 * 60 * 1000));
  const invExp = rfc3339(new Date(now.getTime() + args.ttlSeconds * 1000));

  // Phase-11 budget block. When --realm-budgets is supplied we put it
  // on the UA (org-grant) AND on the invocation (this-run's scope)
  // so both layers carry the same set — the simplest shape that
  // exercises the symmetric narrowing path without bringing in
  // attenuation. A real Hive issuer might narrow on the invocation
  // side; that's a follow-up smoke test.
  let uaBudget: Budget | undefined;
  let invBudget: Budget | undefined;
  if (args.realmBudgets) {
    uaBudget = { realm_budgets: args.realmBudgets };
    invBudget = { realm_budgets: args.realmBudgets };
  }

  // user_authorization grants this user the agent we're about to
  // use. Permissions are deliberately broad enough that the smoke
  // script can vary the agent via flags without regenerating the UA
  // grant — the realistic Hive flow would issue one UA per user
  // with their full granted set, then narrow per invocation.
  const ua: UserAuthorizationUnsigned = {
    user_id: args.userId,
    user_pubkey: { alg: "ed25519", key: userPubHex },
    agents: [args.agent],
    budget: uaBudget,
    iat,
    exp: uaExp,
    nonce: randomNonceHex(),
  };

  const inv: InvocationUnsigned = {
    agents: [args.agent],
    run_id: args.runId,
    max_cost_usd: args.maxCostUsd,
    max_steps: args.maxSteps,
    budget: invBudget,
    iat,
    exp: invExp,
    nonce: randomNonceHex(),
  };

  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);

  const macaroon: Macaroon = {
    v: 1,
    org_id: args.orgId,
    user_authorization: signedUa,
    invocation: signedInv,
    attenuations: [],
  };

  const b64 = bytesToBase64url(
    utf8Bytes(jcs(macaroon as unknown as Record<string, unknown>)),
  );

  // Single line, no trailing newline — the bash side does
  //   MACAROON=$(node --import tsx mint-macaroon.ts ...)
  // and any extra whitespace would land in the header.
  process.stdout.write(b64);
}

main();
