/**
 * Regenerate `gateway/auth/fixtures/*.json` from deterministic seed
 * keys. Run with:
 *
 *     npm run regenerate-fixtures
 *
 * The fixtures are the cross-language byte-equivalence contract.
 * Every implementation (Go, TS, polyglot) reproduces every
 * `expected.*` value byte-for-byte; if the implementations agree,
 * the fixtures pass. If a spec change shifts the bytes, regenerate
 * and commit the new fixtures alongside the spec change.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  attenuate,
  bytesToBase64url,
  bytesToHex,
  computeAttenuationHmac,
  ecdsaPublicKey,
  ed25519PublicKey,
  hexToBytes,
  invocationSigBytes,
  jcs,
  signingBytes,
  signInvocation,
  signUserAuthorizationMultisig,
  signUserAuthorizationSingle,
  utf8Bytes,
} from "../src/index.js";
import type {
  Attenuation,
  AttenuationCaveats,
  Budget,
  Invocation,
  InvocationUnsigned,
  Macaroon,
  UserAuthorization,
  UserAuthorizationUnsigned,
} from "../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(HERE, "..", "..", "fixtures");

// ─── deterministic seed keys ──────────────────────────────────────────
//
// These are FIXED, well-known values. They MUST be reproducible: anyone
// running this script must get the same fixtures. Do not use them for
// anything real — they are published in this repo.

const ORG_PRIV_HEX =
  "0000000000000000000000000000000000000000000000000000000000000001";
const ORG_2_PRIV_HEX =
  "0000000000000000000000000000000000000000000000000000000000000002";
const ORG_3_PRIV_HEX =
  "0000000000000000000000000000000000000000000000000000000000000003";

const USER_PRIV_HEX =
  "1111111111111111111111111111111111111111111111111111111111111111";

// ─── derived pubkeys ──────────────────────────────────────────────────

const orgPriv = hexToBytes(ORG_PRIV_HEX);
const org2Priv = hexToBytes(ORG_2_PRIV_HEX);
const org3Priv = hexToBytes(ORG_3_PRIV_HEX);
const userPriv = hexToBytes(USER_PRIV_HEX);

const orgPubHex = bytesToHex(ecdsaPublicKey(orgPriv));
const org2PubHex = bytesToHex(ecdsaPublicKey(org2Priv));
const org3PubHex = bytesToHex(ecdsaPublicKey(org3Priv));
const userPubHex = bytesToHex(ed25519PublicKey(userPriv));

// ─── shared timestamps (fixed → reproducible) ─────────────────────────

const UA_IAT = "2026-05-14T09:00:00Z";
const UA_EXP = "2026-06-14T09:00:00Z";
const INV_IAT = "2026-05-14T10:00:00Z";
const INV_EXP = "2026-05-14T10:10:00Z";
const ATT1_EXP = "2026-05-14T10:02:00Z";
const ATT2_EXP = "2026-05-14T10:01:30Z";

// ─── shared base objects (used by multiple fixtures) ──────────────────

function baseUserAuthorization(): UserAuthorizationUnsigned {
  return {
    user_id: "u_alice",
    user_pubkey: { alg: "ed25519", key: userPubHex },
    // Phase 11: agents lifted out of the (deleted) `permissions`
    // wrapper. No singular `realm` grant — multi-swarm scoping is
    // encoded in `budget.realm_budgets` when needed.
    agents: ["coder", "browser", "web-search", "repair-agent"],
    iat: UA_IAT,
    exp: UA_EXP,
    nonce: "9f4e1c8b2a3d4e5f6a7b8c9d0e1f2a3b",
  };
}

function baseInvocation(): InvocationUnsigned {
  return {
    // Phase 11: no `realm` field. Single-swarm deployments need no
    // realm scoping; multi-swarm scoping rides on `budget.realm_budgets`.
    agents: ["coder"],
    run_id: "r_01h8alpharootinvocation00",
    max_cost_usd: 5.0,
    max_steps: 100,
    iat: INV_IAT,
    exp: INV_EXP,
    nonce: "7c2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c",
  };
}

// ─── fixture writers ──────────────────────────────────────────────────

interface FixtureExpected {
  ua_signing_bytes_hex: string;
  inv_signing_bytes_hex: string;
  ua_canonical_json: string;
  inv_canonical_json: string;
  macaroon_canonical_json: string;
  macaroon_b64url: string;
  attenuation_hmac_inputs: Array<{
    prev_sig_hex: string;
    caveats_canonical_json: string;
    hmac_input_hex: string;
    hmac_output_hex: string;
  }>;
  claims: {
    org_id: string;
    user_id: string;
    agent_name: string;
    run_id: string;
    effective_caveats: {
      agents: string[];
      max_cost_usd: number;
      max_steps: number;
      budget: Budget | null;
      exp: string;
    };
    ua_nonce: string;
    ua_budget: Budget | null;
    permitted_realms: string[] | null;
    nonces: string[];
    iat: string;
  };
  verify_result: "ok";
}

function buildMacaroon(
  orgId: string,
  ua: UserAuthorization,
  inv: Invocation,
  atts: Attenuation[],
): Macaroon {
  return {
    v: 1,
    org_id: orgId,
    user_authorization: ua,
    invocation: inv,
    attenuations: atts,
  };
}

function computeExpected(
  unsignedUa: UserAuthorizationUnsigned,
  signedUa: UserAuthorization,
  unsignedInv: InvocationUnsigned,
  signedInv: Invocation,
  atts: Attenuation[],
  macaroon: Macaroon,
): FixtureExpected {
  const uaMsg = signingBytes(unsignedUa as Record<string, unknown>, "org_sig");
  const invMsg = signingBytes(unsignedInv as Record<string, unknown>, "user_sig");

  const attHmacInputs: FixtureExpected["attenuation_hmac_inputs"] = [];
  let prevSig = invocationSigBytes(signedInv);
  for (const att of atts) {
    const caveatsCanonical = jcs(att.caveats as unknown as Record<string, unknown>);
    const hmacInput = utf8Bytes(caveatsCanonical);
    const hmacOutput = computeAttenuationHmac(prevSig, att.caveats);
    attHmacInputs.push({
      prev_sig_hex: bytesToHex(prevSig),
      caveats_canonical_json: caveatsCanonical,
      hmac_input_hex: bytesToHex(hmacInput),
      hmac_output_hex: bytesToHex(hmacOutput),
    });
    prevSig = hexToBytes(att.hmac);
  }

  // Build claims by replaying narrowing locally (mirror of the
  // verifier). Phase 11 effective_caveats carries the propagated
  // Budget block; permitted_realms is the sorted keys of the
  // effective budget's realm_budgets, or null. Inherit the UA's
  // budget when the invocation omits its own ("Mixed mode" rule).
  let effectiveBudget: Budget | null =
    signedInv.budget ?? signedUa.budget ?? null;
  let effective: FixtureExpected["claims"]["effective_caveats"] = {
    agents: [...signedInv.agents],
    max_cost_usd: signedInv.max_cost_usd,
    max_steps: signedInv.max_steps,
    budget: effectiveBudget,
    exp: signedInv.exp,
  };
  let runId = signedInv.run_id;
  const attNonces: string[] = [];
  for (const att of atts) {
    effectiveBudget = att.caveats.budget ?? effectiveBudget;
    effective = {
      agents: att.caveats.agents,
      max_cost_usd: att.caveats.max_cost_usd,
      max_steps: att.caveats.max_steps,
      budget: effectiveBudget,
      exp: att.caveats.exp,
    };
    runId = att.caveats.run_id;
    attNonces.push(att.caveats.nonce);
  }
  const agentName = effective.agents[effective.agents.length - 1] ?? "";
  const permittedRealms = effectiveBudget?.realm_budgets
    ? Object.keys(effectiveBudget.realm_budgets).sort()
    : null;

  const macaroonCanonical = jcs(macaroon as unknown as Record<string, unknown>);
  const macaroonB64 = bytesToBase64url(utf8Bytes(macaroonCanonical));

  return {
    ua_signing_bytes_hex: bytesToHex(uaMsg),
    inv_signing_bytes_hex: bytesToHex(invMsg),
    ua_canonical_json: jcs(stripField(signedUa, "org_sig")),
    inv_canonical_json: jcs(stripField(signedInv, "user_sig")),
    macaroon_canonical_json: macaroonCanonical,
    macaroon_b64url: macaroonB64,
    attenuation_hmac_inputs: attHmacInputs,
    claims: {
      org_id: macaroon.org_id,
      user_id: signedUa.user_id,
      agent_name: agentName,
      run_id: runId,
      effective_caveats: effective,
      ua_nonce: signedUa.nonce,
      ua_budget: signedUa.budget ?? null,
      permitted_realms: permittedRealms,
      nonces: [signedUa.nonce, signedInv.nonce, ...attNonces],
      iat: signedInv.iat,
    },
    verify_result: "ok",
  };
}

function stripField(
  obj: Record<string, unknown>,
  field: string,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(obj)) if (k !== field) out[k] = obj[k];
  return out;
}

// ─── fixtures ─────────────────────────────────────────────────────────

function makeFixture01Simple() {
  const ua = baseUserAuthorization();
  const inv = baseInvocation();
  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);
  const macaroon = buildMacaroon("org_acme", signedUa, signedInv, []);
  return {
    description:
      "single-key org, custodial phase 1: one user_authorization, one invocation, zero attenuations. Single-swarm shape — no realm fields anywhere.",
    inputs: {
      org_id: "org_acme",
      org_priv_hex: ORG_PRIV_HEX,
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "single" as const,
        key: { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [] as AttenuationCaveats[],
    },
    expected: computeExpected(ua, signedUa, inv, signedInv, [], macaroon),
  };
}

function makeFixture02OneAttenuation() {
  const ua = baseUserAuthorization();
  const inv = baseInvocation();
  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);

  const childCaveats: AttenuationCaveats = {
    agents: ["coder", "web-search"],
    max_cost_usd: 2.0,
    max_steps: 40,
    run_id: "r_01h8alphachildlevel1run00",
    exp: ATT1_EXP,
    nonce: "1e8d2c3b4a5f6e7d8c9b0a1f2e3d4c5b",
  };
  const att1 = attenuate(invocationSigBytes(signedInv), childCaveats);

  const macaroon = buildMacaroon("org_acme", signedUa, signedInv, [att1]);
  return {
    description:
      "single-key org with one sub-agent attenuation narrowing budget and adding a web-search agent (lineage extension)",
    inputs: {
      org_id: "org_acme",
      org_priv_hex: ORG_PRIV_HEX,
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "single" as const,
        key: { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [childCaveats],
    },
    expected: computeExpected(ua, signedUa, inv, signedInv, [att1], macaroon),
  };
}

function makeFixture03TwoAttenuations() {
  const ua = baseUserAuthorization();
  const inv = baseInvocation();
  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);

  const childCaveats: AttenuationCaveats = {
    agents: ["coder", "web-search"],
    max_cost_usd: 2.0,
    max_steps: 40,
    run_id: "r_01h8alphachildlevel1run00",
    exp: ATT1_EXP,
    nonce: "1e8d2c3b4a5f6e7d8c9b0a1f2e3d4c5b",
  };
  const att1 = attenuate(invocationSigBytes(signedInv), childCaveats);

  const grandchildCaveats: AttenuationCaveats = {
    agents: ["coder", "web-search", "browser"],
    max_cost_usd: 0.5,
    max_steps: 10,
    run_id: "r_01h8alphagrandchild2run00",
    exp: ATT2_EXP,
    nonce: "2f9e3d4c5b6a7f8e9d0c1b2a3f4e5d6c",
  };
  const att2 = attenuate(hexToBytes(att1.hmac), grandchildCaveats);

  const macaroon = buildMacaroon(
    "org_acme",
    signedUa,
    signedInv,
    [att1, att2],
  );
  return {
    description:
      "two-deep attenuation chain: invocation → child (coder+web-search) → grandchild (+browser, smaller budget)",
    inputs: {
      org_id: "org_acme",
      org_priv_hex: ORG_PRIV_HEX,
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "single" as const,
        key: { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [childCaveats, grandchildCaveats],
    },
    expected: computeExpected(
      ua,
      signedUa,
      inv,
      signedInv,
      [att1, att2],
      macaroon,
    ),
  };
}

function makeFixture05BudgetEnvelope() {
  // Cold-storage motivating flow: org leader signs a UA with a
  // weekly cumulative cap + a per-invocation cap, then the
  // employee's hot key signs invocations under it.
  //
  // - max_total_usd:          $1000 cumulative across all invocations
  //                           under this UA (enforced by the adapter
  //                           via Redis cost:ua:<nonce>; the pure
  //                           verifier surfaces it on Claims).
  // - max_per_invocation_usd: $25 per single invocation (enforced at
  //                           signature time; pure field comparison).
  //
  // The invocation asks for $5 < $25 → verifies. The cumulative cap
  // is not exercised by the verifier (no Redis here) but appears on
  // Claims so adapters can plumb it into the hot path.
  const ua: UserAuthorizationUnsigned = {
    ...baseUserAuthorization(),
    budget: {
      max_total_usd: 1000.0,
      max_per_invocation_usd: 25.0,
    },
    // Distinct nonce so cost:ua:<nonce> buckets don't collide with
    // other fixtures in any downstream integration tests that load
    // multiple fixtures into one Redis.
    nonce: "ab1234567890abcdef1234567890abcd",
  };
  const inv: InvocationUnsigned = {
    ...baseInvocation(),
    run_id: "r_01h8budgetenveloperun0000",
    max_cost_usd: 5.0, // within both per-invocation and total caps
    nonce: "cd1234567890abcdef1234567890abcd",
  };
  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);
  const macaroon = buildMacaroon("org_acme", signedUa, signedInv, []);
  return {
    description:
      "single-key org with a UA budget envelope (max_total_usd + max_per_invocation_usd); invocation within both caps verifies and Claims surfaces ua_nonce + ua_budget",
    inputs: {
      org_id: "org_acme",
      org_priv_hex: ORG_PRIV_HEX,
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "single" as const,
        key: { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [] as AttenuationCaveats[],
    },
    expected: computeExpected(ua, signedUa, inv, signedInv, [], macaroon),
  };
}

function makeFixture04Multisig() {
  const ua = baseUserAuthorization();
  const inv = baseInvocation();
  // 2-of-3 multisig: keys 0, 1, 2; signers 0 and 2 participate.
  const signedUa = signUserAuthorizationMultisig(ua, [
    { key_index: 0, privKey: orgPriv },
    { key_index: 2, privKey: org3Priv },
  ]);
  const signedInv = signInvocation(inv, userPriv);
  const macaroon = buildMacaroon("org_multisig", signedUa, signedInv, []);
  return {
    description:
      "2-of-3 multisig org policy, signers 0 and 2 participate, no attenuations",
    inputs: {
      org_id: "org_multisig",
      org_privs_hex: {
        "0": ORG_PRIV_HEX,
        "1": ORG_2_PRIV_HEX,
        "2": ORG_3_PRIV_HEX,
      },
      participating_signers: [0, 2],
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "multisig" as const,
        threshold: 2,
        keys: [
          { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
          { alg: "ecdsa-secp256k1-sha256" as const, key: org2PubHex },
          { alg: "ecdsa-secp256k1-sha256" as const, key: org3PubHex },
        ],
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [] as AttenuationCaveats[],
    },
    expected: computeExpected(ua, signedUa, inv, signedInv, [], macaroon),
  };
}

function makeFixture06MultiRealm() {
  // Phase 11 multi-swarm scenario: the org signs a UA with per-realm
  // caps for two swarms (w1, w2). The user's invocation narrows the
  // realm_budgets to a subset with smaller caps for this run.
  //
  // The verifier checks budget narrowing at the UA→invocation
  // boundary; the plugin (out of scope for the pure verifier) reads
  // claims.permitted_realms to assert this swarm's realm_id is in
  // the set, then enforces the per-realm cap against Redis.
  const ua: UserAuthorizationUnsigned = {
    ...baseUserAuthorization(),
    budget: {
      max_total_usd: 1000.0,
      max_per_invocation_usd: 25.0,
      realm_budgets: {
        w1: { max_total_usd: 500.0 },
        w2: { max_total_usd: 200.0 },
      },
    },
    nonce: "f06e1d2c3b4a5968778899aabbccddee",
  };
  const inv: InvocationUnsigned = {
    ...baseInvocation(),
    run_id: "r_01h8multirealmrun000000000",
    max_cost_usd: 5.0,
    nonce: "f16e1d2c3b4a5968778899aabbccddee",
    budget: {
      // Narrowing per axis: child caps ≤ parent caps; both child
      // realms exist in parent's set.
      realm_budgets: {
        w1: { max_total_usd: 5.0 },
        w2: { max_total_usd: 2.0 },
      },
    },
  };
  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);
  const macaroon = buildMacaroon("org_acme", signedUa, signedInv, []);
  return {
    description:
      "multi-realm UA + invocation: org grants per-realm caps {w1:$500, w2:$200}, invocation narrows to {w1:$5, w2:$2}; no attenuations",
    inputs: {
      org_id: "org_acme",
      org_priv_hex: ORG_PRIV_HEX,
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "single" as const,
        key: { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [] as AttenuationCaveats[],
    },
    expected: computeExpected(ua, signedUa, inv, signedInv, [], macaroon),
  };
}

function makeFixture07CrossRealmAttenuation() {
  // Cross-realm sub-agent spawn: the parent invocation authorizes
  // spend on w1+w2, then attenuates locally to delegate a
  // sub-agent that should only spend on w2 with a smaller cap. No
  // Hive round-trip on the spawn path; the verifier checks the
  // HMAC chain + symmetric budget narrowing.
  const ua: UserAuthorizationUnsigned = {
    ...baseUserAuthorization(),
    budget: {
      max_total_usd: 1000.0,
      max_per_invocation_usd: 25.0,
      realm_budgets: {
        w1: { max_total_usd: 500.0 },
        w2: { max_total_usd: 200.0 },
      },
    },
    nonce: "07e1d2c3b4a596778899aabbccddee01",
  };
  const inv: InvocationUnsigned = {
    ...baseInvocation(),
    run_id: "r_01h8crossrealmparent00000",
    max_cost_usd: 5.0,
    nonce: "07e1d2c3b4a596778899aabbccddee02",
    budget: {
      realm_budgets: {
        w1: { max_total_usd: 5.0 },
        w2: { max_total_usd: 4.0 },
      },
    },
  };
  const signedUa = signUserAuthorizationSingle(ua, orgPriv);
  const signedInv = signInvocation(inv, userPriv);

  const subAgentCaveats: AttenuationCaveats = {
    // Lineage extension: child agents ⊇ parent agents.
    agents: ["coder", "web-search"],
    max_cost_usd: 2.0,
    max_steps: 40,
    run_id: "r_01h8crossrealmsubchild00",
    exp: ATT1_EXP,
    nonce: "07e1d2c3b4a596778899aabbccddee03",
    budget: {
      // Narrow to a single realm with a smaller cap. Parent has w1+w2,
      // child drops w1 (allowed) and tightens w2's cap.
      realm_budgets: {
        w2: { max_total_usd: 1.0 },
      },
    },
  };
  const subAgent = attenuate(invocationSigBytes(signedInv), subAgentCaveats);

  const macaroon = buildMacaroon("org_acme", signedUa, signedInv, [subAgent]);
  return {
    description:
      "cross-realm sub-agent attenuation: parent invocation allows w1+w2, child attenuates to w2 only with a smaller cap (HMAC-chained, no issuer round-trip)",
    inputs: {
      org_id: "org_acme",
      org_priv_hex: ORG_PRIV_HEX,
      user_priv_hex: USER_PRIV_HEX,
      policy: {
        type: "single" as const,
        key: { alg: "ecdsa-secp256k1-sha256" as const, key: orgPubHex },
      },
      ua_unsigned: ua,
      inv_unsigned: inv,
      atts_unsigned: [subAgentCaveats],
    },
    expected: computeExpected(
      ua,
      signedUa,
      inv,
      signedInv,
      [subAgent],
      macaroon,
    ),
  };
}

// ─── keys.json (the deterministic seed set) ───────────────────────────

function makeKeysJson() {
  return {
    description:
      "Deterministic seed keys used by all fixtures. Reproducible: anyone running scripts/regenerate-fixtures.ts gets the same values. NOT secret — published in the repo.",
    org_keys: [
      {
        index: 0,
        alg: "ecdsa-secp256k1-sha256",
        priv_hex: ORG_PRIV_HEX,
        pub_hex: orgPubHex,
      },
      {
        index: 1,
        alg: "ecdsa-secp256k1-sha256",
        priv_hex: ORG_2_PRIV_HEX,
        pub_hex: org2PubHex,
      },
      {
        index: 2,
        alg: "ecdsa-secp256k1-sha256",
        priv_hex: ORG_3_PRIV_HEX,
        pub_hex: org3PubHex,
      },
    ],
    user_keys: [
      {
        index: 0,
        alg: "ed25519",
        priv_hex: USER_PRIV_HEX,
        pub_hex: userPubHex,
      },
    ],
  };
}

// ─── main ─────────────────────────────────────────────────────────────

function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n", "utf8");
  process.stdout.write(`wrote ${path}\n`);
}

function main() {
  writeJson(join(FIXTURES_DIR, "keys.json"), makeKeysJson());
  writeJson(join(FIXTURES_DIR, "01-simple.json"), makeFixture01Simple());
  writeJson(join(FIXTURES_DIR, "02-one-attenuation.json"), makeFixture02OneAttenuation());
  writeJson(join(FIXTURES_DIR, "03-two-attenuations.json"), makeFixture03TwoAttenuations());
  writeJson(join(FIXTURES_DIR, "04-multisig-2of3.json"), makeFixture04Multisig());
  writeJson(join(FIXTURES_DIR, "05-budget-envelope.json"), makeFixture05BudgetEnvelope());
  writeJson(join(FIXTURES_DIR, "06-multi-realm.json"), makeFixture06MultiRealm());
  writeJson(join(FIXTURES_DIR, "07-cross-realm-attenuation.json"), makeFixture07CrossRealmAttenuation());
}

main();
