/**
 * Cross-language byte-equivalence test (TS side).
 *
 * Loads every `gateway/auth/fixtures/*.json`, reproduces every
 * `expected.*` value from the `inputs.*` private keys + unsigned
 * objects, and fails if any byte differs.
 *
 * The Go side runs the same test against the same fixtures. If both
 * pass, the implementations agree.
 */

import { strict as assert } from "node:assert";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  attenuate,
  bytesToBase64url,
  bytesToHex,
  computeAttenuationHmac,
  hexToBytes,
  invocationSigBytes,
  jcs,
  signingBytes,
  signInvocation,
  signUserAuthorizationMultisig,
  signUserAuthorizationSingle,
  utf8Bytes,
  verify,
} from "../src/index.js";
import type {
  Attenuation,
  AttenuationCaveats,
  Budget,
  InvocationUnsigned,
  Macaroon,
  Policy,
  UserAuthorizationUnsigned,
} from "../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(HERE, "..", "..", "fixtures");

interface FixtureShape {
  description: string;
  inputs: {
    org_id: string;
    org_priv_hex?: string;                     // single-sig
    org_privs_hex?: Record<string, string>;    // multisig
    participating_signers?: number[];          // multisig
    user_priv_hex: string;
    policy: Policy;
    ua_unsigned: UserAuthorizationUnsigned;
    inv_unsigned: InvocationUnsigned;
    atts_unsigned: AttenuationCaveats[];
  };
  expected: {
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
  };
}

function fixtureFiles(): string[] {
  return readdirSync(FIXTURES_DIR)
    .filter((f) => /^\d{2}-.*\.json$/.test(f))
    .sort();
}

for (const file of fixtureFiles()) {
  test(`fixture ${file}`, () => {
    const fx: FixtureShape = JSON.parse(
      readFileSync(join(FIXTURES_DIR, file), "utf8"),
    );

    // ─── re-sign ua ──────────────────────────────────────────────
    let signedUa;
    if (fx.inputs.org_priv_hex) {
      signedUa = signUserAuthorizationSingle(
        fx.inputs.ua_unsigned,
        hexToBytes(fx.inputs.org_priv_hex),
      );
    } else if (fx.inputs.org_privs_hex && fx.inputs.participating_signers) {
      const signers = fx.inputs.participating_signers.map((idx) => ({
        key_index: idx,
        privKey: hexToBytes(fx.inputs.org_privs_hex![String(idx)]!),
      }));
      signedUa = signUserAuthorizationMultisig(fx.inputs.ua_unsigned, signers);
    } else {
      throw new Error(`fixture ${file}: no org signing inputs`);
    }

    // ─── re-sign invocation ──────────────────────────────────────
    const signedInv = signInvocation(
      fx.inputs.inv_unsigned,
      hexToBytes(fx.inputs.user_priv_hex),
    );

    // ─── signing bytes ───────────────────────────────────────────
    const uaMsg = signingBytes(
      fx.inputs.ua_unsigned as unknown as Record<string, unknown>,
      "org_sig",
    );
    const invMsg = signingBytes(
      fx.inputs.inv_unsigned as unknown as Record<string, unknown>,
      "user_sig",
    );
    assert.equal(bytesToHex(uaMsg), fx.expected.ua_signing_bytes_hex,
      "ua_signing_bytes_hex");
    assert.equal(bytesToHex(invMsg), fx.expected.inv_signing_bytes_hex,
      "inv_signing_bytes_hex");

    // ─── canonical JSON of signed layers ─────────────────────────
    assert.equal(
      jcs(stripField(signedUa, "org_sig")),
      fx.expected.ua_canonical_json,
      "ua_canonical_json",
    );
    assert.equal(
      jcs(stripField(signedInv, "user_sig")),
      fx.expected.inv_canonical_json,
      "inv_canonical_json",
    );

    // ─── attenuations ────────────────────────────────────────────
    const atts: Attenuation[] = [];
    let prevSig = invocationSigBytes(signedInv);
    fx.inputs.atts_unsigned.forEach((caveats, i) => {
      const exp = fx.expected.attenuation_hmac_inputs[i];
      assert.ok(exp, `expected attenuation_hmac_inputs[${i}] present`);

      assert.equal(bytesToHex(prevSig), exp.prev_sig_hex,
        `attenuation ${i} prev_sig_hex`);
      const caveatsCanonical = jcs(caveats as unknown as Record<string, unknown>);
      assert.equal(caveatsCanonical, exp.caveats_canonical_json,
        `attenuation ${i} caveats_canonical_json`);
      const hmacInput = utf8Bytes(caveatsCanonical);
      assert.equal(bytesToHex(hmacInput), exp.hmac_input_hex,
        `attenuation ${i} hmac_input_hex`);
      const hmacOutput = computeAttenuationHmac(prevSig, caveats);
      assert.equal(bytesToHex(hmacOutput), exp.hmac_output_hex,
        `attenuation ${i} hmac_output_hex`);

      atts.push(attenuate(prevSig, caveats));
      prevSig = hexToBytes(atts[i]!.hmac);
    });

    // ─── full macaroon ───────────────────────────────────────────
    const macaroon: Macaroon = {
      v: 1,
      org_id: fx.inputs.org_id,
      user_authorization: signedUa,
      invocation: signedInv,
      attenuations: atts,
    };
    const macaroonCanonical = jcs(macaroon as unknown as Record<string, unknown>);
    assert.equal(macaroonCanonical, fx.expected.macaroon_canonical_json,
      "macaroon_canonical_json");
    const macaroonB64 = bytesToBase64url(utf8Bytes(macaroonCanonical));
    assert.equal(macaroonB64, fx.expected.macaroon_b64url, "macaroon_b64url");

    // ─── round-trip verify ───────────────────────────────────────
    const claims = verify(
      macaroonB64,
      fx.inputs.policy,
      new Date(Date.parse(fx.inputs.inv_unsigned.iat) + 1000), // 1s after iat: in window
    );
    assert.equal(claims.org_id, fx.expected.claims.org_id);
    assert.equal(claims.user_id, fx.expected.claims.user_id);
    assert.equal(claims.agent_name, fx.expected.claims.agent_name);
    assert.equal(claims.run_id, fx.expected.claims.run_id);
    assert.deepEqual(claims.effective_caveats, fx.expected.claims.effective_caveats);
    assert.equal(claims.ua_nonce, fx.expected.claims.ua_nonce);
    assert.deepEqual(claims.ua_budget, fx.expected.claims.ua_budget);
    assert.deepEqual(claims.permitted_realms, fx.expected.claims.permitted_realms);
    assert.deepEqual(claims.nonces, fx.expected.claims.nonces);
    assert.equal(claims.iat, fx.expected.claims.iat);
  });
}

function stripField(obj: Record<string, unknown>, field: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(obj)) if (k !== field) out[k] = obj[k];
  return out;
}
