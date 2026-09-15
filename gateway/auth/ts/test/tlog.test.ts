/**
 * Transparency-log fixture parity (TS side).
 *
 * The Go gateway is the producer here: `go test ./internal/tlog
 * -update` writes `gateway/auth/fixtures/tlog-*.json` from the Go
 * implementation, which is itself pinned to the RFC 6962 / RFC 9162
 * reference vectors. This test asserts that every value in those
 * files reproduces byte-for-byte from this package.
 */

import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  bytesToHex,
  ecdsaPublicKey,
  emptyRoot,
  hashLeaf,
  hexToBytes,
  jcs,
  leafHash,
  rootFromLeafHashes,
  rootFromLeaves,
  signSth,
  sthSigningBytes,
  utf8Bytes,
  verifyConsistency,
  verifyInclusion,
  verifySth,
} from "../src/index.js";
import type { SignedTreeHead, TlogLeaf } from "../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(HERE, "..", "..", "fixtures");

interface InclusionVector {
  leaf_index: number;
  tree_size: number;
  proof: string[];
}
interface ConsistencyVector {
  old_size: number;
  new_size: number;
  proof: string[];
}
interface VectorFixture {
  description: string;
  inputs: { leaves_hex: string[] };
  expected: {
    leaf_hashes: string[];
    roots: string[];
    inclusion_proofs: InclusionVector[];
    consistency_proofs: ConsistencyVector[];
  };
}
interface LeavesFixture {
  description: string;
  inputs: { log_priv_hex: string; signed_at: string; leaves: TlogLeaf[] };
  expected: {
    log_pubkey: string;
    leaf_canonical_json: string[];
    leaf_hashes: string[];
    roots: string[];
    inclusion_proofs: InclusionVector[];
    consistency_proofs: ConsistencyVector[];
    sths: Array<{ sth: SignedTreeHead; signing_input: string; signing_bytes_hex: string }>;
  };
}

function load<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURES_DIR, name), "utf8")) as T;
}

const UTF8 = new TextDecoder();

function checkProofs(
  leafHashes: string[],
  roots: string[],
  inclusion: InclusionVector[],
  consistency: ConsistencyVector[],
): void {
  assert.ok(inclusion.length > 0 && consistency.length > 0);
  for (const iv of inclusion) {
    const ok = verifyInclusion({
      leafHash: leafHashes[iv.leaf_index]!,
      leafIndex: iv.leaf_index,
      treeSize: iv.tree_size,
      proof: iv.proof,
      root: roots[iv.tree_size]!,
    });
    assert.ok(ok, `inclusion (${iv.leaf_index}, ${iv.tree_size}) must verify`);
    if (iv.tree_size > 1) {
      const other = (iv.leaf_index + 1) % iv.tree_size;
      assert.ok(
        !verifyInclusion({
          leafHash: leafHashes[other]!,
          leafIndex: iv.leaf_index,
          treeSize: iv.tree_size,
          proof: iv.proof,
          root: roots[iv.tree_size]!,
        }),
        `inclusion (${iv.leaf_index}, ${iv.tree_size}) must reject another leaf`,
      );
      assert.ok(
        !verifyInclusion({
          leafHash: leafHashes[iv.leaf_index]!,
          leafIndex: iv.leaf_index,
          treeSize: iv.tree_size,
          proof: iv.proof.slice(0, -1),
          root: roots[iv.tree_size]!,
        }),
        `inclusion (${iv.leaf_index}, ${iv.tree_size}) must reject a truncated proof`,
      );
    }
  }
  for (const cv of consistency) {
    const ok = verifyConsistency({
      oldSize: cv.old_size,
      newSize: cv.new_size,
      proof: cv.proof,
      oldRoot: roots[cv.old_size]!,
      newRoot: roots[cv.new_size]!,
    });
    assert.ok(ok, `consistency (${cv.old_size}, ${cv.new_size}) must verify`);
    if (cv.old_size > 0 && cv.old_size < cv.new_size) {
      assert.ok(
        !verifyConsistency({
          oldSize: cv.old_size,
          newSize: cv.new_size,
          proof: cv.proof,
          oldRoot: roots[cv.old_size - 1]!,
          newRoot: roots[cv.new_size]!,
        }),
        `consistency (${cv.old_size}, ${cv.new_size}) must reject a wrong old root`,
      );
    }
  }
}

// ─── tlog-00: raw-byte reference vectors ────────────────────────────

const vectors = load<VectorFixture>("tlog-00-rfc9162-vectors.json");

test("tlog-00: leaf hashes and roots at every size", () => {
  const hashes = vectors.inputs.leaves_hex.map((h) => leafHash(hexToBytes(h)));
  assert.deepEqual(hashes.map(bytesToHex), vectors.expected.leaf_hashes);
  assert.equal(bytesToHex(emptyRoot()), vectors.expected.roots[0]);
  for (let n = 0; n <= hashes.length; n++) {
    assert.equal(bytesToHex(rootFromLeafHashes(hashes.slice(0, n))), vectors.expected.roots[n], `root ${n}`);
  }
});

test("tlog-00: inclusion and consistency proofs verify", () => {
  checkProofs(
    vectors.expected.leaf_hashes,
    vectors.expected.roots,
    vectors.expected.inclusion_proofs,
    vectors.expected.consistency_proofs,
  );
});

// ─── tlog-01: JSON leaves through the real log ───────────────────────

const fx = load<LeavesFixture>("tlog-01-leaves.json");

test("tlog-01: leaves canonicalize and hash byte-for-byte", () => {
  assert.equal(fx.inputs.leaves.length, 8);
  fx.inputs.leaves.forEach((leaf, i) => {
    const canonical = fx.expected.leaf_canonical_json[i]!;
    assert.equal(jcs(leaf), canonical, `leaf ${i} canonical JSON`);
    assert.equal(bytesToHex(hashLeaf(leaf)), fx.expected.leaf_hashes[i], `leaf ${i} hash from object`);
    assert.equal(bytesToHex(hashLeaf(canonical)), fx.expected.leaf_hashes[i], `leaf ${i} hash from string`);
    assert.equal(bytesToHex(leafHash(utf8Bytes(canonical))), fx.expected.leaf_hashes[i]);
  });
});

test("tlog-01: roots recompute from leaf objects and from canonical strings", () => {
  for (let n = 0; n <= fx.inputs.leaves.length; n++) {
    assert.equal(bytesToHex(rootFromLeaves(fx.inputs.leaves.slice(0, n))), fx.expected.roots[n], `root ${n} (objects)`);
    assert.equal(
      bytesToHex(rootFromLeaves(fx.expected.leaf_canonical_json.slice(0, n))),
      fx.expected.roots[n],
      `root ${n} (strings)`,
    );
  }
});

test("tlog-01: inclusion and consistency proofs verify", () => {
  checkProofs(
    fx.expected.leaf_hashes,
    fx.expected.roots,
    fx.expected.inclusion_proofs,
    fx.expected.consistency_proofs,
  );
});

test("tlog-01: signed tree heads — signing input, deterministic signature, verification", () => {
  const priv = hexToBytes(fx.inputs.log_priv_hex);
  assert.equal(bytesToHex(ecdsaPublicKey(priv)), fx.expected.log_pubkey);
  assert.equal(fx.expected.sths.length, 2);
  assert.equal(fx.expected.sths[0]!.sth.tree_size, 0);
  assert.equal(fx.expected.sths[0]!.sth.root_hash, bytesToHex(emptyRoot()));
  assert.equal(fx.expected.sths[1]!.sth.tree_size, fx.inputs.leaves.length);
  assert.equal(fx.expected.sths[1]!.sth.root_hash, fx.expected.roots[fx.inputs.leaves.length]);

  for (const { sth, signing_input, signing_bytes_hex } of fx.expected.sths) {
    const input = sthSigningBytes(sth);
    assert.equal(UTF8.decode(input), signing_input);
    assert.equal(bytesToHex(input), signing_bytes_hex);
    assert.equal(sth.signed_at, fx.inputs.signed_at);

    // RFC 6979 + low-s: re-signing in TS reproduces the Go bytes.
    const { sig: _drop, ...unsigned } = sth;
    assert.equal(signSth(unsigned, priv).sig, sth.sig);

    assert.ok(verifySth(sth, fx.expected.log_pubkey), `sth at size ${sth.tree_size} must verify`);
    assert.ok(!verifySth({ ...sth, tree_size: sth.tree_size + 1 }, fx.expected.log_pubkey));
    assert.ok(!verifySth({ ...sth, root_hash: fx.expected.roots[1]! }, fx.expected.log_pubkey));
    assert.ok(!verifySth({ ...sth, signed_at: "2030-01-01T00:00:00.000Z" }, fx.expected.log_pubkey));
    assert.ok(!verifySth(sth, bytesToHex(ecdsaPublicKey(hexToBytes("11".repeat(32))))));
    assert.ok(!verifySth({ ...sth, sig: "zz" }, fx.expected.log_pubkey));
    // Extra keys on the object do not change the signing input.
    assert.ok(verifySth({ ...sth, extra: 1 } as SignedTreeHead, fx.expected.log_pubkey));
  }
});
