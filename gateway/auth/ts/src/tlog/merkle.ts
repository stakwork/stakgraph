/**
 * RFC 9162 §2.1 Merkle tree (Certificate Transparency v2, which
 * obsoletes RFC 6962 without changing the hashing):
 *
 *     leaf_hash(d)    = SHA256(0x00 || d)
 *     node_hash(l, r) = SHA256(0x01 || l || r)
 *     MTH({})         = SHA256("")
 *     MTH(D[n])       = node_hash(MTH(D[0:k]), MTH(D[k:n]))   k = largest power of two < n
 *
 * The two verifiers are transcriptions of §2.1.3.2 (inclusion) and
 * §2.1.4.2 (consistency). Pure functions, no I/O. Pinned against the
 * Go producer by `gateway/auth/fixtures/tlog-*.json`, which the Go
 * side in turn pins to the reference implementation's vectors.
 */

import { sha256 } from "@noble/hashes/sha2";

import { hexToBytes, utf8Bytes } from "../encoding.js";
import { jcs } from "../jcs.js";
import type { HashLike, TlogLeaf } from "./types.js";

const LEAF_PREFIX = new Uint8Array([0x00]);
const NODE_PREFIX = new Uint8Array([0x01]);

function concat(...parts: Uint8Array[]): Uint8Array {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** Bytes of a hash given as bytes or hex. */
export function hashBytes(h: HashLike): Uint8Array {
  return typeof h === "string" ? hexToBytes(h) : h;
}

export function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i]! ^ b[i]!;
  return diff === 0;
}

/** MTH({}): the root of the empty tree, SHA-256 of the empty string. */
export function emptyRoot(): Uint8Array {
  return sha256(new Uint8Array(0));
}

/** SHA256(0x00 || data). */
export function leafHash(data: Uint8Array): Uint8Array {
  return sha256(concat(LEAF_PREFIX, data));
}

/** SHA256(0x01 || left || right). */
export function nodeHash(left: Uint8Array, right: Uint8Array): Uint8Array {
  return sha256(concat(NODE_PREFIX, left, right));
}

/**
 * The bytes a leaf hashes over: its canonical (JCS) JSON, UTF-8. A
 * string is taken as already-canonical (store leaves that way and the
 * recompute never depends on a JSON round-trip); an object is
 * canonicalized.
 */
export function leafBytes(leaf: TlogLeaf | string): Uint8Array {
  return utf8Bytes(typeof leaf === "string" ? leaf : jcs(leaf));
}

/** Merkle leaf hash of a transparency-log leaf. */
export function hashLeaf(leaf: TlogLeaf | string): Uint8Array {
  return leafHash(leafBytes(leaf));
}

function largestPowerOfTwoBelow(n: number): number {
  let k = 1;
  while (k * 2 < n) k *= 2;
  return k;
}

function rootRange(hashes: readonly Uint8Array[], lo: number, hi: number): Uint8Array {
  const n = hi - lo;
  if (n === 1) return hashes[lo]!;
  const k = largestPowerOfTwoBelow(n);
  return nodeHash(rootRange(hashes, lo, lo + k), rootRange(hashes, lo + k, hi));
}

/** MTH over leaf hashes, by the recursive definition. */
export function rootFromLeafHashes(hashes: readonly HashLike[]): Uint8Array {
  if (hashes.length === 0) return emptyRoot();
  return rootRange(hashes.map(hashBytes), 0, hashes.length);
}

/**
 * MTH over leaves (canonical strings or leaf objects). This is the
 * full-replica witness check: recompute the root over every leaf held
 * plus the new page and compare to `sth.root_hash`. A match proves the
 * new head extends the old one and that the served leaves are the
 * tree's real leaves; no separate consistency check is needed.
 */
export function rootFromLeaves(leaves: ReadonlyArray<TlogLeaf | string>): Uint8Array {
  return rootFromLeafHashes(leaves.map(hashLeaf));
}

export interface InclusionProofInput {
  /** Hash of the leaf being proven (see `hashLeaf`). */
  leafHash: HashLike;
  leafIndex: number;
  treeSize: number;
  /** Sibling hashes from the leaf up, as served. */
  proof: readonly HashLike[];
  /** Root of the tree at `treeSize`. */
  root: HashLike;
}

/** RFC 9162 §2.1.3.2. */
export function verifyInclusion(input: InclusionProofInput): boolean {
  const { leafIndex, treeSize } = input;
  if (!Number.isSafeInteger(leafIndex) || !Number.isSafeInteger(treeSize)) return false;
  if (leafIndex < 0 || leafIndex >= treeSize) return false;
  let fn = BigInt(leafIndex);
  let sn = BigInt(treeSize - 1);
  let r = hashBytes(input.leafHash);
  for (const p of input.proof) {
    if (sn === 0n) return false;
    const sibling = hashBytes(p);
    if ((fn & 1n) === 1n || fn === sn) {
      r = nodeHash(sibling, r);
      if ((fn & 1n) === 0n) {
        while ((fn & 1n) === 0n && fn !== 0n) {
          fn >>= 1n;
          sn >>= 1n;
        }
      }
    } else {
      r = nodeHash(r, sibling);
    }
    fn >>= 1n;
    sn >>= 1n;
  }
  return sn === 0n && bytesEqual(r, hashBytes(input.root));
}

export interface ConsistencyProofInput {
  oldSize: number;
  newSize: number;
  /** As served for `since = oldSize`. */
  proof: readonly HashLike[];
  oldRoot: HashLike;
  newRoot: HashLike;
}

/**
 * RFC 9162 §2.1.4.2, plus the two cases the algorithm excludes: equal
 * sizes need an empty proof and equal roots, and every tree extends
 * the empty tree (empty proof, `oldRoot` must be the empty root).
 */
export function verifyConsistency(input: ConsistencyProofInput): boolean {
  const { oldSize, newSize } = input;
  if (!Number.isSafeInteger(oldSize) || !Number.isSafeInteger(newSize)) return false;
  if (oldSize < 0 || oldSize > newSize) return false;
  const oldRoot = hashBytes(input.oldRoot);
  const newRoot = hashBytes(input.newRoot);
  if (oldSize === newSize) return input.proof.length === 0 && bytesEqual(oldRoot, newRoot);
  if (oldSize === 0) return input.proof.length === 0 && bytesEqual(oldRoot, emptyRoot());
  if (input.proof.length === 0) return false;

  const path = input.proof.map(hashBytes);
  const isPow2 = (oldSize & (oldSize - 1)) === 0;
  if (isPow2) path.unshift(oldRoot);

  let fn = BigInt(oldSize - 1);
  let sn = BigInt(newSize - 1);
  while ((fn & 1n) === 1n) {
    fn >>= 1n;
    sn >>= 1n;
  }
  let fr = path[0]!;
  let sr = path[0]!;
  for (const c of path.slice(1)) {
    if (sn === 0n) return false;
    if ((fn & 1n) === 1n || fn === sn) {
      fr = nodeHash(c, fr);
      sr = nodeHash(c, sr);
      if ((fn & 1n) === 0n) {
        while ((fn & 1n) === 0n && fn !== 0n) {
          fn >>= 1n;
          sn >>= 1n;
        }
      }
    } else {
      sr = nodeHash(sr, c);
    }
    fn >>= 1n;
    sn >>= 1n;
  }
  return bytesEqual(fr, oldRoot) && bytesEqual(sr, newRoot) && sn === 0n;
}
