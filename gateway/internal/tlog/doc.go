// Package tlog is the gateway's transparency log: an append-only
// Merkle log with one leaf per accounted LLM call, persisted as JSONL
// on the data volume and served to an external witness as a signed
// tree head. See gateway/plans/phases/phase-12-transparency-log.md.
//
// What lives here
// ---------------
//   - merkle.go: the RFC 9162 §2.1 (formerly RFC 6962) SHA-256 Merkle
//     tree — leaf/node hashing, roots, inclusion and consistency
//     proofs, and the verifiers for both. Pure; no I/O.
//   - leaf.go: the wire shape of one leaf and its canonical (JCS)
//     bytes. The canonical bytes are what gets hashed and persisted.
//   - key.go: the per-boot log key. secp256k1, in memory only, never
//     logged. It attests "this gateway served this head" and is not
//     an identity key — Hive countersigns every head it accepts.
//   - sth.go: the signed tree head and its signing input.
//   - log.go: the Log — mutex, tree, leaf file, cached head, paging.
//   - default.go: the process-wide instance the hooks append to and
//     the admin API serves from.
//
// Threat model in one line: the gateway is the party being defended
// against, so nothing here proves anything on its own. What it does
// is make the past un-rewritable once a witness outside the swarm has
// signed a head that covers it.
package tlog
