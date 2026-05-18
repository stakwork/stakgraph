/**
 * `gatekey` — three-layer macaroon for cryptographic LLM governance.
 * Spec: `gateway/plans/phases/phase-4-macaroon-shape.md`.
 */

export type {
  // signature envelopes
  SigAlg,
  Ed25519Sig,
  EcdsaSecp256k1Sig,
  MultisigV1Sig,
  OrgSig,
  UserSig,
  // pubkeys + policies
  Ed25519PubKey,
  EcdsaSecp256k1PubKey,
  PubKey,
  SinglePolicy,
  MultisigPolicy,
  Policy,
  // macaroon layers
  UserPermissions,
  UserAuthorizationUnsigned,
  UserAuthorization,
  InvocationUnsigned,
  Invocation,
  AttenuationCaveats,
  Attenuation,
  Macaroon,
  // verifier output
  EffectiveCaveats,
  Claims,
} from "./types.js";

// Encoding + canonicalization (exposed for advanced use + tests).
export {
  bytesToHex,
  hexToBytes,
  bytesToBase64url,
  base64urlToBytes,
  utf8Bytes,
} from "./encoding.js";

export { jcs, signingBytes } from "./jcs.js";

// Crypto primitives (exposed for keygen / advanced use).
export {
  ed25519Sign,
  ed25519Verify,
  ed25519PublicKey,
  ecdsaSign,
  ecdsaVerify,
  ecdsaPublicKey,
} from "./sigs.js";

// Layer signers (Hive macaroon issuer, custodial phase 1).
export {
  signUserAuthorizationSingle,
  signUserAuthorizationMultisig,
  signInvocation,
  invocationSigBytes,
} from "./sign.js";

// Attenuation (parent agents spawning sub-agents).
export {
  computeAttenuationHmac,
  attenuate,
  attenuationSigBytes,
} from "./attenuate.js";

// Verifier (tests, debug, Hive sanity checks).
export {
  verify,
  decodeMacaroon,
  encodeMacaroon,
  VerifyError,
} from "./verify.js";

export type { VerifyErrorCode } from "./verify.js";
