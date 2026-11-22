/**
 * Anthropic workload identity federation (WIF) on the LLM gateway.
 *
 * With ANTHROPIC_API_KEY empty and ANTHROPIC_FEDERATION_RULE_ID,
 * ANTHROPIC_ORGANIZATION_ID and ANTHROPIC_SERVICE_ACCOUNT_ID set, the real
 * Anthropic credential lives in the gateway (Bifrost, gateway/) and mcp never
 * holds one: every Anthropic call must go through LLM_GATEWAY_URL.
 *
 * What the gateway does with an Anthropic key the client sends (verified in
 * Bifrost transports/v2.5.1):
 *  - It is NOT used as the provider credential. A non-virtual-key
 *    `x-api-key` / `Authorization: Bearer` value only becomes one when the
 *    server has `allow_direct_keys` AND the request carries
 *    `x-bf-direct-key: true` (lib/ctx.go DirectKeyFromHeaders, ~l.59). The
 *    Anthropic route reads `x-api-key` for virtual keys only when it starts
 *    with `sk-bf-` (lib/ctx.go ~l.537-551), and drops it from the headers it
 *    forwards (integrations/router.go ~l.3709).
 *  - It is NOT enough to authenticate, though: with `enforce_auth_on_inference`
 *    (our gateway/data/config.json) the auth middleware admits an inference
 *    request only with a virtual key (`sk-bf-…`, `x-bf-vk`) or admin
 *    credentials (handlers/middlewares.go virtualKeyFromHeaders,
 *    ~l.1428-1452). So the placeholder below only keeps the SDKs from
 *    refusing to build a client; a caller that must pass the gateway's auth
 *    still has to supply its virtual key (the existing request-body / grant
 *    `apiKey` flow, which is always used unchanged when present).
 *  - Our wrapper (gateway/wrapper/authsplit.go) splits `<vk>.<macaroon>` only
 *    when the value matches `sk-bf-…`; a placeholder has no such shape and is
 *    left alone (no macaroon is invented from it).
 */
import { getApiKeyForProvider as aieoGetApiKeyForProvider } from "aieo";
import { env as defaultEnv } from "./env.js";

type Env = Record<string, string | undefined>;

/** Sent as the Anthropic API key when the gateway owns the credential. */
export const GATEWAY_MANAGED_API_KEY = "gateway-managed";

const FEDERATION_VARS = [
  "ANTHROPIC_FEDERATION_RULE_ID",
  "ANTHROPIC_ORGANIZATION_ID",
  "ANTHROPIC_SERVICE_ACCOUNT_ID",
] as const;

const set = (v: string | undefined): boolean => !!v && v.trim() !== "";

/** True iff ANTHROPIC_API_KEY is empty AND the three federation ids are set
 *  (ANTHROPIC_WORKSPACE_ID is optional). Same rule as gateway/wrapper/wif.go. */
export function isAnthropicFederation(e: Env = defaultEnv): boolean {
  return !set(e.ANTHROPIC_API_KEY) && FEDERATION_VARS.every((k) => set(e[k]));
}

/** LLM_GATEWAY_URL without a trailing slash, or undefined. */
export function gatewayUrl(e: Env = defaultEnv): string | undefined {
  const u = e.LLM_GATEWAY_URL?.trim().replace(/\/+$/, "");
  return u || undefined;
}

/** Base URL for an Anthropic SDK through the gateway. Same convention as
 *  aieo's getGatewayBaseURL: `${gateway}/anthropic/v1` (@ai-sdk/anthropic's
 *  baseURL includes `/v1`). */
export function gatewayAnthropicBaseURL(e: Env = defaultEnv): string | undefined {
  const g = gatewayUrl(e);
  return g ? `${g}/anthropic/v1` : undefined;
}

/** The clear error for federation without a gateway. */
export function federationWithoutGatewayError(): Error {
  return new Error(
    "ANTHROPIC_API_KEY is not set and Anthropic workload identity federation is configured " +
      "(ANTHROPIC_FEDERATION_RULE_ID, ANTHROPIC_ORGANIZATION_ID, ANTHROPIC_SERVICE_ACCOUNT_ID), " +
      "but LLM_GATEWAY_URL is not set. Federation only works through the LLM gateway: " +
      "set LLM_GATEWAY_URL, or set ANTHROPIC_API_KEY.",
  );
}

/** Log (once, at boot) a federation setup that cannot work. */
export function warnIfFederationWithoutGateway(e: Env = defaultEnv): void {
  if (isAnthropicFederation(e) && !gatewayUrl(e)) {
    console.error(`[anthropic-federation] ERROR: ${federationWithoutGatewayError().message}`);
  }
}

/**
 * The Anthropic key to hand an SDK. A caller/env key wins, unchanged. Without
 * one: in federation mode through a gateway the placeholder; in federation
 * mode without a gateway the clear error; otherwise `undefined` (callers keep
 * whatever "no key" meant before).
 */
export function resolveAnthropicApiKey(key?: string, e: Env = defaultEnv): string | undefined {
  if (set(key)) return key;
  if (set(e.ANTHROPIC_API_KEY)) return e.ANTHROPIC_API_KEY;
  if (!isAnthropicFederation(e)) return undefined;
  if (!gatewayUrl(e)) throw federationWithoutGatewayError();
  return GATEWAY_MANAGED_API_KEY;
}

/** `{ apiKey, baseURL }` for a direct `createAnthropic(...)`. With a key (or
 *  outside federation) it is exactly `{ apiKey }`: no base URL is added, so
 *  nothing changes for key deployments. In federation mode it is the
 *  placeholder key plus the gateway base URL. */
export function anthropicClientOptions(
  key?: string,
  e: Env = defaultEnv,
): { apiKey: string | undefined; baseURL?: string } {
  const apiKey = resolveAnthropicApiKey(key, e);
  if (apiKey === GATEWAY_MANAGED_API_KEY && !set(key)) {
    return { apiKey, baseURL: gatewayAnthropicBaseURL(e) };
  }
  return { apiKey };
}

/**
 * The environment aieo should read (`useEnv`): `env`, except that in
 * federation mode through a gateway ANTHROPIC_API_KEY reads as the
 * placeholder, so aieo's key lookup (getApiKeyForProvider, resolveLLMConfig,
 * getModel) does not throw. aieo is an npm dependency, so this is the one hook
 * we have. It routes via LLM_GATEWAY_URL by itself (aieo getGatewayBaseURL).
 * Reads are lazy: the env may be frozen/changed after this is installed.
 */
export function federationAwareEnv(base: Env = defaultEnv): NodeJS.ProcessEnv {
  const placeholder = () => isAnthropicFederation(base) && !!gatewayUrl(base);
  return new Proxy({} as NodeJS.ProcessEnv, {
    get: (_t, k) => {
      if (k === "ANTHROPIC_API_KEY" && placeholder()) return GATEWAY_MANAGED_API_KEY;
      return typeof k === "string" ? base[k] : undefined;
    },
    has: (_t, k) =>
      typeof k === "string" && ((k === "ANTHROPIC_API_KEY" && placeholder()) || k in base),
    ownKeys: () => Reflect.ownKeys(base),
    getOwnPropertyDescriptor: (_t, k) => {
      const d = Object.getOwnPropertyDescriptor(base, k);
      return d && { ...d, configurable: true };
    },
  });
}

/** aieo's getApiKeyForProvider, except for Anthropic in federation mode: the
 *  placeholder through a gateway, the clear error without one (instead of
 *  "API key not found for provider: anthropic"). Anything else is aieo's. */
export function getApiKeyForProvider(provider: string, e: Env = defaultEnv): string {
  if (provider === "anthropic" && isAnthropicFederation(e)) {
    if (!gatewayUrl(e)) throw federationWithoutGatewayError();
    return GATEWAY_MANAGED_API_KEY;
  }
  return aieoGetApiKeyForProvider(provider as Parameters<typeof aieoGetApiKeyForProvider>[0]);
}
