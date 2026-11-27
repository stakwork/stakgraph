import { describe, it, afterEach } from "node:test";
import assert from "node:assert/strict";
import { getModel, useEnv } from "aieo";
import {
  GATEWAY_MANAGED_API_KEY,
  anthropicClientOptions,
  federationAwareEnv,
  gatewayAnthropicBaseURL,
  getApiKeyForProvider,
  isAnthropicFederation,
  resolveAnthropicApiKey,
} from "./anthropic-federation.js";

const FED = {
  ANTHROPIC_FEDERATION_RULE_ID: "fdrl_1",
  ANTHROPIC_ORGANIZATION_ID: "org-1",
  ANTHROPIC_SERVICE_ACCOUNT_ID: "svac_1",
};
const GW = { LLM_GATEWAY_URL: "http://gateway:8181/" };

describe("isAnthropicFederation", () => {
  it("needs an empty key and all three ids; workspace is optional", () => {
    assert.equal(isAnthropicFederation({ ...FED }), true);
    assert.equal(isAnthropicFederation({ ...FED, ANTHROPIC_WORKSPACE_ID: "wrkspc_1" }), true);
    assert.equal(isAnthropicFederation({ ...FED, ANTHROPIC_API_KEY: "sk-ant-x" }), false);
    assert.equal(isAnthropicFederation({ ...FED, ANTHROPIC_API_KEY: "  " }), true);
    assert.equal(isAnthropicFederation({ ...FED, ANTHROPIC_ORGANIZATION_ID: "" }), false);
    assert.equal(isAnthropicFederation({}), false);
  });
});

describe("resolveAnthropicApiKey / anthropicClientOptions", () => {
  it("key present: unchanged, no base URL added, even with federation vars and a gateway", () => {
    const e = { ...FED, ...GW, ANTHROPIC_API_KEY: "sk-ant-x" };
    assert.equal(resolveAnthropicApiKey(undefined, e), "sk-ant-x");
    assert.deepEqual(anthropicClientOptions("sk-ant-x", e), { apiKey: "sk-ant-x" });
    assert.deepEqual(anthropicClientOptions(undefined, e), { apiKey: "sk-ant-x" });
  });

  it("a caller key (virtual key from the request/grant) always wins in federation mode", () => {
    const e = { ...FED, ...GW };
    assert.equal(resolveAnthropicApiKey("sk-bf-abc.mac", e), "sk-bf-abc.mac");
    assert.deepEqual(anthropicClientOptions("sk-bf-abc", e), { apiKey: "sk-bf-abc" });
  });

  it("key missing + federation + gateway: placeholder and the gateway's Anthropic base URL", () => {
    const e = { ...FED, ...GW };
    assert.equal(resolveAnthropicApiKey(undefined, e), GATEWAY_MANAGED_API_KEY);
    assert.deepEqual(anthropicClientOptions(undefined, e), {
      apiKey: "gateway-managed",
      baseURL: "http://gateway:8181/anthropic/v1",
    });
    assert.equal(gatewayAnthropicBaseURL(e), "http://gateway:8181/anthropic/v1");
  });

  it("key missing + no federation: as before (no key, no throw here)", () => {
    assert.equal(resolveAnthropicApiKey(undefined, { ...GW }), undefined);
    assert.deepEqual(anthropicClientOptions(undefined, {}), { apiKey: undefined });
    // Incomplete federation is not federation.
    assert.equal(resolveAnthropicApiKey(undefined, { ...GW, ANTHROPIC_FEDERATION_RULE_ID: "x" }), undefined);
  });

  it("federation without a gateway: a clear error naming LLM_GATEWAY_URL", () => {
    assert.throws(() => resolveAnthropicApiKey(undefined, { ...FED }), /LLM_GATEWAY_URL/);
    assert.throws(() => anthropicClientOptions(undefined, { ...FED, LLM_GATEWAY_URL: " " }), /federation/i);
  });
});

describe("getApiKeyForProvider", () => {
  const saved = { ...process.env };
  afterEach(() => {
    for (const k of Object.keys(process.env)) if (!(k in saved)) delete process.env[k];
    Object.assign(process.env, saved);
  });
  const reset = () => {
    for (const k of ["ANTHROPIC_API_KEY", ...Object.keys(FED), "ANTHROPIC_WORKSPACE_ID", "LLM_GATEWAY_URL"]) delete process.env[k];
  };

  it("key present: the key", () => {
    reset();
    process.env.ANTHROPIC_API_KEY = "sk-ant-x";
    assert.equal(getApiKeyForProvider("anthropic"), "sk-ant-x");
  });

  it("no key, no federation: aieo's own error, unchanged", () => {
    reset();
    assert.throws(() => getApiKeyForProvider("anthropic"), /API key not found for provider: anthropic/);
  });

  it("federation + gateway: placeholder; without gateway: clear error", () => {
    reset();
    Object.assign(process.env, FED);
    assert.throws(() => getApiKeyForProvider("anthropic", process.env), /LLM_GATEWAY_URL/);
    process.env.LLM_GATEWAY_URL = "http://gateway:8181";
    assert.equal(getApiKeyForProvider("anthropic", process.env), "gateway-managed");
  });

  it("federation does not make other providers' keys appear", () => {
    reset();
    Object.assign(process.env, FED, GW);
    assert.throws(() => getApiKeyForProvider("openai", process.env), /API key not found for provider: openai/);
  });
});

describe("federationAwareEnv (what aieo reads)", () => {
  afterEach(() => useEnv(process.env));

  it("reads the placeholder only in federation mode through a gateway", () => {
    assert.equal(federationAwareEnv({ ...FED, ...GW }).ANTHROPIC_API_KEY, "gateway-managed");
    assert.ok("ANTHROPIC_API_KEY" in federationAwareEnv({ ...FED, ...GW }));
    assert.equal(federationAwareEnv({ ...FED, ...GW }).LLM_GATEWAY_URL, GW.LLM_GATEWAY_URL);
    assert.equal(federationAwareEnv({ ...FED }).ANTHROPIC_API_KEY, undefined);
    assert.equal(federationAwareEnv({ ...GW }).ANTHROPIC_API_KEY, undefined);
    assert.equal(federationAwareEnv({ ...FED, ...GW, ANTHROPIC_API_KEY: "sk-ant-x" }).ANTHROPIC_API_KEY, "sk-ant-x");
  });

  it("is lazy: follows the underlying env as it changes", () => {
    const base: Record<string, string | undefined> = { ...FED };
    const view = federationAwareEnv(base);
    assert.equal(view.ANTHROPIC_API_KEY, undefined);
    base.LLM_GATEWAY_URL = "http://gateway:8181";
    assert.equal(view.ANTHROPIC_API_KEY, "gateway-managed");
  });

  it("aieo's getModel builds an Anthropic model through it, and throws without it", () => {
    const base = { ...FED, ...GW };
    useEnv({ ...FED } as NodeJS.ProcessEnv);
    assert.throws(() => getModel("anthropic", {}), /API key not found/);
    useEnv(federationAwareEnv(base));
    assert.doesNotThrow(() => getModel("anthropic", {}));
  });
});
