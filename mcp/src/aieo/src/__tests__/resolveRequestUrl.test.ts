import { test, expect } from "@playwright/test";
import { resolveRequestUrl } from "../provider.js";

// ---------------------------------------------------------------------------
// resolveRequestUrl — unit tests
//
// Verifies the fully-qualified HTTP endpoint returned per provider,
// both with an explicit baseUrl (gateway) and without (direct / no env).
// ---------------------------------------------------------------------------

test.describe("resolveRequestUrl — with explicit baseUrl (gateway)", () => {
  test("anthropic → <gatewayBase>/messages", () => {
    expect(resolveRequestUrl("anthropic", "http://gw:3000")).toBe(
      "http://gw:3000/anthropic/v1/messages",
    );
  });

  test("openai → <gatewayBase>/chat/completions", () => {
    expect(resolveRequestUrl("openai", "http://gw:3000")).toBe(
      "http://gw:3000/openai/v1/chat/completions",
    );
  });

  test("openrouter → <gatewayBase>/chat/completions (rides openai path)", () => {
    expect(resolveRequestUrl("openrouter", "http://gw:3000")).toBe(
      "http://gw:3000/openai/v1/chat/completions",
    );
  });

  test("google → <gatewayRoot>/openai/v1/chat/completions (compat path)", () => {
    expect(resolveRequestUrl("google", "http://gw:3000")).toBe(
      "http://gw:3000/openai/v1/chat/completions",
    );
  });

  test("google strips provider suffix to recover gateway root", () => {
    // gatewayUrlFor("google", "http://gw:3000") → "http://gw:3000/genai/v1beta"
    // resolveRequestUrl must strip /genai/v1beta and re-target /openai/v1/chat/completions
    const result = resolveRequestUrl("google", "http://gw:3000");
    expect(result).toBe("http://gw:3000/openai/v1/chat/completions");
    // Confirm the genai path is NOT present in the final URL
    expect(result).not.toContain("/genai/");
  });
});

test.describe("resolveRequestUrl — without baseUrl and no LLM_GATEWAY_URL env", () => {
  // These tests assume LLM_GATEWAY_URL is not set in the test environment.
  // If it is set, getGatewayBaseURL() will return a value and the test
  // would produce a URL instead of undefined — skip gracefully in that case.

  test("anthropic → undefined (no gateway configured)", () => {
    if (process.env.LLM_GATEWAY_URL) {
      test.skip(true, "LLM_GATEWAY_URL is set; skipping no-gateway test");
      return;
    }
    expect(resolveRequestUrl("anthropic", undefined)).toBeUndefined();
  });

  test("openai → undefined (no gateway configured)", () => {
    if (process.env.LLM_GATEWAY_URL) {
      test.skip(true, "LLM_GATEWAY_URL is set; skipping no-gateway test");
      return;
    }
    expect(resolveRequestUrl("openai", undefined)).toBeUndefined();
  });

  test("openrouter → undefined (no gateway configured)", () => {
    if (process.env.LLM_GATEWAY_URL) {
      test.skip(true, "LLM_GATEWAY_URL is set; skipping no-gateway test");
      return;
    }
    expect(resolveRequestUrl("openrouter", undefined)).toBeUndefined();
  });

  test("google → undefined (no gateway configured)", () => {
    if (process.env.LLM_GATEWAY_URL) {
      test.skip(true, "LLM_GATEWAY_URL is set; skipping no-gateway test");
      return;
    }
    expect(resolveRequestUrl("google", undefined)).toBeUndefined();
  });
});
