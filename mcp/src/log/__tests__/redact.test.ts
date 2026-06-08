import { test, expect } from "@playwright/test";
import { redactSecrets, redactSecretsDeep } from "../redact.js";

test.describe("redactSecrets", () => {
  // 1. Exact literal masking
  test("masks exact stakworkApiKey literal", () => {
    const key = "29abcdefgh123456";
    const result = redactSecrets(`curl -H 'X-Api-Key: ${key}'`, { literals: [key] });
    expect(result).not.toContain(key);
    expect(result).toContain("[REDACTED]");
  });

  // 2. Authorization header — Token
  test("masks Authorization Token header in curl string", () => {
    const input = `curl -s -H 'Authorization: Token token=29xxxxxxxxxxxx' https://example.com`;
    const result = redactSecrets(input);
    expect(result).toContain("Authorization: Token token=[REDACTED]");
    expect(result).not.toContain("29xxxxxxxxxxxx");
  });

  // 3. Authorization Bearer
  test("masks Authorization Bearer header", () => {
    const result = redactSecrets("Authorization: Bearer abc123");
    expect(result).toContain("Authorization: Bearer [REDACTED]");
  });

  // 4. AWS key
  test("masks AWS AKIA key", () => {
    const result = redactSecrets("aws_key=AKIAIOSFODNN7EXAMPLE");
    expect(result).not.toContain("AKIAIOSFODNN7EXAMPLE");
    expect(result).toContain("[REDACTED]");
  });

  // 5. JWT
  test("masks JWT token", () => {
    const jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature";
    const result = redactSecrets(`token: ${jwt}`);
    expect(result).not.toContain(jwt);
    expect(result).toContain("[REDACTED]");
  });

  // 6. JSON key-value
  test("masks JSON api_key value", () => {
    const result = redactSecrets('{"api_key": "supersecret"}');
    expect(result).toContain('"api_key": "[REDACTED]"');
  });

  // 7. Query-string token
  test("masks query-string token param", () => {
    const result = redactSecrets("https://api.example.com/data?token=abc123&foo=bar");
    expect(result).toContain("token=[REDACTED]");
    expect(result).not.toContain("abc123");
  });

  // 8. KEY=value env-style
  test("masks KEY=value env variable", () => {
    const result = redactSecrets("SECRET=mysecretvalue");
    expect(result).toContain("SECRET=[REDACTED]");
    expect(result).not.toContain("mysecretvalue");
  });
});

test.describe("redactSecretsDeep", () => {
  test("redacts string values in nested objects", () => {
    const obj = { command: "curl -H 'Authorization: Bearer tok123'", meta: { token: "tok123" } };
    const result = redactSecretsDeep(obj) as any;
    expect(result.command).not.toContain("tok123");
    expect(result.meta.token).not.toContain("tok123");
  });

  test("redacts strings inside arrays", () => {
    const arr = ["api_key=value", "normal text"];
    const result = redactSecretsDeep(arr) as string[];
    expect(result[0]).toContain("[REDACTED]");
    expect(result[1]).toBe("normal text");
  });

  test("passes through non-string primitives unchanged", () => {
    const obj = { count: 42, flag: true, nothing: null };
    expect(redactSecretsDeep(obj)).toEqual(obj);
  });

  test("applies literals option deeply", () => {
    const key = "my-secret-key";
    const obj = { nested: { value: `result with ${key} embedded` } };
    const result = redactSecretsDeep(obj, { literals: [key] }) as any;
    expect(result.nested.value).not.toContain(key);
    expect(result.nested.value).toContain("[REDACTED]");
  });
});
