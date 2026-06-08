/**
 * Redaction utility for scrubbing credentials from log agent egress points.
 */

export interface RedactOpts {
  literals?: string[];
}

const SENSITIVE_KEY_NAMES = new Set([
  "token",
  "secret",
  "password",
  "api_key",
  "apikey",
  "access_token",
  "auth_token",
]);

/**
 * Redact secrets from a string.
 * Applies replacements in order:
 * 1. Exact literals (e.g. stakworkApiKey)
 * 2. Authorization headers (Token/Bearer/Basic)
 * 3. AWS Access Key IDs (AKIA...)
 * 4. JWTs (eyJ... three base64url segments)
 * 5. Generic key=value assignments (JSON, query-string, env-style)
 */
export function redactSecrets(input: string, opts?: RedactOpts): string {
  let result = input;

  // 1. Exact literals
  if (opts?.literals) {
    for (const literal of opts.literals) {
      if (!literal) continue;
      // Escape special regex characters in the literal
      const escaped = literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      result = result.replace(new RegExp(escaped, "g"), "[REDACTED]");
    }
  }

  // 2. Authorization headers (Token/Bearer/Basic)
  // Use a function replacement to avoid extra space when the group already ends with '='
  result = result.replace(
    /Authorization:\s*(Token token=|Bearer|Basic)\s*\S+/gi,
    (_match, group1: string) => {
      // Token token= already contains '=', so no extra separator needed
      if (group1.endsWith("=")) {
        return `Authorization: ${group1}[REDACTED]`;
      }
      return `Authorization: ${group1} [REDACTED]`;
    }
  );

  // 3. AWS Access Key IDs (AKIA followed by 16 uppercase alphanumeric chars)
  result = result.replace(/AKIA[A-Z0-9]{16}/g, "[REDACTED]");

  // 4. JWTs (three base64url segments separated by dots)
  result = result.replace(
    /eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*/g,
    "[REDACTED]"
  );

  // 5a. Generic JSON key-value assignments
  result = result.replace(
    /"(api_key|token|secret|password|apikey|access_token|auth_token)"\s*:\s*"[^"]*"/gi,
    '"$1": "[REDACTED]"'
  );

  // 5b. Query-string params
  result = result.replace(
    /([?&](api_key|token|secret|password|apikey|access_token|auth_token)=)[^&\s"']+/gi,
    "$1[REDACTED]"
  );

  // 5c. KEY=value env-style lines (case-insensitive to catch both api_key= and API_KEY=)
  result = result.replace(
    /\b(API_KEY|TOKEN|SECRET|PASSWORD|APIKEY|ACCESS_TOKEN|AUTH_TOKEN)=\S+/gi,
    "$1=[REDACTED]"
  );

  return result;
}

/**
 * Recursively traverse an object/array and apply redactSecrets to all string values.
 * When an object key is a known sensitive name (token, secret, password, etc.),
 * the string value is replaced with [REDACTED] directly.
 */
export function redactSecretsDeep(obj: unknown, opts?: RedactOpts): unknown {
  if (typeof obj === "string") {
    return redactSecrets(obj, opts);
  }
  if (Array.isArray(obj)) {
    return obj.map((item) => redactSecretsDeep(item, opts));
  }
  if (obj !== null && typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
      if (SENSITIVE_KEY_NAMES.has(key.toLowerCase()) && typeof value === "string") {
        result[key] = "[REDACTED]";
      } else {
        result[key] = redactSecretsDeep(value, opts);
      }
    }
    return result;
  }
  // number, boolean, null, undefined — pass through as-is
  return obj;
}
