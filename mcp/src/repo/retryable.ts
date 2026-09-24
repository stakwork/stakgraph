/**
 * Strict cause-chain check for the AI SDK `isRetryable` flag.
 *
 * True only when the thrown error, or an error in its `cause` chain, has
 * `isRetryable === true`. Missing, `false`, or the string `"true"` is false.
 * Walks only `cause` — not `AggregateError.errors`, message text, or status
 * codes. A cycle returns; a non-object stops the walk.
 *
 * Distinct from `isRetryableError` in `eval/simple-evaluator.ts`, which sniffs
 * message text and status codes. Do not reuse that helper here.
 */
export function errorIsRetryable(error: unknown): boolean {
  const seen = new Set<object>();
  let current: unknown = error;
  while (current !== null && typeof current === "object") {
    if (seen.has(current)) return false;
    seen.add(current);
    if ((current as { isRetryable?: unknown }).isRetryable === true) return true;
    current = (current as { cause?: unknown }).cause;
  }
  return false;
}
