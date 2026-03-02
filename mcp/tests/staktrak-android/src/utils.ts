const LOG_PREFIX = "[staktrak-android]";

export function logInfo(event: string, details?: Record<string, unknown>): void {
  if (details) {
    console.log(`${LOG_PREFIX} ${event}`, details);
    return;
  }
  console.log(`${LOG_PREFIX} ${event}`);
}

export function logError(event: string, error: unknown, details?: Record<string, unknown>): void {
  const message = error instanceof Error ? error.message : String(error);
  if (details) {
    console.error(`${LOG_PREFIX} ${event}: ${message}`, details);
    return;
  }
  console.error(`${LOG_PREFIX} ${event}: ${message}`);
}