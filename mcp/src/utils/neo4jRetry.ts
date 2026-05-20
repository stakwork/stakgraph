import neo4j, { Driver, Session } from "neo4j-driver";

/**
 * Single source of truth for Neo4j driver construction.
 * Uses bolt:// instead of neo4j:// to avoid routing table discovery,
 * which can fail permanently after DNS resolution errors (EAI_AGAIN).
 */
export function createNeo4jDriver(): Driver {
  const host = process.env.NEO4J_HOST || "localhost:7687";
  const user = process.env.NEO4J_USER || "neo4j";
  const pswd = process.env.NEO4J_PASSWORD || "testtest";
  const uri = `bolt://${host}`;
  return neo4j.driver(uri, neo4j.auth.basic(user, pswd));
}

const TRANSIENT_CODES = new Set([
  "ServiceUnavailable",
  "SessionExpired",
  "Neo.TransientError.General.DatabaseUnavailable",
]);

function isTransient(err: any): boolean {
  if (!err) return false;
  const code: string = err.code || err.name || "";
  if (TRANSIENT_CODES.has(code)) return true;
  const msg: string = err.message || "";
  return msg.includes("EAI_AGAIN") || msg.includes("ServiceUnavailable") || msg.includes("SessionExpired");
}

/**
 * Wraps a Neo4j session operation with exponential-backoff retry and
 * driver recreation on transient errors (ServiceUnavailable, SessionExpired,
 * EAI_AGAIN). Mirrors the Rust `with_transient_retry_reconnect` pattern.
 *
 * @param getDriver  Returns the current driver instance
 * @param setDriver  Called with a freshly created driver after recreation
 * @param op         The session operation to run
 * @param label      Human-readable label for log messages
 * @param maxAttempts Number of total attempts (default from NEO4J_RETRY_ATTEMPTS env, fallback 3)
 */
export async function withNeo4jRetry<T>(
  getDriver: () => Driver,
  setDriver: (d: Driver) => void,
  op: (session: Session) => Promise<T>,
  label: string,
  maxAttempts: number = parseInt(process.env.NEO4J_RETRY_ATTEMPTS || "3", 10)
): Promise<T> {
  let attempt = 0;

  while (true) {
    const session = getDriver().session();
    try {
      const result = await op(session);
      return result;
    } catch (err: any) {
      await session.close().catch(() => {});

      if (!isTransient(err) || attempt >= maxAttempts - 1) {
        throw err;
      }

      // Exponential backoff: 50ms * 2^attempt, capped at 2^6 doublings
      const backoffMs = 50 * Math.pow(2, Math.min(attempt, 6));
      console.warn(
        `[neo4j-retry] transient error on '${label}' (attempt ${attempt + 1}/${maxAttempts}), retrying in ${backoffMs}ms: ${err?.message || err}`
      );

      // Recreate the driver to clear stale routing/connection state
      try {
        await getDriver().close();
      } catch (_) {
        // ignore close errors
      }
      setDriver(createNeo4jDriver());

      await new Promise((resolve) => setTimeout(resolve, backoffMs));
      attempt++;
    } finally {
      // Session may already be closed on error path — close is idempotent
      await session.close().catch(() => {});
    }
  }
}
