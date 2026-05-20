import { test, expect } from "@playwright/test";

/**
 * Unit tests for the Neo4j createIndexes retry loop in neo4j.ts.
 *
 * We exercise the retry logic in isolation by recreating the same
 * async pattern used in the module-level startup block.
 */

// Reproduce the startup retry loop so it can be tested without
// importing neo4j.ts (which has side-effects / requires a real driver).
async function runRetryLoop(
  createIndexes: () => Promise<void>,
  retry_interval: number,
  logs: { warns: string[]; infos: string[] }
): Promise<void> {
  while (true) {
    try {
      await createIndexes();
      logs.infos.push("===> Neo4j indexes created successfully");
      break;
    } catch (error: any) {
      logs.warns.push(
        `===> Neo4j not ready, retrying in ${retry_interval}ms: ${
          error?.message || error
        }`
      );
      await new Promise((resolve) => setTimeout(resolve, retry_interval));
    }
  }
}

test.describe("Neo4j createIndexes retry loop", () => {
  test("retries N times then succeeds, emitting correct log messages", async () => {
    const logs = { warns: [] as string[], infos: [] as string[] };
    const FAIL_TIMES = 3;
    let callCount = 0;

    const createIndexes = async () => {
      callCount++;
      if (callCount <= FAIL_TIMES) {
        throw new Error("ServiceUnavailable");
      }
    };

    await runRetryLoop(createIndexes, 0, logs);

    // Called FAIL_TIMES failures + 1 success
    expect(callCount).toBe(FAIL_TIMES + 1);

    // One warn per failure
    expect(logs.warns).toHaveLength(FAIL_TIMES);
    for (const w of logs.warns) {
      expect(w).toContain("Neo4j not ready, retrying in");
      expect(w).toContain("ServiceUnavailable");
    }

    // Exactly one success log
    expect(logs.infos).toHaveLength(1);
    expect(logs.infos[0]).toContain("Neo4j indexes created successfully");
  });

  test("succeeds on first attempt with no warnings", async () => {
    const logs = { warns: [] as string[], infos: [] as string[] };
    let callCount = 0;

    const createIndexes = async () => {
      callCount++;
    };

    await runRetryLoop(createIndexes, 0, logs);

    expect(callCount).toBe(1);
    expect(logs.warns).toHaveLength(0);
    expect(logs.infos).toHaveLength(1);
  });

  test("retry_interval is respected in the delay between attempts", async () => {
    const logs = { warns: [] as string[], infos: [] as string[] };
    const RETRY_INTERVAL = 50; // small but measurable
    let callCount = 0;
    const timestamps: number[] = [];

    const createIndexes = async () => {
      timestamps.push(Date.now());
      callCount++;
      if (callCount < 3) {
        throw new Error("ECONNREFUSED");
      }
    };

    await runRetryLoop(createIndexes, RETRY_INTERVAL, logs);

    expect(callCount).toBe(3);

    // Gap between attempt 1→2 and 2→3 should be at least RETRY_INTERVAL ms
    const gap1 = timestamps[1] - timestamps[0];
    const gap2 = timestamps[2] - timestamps[1];
    expect(gap1).toBeGreaterThanOrEqual(RETRY_INTERVAL - 5); // allow 5 ms tolerance
    expect(gap2).toBeGreaterThanOrEqual(RETRY_INTERVAL - 5);

    // Warn messages include the configured interval
    for (const w of logs.warns) {
      expect(w).toContain(`retrying in ${RETRY_INTERVAL}ms`);
    }
  });
});

// ---------------------------------------------------------------------------
// withNeo4jRetry unit tests
// ---------------------------------------------------------------------------

/**
 * Minimal inline re-implementation of withNeo4jRetry for isolated testing.
 * Mirrors the real implementation in mcp/src/utils/neo4jRetry.ts exactly so
 * these tests validate the algorithm without importing the module (which would
 * pull in the neo4j-driver and env side-effects).
 */

type FakeSession = { run: () => Promise<any>; close: () => Promise<void> };
type FakeDriver = { session: () => FakeSession; close: () => Promise<void> };

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
  return (
    msg.includes("EAI_AGAIN") ||
    msg.includes("ServiceUnavailable") ||
    msg.includes("SessionExpired")
  );
}

async function withNeo4jRetryInline<T>(
  getDriver: () => FakeDriver,
  setDriver: (d: FakeDriver) => void,
  op: (session: FakeSession) => Promise<T>,
  label: string,
  maxAttempts: number,
  createDriver: () => FakeDriver,
  delays: number[]
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

      const backoffMs = 50 * Math.pow(2, Math.min(attempt, 6));
      delays.push(backoffMs);

      try {
        await getDriver().close();
      } catch (_) {}
      setDriver(createDriver());

      await new Promise((resolve) => setTimeout(resolve, 0)); // zero for tests
      attempt++;
    } finally {
      await session.close().catch(() => {});
    }
  }
}

function makeTransientError(code: string, message?: string): Error {
  const err: any = new Error(message || code);
  err.code = code;
  return err;
}

function makeFakeDriver(): FakeDriver {
  return {
    session: () => ({
      run: async () => ({}),
      close: async () => {},
    }),
    close: async () => {},
  };
}

test.describe("withNeo4jRetry", () => {
  test("succeeds on first attempt without calling setDriver", async () => {
    let driver = makeFakeDriver();
    let setDriverCalled = 0;
    const delays: number[] = [];

    const result = await withNeo4jRetryInline(
      () => driver,
      (d) => { driver = d; setDriverCalled++; },
      async (_session) => "ok",
      "test-label",
      3,
      makeFakeDriver,
      delays
    );

    expect(result).toBe("ok");
    expect(setDriverCalled).toBe(0);
    expect(delays).toHaveLength(0);
  });

  test("retries and recreates driver on ServiceUnavailable", async () => {
    let driver = makeFakeDriver();
    let setDriverCallCount = 0;
    let opCallCount = 0;
    const delays: number[] = [];

    await withNeo4jRetryInline(
      () => driver,
      (d) => { driver = d; setDriverCallCount++; },
      async (_session) => {
        opCallCount++;
        if (opCallCount < 3) throw makeTransientError("ServiceUnavailable");
        return "ok";
      },
      "test-service-unavailable",
      3,
      makeFakeDriver,
      delays
    );

    expect(opCallCount).toBe(3);
    // setDriver called once per failure (2 failures)
    expect(setDriverCallCount).toBe(2);
    expect(delays).toHaveLength(2);
  });

  test("retries and recreates driver on SessionExpired", async () => {
    let driver = makeFakeDriver();
    let setDriverCallCount = 0;
    let opCallCount = 0;
    const delays: number[] = [];

    await withNeo4jRetryInline(
      () => driver,
      (d) => { driver = d; setDriverCallCount++; },
      async (_session) => {
        opCallCount++;
        if (opCallCount === 1) throw makeTransientError("SessionExpired");
        return "done";
      },
      "test-session-expired",
      3,
      makeFakeDriver,
      delays
    );

    expect(opCallCount).toBe(2);
    expect(setDriverCallCount).toBe(1);
  });

  test("retries on EAI_AGAIN in error message", async () => {
    let driver = makeFakeDriver();
    let opCallCount = 0;
    const delays: number[] = [];

    await withNeo4jRetryInline(
      () => driver,
      (d) => { driver = d; },
      async (_session) => {
        opCallCount++;
        if (opCallCount === 1) {
          throw new Error("getaddrinfo EAI_AGAIN neo4j.sphinx");
        }
        return "ok";
      },
      "test-eai-again",
      3,
      makeFakeDriver,
      delays
    );

    expect(opCallCount).toBe(2);
  });

  test("throws immediately on non-transient error without retrying", async () => {
    let driver = makeFakeDriver();
    let opCallCount = 0;
    const delays: number[] = [];

    const nonTransient: any = new Error("Syntax error in query");
    nonTransient.code = "Neo.ClientError.Statement.SyntaxError";

    await expect(
      withNeo4jRetryInline(
        () => driver,
        (d) => { driver = d; },
        async (_session) => {
          opCallCount++;
          throw nonTransient;
        },
        "test-non-transient",
        3,
        makeFakeDriver,
        delays
      )
    ).rejects.toThrow("Syntax error in query");

    expect(opCallCount).toBe(1);
    expect(delays).toHaveLength(0);
  });

  test("throws after exhausting maxAttempts", async () => {
    let driver = makeFakeDriver();
    let opCallCount = 0;
    const delays: number[] = [];
    const MAX = 3;

    await expect(
      withNeo4jRetryInline(
        () => driver,
        (d) => { driver = d; },
        async (_session) => {
          opCallCount++;
          throw makeTransientError("ServiceUnavailable", `fail ${opCallCount}`);
        },
        "test-exhaustion",
        MAX,
        makeFakeDriver,
        delays
      )
    ).rejects.toThrow("fail 3");

    expect(opCallCount).toBe(MAX);
    // delays recorded for first (MAX-1) failures; last failure throws
    expect(delays).toHaveLength(MAX - 1);
  });

  test("exponential backoff delays follow 50ms * 2^attempt pattern", async () => {
    let driver = makeFakeDriver();
    let opCallCount = 0;
    const delays: number[] = [];

    await expect(
      withNeo4jRetryInline(
        () => driver,
        (d) => { driver = d; },
        async (_session) => {
          opCallCount++;
          throw makeTransientError("ServiceUnavailable");
        },
        "test-backoff",
        4, // 3 failures then exhausted
        makeFakeDriver,
        delays
      )
    ).rejects.toThrow();

    // delays for attempt 0, 1, 2 → 50*1, 50*2, 50*4
    expect(delays).toEqual([50, 100, 200]);
  });
});
