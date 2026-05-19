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
