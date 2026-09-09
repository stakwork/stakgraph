/**
 * Tests for withNeo4jRetry, which wraps every Neo4j query in the app
 * (graph/neo4j.ts and lab/concepts/store/graphStorage.ts).
 *
 * Retry classification is exercised through the public function rather than
 * the private isTransient, so these assert behavior rather than internals.
 */
import { test, expect } from "../../testkit.js";
import type { Driver, Session } from "neo4j-driver";
import { withNeo4jRetry } from "../neo4jRetry.js";

interface Harness {
  driver: Driver;
  sessionsOpened: number;
  sessionsClosed: number;
  driversSet: number;
  getDriver: () => Driver;
  setDriver: (d: Driver) => void;
}

function harness(): Harness {
  const h: Partial<Harness> = { sessionsOpened: 0, sessionsClosed: 0, driversSet: 0 };

  const makeDriver = (): Driver =>
    ({
      session: () => {
        h.sessionsOpened!++;
        return {
          close: async () => {
            h.sessionsClosed!++;
          },
        } as unknown as Session;
      },
      close: async () => {},
    } as unknown as Driver);

  h.driver = makeDriver();
  h.getDriver = () => h.driver!;
  h.setDriver = (d: Driver) => {
    h.driversSet!++;
    // withNeo4jRetry builds a real (lazy, unconnected) driver on each retry.
    // Close it so no handle outlives the test.
    void d.close();
    h.driver = makeDriver();
  };

  return h as Harness;
}

function transient(message = "ServiceUnavailable"): Error {
  const e = new Error(message);
  (e as Error & { code: string }).code = "ServiceUnavailable";
  return e;
}

test.describe("withNeo4jRetry", () => {
  test("returns the result and closes the session on first success", async () => {
    const h = harness();
    const result = await withNeo4jRetry(h.getDriver, h.setDriver, async () => "ok", "label", 3);

    expect(result).toBe("ok");
    expect(h.sessionsOpened).toBe(1);
    expect(h.sessionsClosed).toBe(1);
    expect(h.driversSet).toBe(0);
  });

  test("retries a transient error and succeeds on a later attempt", async () => {
    const h = harness();
    let calls = 0;

    const result = await withNeo4jRetry(
      h.getDriver,
      h.setDriver,
      async () => {
        calls++;
        if (calls < 3) throw transient();
        return "recovered";
      },
      "label",
      3
    );

    expect(result).toBe("recovered");
    expect(calls).toBe(3);
    expect(h.sessionsOpened).toBe(3);
  });

  test("recreates the driver on each retry", async () => {
    const h = harness();
    let calls = 0;

    await withNeo4jRetry(
      h.getDriver,
      h.setDriver,
      async () => {
        calls++;
        if (calls < 3) throw transient();
        return "recovered";
      },
      "label",
      3
    );

    expect(h.driversSet).toBe(2);
  });

  test("rethrows the original error once maxAttempts is exhausted", async () => {
    const h = harness();
    let calls = 0;
    const err = transient("still down");

    await expect(
      withNeo4jRetry(
        h.getDriver,
        h.setDriver,
        async () => {
          calls++;
          throw err;
        },
        "label",
        3
      )
    ).rejects.toThrow("still down");

    expect(calls).toBe(3);
  });

  test("does not retry a non-transient error", async () => {
    const h = harness();
    let calls = 0;

    await expect(
      withNeo4jRetry(
        h.getDriver,
        h.setDriver,
        async () => {
          calls++;
          throw new Error("Neo.ClientError.Statement.SyntaxError");
        },
        "label",
        3
      )
    ).rejects.toThrow("SyntaxError");

    expect(calls).toBe(1);
    expect(h.driversSet).toBe(0);
  });

  test("treats SessionExpired and DatabaseUnavailable codes as transient", async () => {
    for (const code of [
      "SessionExpired",
      "Neo.TransientError.General.DatabaseUnavailable",
    ]) {
      const h = harness();
      let calls = 0;

      await withNeo4jRetry(
        h.getDriver,
        h.setDriver,
        async () => {
          calls++;
          if (calls < 2) {
            const e = new Error("boom");
            (e as Error & { code: string }).code = code;
            throw e;
          }
          return "ok";
        },
        "label",
        3
      );

      expect(calls).toBe(2);
    }
  });

  test("treats an EAI_AGAIN message as transient even without a code", async () => {
    const h = harness();
    let calls = 0;

    await withNeo4jRetry(
      h.getDriver,
      h.setDriver,
      async () => {
        calls++;
        if (calls < 2) throw new Error("getaddrinfo EAI_AGAIN neo4j");
        return "ok";
      },
      "label",
      3
    );

    expect(calls).toBe(2);
  });

  test("maxAttempts of 1 means no retry at all", async () => {
    const h = harness();
    let calls = 0;

    await expect(
      withNeo4jRetry(
        h.getDriver,
        h.setDriver,
        async () => {
          calls++;
          throw transient();
        },
        "label",
        1
      )
    ).rejects.toThrow();

    expect(calls).toBe(1);
    expect(h.driversSet).toBe(0);
  });
});
