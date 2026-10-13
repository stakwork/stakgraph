import { test } from "node:test";
import assert from "node:assert/strict";
import recordUsage from "../steps/record-usage.js";

function harness() {
  const calls: Array<{ repo: string; usage: unknown }> = [];
  const ctx = {
    services: {
      storage: {
        async addToTotalUsage(repo: string, usage: unknown) {
          calls.push({ repo, usage });
        },
      },
    },
  };
  return { ctx, calls };
}

test("sums the step usages it is given and adds them to the repo's total", async () => {
  const { ctx, calls } = harness();
  const out = await recordUsage.run(
    recordUsage.input.parse({
      owner: "o",
      repo: "r",
      usage: [
        { input: 10, cache_read: 2, cache_write: 1, output: 5, total: 18 },
        undefined, // a skipped step
        { input: "x", total: NaN }, // malformed
        { input: 1, output: 1, total: 2, inputTokens: 999 }, // legacy fields are not summed
      ],
    }),
    ctx as never,
  );
  const usage = { input: 11, cache_read: 2, cache_write: 1, output: 6, total: 20, inputTokens: 14, outputTokens: 6, totalTokens: 20 };
  assert.deepEqual(calls, [{ repo: "o/r", usage }]);
  assert.deepEqual(out, { repo: "o/r", usage });
});

test("records zero when given nothing", async () => {
  const { ctx, calls } = harness();
  await recordUsage.run(recordUsage.input.parse({ owner: "o", repo: "r" }), ctx as never);
  assert.equal(calls.length, 1);
  assert.equal((calls[0]!.usage as { total: number }).total, 0);
});
