/**
 * Dev smoke for the gitsee harness services' LIFECYCLE (no docker / playwright /
 * LLM): per-run session keying + onRunEnd disposal idempotency. Run:
 *   npx tsx src/lab/gitsee/services/smoke-services.ts
 */
import assert from "node:assert/strict";
import { buildGitseeServices } from "./index.js";

async function main() {
  const { gitsee, disposeRun } = buildGitseeServices();

  // Two concurrent runs get DISTINCT sessions (no singleton collision). keepUp
  // so teardown is a no-op (no docker calls in this smoke).
  const sA = gitsee.stack.session("runA", "/tmp/wsA", { keepUp: true });
  const sB = gitsee.stack.session("runB", "/tmp/wsB", { keepUp: true });
  assert.notEqual(sA, sB, "distinct stack sessions per runId");
  assert.equal(gitsee.stack.session("runA", "/tmp/wsA", { keepUp: true }), sA, "same session reused for a runId");
  assert.equal(sA.workspacePath, "/tmp/wsA");
  assert.equal(sB.workspacePath, "/tmp/wsB");

  const bA = gitsee.browser.session("runA", "http://localhost:3000", "/tmp/wsA/.shots");
  assert.equal(gitsee.browser.session("runA", "x", "y"), bA, "same browser session reused for a runId");

  assert.ok(gitsee.stack.has("runA") && gitsee.browser.has("runA"));

  // Disposing runA frees ONLY runA's sessions; runB untouched.
  await disposeRun("runA");
  assert.equal(gitsee.stack.has("runA"), false, "runA stack disposed");
  assert.equal(gitsee.browser.has("runA"), false, "runA browser disposed");
  assert.equal(gitsee.stack.has("runB"), true, "runB stack untouched");

  // Idempotent: disposing again (or an unknown run) is a no-op, not a throw.
  await disposeRun("runA");
  await disposeRun("never-existed");

  await disposeRun("runB");
  assert.equal(gitsee.stack.has("runB"), false);

  console.log("[smoke-services] OK — per-run keying + onRunEnd disposal verified");
}

main().catch((e) => {
  console.error("[smoke-services] FAILED:", e);
  process.exit(1);
});
