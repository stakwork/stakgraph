import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileArtifactsCapability } from "./capabilities.js";

describe("fileArtifactsCapability (per-run artifact files)", () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "vein-artifacts-"));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("dir() creates the run dir on demand and returns its absolute path", async () => {
    const a = fileArtifactsCapability(root);
    const d = await a.dir("run1");
    assert.equal(d, join(root, "run1"));
    // Idempotent.
    assert.equal(await a.dir("run1"), d);
  });

  it("write → read round-trips text (and creates subdirectories)", async () => {
    const a = fileArtifactsCapability(root);
    const abs = await a.write("run1", "reports/summary.md", "# hello");
    assert.equal(abs, join(root, "run1", "reports", "summary.md"));
    const bytes = await a.read("run1", "reports/summary.md");
    assert.equal(Buffer.from(bytes).toString(), "# hello");
  });

  it("write accepts binary content", async () => {
    const a = fileArtifactsCapability(root);
    const payload = new Uint8Array([0, 1, 2, 255]);
    await a.write("run1", "blob.bin", payload);
    assert.deepEqual(await a.read("run1", "blob.bin"), payload);
  });

  it("list() returns recursive relative paths, sorted; [] for an unknown run", async () => {
    const a = fileArtifactsCapability(root);
    await a.write("run1", "b.txt", "b");
    await a.write("run1", "sub/a.txt", "a");
    assert.deepEqual(await a.list("run1"), ["b.txt", join("sub", "a.txt")]);
    assert.deepEqual(await a.list("never-ran"), []);
  });

  it("runs are isolated from each other", async () => {
    const a = fileArtifactsCapability(root);
    await a.write("run1", "x.txt", "one");
    await a.write("run2", "x.txt", "two");
    assert.equal(Buffer.from(await a.read("run2", "x.txt")).toString(), "two");
    assert.deepEqual(await a.list("run1"), ["x.txt"]);
  });

  it("rejects runIds containing path separators or ..", async () => {
    const a = fileArtifactsCapability(root);
    for (const bad of ["", "a/b", "a\\b", "..", "x..y"]) {
      await assert.rejects(() => a.dir(bad), /invalid runId/);
    }
  });

  it("rejects relPaths that escape the run dir", async () => {
    const a = fileArtifactsCapability(root);
    await assert.rejects(
      () => a.write("run1", "../elsewhere.txt", "nope"),
      /escapes the artifact root/,
    );
    await assert.rejects(
      () => a.read("run1", "../../etc/passwd"),
      /escapes the artifact root/,
    );
  });
});
