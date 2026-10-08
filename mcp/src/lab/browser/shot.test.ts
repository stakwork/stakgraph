/**
 * saveShot (shot.ts): where the PNG lands and what the step reports — under
 * the run's artifacts on a plain run, in the job's directory on a run
 * launched under a job — and the counter each takes.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileArtifactsCapability, jobRoot } from "strut";
import { saveShot } from "./shot.js";

async function frame(width = 40, height = 20): Promise<Buffer> {
  const { default: sharp } = await import("sharp");
  return sharp({ create: { width, height, channels: 3, background: "#336699" } }).png().toBuffer();
}

test("on a plain run the PNG goes under the run's artifacts, counted per run, named when asked", async () => {
  const root = await mkdtemp(join(tmpdir(), "lab-shot-"));
  try {
    const artifacts = fileArtifactsCapability(root);
    const sink = { artifacts, runId: "r1" };
    const png = await frame();
    const first = await saveShot(sink, png);
    assert.deepEqual(
      { ...first, small: undefined },
      { path: "shots/001.png", dir: join(root, "r1"), url: "/artifacts/r1/shots/001.png", width: 40, height: 20, bytes: png.length, small: undefined },
    );
    assert.equal((await readFile(join(root, "r1", "shots", "001.png"))).length, png.length);
    assert.ok(first.small.length > 0, "the model's copy comes back as bytes");
    assert.equal((await saveShot(sink, png)).path, "shots/002.png");
    assert.equal((await saveShot(sink, png, "Hero shot.png")).path, "shots/Hero_shot.png");
    assert.equal((await saveShot({ artifacts, runId: "r2" }, png)).path, "shots/001.png", "the counter is the run's");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("under a job the PNG goes to the job's directory and the link is the job file's", async () => {
  const dataDir = await mkdtemp(join(tmpdir(), "lab-shot-job-"));
  try {
    const artifacts = fileArtifactsCapability(join(dataDir, "artifacts"));
    const job = "abc/review";
    const root = jobRoot(dataDir, job);
    const sink = { artifacts, runId: "child-run", job: { name: job, root } };
    const png = await frame();
    const first = await saveShot(sink, png);
    assert.deepEqual(
      { ...first, small: undefined },
      { path: "shots/001.png", dir: root, url: "/jobs/abc%2Freview/files/shots/001.png", width: 40, height: 20, bytes: png.length, small: undefined },
    );
    assert.equal((await readFile(join(root, "shots", "001.png"))).length, png.length);
    assert.equal((await saveShot(sink, png)).path, "shots/002.png", "the counter is the job directory's");
    assert.equal((await saveShot(sink, png, "after")).url, "/jobs/abc%2Freview/files/shots/after.png");
    assert.deepEqual(await artifacts.list("child-run"), [], "nothing under the run's own artifacts");
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});
