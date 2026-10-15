/**
 * Recording (walkthrough.ts): the directory and its manifest — numbered,
 * never overwritten, rewritten after every frame and told to the job's host
 * each time — on a job's directory and under a run's artifacts.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileArtifactsCapability, jobRoot } from "strut";
import { Recording } from "./walkthrough.js";

async function frame(color = "#336699"): Promise<Buffer> {
  const { default: sharp } = await import("sharp");
  return sharp({ create: { width: 40, height: 20, channels: 3, background: color } }).png().toBuffer();
}

test("on a job: walkthroughs/001/ with numbered frames and walkthrough.json, rewritten and told after every frame; the next recording is 002", async () => {
  const dataDir = await mkdtemp(join(tmpdir(), "lab-walk-job-"));
  try {
    const root = jobRoot(dataDir, "j-1");
    const told: Array<Record<string, unknown>> = [];
    const sink = { artifacts: fileArtifactsCapability(join(dataDir, "artifacts")), runId: "r1", job: { name: "j-1", root }, notify: (u: Record<string, unknown>) => void told.push(u) };
    const rec = await Recording.start(sink, { title: "Switch billing", url: "https://pod/settings" });
    assert.deepEqual([rec.dir, rec.path, rec.url], ["walkthroughs/001", "walkthroughs/001/walkthrough.json", "/jobs/j-1/files/walkthroughs/001/walkthrough.json"]);

    const one = await rec.addFrame(await frame(), { caption: "Open settings", pageUrl: "https://pod/settings" });
    assert.deepEqual([one.file, one.url], ["01.png", "/jobs/j-1/files/walkthroughs/001/01.png"]);
    assert.ok(one.small.length > 0);
    await rec.addFrame(await frame("#993333"), { caption: "Billing", pageUrl: "https://pod/settings/billing", step: { do: "click", selector: "text=Billing" } });
    await rec.finish([]);

    assert.deepEqual((await readdir(join(root, "walkthroughs", "001"))).sort(), ["01.png", "02.png", "walkthrough.json"], "no temp file left behind");
    const m = JSON.parse(await readFile(join(root, rec.path), "utf8"));
    assert.deepEqual(
      { ...m, frames: m.frames.map((f: { ms: number }) => ({ ...f, ms: typeof f.ms })) },
      {
        title: "Switch billing",
        url: "https://pod/settings",
        status: "done",
        frames: [
          { file: "01.png", caption: "Open settings", pageUrl: "https://pod/settings", ms: "number" },
          { file: "02.png", caption: "Billing", pageUrl: "https://pod/settings/billing", step: { do: "click", selector: "text=Billing" }, ms: "number" },
        ],
        failed: [],
      },
    );
    assert.deepEqual(
      told.map((u) => [u.kind, u.url, u.state, u.title, (u.detail as { frames: number }).frames]),
      [
        ["walkthrough", rec.url, "recording", "Switch billing", 1],
        ["walkthrough", rec.url, "recording", "Switch billing", 2],
        ["walkthrough", rec.url, "done", "Switch billing", 2],
      ],
    );
    assert.equal((await Recording.start(sink, { url: "https://pod/" })).dir, "walkthroughs/002", "a new directory per recording, never overwritten");
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

test("under a run's artifacts: the run's link; a stop is `stopped`, its reasons on the manifest and on the last frame", async () => {
  const dir = await mkdtemp(join(tmpdir(), "lab-walk-run-"));
  try {
    const artifacts = fileArtifactsCapability(dir);
    const sink = { artifacts, runId: "r9" };
    const rec = await Recording.start(sink, { url: "https://x/" });
    assert.equal(rec.url, "/artifacts/r9/walkthroughs/001/walkthrough.json");
    await rec.addFrame(await frame(), { caption: "Home", pageUrl: "https://x/" });
    await rec.addFrame(await frame(), { caption: "Stopped at step 1", pageUrl: "https://x/" });
    await rec.finish(["step 1 (click text=Nope) failed: matched nothing"]);
    const m = JSON.parse(Buffer.from(await artifacts.read("r9", rec.path)).toString());
    assert.equal(m.status, "stopped");
    assert.deepEqual(m.failed, ["step 1 (click text=Nope) failed: matched nothing"]);
    assert.deepEqual(m.frames.map((f: { failed?: string }) => f.failed), [undefined, "step 1 (click text=Nope) failed: matched nothing"]);
    assert.deepEqual(await artifacts.list("r9"), ["walkthroughs/001/01.png", "walkthroughs/001/02.png", "walkthroughs/001/walkthrough.json"]);
    assert.equal((await Recording.start(sink, { url: "https://x/" })).dir, "walkthroughs/002");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
