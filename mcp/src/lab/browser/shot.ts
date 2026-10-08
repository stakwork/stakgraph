/**
 * A screenshot has two readers: the record wants it sharp, the model pays
 * for it in pixels on every later turn. So the full-frame PNG is written as
 * `shots/NNN.png` and a 0.6× copy comes back as bytes for the step to hand
 * the model.
 *
 * WHERE the PNG goes is the sink's. On a plain run, under the run's
 * artifacts, served at `/artifacts/<runId>/shots/NNN.png`. On a run launched
 * under a JOB (strut plans/jobs.md), in the job's own directory, served at
 * `/jobs/<job>/files/shots/NNN.png`. A job's host (hive's Jamie chat) reads
 * a file off the swarm only when it can tie the link to something it
 * launched — the job, or a run id it was handed — and a child run a job
 * agent starts through meta/run-workflow has a run id the host never saw.
 * A shot under THAT run's artifacts would be unreachable there; a job file
 * is the job's, whichever run wrote it, and stays with the job's files.
 *
 * `sharp` is a native module, imported lazily: a host where its binary is
 * missing fails the screenshot, not the boot.
 */
import { mkdir, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ArtifactsCapability } from "strut";

/** Where a shot is written: the run's artifacts, or the job's directory. */
export interface ShotSink {
  artifacts: ArtifactsCapability;
  runId: string;
  /** Set when the run carries a job: its id and its directory (strut's `jobRoot`). */
  job?: { name: string; root: string };
}

/** What `browser/screenshot` (and `capture`) return: the file, where it is,
 *  its served form, and the frame's size. */
export interface Shot {
  /** `shots/NNN.png`, relative to `dir`. */
  path: string;
  /** The absolute directory `path` is relative to: the run's artifact dir, or the job's. */
  dir: string;
  /** The served form — the link strut's UI and a job's host render as an artifact. */
  url: string;
  width: number;
  height: number;
  bytes: number;
}

/** The model's copy is this fraction of the frame: 1280×800 → 768×480,
 *  about 490 tokens instead of 1,365. */
export const MODEL_SCALE = 0.6;

const SHOTS = "shots";

function fileName(count: number, name?: string): string {
  return name ? `${name.replace(/\.png$/i, "").replace(/[^\w.-]+/g, "_")}.png` : `${String(count + 1).padStart(3, "0")}.png`;
}

/** The PNGs already under `shots/` — the number the next unnamed shot takes. */
async function countArtifactShots(sink: ShotSink): Promise<number> {
  return (await sink.artifacts.list(sink.runId)).filter((p) => p.startsWith(`${SHOTS}/`) && p.endsWith(".png")).length;
}

async function countJobShots(shotsDir: string): Promise<number> {
  try {
    return (await readdir(shotsDir)).filter((f) => f.endsWith(".png")).length;
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code === "ENOENT") return 0;
    throw err;
  }
}

export async function saveShot(sink: ShotSink, png: Buffer, name?: string): Promise<Shot & { small: Buffer }> {
  let path: string;
  let dir: string;
  let url: string;
  if (sink.job) {
    const shotsDir = join(sink.job.root, SHOTS);
    const file = fileName(await countJobShots(shotsDir), name);
    await mkdir(shotsDir, { recursive: true });
    await writeFile(join(shotsDir, file), png);
    path = `${SHOTS}/${file}`;
    dir = sink.job.root;
    url = `/jobs/${encodeURIComponent(sink.job.name)}/files/${path}`;
  } else {
    path = `${SHOTS}/${fileName(await countArtifactShots(sink), name)}`;
    await sink.artifacts.write(sink.runId, path, png);
    dir = await sink.artifacts.dir(sink.runId);
    url = `/artifacts/${sink.runId}/${path}`;
  }
  const { default: sharp } = await import("sharp");
  const meta = await sharp(png).metadata();
  const width = meta.width ?? 0;
  const height = meta.height ?? 0;
  const small = await sharp(png)
    .resize({ width: Math.max(1, Math.round(width * MODEL_SCALE)) })
    .png()
    .toBuffer();
  return { path, dir, url, width, height, bytes: png.length, small };
}
