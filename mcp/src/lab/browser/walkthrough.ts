/**
 * A walkthrough recording (strut plans/live-artifacts.md §4): a DIRECTORY
 * of frames plus one manifest naming them in order —
 *
 *   walkthroughs/004/
 *     walkthrough.json
 *     01.png  02.png  03.png
 *
 * — under the shot sink's root (the job's directory on a job run, else the
 * run's artifacts), a new directory per recording, never overwritten: a
 * claim's evidence from turn 3 keeps showing turn 3's frames. The manifest
 * is rewritten after every frame (status `recording`) and once at the end
 * (`done` or `stopped`), and every write is one `job.update` to a host that
 * listens, so a person watching sees the frames arrive. Strut's viewer and
 * hive's card read the manifest by its name (`walkthrough.json`) and scrub.
 */
import { mkdir, readdir, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { WALKTHROUGH_MANIFEST, jobFileUrl } from "strut";
import { modelCopy, type ShotSink } from "./shot.js";

const ROOT = "walkthroughs";

/** One frame as the manifest lists it. */
export interface WalkthroughFrame {
  /** `01.png`, relative to the recording's directory. */
  file: string;
  caption: string;
  /** Where the page was when the frame was taken. */
  pageUrl: string;
  /** Since the page was opened. */
  ms: number;
  /** The step the frame follows (absent on the opening frame). */
  step?: Record<string, unknown>;
  /** Set on the frame where the recording stopped: why. */
  failed?: string;
}

export interface WalkthroughManifest {
  title?: string;
  url: string;
  status: "recording" | "done" | "stopped";
  frames: WalkthroughFrame[];
  failed: string[];
}

/** The next recording's number: one past the highest directory already there. */
function nextNumber(names: string[]): string {
  const taken = names.map((n) => Number.parseInt(n, 10)).filter((n) => Number.isFinite(n));
  return String((taken.length ? Math.max(...taken) : 0) + 1).padStart(3, "0");
}

export class Recording {
  /** `walkthroughs/004`, relative to the sink's root. */
  readonly dir: string;
  /** `walkthroughs/004/walkthrough.json`. */
  readonly path: string;
  /** The manifest's served link. */
  readonly url: string;
  readonly manifest: WalkthroughManifest;
  private readonly started = Date.now();

  private constructor(
    private readonly sink: ShotSink,
    dir: string,
    meta: { title?: string; url: string },
  ) {
    this.dir = dir;
    this.path = `${dir}/${WALKTHROUGH_MANIFEST}`;
    this.url = sink.job ? jobFileUrl(sink.job.name, this.path) : `/artifacts/${sink.runId}/${this.path}`;
    this.manifest = { ...(meta.title ? { title: meta.title } : {}), url: meta.url, status: "recording", frames: [], failed: [] };
  }

  /** A new recording in the next free directory. Nothing is written until the first frame. */
  static async start(sink: ShotSink, meta: { title?: string; url: string }): Promise<Recording> {
    let names: string[];
    if (sink.job) {
      try {
        names = await readdir(join(sink.job.root, ROOT));
      } catch (err) {
        if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") throw err;
        names = [];
      }
    } else {
      names = (await sink.artifacts.list(sink.runId)).filter((p) => p.startsWith(`${ROOT}/`)).map((p) => p.split("/")[1]!);
    }
    return new Recording(sink, `${ROOT}/${nextNumber(names)}`, meta);
  }

  /** Write a frame, then the manifest that names it. Returns the frame's
   *  served link and the model's copy. */
  async addFrame(png: Buffer, frame: Omit<WalkthroughFrame, "file" | "ms">): Promise<{ file: string; url: string; small: Buffer }> {
    const file = `${String(this.manifest.frames.length + 1).padStart(2, "0")}.png`;
    await this.write(file, png);
    this.manifest.frames.push({ file, ...frame, ms: Date.now() - this.started });
    await this.save();
    const rel = `${this.dir}/${file}`;
    const url = this.sink.job ? jobFileUrl(this.sink.job.name, rel) : `/artifacts/${this.sink.runId}/${rel}`;
    return { file, url, small: (await modelCopy(png)).small };
  }

  /** The end: `done`, or `stopped` with what failed — marked on the last
   *  frame too, so the viewer shows the reason under the frame it is about. */
  async finish(failed: string[]): Promise<void> {
    this.manifest.status = failed.length ? "stopped" : "done";
    this.manifest.failed = failed;
    const last = this.manifest.frames[this.manifest.frames.length - 1];
    if (last && failed.length) last.failed = failed.join("; ");
    await this.save();
  }

  /** The manifest, whole — through a temp file and a rename on a job's
   *  directory, so a reader polling it never sees half of one. */
  private async save(): Promise<void> {
    const text = JSON.stringify(this.manifest, null, 2);
    if (this.sink.job) {
      const abs = join(this.sink.job.root, this.path);
      await writeFile(`${abs}.tmp`, text);
      await rename(`${abs}.tmp`, abs);
    } else {
      await this.sink.artifacts.write(this.sink.runId, this.path, text);
    }
    this.sink.notify?.({
      kind: "walkthrough",
      url: this.url,
      state: this.manifest.status,
      ...(this.manifest.title ? { title: this.manifest.title } : {}),
      detail: { frames: this.manifest.frames.length },
    });
  }

  private async write(file: string, bytes: Buffer): Promise<void> {
    if (this.sink.job) {
      const dir = join(this.sink.job.root, this.dir);
      await mkdir(dir, { recursive: true });
      await writeFile(join(dir, file), bytes);
    } else {
      await this.sink.artifacts.write(this.sink.runId, `${this.dir}/${file}`, bytes);
    }
  }
}
