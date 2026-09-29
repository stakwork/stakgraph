/**
 * A screenshot has two readers: the run record wants it sharp, the model
 * pays for it in pixels on every later turn. So the full-frame PNG goes to
 * the run's artifacts as `shots/NNN.png` and a 0.6× copy comes back as bytes
 * for the step to hand the model.
 *
 * `sharp` is a native module, imported lazily: a host where its binary is
 * missing fails the screenshot, not the boot.
 */
import type { ArtifactsCapability } from "strut";

/** What `browser/screenshot` (and `capture`) return: the artifact, its served
 *  form, and the frame's size. */
export interface Shot {
  /** Relative to the run's artifact dir — what `artifacts.read` and a later
   *  step's `cwd` see. */
  path: string;
  /** The served form, the convention strut's UI renders as an artifact link. */
  url: string;
  width: number;
  height: number;
  bytes: number;
}

/** The model's copy is this fraction of the frame: 1280×800 → 768×480,
 *  about 490 tokens instead of 1,365. */
export const MODEL_SCALE = 0.6;

export async function saveShot(
  artifacts: ArtifactsCapability,
  runId: string,
  png: Buffer,
  name?: string,
): Promise<Shot & { small: Buffer }> {
  const count = (await artifacts.list(runId)).filter((p) => p.startsWith("shots/") && p.endsWith(".png")).length;
  const file = name ? `${name.replace(/\.png$/i, "").replace(/[^\w.-]+/g, "_")}.png` : `${String(count + 1).padStart(3, "0")}.png`;
  const path = `shots/${file}`;
  await artifacts.write(runId, path, png);
  const { default: sharp } = await import("sharp");
  const meta = await sharp(png).metadata();
  const width = meta.width ?? 0;
  const height = meta.height ?? 0;
  const small = await sharp(png)
    .resize({ width: Math.max(1, Math.round(width * MODEL_SCALE)) })
    .png()
    .toBuffer();
  return { path, url: `/artifacts/${runId}/${path}`, width, height, bytes: png.length, small };
}
