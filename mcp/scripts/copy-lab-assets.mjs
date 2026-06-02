// Copy lab runtime assets that `tsc` does not emit into `build/`.
//
// The lab seeds workflow templates and step sources into the vein workspace
// at boot, locating them relative to the compiled module (`import.meta.url`).
// `tsc` only emits `.js`, so without this step the prod build (`node
// build/index.js`) can't find:
//   - `*.yaml` workflow templates
//   - `steps/*.ts` step sources (read as text, published as vein custom steps)
// Run after `tsc` (see the `build` script).

import { cp, mkdir, readdir } from "node:fs/promises";
import { dirname, join, sep } from "node:path";

const SRC = join("src", "lab");
const OUT = join("build", "lab");

async function* walk(dir) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return; // dir doesn't exist
  }
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* walk(p);
    else yield p;
  }
}

function isAsset(path) {
  if (path.endsWith(".yaml") || path.endsWith(".yml")) return true;
  // Step sources are read as text and published as vein custom steps.
  if (path.endsWith(".ts") && !path.endsWith(".test.ts") && path.split(sep).includes("steps")) {
    return true;
  }
  return false;
}

let copied = 0;
for await (const src of walk(SRC)) {
  if (!isAsset(src)) continue;
  const dest = OUT + src.slice(SRC.length);
  await mkdir(dirname(dest), { recursive: true });
  await cp(src, dest);
  copied++;
}
console.log(`[copy-lab-assets] copied ${copied} file(s) into ${OUT}`);
