// Default vein server — a thin wrapper over createVein() that boots
// vein with its filesystem-backed defaults (FileRunStore, workspace
// loaded from VEIN_WORKSPACE, registry built by scanning steps/).
//
// This file is the canonical "library usage" example: anything you
// see here, your own consumer code can do too. Pass your own
// `services`, your own in-code `registry`, your own store — and mount
// the returned `app` wherever you like (under another Hono router,
// behind Express, or just call `vein.listen(port)`).

import { createVein, type Vein } from "./createVein.js";

let veinInstance: Vein | null = null;

/** Lazily build the default Vein instance. Kept lazy so importing this
 *  module doesn't kick off filesystem I/O at module-load time. */
async function getDefault(): Promise<Vein> {
  if (!veinInstance) {
    veinInstance = await createVein();
  }
  return veinInstance;
}

/**
 * Hono app for the default filesystem-backed vein. Returns the same
 * instance on repeated calls. Most code should call `createVein()`
 * directly and mount `vein.app` — this helper exists for ergonomic
 * scripting and backwards compatibility.
 */
export async function getApp() {
  return (await getDefault()).app;
}

/**
 * Boot the default filesystem-backed vein server on `port` (defaults to
 * `VEIN_PORT` or `3000`). Equivalent to:
 *
 * ```ts
 * const vein = await createVein();
 * await vein.listen(port);
 * ```
 */
export async function startServer(port?: number): Promise<void> {
  const vein = await getDefault();
  await vein.listen(port);
}

// Run directly when invoked as a script.
const isMain =
  process.argv[1]?.endsWith("server.ts") ||
  process.argv[1]?.endsWith("server.js");
if (isMain) {
  startServer();
}
