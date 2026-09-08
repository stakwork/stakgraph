#!/usr/bin/env node
/**
 * Desktop launcher — strut with the defaults a native host (or someone who
 * just downloaded the tarball) wants, with no env to set
 * (plans/local-desktop-and-stt.md §2.5). Staged beside `build/` by
 * `package:desktop`; a host spawns `node desktop.js` and reads one stdout line:
 *
 *   {"event":"ready","port":51234,"host":"127.0.0.1","key":"…"}
 *
 * Defaults (each one an env override): filesystem workspace (no Neo4j),
 * bind 127.0.0.1 on an OS-picked port, workspace under the platform
 * app-support dir, models under the platform cache dir, a random API key
 * per launch. `--open` launches the web UI in the default browser, which
 * has dictation built in — the quickest way to try the recognizer.
 */
import { randomBytes } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { homedir, platform } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
  console.log(`usage: strut [--open]

Runs strut locally with desktop defaults. Prints a JSON ready line on stdout
({"event":"ready","port","host","key"}) and the UI URL on stderr.

  --open   open the web UI in the default browser

Env overrides (all optional): STRUT_WORKSPACE, STRUT_CACHE_DIR, STRUT_API_KEY,
STRUT_HOST, STRUT_PORT, STRUT_WORKSPACE_BACKEND, STRUT_SECRET_KEY,
STRUT_STT_MODEL, STRUT_STT_PARTIAL_MODEL, ANTHROPIC_API_KEY.`);
  process.exit(0);
}

const env = process.env;
const home = homedir();
const os = platform();
const dflt = (k, v) => {
  if (env[k] === undefined || env[k] === "") env[k] = v;
};
const appSupport =
  os === "darwin"
    ? join(home, "Library", "Application Support")
    : os === "win32"
      ? env["APPDATA"] || join(home, "AppData", "Roaming")
      : env["XDG_DATA_HOME"] || join(home, ".local", "share");
const cacheRoot =
  os === "darwin"
    ? join(home, "Library", "Caches")
    : os === "win32"
      ? env["LOCALAPPDATA"] || join(home, "AppData", "Local")
      : env["XDG_CACHE_HOME"] || join(home, ".cache");

dflt("STRUT_WORKSPACE_BACKEND", "fs");
dflt("STRUT_HOST", "127.0.0.1");
dflt("STRUT_PORT", "0");
dflt("STRUT_WORKSPACE", join(appSupport, "strut", "workspace"));
dflt("STRUT_CACHE_DIR", cacheRoot); // models land at <cacheRoot>/strut/models
dflt("STRUT_WEB_DIST", join(here, "web", "dist"));
dflt("STRUT_API_KEY", randomBytes(24).toString("hex"));
env["STRUT_READY_KEY"] = "1"; // put the key on the ready line for the host

await mkdir(env["STRUT_WORKSPACE"], { recursive: true });

const { startServer } = await import("./build/server.js");
const port = await startServer();
const url = `http://${env["STRUT_HOST"]}:${port}/?key=${env["STRUT_API_KEY"]}`;
console.error(`strut ui: ${url}`);
if (args.includes("--open")) {
  const [cmd, cmdArgs] =
    os === "darwin" ? ["open", [url]] : os === "win32" ? ["cmd", ["/c", "start", "", url]] : ["xdg-open", [url]];
  spawn(cmd, cmdArgs, { stdio: "ignore", detached: true }).on("error", () => {}).unref();
}
