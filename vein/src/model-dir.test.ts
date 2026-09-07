import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { homedir } from "node:os";
import { modelDirFromEnv, LEGACY_MODEL_DIR } from "./model-dir.js";

describe("modelDirFromEnv", () => {
  it("prefers VEIN_MODEL_DIR, then the VEIN_MODEL_CACHE alias", () => {
    assert.equal(modelDirFromEnv({ VEIN_MODEL_DIR: "/m", VEIN_MODEL_CACHE: "/c", VEIN_CACHE_DIR: "/r" }), "/m");
    assert.equal(modelDirFromEnv({ VEIN_MODEL_CACHE: "/c", VEIN_CACHE_DIR: "/r" }), "/c");
  });
  it("falls back to <VEIN_CACHE_DIR>/vein/models, then XDG_CACHE_HOME", () => {
    assert.equal(modelDirFromEnv({ VEIN_CACHE_DIR: "/srv/cache" }), join("/srv/cache", "vein", "models"));
    assert.equal(modelDirFromEnv({ XDG_CACHE_HOME: "/xdg" }), join("/xdg", "vein", "models"));
    assert.equal(modelDirFromEnv({ VEIN_CACHE_DIR: "/r", XDG_CACHE_HOME: "/xdg" }), join("/r", "vein", "models"));
  });
  it("with nothing set uses ~/.cache/vein/models (or the legacy dir if only that exists)", () => {
    const got = modelDirFromEnv({});
    assert.ok(got === join(homedir(), ".cache", "vein", "models") || got === LEGACY_MODEL_DIR, got);
  });
});
