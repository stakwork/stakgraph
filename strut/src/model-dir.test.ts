import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { homedir } from "node:os";
import { modelDirFromEnv, LEGACY_MODEL_DIRS } from "./model-dir.js";

describe("modelDirFromEnv", () => {
  it("prefers STRUT_MODEL_DIR, then the STRUT_MODEL_CACHE alias", () => {
    assert.equal(modelDirFromEnv({ STRUT_MODEL_DIR: "/m", STRUT_MODEL_CACHE: "/c", STRUT_CACHE_DIR: "/r" }), "/m");
    assert.equal(modelDirFromEnv({ STRUT_MODEL_CACHE: "/c", STRUT_CACHE_DIR: "/r" }), "/c");
  });
  it("falls back to <STRUT_CACHE_DIR>/strut/models, then XDG_CACHE_HOME", () => {
    assert.equal(modelDirFromEnv({ STRUT_CACHE_DIR: "/srv/cache" }), join("/srv/cache", "strut", "models"));
    assert.equal(modelDirFromEnv({ XDG_CACHE_HOME: "/xdg" }), join("/xdg", "strut", "models"));
    assert.equal(modelDirFromEnv({ STRUT_CACHE_DIR: "/r", XDG_CACHE_HOME: "/xdg" }), join("/r", "strut", "models"));
  });
  it("with nothing set uses ~/.cache/strut/models (or the legacy dir if only that exists)", () => {
    const got = modelDirFromEnv({});
    assert.ok(got === join(homedir(), ".cache", "strut", "models") || LEGACY_MODEL_DIRS.includes(got), got);
  });
});
