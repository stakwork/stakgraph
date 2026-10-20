import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import { env, envFrozen, freezeEnv } from "./env.js";

const NAME = "MCP_ENV_TEST_VARIABLE";

describe("env", () => {
  after(() => {
    delete process.env[NAME];
  });

  it("is process.env until frozen: a variable set after import is readable, and a spread carries it", () => {
    assert.equal(envFrozen(), false);
    process.env[NAME] = "live";
    assert.equal(env[NAME], "live");
    assert.ok(NAME in env);
    assert.equal({ ...env }[NAME], "live");
    assert.ok(Object.keys(env).includes(NAME));
    delete process.env[NAME];
    assert.equal(env[NAME], undefined);
    assert.equal(NAME in env, false);
  });

  it("after freezeEnv, reads the copy: a variable taken out of process.env is still there, one added after is not", () => {
    process.env[NAME] = "frozen";
    assert.equal(freezeEnv(), true);
    assert.equal(envFrozen(), true);
    assert.equal(freezeEnv(), false, "idempotent");
    delete process.env[NAME];
    assert.equal(env[NAME], "frozen");
    assert.equal({ ...env }[NAME], "frozen");
    process.env[`${NAME}_LATER`] = "late";
    assert.equal(env[`${NAME}_LATER`], undefined);
    delete process.env[`${NAME}_LATER`];
    // A write goes to the copy, never to process.env.
    env[`${NAME}_WRITE`] = "w";
    assert.equal(env[`${NAME}_WRITE`], "w");
    assert.equal(process.env[`${NAME}_WRITE`], undefined);
    delete env[`${NAME}_WRITE`];
    assert.equal(env[`${NAME}_WRITE`], undefined);
  });
});
