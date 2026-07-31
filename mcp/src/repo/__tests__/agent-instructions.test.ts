import { test, expect } from "../../testkit.js";
import { prepareAgent } from "../agent.js";
import fs from "fs";
import os from "os";
import path from "path";

// `settings` is private on ToolLoopAgent but present at runtime.
function agentInstructions(prepared: unknown): unknown {
  return ((prepared as { agent: { settings: { instructions?: unknown } } }).agent
    .settings).instructions;
}

test.describe("agent instructions wiring", () => {
  let repo: string;

  test.beforeEach(async () => {
    repo = await fs.promises.mkdtemp(path.join(os.tmpdir(), "agent-instructions-"));
    await fs.promises.writeFile(path.join(repo, "a.txt"), "hello\n");
  });

  test.afterEach(async () => {
    await fs.promises.rm(repo, { recursive: true, force: true });
  });

  const OPTS = { apiKey: "test-key-unused" };

  test("passes the assembled system prompt to the agent", async () => {
    const prepared = await prepareAgent("what is this repo?", repo, {
      ...OPTS,
      systemOverride: "SENTINEL_SYSTEM_PROMPT",
    } as never);

    const instructions = agentInstructions(prepared);
    expect(typeof instructions).toBe("string");
    expect(instructions as string).toContain("SENTINEL_SYSTEM_PROMPT");
  });

  test("builds a non-empty default prompt when no override is given", async () => {
    const prepared = await prepareAgent("what is this repo?", repo, OPTS as never);

    const instructions = agentInstructions(prepared);
    expect(typeof instructions).toBe("string");
    expect((instructions as string).length).toBeGreaterThan(0);
  });

  test("transparent replay sends no generated system prompt", async () => {
    const prepared = await prepareAgent(
      [{ role: "user", content: "hi" }] as never,
      repo,
      { ...OPTS, transparent: true } as never,
    );

    expect(agentInstructions(prepared)).toBeUndefined();
  });
});
