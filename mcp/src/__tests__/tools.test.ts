import { test, expect } from '@playwright/test';
import { get_tools } from "../repo/tools.js";
import path from 'path';
import fs from 'fs';
import os from 'os';

// Helper to create a temporary test repository
async function createTestRepo(): Promise<string> {
  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'bash-tool-test-'));
  await fs.promises.writeFile(path.join(tmpDir, 'test.txt'), 'test content\n');
  return tmpDir;
}

// Helper to cleanup test repository
async function cleanupTestRepo(repoPath: string) {
  try {
    await fs.promises.rm(repoPath, { recursive: true, force: true });
  } catch (e) {
    // Ignore cleanup errors
  }
}

test.describe("bash tool registration", () => {
  let testRepo: string;

  test.beforeEach(async () => {
    testRepo = await createTestRepo();
  });

  test.afterEach(async () => {
    await cleanupTestRepo(testRepo);
  });

  test.describe("non-Anthropic providers", () => {
    test("should register bash tool for openai provider", async () => {
      const tools = await get_tools(
        testRepo,
        "", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "openai"
      );

      expect(tools.bash).toBeDefined();
      expect(tools.bash).toHaveProperty("execute");
      expect(typeof tools.bash.execute).toBe("function");
    });

    test("should execute bash command successfully", async () => {
      const tools = await get_tools(
        testRepo,
        "", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "openai"
      );

      const result = await tools.bash.execute({ command: "echo hello" });

      expect(result).toContain("hello");
    });

    test("should handle failing command gracefully", async () => {
      const tools = await get_tools(
        testRepo,
        "", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "openai"
      );

      const result = await tools.bash.execute({ command: "nonexistent_command_xyz_123" });

      expect(typeof result).toBe("string");
      expect(result).toMatch(/Error executing command:/);
    });

    test("should execute commands in the correct repo directory", async () => {
      const tools = await get_tools(
        testRepo,
        "", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "openai"
      );

      const result = await tools.bash.execute({ command: "cat test.txt" });

      expect(result).toContain("test content");
    });

    test("should be available for deepseek provider", async () => {
      const tools = await get_tools(
        testRepo,
        "", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "deepseek"
      );

      expect(tools.bash).toBeDefined();
      expect(tools.bash).toHaveProperty("execute");
    });

    test("should be available for google provider", async () => {
      const tools = await get_tools(
        testRepo,
        "", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "google"
      );

      expect(tools.bash).toBeDefined();
      expect(tools.bash).toHaveProperty("execute");
    });
  });

  test.describe("Anthropic provider", () => {
    test("should register bash tool for anthropic provider", async () => {
      const tools = await get_tools(
        testRepo,
        "test-key", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "anthropic"
      );

      expect(tools.bash).toBeDefined();
      expect(tools.bash).toHaveProperty("execute");
      expect(typeof tools.bash.execute).toBe("function");
    });

    test("should execute bash command for anthropic provider", async () => {
      const tools = await get_tools(
        testRepo,
        "test-key", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "anthropic"
      );

      const result = await tools.bash.execute({ command: "echo anthropic" });

      expect(result).toContain("anthropic");
    });

    test("should handle errors for anthropic provider", async () => {
      const tools = await get_tools(
        testRepo,
        "test-key", // apiKey
        undefined, // pat
        undefined, // toolsConfig
        "anthropic"
      );

      const result = await tools.bash.execute({ command: "nonexistent_command_xyz_123" });

      expect(typeof result).toBe("string");
      expect(result).toMatch(/Error executing command:/);
    });
  });

  test.describe("bash tool presence across all providers", () => {
    test("should be available for all provider types", async () => {
      const providers = ["openai", "anthropic", "google", "deepseek"];

      for (const provider of providers) {
        const tools = await get_tools(
          testRepo,
          provider === "anthropic" ? "test-key" : "", // apiKey
          undefined, // pat
          undefined, // toolsConfig
          provider as any
        );

        expect(tools.bash, `bash tool should be defined for ${provider}`).toBeDefined();
        expect(tools.bash.execute, `bash.execute should be a function for ${provider}`).toBeInstanceOf(Function);
        
        // Verify it actually works
        const result = await tools.bash.execute({ command: "echo test" });
        expect(result, `bash tool should execute for ${provider}`).toContain("test");
      }
    });
  });
});
