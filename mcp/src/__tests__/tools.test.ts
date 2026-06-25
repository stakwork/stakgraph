import { test, expect } from '@playwright/test';
import { get_tools } from "../repo/tools.js";
import path from 'path';
import fs from 'fs';
import os from 'os';
import { execSync } from 'child_process';

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

test.describe("editing tools", () => {
  let testRepo: string;

  test.beforeEach(async () => {
    testRepo = await createTestRepo();
  });

  test.afterEach(async () => {
    await cleanupTestRepo(testRepo);
  });

  test.describe("str_replace_based_edit_tool (openai provider)", () => {
    test("should NOT be present when toolsConfig is undefined", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        undefined, // no toolsConfig
        "openai"
      );
      expect(tools.str_replace_based_edit_tool).toBeUndefined();
    });

    test("should be defined when toolsConfig.str_replace_based_edit_tool is true", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { str_replace_based_edit_tool: true },
        "openai"
      );
      expect(tools.str_replace_based_edit_tool).toBeDefined();
      expect(typeof tools.str_replace_based_edit_tool.execute).toBe("function");
    });

    test("execute view returns file contents", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { str_replace_based_edit_tool: true },
        "openai"
      );
      const result = await tools.str_replace_based_edit_tool.execute({
        command: "view",
        path: "test.txt",
      });
      expect(result).toContain("test content");
    });

    test("execute str_replace modifies file on disk", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { str_replace_based_edit_tool: true },
        "openai"
      );
      const replaceResult = await tools.str_replace_based_edit_tool.execute({
        command: "str_replace",
        path: "test.txt",
        old_str: "test content",
        new_str: "replaced content",
      });
      expect(replaceResult).toContain("Successfully replaced");

      // Confirm change on disk via view
      const viewResult = await tools.str_replace_based_edit_tool.execute({
        command: "view",
        path: "test.txt",
      });
      expect(viewResult).toContain("replaced content");
      expect(viewResult).not.toContain("test content");
    });

    test("execute create creates new file on disk", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { str_replace_based_edit_tool: true },
        "openai"
      );
      const result = await tools.str_replace_based_edit_tool.execute({
        command: "create",
        path: "newfile.txt",
        file_text: "hello",
      });
      expect(result).toContain("Successfully created");
      expect(fs.existsSync(path.join(testRepo, "newfile.txt"))).toBe(true);
    });

    test("str_replace with non-matching old_str returns error string (not a throw)", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { str_replace_based_edit_tool: true },
        "openai"
      );
      const result = await tools.str_replace_based_edit_tool.execute({
        command: "str_replace",
        path: "test.txt",
        old_str: "this text does not exist in the file",
        new_str: "replacement",
      });
      expect(typeof result).toBe("string");
      expect(result).toMatch(/Error/);
    });

    test("path traversal attempt returns error string (not a throw)", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { str_replace_based_edit_tool: true },
        "openai"
      );
      const result = await tools.str_replace_based_edit_tool.execute({
        command: "view",
        path: "../../etc/passwd",
      });
      expect(typeof result).toBe("string");
      expect(result).toMatch(/Error/);
    });
  });

  test.describe("apply_patch", () => {
    test("should NOT be present when toolsConfig is undefined", async () => {
      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        undefined, // no toolsConfig
        "openai"
      );
      expect(tools.apply_patch).toBeUndefined();
    });

    test("should be defined when toolsConfig.apply_patch is true", async () => {
      // Initialize git repo so `git apply` works
      await new Promise<void>((resolve, reject) => {
        try {
          execSync("git init && git add . && git commit -m init", {
            cwd: testRepo,
            stdio: "pipe",
            env: {
              ...process.env,
              GIT_AUTHOR_NAME: "test",
              GIT_AUTHOR_EMAIL: "test@test.com",
              GIT_COMMITTER_NAME: "test",
              GIT_COMMITTER_EMAIL: "test@test.com",
            },
          });
          resolve();
        } catch (e) {
          reject(e);
        }
      });

      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { apply_patch: true },
        "openai"
      );
      expect(tools.apply_patch).toBeDefined();
      expect(typeof tools.apply_patch.execute).toBe("function");
    });

    test("execute with valid unified diff modifies file on disk", async () => {
      // Initialize git repo
      execSync("git init && git add . && git commit -m init", {
        cwd: testRepo,
        stdio: "pipe",
        env: {
          ...process.env,
          GIT_AUTHOR_NAME: "test",
          GIT_AUTHOR_EMAIL: "test@test.com",
          GIT_COMMITTER_NAME: "test",
          GIT_COMMITTER_EMAIL: "test@test.com",
        },
      });

      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { apply_patch: true },
        "openai"
      );

      const patch = `--- a/test.txt\n+++ b/test.txt\n@@ -1 +1 @@\n-test content\n+patched content\n`;
      const result = await tools.apply_patch.execute({ patch });
      // git apply succeeds silently (empty stdout on success)
      expect(typeof result).toBe("string");
      // File should be modified
      const content = fs.readFileSync(path.join(testRepo, "test.txt"), "utf-8");
      expect(content).toContain("patched content");
    });

    test("execute with invalid patch returns error string (not a throw)", async () => {
      // Initialize git repo
      execSync("git init && git add . && git commit -m init", {
        cwd: testRepo,
        stdio: "pipe",
        env: {
          ...process.env,
          GIT_AUTHOR_NAME: "test",
          GIT_AUTHOR_EMAIL: "test@test.com",
          GIT_COMMITTER_NAME: "test",
          GIT_COMMITTER_EMAIL: "test@test.com",
        },
      });

      const tools = await get_tools(
        testRepo,
        "",
        undefined,
        { apply_patch: true },
        "openai"
      );

      const result = await tools.apply_patch.execute({ patch: "invalid patch content" });
      expect(typeof result).toBe("string");
      // Should contain an error indication
      expect(result.toLowerCase()).toMatch(/error|failed|apply_patch/);
    });
  });
});
