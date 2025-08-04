import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";
import { Request, Response } from "express";
import { fileURLToPath } from "url";

const execAsync = promisify(exec);

// curl "http://localhost:3000/test?test=all"

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const getBaseDir = (): string => {
  return path.join(__dirname, "../..");
};

export const getTestsDir = (): string => {
  if (process.env.TESTS_DIR) {
    // Return absolute path for TESTS_DIR
    return path.resolve(process.env.TESTS_DIR);
  } else {
    return path.join(getBaseDir(), "tests/generated_tests");
  }
};

// Run Playwright test
export async function runPlaywrightTest(
  req: Request,
  res: Response
): Promise<void> {
  try {
    const { test } = req.query;
    console.log("===> runPlaywrightTest", test);

    if (!test || typeof test !== "string") {
      res.status(400).json({ error: "Test name is required" });
      return;
    }

    // Validate test parameter to prevent command injection
    const validTestPattern = /^[a-zA-Z0-9_\-\/\*\.]+$/;
    if (!validTestPattern.test(test)) {
      res.status(400).json({ error: "Invalid test name format" });
      return;
    }

    // Check if tests directory exists
    const testsDir = getTestsDir();
    try {
      await fs.access(testsDir);
    } catch {
      res.status(404).json({ error: "Tests directory not found" });
      return;
    }

    // Construct the playwright command
    let testPath: string;
    const baseDir = getBaseDir();
    const configPath = path.join(baseDir, "tests/playwright.config.js");

    if (test === "all") {
      // Run all tests in the TESTS_DIR
      testPath = testsDir;
    } else if (test.includes("*")) {
      // Handle glob patterns - construct full path
      testPath = path.join(testsDir, test);
    } else {
      // If it's a specific test file, ensure it has proper extension and full path
      const testFileName =
        test.endsWith(".spec.js") || test.endsWith(".spec.ts")
          ? test
          : `${test}.spec.js`;
      testPath = path.join(testsDir, testFileName);
    }

    // Use the config file from the base directory but run tests from TESTS_DIR
    const command = `npx playwright test "${testPath}" --config="${configPath}"`;

    console.log("Executing command:", command);
    console.log("Tests directory:", testsDir);
    console.log("Config path:", configPath);

    // Set timeout for the command
    const { stdout, stderr } = await execAsync(command, {
      cwd: baseDir, // Keep cwd as base directory for config resolution
      timeout: 60000,
      env: { ...process.env, CI: "true" }, // Set CI mode for consistent output
    });

    res.json({
      success: true,
      testPath,
      testsDir,
      configPath,
      output: stdout,
      errors: stderr || null,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    // Handle different types of errors
    if (error.code === "ENOENT") {
      res.status(500).json({
        success: false,
        error: "Playwright not found. Make sure it's installed.",
        timestamp: new Date().toISOString(),
      });
    } else if (error.killed && error.signal === "SIGTERM") {
      res.status(408).json({
        success: false,
        error: "Test execution timed out",
        timestamp: new Date().toISOString(),
      });
    } else {
      // Test failures will come through here since playwright exits with non-zero code
      res.json({
        success: false,
        testPath: req.query.test as string,
        testsDir: getTestsDir(),
        output: error.stdout || "",
        errors: error.stderr || error.message,
        exitCode: error.code,
        timestamp: new Date().toISOString(),
      });
    }
  }
}
