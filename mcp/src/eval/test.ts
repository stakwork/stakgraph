import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";
import express, { Request, Response, Express } from "express";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const execAsync = promisify(exec);

// curl "http://localhost:3000/test?test=all"

// Configurable base directory
const getBaseDir = (): string => {
  return process.env.PROJECT_ROOT || path.join(__dirname, "../..");
};

const getTestsDir = (): string => {
  return path.join(getBaseDir(), "tests");
};

// Utility function to sanitize filename
const sanitizeFilename = (filename: string): string => {
  return filename.replace(/[^a-zA-Z0-9_\-\.]/g, "_");
};

// Utility function to ensure .spec.js extension
const ensureSpecExtension = (filename: string): string => {
  if (filename.endsWith(".spec.js") || filename.endsWith(".spec.ts")) {
    return filename;
  }
  return `${filename}.spec.js`;
};

// Run Playwright tests
export async function runPlaywrightTests(
  req: Request,
  res: Response
): Promise<void> {
  try {
    const { test } = req.query;

    if (!test || typeof test !== "string") {
      res.status(400).json({ error: "Test parameter is required" });
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
    if (test === "all") {
      testPath = ".";
    } else if (test.includes("*")) {
      testPath = test;
    } else {
      // If it's a specific test file, ensure it has proper extension
      testPath =
        test.endsWith(".spec.js") || test.endsWith(".spec.ts")
          ? `tests/${test}`
          : `tests/${test}.spec.js`;
    }

    const command = `npx playwright test --config=tests/playwright.config.js ${testPath}`;

    // Set timeout for the command
    const { stdout, stderr } = await execAsync(command, {
      cwd: getBaseDir(),
      timeout: 60000,
      env: { ...process.env, CI: "true" }, // Set CI mode for consistent output
    });

    res.json({
      success: true,
      testPath,
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
        output: error.stdout || "",
        errors: error.stderr || error.message,
        exitCode: error.code,
        timestamp: new Date().toISOString(),
      });
    }
  }
}

// Save a test file
export async function saveTest(req: Request, res: Response): Promise<void> {
  try {
    const { name, text } = req.body;

    if (!name || !text) {
      res.status(400).json({ error: "Name and text are required" });
      return;
    }

    if (typeof name !== "string" || typeof text !== "string") {
      res.status(400).json({ error: "Name and text must be strings" });
      return;
    }

    // Sanitize filename and ensure proper extension
    const sanitizedName = sanitizeFilename(name);
    const filename = ensureSpecExtension(sanitizedName);

    // Check if tests directory exists, create if it doesn't
    const testsDir = getTestsDir();
    try {
      await fs.access(testsDir);
    } catch {
      await fs.mkdir(testsDir, { recursive: true });
    }

    const filePath = path.join(testsDir, filename);

    // Write the test file
    await fs.writeFile(filePath, text, "utf8");

    res.json({
      success: true,
      message: "Test saved successfully",
      filename,
      path: filePath,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

// List all test files
export async function listTests(req: Request, res: Response): Promise<void> {
  try {
    const testsDir = getTestsDir();

    // Check if tests directory exists
    try {
      await fs.access(testsDir);
    } catch {
      res.status(404).json({ error: "Tests directory not found" });
      return;
    }

    // Read directory contents
    const files = await fs.readdir(testsDir);

    // Filter for test files
    const testFiles = files.filter(
      (file) => file.endsWith(".spec.js") || file.endsWith(".spec.ts")
    );

    // Get file stats for each test file
    const testsWithInfo = await Promise.all(
      testFiles.map(async (filename) => {
        const filePath = path.join(testsDir, filename);
        const stats = await fs.stat(filePath);
        return {
          name: filename,
          size: stats.size,
          modified: stats.mtime.toISOString(),
          created: stats.birthtime.toISOString(),
        };
      })
    );

    res.json({
      success: true,
      tests: testsWithInfo,
      count: testsWithInfo.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

// Get a specific test file by name
export async function getTestByName(
  req: Request,
  res: Response
): Promise<void> {
  try {
    const { name } = req.params;

    if (!name) {
      res.status(400).json({ error: "Test name is required" });
      return;
    }

    // Sanitize and ensure proper extension
    const sanitizedName = sanitizeFilename(name);
    const filename = ensureSpecExtension(sanitizedName);
    const filePath = path.join(getTestsDir(), filename);

    try {
      // Check if file exists and read it
      const content = await fs.readFile(filePath, "utf8");
      const stats = await fs.stat(filePath);

      res.json({
        success: true,
        name: filename,
        content,
        size: stats.size,
        modified: stats.mtime.toISOString(),
        created: stats.birthtime.toISOString(),
        timestamp: new Date().toISOString(),
      });
    } catch (error: any) {
      if (error.code === "ENOENT") {
        res.status(404).json({
          success: false,
          error: "Test file not found",
          timestamp: new Date().toISOString(),
        });
      } else {
        throw error;
      }
    }
  } catch (error: any) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

// Delete a test file by name
export async function deleteTestByName(
  req: Request,
  res: Response
): Promise<void> {
  try {
    const { name } = req.params;

    if (!name) {
      res.status(400).json({ error: "Test name is required" });
      return;
    }

    // Sanitize and ensure proper extension
    const sanitizedName = sanitizeFilename(name);
    const filename = ensureSpecExtension(sanitizedName);
    const filePath = path.join(getTestsDir(), filename);

    try {
      // Check if file exists first
      await fs.access(filePath);

      // Delete the file
      await fs.unlink(filePath);

      res.json({
        success: true,
        message: "Test file deleted successfully",
        filename,
        timestamp: new Date().toISOString(),
      });
    } catch (error: any) {
      if (error.code === "ENOENT") {
        res.status(404).json({
          success: false,
          error: "Test file not found",
          timestamp: new Date().toISOString(),
        });
      } else {
        throw error;
      }
    }
  } catch (error: any) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

export function test_routes(app: Express) {
  app.get("/test", runPlaywrightTests);
  app.get("/test/list", listTests);
  app.get("/test/get", getTestByName);
  app.get("/test/delete", deleteTestByName);
  app.get("/test/save", saveTest);

  app.get("/tests", (req, res) => {
    res.sendFile(path.join(__dirname, "../../tests/tests.html"));
  });
  const static_files = ["app.js", "style.css", "hooks.js"];
  serveStaticFiles(app, static_files);
}

function serveStaticFiles(
  app: Express,
  files: string[],
  basePath: string = "../../tests"
) {
  files.forEach((file) => {
    app.get(`/tests/${file}`, (req, res) => {
      res.sendFile(path.join(__dirname, basePath, file));
    });
  });
}
