import { getTestsDir, runPlaywrightTest } from "./runPlaywright.js";
import fs from "fs/promises";
import path from "path";
import { Request, Response, Express } from "express";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

// Save a test file
export async function saveTest(req: Request, res: Response): Promise<void> {
  try {
    const { name, text } = req.body;
    console.log("===> saveTest", name);

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
    console.log("===> listTests", testsDir);

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
    const { name } = req.query;
    console.log("===> getTestByName", name);

    if (!name || typeof name !== "string") {
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
    const { name } = req.query;
    console.log("===> deleteTestByName", name);

    if (!name || typeof name !== "string") {
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

export async function generatePlaywrightTest(
  req: Request,
  res: Response
): Promise<void> {
  try {
    const { url, trackingData } = req.body;

    if (!url || !trackingData) {
      res.status(400).json({
        success: false,
        error: "URL and tracking data are required",
      });
      return;
    }

    const playwrightGeneratorPath = path.join(
      __dirname,
      "../../tests/playwright-generator.js"
    );

    try {
      const module = await import(playwrightGeneratorPath);
      const testCode = module.generatePlaywrightTest(url, trackingData);

      res.json({
        success: true,
        testCode,
        timestamp: new Date().toISOString(),
      });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: error.message || "Error generating test code",
        timestamp: new Date().toISOString(),
      });
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
  app.get("/test", runPlaywrightTest);
  app.get("/test/list", listTests);
  app.get("/test/get", getTestByName);
  app.get("/test/delete", deleteTestByName);
  app.post("/test/save", saveTest);

  const base = path.join(__dirname, "../../tests");

  app.get("/tests", (_req, res) => {
    res.sendFile(path.join(base, "tests.html"));
  });
  app.get("/tests/frame/frame.html", (_req, res) => {
    res.sendFile(path.join(base, "frame/frame.html"));
  });

  const static_files = [
    "app.js",
    "style.css",
    "hooks.js",
    "frame/app.js",
    "frame/style.css",
    "staktrak/dist/staktrak.js",
    "staktrak/dist/replay.js",
    "staktrak/dist/playwright-generator.js",
  ];

  serveStaticFiles(app, static_files, base);
}

function serveStaticFiles(app: Express, files: string[], basePath: string) {
  files.forEach((file) => {
    app.get(`/tests/${file}`, (req, res) => {
      res.sendFile(path.join(basePath, file));
    });
  });
}
