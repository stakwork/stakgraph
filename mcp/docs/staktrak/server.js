import express from "express";
import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(express.json());
app.use(express.static(__dirname));

// Routes
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.get("/preact/frame.html", (req, res) => {
  console.log("=> frame.html");
  res.sendFile(path.join(__dirname, "preact/frame.html"));
});

app.get("/playwright-generator.js", (req, res) => {
  console.log("=> playwright-generator.js");
  res.sendFile(path.join(__dirname, "playwright-generator.js"));
});

// API route to save the generated test
app.post("/api/save-test", async (req, res) => {
  try {
    const { testCode, filename } = req.body;

    if (!testCode) {
      return res.status(400).json({ error: "Test code is required" });
    }

    // Generate filename if not provided
    const testFilename = filename || `generated-test-${Date.now()}.spec.js`;

    // Ensure the tests directory exists
    const testsDir = path.join(__dirname, "tests");
    try {
      await fs.access(testsDir);
    } catch {
      await fs.mkdir(testsDir, { recursive: true });
    }

    // Write the test file
    const filePath = path.join(testsDir, testFilename);
    await fs.writeFile(filePath, testCode, "utf8");

    res.json({
      success: true,
      message: "Test saved successfully",
      filename: testFilename,
      path: filePath,
    });
  } catch (error) {
    console.error("Error saving test:", error);
    res.status(500).json({ error: "Failed to save test file" });
  }
});

// API route to list saved tests
app.get("/api/tests", async (req, res) => {
  try {
    const testsDir = path.join(__dirname, "tests");

    try {
      const files = await fs.readdir(testsDir);
      const testFiles = files.filter((file) => file.endsWith(".spec.js"));

      const fileDetails = await Promise.all(
        testFiles.map(async (file) => {
          const filePath = path.join(testsDir, file);
          const stats = await fs.stat(filePath);
          return {
            filename: file,
            created: stats.birthtime,
            modified: stats.mtime,
            size: stats.size,
          };
        })
      );

      res.json(fileDetails);
    } catch (error) {
      // Tests directory doesn't exist yet
      res.json([]);
    }
  } catch (error) {
    console.error("Error listing tests:", error);
    res.status(500).json({ error: "Failed to list test files" });
  }
});

// API route to get a specific test file
app.get("/api/tests/:filename", async (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(__dirname, "tests", filename);

    const testCode = await fs.readFile(filePath, "utf8");
    res.json({ filename, testCode });
  } catch (error) {
    console.error("Error reading test file:", error);
    res.status(404).json({ error: "Test file not found" });
  }
});

// API route to delete a test file
app.delete("/api/tests/:filename", async (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(__dirname, "tests", filename);

    await fs.unlink(filePath);
    res.json({ success: true, message: "Test file deleted successfully" });
  } catch (error) {
    console.error("Error deleting test file:", error);
    res.status(404).json({ error: "Test file not found" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
