#!/usr/bin/env node

import fs from "fs";
import path from "path";
import clipboard from "clipboardy";

// Parse command line arguments
const args = process.argv.slice(2);
const jsonFlag = args.includes("--json");

// Extract ignore patterns
const ignorePatterns = [];
for (let i = 0; i < args.length - 1; i++) {
  if (args[i] === "--ignore") {
    ignorePatterns.push(args[i + 1]);
  }
}

// Extract extra gitignore path
let extraGitignorePath = null;
for (let i = 0; i < args.length - 1; i++) {
  if (args[i] === "--gitignore") {
    extraGitignorePath = args[i + 1];
  }
}

// Extract the directory path (first non-flag argument)
const dirPath = args.filter(
  (arg) =>
    !arg.startsWith("--") &&
    !ignorePatterns.includes(arg) &&
    arg !== extraGitignorePath
)[0];

// Check if path was provided
if (!dirPath) {
  console.error("Please provide a directory path as an argument");
  process.exit(1);
}

// Check if path exists and is a directory
if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
  console.error(`Error: ${dirPath} is not a valid directory`);
  process.exit(1);
}

// Default directories to skip
const dirsToSkip = ["node_modules", "vendor"];

// Cache for .gitignore patterns to avoid re-reading files
const gitignoreCache = new Map();
// Store extra gitignore patterns separately
const extraGitignorePatterns = [];

// Function to parse gitignore content into patterns
function parseGitignoreContent(content) {
  const patterns = [];
  const lines = content.split("\n");

  for (let line of lines) {
    // Remove comments and trim whitespace
    line = line.split("#")[0].trim();

    // Skip empty lines and negation patterns (for simplicity)
    if (!line || line.startsWith("!")) continue;

    patterns.push(line);
  }

  return patterns;
}

// Load extra gitignore if specified
if (extraGitignorePath) {
  try {
    const absGitignorePath = path.resolve(extraGitignorePath);
    let gitignoreFilePath;

    if (fs.existsSync(absGitignorePath)) {
      if (fs.statSync(absGitignorePath).isDirectory()) {
        // If it's a directory, look for .gitignore inside it
        gitignoreFilePath = path.join(absGitignorePath, ".gitignore");
      } else {
        // If it's a file, use it directly
        gitignoreFilePath = absGitignorePath;
      }

      if (fs.existsSync(gitignoreFilePath)) {
        const content = fs.readFileSync(gitignoreFilePath, "utf8");
        const patterns = parseGitignoreContent(content);
        extraGitignorePatterns.push(...patterns);
      } else {
        console.error(`No .gitignore found at ${gitignoreFilePath}`);
      }
    } else {
      console.error(
        `Extra gitignore path does not exist: ${extraGitignorePath}`
      );
    }
  } catch (err) {
    console.error(`Error processing extra gitignore: ${err.message}`);
  }
}

// Function to read .gitignore patterns from a directory
function getGitignorePatterns(dir) {
  if (gitignoreCache.has(dir)) {
    return gitignoreCache.get(dir);
  }

  const patterns = [];
  const gitignorePath = path.join(dir, ".gitignore");

  if (fs.existsSync(gitignorePath)) {
    try {
      const content = fs.readFileSync(gitignorePath, "utf8");
      patterns.push(...parseGitignoreContent(content));
    } catch (err) {
      console.error(`Error reading .gitignore in ${dir}: ${err.message}`);
    }
  }

  gitignoreCache.set(dir, patterns);
  return patterns;
}

// Function to match a path against a gitignore pattern
function matchPattern(pattern, filePath, fileName, basePath) {
  // Simple exact match for the filename
  if (!pattern.includes("/") && pattern === fileName) {
    return true;
  }

  // Get path relative to the base directory
  const relativePath = path.relative(basePath, filePath);

  // Simple exact match
  if (pattern === relativePath) return true;

  // Directory pattern (ends with /)
  if (pattern.endsWith("/")) {
    const dirPattern = pattern.slice(0, -1);
    if (
      relativePath === dirPattern ||
      relativePath.startsWith(dirPattern + path.sep) ||
      relativePath.startsWith(dirPattern + "/")
    ) {
      return true;
    }
  }

  // Basic wildcard pattern matching
  if (pattern.includes("*")) {
    try {
      const regexPattern = pattern.replace(/\./g, "\\.").replace(/\*/g, ".*");

      const regex = new RegExp(`^${regexPattern}$`);
      if (regex.test(relativePath) || regex.test(fileName)) {
        return true;
      }
    } catch (e) {
      // If regex fails, continue with other patterns
    }
  }

  return false;
}

// Function to check if a path should be ignored
function shouldIgnore(filePath) {
  const relativePath = path.relative(dirPath, filePath);
  const fileName = path.basename(filePath);

  // Check if file name or any parent directory matches an ignore pattern
  for (const pattern of ignorePatterns) {
    // Check for exact filename match
    if (fileName === pattern) return true;

    // Check if the relative path starts with or equals the pattern
    if (
      relativePath === pattern ||
      relativePath.startsWith(pattern + path.sep) ||
      relativePath.startsWith(pattern + "/")
    ) {
      return true;
    }
  }

  // First check extra gitignore patterns (applied to all files)
  const resolvedDirPath = path.resolve(dirPath);
  for (const pattern of extraGitignorePatterns) {
    if (matchPattern(pattern, filePath, fileName, resolvedDirPath)) {
      return true;
    }
  }

  // Check .gitignore patterns in all parent directories
  let currentDir = path.dirname(filePath);
  const rootDir = path.resolve(dirPath);

  // Check all parent directories up to the root directory
  while (currentDir && currentDir.startsWith(rootDir)) {
    const gitignorePatterns = getGitignorePatterns(currentDir);

    for (const pattern of gitignorePatterns) {
      if (matchPattern(pattern, filePath, fileName, currentDir)) {
        return true;
      }
    }

    // Stop if we've reached the root directory
    if (currentDir === rootDir) break;

    // Move up to parent directory
    currentDir = path.dirname(currentDir);
  }

  // Also check the default dirs to skip
  const pathParts = relativePath.split(path.sep);
  return pathParts.some((part) => dirsToSkip.includes(part));
}

// Function to recursively walk a directory
function walkDirectory(dir) {
  let files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    // Skip file_contents.txt to avoid including it in the output
    if (entry.name === "file_contents.txt") continue;

    // Skip files that match ignore patterns
    if (shouldIgnore(fullPath)) continue;

    if (entry.isDirectory()) {
      // Recursively walk subdirectories
      files = files.concat(walkDirectory(fullPath));
    } else {
      files.push(fullPath);
    }
  }

  return files;
}

try {
  // Get all files in the directory and subdirectories
  const allFiles = walkDirectory(dirPath);
  let output = "";
  let jsonOutput = {}; // Changed from array to object

  // Process each file
  for (const file of allFiles) {
    try {
      const content = fs.readFileSync(file, "utf8");

      if (jsonFlag) {
        // Use the file path as key and content as value
        jsonOutput[file] = content;
      } else {
        output += `${file}\n\`\`\`\n${content}\n\`\`\`\n\n`;
      }
    } catch (err) {
      console.error(`Error reading file ${file}: ${err.message}`);

      if (jsonFlag) {
        // For files with errors, include error message in the value
        jsonOutput[file] = `// Error reading file: ${err.message}`;
      } else {
        output += `${file}\n\`\`\`\n// Error reading file: ${err.message}\n\`\`\`\n\n`;
      }
    }
  }

  // Log the ignore patterns that were applied
  if (ignorePatterns.length > 0) {
    console.log(`Ignored patterns: ${ignorePatterns.join(", ")}`);
  }
  if (extraGitignorePatterns.length > 0) {
    console.log(`Extra gitignore patterns: ${extraGitignorePatterns.length}`);
  }

  // Write the output to the clipboard
  if (jsonFlag) {
    clipboard.write(JSON.stringify(jsonOutput, null, 2));
    console.log("Copied JSON to clipboard");
    console.log(`Total files copied: ${Object.keys(jsonOutput).length}`);
  } else {
    clipboard.write(output);
    const lineCount = output.split("\n").length;
    console.log("Copied to clipboard");
    console.log(`Total files copied: ${allFiles.length}`);
    console.log(`Total lines copied: ${lineCount}`);
  }
} catch (err) {
  console.error(`Error: ${err.message}`);
  process.exit(1);
}
