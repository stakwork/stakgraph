import * as fs from "fs";
import * as path from "path";
import clipboard from "clipboardy";

// Get the directory path from command line argument
const dirPath = process.argv[2];

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

// Function to recursively walk a directory
function walkDirectory(dir: string): string[] {
  let files: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    // Skip file_contents.txt to avoid including it in the output
    if (entry.name === "file_contents.txt") continue;

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

  // Process each file
  for (const file of allFiles) {
    try {
      const content = fs.readFileSync(file, "utf8");
      output += `${file}\n\`\`\`\n${content}\n\`\`\`\n\n`;
    } catch (err: any) {
      console.error(`Error reading file ${file}: ${err.message}`);
      output += `${file}\n\`\`\`\n// Error reading file: ${err.message}\n\`\`\`\n\n`;
    }
  }

  // Write the output to the clipboard
  clipboard.write(output);

  // Count the total number of lines
  const lineCount = output.split("\n").length;

  console.log("Copied to clipboard");
  console.log(`Total lines copied: ${lineCount}`);
} catch (err: any) {
  console.error(`Error: ${err.message}`);
  process.exit(1);
}
