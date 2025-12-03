import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

// Execute ripgrep with args array directly
function execRipgrepCommandDirect(
  args: string[],
  cwd: string,
  timeoutMs: number = 10000
): Promise<string> {
  return new Promise((resolve, reject) => {
    const process = spawn("rg", args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let resolved = false;

    const timeout = setTimeout(() => {
      if (!resolved) {
        process.kill("SIGKILL");
        resolved = true;
        reject(new Error(`Command timed out after ${timeoutMs}ms`));
      }
    }, timeoutMs);

    process.stdout.on("data", (data) => {
      stdout += data.toString();

      if (stdout.length > 10000) {
        process.kill("SIGKILL");
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          const truncated =
            stdout.substring(0, 10000) +
            "\n\n[... output truncated due to size limit ...]";
          resolve(truncated);
        }
        return;
      }
    });

    process.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    process.on("close", (code) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);

        if (code === 0) {
          if (stdout.length > 10000) {
            const truncated =
              stdout.substring(0, 10000) +
              "\n\n[... output truncated to 10,000 characters ...]";
            resolve(truncated);
          } else {
            resolve(stdout);
          }
        } else if (code === 1) {
          resolve("No matches found");
        } else {
          reject(new Error(`Command failed with code ${code}: ${stderr}`));
        }
      }
    });

    process.on("error", (error) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(error);
      }
    });
  });
}

// Execute any shell command with proper streaming
function execShellCommand(
  command: string,
  cwd: string,
  timeoutMs: number = 10000
): Promise<string> {
  return new Promise((resolve, reject) => {
    const process = spawn(command, {
      cwd,
      shell: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let resolved = false;

    const timeout = setTimeout(() => {
      if (!resolved) {
        process.kill("SIGKILL");
        resolved = true;
        reject(new Error(`Command timed out after ${timeoutMs}ms`));
      }
    }, timeoutMs);

    process.stdout.on("data", (data) => {
      stdout += data.toString();

      if (stdout.length > 10000) {
        process.kill("SIGKILL");
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          const truncated =
            stdout.substring(0, 10000) +
            "\n\n[... output truncated due to size limit ...]";
          resolve(truncated);
        }
        return;
      }
    });

    process.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    process.on("close", (code) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);

        // Truncate if needed
        let output = stdout;
        if (output.length > 10000) {
          output =
            output.substring(0, 10000) +
            "\n\n[... output truncated to 10,000 characters ...]";
        }

        if (code === 0) {
          resolve(output);
        } else if (code === 1 && !stderr) {
          // Exit code 1 with no stderr often means "no matches" (grep, find, etc.)
          resolve(output || "No matches found");
        } else {
          // Include both stdout and stderr in error for debugging
          const errorOutput = stderr || stdout || "Unknown error";
          reject(new Error(`Command failed with code ${code}: ${errorOutput}`));
        }
      }
    });

    process.on("error", (error) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(error);
      }
    });
  });
}

// Get repository map
export async function getRepoMap(repoPath: string): Promise<string> {
  if (!repoPath) {
    return "No repository path provided";
  }

  if (!fs.existsSync(repoPath)) {
    return "Repository not cloned yet";
  }

  try {
    const result = await execShellCommand(
      "git ls-tree -r --name-only HEAD | tree -L 3 --fromfile",
      repoPath
    );
    return result;
  } catch (error: any) {
    return `Error getting repo map: ${error.message}`;
  }
}

// Get file summary by reading first 40 lines
export function getFileSummary(
  filePath: string,
  repoPath: string,
  linesLimit: number
): string {
  if (!repoPath) {
    return "No repository path provided";
  }

  const fullPath = path.join(repoPath, filePath);

  if (!fs.existsSync(fullPath)) {
    return "File not found";
  }

  try {
    const content = fs.readFileSync(fullPath, "utf-8");
    const lines = content
      .split("\n")
      .slice(0, linesLimit || 40)
      .map((line) => {
        return line.length > 200 ? line.substring(0, 200) + "..." : line;
      });

    return lines.join("\n");
  } catch (error: any) {
    return `Error reading file: ${error.message}`;
  }
}

// Fulltext search using ripgrep
export async function fulltextSearch(
  query: string,
  repoPath: string
): Promise<string> {
  if (!repoPath) {
    return "No repository path provided";
  }

  if (!fs.existsSync(repoPath)) {
    return "Repository not cloned yet";
  }

  try {
    const args = [
      "--glob",
      "!dist",
      "--ignore-file",
      ".gitignore",
      "-C",
      "2",
      "-n",
      "--max-count",
      "10",
      "--max-columns",
      "200",
      query,
      "./",
    ];

    const result = await execRipgrepCommandDirect(args, repoPath, 5000);

    if (result.length > 10000) {
      return (
        result.substring(0, 10000) +
        "\n\n[... output truncated to 10,000 characters ...]"
      );
    }

    return result;
  } catch (error: any) {
    if (error.message.includes("code 1")) {
      return `No matches found for "${query}"`;
    }
    return `Error searching: ${error.message}`;
  }
}

// Execute arbitrary bash command
export async function executeBashCommand(
  command: string,
  repoPath: string,
  timeoutMs?: number
): Promise<string> {
  if (!repoPath) {
    return "No repository path provided";
  }

  if (!fs.existsSync(repoPath)) {
    return "Repository not cloned yet";
  }

  try {
    const result = await execShellCommand(command, repoPath, timeoutMs);
    return result;
  } catch (error: any) {
    return `Error executing command: ${error.message}`;
  }
}
