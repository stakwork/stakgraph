import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

export interface CloneResult {
  success: boolean;
  localPath: string;
  error?: string;
  duration?: number;
}

export interface CloneOptions {
  username?: string;
  token?: string;
  branch?: string;
}

export class RepoCloner {
  private static readonly BASE_PATH =
    process.env.GITSEE_BASE_PATH || "/tmp/gitsee";
  private static clonePromises: Map<string, Promise<CloneResult>> = new Map();

  /**
   * Clone a repository in the background (fire-and-forget)
   */
  static async cloneInBackground(
    owner: string,
    repo: string,
    options?: CloneOptions
  ): Promise<void> {
    const repoKey = `${owner}/${repo}`;

    // If already cloning, don't start another clone
    if (this.clonePromises.has(repoKey)) {
      return;
    }

    // Start clone and store promise
    const clonePromise = this.cloneRepo(owner, repo, options);
    this.clonePromises.set(repoKey, clonePromise);

    // Handle completion (success or failure)
    clonePromise
      .finally(() => {
        // Clean up the promise from the map after completion
        setTimeout(() => {
          this.clonePromises.delete(repoKey);
        }, 5000); // Keep for 5 seconds to allow quick access
      })
      .catch((error) => {
        console.error(
          `🚨 Background clone failed for ${owner}/${repo}:`,
          error.message
        );
      });
  }

  /**
   * Clone a repository to /tmp/gitsee/{owner}/{repo}
   */
  static async cloneRepo(
    owner: string,
    repo: string,
    options?: CloneOptions
  ): Promise<CloneResult> {
    const startTime = Date.now();
    const repoPath = path.join(this.BASE_PATH, owner, repo);

    // Build GitHub URL with authentication if provided
    let githubUrl: string;
    if (options?.username && options?.token) {
      githubUrl = `https://${options.username}:${options.token}@github.com/${owner}/${repo}.git`;
    } else {
      githubUrl = `https://github.com/${owner}/${repo}.git`;
    }

    console.log(`📥 Starting clone of ${owner}/${repo} to ${repoPath}`);

    try {
      // Check if already exists AND is a valid clone (has .git or at least some files)
      if (fs.existsSync(repoPath)) {
        const hasGit = fs.existsSync(path.join(repoPath, ".git"));
        const hasFiles = fs.readdirSync(repoPath).length > 0;

        if (hasGit || hasFiles) {
          console.log(
            `📂 Repository ${owner}/${repo} already exists at ${repoPath}`
          );
          return {
            success: true,
            localPath: repoPath,
            duration: Date.now() - startTime,
          };
        } else {
          console.log(
            `🗑️ Repository ${owner}/${repo} exists but appears invalid, removing...`
          );
          fs.rmSync(repoPath, { recursive: true, force: true });
        }
      }

      // Ensure parent directory exists
      const parentDir = path.dirname(repoPath);
      fs.mkdirSync(parentDir, { recursive: true });

      // Clone with shallow copy (depth 1) and single branch for speed
      const result = await this.executeGitClone(
        githubUrl,
        repoPath,
        options?.branch
      );

      const duration = Date.now() - startTime;

      if (result.success) {
        console.log(`✅ Successfully cloned ${owner}/${repo} in ${duration}ms`);
        return {
          success: true,
          localPath: repoPath,
          duration,
        };
      } else {
        console.error(`❌ Failed to clone ${owner}/${repo}:`, result.error);
        return {
          success: false,
          localPath: repoPath,
          error: result.error,
          duration,
        };
      }
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`💥 Clone error for ${owner}/${repo}:`, error.message);

      return {
        success: false,
        localPath: repoPath,
        error: error.message,
        duration,
      };
    }
  }

  /**
   * Execute git clone command with shallow clone and single branch
   */
  private static executeGitClone(
    githubUrl: string,
    targetPath: string,
    branch?: string
  ): Promise<{ success: boolean; error?: string }> {
    return new Promise((resolve) => {
      // Build git clone arguments
      const gitArgs = [
        "clone",
        "--depth",
        "1", // Shallow clone (only latest commit)
        "--single-branch", // Only clone the specified branch
        "--no-tags", // Skip tags for speed
      ];

      // Add branch specification if provided
      if (branch) {
        gitArgs.push("--branch", branch);
      }

      gitArgs.push(githubUrl, targetPath);

      // Use shallow clone with single branch for maximum speed
      const gitProcess = spawn("git", gitArgs);

      let errorOutput = "";

      gitProcess.stderr.on("data", (data) => {
        errorOutput += data.toString();
      });

      gitProcess.stdout.on("data", (data) => {
        // Git clone sends progress to stderr, but we can capture stdout too
        const output = data.toString();
        if (output.includes("Cloning") || output.includes("Receiving")) {
          console.log(`📥 ${output.trim()}`);
        }
      });

      gitProcess.on("close", (code) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          resolve({
            success: false,
            error: errorOutput || `Git clone exited with code ${code}`,
          });
        }
      });

      gitProcess.on("error", (error) => {
        resolve({
          success: false,
          error: `Failed to start git process: ${error.message}`,
        });
      });
    });
  }

  /**
   * Check if a repository is already cloned
   */
  static isRepoCloned(owner: string, repo: string): boolean {
    const repoPath = path.join(this.BASE_PATH, owner, repo);
    return (
      fs.existsSync(repoPath) && fs.existsSync(path.join(repoPath, ".git"))
    );
  }

  /**
   * Get the local path for a repository
   */
  static getRepoPath(owner: string, repo: string): string {
    return path.join(this.BASE_PATH, owner, repo);
  }

  /**
   * Wait for a repository clone to complete
   */
  static async waitForClone(
    owner: string,
    repo: string,
    options?: CloneOptions
  ): Promise<CloneResult> {
    const repoKey = `${owner}/${repo}`;

    // Check if already cloned
    if (this.isRepoCloned(owner, repo)) {
      return {
        success: true,
        localPath: this.getRepoPath(owner, repo),
      };
    }

    // Check if currently cloning
    const clonePromise = this.clonePromises.get(repoKey);
    if (clonePromise) {
      console.log(`⏳ Waiting for ongoing clone of ${owner}/${repo}...`);
      return await clonePromise;
    }

    // Start a new clone
    console.log(`🚀 Starting new clone for ${owner}/${repo}...`);
    return await this.cloneRepo(owner, repo, options);
  }

  /**
   * Get clone result if available (non-blocking)
   */
  static async getCloneResult(
    owner: string,
    repo: string
  ): Promise<CloneResult | null> {
    const repoKey = `${owner}/${repo}`;

    // Check if already cloned
    if (this.isRepoCloned(owner, repo)) {
      return {
        success: true,
        localPath: this.getRepoPath(owner, repo),
      };
    }

    // Check if currently cloning
    const clonePromise = this.clonePromises.get(repoKey);
    if (clonePromise) {
      try {
        return await clonePromise;
      } catch (error) {
        return {
          success: false,
          localPath: this.getRepoPath(owner, repo),
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    }

    // Not cloned and not cloning
    return null;
  }

  /**
   * Clean up old repositories (optional utility)
   */
  static async cleanupOldRepos(maxAgeHours: number = 24): Promise<void> {
    try {
      if (!fs.existsSync(this.BASE_PATH)) {
        return;
      }

      const cutoffTime = Date.now() - maxAgeHours * 60 * 60 * 1000;

      // Walk through /tmp/gitsee/{owner}/{repo} directories
      const owners = fs.readdirSync(this.BASE_PATH);

      for (const owner of owners) {
        const ownerPath = path.join(this.BASE_PATH, owner);
        if (!fs.statSync(ownerPath).isDirectory()) continue;

        const repos = fs.readdirSync(ownerPath);

        for (const repo of repos) {
          const repoPath = path.join(ownerPath, repo);
          const stats = fs.statSync(repoPath);

          if (stats.isDirectory() && stats.mtime.getTime() < cutoffTime) {
            console.log(`🗑️ Cleaning up old repo: ${owner}/${repo}`);
            fs.rmSync(repoPath, { recursive: true, force: true });
          }
        }
      }
    } catch (error: any) {
      console.error("Error cleaning up old repos:", error.message);
    }
  }
}
