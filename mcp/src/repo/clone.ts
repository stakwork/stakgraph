import { SimpleGitOptions, SimpleGit, simpleGit } from "simple-git";
import path from "path";
import fs from "fs";

// Fail immediately on credential prompts instead of blocking on a TTY
process.env.GIT_TERMINAL_PROMPT = "0";
process.env.GIT_ASKPASS = "echo";

const OPTIONS: SimpleGitOptions = {
  baseDir: "/tmp",
  binary: "git",
  maxConcurrentProcesses: 10,
  trimmed: true,
  config: [],
  timeout: { block: 600_000 }, // 10 min — kill git if no output
};

const CLONE_TIMEOUT_MS = 10 * 60 * 1000;

function cloneWithTimeout(promise: Promise<string>, label: string): Promise<string> {
  return Promise.race([
    promise,
    new Promise<string>((_, reject) =>
      setTimeout(
        () => reject(new Error(`[clone] Timeout after 10 min: ${label}`)),
        CLONE_TIMEOUT_MS
      )
    ),
  ]);
}

// Lock map to prevent concurrent clones to the same directory
const cloneLocks = new Map<string, Promise<string>>();

/**
 * Clone or update one or more repositories.
 * If repoUrls is a comma-separated list, clones all repos to /tmp/owner/repo each
 * and returns /tmp as the parent directory.
 * If repoUrls is a single URL, returns the specific repo directory (backward compatible).
 */
export async function cloneOrUpdateRepo(
  repoUrls: string,
  username?: string,
  pat?: string,
  commit?: string,
  abortSignal?: AbortSignal
): Promise<string> {
  // Split by comma and trim whitespace
  const urls = repoUrls.split(",").map((url) => url.trim()).filter((url) => url.length > 0);

  if (urls.length === 1) {
    // Single repo - use original behavior for backward compatibility
    return cloneSingleRepo(urls[0], username, pat, commit, abortSignal);
  }

  // Multiple repos - clone all in parallel, return /tmp
  return cloneMultipleRepos(urls, username, pat, commit, abortSignal);
}

/**
 * Clone a single repository (original behavior)
 */
async function cloneSingleRepo(
  repoUrl: string,
  username?: string,
  pat?: string,
  commit?: string,
  abortSignal?: AbortSignal
): Promise<string> {
  // Extract owner and repo name from URL
  const urlParts = repoUrl.replace(/\.git$/, "").split("/");
  const repoName = urlParts.pop() || "repo";
  const owner = urlParts.pop() || "";

  // Create directory structure: /tmp/owner/repo
  const cloneDir = path.join("/tmp", owner, repoName);

  // Check if there's already a clone operation in progress for this directory
  const existingLock = cloneLocks.get(cloneDir);
  if (existingLock) {
    console.log(`Clone already in progress for ${cloneDir}, waiting...`);
    return existingLock;
  }

  if (abortSignal?.aborted) {
    return Promise.reject(new Error(`[clone] Aborted before start: ${cloneDir}`));
  }

  const clonePromise = cloneWithTimeout(
    doCloneOrUpdate(repoUrl, cloneDir, username, pat, commit, abortSignal),
    cloneDir
  );
  cloneLocks.set(cloneDir, clonePromise);

  console.log("===> cloning into", cloneDir);
  try {
    return await clonePromise;
  } finally {
    cloneLocks.delete(cloneDir);
  }
}

/**
 * Clone multiple repositories to /tmp/owner/repo each.
 * Returns /tmp as the parent directory containing all repos.
 */
async function cloneMultipleRepos(
  repoUrls: string[],
  username?: string,
  pat?: string,
  commit?: string,
  abortSignal?: AbortSignal
): Promise<string> {
  console.log(`===> cloning ${repoUrls.length} repos into /tmp`);

  // Clone all repos in parallel using the same structure as single repos
  const clonePromises = repoUrls.map((repoUrl) => 
    cloneSingleRepo(repoUrl, username, pat, commit, abortSignal)
  );

  await Promise.all(clonePromises);

  // Return /tmp as the parent directory containing all repos
  return "/tmp";
}

async function doCloneOrUpdate(
  repoUrl: string,
  cloneDir: string,
  username?: string,
  pat?: string,
  commit?: string,
  abortSignal?: AbortSignal
): Promise<string> {
  const opts: SimpleGitOptions = abortSignal
    ? { ...OPTIONS, abort: abortSignal }
    : OPTIONS;
  const git = simpleGit(opts);
  let url = repoUrl;
  if (pat) {
    // GitHub accepts PAT with any username (or just the token as username)
    const user = username || "x-access-token";
    url = repoUrl.replace("https://", `https://${user}:${pat}@`);
  }

  // Check if directory exists and is a git repo
  if (fs.existsSync(cloneDir)) {
    try {
      // Only create simpleGit instance if directory exists
      const repoGit = simpleGit(cloneDir, { timeout: { block: 600_000 }, ...(abortSignal ? { abort: abortSignal } : {}) });
      const isRepo = await repoGit.checkIsRepo();

      if (isRepo) {
        console.log(`Repository already exists at ${cloneDir}, updating...`);

        // Fetch latest changes
        await repoGit.fetch();

        // If no specific commit, pull latest
        if (!commit) {
          await repoGit.pull();
        } else {
          // Checkout specific commit
          await repoGit.checkout(commit);
        }

        return cloneDir;
      } else {
        // Directory exists but is not a git repo, remove it
        fs.rmSync(cloneDir, { recursive: true, force: true });
      }
    } catch (error) {
      // If there's an error checking, try to remove and re-clone
      console.log(`Error checking repo, re-cloning... (${(error as Error).message})`);
      fs.rmSync(cloneDir, { recursive: true, force: true });
    }
  }

  // Clone the repo (either directory didn't exist or was removed)
  await git.clone(url, cloneDir, ["--single-branch"]);

  // Checkout specific commit if provided
  if (commit) {
    const repoGit = simpleGit(cloneDir, { timeout: { block: 600_000 }, ...(abortSignal ? { abort: abortSignal } : {}) });
    await repoGit.checkout(commit);
  }

  return cloneDir;
}
