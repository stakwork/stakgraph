import { SimpleGitOptions, SimpleGit, simpleGit } from "simple-git";
import path from "path";
import fs from "fs";

const OPTIONS: SimpleGitOptions = {
  baseDir: "/tmp",
  binary: "git",
  maxConcurrentProcesses: 10,
  trimmed: true,
  config: [],
};

const git: SimpleGit = simpleGit(OPTIONS);

// Lock map to prevent concurrent clones to the same directory
const cloneLocks = new Map<string, Promise<string>>();

export async function cloneOrUpdateRepo(
  repoUrl: string,
  username?: string,
  pat?: string,
  commit?: string
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

  // Create and store the promise for this clone operation
  const clonePromise = doCloneOrUpdate(repoUrl, cloneDir, username, pat, commit);
  cloneLocks.set(cloneDir, clonePromise);

  console.log("===> cloning into", cloneDir);
  try {
    return await clonePromise;
  } finally {
    cloneLocks.delete(cloneDir);
  }
}

async function doCloneOrUpdate(
  repoUrl: string,
  cloneDir: string,
  username?: string,
  pat?: string,
  commit?: string
): Promise<string> {
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
      const repoGit = simpleGit(cloneDir);
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
      console.log(`Error checking repo, re-cloning...`);
      fs.rmSync(cloneDir, { recursive: true, force: true });
    }
  }

  // Clone the repo (either directory didn't exist or was removed)
  await git.clone(url, cloneDir, ["--single-branch"]);

  // Checkout specific commit if provided
  if (commit) {
    const repoGit = simpleGit(cloneDir);
    await repoGit.checkout(commit);
  }

  return cloneDir;
}
