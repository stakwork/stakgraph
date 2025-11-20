import { Octokit } from "@octokit/rest";

interface CommitInfo {
  owner: string;
  repo: string;
  sha: string;
}

interface CommitContentOptions {
  maxPatchLines?: number; // Max lines per file patch before truncation
}

const DEFAULT_OPTIONS: CommitContentOptions = {
  maxPatchLines: 100,
};

/**
 * Fetches comprehensive commit data and formats it as markdown for LLM consumption
 */
export async function fetchCommitContent(
  octokit: Octokit,
  commitInfo: CommitInfo,
  options: CommitContentOptions = {}
): Promise<string> {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  const { data: commit } = await octokit.repos.getCommit({
    owner: commitInfo.owner,
    repo: commitInfo.repo,
    ref: commitInfo.sha,
  });

  return formatCommitContent(commit, opts);
}

/**
 * Fetch commits not associated with any pull request
 * Returns commits in chronological order (oldest first)
 */
export async function fetchOrphanCommits(
  octokit: Octokit,
  owner: string,
  repo: string,
  since?: string // SHA to start from (exclusive)
): Promise<Array<{
  sha: string;
  message: string;
  author: string;
  committedAt: Date;
  url: string;
}>> {
  const orphanCommits: Array<{
    sha: string;
    message: string;
    author: string;
    committedAt: Date;
    url: string;
  }> = [];

  // Fetch all commits from the main branch in chronological order
  const commits = await octokit.paginate(octokit.repos.listCommits, {
    owner,
    repo,
    per_page: 100,
    // Note: GitHub API returns newest first, we'll reverse later
  });

  // Reverse to get chronological order (oldest first)
  commits.reverse();

  // If since is provided, skip commits until we find it
  let foundSince = !since;

  for (const commit of commits) {
    // Skip until we find the "since" commit
    if (!foundSince) {
      if (commit.sha === since) {
        foundSince = true;
      }
      continue;
    }

    // Check if this commit is associated with any PR
    try {
      const { data: prs } =
        await octokit.repos.listPullRequestsAssociatedWithCommit({
          owner,
          repo,
          commit_sha: commit.sha,
        });

      // Only include if no merged PRs are associated
      const mergedPRs = prs.filter((pr) => pr.merged_at);
      if (mergedPRs.length === 0) {
        orphanCommits.push({
          sha: commit.sha,
          message: commit.commit.message,
          author:
            commit.commit.author?.name ||
            commit.author?.login ||
            "Unknown",
          committedAt: new Date(commit.commit.author?.date || Date.now()),
          url: commit.html_url,
        });
      }
    } catch (error) {
      console.error(`Error checking PR association for ${commit.sha}:`, error);
      // If we can't check, assume it's an orphan
      orphanCommits.push({
        sha: commit.sha,
        message: commit.commit.message,
        author:
          commit.commit.author?.name ||
          commit.author?.login ||
          "Unknown",
        committedAt: new Date(commit.commit.author?.date || Date.now()),
        url: commit.html_url,
      });
    }
  }

  return orphanCommits;
}

/**
 * Format commit data as markdown
 */
function formatCommitContent(
  commit: Record<string, any>,
  options: CommitContentOptions
): string {
  const sections: string[] = [];

  // Header
  sections.push(formatHeader(commit));

  // Commit Message
  if (commit.commit?.message) {
    sections.push(formatCommitMessage(commit.commit.message));
  }

  // Files Changed
  if (commit.files && Array.isArray(commit.files)) {
    sections.push(formatFilesChanged(commit.files, options));
  }

  // Stats
  if (commit.stats) {
    sections.push(formatStats(commit.stats));
  }

  return sections.join("\n\n");
}

/**
 * Format commit header
 */
function formatHeader(commit: Record<string, any>): string {
  const sha = commit.sha?.substring(0, 7) || "unknown";
  const author = commit.commit?.author?.name || commit.author?.login || "Unknown";
  const date = commit.commit?.author?.date || new Date().toISOString();
  const message = commit.commit?.message?.split('\n')[0] || "No message";

  return `# Commit ${sha}: ${message}

**Author**: ${author}
**Date**: ${date}
**URL**: ${commit.html_url || ""}`;
}

/**
 * Format commit message
 */
function formatCommitMessage(message: string): string {
  const lines = message.split('\n');
  const title = lines[0];
  const body = lines.slice(1).join('\n').trim();

  if (!body) {
    return `## Commit Message\n\n${title}`;
  }

  return `## Commit Message\n\n**${title}**\n\n${body}`;
}

/**
 * Format files changed with patches
 */
function formatFilesChanged(
  files: Record<string, any>[],
  options: CommitContentOptions
): string {
  if (files.length === 0) {
    return "## Files Changed\n\nNo files changed.";
  }

  const fileList = files
    .map((file) => {
      const status = file.status || "modified";
      const additions = file.additions || 0;
      const deletions = file.deletions || 0;
      const filename = file.filename || "unknown";

      let summary = `- **${filename}** (${status}, +${additions} -${deletions})`;

      // Add patch if available (truncated)
      if (file.patch) {
        const patchLines = file.patch.split("\n");
        const truncated = patchLines.length > options.maxPatchLines!;
        const displayLines = truncated
          ? patchLines.slice(0, options.maxPatchLines!)
          : patchLines;

        summary += `\n\`\`\`diff\n${displayLines.join("\n")}${
          truncated ? `\n... (truncated ${patchLines.length - options.maxPatchLines!} lines)` : ""
        }\n\`\`\``;
      }

      return summary;
    })
    .join("\n\n");

  return `## Files Changed (${files.length})

${fileList}`;
}

/**
 * Format commit stats
 */
function formatStats(stats: Record<string, any>): string {
  const additions = stats.additions || 0;
  const deletions = stats.deletions || 0;
  const total = stats.total || 0;

  return `## Stats

- **Total changes**: ${total}
- **Additions**: ${additions}
- **Deletions**: ${deletions}`;
}
