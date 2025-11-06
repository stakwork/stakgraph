import { Octokit } from "@octokit/rest";

interface PullRequestInfo {
  owner: string;
  repo: string;
  pull_number: number;
}

interface PRContentOptions {
  maxPatchLines?: number; // Max lines per file patch before truncation
  includeContext?: boolean; // Include diff context lines
}

const DEFAULT_OPTIONS: PRContentOptions = {
  maxPatchLines: 500,
  includeContext: true,
};

/**
 * Fetches comprehensive PR data and formats it as markdown for LLM consumption
 */
export async function fetchPullRequestContent(
  octokit: Octokit,
  prInfo: PullRequestInfo,
  options: PRContentOptions = {}
): Promise<string> {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  const [prData, files, reviewComments, issueComments, reviews, commits] =
    await Promise.all([
      fetchPRData(octokit, prInfo),
      fetchFiles(octokit, prInfo),
      fetchReviewComments(octokit, prInfo),
      fetchIssueComments(octokit, prInfo),
      fetchReviews(octokit, prInfo),
      fetchCommits(octokit, prInfo),
    ]);

  return formatPRContent(
    prData,
    files,
    reviewComments,
    issueComments,
    reviews,
    commits,
    opts
  );
}

/**
 * Fetch basic PR data
 */
async function fetchPRData(octokit: Octokit, prInfo: PullRequestInfo) {
  const { data } = await octokit.pulls.get({
    owner: prInfo.owner,
    repo: prInfo.repo,
    pull_number: prInfo.pull_number,
  });
  return data;
}

/**
 * Fetch changed files with patches
 */
async function fetchFiles(octokit: Octokit, prInfo: PullRequestInfo) {
  const { data } = await octokit.pulls.listFiles({
    owner: prInfo.owner,
    repo: prInfo.repo,
    pull_number: prInfo.pull_number,
    per_page: 100,
  });
  return data;
}

/**
 * Fetch line-specific code review comments
 */
async function fetchReviewComments(octokit: Octokit, prInfo: PullRequestInfo) {
  const { data } = await octokit.pulls.listReviewComments({
    owner: prInfo.owner,
    repo: prInfo.repo,
    pull_number: prInfo.pull_number,
    per_page: 100,
  });
  return data;
}

/**
 * Fetch general PR conversation comments
 */
async function fetchIssueComments(octokit: Octokit, prInfo: PullRequestInfo) {
  const { data } = await octokit.issues.listComments({
    owner: prInfo.owner,
    repo: prInfo.repo,
    issue_number: prInfo.pull_number,
    per_page: 100,
  });
  return data;
}

/**
 * Fetch PR reviews (approved, changes requested, etc)
 */
async function fetchReviews(octokit: Octokit, prInfo: PullRequestInfo) {
  const { data } = await octokit.pulls.listReviews({
    owner: prInfo.owner,
    repo: prInfo.repo,
    pull_number: prInfo.pull_number,
    per_page: 100,
  });
  return data;
}

/**
 * Fetch all commits in the PR
 */
async function fetchCommits(octokit: Octokit, prInfo: PullRequestInfo) {
  const { data } = await octokit.pulls.listCommits({
    owner: prInfo.owner,
    repo: prInfo.repo,
    pull_number: prInfo.pull_number,
    per_page: 100,
  });
  return data;
}

/**
 * Format all PR data as markdown
 */
function formatPRContent(
  prData: Record<string, unknown>,
  files: Record<string, unknown>[],
  reviewComments: Record<string, unknown>[],
  issueComments: Record<string, unknown>[],
  reviews: Record<string, unknown>[],
  commits: Record<string, unknown>[],
  options: PRContentOptions
): string {
  const sections: string[] = [];

  // Header
  sections.push(formatHeader(prData));

  // Description
  if (prData.body && typeof prData.body === "string") {
    sections.push(formatDescription(prData.body));
  }

  // Files Changed
  sections.push(formatFilesChanged(files, options));

  // Review Comments (line-specific)
  if (reviewComments.length > 0) {
    sections.push(formatReviewComments(reviewComments));
  }

  // Reviews (approval/changes requested)
  if (reviews.length > 0) {
    sections.push(formatReviews(reviews));
  }

  // General Discussion
  if (issueComments.length > 0) {
    sections.push(formatIssueComments(issueComments));
  }

  // Commits
  sections.push(formatCommits(commits));

  return sections.join("\n\n");
}

/**
 * Format PR header with metadata
 */
function formatHeader(prData: Record<string, unknown>): string {
  const totalAdditions = (prData.additions as number) || 0;
  const totalDeletions = (prData.deletions as number) || 0;
  const changedFiles = (prData.changed_files as number) || 0;
  const user = prData.user as { login: string } | undefined;
  const mergedBy = prData.merged_by as { login: string } | undefined;
  const base = prData.base as { ref: string } | undefined;
  const head = prData.head as { ref: string } | undefined;

  return `# Pull Request #${prData.number}: ${prData.title}

**Author:** @${user?.login || "unknown"}
**Merged by:** ${mergedBy ? `@${mergedBy.login}` : "N/A"}
**Merged at:** ${prData.merged_at || "N/A"}
**Base branch:** ${base?.ref || "unknown"} → **Head branch:** ${
    head?.ref || "unknown"
  }
**Changes:** ${changedFiles} files changed, +${totalAdditions} -${totalDeletions}
**PR URL:** ${prData.html_url}`;
}

/**
 * Format PR description
 */
function formatDescription(body: string): string {
  return `## Description

${body}`;
}

/**
 * Format files changed with patches
 */
function formatFilesChanged(
  files: Record<string, unknown>[],
  options: PRContentOptions
): string {
  const sections = [`## Files Changed (${files.length} files)`];

  for (const file of files) {
    const status = (file.status as string) || "unknown";
    const additions = (file.additions as number) || 0;
    const deletions = (file.deletions as number) || 0;
    const filename = (file.filename as string) || "unknown";
    const patch = file.patch as string | undefined;

    sections.push(`### ${filename}`);
    sections.push(
      `**Status:** ${status} | **Changes:** +${additions} -${deletions}`
    );

    if (patch) {
      const truncatedPatch = truncatePatch(patch, options.maxPatchLines!);
      sections.push("\n```diff");
      sections.push(truncatedPatch);
      sections.push("```");
    } else {
      sections.push("\n*No patch available (binary file or too large)*");
    }
  }

  return sections.join("\n");
}

/**
 * Truncate patch if too long
 */
function truncatePatch(patch: string, maxLines: number): string {
  const lines = patch.split("\n");
  if (lines.length <= maxLines) {
    return patch;
  }

  const truncated = lines.slice(0, maxLines);
  const remaining = lines.length - maxLines;
  truncated.push(`\n... truncated ${remaining} lines ...`);
  return truncated.join("\n");
}

/**
 * Format line-specific review comments
 */
function formatReviewComments(comments: Record<string, unknown>[]): string {
  const sections = ["## Code Review Comments"];

  // Group by file
  const commentsByFile: Record<string, Record<string, unknown>[]> = {};
  for (const comment of comments) {
    const file = (comment.path as string) || "unknown";
    if (!commentsByFile[file]) {
      commentsByFile[file] = [];
    }
    commentsByFile[file].push(comment);
  }

  for (const [file, fileComments] of Object.entries(commentsByFile)) {
    sections.push(`\n### ${file}`);

    for (const comment of fileComments) {
      const line =
        (comment.line as number) || (comment.original_line as number) || "?";
      const user = comment.user as { login: string } | undefined;
      const author = user?.login || "unknown";
      const createdAt = comment.created_at
        ? new Date(comment.created_at as string).toLocaleString()
        : "unknown";
      const diffHunk = comment.diff_hunk as string | undefined;
      const body = (comment.body as string) || "";

      sections.push(`\n**@${author}** on line ${line} - ${createdAt}`);
      if (diffHunk) {
        sections.push("```diff");
        sections.push(diffHunk);
        sections.push("```");
      }
      sections.push(`> ${body.replace(/\n/g, "\n> ")}`);
    }
  }

  return sections.join("\n");
}

/**
 * Format PR reviews (approved, changes requested, etc)
 */
function formatReviews(reviews: Record<string, unknown>[]): string {
  const sections = ["## Reviews"];

  for (const review of reviews) {
    const body = review.body as string | undefined;
    const state = (review.state as string) || "";

    if (!body || state === "COMMENTED") continue; // Skip empty or pure comment reviews

    const user = review.user as { login: string } | undefined;
    const reviewer = user?.login || "unknown";
    const createdAt = review.submitted_at
      ? new Date(review.submitted_at as string).toLocaleString()
      : "unknown";

    const stateEmoji: Record<string, string> = {
      APPROVED: "✅",
      CHANGES_REQUESTED: "🔄",
      COMMENTED: "💬",
    };
    const emoji = stateEmoji[state] || "";

    sections.push(
      `\n${emoji} **@${reviewer}** ${state
        .toLowerCase()
        .replace("_", " ")} - ${createdAt}`
    );
    sections.push(`> ${body.replace(/\n/g, "\n> ")}`);
  }

  return sections.join("\n");
}

/**
 * Format general PR discussion comments
 */
function formatIssueComments(comments: Record<string, unknown>[]): string {
  const sections = ["## Discussion"];

  for (const comment of comments) {
    const user = comment.user as { login: string } | undefined;
    const author = user?.login || "unknown";
    const createdAt = comment.created_at
      ? new Date(comment.created_at as string).toLocaleString()
      : "unknown";
    const body = (comment.body as string) || "";

    sections.push(`\n**@${author}** - ${createdAt}`);
    sections.push(`> ${body.replace(/\n/g, "\n> ")}`);
  }

  return sections.join("\n");
}

/**
 * Format commit history
 */
function formatCommits(commits: Record<string, unknown>[]): string {
  const sections = [`## Commits (${commits.length} commits)`];

  for (const commit of commits) {
    const sha = ((commit.sha as string) || "").substring(0, 7);
    const commitData = commit.commit as
      | { author?: { name?: string }; message?: string }
      | undefined;
    const author = commitData?.author?.name || "unknown";
    const message = (commitData?.message || "").split("\n")[0]; // First line only

    sections.push(`- \`${sha}\` @${author}: ${message}`);
  }

  return sections.join("\n");
}
