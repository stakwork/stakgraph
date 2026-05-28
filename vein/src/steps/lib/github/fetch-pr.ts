import { z } from "zod";
import { Octokit } from "@octokit/rest";
import { defineStep } from "../../../core.js";

const EXAMPLE = `- id: pr
  type: github/fetch-pr
  config:
    owner: "facebook"
    repo: "react"
    pull_number: 1234
    token: "{{ input.githubToken }}"`;

const limitsSchema = z
  .object({
    maxPatchLines: z.number().int().positive().default(500),
    maxFiles: z.number().int().positive().default(50),
    maxDescriptionChars: z.number().int().positive().default(5000),
    maxCommentChars: z.number().int().positive().default(500),
    maxComments: z.number().int().positive().default(50),
    maxReviews: z.number().int().positive().default(20),
  })
  .default({});

type Limits = z.infer<typeof limitsSchema>;

export default defineStep({
  type: "github/fetch-pr",
  description: `Fetch a GitHub pull request with files, reviews, comments, and commits, and format it as markdown for LLM consumption. Output: { markdown, pr: { number, title, mergedAt, author, htmlUrl, additions, deletions, changedFiles } }.\n\n${EXAMPLE}`,
  input: z.object({
    owner: z.string().min(1),
    repo: z.string().min(1),
    pull_number: z.number().int().positive(),
    token: z.string().optional(),
    limits: limitsSchema,
  }),
  output: z.object({
    markdown: z.string(),
    pr: z.object({
      number: z.number(),
      title: z.string(),
      mergedAt: z.string().nullable(),
      author: z.string(),
      htmlUrl: z.string(),
      additions: z.number(),
      deletions: z.number(),
      changedFiles: z.number(),
    }),
  }),
  async run(cfg) {
    const auth = cfg.token ?? process.env["GITHUB_TOKEN"];
    const octokit = new Octokit(auth ? { auth } : {});

    const prInfo = {
      owner: cfg.owner,
      repo: cfg.repo,
      pull_number: cfg.pull_number,
    };

    const [
      { data: prData },
      { data: files },
      { data: reviewComments },
      { data: issueComments },
      { data: reviews },
      { data: commits },
    ] = await Promise.all([
      octokit.pulls.get(prInfo),
      octokit.pulls.listFiles({ ...prInfo, per_page: 100 }),
      octokit.pulls.listReviewComments({ ...prInfo, per_page: 100 }),
      octokit.issues.listComments({
        owner: prInfo.owner,
        repo: prInfo.repo,
        issue_number: prInfo.pull_number,
        per_page: 100,
      }),
      octokit.pulls.listReviews({ ...prInfo, per_page: 100 }),
      octokit.pulls.listCommits({ ...prInfo, per_page: 100 }),
    ]);

    const markdown = formatPRContent(
      prData as Record<string, unknown>,
      files as Record<string, unknown>[],
      reviewComments as Record<string, unknown>[],
      issueComments as Record<string, unknown>[],
      reviews as Record<string, unknown>[],
      commits as Record<string, unknown>[],
      cfg.limits,
    );

    return {
      markdown,
      pr: {
        number: prData.number,
        title: prData.title,
        mergedAt: prData.merged_at,
        author: prData.user?.login ?? "unknown",
        htmlUrl: prData.html_url,
        additions: prData.additions ?? 0,
        deletions: prData.deletions ?? 0,
        changedFiles: prData.changed_files ?? 0,
      },
    };
  },
});

// ── Formatting helpers (adapted from mcp/src/gitree/pr.ts) ─────────────────

function formatPRContent(
  prData: Record<string, unknown>,
  files: Record<string, unknown>[],
  reviewComments: Record<string, unknown>[],
  issueComments: Record<string, unknown>[],
  reviews: Record<string, unknown>[],
  commits: Record<string, unknown>[],
  limits: Limits,
): string {
  const sections: string[] = [formatHeader(prData)];

  if (typeof prData.body === "string" && prData.body.length > 0) {
    sections.push(formatDescription(prData.body, limits.maxDescriptionChars));
  }

  sections.push(
    formatFilesChanged(files.slice(0, limits.maxFiles), files.length, limits),
  );

  if (reviewComments.length > 0) {
    sections.push(
      formatReviewComments(
        reviewComments.slice(0, limits.maxComments),
        reviewComments.length,
        limits.maxCommentChars,
      ),
    );
  }

  if (reviews.length > 0) {
    sections.push(
      formatReviews(
        reviews.slice(0, limits.maxReviews),
        reviews.length,
        limits.maxCommentChars,
      ),
    );
  }

  if (issueComments.length > 0) {
    sections.push(
      formatIssueComments(
        issueComments.slice(0, limits.maxComments),
        issueComments.length,
        limits.maxCommentChars,
      ),
    );
  }

  sections.push(formatCommits(commits));
  return sections.join("\n\n");
}

function formatHeader(prData: Record<string, unknown>): string {
  const user = prData.user as { login: string } | undefined;
  const mergedBy = prData.merged_by as { login: string } | undefined;
  const base = prData.base as { ref: string } | undefined;
  const head = prData.head as { ref: string } | undefined;
  const additions = (prData.additions as number) || 0;
  const deletions = (prData.deletions as number) || 0;
  const changedFiles = (prData.changed_files as number) || 0;

  return `# Pull Request #${prData.number}: ${prData.title}

**Author:** @${user?.login ?? "unknown"}
**Merged by:** ${mergedBy ? `@${mergedBy.login}` : "N/A"}
**Merged at:** ${prData.merged_at ?? "N/A"}
**Base branch:** ${base?.ref ?? "unknown"} → **Head branch:** ${head?.ref ?? "unknown"}
**Changes:** ${changedFiles} files changed, +${additions} -${deletions}
**PR URL:** ${prData.html_url}`;
}

function formatDescription(body: string, maxChars: number): string {
  const truncated =
    body.length > maxChars
      ? `${body.substring(0, maxChars)}\n\n... [truncated ${body.length - maxChars} characters]`
      : body;
  return `## Description\n\n${truncated}`;
}

function truncatePatch(patch: string, maxLines: number): string {
  const lines = patch.split("\n");
  if (lines.length <= maxLines) return patch;
  const remaining = lines.length - maxLines;
  return [...lines.slice(0, maxLines), `\n... truncated ${remaining} lines ...`].join("\n");
}

function formatFilesChanged(
  files: Record<string, unknown>[],
  total: number,
  limits: Limits,
): string {
  const sections = [`## Files Changed (showing ${files.length} of ${total} files)`];
  if (files.length < total) {
    sections.push(`\n*Note: ${total - files.length} files omitted for brevity*\n`);
  }
  for (const file of files) {
    const filename = (file.filename as string) ?? "unknown";
    const status = (file.status as string) ?? "unknown";
    const additions = (file.additions as number) ?? 0;
    const deletions = (file.deletions as number) ?? 0;
    const patch = file.patch as string | undefined;

    sections.push(`### ${filename}`);
    sections.push(`**Status:** ${status} | **Changes:** +${additions} -${deletions}`);
    if (patch) {
      sections.push("\n```diff");
      sections.push(truncatePatch(patch, limits.maxPatchLines));
      sections.push("```");
    } else {
      sections.push("\n*No patch available (binary file or too large)*");
    }
  }
  return sections.join("\n");
}

function truncate(body: string, max: number): string {
  return body.length > max ? `${body.substring(0, max)}... [truncated]` : body;
}

function formatReviewComments(
  comments: Record<string, unknown>[],
  total: number,
  maxChars: number,
): string {
  const sections = [`## Code Review Comments (showing ${comments.length} of ${total})`];
  if (comments.length < total) {
    sections.push(`\n*Note: ${total - comments.length} comments omitted for brevity*\n`);
  }

  const byFile: Record<string, Record<string, unknown>[]> = {};
  for (const c of comments) {
    const file = (c.path as string) ?? "unknown";
    (byFile[file] ??= []).push(c);
  }

  for (const [file, fileComments] of Object.entries(byFile)) {
    sections.push(`\n### ${file}`);
    for (const comment of fileComments) {
      const line =
        (comment.line as number) ?? (comment.original_line as number) ?? "?";
      const author = (comment.user as { login: string } | undefined)?.login ?? "unknown";
      const createdAt = comment.created_at
        ? new Date(comment.created_at as string).toLocaleString()
        : "unknown";
      const diffHunk = comment.diff_hunk as string | undefined;
      const body = truncate((comment.body as string) ?? "", maxChars);

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

function formatReviews(
  reviews: Record<string, unknown>[],
  total: number,
  maxChars: number,
): string {
  const sections = [`## Reviews (showing ${reviews.length} of ${total})`];
  if (reviews.length < total) {
    sections.push(`\n*Note: ${total - reviews.length} reviews omitted for brevity*\n`);
  }

  const stateEmoji: Record<string, string> = {
    APPROVED: "✅",
    CHANGES_REQUESTED: "🔄",
    COMMENTED: "💬",
  };

  for (const review of reviews) {
    const body = review.body as string | undefined;
    const state = (review.state as string) ?? "";
    if (!body || state === "COMMENTED") continue;

    const reviewer = (review.user as { login: string } | undefined)?.login ?? "unknown";
    const createdAt = review.submitted_at
      ? new Date(review.submitted_at as string).toLocaleString()
      : "unknown";
    const emoji = stateEmoji[state] ?? "";
    const truncated = truncate(body, maxChars);

    sections.push(
      `\n${emoji} **@${reviewer}** ${state.toLowerCase().replace("_", " ")} - ${createdAt}`,
    );
    sections.push(`> ${truncated.replace(/\n/g, "\n> ")}`);
  }
  return sections.join("\n");
}

function formatIssueComments(
  comments: Record<string, unknown>[],
  total: number,
  maxChars: number,
): string {
  const sections = [`## Discussion (showing ${comments.length} of ${total})`];
  if (comments.length < total) {
    sections.push(`\n*Note: ${total - comments.length} comments omitted for brevity*\n`);
  }
  for (const comment of comments) {
    const author = (comment.user as { login: string } | undefined)?.login ?? "unknown";
    const createdAt = comment.created_at
      ? new Date(comment.created_at as string).toLocaleString()
      : "unknown";
    const body = truncate((comment.body as string) ?? "", maxChars);
    sections.push(`\n**@${author}** - ${createdAt}`);
    sections.push(`> ${body.replace(/\n/g, "\n> ")}`);
  }
  return sections.join("\n");
}

function formatCommits(commits: Record<string, unknown>[]): string {
  const sections = [`## Commits (${commits.length} commits)`];
  for (const commit of commits) {
    const sha = ((commit.sha as string) ?? "").substring(0, 7);
    const data = commit.commit as
      | { author?: { name?: string }; message?: string }
      | undefined;
    const author = data?.author?.name ?? "unknown";
    const message = (data?.message ?? "").split("\n")[0];
    sections.push(`- \`${sha}\` @${author}: ${message}`);
  }
  return sections.join("\n");
}
