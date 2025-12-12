import { Octokit } from "@octokit/rest";
import { Storage } from "./store/index.js";
import { LLMClient, SYSTEM_PROMPT, DECISION_GUIDELINES } from "./llm.js";
import {
  Feature,
  PRRecord,
  CommitRecord,
  LLMDecision,
  GitHubPR,
  Usage,
  ChronologicalCheckpoint,
} from "./types.js";
import { fetchPullRequestContent } from "./pr.js";
import { fetchCommitContent } from "./commit.js";

/**
 * Main class for building the feature knowledge base from PRs and commits
 */
export class StreamingFeatureBuilder {
  private clueAnalyzer?: any; // ClueAnalyzer instance (lazily initialized)

  constructor(
    private storage: Storage,
    private llm: LLMClient,
    private octokit: Octokit,
    private repoPath?: string,
    private shouldAnalyzeClues: boolean = false
  ) {}

  /**
   * Main entry point: process a repo (both PRs and commits chronologically)
   */
  async processRepo(owner: string, repo: string): Promise<Usage> {
    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    // Get chronological checkpoint (or migrate from old checkpoints)
    let checkpoint = await this.storage.getChronologicalCheckpoint();

    // Backwards compatibility: migrate old checkpoints if needed
    if (!checkpoint) {
      checkpoint = await this.migrateOldCheckpoint();
    }

    console.log(`\n📋 Fetching changes from ${owner}/${repo}...`);
    const changes = await this.fetchAllChanges(owner, repo, checkpoint);

    if (changes.length === 0) {
      console.log(`   No new changes to process.`);
      const features = await this.storage.getAllFeatures();
      console.log(`\n🎉 Repository processing complete!`);
      console.log(`   Total features: ${features.length}`);
      return totalUsage;
    }

    console.log(`   Processing ${changes.length} changes chronologically...\n`);

    // Process each change in chronological order
    for (let i = 0; i < changes.length; i++) {
      const change = changes[i];
      const progress = `[${i + 1}/${changes.length}]`;

      if (change.type === "pr") {
        const pr = change.data as GitHubPR;
        console.log(`\n${progress} Processing PR #${pr.number}: ${pr.title}`);

        try {
          const usage = await this.processPR(owner, repo, pr);
          totalUsage.inputTokens += usage.inputTokens;
          totalUsage.outputTokens += usage.outputTokens;
          totalUsage.totalTokens += usage.totalTokens;
          console.log(
            `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`
          );
        } catch (error) {
          console.error(
            `   ❌ Error processing PR #${pr.number}:`,
            error instanceof Error ? error.message : error
          );
          console.log(`   ⏭️  Skipping and continuing with next change...`);

          // Save a minimal PR record so we know it was attempted
          await this.storage.savePR({
            number: pr.number,
            title: pr.title,
            summary: `Error during processing: ${
              error instanceof Error ? error.message : "Unknown error"
            }`,
            mergedAt: pr.mergedAt,
            url: pr.url,
            files: pr.filesChanged,
          });
        }
      } else {
        const commit = change.data as {
          sha: string;
          message: string;
          author: string;
          committedAt: Date;
          url: string;
        };
        console.log(
          `\n${progress} Processing commit ${commit.sha.substring(0, 7)}: ${
            commit.message.split("\n")[0]
          }`
        );

        try {
          const usage = await this.processCommit(owner, repo, commit);
          totalUsage.inputTokens += usage.inputTokens;
          totalUsage.outputTokens += usage.outputTokens;
          totalUsage.totalTokens += usage.totalTokens;
          console.log(
            `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`
          );
        } catch (error) {
          console.error(
            `   ❌ Error processing commit ${commit.sha.substring(0, 7)}:`,
            error instanceof Error ? error.message : error
          );
          console.log(`   ⏭️  Skipping and continuing with next change...`);

          // Save a minimal commit record so we know it was attempted
          await this.storage.saveCommit({
            sha: commit.sha,
            message: commit.message,
            summary: `Error during processing: ${
              error instanceof Error ? error.message : "Unknown error"
            }`,
            author: commit.author,
            committedAt: commit.committedAt,
            url: commit.url,
            files: [],
          });
        }
      }

      // Update checkpoint after processing each change
      await this.updateCheckpoint(change.date, change.id);
    }

    // Show final summary
    const features = await this.storage.getAllFeatures();
    console.log(`\n🎉 Repository processing complete!`);
    console.log(`   Total features: ${features.length}`);
    console.log(
      `   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`
    );

    return totalUsage;
  }

  /**
   * Fetch all changes (PRs and commits) chronologically from checkpoint
   */
  private async fetchAllChanges(
    owner: string,
    repo: string,
    checkpoint: ChronologicalCheckpoint | null
  ): Promise<
    Array<{ type: "pr" | "commit"; data: any; date: Date; id: string }>
  > {
    const changes: Array<{
      type: "pr" | "commit";
      data: any;
      date: Date;
      id: string;
    }> = [];

    const sinceDate = checkpoint
      ? new Date(checkpoint.lastProcessedTimestamp)
      : null;
    const processedIds = new Set(checkpoint?.processedAtTimestamp || []);

    console.log(
      `   Fetching changes since ${
        sinceDate ? sinceDate.toISOString() : "beginning"
      }...`
    );

    // Fetch PRs
    console.log(`   📡 Fetching PRs from GitHub API...`);
    const allPRs = await this.octokit.paginate(this.octokit.pulls.list, {
      owner,
      repo,
      state: "closed",
      sort: "created",
      direction: "asc",
      per_page: 100,
    });
    console.log(`   ✓ Fetched ${allPRs.length} total closed PRs`);

    console.log(`   🔍 Filtering for merged PRs after checkpoint...`);
    const mergedPRs = allPRs.filter((pr) => {
      if (!pr.merged_at) return false;
      const mergedDate = new Date(pr.merged_at);

      if (!sinceDate) return true;
      if (mergedDate > sinceDate) return true;
      if (
        mergedDate.getTime() === sinceDate.getTime() &&
        !processedIds.has(pr.number.toString())
      ) {
        return true;
      }
      return false;
    });
    console.log(`   ✓ Found ${mergedPRs.length} merged PRs to process`);

    console.log(`   📝 Adding ${mergedPRs.length} PRs to changes list...`);
    for (const pr of mergedPRs) {
      changes.push({
        type: "pr",
        data: {
          number: pr.number,
          title: pr.title,
          body: pr.body,
          url: pr.html_url,
          mergedAt: new Date(pr.merged_at!),
          additions: 0,
          deletions: 0,
          filesChanged: [],
        },
        date: new Date(pr.merged_at!),
        id: pr.number.toString(),
      });
    }

    // Fetch commits from default branch (use since parameter if we have a checkpoint)
    // Note: When no 'sha' is specified, GitHub API defaults to the repository's default branch
    console.log(`   📡 Fetching commits from default branch...`);
    const commits = await this.octokit.paginate(
      this.octokit.repos.listCommits,
      {
        owner,
        repo,
        per_page: 100,
        ...(sinceDate ? { since: sinceDate.toISOString() } : {}),
      }
    );
    console.log(`   ✓ Fetched ${commits.length} total commits`);

    // Check each commit to see if it's associated with a PR
    console.log(`   🔍 Checking commits for PR associations...`);
    let skippedPRCommits = 0;
    let standaloneCommits = 0;
    let processedCount = 0;
    for (const commit of commits) {
      try {
        // Type guard - ensure commit has expected structure
        if (!commit?.sha || !commit?.commit) continue;

        processedCount++;
        // Log progress every 50 commits
        if (processedCount % 50 === 0) {
          console.log(
            `      Progress: ${processedCount}/${commits.length} commits checked (${standaloneCommits} standalone, ${skippedPRCommits} PR-associated)`
          );
        }

        const { data: prs } =
          await this.octokit.repos.listPullRequestsAssociatedWithCommit({
            owner,
            repo,
            commit_sha: commit.sha,
          });

        const mergedPRs = prs.filter((pr) => pr.merged_at);

        // ONLY process commits that are not associated with any PR
        if (mergedPRs.length === 0) {
          const committedAt = new Date(
            commit.commit.author?.date || Date.now()
          );

          // Filter by checkpoint
          if (sinceDate) {
            if (committedAt < sinceDate) continue;
            if (
              committedAt.getTime() === sinceDate.getTime() &&
              processedIds.has(commit.sha)
            ) {
              continue;
            }
          }

          changes.push({
            type: "commit",
            data: {
              sha: commit.sha,
              message: commit.commit.message,
              author:
                commit.commit.author?.name || commit.author?.login || "Unknown",
              committedAt: committedAt,
              url: commit.html_url,
            },
            date: committedAt,
            id: commit.sha,
          });
          standaloneCommits++;
        } else {
          skippedPRCommits++;
        }
      } catch (error) {
        console.error(
          `   ❌ Error checking PR association for ${commit?.sha}:`,
          error
        );
      }
    }
    console.log(
      `   ✓ Found ${standaloneCommits} standalone commits (skipped ${skippedPRCommits} PR-associated commits)`
    );

    // Sort chronologically (oldest first)
    console.log(`   🔄 Sorting changes chronologically...`);
    changes.sort((a, b) => a.date.getTime() - b.date.getTime());

    console.log(
      `   ✅ Found ${changes.filter((c) => c.type === "pr").length} PRs and ${
        changes.filter((c) => c.type === "commit").length
      } commits to process`
    );

    return changes;
  }

  /**
   * Process a single PR
   */
  private async processPR(
    owner: string,
    repo: string,
    pr: GitHubPR
  ): Promise<Usage> {
    // Skip obvious noise
    if (this.shouldSkip(pr)) {
      console.log(`   ⏭️  Skipped (maintenance/trivial)`);

      // Still save the PR record for completeness
      await this.storage.savePR({
        number: pr.number,
        title: pr.title,
        summary: "Skipped (maintenance/trivial)",
        mergedAt: pr.mergedAt,
        url: pr.url,
        files: pr.filesChanged,
      });
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }

    // Get current features for context
    const features = await this.storage.getAllFeatures();

    // Fetch detailed PR info (additions/deletions/files) - done lazily per PR
    console.log(`   📥 Fetching PR details...`);
    const { data: fullPR } = await this.octokit.pulls.get({
      owner,
      repo,
      pull_number: pr.number,
    });
    pr.additions = fullPR.additions || 0;
    pr.deletions = fullPR.deletions || 0;

    // Fetch full PR content using the existing pr.ts module
    const prContent = await fetchPullRequestContent(
      this.octokit,
      {
        owner,
        repo,
        pull_number: pr.number,
      },
      {
        maxPatchLines: 100, // Reduce to 100 lines per file to save tokens
      }
    );

    // Build decision prompt
    const prompt = await this.buildDecisionPrompt(prContent, features);

    // Ask LLM what to do
    console.log(`   🤖 Asking LLM for decision...`);
    const { decision, usage } = await this.llm.decide(prompt);

    // Apply decision
    await this.applyPrDecision(owner, repo, pr, decision);

    // Analyze for clues if enabled
    if (this.shouldAnalyzeClues) {
      try {
        const changeContext = this.extractPRChangeContext(
          prContent,
          pr,
          decision
        );
        await this.analyzeChangeForClues(changeContext);
      } catch (error) {
        console.error(`   ⚠️  Clue analysis failed:`, error);
        // Continue processing
      }
    }

    return usage;
  }

  /**
   * Quick heuristic filter (no LLM needed)
   */
  private shouldSkip(pr: GitHubPR): boolean {
    const skipPatterns = [
      /^bump/i,
      /^chore:/i,
      /dependabot/i,
      /^docs:/i,
      /typo/i,
      /^ci:/i,
    ];

    return skipPatterns.some((pattern) => pattern.test(pr.title));
  }

  /**
   * Build the decision prompt
   */
  private async buildDecisionPrompt(
    prContent: string,
    features: Feature[]
  ): Promise<string> {
    const themesContext = await this.formatThemeContext();

    return `${SYSTEM_PROMPT}

${this.formatFeatureContext(features)}

${themesContext}

${prContent}

${DECISION_GUIDELINES}`;
  }

  /**
   * Format feature list for context
   */
  private formatFeatureContext(features: Feature[]): string {
    if (features.length === 0) {
      return "## Current Features\n\nNo features yet.";
    }

    const featureList = features
      .sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
      .map((f) => {
        const prCount = f.prNumbers.length;
        const commitCount = (f.commitShas || []).length;
        const changesSummary =
          commitCount > 0
            ? `[${prCount} PRs, ${commitCount} commits]`
            : `[${prCount} PRs]`;
        return `- **${f.name}** (\`${f.id}\`): ${f.description} ${changesSummary}`;
      })
      .join("\n");

    return `## Current Features\n\n${featureList}`;
  }

  /**
   * Format recent themes for context
   */
  private async formatThemeContext(): Promise<string> {
    const themes = await this.storage.getRecentThemes();

    if (themes.length === 0) {
      return "## Recent Technical Themes\n\nNo recent themes.";
    }

    // Show themes in reverse order (most recent first), limit to 100 for display
    const themeList = themes.slice().reverse().slice(0, 100).join(", ");

    return `## Recent Technical Themes (last 100 of ${themes.length})\n\n${themeList}`;
  }

  /**
   * Apply LLM decision for a PR
   */
  private async applyPrDecision(
    owner: string,
    repo: string,
    pr: GitHubPR,
    decision: LLMDecision
  ): Promise<void> {
    // Fetch file list for this PR
    const { data: files } = await this.octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: pr.number,
      per_page: 100,
    });

    // Save PR record
    const prRecord: PRRecord = {
      number: pr.number,
      title: pr.title,
      summary: decision.summary,
      mergedAt: pr.mergedAt,
      url: pr.url,
      files: files.map((f) => f.filename),
      newDeclarations: decision.newDeclarations,
    };
    await this.storage.savePR(prRecord);

    // Apply the decision using shared logic
    await this.applyDecisionToFeatures(decision, {
      changeDate: pr.mergedAt,
      addToFeature: async (feature) => {
        if (!feature.prNumbers.includes(pr.number)) {
          feature.prNumbers.push(pr.number);
          return true;
        }
        return false;
      },
      createFeatureWith: (baseFeature) => ({
        ...baseFeature,
        prNumbers: [pr.number],
        commitShas: [],
      }),
    });
  }

  /**
   * Process a single commit
   */
  private async processCommit(
    owner: string,
    repo: string,
    commit: {
      sha: string;
      message: string;
      author: string;
      committedAt: Date;
      url: string;
    }
  ): Promise<Usage> {
    // Skip obvious noise
    if (this.shouldSkipCommit(commit)) {
      console.log(`   ⏭️  Skipped (maintenance/trivial)`);

      // Still save the commit record for completeness
      await this.storage.saveCommit({
        sha: commit.sha,
        message: commit.message,
        summary: "Skipped (maintenance/trivial)",
        author: commit.author,
        committedAt: commit.committedAt,
        url: commit.url,
        files: [],
      });
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }

    // Get current features for context
    const features = await this.storage.getAllFeatures();

    // Fetch full commit content
    console.log(`   📥 Fetching commit details...`);
    const commitContent = await fetchCommitContent(
      this.octokit,
      {
        owner,
        repo,
        sha: commit.sha,
      },
      {
        maxPatchLines: 100, // Same as PRs
      }
    );

    // Build decision prompt
    const prompt = await this.buildDecisionPrompt(commitContent, features);

    // Ask LLM what to do
    console.log(`   🤖 Asking LLM for decision...`);
    const { decision, usage } = await this.llm.decide(prompt);

    // Apply decision
    await this.applyCommitDecision(owner, repo, commit, decision);

    // Analyze for clues if enabled
    if (this.shouldAnalyzeClues) {
      try {
        const changeContext = this.extractCommitChangeContext(
          commitContent,
          commit,
          decision
        );
        await this.analyzeChangeForClues(changeContext);
      } catch (error) {
        console.error(`   ⚠️  Clue analysis failed:`, error);
        // Continue processing
      }
    }

    return usage;
  }

  /**
   * Quick heuristic filter for commits (same as PRs)
   */
  private shouldSkipCommit(commit: { message: string }): boolean {
    const skipPatterns = [
      /^bump/i,
      /^chore:/i,
      /dependabot/i,
      /^docs:/i,
      /typo/i,
      /^ci:/i,
    ];

    return skipPatterns.some((pattern) => pattern.test(commit.message));
  }

  /**
   * Apply LLM decision for a commit
   */
  private async applyCommitDecision(
    owner: string,
    repo: string,
    commit: {
      sha: string;
      message: string;
      author: string;
      committedAt: Date;
      url: string;
    },
    decision: LLMDecision
  ): Promise<void> {
    // Fetch file list for this commit
    const { data: commitData } = await this.octokit.repos.getCommit({
      owner,
      repo,
      ref: commit.sha,
    });

    const files = commitData.files || [];

    // Save commit record
    const commitRecord: CommitRecord = {
      sha: commit.sha,
      message: commit.message,
      summary: decision.summary,
      author: commit.author,
      committedAt: commit.committedAt,
      url: commit.url,
      files: files.map((f: any) => f.filename),
      newDeclarations: decision.newDeclarations,
    };
    await this.storage.saveCommit(commitRecord);

    // Apply the decision using shared logic
    await this.applyDecisionToFeatures(decision, {
      changeDate: commit.committedAt,
      addToFeature: async (feature) => {
        // Initialize commitShas if it doesn't exist (legacy features)
        if (!feature.commitShas) {
          feature.commitShas = [];
        }
        if (!feature.commitShas.includes(commit.sha)) {
          feature.commitShas.push(commit.sha);
          return true;
        }
        return false;
      },
      createFeatureWith: (baseFeature) => ({
        ...baseFeature,
        prNumbers: [],
        commitShas: [commit.sha],
      }),
    });
  }

  /**
   * Shared logic for applying LLM decision to features
   */
  private async applyDecisionToFeatures(
    decision: LLMDecision,
    config: {
      changeDate: Date;
      addToFeature: (feature: Feature) => Promise<boolean>;
      createFeatureWith: (base: Omit<Feature, 'prNumbers' | 'commitShas'>) => Feature;
    }
  ): Promise<void> {
    console.log(`   📝 Summary: ${decision.summary}`);
    console.log(`   💭 Reasoning: ${decision.reasoning}`);

    // Process each action
    for (const action of decision.actions) {
      if (action === "ignore") {
        console.log(`   ⏭️  Ignored`);
        continue;
      }

      if (action === "add_to_existing") {
        // Add to existing feature(s)
        if (
          decision.existingFeatureIds &&
          decision.existingFeatureIds.length > 0
        ) {
          for (const featureId of decision.existingFeatureIds) {
            const feature = await this.storage.getFeature(featureId);
            if (feature) {
              const wasAdded = await config.addToFeature(feature);
              if (wasAdded) {
                feature.lastUpdated = config.changeDate;
                await this.storage.saveFeature(feature);
                console.log(`   → Added to feature: ${feature.name}`);
              }
            } else {
              console.log(
                `   ⚠️  Warning: Feature ${featureId} not found, skipping`
              );
            }
          }
        }
      }

      if (action === "create_new") {
        // Create new feature(s)
        if (decision.newFeatures && decision.newFeatures.length > 0) {
          for (const newFeatureData of decision.newFeatures) {
            const baseFeature = {
              id: this.generateFeatureId(newFeatureData.name),
              name: newFeatureData.name,
              description: newFeatureData.description,
              createdAt: config.changeDate,
              lastUpdated: config.changeDate,
            };
            const newFeature = config.createFeatureWith(baseFeature);
            await this.storage.saveFeature(newFeature);
            console.log(`   ✨ Created new feature: ${newFeature.name}`);
          }
        }
      }
    }

    // Update feature descriptions
    if (decision.updateFeatures && decision.updateFeatures.length > 0) {
      for (const update of decision.updateFeatures) {
        const feature = await this.storage.getFeature(update.featureId);
        if (feature) {
          feature.description = update.newDescription;
          feature.lastUpdated = config.changeDate;
          await this.storage.saveFeature(feature);
          console.log(`   🔄 Updated feature description: ${feature.name}`);
          console.log(`      ${update.reasoning}`);
        } else {
          console.log(
            `   ⚠️  Warning: Cannot update feature ${update.featureId} - not found`
          );
        }
      }
    }

    // Save themes
    if (decision.themes && decision.themes.length > 0) {
      await this.storage.addThemes(decision.themes);
      console.log(`   🏷️  Tagged: ${decision.themes.join(", ")}`);
    }
  }

  /**
   * Generate slug-style feature ID from name
   */
  private generateFeatureId(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
  }

  /**
   * Migrate old checkpoint format to new chronological checkpoint
   */
  private async migrateOldCheckpoint(): Promise<ChronologicalCheckpoint | null> {
    const lastPR = await this.storage.getLastProcessedPR();
    const lastCommit = await this.storage.getLastProcessedCommit();

    // If no old checkpoints exist, return null (start from beginning)
    if (lastPR === 0 && !lastCommit) {
      return null;
    }

    console.log(`   Migrating old checkpoints to chronological format...`);

    // Get the dates of the last processed PR and commit
    let latestDate: Date | null = null;

    if (lastPR > 0) {
      const pr = await this.storage.getPR(lastPR);
      if (pr) {
        latestDate = pr.mergedAt;
      }
    }

    if (lastCommit) {
      const commit = await this.storage.getCommit(lastCommit);
      if (commit) {
        if (!latestDate || commit.committedAt > latestDate) {
          latestDate = commit.committedAt;
        }
      }
    }

    if (!latestDate) {
      return null;
    }

    // Create new checkpoint from the latest date
    const checkpoint: ChronologicalCheckpoint = {
      lastProcessedTimestamp: latestDate.toISOString(),
      processedAtTimestamp: [], // Don't include the last processed items (they're already done)
    };

    await this.storage.setChronologicalCheckpoint(checkpoint);
    console.log(
      `   Migrated to chronological checkpoint: ${checkpoint.lastProcessedTimestamp}`
    );

    return checkpoint;
  }

  /**
   * Update checkpoint after processing a change
   */
  private async updateCheckpoint(date: Date, id: string): Promise<void> {
    const currentCheckpoint = await this.storage.getChronologicalCheckpoint();
    const dateString = date.toISOString();

    if (
      !currentCheckpoint ||
      currentCheckpoint.lastProcessedTimestamp < dateString
    ) {
      // New timestamp - replace checkpoint
      await this.storage.setChronologicalCheckpoint({
        lastProcessedTimestamp: dateString,
        processedAtTimestamp: [id],
      });
    } else if (currentCheckpoint.lastProcessedTimestamp === dateString) {
      // Same timestamp - add to processedAtTimestamp array
      if (!currentCheckpoint.processedAtTimestamp.includes(id)) {
        currentCheckpoint.processedAtTimestamp.push(id);
        await this.storage.setChronologicalCheckpoint(currentCheckpoint);
      }
    }
    // If date < checkpoint, this is an old item (shouldn't happen, but ignore)
  }

  /**
   * Extract change context from PR for clue analysis
   */
  private extractPRChangeContext(
    prContent: string,
    pr: any,
    decision: LLMDecision
  ): any {
    // Extract comments section from prContent
    const commentsMatch = prContent.match(
      /## Code Review Comments([\s\S]*?)(?=##|$)/
    );
    const comments = commentsMatch ? commentsMatch[1].trim() : undefined;

    // Extract reviews section from prContent
    const reviewsMatch = prContent.match(/## Reviews([\s\S]*?)(?=##|$)/);
    const reviews = reviewsMatch ? reviewsMatch[1].trim() : undefined;

    return {
      type: "pr" as const,
      identifier: `#${pr.number}`,
      title: pr.title,
      summary: decision.summary,
      files: pr.files.map((f: any) => f.filename),
      comments,
      reviews,
    };
  }

  /**
   * Extract change context from commit for clue analysis
   */
  private extractCommitChangeContext(
    commitContent: string,
    commit: any,
    decision: LLMDecision
  ): any {
    return {
      type: "commit" as const,
      identifier: commit.sha.substring(0, 7),
      title: commit.message.split("\n")[0],
      summary: decision.summary,
      files: commit.files.map((f: any) => f.filename),
      // Commits don't have reviews/comments
    };
  }

  /**
   * Analyze change (PR or commit) for clues
   */
  private async analyzeChangeForClues(changeContext: any): Promise<void> {
    console.log(`   💡 Analyzing for clues...`);

    // Initialize clue analyzer if needed
    if (!this.clueAnalyzer) {
      if (!this.repoPath) {
        console.log(`   ⏭️  Skipping clue analysis (no repo path)`);
        return;
      }
      const { ClueAnalyzer } = await import("./clueAnalyzer.js");
      this.clueAnalyzer = new ClueAnalyzer(this.storage, this.repoPath);
    }

    // Analyze change
    const result = await this.clueAnalyzer.analyzeChange(changeContext);

    if (result.clues.length === 0) {
      console.log(`   ℹ️  No new clues found`);
      return;
    }

    console.log(`   ✨ Found ${result.clues.length} clue(s)`);

    // Auto-link clues to relevant features
    const { ClueLinker } = await import("./clueLinker.js");
    const linker = new ClueLinker(this.storage);
    const clueIds = result.clues.map((c: any) => c.id);

    console.log(
      `   🔗 Linking ${clueIds.length} clue(s) to relevant features...`
    );
    await linker.linkClues(clueIds);
  }
}
