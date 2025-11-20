import { Octokit } from "@octokit/rest";
import { Storage } from "./store/index.js";
import { LLMClient, SYSTEM_PROMPT, DECISION_GUIDELINES } from "./llm.js";
import { Feature, PRRecord, CommitRecord, LLMDecision, GitHubPR, Usage } from "./types.js";
import { fetchPullRequestContent } from "./pr.js";
import { fetchCommitContent, fetchOrphanCommits } from "./commit.js";

/**
 * Main class for building the feature knowledge base from PRs and commits
 */
export class StreamingFeatureBuilder {
  constructor(
    private storage: Storage,
    private llm: LLMClient,
    private octokit: Octokit
  ) {}

  /**
   * Main entry point: process a repo (both PRs and commits)
   */
  async processRepo(owner: string, repo: string): Promise<Usage> {
    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    // Process PRs first
    const prUsage = await this.processPRs(owner, repo);
    totalUsage.inputTokens += prUsage.inputTokens;
    totalUsage.outputTokens += prUsage.outputTokens;
    totalUsage.totalTokens += prUsage.totalTokens;

    // Process orphan commits
    const commitUsage = await this.processCommits(owner, repo);
    totalUsage.inputTokens += commitUsage.inputTokens;
    totalUsage.outputTokens += commitUsage.outputTokens;
    totalUsage.totalTokens += commitUsage.totalTokens;

    // Show final summary
    const features = await this.storage.getAllFeatures();
    console.log(`\n🎉 Repository processing complete!`);
    console.log(`   Total features: ${features.length}`);
    console.log(`   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`);

    return totalUsage;
  }

  /**
   * Process PRs for a repo
   */
  async processPRs(owner: string, repo: string): Promise<Usage> {
    const lastProcessed = await this.storage.getLastProcessedPR();

    console.log(`\n📋 Fetching PRs from ${owner}/${repo}...`);
    const prs = await this.fetchPRs(owner, repo, lastProcessed);

    if (prs.length === 0) {
      console.log(`   No new PRs to process.`);
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }

    console.log(
      `   Processing ${prs.length} PRs starting from #${lastProcessed + 1}...\n`
    );

    // Accumulate usage across all PRs
    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    for (let i = 0; i < prs.length; i++) {
      const pr = prs[i];
      const progress = `[${i + 1}/${prs.length}]`;
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
        console.log(`   ⏭️  Skipping and continuing with next PR...`);

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

      await this.storage.setLastProcessedPR(pr.number);
    }

    const features = await this.storage.getAllFeatures();
    console.log(`\n✅ PRs done! Total features: ${features.length}`);

    return totalUsage;
  }

  /**
   * Fetch PRs from GitHub (with pagination) - lightweight, just the list
   */
  private async fetchPRs(
    owner: string,
    repo: string,
    since: number
  ): Promise<GitHubPR[]> {
    console.log(`   Fetching PR list (paginated)...`);

    // Use octokit pagination to get ALL PRs
    const allPRs = await this.octokit.paginate(this.octokit.pulls.list, {
      owner,
      repo,
      state: "closed",
      sort: "created",
      direction: "asc",
      per_page: 100,
    });

    console.log(`   Found ${allPRs.length} closed PRs total`);

    // Filter to only merged PRs after the last processed
    const mergedPRs = allPRs.filter((pr) => pr.merged_at && pr.number > since);

    console.log(
      `   ${mergedPRs.length} merged PRs to process (after #${since})`
    );

    // Convert to lightweight GitHubPR type (detailed info fetched when processing)
    return mergedPRs.map((pr) => ({
      number: pr.number,
      title: pr.title,
      body: pr.body,
      url: pr.html_url,
      mergedAt: new Date(pr.merged_at!),
      additions: 0, // Will fetch when processing
      deletions: 0, // Will fetch when processing
      filesChanged: [], // Will fetch when processing
    }));
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
    await this.applyDecision(owner, repo, pr, decision);

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
        const commitCount = f.commitShas.length;
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
   * Apply LLM decision
   */
  private async applyDecision(
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
              if (!feature.prNumbers.includes(pr.number)) {
                feature.prNumbers.push(pr.number);
                feature.lastUpdated = pr.mergedAt;
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
            const newFeature: Feature = {
              id: this.generateFeatureId(newFeatureData.name),
              name: newFeatureData.name,
              description: newFeatureData.description,
              prNumbers: [pr.number],
              commitShas: [],
              createdAt: pr.mergedAt,
              lastUpdated: pr.mergedAt,
            };
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
          feature.lastUpdated = pr.mergedAt;
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
   * Process commits not associated with PRs
   */
  async processCommits(owner: string, repo: string): Promise<Usage> {
    const lastProcessed = await this.storage.getLastProcessedCommit();

    console.log(`\n💾 Fetching orphan commits from ${owner}/${repo}...`);
    const commits = await fetchOrphanCommits(
      this.octokit,
      owner,
      repo,
      lastProcessed || undefined
    );

    if (commits.length === 0) {
      console.log(`   No new orphan commits to process.`);
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }

    console.log(
      `   Processing ${commits.length} orphan commits${
        lastProcessed ? ` starting after ${lastProcessed.substring(0, 7)}` : ""
      }...\n`
    );

    // Accumulate usage across all commits
    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    for (let i = 0; i < commits.length; i++) {
      const commit = commits[i];
      const progress = `[${i + 1}/${commits.length}]`;
      console.log(
        `\n${progress} Processing commit ${commit.sha.substring(0, 7)}: ${commit.message.split('\n')[0]}`
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
        console.log(`   ⏭️  Skipping and continuing with next commit...`);

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

      await this.storage.setLastProcessedCommit(commit.sha);
    }

    console.log(`\n✅ Commits done!`);

    return totalUsage;
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
              if (!feature.commitShas.includes(commit.sha)) {
                feature.commitShas.push(commit.sha);
                feature.lastUpdated = commit.committedAt;
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
            const newFeature: Feature = {
              id: this.generateFeatureId(newFeatureData.name),
              name: newFeatureData.name,
              description: newFeatureData.description,
              prNumbers: [],
              commitShas: [commit.sha],
              createdAt: commit.committedAt,
              lastUpdated: commit.committedAt,
            };
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
          feature.lastUpdated = commit.committedAt;
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
}
