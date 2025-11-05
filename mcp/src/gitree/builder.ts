import { Octokit } from "@octokit/rest";
import { Storage } from "./storage.js";
import { LLMClient, SYSTEM_PROMPT, DECISION_FORMAT } from "./llm.js";
import { Feature, PRRecord, LLMDecision, GitHubPR } from "./types.js";
import { fetchPullRequestContent } from "./pr.js";

/**
 * Main class for building the feature knowledge base from PRs
 */
export class StreamingFeatureBuilder {
  constructor(
    private storage: Storage,
    private llm: LLMClient,
    private octokit: Octokit
  ) {}

  /**
   * Main entry point: process a repo
   */
  async processRepo(owner: string, repo: string): Promise<void> {
    const lastProcessed = await this.storage.getLastProcessedPR();

    console.log(`Fetching PRs from ${owner}/${repo}...`);
    const prs = await this.fetchPRs(owner, repo, lastProcessed);

    console.log(
      `Processing ${prs.length} PRs starting from #${lastProcessed + 1}...`
    );

    for (const pr of prs) {
      await this.processPR(owner, repo, pr);
      await this.storage.setLastProcessedPR(pr.number);
      console.log(`✅ Processed PR #${pr.number}`);
    }

    const features = await this.storage.getAllFeatures();
    console.log(`\nDone! Total features: ${features.length}`);
  }

  /**
   * Fetch PRs from GitHub
   */
  private async fetchPRs(
    owner: string,
    repo: string,
    since: number
  ): Promise<GitHubPR[]> {
    const { data } = await this.octokit.pulls.list({
      owner,
      repo,
      state: "closed",
      sort: "created",
      direction: "asc",
      per_page: 100,
    });

    // Filter to only merged PRs after the last processed
    const mergedPRs = data.filter(
      (pr) => pr.merged_at && pr.number > since
    );

    // Convert to our GitHubPR type
    const gitHubPRs: GitHubPR[] = [];
    for (const pr of mergedPRs) {
      // Fetch full PR details to get additions/deletions
      const { data: fullPR } = await this.octokit.pulls.get({
        owner,
        repo,
        pull_number: pr.number,
      });

      // Fetch files for this PR
      const { data: files } = await this.octokit.pulls.listFiles({
        owner,
        repo,
        pull_number: pr.number,
        per_page: 100,
      });

      gitHubPRs.push({
        number: fullPR.number,
        title: fullPR.title,
        body: fullPR.body,
        url: fullPR.html_url,
        mergedAt: new Date(fullPR.merged_at!),
        additions: fullPR.additions || 0,
        deletions: fullPR.deletions || 0,
        filesChanged: files.map((f) => f.filename),
      });
    }

    return gitHubPRs;
  }

  /**
   * Process a single PR
   */
  private async processPR(
    owner: string,
    repo: string,
    pr: GitHubPR
  ): Promise<void> {
    // Skip obvious noise
    if (this.shouldSkip(pr)) {
      console.log(`⏭️  Skipping #${pr.number}: ${pr.title}`);

      // Still save the PR record for completeness
      await this.storage.savePR({
        number: pr.number,
        title: pr.title,
        summary: "Skipped (maintenance/trivial)",
        mergedAt: pr.mergedAt,
        url: pr.url,
      });
      return;
    }

    console.log(`\n🔍 Analyzing PR #${pr.number}: ${pr.title}`);

    // Get current features for context
    const features = await this.storage.getAllFeatures();

    // Fetch full PR content using the existing pr.ts module
    const prContent = await fetchPullRequestContent(this.octokit, {
      owner,
      repo,
      pull_number: pr.number,
    });

    // Build decision prompt
    const prompt = this.buildDecisionPrompt(prContent, features);

    // Ask LLM what to do
    console.log(`   🤖 Asking LLM for decision...`);
    const decision = await this.llm.decide(prompt);

    // Apply decision
    await this.applyDecision(pr, decision);
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
  private buildDecisionPrompt(prContent: string, features: Feature[]): string {
    return `${SYSTEM_PROMPT}

${this.formatFeatureContext(features)}

${prContent}

${DECISION_FORMAT}`;
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
      .map(
        (f) =>
          `- **${f.name}** (\`${f.id}\`): ${f.description} [${f.prNumbers.length} PRs]`
      )
      .join("\n");

    return `## Current Features\n\n${featureList}`;
  }

  /**
   * Apply LLM decision
   */
  private async applyDecision(
    pr: GitHubPR,
    decision: LLMDecision
  ): Promise<void> {
    // Save PR record
    const prRecord: PRRecord = {
      number: pr.number,
      title: pr.title,
      summary: decision.summary,
      mergedAt: pr.mergedAt,
      url: pr.url,
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
              createdAt: pr.mergedAt,
              lastUpdated: pr.mergedAt,
            };
            await this.storage.saveFeature(newFeature);
            console.log(`   ✨ Created new feature: ${newFeature.name}`);
          }
        }
      }
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
