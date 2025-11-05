import { Octokit } from "@octokit/rest";
import { Storage } from "./storage.js";
import { LLMClient, SYSTEM_PROMPT, DECISION_GUIDELINES } from "./llm.js";
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

    if (prs.length === 0) {
      console.log(`No new PRs to process.`);
      return;
    }

    console.log(
      `Processing ${prs.length} PRs starting from #${lastProcessed + 1}...\n`
    );

    for (let i = 0; i < prs.length; i++) {
      const pr = prs[i];
      const progress = `[${i + 1}/${prs.length}]`;
      console.log(`\n${progress} Processing PR #${pr.number}: ${pr.title}`);

      try {
        await this.processPR(owner, repo, pr);
      } catch (error) {
        console.error(`   ❌ Error processing PR #${pr.number}:`, error instanceof Error ? error.message : error);
        console.log(`   ⏭️  Skipping and continuing with next PR...`);

        // Save a minimal PR record so we know it was attempted
        await this.storage.savePR({
          number: pr.number,
          title: pr.title,
          summary: `Error during processing: ${error instanceof Error ? error.message : 'Unknown error'}`,
          mergedAt: pr.mergedAt,
          url: pr.url,
        });
      }

      await this.storage.setLastProcessedPR(pr.number);
    }

    const features = await this.storage.getAllFeatures();
    console.log(`\n✅ Done! Total features: ${features.length}`);
  }

  /**
   * Fetch PRs from GitHub (with pagination)
   */
  private async fetchPRs(
    owner: string,
    repo: string,
    since: number
  ): Promise<GitHubPR[]> {
    console.log(`   Fetching all PRs (paginated)...`);

    // Use octokit pagination to get ALL PRs
    const allPRs = await this.octokit.paginate(
      this.octokit.pulls.list,
      {
        owner,
        repo,
        state: "closed",
        sort: "created",
        direction: "asc",
        per_page: 100,
      }
    );

    console.log(`   Found ${allPRs.length} closed PRs total`);

    // Filter to only merged PRs after the last processed
    const mergedPRs = allPRs.filter(
      (pr) => pr.merged_at && pr.number > since
    );

    console.log(`   ${mergedPRs.length} merged PRs to process (after #${since})`);

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
      console.log(`   ⏭️  Skipped (maintenance/trivial)`);

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

    // Get current features for context
    const features = await this.storage.getAllFeatures();

    // Fetch full PR content using the existing pr.ts module
    console.log(`   📥 Fetching PR content...`);
    const prContent = await fetchPullRequestContent(this.octokit, {
      owner,
      repo,
      pull_number: pr.number,
    }, {
      maxPatchLines: 100, // Reduce to 100 lines per file to save tokens
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
