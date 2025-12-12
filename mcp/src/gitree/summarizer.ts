import { Storage } from "./store/index.js";
import { callGenerateText } from "../aieo/src/stream.js";
import { Provider } from "../aieo/src/provider.js";
import { Feature, PRRecord, CommitRecord, Usage } from "./types.js";

/**
 * Generates comprehensive documentation for features based on their PR and commit history
 */
export class Summarizer {
  constructor(
    private storage: Storage,
    private provider: Provider,
    private apiKey: string
  ) {}

  /**
   * Generate documentation for a single feature
   */
  async summarizeFeature(featureId: string): Promise<Usage> {
    // Load feature
    const feature = await this.storage.getFeature(featureId);
    if (!feature) {
      throw new Error(`Feature ${featureId} not found`);
    }

    console.log(`\n📝 Summarizing feature: ${feature.name}`);

    // Get all PRs and commits for this feature
    const allPRs = await this.storage.getPRsForFeature(featureId);
    const allCommits = await this.storage.getCommitsForFeature(featureId);

    if (allPRs.length === 0 && allCommits.length === 0) {
      console.log(`   ⚠️  No PRs or commits found for this feature`);
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }

    // Sort PRs and commits chronologically
    const sortedPRs = allPRs.sort((a, b) => a.number - b.number);
    const sortedCommits = allCommits.sort((a, b) => a.committedAt.getTime() - b.committedAt.getTime());

    // Combine and sort chronologically (oldest to newest)
    const combined = [
      ...sortedPRs.map(pr => ({ type: 'pr' as const, data: pr, date: pr.mergedAt })),
      ...sortedCommits.map(commit => ({ type: 'commit' as const, data: commit, date: commit.committedAt }))
    ].sort((a, b) => a.date.getTime() - b.date.getTime());

    // Bookend strategy: First 8 (foundational) + Last 100 (recent) = 108 total
    let selected;
    if (combined.length <= 108) {
      selected = combined; // Use all if under limit
    } else {
      const first8 = combined.slice(0, 8);   // Foundation
      const last100 = combined.slice(-100);  // Current state
      selected = [...first8, ...last100];
    }

    console.log(
      `   Found ${allPRs.length} PRs and ${allCommits.length} commits (using ${selected.length}: ${combined.length <= 108 ? 'all' : 'first 8 + last 100'})`
    );

    // Build prompt with selected changes (in chronological order)
    const isBookended = combined.length > 108;
    const prompt = this.buildSummaryPrompt(feature, selected, isBookended);

    // Generate documentation using LLM
    console.log(`   🤖 Generating documentation...`);
    const result = await callGenerateText({
      provider: this.provider,
      apiKey: this.apiKey,
      prompt,
    });

    const documentation = result.text;

    // Save documentation to feature JSON
    feature.documentation = documentation;
    await this.storage.saveFeature(feature);

    // Save documentation as markdown file
    await this.storage.saveDocumentation(feature.id, documentation);

    console.log(`   ✅ Documentation generated (${documentation.length} chars)`);

    return result.usage;
  }

  /**
   * Generate documentation for specific modified features
   */
  async summarizeModifiedFeatures(featureIds: string[]): Promise<Usage> {
    if (featureIds.length === 0) {
      console.log(`\n⏭️  No features to summarize`);
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }

    console.log(`\n📚 Summarizing ${featureIds.length} modified feature(s)...\n`);

    // Accumulate usage across all features
    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    for (let i = 0; i < featureIds.length; i++) {
      const featureId = featureIds[i];
      const progress = `[${i + 1}/${featureIds.length}]`;

      const feature = await this.storage.getFeature(featureId);
      if (!feature) {
        console.log(`${progress} Feature ${featureId} not found, skipping`);
        continue;
      }

      console.log(`${progress} Processing: ${feature.name} (${feature.id})`);

      try {
        const usage = await this.summarizeFeature(feature.id);
        totalUsage.inputTokens += usage.inputTokens;
        totalUsage.outputTokens += usage.outputTokens;
        totalUsage.totalTokens += usage.totalTokens;
        console.log(
          `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`
        );
      } catch (error) {
        console.error(
          `   ❌ Error:`,
          error instanceof Error ? error.message : error
        );
        console.log(`   ⏭️  Skipping and continuing...`);
      }
    }

    console.log(`\n✅ Done summarizing modified features!`);

    return totalUsage;
  }

  /**
   * Generate documentation for all features
   */
  async summarizeAllFeatures(): Promise<Usage> {
    const features = await this.storage.getAllFeatures();

    console.log(`\n📚 Summarizing ${features.length} features...\n`);

    // Accumulate usage across all features
    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const progress = `[${i + 1}/${features.length}]`;

      console.log(`${progress} Processing: ${feature.name} (${feature.id})`);

      try {
        const usage = await this.summarizeFeature(feature.id);
        totalUsage.inputTokens += usage.inputTokens;
        totalUsage.outputTokens += usage.outputTokens;
        totalUsage.totalTokens += usage.totalTokens;
        console.log(
          `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`
        );
      } catch (error) {
        console.error(
          `   ❌ Error:`,
          error instanceof Error ? error.message : error
        );
        console.log(`   ⏭️  Skipping and continuing...`);
      }
    }

    console.log(`\n✅ Done summarizing all features!`);

    return totalUsage;
  }

  /**
   * Build the prompt for generating documentation
   */
  private buildSummaryPrompt(
    feature: Feature,
    selected: Array<{ type: 'pr' | 'commit', data: PRRecord | CommitRecord, date: Date }>,
    isBookended: boolean
  ): string {
    // Format changes in chronological order
    const formattedChanges = selected.map((item, index) => {
      const content = item.type === 'pr'
        ? this.formatPRForSummary(item.data as PRRecord)
        : this.formatCommitForSummary(item.data as CommitRecord);

      // Add section marker after first 8 if bookended
      if (isBookended && index === 7) {
        return content + '\n\n---\n**[NOTE: Gap in history - continuing with most recent 100 changes]**\n---';
      }

      return content;
    });

    const changesText = formattedChanges.join("\n\n");
    const totalChanges = selected.length;

    const prs = selected.filter(c => c.type === 'pr');
    const commits = selected.filter(c => c.type === 'commit');

    return `You are generating SUCCINCT documentation for a software feature to help developers quickly understand and continue working on it.

**Feature**: ${feature.name}
**ID**: ${feature.id}
**Description**: ${feature.description}
**Total changes in history**: ${totalChanges} (${prs.length} PRs, ${commits.length} commits)

Below is ${isBookended ? 'the FOUNDATIONAL (first 8) and RECENT (last 100) changes' : 'the COMPLETE chronological history'} (PRs and commits) that built this feature (from oldest to newest):
${isBookended ? '\n**NOTE**: The first 8 changes show initial architecture/foundation. After a gap, the remaining changes show the recent state.\n' : ''}
${changesText}

---

**Your task**: Generate HIGH-LEVEL documentation for the CURRENT state of this feature.

**CRITICAL REQUIREMENTS**:
1. **Be SUCCINCT** - Target length: 100-200 lines MAXIMUM
2. **NO code snippets** - Focus on concepts, not implementation details
3. **High-level only** - What it does, not how it's coded
4. **Actionable** - What developers need to know to work on this feature
5. **Focus on CURRENT state** - Ignore historical implementation details

**What to include**:
- Brief overview (2-3 sentences max)
- List the 5-15 core files (just paths and 1-line purposes)
- Key concepts/components (high-level only)
- Main API endpoints/functions (names only, no implementations)
- Core data models (names only, brief purpose)

**What to AVOID**:
- Long explanations of how things work internally
- Code snippets or implementation details
- Historical information about how it evolved
- Detailed API documentation
- Step-by-step flows unless absolutely essential

Generate the documentation in markdown format:`;
  }

  /**
   * Format a PR for the summary prompt (lighter weight than full markdown)
   */
  private formatPRForSummary(pr: PRRecord): string {
    let content = `### PR #${pr.number}: ${pr.title}
**Merged**: ${pr.mergedAt.toISOString().split("T")[0]}
**Summary**: ${pr.summary}`;

    // Include files if reasonable number
    if (pr.files.length > 0 && pr.files.length <= 30) {
      content += `\n**Files changed**: ${pr.files.join(", ")}`;
    } else if (pr.files.length > 30) {
      content += `\n**Files changed**: ${pr.files.slice(0, 20).join(", ")} ... (${pr.files.length - 20} more)`;
    }

    // Include new declarations if any
    if (pr.newDeclarations && pr.newDeclarations.length > 0) {
      const declSummary = pr.newDeclarations
        .map((d) => `${d.file}: ${d.declarations.join(", ")}`)
        .join("; ");
      content += `\n**New declarations**: ${declSummary}`;
    }

    return content;
  }

  /**
   * Format a commit for the summary prompt (lighter weight than full markdown)
   */
  private formatCommitForSummary(commit: CommitRecord): string {
    let content = `### Commit ${commit.sha.substring(0, 7)}: ${commit.message.split('\n')[0]}
**Author**: ${commit.author}
**Committed**: ${commit.committedAt.toISOString().split("T")[0]}
**Summary**: ${commit.summary}`;

    // Include files if reasonable number
    if (commit.files.length > 0 && commit.files.length <= 30) {
      content += `\n**Files changed**: ${commit.files.join(", ")}`;
    } else if (commit.files.length > 30) {
      content += `\n**Files changed**: ${commit.files.slice(0, 20).join(", ")} ... (${commit.files.length - 20} more)`;
    }

    // Include new declarations if any
    if (commit.newDeclarations && commit.newDeclarations.length > 0) {
      const declSummary = commit.newDeclarations
        .map((d) => `${d.file}: ${d.declarations.join(", ")}`)
        .join("; ");
      content += `\n**New declarations**: ${declSummary}`;
    }

    return content;
  }
}
