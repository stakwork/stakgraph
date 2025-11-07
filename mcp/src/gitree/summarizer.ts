import { Storage } from "./store/index.js";
import { callGenerateText } from "../aieo/src/stream.js";
import { Provider } from "../aieo/src/provider.js";
import { Feature, PRRecord } from "./types.js";

/**
 * Generates comprehensive documentation for features based on their PR history
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
  async summarizeFeature(featureId: string): Promise<void> {
    // Load feature
    const feature = await this.storage.getFeature(featureId);
    if (!feature) {
      throw new Error(`Feature ${featureId} not found`);
    }

    console.log(`\n📝 Summarizing feature: ${feature.name}`);

    // Get all PRs for this feature
    const allPRs = await this.storage.getPRsForFeature(featureId);

    if (allPRs.length === 0) {
      console.log(`   ⚠️  No PRs found for this feature`);
      return;
    }

    // Sort chronologically and take last 100 if more than 100
    const sortedPRs = allPRs.sort((a, b) => a.number - b.number);
    const prs = sortedPRs.slice(-100);

    console.log(
      `   Found ${allPRs.length} PRs (using ${prs.length} most recent)`
    );

    // Build prompt with all PR content
    const prompt = this.buildSummaryPrompt(feature, prs);

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
  }

  /**
   * Generate documentation for all features
   */
  async summarizeAllFeatures(): Promise<void> {
    const features = await this.storage.getAllFeatures();

    console.log(`\n📚 Summarizing ${features.length} features...\n`);

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const progress = `[${i + 1}/${features.length}]`;

      console.log(`${progress} Processing: ${feature.name} (${feature.id})`);

      try {
        await this.summarizeFeature(feature.id);
      } catch (error) {
        console.error(
          `   ❌ Error:`,
          error instanceof Error ? error.message : error
        );
        console.log(`   ⏭️  Skipping and continuing...`);
      }
    }

    console.log(`\n✅ Done summarizing all features!`);
  }

  /**
   * Build the prompt for generating documentation
   */
  private buildSummaryPrompt(feature: Feature, prs: PRRecord[]): string {
    // Format all PRs in a concise format
    const prContents = prs.map((pr) => this.formatPRForSummary(pr)).join("\n\n");

    return `You are generating SUCCINCT documentation for a software feature to help developers quickly understand and continue working on it.

**Feature**: ${feature.name}
**ID**: ${feature.id}
**Description**: ${feature.description}
**Total PRs in history**: ${prs.length}

Below is the COMPLETE chronological history of PRs that built this feature (from oldest to newest):

${prContents}

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
}
