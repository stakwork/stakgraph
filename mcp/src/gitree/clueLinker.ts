import { Storage } from "./store/index.js";
import { Clue, Usage } from "./types.js";
import { get_context } from "../repo/agent.js";

/**
 * Links clues to relevant features based on semantic similarity and context
 */
export class ClueLinker {
  constructor(
    private storage: Storage,
    private repoPath: string
  ) {}

  /**
   * Link all clues to relevant features
   */
  async linkAllClues(force: boolean = false): Promise<Usage> {
    const features = await this.storage.getAllFeatures();
    const allClues = await this.storage.getAllClues();

    console.log(`\n🔗 Linking ${allClues.length} clues to ${features.length} features...\n`);

    const totalUsage: Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };

    // Process in batches of 20 clues
    const batchSize = 20;
    for (let i = 0; i < allClues.length; i += batchSize) {
      const batch = allClues.slice(i, i + batchSize);
      const progress = `[${i + 1}-${Math.min(i + batchSize, allClues.length)}/${allClues.length}]`;

      console.log(`${progress} Processing batch...`);

      try {
        const result = await this.linkClueBatch(batch, features, force);
        totalUsage.inputTokens += result.usage.inputTokens;
        totalUsage.outputTokens += result.usage.outputTokens;
        totalUsage.totalTokens += result.usage.totalTokens;
      } catch (error) {
        console.error(`   ❌ Error:`, error instanceof Error ? error.message : error);
        console.log(`   ⏭️  Skipping batch and continuing...`);
      }
    }

    console.log(`\n✅ Done linking all clues!`);
    console.log(`   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`);

    return totalUsage;
  }

  /**
   * Link a batch of clues to relevant features
   */
  private async linkClueBatch(
    clues: Clue[],
    features: any[],
    force: boolean
  ): Promise<{ usage: Usage }> {
    // Skip clues that already have multiple links (unless force)
    const cluesToLink = force
      ? clues
      : clues.filter((c) => c.relatedFeatures.length <= 1);

    if (cluesToLink.length === 0) {
      console.log(`   ⏭️  All clues already linked, skipping...`);
      return {
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      };
    }

    const prompt = this.buildLinkingPrompt(cluesToLink, features);
    const schema = this.buildLinkingSchema(cluesToLink, features);

    console.log(`   🤖 Analyzing ${cluesToLink.length} clues for relevance...`);

    const result = await get_context(prompt, this.repoPath, {
      schema,
      systemOverride: this.buildSystemPrompt(),
    });

    const decision = result.content as any;

    // Update clues with new links
    for (const link of decision.links || []) {
      const clue = clues.find((c) => c.id === link.clueId);
      if (!clue) continue;

      // Merge with existing relatedFeatures (keep discovering feature)
      const newFeatures = new Set([
        clue.featureId, // Always keep discovering feature
        ...link.featureIds,
      ]);

      clue.relatedFeatures = Array.from(newFeatures);
      clue.updatedAt = new Date();

      await this.storage.saveClue(clue);

      console.log(
        `   🔗 Linked "${clue.title}" to ${clue.relatedFeatures.length} feature(s)`
      );
    }

    return { usage: result.usage };
  }

  /**
   * Build the linking prompt
   */
  private buildLinkingPrompt(clues: Clue[], features: any[]): string {
    const cluesList = clues
      .map(
        (c) =>
          `  - ${c.id}: "${c.title}" [${c.type}]\n` +
          `    Discovered in: ${c.featureId}\n` +
          `    Content: ${c.content.substring(0, 150)}...\n` +
          `    Keywords: ${c.keywords.slice(0, 5).join(", ")}\n` +
          `    Entities: ${Object.entries(c.entities)
            .map(([type, values]) =>
              values && values.length > 0
                ? `${type}=[${values.slice(0, 3).join(", ")}]`
                : ""
            )
            .filter(Boolean)
            .join(", ")}`
      )
      .join("\n\n");

    const featuresList = features
      .map(
        (f) =>
          `  - ${f.id}: "${f.name}"\n` +
          `    ${f.description}\n` +
          (f.documentation
            ? `    ${f.documentation.substring(0, 200)}...\n`
            : "")
      )
      .join("\n");

    return `Analyze which features each clue is relevant to, based on semantic similarity and usage context.

**Clues** (${clues.length}):
${cluesList}

**Features** (${features.length}):
${featuresList}

**Your Task**:
For each clue, determine which features it's relevant to. A clue is relevant if:
1. The pattern/utility/abstraction is used by that feature
2. The clue's entities (functions, classes, types) appear in the feature's files
3. The clue's keywords match the feature's domain
4. The architectural concept applies to the feature

**Guidelines**:
- Cross-cutting concerns (auth, logging, error handling) are relevant to many features
- Feature-specific implementations are relevant to 1-2 features
- General utilities/patterns may be relevant to 3-5 features
- Don't over-link - only include truly relevant features
- Always include the discovering feature (featureId) in the list

Return your analysis.`;
  }

  /**
   * Build the linking schema
   */
  private buildLinkingSchema(clues: Clue[], features: any[]): any {
    const clueIds = clues.map((c) => c.id);
    const featureIds = features.map((f) => f.id);

    return {
      type: "object",
      properties: {
        links: {
          type: "array",
          items: {
            type: "object",
            properties: {
              clueId: {
                type: "string",
                enum: clueIds,
              },
              featureIds: {
                type: "array",
                items: {
                  type: "string",
                  enum: featureIds,
                },
              },
              reasoning: {
                type: "string",
              },
            },
            required: ["clueId", "featureIds", "reasoning"],
          },
        },
      },
      required: ["links"],
    };
  }

  /**
   * Build the system prompt
   */
  private buildSystemPrompt(): string {
    return `You are a codebase architecture analyzer that links architectural patterns and utilities to the features they're relevant to.

Your goal is to create RELEVANT_TO relationships between Clues and Features based on:
1. Entity usage - which features use the clue's functions/classes/types
2. Semantic similarity - keywords and concepts match
3. Architectural applicability - the pattern applies to the feature's domain
4. File overlap - clue files intersect with feature files

**Linking Principles:**
- Cross-cutting concerns (authentication, logging, error handling, testing) → many features
- Domain-specific utilities (payment processing, email templates) → 1-3 related features
- Generic patterns (state management, data flow) → 3-5 features where pattern is used
- Feature-specific implementations → 1-2 features only

**Quality over quantity:**
- Only link clues that are genuinely useful for understanding/working with that feature
- Don't over-link - specificity is valuable
- Always include the discovering feature in the links

Analyze the clues and features, then return the relevance links.`;
  }
}
