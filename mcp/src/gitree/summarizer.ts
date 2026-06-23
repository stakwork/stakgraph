import { Storage } from "./store/index.js";
import { callGenerateText } from "../aieo/src/stream.js";
import { Provider } from "../aieo/src/provider.js";
import { addUsage, normalizeUsage } from "../aieo/src/usage.js";
import { Concept, PRRecord, CommitRecord, Usage } from "./types.js";
import {
  appendGitreeLlmExchange,
  DOC_GUIDELINES,
  GitreeSessionTracker,
} from "./llm.js";
import { appendMessages } from "../repo/session.js";

/**
 * Generates comprehensive documentation for concepts based on their PR and commit history
 */
export class Summarizer {
  constructor(
    private storage: Storage,
    private provider: Provider,
    private apiKey: string,
    private sessionTracker?: GitreeSessionTracker,
  ) {}

  /**
   * Generate documentation for a single concept
   */
  async summarizeConcept(
    conceptId: string,
    sessionId?: string,
  ): Promise<Usage> {
    // Load concept
    const concept = await this.storage.getConcept(conceptId);
    if (!concept) {
      throw new Error(`Concept ${conceptId} not found`);
    }

    console.log(`\n📝 Summarizing concept: ${concept.name}`);

    // Get all PRs and commits for this concept
    const allPRs = await this.storage.getPRsForConcept(conceptId);
    const allCommits = await this.storage.getCommitsForConcept(conceptId);

    if (allPRs.length === 0 && allCommits.length === 0) {
      console.log(`   ⚠️  No PRs or commits found for this concept`);
      return normalizeUsage();
    }

    // Sort PRs and commits chronologically
    const sortedPRs = allPRs.sort((a, b) => a.number - b.number);
    const sortedCommits = allCommits.sort(
      (a, b) => a.committedAt.getTime() - b.committedAt.getTime(),
    );

    // Combine and sort chronologically (oldest to newest)
    const combined = [
      ...sortedPRs.map((pr) => ({
        type: "pr" as const,
        data: pr,
        date: pr.mergedAt,
      })),
      ...sortedCommits.map((commit) => ({
        type: "commit" as const,
        data: commit,
        date: commit.committedAt,
      })),
    ].sort((a, b) => a.date.getTime() - b.date.getTime());

    // Bookend strategy: First 8 (foundational) + Last 100 (recent) = 108 total
    let selected;
    if (combined.length <= 108) {
      selected = combined; // Use all if under limit
    } else {
      const first8 = combined.slice(0, 8); // Foundation
      const last100 = combined.slice(-100); // Current state
      selected = [...first8, ...last100];
    }

    console.log(
      `   Found ${allPRs.length} PRs and ${allCommits.length} commits (using ${selected.length}: ${combined.length <= 108 ? "all" : "first 8 + last 100"})`,
    );

    // Build prompt with selected changes (in chronological order)
    const isBookended = combined.length > 108;
    const prompt = this.buildSummaryPrompt(concept, selected, isBookended);

    // Generate documentation using LLM
    console.log(`   🤖 Generating documentation...`);
    const result = await callGenerateText({
      provider: this.provider,
      apiKey: this.apiKey,
      prompt,
    });

    if (this.sessionTracker) {
      appendGitreeLlmExchange(
        this.sessionTracker,
        prompt,
        result.text,
        result.usage,
        `gitree summary: ${concept.name}`,
      );
    } else if (sessionId) {
      appendMessages(sessionId, [
        { role: "user", content: prompt },
        { role: "assistant", content: result.text },
      ]);
    }

    const documentation = result.text.trim();

    // LLM responded "OK" — existing docs are still accurate, no update needed
    if (
      documentation.toUpperCase() === "OK" ||
      documentation.toUpperCase() === "\`OK\`"
    ) {
      console.log(`   ⏭️  No doc update needed (existing docs are current)`);
      return result.usage;
    }

    // Save documentation and usage to concept
    concept.documentation = documentation;
    concept.usage = result.usage;
    await this.storage.saveConcept(concept);

    // Save documentation as markdown file
    await this.storage.saveDocumentation(concept.id, documentation);

    console.log(
      `   ✅ Documentation generated (${documentation.length} chars)`,
    );

    return result.usage;
  }

  /**
   * Generate documentation for specific modified concepts
   */
  async summarizeModifiedConcepts(
    conceptIds: string[],
    sessionId?: string,
  ): Promise<Usage> {
    if (conceptIds.length === 0) {
      console.log(`\n⏭️  No concepts to summarize`);
      return normalizeUsage();
    }

    console.log(
      `\n📚 Summarizing ${conceptIds.length} modified concept(s)...\n`,
    );

    // Accumulate usage across all concepts
    let totalUsage: Usage = normalizeUsage();

    for (let i = 0; i < conceptIds.length; i++) {
      const conceptId = conceptIds[i];
      const progress = `[${i + 1}/${conceptIds.length}]`;

      const concept = await this.storage.getConcept(conceptId);
      if (!concept) {
        console.log(`${progress} Concept ${conceptId} not found, skipping`);
        continue;
      }

      console.log(`${progress} Processing: ${concept.name} (${concept.id})`);

      try {
        const usage = await this.summarizeConcept(concept.id, sessionId);
        totalUsage = normalizeUsage(addUsage(totalUsage, usage));
        console.log(
          `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`,
        );
      } catch (error) {
        console.error(
          `   ❌ Error:`,
          error instanceof Error ? error.message : error,
        );
        console.log(`   ⏭️  Skipping and continuing...`);
      }
    }

    console.log(`\n✅ Done summarizing modified concepts!`);

    return totalUsage;
  }

  /**
   * Generate documentation for all concepts
   * @param repo - Optional repo to filter concepts
   */
  async summarizeAllConcepts(repo?: string): Promise<Usage> {
    const concepts = await this.storage.getAllConcepts(repo);

    console.log(`\n📚 Summarizing ${concepts.length} concepts...\n`);

    // Accumulate usage across all concepts
    let totalUsage: Usage = normalizeUsage();

    for (let i = 0; i < concepts.length; i++) {
      const concept = concepts[i];
      const progress = `[${i + 1}/${concepts.length}]`;

      console.log(`${progress} Processing: ${concept.name} (${concept.id})`);

      try {
        const usage = await this.summarizeConcept(concept.id);
        totalUsage = normalizeUsage(addUsage(totalUsage, usage));
        console.log(
          `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`,
        );
      } catch (error) {
        console.error(
          `   ❌ Error:`,
          error instanceof Error ? error.message : error,
        );
        console.log(`   ⏭️  Skipping and continuing...`);
      }
    }

    console.log(`\n✅ Done summarizing all concepts!`);

    return totalUsage;
  }

  /**
   * Build the prompt for generating documentation
   */
  private buildSummaryPrompt(
    concept: Concept,
    selected: Array<{
      type: "pr" | "commit";
      data: PRRecord | CommitRecord;
      date: Date;
    }>,
    isBookended: boolean,
  ): string {
    // Format changes in chronological order
    const formattedChanges = selected.map((item, index) => {
      const content =
        item.type === "pr"
          ? this.formatPRForSummary(item.data as PRRecord)
          : this.formatCommitForSummary(item.data as CommitRecord);

      // Add section marker after first 8 if bookended
      if (isBookended && index === 7) {
        return (
          content +
          "\n\n---\n**[NOTE: Gap in history - continuing with most recent 100 changes]**\n---"
        );
      }

      return content;
    });

    const changesText = formattedChanges.join("\n\n");
    const totalChanges = selected.length;

    const prs = selected.filter((c) => c.type === "pr");
    const commits = selected.filter((c) => c.type === "commit");

    const hasExistingDocs =
      concept.documentation && concept.documentation.trim().length > 0;

    const existingDocsSection = hasExistingDocs
      ? `\n## Existing Documentation\n\nThe following documentation already exists for this concept. Use it as your starting point — preserve what is still accurate, update what has changed, and add any new information from the changes below.\n\n${concept.documentation}\n\n---\n`
      : "";

    const taskDescription = hasExistingDocs
      ? `**Your task**: UPDATE the existing documentation based on the new changes below. Preserve the structure and content that is still accurate. Integrate new information naturally — don't append a changelog, rewrite the relevant sections to reflect the current state.

If the new changes are minor (bug fixes, small tweaks, refactors) and the existing documentation already accurately describes the concept, respond with exactly \`OK\` and nothing else.`
      : `**Your task**: Generate HIGH-LEVEL documentation for the CURRENT state of this concept.`;

    return `You are generating SUCCINCT documentation for a software concept to help developers quickly understand and continue working on it.

**Concept**: ${concept.name}
**ID**: ${concept.id}
**Description**: ${concept.description}
**Total changes in history**: ${totalChanges} (${prs.length} PRs, ${commits.length} commits)
${existingDocsSection}
Below is ${isBookended ? "the FOUNDATIONAL (first 8) and RECENT (last 100) changes" : "the COMPLETE chronological history"} (PRs and commits) that built this concept (from oldest to newest):
${isBookended ? "\n**NOTE**: The first 8 changes show initial architecture/foundation. After a gap, the remaining changes show the recent state.\n" : ""}
${changesText}

---

${taskDescription}

**CRITICAL REQUIREMENTS**:
1. **Be SUCCINCT** - Target length: 100-200 lines MAXIMUM
2. **NO code snippets** - Focus on concepts, not implementation details
3. **High-level only** - What it does, not how it's coded
4. **Actionable** - What developers need to know to work on this concept
5. **Focus on CURRENT state** - Ignore historical implementation details

${DOC_GUIDELINES.include}

${DOC_GUIDELINES.avoid}

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
    let content = `### Commit ${commit.sha.substring(0, 7)}: ${commit.message.split("\n")[0]}
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
