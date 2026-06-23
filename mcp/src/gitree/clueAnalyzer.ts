import { Storage } from "./store/index.js";
import { Clue, ClueAnalysisResult, Usage } from "./types.js";
import { addUsage, normalizeUsage } from "../aieo/src/usage.js";
import { get_context } from "../repo/agent.js";
import { vectorizeQuery } from "../vector/index.js";

/**
 * Analyzes concept codebases to extract architectural clues
 */
export class ClueAnalyzer {
  private repo?: string; // Repository identifier "owner/repo"
  
  constructor(
    private storage: Storage,
    private repoPath: string,
    repo?: string,
    private sessionId?: string
  ) {
    this.repo = repo;
  }
  
  /**
   * Set the repo identifier (for multi-repo support)
   */
  setRepo(repo: string): void {
    this.repo = repo;
  }

  /**
   * Analyze a single concept for clues (iterative)
   */
  async analyzeConcept(conceptId: string): Promise<ClueAnalysisResult> {
    const concept = await this.storage.getConcept(conceptId);
    if (!concept) {
      throw new Error(`Concept ${conceptId} not found`);
    }

    console.log(`\n💡 Analyzing concept for clues: ${concept.name}`);

    // Get existing clues
    const existingClues = await this.storage.getCluesForConcept(conceptId);
    console.log(`   Found ${existingClues.length} existing clues`);

    // Check if we've hit the limit
    if (existingClues.length >= 40) {
      console.log(`   ⚠️  Concept already has 40 clues (limit reached)`);
      return {
        clues: [],
        complete: true,
        reasoning: "Maximum clue limit (40) reached",
        usage: normalizeUsage(),
      };
    }

    // Get files associated with this concept
    const prs = await this.storage.getPRsForConcept(conceptId);
    const commits = await this.storage.getCommitsForConcept(conceptId);
    const allFiles = new Set<string>();
    prs.forEach((pr) => pr.files.forEach((f) => allFiles.add(f)));
    commits.forEach((c) => c.files.forEach((f) => allFiles.add(f)));

    if (allFiles.size === 0) {
      console.log(`   ⚠️  No files found for this concept`);
      return {
        clues: [],
        complete: true,
        reasoning: "No files associated with this concept",
        usage: normalizeUsage(),
      };
    }

    console.log(`   Analyzing ${allFiles.size} files...`);

    // Build prompt for get_context
    const prompt = this.buildAnalysisPrompt(
      concept,
      existingClues,
      Array.from(allFiles)
    );

    // Define schema for structured output
    const schema = {
      type: "object",
      properties: {
        clues: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              type: {
                type: "string",
                enum: [
                  "utility",
                  "abstraction",
                  "integration",
                  "convention",
                  "gotcha",
                  "data-flow",
                  "state-pattern",
                ],
              },
              content: { type: "string" },
              entities: {
                type: "object",
                properties: {
                  functions: { type: "array", items: { type: "string" } },
                  classes: { type: "array", items: { type: "string" } },
                  types: { type: "array", items: { type: "string" } },
                  interfaces: { type: "array", items: { type: "string" } },
                  components: { type: "array", items: { type: "string" } },
                  endpoints: { type: "array", items: { type: "string" } },
                  tables: { type: "array", items: { type: "string" } },
                  constants: { type: "array", items: { type: "string" } },
                  hooks: { type: "array", items: { type: "string" } },
                },
              },
              files: { type: "array", items: { type: "string" } },
              keywords: { type: "array", items: { type: "string" } },
              centrality: { type: "number" },
              usageFrequency: { type: "number" },
              relatedClues: { type: "array", items: { type: "string" } },
              dependsOn: { type: "array", items: { type: "string" } },
            },
            required: ["title", "type", "content", "entities", "files", "keywords"],
          },
        },
        complete: { type: "boolean" },
        reasoning: { type: "string" },
      },
      required: ["clues", "complete", "reasoning"],
    };

    // Call get_context with schema
    console.log(`   🤖 Calling codebase analysis agent...`);
    const result = await get_context(prompt, this.repoPath, {
      schema,
      systemOverride: this.buildSystemPrompt(),
      sessionId: this.sessionId,
      isolatedContext: true,
    });

    const decision = result.content as any;

    // Create and save clues
    const now = new Date();
    const savedClues: Clue[] = [];

    for (const clueData of decision.clues || []) {
      // Generate embedding from title + content
      const embedding = await vectorizeQuery(`${clueData.title}\n\n${clueData.content}`);

      // Get repo from concept if not set
      const clueRepo = this.repo || concept.repo;
      
      const clue: Clue = {
        id: this.generateClueId(clueData.title, clueRepo),
        repo: clueRepo,
        conceptId,
        type: clueData.type,
        title: clueData.title,
        content: clueData.content,
        entities: clueData.entities || {},
        files: clueData.files || [],
        keywords: clueData.keywords || [],
        centrality: clueData.centrality,
        usageFrequency: clueData.usageFrequency,
        relatedConcepts: [conceptId], // Initially link to discovering concept only
        relatedClues: clueData.relatedClues || [],
        dependsOn: clueData.dependsOn || [],
        embedding,
        createdAt: now,
        updatedAt: now,
      };

      await this.storage.saveClue(clue);
      savedClues.push(clue);
      console.log(`   ✨ Created clue: ${clue.title} [${clue.type}]`);
    }

    // Update concept metadata
    concept.cluesCount = existingClues.length + savedClues.length;
    concept.cluesLastAnalyzedAt = now;
    await this.storage.saveConcept(concept);

    console.log(
      `   📊 Created ${savedClues.length} new clues (total: ${concept.cluesCount})`
    );
    console.log(
      `   ${
        decision.complete
          ? "✅ Analysis complete"
          : "🔄 Can run again for more clues"
      }`
    );

    return {
      clues: savedClues,
      complete: decision.complete,
      reasoning: decision.reasoning,
      usage: result.usage,
    };
  }

  /**
   * Analyze all concepts that need clue analysis
   * @param force - Force re-analysis even if already analyzed
   * @param autoLink - Automatically link clues to concepts
   * @param repo - Optional repo to filter concepts
   */
  async analyzeAllConcepts(
    force: boolean = false,
    autoLink: boolean = true,
    repo?: string
  ): Promise<Usage> {
    const concepts = await this.storage.getAllConcepts(repo || this.repo);
    console.log(`\n📚 Analyzing clues for ${concepts.length} concepts...\n`);

    let totalUsage: Usage = normalizeUsage();

    for (let i = 0; i < concepts.length; i++) {
      const concept = concepts[i];
      const progress = `[${i + 1}/${concepts.length}]`;

      // Skip if already analyzed (unless force)
      if (
        !force &&
        concept.cluesLastAnalyzedAt &&
        (concept.cluesCount || 0) >= 5
      ) {
        console.log(
          `${progress} Skipping ${concept.name} (already has ${concept.cluesCount} clues)`
        );
        continue;
      }

      console.log(`${progress} Processing: ${concept.name} (${concept.id})`);

      try {
        const result = await this.analyzeConcept(concept.id);
        totalUsage = normalizeUsage(addUsage(totalUsage, result.usage));
      } catch (error) {
        console.error(
          `   ❌ Error:`,
          error instanceof Error ? error.message : error
        );
        console.log(`   ⏭️  Skipping and continuing...`);
      }
    }

    console.log(`\n✅ Done analyzing all concepts!`);
    console.log(
      `   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`
    );

    // Automatically link clues to concepts after discovery
    if (autoLink) {
      console.log(`\n🔗 Automatically linking clues to relevant concepts...\n`);
      try {
        const { ClueLinker } = await import("./clueLinker.js");
        const linker = new ClueLinker(this.storage);
        const linkUsage = await linker.linkAllClues(false);

        totalUsage = normalizeUsage(addUsage(totalUsage, linkUsage));

        console.log(
          `\n✅ Total usage (analysis + linking): ${totalUsage.totalTokens.toLocaleString()}\n`
        );
      } catch (error) {
        console.error(
          `\n⚠️  Auto-linking failed:`,
          error instanceof Error ? error.message : error
        );
        console.log(`   Clues were created but not linked. Run link-clues manually.\n`);
      }
    }

    return totalUsage;
  }

  /**
   * Analyze a PR or commit for architectural clues
   * @param changeContext - Context about what changed (includes full formatted content if available)
   * @param conceptIds - Optional list of concept IDs to scope clues to (optimization)
   * @returns Analysis result with discovered clues
   */
  async analyzeChange(changeContext: {
    type: "pr" | "commit";
    identifier: string;
    title: string;
    summary: string;
    files: string[];
    fullContent?: string; // Full formatted PR/commit content (optional for retroactive analysis)
  }, conceptIds?: string[]): Promise<ClueAnalysisResult> {
    console.log(
      `\n💡 Analyzing ${changeContext.type} for clues: ${changeContext.identifier}`
    );
    console.log(`   ${changeContext.title}`);

    // Get existing clues to avoid duplicates
    // If conceptIds provided, only fetch clues for those concepts (optimization)
    let existingClues: Clue[];
    if (conceptIds && conceptIds.length > 0) {
      console.log(`   Fetching recent clues (max 100 per concept) for ${conceptIds.length} linked concept(s)...`);
      const cluesByConcept = await Promise.all(
        conceptIds.map((fid) => this.storage.getCluesForConcept(fid, 100))
      );
      // Deduplicate clues (a clue can be linked to multiple concepts)
      const clueMap = new Map<string, Clue>();
      for (const clues of cluesByConcept) {
        for (const clue of clues) {
          clueMap.set(clue.id, clue);
        }
      }
      existingClues = Array.from(clueMap.values());
      console.log(`   Found ${existingClues.length} existing clues for linked concepts`);
    } else {
      // Fallback: fetch all clues (less efficient but works if no concepts linked)
      existingClues = await this.storage.getAllClues();
      console.log(`   Found ${existingClues.length} existing clues in system (all concepts)`);
    }

    // Build prompt with change context
    const prompt = this.buildChangeAnalysisPrompt(changeContext, existingClues);

    // Use same schema as analyzeConcept
    const schema = {
      type: "object",
      properties: {
        clues: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              type: {
                type: "string",
                enum: [
                  "utility",
                  "abstraction",
                  "integration",
                  "convention",
                  "gotcha",
                  "data-flow",
                  "state-pattern",
                ],
              },
              content: { type: "string" },
              entities: {
                type: "object",
                properties: {
                  functions: { type: "array", items: { type: "string" } },
                  classes: { type: "array", items: { type: "string" } },
                  types: { type: "array", items: { type: "string" } },
                  interfaces: { type: "array", items: { type: "string" } },
                  components: { type: "array", items: { type: "string" } },
                  endpoints: { type: "array", items: { type: "string" } },
                  tables: { type: "array", items: { type: "string" } },
                  constants: { type: "array", items: { type: "string" } },
                  hooks: { type: "array", items: { type: "string" } },
                },
              },
              files: { type: "array", items: { type: "string" } },
              keywords: { type: "array", items: { type: "string" } },
              centrality: { type: "number" },
              usageFrequency: { type: "number" },
              relatedClues: { type: "array", items: { type: "string" } },
              dependsOn: { type: "array", items: { type: "string" } },
            },
            required: ["title", "type", "content", "entities", "files", "keywords"],
          },
        },
        complete: { type: "boolean" },
        reasoning: { type: "string" },
      },
      required: ["clues", "complete", "reasoning"],
    };

    // Call get_context with enhanced prompt
    console.log(`   🤖 Calling codebase analysis agent...`);
    const result = await get_context(prompt, this.repoPath, {
      schema,
      systemOverride: this.buildChangeSystemPrompt(),
      sessionId: this.sessionId,
      isolatedContext: true,
    });

    const decision = result.content as any;

    // Create and save clues
    const now = new Date();
    const savedClues: Clue[] = [];

    for (const clueData of decision.clues || []) {
      // Generate embedding from title + content
      const embedding = await vectorizeQuery(`${clueData.title}\n\n${clueData.content}`);

      const clue: Clue = {
        id: this.generateClueId(clueData.title, this.repo),
        repo: this.repo,
        conceptId: "unassigned", // Temporary, will be replaced by linking
        type: clueData.type,
        title: clueData.title,
        content: clueData.content,
        entities: clueData.entities || {},
        files: clueData.files || changeContext.files,
        keywords: clueData.keywords || [],
        centrality: clueData.centrality,
        usageFrequency: clueData.usageFrequency,
        relatedConcepts: [], // Empty initially - linking happens afterward
        relatedClues: clueData.relatedClues || [],
        dependsOn: clueData.dependsOn || [],
        embedding,
        createdAt: now,
        updatedAt: now,
      };

      await this.storage.saveClue(clue);
      savedClues.push(clue);
      console.log(`   ✨ Created clue: ${clue.title} [${clue.type}]`);
    }

    console.log(`   📊 Created ${savedClues.length} new clue(s)`);

    return {
      clues: savedClues,
      complete: decision.complete,
      reasoning: decision.reasoning,
      usage: result.usage,
    };
  }

  /**
   * Build the analysis prompt for get_context
   */
  private buildAnalysisPrompt(
    concept: any,
    existingClues: Clue[],
    files: string[]
  ): string {
    const existingCluesList = existingClues
      .map((c) => `  - ${c.title} (${c.id}) [${c.type}]`)
      .join("\n");

    const filesList = files.slice(0, 50).join("\n  ");
    const filesNote =
      files.length > 50 ? `\n  ... and ${files.length - 50} more files` : "";

    return `Analyze the codebase for the concept "${concept.name}" and identify architectural utilities, key abstractions, and patterns.

**Concept**: ${concept.name}
**Description**: ${concept.description}
${concept.documentation ? `\n**Documentation**:\n${concept.documentation}\n` : ""}
**Existing Clues** (${existingClues.length}/40):
${existingClues.length > 0 ? existingCluesList : "  (none yet)"}

**Files to Analyze** (${files.length} total):
  ${filesList}${filesNote}

**Your Task**:
1. Read and analyze the codebase files for this concept
2. Identify NEW clues that don't overlap with existing ones
3. Focus on the most important patterns and utilities (5-10 clues)
4. For each clue, extract:
   - Title (concise, descriptive, GENERIC - no concept name)
   - Type (utility, abstraction, integration, convention, gotcha, data-flow, state-pattern)
   - Content (GENERIC explanation - WHY, WHEN, HOW - written for ANY concept that uses this pattern)
   - Entities (actual function/class/type/endpoint names from the code)
   - Files (list of files where these entities are defined or this pattern is used)
   - Keywords (for searchability)

5. Set "complete: true" if you believe this concept is comprehensively covered (or if creating fewer than 2 new clues)

**CRITICAL - Content Writing Guidelines**:
- include exact function names, type names, class names, etc. But not big code snippets.
- Write GENERIC, reusable explanations that work for ANY concept using this pattern
- Focus on the utility/abstraction ITSELF, not how this particular concept uses it
- Think: "How would I explain this to someone working on ANY concept?"

**IMPORTANT**:
- DO NOT create code snippets - only reference entity names
- Avoid duplicating existing clues
- Prioritize reusable utilities and common patterns
- Target 5-10 high-value clues per analysis

Analyze the codebase and return your structured response.`;
  }

  /**
   * Build the system prompt for the agent
   */
  private buildSystemPrompt(): string {
    return `You are a codebase analyzer that extracts GENERIC, REUSABLE architectural patterns and development insights.

Your goal is to identify "Clues" - knowledge nuggets that help developers understand:
- Reusable utilities and helpers
- Architectural conventions
- Key abstractions (interfaces, base classes)
- Integration patterns
- Common gotchas and edge cases
- Data flow patterns
- State management approaches

**CRITICAL: Write GENERIC content that works for ANY concept using the pattern!**
- Think of clues as reusable documentation that will be linked to MULTIPLE concepts
- DO NOT mention specific concept names in the content
- Focus on the utility/abstraction itself, not one specific usage
- Example: "This utility ensures workspace isolation by..." NOT "Quick Ask uses this to..."

**Clue Types:**
- **utility**: Reusable functions/classes used across multiple concepts
- **abstraction**: Interface/type/base class meant to be extended
- **integration**: How to integrate with external systems/concepts
- **convention**: Coding style or naming convention
- **gotcha**: Common mistake or edge case to avoid
- **data-flow**: How data transforms through the system
- **state-pattern**: State management approach

**Entity Types to Extract:**
- functions: Function names (e.g., "generateToken", "validateUser")
- classes: Class names (e.g., "JWTManager", "AuthService")
- types: Type/Interface names (e.g., "TokenPayload", "AuthConfig")
- interfaces: Interface names
- components: Component names (React/Vue)
- endpoints: API endpoint paths (e.g., "POST /auth/login")
- tables: Database table names
- constants: Constant names
- hooks: React/Vue hook names

**Guidelines:**
1. Extract entity names exactly as they appear in code
2. Focus on patterns that are reused 3+ times or are architecturally significant
3. Keep content GENERIC and concise (2-3 sentences explaining WHY, WHEN, HOW)
4. Don't include code snippets - only entity names and file paths
5. Prioritize high-impact clues over completeness
6. Set complete=true if fewer than 2 new clues found

Analyze the provided files and extract valuable, GENERIC clues.`;
  }

  /**
   * Build minimal content when full formatted content is not available
   */
  private buildMinimalContent(changeContext: {
    type: "pr" | "commit";
    identifier: string;
    title: string;
    files: string[];
  }): string {
    const changeType = changeContext.type === "pr" ? "Pull Request" : "Commit";
    const filesList = changeContext.files.slice(0, 50).join("\n  ");
    const filesNote = changeContext.files.length > 50
      ? `\n  ... and ${changeContext.files.length - 50} more files`
      : "";

    return `# ${changeType} ${changeContext.identifier}: ${changeContext.title}

**Files Changed** (${changeContext.files.length} total):
  ${filesList}${filesNote}

*Note: Full change details not available (retroactive analysis)*`;
  }

  /**
   * Build prompt for analyzing a change (PR or commit)
   */
  private buildChangeAnalysisPrompt(
    changeContext: {
      type: "pr" | "commit";
      identifier: string;
      title: string;
      summary: string;
      files: string[];
      fullContent?: string;
    },
    existingClues: Clue[]
  ): string {
    const existingCluesList = existingClues
      .map((c) => `  - ${c.title} [${c.type}]`)
      .join("\n");

    const changeType = changeContext.type === "pr" ? "Pull Request" : "Commit";

    // Use full content if available, otherwise construct minimal version
    const contentSection = changeContext.fullContent || this.buildMinimalContent(changeContext);

    return `Analyze the codebase based on this ${changeType} and extract architectural utilities, key abstractions, and patterns.

**${changeType} Details**:

${contentSection}

---

**LLM Analysis Summary**: ${changeContext.summary}

**Existing Clues** (${existingClues.length} total - avoid duplicates):
${existingClues.length > 0 ? existingCluesList : "  (none yet)"}

**Your Task**:
1. Read and analyze the codebase files that changed (shown above)
2. **FOCUS on utilities/abstractions relevant to what changed**
3. Extract architectural insights that would help developers working on similar changes
4. Identify NEW clues that don't overlap with existing ones

For each clue, extract:
- Title (concise, descriptive, GENERIC)
- Type (utility, abstraction, integration, convention, gotcha, data-flow, state-pattern)
- Content (GENERIC explanation - WHY, WHEN, HOW - works for ANY concept)
- Entities (actual function/class/type/endpoint names from the code)
- Files (list of files where entities are defined)
- Keywords (for searchability)

**IMPORTANT - Focus on Actionable Insights**:
- Prioritize patterns VISIBLE in these changes
- Extract utilities that were INTRODUCED or USED in this change
- Identify gotchas mentioned in code reviews
- Focus on patterns that make the code work
- Avoid generic/abstract patterns unless clearly demonstrated

**CRITICAL - Content Writing Guidelines**:
- Write GENERIC, reusable explanations that work for ANY codebase/concept
- DO NOT mention specific concept names or this specific change
- Focus on the utility/abstraction ITSELF as a general concept
- Use generic examples: "This pattern..." instead of "This PR..."
- Think: "How would I explain this pattern to someone in a different codebase?"

**Quality over Quantity**:
- Target 2-5 high-value, actionable clues per change
- Skip generic patterns unless clearly demonstrated
- Each clue should be reusable knowledge

Analyze the codebase and return your structured response.`;
  }

  /**
   * Build system prompt for change-based analysis
   */
  private buildChangeSystemPrompt(): string {
    return `You are a codebase analyzer that extracts ACTIONABLE, change-focused architectural insights.

Your goal is to identify "Clues" based on what changed in a specific PR or commit. Focus on:
- Patterns and utilities VISIBLE in the changes
- Architectural decisions DEMONSTRATED by the code
- Gotchas and edge cases MENTIONED in reviews or error handling
- Integrations and abstractions INTRODUCED or USED

**CRITICAL: Focus on Actionable Insights!**
- Prioritize patterns directly relevant to the changes
- Avoid generic/abstract patterns unless clearly demonstrated
- Extract insights useful for developers working on similar changes
- Quality over quantity - 2-5 excellent, focused clues better than many generic ones

**CRITICAL: Write GENERIC content!**
- Clues are standalone knowledge not tied to specific concepts
- DO NOT mention concept names, PR numbers, or commit SHAs
- Focus on the utility/abstraction itself as a general concept
- Example: "This pattern ensures workspace isolation by..." NOT "This PR adds..."

**Clue Types:**
- **utility**: Reusable functions/classes
- **abstraction**: Interface/type/base class introduced or extended
- **integration**: How external systems are integrated
- **convention**: Coding style visible in changes
- **gotcha**: Common mistakes or error handling patterns
- **data-flow**: How data transforms
- **state-pattern**: State management approach

**Guidelines:**
1. Extract 2-5 high-impact clues focused on changes
2. Prioritize actionable patterns over generic ones
3. Keep content GENERIC and concise (2-3 sentences: WHY, WHEN, HOW)
4. Extract entity names exactly as they appear in code
5. Set complete=true if fewer than 2 new actionable clues found

Analyze the codebase with the change context in mind and extract valuable, ACTIONABLE, GENERIC clues.`;
  }

  /**
   * Generate repo-prefixed slug-style clue ID from title
   * e.g., "owner/repo/clue-slug"
   */
  private generateClueId(title: string, repo?: string): string {
    const slug = title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    
    // Return repo-prefixed ID if repo is provided
    return repo ? `${repo}/${slug}` : slug;
  }
}
