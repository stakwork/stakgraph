import { Octokit } from "@octokit/rest";
import { Storage } from "./store/index.js";
import {
  GitreeSessionTracker,
  LLMClient,
  SYSTEM_PROMPT,
  DECISION_GUIDELINES,
} from "./llm.js";
import {
  Concept,
  PRRecord,
  CommitRecord,
  LLMDecision,
  GitHubPR,
  Usage,
  ChronologicalCheckpoint,
} from "./types.js";
import { fetchPullRequestContent } from "./pr.js";
import { fetchCommitContent } from "./commit.js";
import {ClueAnalyzer} from "./clueAnalyzer.js";
import { exploreNewConcept } from "./bootstrap.js";
import { addUsage, normalizeUsage } from "../aieo/src/usage.js";

/**
 * Main class for building the concept knowledge base from PRs and commits
 */
export class StreamingConceptBuilder {
  private clueAnalyzer?: ClueAnalyzer; // ClueAnalyzer instance (lazily initialized)
  private repo: string = ""; // Repository identifier "owner/repo"
  private sessionId?: string;

  constructor(
    private storage: Storage,
    private llm: LLMClient,
    private octokit: Octokit,
    private repoPath?: string,
    private shouldAnalyzeClues: boolean = false,
    private sessionTracker?: GitreeSessionTracker,
  ) {}

  /**
   * Main entry point: process a repo (both PRs and commits chronologically)
   */
  async processRepo(
    owner: string,
    repo: string,
    sessionId?: string,
  ): Promise<{ usage: Usage; modifiedConceptIds: Set<string> }> {
    // Set repo identifier for use throughout processing
    this.repo = `${owner}/${repo}`;
    this.sessionId = sessionId;

    let totalUsage: Usage = normalizeUsage();

    // Track which concepts were modified during processing
    const modifiedConceptIds = new Set<string>();

    // Get chronological checkpoint for THIS repo (or migrate from old checkpoints)
    let checkpoint = await this.storage.getChronologicalCheckpoint(this.repo);

    // Backwards compatibility: migrate old checkpoints if needed
    if (!checkpoint) {
      checkpoint = await this.migrateOldCheckpoint();
    }

    console.log(`\n📋 Fetching changes from ${owner}/${repo}...`);
    const changes = await this.fetchAllChanges(owner, repo, checkpoint);

    if (changes.length === 0) {
      console.log(`   No new changes to process.`);
      const concepts = await this.storage.getAllConcepts();
      console.log(`\n🎉 Repository processing complete!`);
      console.log(`   Total concepts: ${concepts.length}`);
      return { usage: totalUsage, modifiedConceptIds };
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
          const usage = await this.processPR(
            owner,
            repo,
            pr,
            modifiedConceptIds,
          );
          totalUsage = normalizeUsage(addUsage(totalUsage, usage));
          console.log(
            `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`,
          );
        } catch (error) {
          console.error(
            `   ❌ Error processing PR #${pr.number}:`,
            error instanceof Error ? error.message : error,
          );
          console.log(`   ⏭️  Skipping and continuing with next change...`);

          // Save a minimal PR record so we know it was attempted
          await this.storage.savePR({
            number: pr.number,
            repo: this.repo,
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
          }`,
        );

        try {
          const usage = await this.processCommit(
            owner,
            repo,
            commit,
            modifiedConceptIds,
          );
          totalUsage = normalizeUsage(addUsage(totalUsage, usage));
          console.log(
            `   📊 Input Usage: ${totalUsage.inputTokens.toLocaleString()} tokens. Output Usage: ${totalUsage.outputTokens.toLocaleString()} tokens`,
          );
        } catch (error) {
          console.error(
            `   ❌ Error processing commit ${commit.sha.substring(0, 7)}:`,
            error instanceof Error ? error.message : error,
          );
          console.log(`   ⏭️  Skipping and continuing with next change...`);

          // Save a minimal commit record so we know it was attempted
          await this.storage.saveCommit({
            sha: commit.sha,
            repo: this.repo,
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
    const concepts = await this.storage.getAllConcepts(this.repo);
    console.log(`\n🎉 Repository processing complete!`);
    console.log(`   Total concepts: ${concepts.length}`);
    console.log(`   Modified concepts: ${modifiedConceptIds.size}`);
    console.log(
      `   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`,
    );

    return { usage: totalUsage, modifiedConceptIds };
  }

  /**
   * Fetch all changes (PRs and commits) chronologically from checkpoint
   */
  private async fetchAllChanges(
    owner: string,
    repo: string,
    checkpoint: ChronologicalCheckpoint | null,
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
      }...`,
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
      },
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
            `      Progress: ${processedCount}/${commits.length} commits checked (${standaloneCommits} standalone, ${skippedPRCommits} PR-associated)`,
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
            commit.commit.author?.date || Date.now(),
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
          error,
        );
      }
    }
    console.log(
      `   ✓ Found ${standaloneCommits} standalone commits (skipped ${skippedPRCommits} PR-associated commits)`,
    );

    // Sort chronologically (oldest first)
    console.log(`   🔄 Sorting changes chronologically...`);
    changes.sort((a, b) => a.date.getTime() - b.date.getTime());

    console.log(
      `   ✅ Found ${changes.filter((c) => c.type === "pr").length} PRs and ${
        changes.filter((c) => c.type === "commit").length
      } commits to process`,
    );

    return changes;
  }

  /**
   * Process a single PR
   */
  private async processPR(
    owner: string,
    repo: string,
    pr: GitHubPR,
    modifiedConceptIds: Set<string>,
  ): Promise<Usage> {
    // Skip obvious noise
    if (this.shouldSkip(pr)) {
      console.log(`   ⏭️  Skipped (maintenance/trivial)`);

      // Still save the PR record for completeness
      await this.storage.savePR({
        number: pr.number,
        repo: this.repo,
        title: pr.title,
        summary: "Skipped (maintenance/trivial)",
        mergedAt: pr.mergedAt,
        url: pr.url,
        files: pr.filesChanged,
      });
      return normalizeUsage();
    }

    // Get current concepts for context
    const concepts = await this.storage.getAllConcepts(this.repo);

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
      },
    );

    // Build decision prompt
    const prompt = await this.buildDecisionPrompt(prContent, concepts);

    // Ask LLM what to do
    console.log(`   🤖 Asking LLM for decision...`);
    const { decision, usage } = await this.llm.decide(
      prompt,
      undefined,
      this.sessionId,
      `gitree decision: PR #${pr.number}`,
    );

    // Apply decision (pass usage to save with PR record)
    await this.applyPrDecision(
      owner,
      repo,
      pr,
      decision,
      modifiedConceptIds,
      usage,
    );

    // Analyze for clues if enabled
    if (this.shouldAnalyzeClues) {
      try {
        // Get concepts linked to this PR (after decision has been applied)
        const linkedConcepts = await this.storage.getConceptsForPR(
          pr.number,
          this.repo,
        );
        const conceptIds = linkedConcepts.map((f) => f.id);

        // Fetch file list for context
        const { data: files } = await this.octokit.pulls.listFiles({
          owner,
          repo,
          pull_number: pr.number,
          per_page: 100,
        });

        const clueUsage = await this.analyzeChangeForClues(
          {
            type: "pr" as const,
            identifier: `#${pr.number}`,
            title: pr.title,
            summary: decision.summary,
            files: files.map((f) => f.filename),
            fullContent: prContent,
          },
          {
            date: pr.mergedAt,
            id: pr.number.toString(),
          },
          conceptIds,
        );
        return normalizeUsage(addUsage(usage, clueUsage));
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
    concepts: Concept[],
  ): Promise<string> {
    const themesContext = await this.formatThemeContext();

    return `${SYSTEM_PROMPT}

${this.formatConceptContext(concepts)}

${themesContext}

${prContent}

${DECISION_GUIDELINES}`;
  }

  /**
   * Format concept list for context
   */
  private formatConceptContext(concepts: Concept[]): string {
    if (concepts.length === 0) {
      return "## Current Concepts\n\nNo concepts yet.";
    }

    const conceptList = concepts
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

    return `## Current Concepts\n\n${conceptList}`;
  }

  /**
   * Format recent themes for context
   */
  private async formatThemeContext(): Promise<string> {
    const themes = await this.storage.getRecentThemes(this.repo);

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
    decision: LLMDecision,
    modifiedConceptIds: Set<string>,
    usage?: Usage,
  ): Promise<void> {
    // Fetch file list for this PR
    const { data: files } = await this.octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: pr.number,
      per_page: 100,
    });

    // Save PR record with repo and usage
    const prRecord: PRRecord = {
      number: pr.number,
      repo: this.repo,
      title: pr.title,
      summary: decision.summary,
      mergedAt: pr.mergedAt,
      url: pr.url,
      files: files.map((f) => f.filename),
      newDeclarations: decision.newDeclarations,
      usage: usage,
    };
    await this.storage.savePR(prRecord);

    // Apply the decision using shared logic
    await this.applyDecisionToConcepts(
      decision,
      {
        changeDate: pr.mergedAt,
        addToConcept: async (concept) => {
          if (!concept.prNumbers.includes(pr.number)) {
            concept.prNumbers.push(pr.number);
            return true;
          }
          return false;
        },
        createConceptWith: (baseConcept) => ({
          ...baseConcept,
          prNumbers: [pr.number],
          commitShas: [],
        }),
      },
      modifiedConceptIds,
    );
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
    },
    modifiedConceptIds: Set<string>,
  ): Promise<Usage> {
    // Skip obvious noise
    if (this.shouldSkipCommit(commit)) {
      console.log(`   ⏭️  Skipped (maintenance/trivial)`);

      // Still save the commit record for completeness
      await this.storage.saveCommit({
        sha: commit.sha,
        repo: this.repo,
        message: commit.message,
        summary: "Skipped (maintenance/trivial)",
        author: commit.author,
        committedAt: commit.committedAt,
        url: commit.url,
        files: [],
      });
      return normalizeUsage();
    }

    // Get current concepts for context
    const concepts = await this.storage.getAllConcepts(this.repo);

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
      },
    );

    // Build decision prompt
    const prompt = await this.buildDecisionPrompt(commitContent, concepts);

    // Ask LLM what to do
    console.log(`   🤖 Asking LLM for decision...`);
    const { decision, usage } = await this.llm.decide(
      prompt,
      undefined,
      this.sessionId,
      `gitree decision: commit ${commit.sha.substring(0, 7)}`,
    );

    // Apply decision (pass usage to save with commit record)
    await this.applyCommitDecision(
      owner,
      repo,
      commit,
      decision,
      modifiedConceptIds,
      usage,
    );

    // Analyze for clues if enabled
    if (this.shouldAnalyzeClues) {
      try {
        // Get concepts linked to this commit (after decision has been applied)
        const linkedConcepts = await this.storage.getConceptsForCommit(
          commit.sha,
          this.repo,
        );
        const conceptIds = linkedConcepts.map((f) => f.id);

        // Fetch file list for context
        const { data: commitData } = await this.octokit.repos.getCommit({
          owner,
          repo,
          ref: commit.sha,
        });
        const files = commitData.files || [];

        const clueUsage = await this.analyzeChangeForClues(
          {
            type: "commit" as const,
            identifier: commit.sha.substring(0, 7),
            title: commit.message.split("\n")[0],
            summary: decision.summary,
            files: files.map((f: any) => f.filename),
            fullContent: commitContent,
          },
          {
            date: commit.committedAt,
            id: commit.sha,
          },
          conceptIds,
        );
        return normalizeUsage(addUsage(usage, clueUsage));
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
    decision: LLMDecision,
    modifiedConceptIds: Set<string>,
    usage?: Usage,
  ): Promise<void> {
    // Fetch file list for this commit
    const { data: commitData } = await this.octokit.repos.getCommit({
      owner,
      repo,
      ref: commit.sha,
    });

    const files = commitData.files || [];

    // Save commit record with repo and usage
    const commitRecord: CommitRecord = {
      sha: commit.sha,
      repo: this.repo,
      message: commit.message,
      summary: decision.summary,
      author: commit.author,
      committedAt: commit.committedAt,
      url: commit.url,
      files: files.map((f: any) => f.filename),
      newDeclarations: decision.newDeclarations,
      usage: usage,
    };
    await this.storage.saveCommit(commitRecord);

    // Apply the decision using shared logic
    await this.applyDecisionToConcepts(
      decision,
      {
        changeDate: commit.committedAt,
        addToConcept: async (concept) => {
          // Initialize commitShas if it doesn't exist (legacy concepts)
          if (!concept.commitShas) {
            concept.commitShas = [];
          }
          if (!concept.commitShas.includes(commit.sha)) {
            concept.commitShas.push(commit.sha);
            return true;
          }
          return false;
        },
        createConceptWith: (baseConcept) => ({
          ...baseConcept,
          prNumbers: [],
          commitShas: [commit.sha],
        }),
      },
      modifiedConceptIds,
    );
  }

  /**
   * Shared logic for applying LLM decision to concepts
   */
  private async applyDecisionToConcepts(
    decision: LLMDecision,
    config: {
      changeDate: Date;
      addToConcept: (concept: Concept) => Promise<boolean>;
      createConceptWith: (
        base: Omit<Concept, "prNumbers" | "commitShas">,
      ) => Concept;
    },
    modifiedConceptIds: Set<string>,
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
        // Add to existing concept(s)
        if (
          decision.existingConceptIds &&
          decision.existingConceptIds.length > 0
        ) {
          for (const conceptId of decision.existingConceptIds) {
            const concept = await this.storage.getConcept(conceptId, this.repo);
            if (concept) {
              const wasAdded = await config.addToConcept(concept);
              if (wasAdded) {
                concept.lastUpdated = config.changeDate;
                await this.storage.saveConcept(concept);
                modifiedConceptIds.add(concept.id);
                console.log(`   → Added to concept: ${concept.name}`);
              }
            } else {
              console.log(
                `   ⚠️  Warning: Concept ${conceptId} not found, skipping`,
              );
            }
          }
        }
      }

      if (action === "create_new") {
        // Create new concept(s)
        if (decision.newConcepts && decision.newConcepts.length > 0) {
          for (const newConceptData of decision.newConcepts) {
            const baseConcept = {
              id: this.generateConceptId(newConceptData.name),
              repo: this.repo,
              name: newConceptData.name,
              description: newConceptData.description,
              createdAt: config.changeDate,
              lastUpdated: config.changeDate,
            };
            const newConcept = config.createConceptWith(baseConcept);
            await this.storage.saveConcept(newConcept);
            modifiedConceptIds.add(newConcept.id);
            console.log(`   ✨ Created new concept: ${newConcept.name}`);

            // Explore codebase to generate initial docs (only if we have a local clone)
            if (this.repoPath) {
              await exploreNewConcept(
                newConcept,
                this.repoPath,
                this.storage,
                this.sessionId,
              );
            }
          }
        }
      }
    }

    // Update concept descriptions (and attach the PR/commit that caused the update)
    if (decision.updateConcepts && decision.updateConcepts.length > 0) {
      for (const update of decision.updateConcepts) {
        const concept = await this.storage.getConcept(
          update.conceptId,
          this.repo,
        );
        if (concept) {
          concept.description = update.newDescription;
          concept.lastUpdated = config.changeDate;
          await config.addToConcept(concept);
          await this.storage.saveConcept(concept);
          modifiedConceptIds.add(concept.id);
          console.log(`   🔄 Updated concept description: ${concept.name}`);
          console.log(`      ${update.reasoning}`);
        } else {
          console.log(
            `   ⚠️  Warning: Cannot update concept ${update.conceptId} - not found`,
          );
        }
      }
    }

    // Save themes (per-repo)
    if (decision.themes && decision.themes.length > 0) {
      await this.storage.addThemes(this.repo, decision.themes);
      console.log(`   🏷️  Tagged: ${decision.themes.join(", ")}`);
    }
  }

  /**
   * Generate repo-prefixed slug-style concept ID from name
   * e.g., "owner/repo/concept-slug"
   */
  private generateConceptId(name: string): string {
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");

    // Return repo-prefixed ID
    return `${this.repo}/${slug}`;
  }

  /**
   * Migrate old checkpoint format to new chronological checkpoint (per-repo)
   */
  private async migrateOldCheckpoint(): Promise<ChronologicalCheckpoint | null> {
    const lastPR = await this.storage.getLastProcessedPR(this.repo);
    const lastCommit = await this.storage.getLastProcessedCommit(this.repo);

    // If no old checkpoints exist, return null (start from beginning)
    if (lastPR === 0 && !lastCommit) {
      return null;
    }

    console.log(
      `   Migrating old checkpoints to chronological format for ${this.repo}...`,
    );

    // Get the dates of the last processed PR and commit
    let latestDate: Date | null = null;

    if (lastPR > 0) {
      const pr = await this.storage.getPR(lastPR, this.repo);
      if (pr) {
        latestDate = pr.mergedAt;
      }
    }

    if (lastCommit) {
      const commit = await this.storage.getCommit(lastCommit, this.repo);
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

    await this.storage.setChronologicalCheckpoint(this.repo, checkpoint);
    console.log(
      `   Migrated to chronological checkpoint: ${checkpoint.lastProcessedTimestamp}`,
    );

    return checkpoint;
  }

  /**
   * Update checkpoint after processing a change (per-repo)
   */
  private async updateCheckpoint(date: Date, id: string): Promise<void> {
    const currentCheckpoint = await this.storage.getChronologicalCheckpoint(
      this.repo,
    );
    const dateString = date.toISOString();

    if (
      !currentCheckpoint ||
      currentCheckpoint.lastProcessedTimestamp < dateString
    ) {
      // New timestamp - replace checkpoint
      await this.storage.setChronologicalCheckpoint(this.repo, {
        lastProcessedTimestamp: dateString,
        processedAtTimestamp: [id],
      });
    } else if (currentCheckpoint.lastProcessedTimestamp === dateString) {
      // Same timestamp - add to processedAtTimestamp array
      if (!currentCheckpoint.processedAtTimestamp.includes(id)) {
        currentCheckpoint.processedAtTimestamp.push(id);
        await this.storage.setChronologicalCheckpoint(
          this.repo,
          currentCheckpoint,
        );
      }
    }
    // If date < checkpoint, this is an old item (shouldn't happen, but ignore)
  }

  /**
   * Analyze change (PR or commit) for clues
   */
  private async analyzeChangeForClues(
    changeContext: {
      type: "pr" | "commit";
      identifier: string;
      title: string;
      summary: string;
      files: string[];
      fullContent?: string;
    },
    checkpoint: { date: Date; id: string },
    conceptIds?: string[],
  ): Promise<Usage> {
    let clueUsage: Usage = normalizeUsage();
    console.log(`   💡 Analyzing for clues...`);

    // Initialize clue analyzer if needed
    if (!this.clueAnalyzer) {
      if (!this.repoPath) {
        console.log(`   ⏭️  Skipping clue analysis (no repo path)`);
        return clueUsage;
      }
      const { ClueAnalyzer } = await import("./clueAnalyzer.js");
      this.clueAnalyzer = new ClueAnalyzer(
        this.storage,
        this.repoPath,
        undefined,
        this.sessionId,
      );
    }

    // Analyze change (pass conceptIds to scope clues)
    const result = await this.clueAnalyzer.analyzeChange(
      changeContext,
      conceptIds,
    );
    clueUsage = normalizeUsage(addUsage(clueUsage, result.usage));

    if (result.clues.length === 0) {
      console.log(`   ℹ️  No new clues found`);
    } else {
      console.log(`   ✨ Found ${result.clues.length} clue(s)`);

      // Auto-link clues to relevant concepts
      const { ClueLinker } = await import("./clueLinker.js");
      const linker = new ClueLinker(
        this.storage,
        this.sessionId,
        this.sessionTracker,
      );
      const clueIds = result.clues.map((c: any) => c.id);

      console.log(
        `   🔗 Linking ${clueIds.length} clue(s) to relevant concepts...`,
      );
      const linkUsage = await linker.linkClues(clueIds);
      clueUsage = normalizeUsage(addUsage(clueUsage, linkUsage));
    }

    // Save checkpoint after analyzing (regardless of whether clues were found)
    await this.updateClueAnalysisCheckpoint(checkpoint.date, checkpoint.id);
    return clueUsage;
  }

  /**
   * Update clue analysis checkpoint (similar to updateCheckpoint but for clues, per-repo)
   */
  private async updateClueAnalysisCheckpoint(
    date: Date,
    id: string,
  ): Promise<void> {
    const currentCheckpoint = await this.storage.getClueAnalysisCheckpoint(
      this.repo,
    );
    const dateString = date.toISOString();

    if (
      !currentCheckpoint ||
      currentCheckpoint.lastProcessedTimestamp < dateString
    ) {
      // New timestamp - create new checkpoint
      await this.storage.setClueAnalysisCheckpoint(this.repo, {
        lastProcessedTimestamp: dateString,
        processedAtTimestamp: [id],
      });
    } else if (currentCheckpoint.lastProcessedTimestamp === dateString) {
      // Same timestamp - add to processedAtTimestamp array
      if (!currentCheckpoint.processedAtTimestamp.includes(id)) {
        currentCheckpoint.processedAtTimestamp.push(id);
        await this.storage.setClueAnalysisCheckpoint(
          this.repo,
          currentCheckpoint,
        );
      }
    }
    // If date < checkpoint, this is an old item (shouldn't happen, but ignore)
  }
}
