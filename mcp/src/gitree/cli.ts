#!/usr/bin/env node

import { Command } from "commander";
import { Octokit } from "@octokit/rest";
import { Storage, FileSystemStore, GraphStorage } from "./store/index.js";
import { LLMClient } from "./llm.js";
import { StreamingConceptBuilder } from "./builder.js";
import { Summarizer } from "./summarizer.js";
import { FileLinker } from "./fileLinker.js";
import { ClueAnalyzer } from "./clueAnalyzer.js";
import { getApiKeyForProvider } from "../aieo/src/provider.js";
import { addUsage, normalizeUsage } from "../aieo/src/usage.js";
import { Usage } from "./types.js";

const program = new Command();

program
  .name("gitree")
  .description("GitHub Concept Knowledge Base - Extract concepts from PR history")
  .version("1.0.0");

/**
 * Create storage instance based on options
 */
async function createStorage(options: any): Promise<Storage> {
  let storage: Storage;

  if (options.graph) {
    console.log("   Using: Neo4j GraphStorage");
    storage = new GraphStorage();
  } else {
    console.log(`   Using: FileSystemStorage (${options.dir})`);
    storage = new FileSystemStore(options.dir);
  }

  await storage.initialize();
  return storage;
}

/**
 * Process a repository
 */
program
  .command("process")
  .description("Process a GitHub repository to extract concepts")
  .argument("<owner>", "Repository owner")
  .argument("<repo>", "Repository name")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .option("-t, --token <token>", "GitHub token (or set GITHUB_TOKEN env)")
  .action(async (owner: string, repo: string, options) => {
    try {
      // Get GitHub token
      const githubToken =
        options.token || process.env.GITHUB_TOKEN;
      if (!githubToken) {
        console.error(
          "❌ GitHub token required. Set GITHUB_TOKEN env or use --token"
        );
        process.exit(1);
      }

      // Get Anthropic API key
      const anthropicKey = getApiKeyForProvider("anthropic");

      // Initialize components
      console.log(`\n🚀 Initializing GitHub Concept Knowledge Base...`);
      console.log(`   Repository: ${owner}/${repo}`);

      const storage = await createStorage(options);

      const octokit = new Octokit({ auth: githubToken });
      const llm = new LLMClient("anthropic", anthropicKey);
      const builder = new StreamingConceptBuilder(storage, llm, octokit);

      // Process repo (both PRs and commits)
      await builder.processRepo(owner, repo);

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * List all concepts
 */
program
  .command("list-concepts")
  .description("List all concepts in the knowledge base")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (options) => {
    try {
      const storage = await createStorage(options);

      const concepts = await storage.getAllConcepts();

      if (concepts.length === 0) {
        console.log("No concepts found.");
        return;
      }

      console.log(`\n📚 Concepts (${concepts.length} total):\n`);

      const sorted = concepts.sort(
        (a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime()
      );

      for (const concept of sorted) {
        console.log(`🔹 ${concept.name} (${concept.id})`);
        console.log(`   ${concept.description}`);
        const commitCount = (concept.commitShas || []).length;
        const changesSummary =
          commitCount > 0
            ? `PRs: ${concept.prNumbers.length} | Commits: ${commitCount}`
            : `PRs: ${concept.prNumbers.length}`;
        console.log(
          `   ${changesSummary} | Last updated: ${concept.lastUpdated.toISOString().split("T")[0]}`
        );
        console.log();
      }
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Show details of a specific concept
 */
program
  .command("show-concept")
  .description("Show details of a specific concept")
  .argument("<conceptId>", "Concept ID")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (conceptId: string, options) => {
    try {
      const storage = await createStorage(options);

      const concept = await storage.getConcept(conceptId);

      if (!concept) {
        console.log(`❌ Concept not found: ${conceptId}`);
        return;
      }

      console.log(`\n📖 Concept: ${concept.name}\n`);
      console.log(`ID: ${concept.id}`);
      console.log(`Description: ${concept.description}`);
      console.log(`Created: ${concept.createdAt.toISOString().split("T")[0]}`);
      console.log(`Last Updated: ${concept.lastUpdated.toISOString().split("T")[0]}`);
      console.log(`\nPull Requests (${concept.prNumbers.length}):\n`);

      const prs = await storage.getPRsForConcept(conceptId);
      for (const pr of prs) {
        console.log(`  #${pr.number}: ${pr.title}`);
        console.log(`     ${pr.summary}`);
        console.log(`     ${pr.url}`);
        console.log();
      }

      const commitCount = (concept.commitShas || []).length;
      if (commitCount > 0) {
        console.log(`\nCommits (${commitCount}):\n`);

        const commits = await storage.getCommitsForConcept(conceptId);
        for (const commit of commits) {
          console.log(`  ${commit.sha.substring(0, 7)}: ${commit.message.split('\n')[0]}`);
          console.log(`     ${commit.summary}`);
          console.log(`     ${commit.url}`);
          console.log();
        }
      }
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Show details of a specific PR
 */
program
  .command("show-pr")
  .description("Show details of a specific PR")
  .argument("<number>", "PR number")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (number: string, options) => {
    try {
      const prNumber = parseInt(number);
      const storage = await createStorage(options);

      const pr = await storage.getPR(prNumber);

      if (!pr) {
        console.log(`❌ PR not found: #${prNumber}`);
        return;
      }

      const concepts = await storage.getConceptsForPR(prNumber);

      console.log(`\n📄 PR #${pr.number}: ${pr.title}\n`);
      console.log(`Summary: ${pr.summary}`);
      console.log(`Merged: ${pr.mergedAt.toISOString().split("T")[0]}`);
      console.log(`URL: ${pr.url}`);

      if (concepts.length > 0) {
        console.log(`\nPart of ${concepts.length} concept(s):\n`);
        for (const concept of concepts) {
          console.log(`  🔹 ${concept.name} (${concept.id})`);
        }
      } else {
        console.log(`\nNot associated with any concepts.`);
      }
      console.log();
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Show details of a specific commit
 */
program
  .command("show-commit")
  .description("Show details of a specific commit")
  .argument("<sha>", "Commit SHA")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (sha: string, options) => {
    try {
      const storage = await createStorage(options);

      const commit = await storage.getCommit(sha);

      if (!commit) {
        console.log(`❌ Commit not found: ${sha}`);
        return;
      }

      const concepts = await storage.getConceptsForCommit(sha);

      console.log(`\n📝 Commit ${commit.sha.substring(0, 7)}: ${commit.message.split('\n')[0]}\n`);
      console.log(`Summary: ${commit.summary}`);
      console.log(`Author: ${commit.author}`);
      console.log(`Committed: ${commit.committedAt.toISOString().split("T")[0]}`);
      console.log(`URL: ${commit.url}`);

      if (concepts.length > 0) {
        console.log(`\nPart of ${concepts.length} concept(s):\n`);
        for (const concept of concepts) {
          console.log(`  🔹 ${concept.name} (${concept.id})`);
        }
      } else {
        console.log(`\nNot associated with any concepts.`);
      }
      console.log();
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Show statistics
 */
program
  .command("stats")
  .description("Show knowledge base statistics")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .option("--repo <owner/repo>", "Repository identifier (e.g., stakwork/hive)")
  .action(async (options) => {
    try {
      const storage = await createStorage(options);
      const repo = options.repo as string | undefined;

      const concepts = await storage.getAllConcepts(repo);
      const prs = await storage.getAllPRs(repo);

      console.log(`\n📊 Knowledge Base Statistics\n`);
      if (repo) {
        console.log(`Repository: ${repo}`);
      }
      console.log(`Total Concepts: ${concepts.length}`);
      console.log(`Total PRs: ${prs.length}`);
      if (repo) {
        const lastProcessed = await storage.getLastProcessedPR(repo);
        console.log(`Last Processed PR: #${lastProcessed}`);
      }

      if (concepts.length > 0) {
        const avgPRsPerConcept =
          concepts.reduce((sum, f) => sum + f.prNumbers.length, 0) /
          concepts.length;
        console.log(
          `Average PRs per Concept: ${avgPRsPerConcept.toFixed(1)}`
        );

        // Find most active concept
        const mostActive = concepts.reduce((max, f) =>
          f.prNumbers.length > max.prNumbers.length ? f : max
        );
        console.log(
          `\nMost Active Concept: ${mostActive.name} (${mostActive.prNumbers.length} PRs)`
        );
      }

      console.log();
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Summarize a single concept
 */
program
  .command("summarize")
  .description("Generate comprehensive documentation for a concept")
  .argument("<conceptId>", "Concept ID to summarize")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (conceptId: string, options) => {
    try {
      // Get Anthropic API key
      const anthropicKey = getApiKeyForProvider("anthropic");

      // Initialize components
      const storage = await createStorage(options);

      const summarizer = new Summarizer(storage, "anthropic", anthropicKey);

      // Summarize the concept
      await summarizer.summarizeConcept(conceptId);

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Summarize all concepts
 */
program
  .command("summarize-all")
  .description("Generate comprehensive documentation for all concepts")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (options) => {
    try {
      // Get Anthropic API key
      const anthropicKey = getApiKeyForProvider("anthropic");

      // Initialize components
      console.log(`\n🚀 Generating documentation for all concepts...`);

      const storage = await createStorage(options);

      const summarizer = new Summarizer(storage, "anthropic", anthropicKey);

      // Summarize all concepts
      await summarizer.summarizeAllConcepts();

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Link concepts to files
 */
program
  .command("link-files")
  .description("Link concepts to file nodes in the graph based on PR changes")
  .argument("[conceptId]", "Concept ID to link (optional, links all if omitted)")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (conceptId: string | undefined, options) => {
    try {
      console.log(`\n🚀 Linking concepts to files...`);

      const storage = await createStorage(options);
      const linker = new FileLinker(storage);

      // Link single concept or all
      if (conceptId) {
        await linker.linkConcept(conceptId);
      } else {
        await linker.linkAllConcepts();
      }

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Analyze concept(s) for architectural clues
 */
program
  .command("analyze-clues")
  .description("Analyze concept(s) for architectural clues (auto-links by default)")
  .argument("[conceptId]", "Concept ID to analyze (optional, analyzes all if omitted)")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .option("-f, --force", "Force re-analysis even if concept already has clues")
  .option("-r, --repo-path <path>", "Path to repository", process.cwd())
  .option("--no-link", "Skip automatic linking after analysis")
  .action(async (conceptId: string | undefined, options) => {
    try {
      const storage = await createStorage(options);
      const analyzer = new ClueAnalyzer(storage, options.repoPath);

      const autoLink = options.link !== false; // Commander sets to false with --no-link

      if (conceptId) {
        // Analyze single concept
        const result = await analyzer.analyzeConcept(conceptId);

        // Auto-link after single concept analysis
        if (autoLink && result.clues.length > 0) {
          console.log(`\n🔗 Auto-linking new clues to relevant concepts...\n`);
          const { ClueLinker } = await import("./clueLinker.js");
          const linker = new ClueLinker(storage);
          const newClueIds = result.clues.map((c) => c.id);
          await linker.linkClues(newClueIds);
        }
      } else {
        // Analyze all concepts (with auto-linking by default)
        await analyzer.analyzeAllConcepts(options.force, autoLink);
      }

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Retroactively analyze PRs/commits for clues
 */
program
  .command("analyze-changes")
  .description("Retroactively analyze historical PRs/commits for clues")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .option("-r, --repo-path <path>", "Path to repository", process.cwd())
  .option("--repo <owner/repo>", "Repository identifier (e.g., stakwork/hive) - required for checkpoint tracking")
  .option("-f, --force", "Force re-analysis of all changes (ignore checkpoint)")
  .action(async (options) => {
    try {
      const storage = await createStorage(options);
      const analyzer = new ClueAnalyzer(storage, options.repoPath);
      const repo = options.repo as string | undefined;

      if (!repo) {
        console.log(`\n⚠️  Warning: --repo not specified, checkpoint tracking will be disabled.\n`);
      }

      // Get checkpoint (unless force or no repo specified)
      const checkpoint = options.force || !repo
        ? null
        : await storage.getClueAnalysisCheckpoint(repo);

      if (checkpoint) {
        console.log(
          `\n📌 Resuming from checkpoint: ${checkpoint.lastProcessedTimestamp}`
        );
      } else if (options.force) {
        console.log(`\n🔄 Force mode: analyzing all changes\n`);
      } else {
        console.log(`\n🆕 No checkpoint found, starting from beginning\n`);
      }

      // Fetch all PRs and commits
      const allPRs = await storage.getAllPRs();
      const allCommits = await storage.getAllCommits();

      // Combine and sort chronologically
      const changes: Array<{
        type: "pr" | "commit";
        date: Date;
        id: string;
        data: any;
      }> = [];

      for (const pr of allPRs) {
        changes.push({
          type: "pr",
          date: pr.mergedAt,
          id: pr.number.toString(),
          data: pr,
        });
      }

      for (const commit of allCommits) {
        changes.push({
          type: "commit",
          date: commit.committedAt,
          id: commit.sha,
          data: commit,
        });
      }

      // Sort chronologically (oldest first)
      changes.sort((a, b) => a.date.getTime() - b.date.getTime());

      // Filter by checkpoint
      let changesToProcess = changes;
      if (checkpoint) {
        changesToProcess = changes.filter((change) => {
          const changeTime = change.date.toISOString();
          if (changeTime > checkpoint.lastProcessedTimestamp) {
            return true;
          }
          if (changeTime === checkpoint.lastProcessedTimestamp) {
            // Skip if already processed at this exact timestamp
            return !checkpoint.processedAtTimestamp.includes(change.id);
          }
          return false;
        });
      }

      if (changesToProcess.length === 0) {
        console.log(`\n✅ No new changes to analyze!\n`);
        return;
      }

      console.log(
        `\n📊 Analyzing ${changesToProcess.length} change(s) for clues...\n`
      );

      let totalClues = 0;
      let totalUsage: Usage = normalizeUsage();

      for (let i = 0; i < changesToProcess.length; i++) {
        const change = changesToProcess[i];
        const progress = `[${i + 1}/${changesToProcess.length}]`;

        if (change.type === "pr") {
          const pr = change.data;
          console.log(`\n${progress} PR #${pr.number}: ${pr.title}`);

          try {
            const changeContext = {
              type: "pr" as const,
              identifier: `#${pr.number}`,
              title: pr.title,
              summary: pr.summary,
              files: pr.files,
              // Note: We don't have comments/reviews saved in PRRecord
            };

            const result = await analyzer.analyzeChange(changeContext);
            totalClues += result.clues.length;
            totalUsage = normalizeUsage(addUsage(totalUsage, result.usage));

            // Link clues if any were created
            if (result.clues.length > 0) {
              const { ClueLinker } = await import("./clueLinker.js");
              const linker = new ClueLinker(storage);
              await linker.linkClues(result.clues.map((c) => c.id));
            }

            // Update checkpoint (only if repo is specified)
            if (repo) {
              await storage.setClueAnalysisCheckpoint(repo, {
                lastProcessedTimestamp: pr.mergedAt.toISOString(),
                processedAtTimestamp: [pr.number.toString()],
              });
            }
          } catch (error) {
            console.error(
              `   ❌ Error:`,
              error instanceof Error ? error.message : error
            );
            console.log(`   ⏭️  Skipping...`);
          }
        } else {
          const commit = change.data;
          console.log(
            `\n${progress} Commit ${commit.sha.substring(0, 7)}: ${
              commit.message.split("\n")[0]
            }`
          );

          try {
            const changeContext = {
              type: "commit" as const,
              identifier: commit.sha.substring(0, 7),
              title: commit.message.split("\n")[0],
              summary: commit.summary,
              files: commit.files,
            };

            const result = await analyzer.analyzeChange(changeContext);
            totalClues += result.clues.length;
            totalUsage = normalizeUsage(addUsage(totalUsage, result.usage));

            // Link clues if any were created
            if (result.clues.length > 0) {
              const { ClueLinker } = await import("./clueLinker.js");
              const linker = new ClueLinker(storage);
              await linker.linkClues(result.clues.map((c) => c.id));
            }

            // Update checkpoint (only if repo is specified)
            if (repo) {
              await storage.setClueAnalysisCheckpoint(repo, {
                lastProcessedTimestamp: commit.committedAt.toISOString(),
                processedAtTimestamp: [commit.sha],
              });
            }
          } catch (error) {
            console.error(
              `   ❌ Error:`,
              error instanceof Error ? error.message : error
            );
            console.log(`   ⏭️  Skipping...`);
          }
        }
      }

      console.log(`\n🎉 Analysis complete!`);
      console.log(`   Total clues created: ${totalClues}`);
      console.log(
        `   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`
      );
      console.log(`\n✅ Done!\n`);
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * List all clues or clues for a specific concept
 */
program
  .command("list-clues")
  .description("List all clues or clues for a specific concept")
  .argument("[conceptId]", "Concept ID to list clues for (optional)")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage")
  .action(async (conceptId: string | undefined, options) => {
    try {
      const storage = await createStorage(options);

      const clues = conceptId
        ? await storage.getCluesForConcept(conceptId)
        : await storage.getAllClues();

      if (clues.length === 0) {
        console.log("\n📭 No clues found.\n");
        return;
      }

      console.log(`\n💡 Clues (${clues.length} total):\n`);

      // Group by concept if listing all
      if (!conceptId) {
        const byConcept = new Map<string, typeof clues>();
        for (const clue of clues) {
          if (!byConcept.has(clue.conceptId)) {
            byConcept.set(clue.conceptId, []);
          }
          byConcept.get(clue.conceptId)!.push(clue);
        }

        for (const [fid, fclues] of byConcept.entries()) {
          const concept = await storage.getConcept(fid);
          console.log(`\n🔹 ${concept?.name || fid} (${fclues.length} clues)`);
          for (const clue of fclues) {
            console.log(`   - ${clue.title} [${clue.type}]`);
          }
        }
      } else {
        for (const clue of clues) {
          console.log(`🔹 ${clue.title} (${clue.id})`);
          console.log(`   Type: ${clue.type}`);
          console.log(`   Content: ${clue.content.substring(0, 100)}...`);
          console.log(`   Keywords: ${clue.keywords.join(", ")}`);
          console.log();
        }
      }

      console.log();
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Show details of a specific clue
 */
program
  .command("show-clue")
  .description("Show details of a specific clue")
  .argument("<clueId>", "Clue ID to show")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage")
  .action(async (clueId: string, options) => {
    try {
      const storage = await createStorage(options);
      const clue = await storage.getClue(clueId);

      if (!clue) {
        console.error(`\n❌ Clue not found: ${clueId}\n`);
        process.exit(1);
      }

      console.log(`\n💡 ${clue.title}`);
      console.log(`${"=".repeat(clue.title.length + 3)}\n`);
      console.log(`ID: ${clue.id}`);
      console.log(`Type: ${clue.type}`);
      console.log(`Concept: ${clue.conceptId}\n`);
      console.log(`Content:\n${clue.content}\n`);

      if (Object.keys(clue.entities).length > 0) {
        console.log("Entities:");
        for (const [key, values] of Object.entries(clue.entities)) {
          if (values && values.length > 0) {
            console.log(`  ${key}: ${values.join(", ")}`);
          }
        }
        console.log();
      }

      if (clue.files.length > 0) {
        console.log(`Files:\n  ${clue.files.join("\n  ")}\n`);
      }

      console.log(`Keywords: ${clue.keywords.join(", ")}\n`);

      if (clue.relatedClues.length > 0) {
        console.log(`Related Clues: ${clue.relatedClues.join(", ")}\n`);
      }

      if (clue.dependsOn.length > 0) {
        console.log(`Depends On: ${clue.dependsOn.join(", ")}\n`);
      }

      console.log(`Created: ${clue.createdAt.toISOString()}`);
      console.log(`Updated: ${clue.updatedAt.toISOString()}\n`);
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

program
  .command("link-clues")
  .description("Link clues to relevant concepts (Step 2 after analyze-clues)")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .option("-f, --force", "Force re-linking even if clues already have links")
  .action(async (options) => {
    try {
      const storage = await createStorage(options);
      const { ClueLinker } = await import("./clueLinker.js");
      const linker = new ClueLinker(storage);

      await linker.linkAllClues(options.force);

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

program
  .command("search-clues")
  .description("Search clues by relevance (embeddings + keywords + centrality)")
  .argument("<query>", "Search query")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage")
  .option("-f, --concept <id>", "Filter by concept ID")
  .option("-l, --limit <number>", "Maximum number of results", "10")
  .option("-t, --threshold <number>", "Similarity threshold (0-1)", "0.5")
  .action(async (query: string, options) => {
    try {
      const storage = await createStorage(options);
      const { vectorizeQuery } = await import("../vector/index.js");

      console.log(`\n🔍 Searching for: "${query}"\n`);

      // Generate embeddings
      const embeddings = await vectorizeQuery(query);

      // Search
      const results = await storage.searchClues(
        query,
        embeddings,
        options.concept,
        parseInt(options.limit),
        parseFloat(options.threshold)
      );

      if (results.length === 0) {
        console.log("📭 No clues found matching your query.\n");
        return;
      }

      console.log(`Found ${results.length} relevant clue(s):\n`);

      for (const result of results) {
        console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
        console.log(`📍 ${result.title} (${result.id})`);
        console.log(`   Score: ${result.score.toFixed(3)} | Type: ${result.type}`);

        if (result.relevanceBreakdown) {
          console.log(
            `   📊 Breakdown: Vector=${result.relevanceBreakdown.vector.toFixed(2)}, ` +
              `Keyword=${result.relevanceBreakdown.keyword.toFixed(2)}, ` +
              `Title=${result.relevanceBreakdown.title.toFixed(2)}, ` +
              `Centrality=${result.relevanceBreakdown.centrality.toFixed(2)}`
          );
        }

        console.log(`   ${result.content.substring(0, 150)}...`);
        console.log(`   Keywords: ${result.keywords.slice(0, 5).join(", ")}`);
        console.log();
      }

      console.log();
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

program.parse();
