#!/usr/bin/env node

import { Command } from "commander";
import { Octokit } from "@octokit/rest";
import { Storage, FileSystemStore, GraphStorage } from "./store/index.js";
import { LLMClient } from "./llm.js";
import { StreamingFeatureBuilder } from "./builder.js";
import { Summarizer } from "./summarizer.js";
import { FileLinker } from "./fileLinker.js";
import { getApiKeyForProvider } from "../aieo/src/provider.js";

const program = new Command();

program
  .name("gitree")
  .description("GitHub Feature Knowledge Base - Extract features from PR history")
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
  .description("Process a GitHub repository to extract features")
  .argument("<owner>", "Repository owner")
  .argument("<repo>", "Repository name")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .option("-t, --token <token>", "GitHub token (or set GITHUB_TOKEN env)")
  .option("-c, --commits", "Also process orphan commits (not in PRs)")
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
      console.log(`\n🚀 Initializing GitHub Feature Knowledge Base...`);
      console.log(`   Repository: ${owner}/${repo}`);

      const storage = await createStorage(options);

      const octokit = new Octokit({ auth: githubToken });
      const llm = new LLMClient("anthropic", anthropicKey);
      const builder = new StreamingFeatureBuilder(storage, llm, octokit);

      // Process repo (PRs)
      await builder.processRepo(owner, repo);

      // Process commits if requested
      if (options.commits) {
        console.log("\n\n🔍 Processing orphan commits...\n");
        await builder.processCommits(owner, repo);
      }

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * List all features
 */
program
  .command("list-features")
  .description("List all features in the knowledge base")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (options) => {
    try {
      const storage = await createStorage(options);

      const features = await storage.getAllFeatures();

      if (features.length === 0) {
        console.log("No features found.");
        return;
      }

      console.log(`\n📚 Features (${features.length} total):\n`);

      const sorted = features.sort(
        (a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime()
      );

      for (const feature of sorted) {
        console.log(`🔹 ${feature.name} (${feature.id})`);
        console.log(`   ${feature.description}`);
        const changesSummary =
          feature.commitShas.length > 0
            ? `PRs: ${feature.prNumbers.length} | Commits: ${feature.commitShas.length}`
            : `PRs: ${feature.prNumbers.length}`;
        console.log(
          `   ${changesSummary} | Last updated: ${feature.lastUpdated.toISOString().split("T")[0]}`
        );
        console.log();
      }
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Show details of a specific feature
 */
program
  .command("show-feature")
  .description("Show details of a specific feature")
  .argument("<featureId>", "Feature ID")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (featureId: string, options) => {
    try {
      const storage = await createStorage(options);

      const feature = await storage.getFeature(featureId);

      if (!feature) {
        console.log(`❌ Feature not found: ${featureId}`);
        return;
      }

      console.log(`\n📖 Feature: ${feature.name}\n`);
      console.log(`ID: ${feature.id}`);
      console.log(`Description: ${feature.description}`);
      console.log(`Created: ${feature.createdAt.toISOString().split("T")[0]}`);
      console.log(`Last Updated: ${feature.lastUpdated.toISOString().split("T")[0]}`);
      console.log(`\nPull Requests (${feature.prNumbers.length}):\n`);

      const prs = await storage.getPRsForFeature(featureId);
      for (const pr of prs) {
        console.log(`  #${pr.number}: ${pr.title}`);
        console.log(`     ${pr.summary}`);
        console.log(`     ${pr.url}`);
        console.log();
      }

      if (feature.commitShas.length > 0) {
        console.log(`\nCommits (${feature.commitShas.length}):\n`);

        const commits = await storage.getCommitsForFeature(featureId);
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

      const features = await storage.getFeaturesForPR(prNumber);

      console.log(`\n📄 PR #${pr.number}: ${pr.title}\n`);
      console.log(`Summary: ${pr.summary}`);
      console.log(`Merged: ${pr.mergedAt.toISOString().split("T")[0]}`);
      console.log(`URL: ${pr.url}`);

      if (features.length > 0) {
        console.log(`\nPart of ${features.length} feature(s):\n`);
        for (const feature of features) {
          console.log(`  🔹 ${feature.name} (${feature.id})`);
        }
      } else {
        console.log(`\nNot associated with any features.`);
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

      const features = await storage.getFeaturesForCommit(sha);

      console.log(`\n📝 Commit ${commit.sha.substring(0, 7)}: ${commit.message.split('\n')[0]}\n`);
      console.log(`Summary: ${commit.summary}`);
      console.log(`Author: ${commit.author}`);
      console.log(`Committed: ${commit.committedAt.toISOString().split("T")[0]}`);
      console.log(`URL: ${commit.url}`);

      if (features.length > 0) {
        console.log(`\nPart of ${features.length} feature(s):\n`);
        for (const feature of features) {
          console.log(`  🔹 ${feature.name} (${feature.id})`);
        }
      } else {
        console.log(`\nNot associated with any features.`);
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
  .action(async (options) => {
    try {
      const storage = await createStorage(options);

      const features = await storage.getAllFeatures();
      const prs = await storage.getAllPRs();
      const lastProcessed = await storage.getLastProcessedPR();

      console.log(`\n📊 Knowledge Base Statistics\n`);
      console.log(`Total Features: ${features.length}`);
      console.log(`Total PRs: ${prs.length}`);
      console.log(`Last Processed PR: #${lastProcessed}`);

      if (features.length > 0) {
        const avgPRsPerFeature =
          features.reduce((sum, f) => sum + f.prNumbers.length, 0) /
          features.length;
        console.log(
          `Average PRs per Feature: ${avgPRsPerFeature.toFixed(1)}`
        );

        // Find most active feature
        const mostActive = features.reduce((max, f) =>
          f.prNumbers.length > max.prNumbers.length ? f : max
        );
        console.log(
          `\nMost Active Feature: ${mostActive.name} (${mostActive.prNumbers.length} PRs)`
        );
      }

      console.log();
    } catch (error) {
      console.error("❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Summarize a single feature
 */
program
  .command("summarize")
  .description("Generate comprehensive documentation for a feature")
  .argument("<featureId>", "Feature ID to summarize")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (featureId: string, options) => {
    try {
      // Get Anthropic API key
      const anthropicKey = getApiKeyForProvider("anthropic");

      // Initialize components
      const storage = await createStorage(options);

      const summarizer = new Summarizer(storage, "anthropic", anthropicKey);

      // Summarize the feature
      await summarizer.summarizeFeature(featureId);

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Summarize all features
 */
program
  .command("summarize-all")
  .description("Generate comprehensive documentation for all features")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (options) => {
    try {
      // Get Anthropic API key
      const anthropicKey = getApiKeyForProvider("anthropic");

      // Initialize components
      console.log(`\n🚀 Generating documentation for all features...`);

      const storage = await createStorage(options);

      const summarizer = new Summarizer(storage, "anthropic", anthropicKey);

      // Summarize all features
      await summarizer.summarizeAllFeatures();

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

/**
 * Link features to files
 */
program
  .command("link-files")
  .description("Link features to file nodes in the graph based on PR changes")
  .argument("[featureId]", "Feature ID to link (optional, links all if omitted)")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .option("-g, --graph", "Use Neo4j GraphStorage instead of FileSystemStorage")
  .action(async (featureId: string | undefined, options) => {
    try {
      console.log(`\n🚀 Linking features to files...`);

      const storage = await createStorage(options);
      const linker = new FileLinker(storage);

      // Link single feature or all
      if (featureId) {
        await linker.linkFeature(featureId);
      } else {
        await linker.linkAllFeatures();
      }

      console.log("\n✅ Done!\n");
    } catch (error) {
      console.error("\n❌ Error:", error);
      process.exit(1);
    }
  });

program.parse();
