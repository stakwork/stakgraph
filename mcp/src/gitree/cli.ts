#!/usr/bin/env node

import { Command } from "commander";
import { Octokit } from "@octokit/rest";
import { FileSystemStore } from "./storage.js";
import { LLMClient } from "./llm.js";
import { StreamingFeatureBuilder } from "./builder.js";
import { getApiKeyForProvider } from "../aieo/src/provider.js";

const program = new Command();

program
  .name("gitree")
  .description("GitHub Feature Knowledge Base - Extract features from PR history")
  .version("1.0.0");

/**
 * Process a repository
 */
program
  .command("process")
  .description("Process a GitHub repository to extract features")
  .argument("<owner>", "Repository owner")
  .argument("<repo>", "Repository name")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
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
      console.log(`\n🚀 Initializing GitHub Feature Knowledge Base...`);
      console.log(`   Repository: ${owner}/${repo}`);
      console.log(`   Storage: ${options.dir}\n`);

      const storage = new FileSystemStore(options.dir);
      await storage.initialize();

      const octokit = new Octokit({ auth: githubToken });
      const llm = new LLMClient("anthropic", anthropicKey);
      const builder = new StreamingFeatureBuilder(storage, llm, octokit);

      // Process repo
      await builder.processRepo(owner, repo);

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
  .action(async (options) => {
    try {
      const storage = new FileSystemStore(options.dir);
      await storage.initialize();

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
        console.log(
          `   PRs: ${feature.prNumbers.length} | Last updated: ${feature.lastUpdated.toISOString().split("T")[0]}`
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
  .action(async (featureId: string, options) => {
    try {
      const storage = new FileSystemStore(options.dir);
      await storage.initialize();

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
  .action(async (number: string, options) => {
    try {
      const prNumber = parseInt(number);
      const storage = new FileSystemStore(options.dir);
      await storage.initialize();

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
 * Show statistics
 */
program
  .command("stats")
  .description("Show knowledge base statistics")
  .option("-d, --dir <path>", "Knowledge base directory", "./knowledge-base")
  .action(async (options) => {
    try {
      const storage = new FileSystemStore(options.dir);
      await storage.initialize();

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

program.parse();
