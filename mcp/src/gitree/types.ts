/**
 * Core types for the GitHub Feature Knowledge Base
 */

export interface Feature {
  id: string; // Slug from name (e.g., "auth-system")
  name: string; // Human-readable (e.g., "Authentication System")
  description: string; // What this feature is about
  prNumbers: number[]; // All PRs that touched this feature
  createdAt: Date;
  lastUpdated: Date;
}

export interface PRRecord {
  number: number;
  title: string;
  summary: string; // LLM-generated summary of what this PR does
  mergedAt: Date;
  url: string;
}

export interface LLMDecision {
  actions: ("add_to_existing" | "create_new" | "ignore")[]; // Can have multiple actions
  existingFeatureIds?: string[]; // Which features to add to
  newFeatures?: Array<{
    // Which features to create
    name: string;
    description: string;
  }>;
  summary: string; // Summary of the PR itself
  reasoning: string;
}

/**
 * GitHub PR data structure (from Octokit)
 */
export interface GitHubPR {
  number: number;
  title: string;
  body: string | null;
  url: string;
  mergedAt: Date;
  additions: number;
  deletions: number;
  filesChanged: string[];
}

/**
 * Options for fetching PRs
 */
export interface FetchPROptions {
  since: number;
  state: "open" | "closed" | "all";
  sort: "created" | "updated";
  direction: "asc" | "desc";
}
