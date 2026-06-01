import type { AiUsageWithLegacy } from "../../aieo/src/usage.js";

export type Usage = AiUsageWithLegacy;

export interface LinkResult {
  featuresProcessed: number;
  filesLinked: number;
  filesInDocs: number;
  filesNotInDocs: number;
  featureFileLinks: Array<{
    featureId: string;
    filesLinked: number;
    filesInDocs: number;
    filesNotInDocs: number;
  }>;
}

/**
 * Core types for the GitHub Concept Knowledge Base
 */

export interface Concept {
  id: string; // Slug from name, repo-prefixed for uniqueness (e.g., "owner/repo/auth-system")
  repo?: string; // Repository identifier "owner/repo" - optional for backwards compat
  ref_id?: string; // Reference ID from the repository (e.g., "1234567890")
  name: string; // Human-readable (e.g., "Authentication System")
  description: string; // What this feature is about
  prNumbers: number[]; // All PRs that touched this feature
  commitShas?: string[]; // All commits (not in PRs) that touched this feature (optional for legacy features)
  createdAt: Date;
  lastUpdated: Date;
  documentation?: string; // LLM-generated comprehensive documentation of current state
  usage?: Usage; // Token usage for summarizing this concept
}

export interface PRRecord {
  number: number;
  repo?: string; // Repository identifier "owner/repo" - optional for backwards compat
  title: string;
  summary: string; // LLM-generated summary of what this PR does
  mergedAt: Date;
  url: string;
  files: string[]; // List of files changed in this PR
  newDeclarations?: Array<{
    file: string;
    declarations: string[];
  }>;
  usage?: Usage; // Token usage for processing this PR
}

export interface CommitRecord {
  sha: string;
  repo?: string; // Repository identifier "owner/repo" - optional for backwards compat
  message: string;
  summary: string; // LLM-generated summary of what this commit does
  committedAt: Date;
  author: string;
  url: string;
  files: string[]; // List of files changed in this commit
  newDeclarations?: Array<{
    file: string;
    declarations: string[];
  }>;
  usage?: Usage; // Token usage for processing this commit
}

export interface LLMDecision {
  actions: ("add_to_existing" | "create_new" | "ignore")[]; // Can have multiple actions
  existingConceptIds?: string[]; // Which features to add to
  newConcepts?: Array<{
    // Which features to create
    name: string;
    description: string;
  }>;
  updateConcepts?: Array<{
    // Concepts whose descriptions need updating
    conceptId: string;
    newDescription: string;
    reasoning: string;
  }>;
  themes?: string[]; // 1-3 theme tags for context (can be new or existing)
  summary: string; // Summary of the PR itself
  reasoning: string;
  newDeclarations?: Array<{
    file: string;
    declarations: string[];
  }>;
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
 * GitHub Commit data structure (from Octokit)
 */
export interface GitHubCommit {
  sha: string;
  message: string;
  url: string;
  committedAt: Date;
  author: string;
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

/**
 * Chronological checkpoint for unified PR and commit processing
 */
export interface ChronologicalCheckpoint {
  lastProcessedTimestamp: string; // ISO date string of last processed item
  processedAtTimestamp: string[]; // IDs (PR numbers or commit SHAs) processed at exact timestamp
}

/**
 * Request body for /concepts/provenance endpoint
 */
export interface ProvenanceRequest {
  conceptIds: string[]; // Array of feature IDs (slugs)
}

/**
 * Code entity in provenance response
 */
export interface ProvenanceCodeEntity {
  refId: string;
  name: string;
  nodeType:
    | "Function"
    | "Page"
    | "Endpoint"
    | "Datamodel"
    | "UnitTest"
    | "IntegrationTest"
    | "E2etest";
  file: string;
  start: number;
  end: number;
}

/**
 * File with code entities in provenance response
 */
export interface ProvenanceFile {
  refId: string;
  name: string;
  path: string;
  codeEntities: ProvenanceCodeEntity[];
}

/**
 * Concept with files in provenance response
 */
export interface ProvenanceConcept {
  id: string; // Concept ID (slug, e.g., "auth-system")
  name: string;
  description?: string;
  files: ProvenanceFile[];
}

/**
 * Response from /gitree/provenance endpoint
 */
export interface ProvenanceResponse {
  concepts: ProvenanceConcept[];
}
