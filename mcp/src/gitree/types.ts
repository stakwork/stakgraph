export interface Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

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
 * Core types for the GitHub Feature Knowledge Base
 */

export interface Feature {
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
  cluesCount?: number; // Number of clues for this feature
  cluesLastAnalyzedAt?: Date; // Last time clues were generated
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
}

export interface LLMDecision {
  actions: ("add_to_existing" | "create_new" | "ignore")[]; // Can have multiple actions
  existingFeatureIds?: string[]; // Which features to add to
  newFeatures?: Array<{
    // Which features to create
    name: string;
    description: string;
  }>;
  updateFeatures?: Array<{
    // Features whose descriptions need updating
    featureId: string;
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
 * Clue types for categorizing architectural insights
 */
export type ClueType =
  | "utility" // Reusable functions/classes
  | "pattern" // Architectural or code pattern
  | "abstraction" // Interface/type/base class to extend
  | "integration" // How to integrate with external system
  | "convention" // Coding convention/style
  | "gotcha" // Common mistake or edge case
  | "data-flow" // How data moves through code
  | "state-pattern"; // State management approach

/**
 * Clue entity types - actual code entities referenced (not code snippets!)
 */
export interface ClueEntities {
  functions?: string[]; // Function names (e.g., "generateToken", "verifyToken")
  classes?: string[]; // Class names (e.g., "JWTManager", "TokenValidator")
  types?: string[]; // Type names (e.g., "TokenPayload", "JWTConfig")
  interfaces?: string[]; // Interface names (e.g., "ITokenProvider")
  components?: string[]; // Component names (e.g., "AuthProvider", "TokenRefresher")
  endpoints?: string[]; // API endpoint paths (e.g., "POST /auth/refresh")
  tables?: string[]; // Database table names (e.g., "users", "refresh_tokens")
  constants?: string[]; // Constant names (e.g., "TOKEN_EXPIRY")
  hooks?: string[]; // React/Vue hook names (e.g., "useAuth")
}

/**
 * Clue - a knowledge node about architectural patterns and utilities
 */
export interface Clue {
  id: string; // Slug from title, repo-prefixed for uniqueness (e.g., "owner/repo/auth-jwt-utils")
  repo?: string; // Repository identifier "owner/repo" - optional for backwards compat
  featureId: string; // Feature where this clue was discovered (provenance)
  type: ClueType;
  title: string; // e.g., "JWT Token Management Utilities"
  content: string; // Markdown explanation (WHY, WHEN, CONTEXT)

  // Code entity references (NOT code snippets!)
  entities: ClueEntities;

  // File organization
  files: string[]; // Associated files

  // Discovery metadata
  keywords: string[]; // For searching
  centrality?: number; // How central/important (0-1)
  usageFrequency?: number; // How often entities are used
  embedding?: number[]; // Vector embedding of title for semantic search

  // Relationships
  relatedFeatures: string[]; // Feature IDs this clue is relevant to (for RELEVANT_TO edges)
  relatedClues: string[]; // Other clue IDs
  dependsOn: string[]; // Clues to understand first

  createdAt: Date;
  updatedAt: Date;
}

/**
 * Result from clue analysis
 */
export interface ClueAnalysisResult {
  clues: Clue[];
  complete: boolean; // Agent thinks feature is comprehensive
  reasoning: string;
  usage: Usage;
}

/**
 * Request body for /gitree/provenance endpoint
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
  id: string; // Feature ID (slug, e.g., "auth-system")
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
