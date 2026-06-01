import { Concept, PRRecord, CommitRecord, LinkResult, ChronologicalCheckpoint, Usage } from "../types.js";

/**
 * Abstract storage interface for features, PRs, and commits
 * 
 * Multi-repo support:
 * - All query methods accept an optional `repo` parameter to filter by repository
 * - When `repo` is not provided, queries return data from all repositories
 * - Metadata methods (checkpoints, themes) require a `repo` parameter for per-repo isolation
 */
export abstract class Storage {
  // Initialization
  abstract initialize(): Promise<void>;

  // Concepts
  abstract saveConcept(feature: Concept): Promise<void>;
  abstract getConcept(id: string, repo?: string): Promise<Concept | null>;
  abstract getAllConcepts(repo?: string): Promise<Concept[]>;
  abstract deleteConcept(id: string, repo?: string): Promise<void>;

  // PRs
  abstract savePR(pr: PRRecord): Promise<void>;
  abstract getPR(number: number, repo?: string): Promise<PRRecord | null>;
  abstract getAllPRs(repo?: string): Promise<PRRecord[]>;

  // Commits
  abstract saveCommit(commit: CommitRecord): Promise<void>;
  abstract getCommit(sha: string, repo?: string): Promise<CommitRecord | null>;
  abstract getAllCommits(repo?: string): Promise<CommitRecord[]>;

  // Metadata - now per-repo
  abstract getLastProcessedPR(repo: string): Promise<number>;
  abstract setLastProcessedPR(repo: string, number: number): Promise<void>;
  abstract getLastProcessedCommit(repo: string): Promise<string | null>;
  abstract setLastProcessedCommit(repo: string, sha: string): Promise<void>;

  // Chronological checkpoint - per-repo
  abstract getChronologicalCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null>;
  abstract setChronologicalCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void>;

  // Themes - per-repo
  abstract addThemes(repo: string, themes: string[]): Promise<void>;
  abstract getRecentThemes(repo: string): Promise<string[]>;

  // Total Usage - cumulative token usage across all processing runs, per-repo
  abstract getTotalUsage(repo: string): Promise<Usage>;
  abstract addToTotalUsage(repo: string, usage: Usage): Promise<void>;

  // Get aggregated metadata across all repos (latest timestamp, summed usage)
  abstract getAggregatedMetadata(): Promise<{
    lastProcessedTimestamp: string | null;
    cumulativeUsage: Usage;
  }>;

  // Documentation
  abstract saveDocumentation(featureId: string, documentation: string): Promise<void>;

  // Concept-File Linking
  abstract linkConceptsToFiles(featureId?: string, repo?: string): Promise<LinkResult>;

  // Link a feature to files by explicit file paths (used by bootstrap)
  abstract linkConceptToFilesByPaths(featureId: string, filePaths: string[]): Promise<number>;

  // Get Files for Concept
  abstract getFilesForConcept(featureId: string, expand?: string[]): Promise<any[]>;

  // Query helpers (derived from the graph)
  async getPRsForConcept(featureId: string, repo?: string): Promise<PRRecord[]> {
    const feature = await this.getConcept(featureId, repo);
    if (!feature) return [];

    const prs: PRRecord[] = [];
    for (const prNumber of feature.prNumbers) {
      const pr = await this.getPR(prNumber, feature.repo);
      if (pr) prs.push(pr);
    }
    return prs;
  }

  async getConceptsForPR(prNumber: number, repo?: string): Promise<Concept[]> {
    const allConcepts = await this.getAllConcepts(repo);
    return allConcepts.filter((f) => f.prNumbers.includes(prNumber));
  }

  async getCommitsForConcept(featureId: string, repo?: string): Promise<CommitRecord[]> {
    const feature = await this.getConcept(featureId, repo);
    if (!feature) return [];

    const commits: CommitRecord[] = [];
    // Handle legacy features without commitShas
    const commitShas = feature.commitShas || [];
    for (const sha of commitShas) {
      const commit = await this.getCommit(sha, feature.repo);
      if (commit) commits.push(commit);
    }
    return commits;
  }

  async getConceptsForCommit(sha: string, repo?: string): Promise<Concept[]> {
    const allConcepts = await this.getAllConcepts(repo);
    // Handle legacy concepts without commitShas
    return allConcepts.filter((f) => (f.commitShas || []).includes(sha));
  }
}
