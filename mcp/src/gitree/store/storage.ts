import { Feature, PRRecord, CommitRecord, Clue, LinkResult, ChronologicalCheckpoint } from "../types.js";

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

  // Features
  abstract saveFeature(feature: Feature): Promise<void>;
  abstract getFeature(id: string, repo?: string): Promise<Feature | null>;
  abstract getAllFeatures(repo?: string): Promise<Feature[]>;
  abstract deleteFeature(id: string, repo?: string): Promise<void>;

  // PRs
  abstract savePR(pr: PRRecord): Promise<void>;
  abstract getPR(number: number, repo?: string): Promise<PRRecord | null>;
  abstract getAllPRs(repo?: string): Promise<PRRecord[]>;

  // Commits
  abstract saveCommit(commit: CommitRecord): Promise<void>;
  abstract getCommit(sha: string, repo?: string): Promise<CommitRecord | null>;
  abstract getAllCommits(repo?: string): Promise<CommitRecord[]>;

  // Clues
  abstract saveClue(clue: Clue): Promise<void>;
  abstract getClue(id: string, repo?: string): Promise<Clue | null>;
  abstract getAllClues(repo?: string): Promise<Clue[]>;
  abstract deleteClue(id: string, repo?: string): Promise<void>;
  abstract searchClues(
    query: string,
    embeddings: number[],
    featureId?: string,
    limit?: number,
    similarityThreshold?: number,
    repo?: string
  ): Promise<Array<Clue & { score: number; relevanceBreakdown?: any }>>;

  // Metadata - now per-repo
  abstract getLastProcessedPR(repo: string): Promise<number>;
  abstract setLastProcessedPR(repo: string, number: number): Promise<void>;
  abstract getLastProcessedCommit(repo: string): Promise<string | null>;
  abstract setLastProcessedCommit(repo: string, sha: string): Promise<void>;

  // Chronological checkpoint - per-repo
  abstract getChronologicalCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null>;
  abstract setChronologicalCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void>;

  // Clue analysis checkpoint - per-repo
  abstract getClueAnalysisCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null>;
  abstract setClueAnalysisCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void>;

  // Themes - per-repo
  abstract addThemes(repo: string, themes: string[]): Promise<void>;
  abstract getRecentThemes(repo: string): Promise<string[]>;

  // Total Usage - cumulative token usage across all processing runs, per-repo
  abstract getTotalUsage(repo: string): Promise<{ inputTokens: number; outputTokens: number; totalTokens: number }>;
  abstract addToTotalUsage(repo: string, usage: { inputTokens: number; outputTokens: number; totalTokens: number }): Promise<void>;

  // Get aggregated metadata across all repos (latest timestamp, summed usage)
  abstract getAggregatedMetadata(): Promise<{
    lastProcessedTimestamp: string | null;
    cumulativeUsage: { inputTokens: number; outputTokens: number; totalTokens: number };
  }>;

  // Documentation
  abstract saveDocumentation(featureId: string, documentation: string): Promise<void>;

  // Feature-File Linking
  abstract linkFeaturesToFiles(featureId?: string, repo?: string): Promise<LinkResult>;

  // Get Files for Feature
  abstract getFilesForFeature(featureId: string, expand?: string[]): Promise<any[]>;

  // Query helpers (derived from the graph)
  async getPRsForFeature(featureId: string, repo?: string): Promise<PRRecord[]> {
    const feature = await this.getFeature(featureId, repo);
    if (!feature) return [];

    const prs: PRRecord[] = [];
    for (const prNumber of feature.prNumbers) {
      const pr = await this.getPR(prNumber, feature.repo);
      if (pr) prs.push(pr);
    }
    return prs;
  }

  async getFeaturesForPR(prNumber: number, repo?: string): Promise<Feature[]> {
    const allFeatures = await this.getAllFeatures(repo);
    return allFeatures.filter((f) => f.prNumbers.includes(prNumber));
  }

  async getCommitsForFeature(featureId: string, repo?: string): Promise<CommitRecord[]> {
    const feature = await this.getFeature(featureId, repo);
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

  async getFeaturesForCommit(sha: string, repo?: string): Promise<Feature[]> {
    const allFeatures = await this.getAllFeatures(repo);
    // Handle legacy features without commitShas
    return allFeatures.filter((f) => (f.commitShas || []).includes(sha));
  }

  async getCluesForFeature(featureId: string, limit?: number, repo?: string): Promise<Clue[]> {
    const allClues = await this.getAllClues(repo);
    // Get clues that are RELEVANT to this feature (not just discovered in it)
    const filtered = allClues.filter((c) => c.relatedFeatures.includes(featureId));
    // Apply limit if specified (most recent first, already ordered by createdAt DESC)
    return limit ? filtered.slice(0, limit) : filtered;
  }
}
