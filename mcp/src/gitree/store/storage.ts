import { Feature, PRRecord, CommitRecord, LinkResult, ChronologicalCheckpoint } from "../types.js";

/**
 * Abstract storage interface for features, PRs, and commits
 */
export abstract class Storage {
  // Initialization
  abstract initialize(): Promise<void>;

  // Features
  abstract saveFeature(feature: Feature): Promise<void>;
  abstract getFeature(id: string): Promise<Feature | null>;
  abstract getAllFeatures(): Promise<Feature[]>;
  abstract deleteFeature(id: string): Promise<void>;

  // PRs
  abstract savePR(pr: PRRecord): Promise<void>;
  abstract getPR(number: number): Promise<PRRecord | null>;
  abstract getAllPRs(): Promise<PRRecord[]>;

  // Commits
  abstract saveCommit(commit: CommitRecord): Promise<void>;
  abstract getCommit(sha: string): Promise<CommitRecord | null>;
  abstract getAllCommits(): Promise<CommitRecord[]>;

  // Metadata (legacy - kept for backwards compatibility)
  abstract getLastProcessedPR(): Promise<number>;
  abstract setLastProcessedPR(number: number): Promise<void>;
  abstract getLastProcessedCommit(): Promise<string | null>;
  abstract setLastProcessedCommit(sha: string): Promise<void>;

  // Chronological checkpoint (new unified approach)
  abstract getChronologicalCheckpoint(): Promise<ChronologicalCheckpoint | null>;
  abstract setChronologicalCheckpoint(checkpoint: ChronologicalCheckpoint): Promise<void>;

  // Themes (sliding window of recent technical tags)
  abstract addThemes(themes: string[]): Promise<void>;
  abstract getRecentThemes(): Promise<string[]>;

  // Documentation
  abstract saveDocumentation(featureId: string, documentation: string): Promise<void>;

  // Feature-File Linking
  abstract linkFeaturesToFiles(featureId?: string): Promise<LinkResult>;

  // Get Files for Feature
  abstract getFilesForFeature(featureId: string, expand?: string[]): Promise<any[]>;

  // Query helpers (derived from the graph)
  async getPRsForFeature(featureId: string): Promise<PRRecord[]> {
    const feature = await this.getFeature(featureId);
    if (!feature) return [];

    const prs: PRRecord[] = [];
    for (const prNumber of feature.prNumbers) {
      const pr = await this.getPR(prNumber);
      if (pr) prs.push(pr);
    }
    return prs;
  }

  async getFeaturesForPR(prNumber: number): Promise<Feature[]> {
    const allFeatures = await this.getAllFeatures();
    return allFeatures.filter((f) => f.prNumbers.includes(prNumber));
  }

  async getCommitsForFeature(featureId: string): Promise<CommitRecord[]> {
    const feature = await this.getFeature(featureId);
    if (!feature) return [];

    const commits: CommitRecord[] = [];
    for (const sha of feature.commitShas) {
      const commit = await this.getCommit(sha);
      if (commit) commits.push(commit);
    }
    return commits;
  }

  async getFeaturesForCommit(sha: string): Promise<Feature[]> {
    const allFeatures = await this.getAllFeatures();
    return allFeatures.filter((f) => f.commitShas.includes(sha));
  }
}
