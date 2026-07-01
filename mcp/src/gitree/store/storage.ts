import { Concept, PRRecord, CommitRecord, Clue, LinkResult, ChronologicalCheckpoint, Usage } from "../types.js";

/**
 * Abstract storage interface for concepts, PRs, and commits
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
  abstract saveConcept(concept: Concept): Promise<void>;
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

  // Clues
  abstract saveClue(clue: Clue): Promise<void>;
  abstract getClue(id: string, repo?: string): Promise<Clue | null>;
  abstract getAllClues(repo?: string): Promise<Clue[]>;
  abstract deleteClue(id: string, repo?: string): Promise<void>;
  abstract searchClues(
    query: string,
    embeddings: number[],
    conceptId?: string,
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
  abstract getTotalUsage(repo: string): Promise<Usage>;
  abstract addToTotalUsage(repo: string, usage: Usage): Promise<void>;

  // Get aggregated metadata across all repos (latest timestamp, summed usage)
  abstract getAggregatedMetadata(): Promise<{
    lastProcessedTimestamp: string | null;
    cumulativeUsage: Usage;
  }>;

  // Documentation
  abstract saveDocumentation(conceptId: string, documentation: string): Promise<void>;

  // Concept-File Linking
  abstract linkConceptsToFiles(conceptId?: string, repo?: string): Promise<LinkResult>;

  // Link a concept to files by explicit file paths (used by bootstrap)
  abstract linkConceptToFilesByPaths(conceptId: string, filePaths: string[]): Promise<number>;

  // Direct PR->File Linking (deterministic, based on each PR's `files` array).
  // Scopes matching to each PR's own repo, so a repo-less (null) `repo` arg is
  // still safe to run across a multi-repo swarm. Used both incrementally after
  // ingestion and as a backfill for existing PRs.
  abstract linkPRsToFiles(repo?: string): Promise<{ prsProcessed: number; edgesLinked: number }>;

  // Get Files for Concept
  abstract getFilesForConcept(conceptId: string, expand?: string[]): Promise<any[]>;

  // Query helpers (derived from the graph)
  async getPRsForConcept(conceptId: string, repo?: string): Promise<PRRecord[]> {
    const concept = await this.getConcept(conceptId, repo);
    if (!concept) return [];

    const prs: PRRecord[] = [];
    for (const prNumber of concept.prNumbers) {
      const pr = await this.getPR(prNumber, concept.repo);
      if (pr) prs.push(pr);
    }
    return prs;
  }

  async getConceptsForPR(prNumber: number, repo?: string): Promise<Concept[]> {
    const allConcepts = await this.getAllConcepts(repo);
    return allConcepts.filter((f) => f.prNumbers.includes(prNumber));
  }

  async getCommitsForConcept(conceptId: string, repo?: string): Promise<CommitRecord[]> {
    const concept = await this.getConcept(conceptId, repo);
    if (!concept) return [];

    const commits: CommitRecord[] = [];
    // Handle legacy concepts without commitShas
    const commitShas = concept.commitShas || [];
    for (const sha of commitShas) {
      const commit = await this.getCommit(sha, concept.repo);
      if (commit) commits.push(commit);
    }
    return commits;
  }

  async getConceptsForCommit(sha: string, repo?: string): Promise<Concept[]> {
    const allConcepts = await this.getAllConcepts(repo);
    // Handle legacy concepts without commitShas
    return allConcepts.filter((f) => (f.commitShas || []).includes(sha));
  }

  async getCluesForConcept(conceptId: string, limit?: number, repo?: string): Promise<Clue[]> {
    const allClues = await this.getAllClues(repo);
    // Get clues that are RELEVANT to this concept (not just discovered in it)
    const filtered = allClues.filter((c) => c.relatedConcepts.includes(conceptId));
    // Apply limit if specified (most recent first, already ordered by createdAt DESC)
    return limit ? filtered.slice(0, limit) : filtered;
  }
}
