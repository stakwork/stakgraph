import fs from "fs/promises";
import path from "path";
import { Concept, PRRecord, CommitRecord, Clue, LinkResult, ChronologicalCheckpoint, Usage } from "../types.js";
import { Storage } from "./storage.js";
import { formatPRMarkdown, parsePRMarkdown, formatCommitMarkdown, parseCommitMarkdown } from "./utils.js";
import { addUsage, normalizeUsage } from "../../aieo/src/usage.js";

function normalizeConcept(concept: any): Concept {
  return {
    ...concept,
    createdAt: new Date(concept.createdAt),
    lastUpdated: new Date(concept.lastUpdated),
    cluesLastAnalyzedAt: concept.cluesLastAnalyzedAt
      ? new Date(concept.cluesLastAnalyzedAt)
      : undefined,
    usage: concept.usage ? normalizeUsage(concept.usage) : undefined,
  };
}

/**
 * File system based storage implementation
 *
 * Directory structure:
 * ./knowledge-base/
 *   ├── metadata.json
 *   ├── concepts/
 *   │   ├── auth-system.json
 *   │   └── ...
 *   ├── prs/
 *   │   ├── 1.md
 *   │   └── ...
 *   ├── commits/
 *   │   ├── abc1234.md
 *   │   └── ...
 *   └── docs/
 *       ├── auth-system.md
 *       └── ...
 */
export class FileSystemStore extends Storage {
  private conceptsDir: string;
  private prsDir: string;
  private commitsDir: string;
  private cluesDir: string;
  private docsDir: string;
  private metadataPath: string;

  constructor(baseDir: string = "./knowledge-base") {
    super();
    this.conceptsDir = path.join(baseDir, "concepts");
    this.prsDir = path.join(baseDir, "prs");
    this.commitsDir = path.join(baseDir, "commits");
    this.cluesDir = path.join(baseDir, "clues");
    this.docsDir = path.join(baseDir, "docs");
    this.metadataPath = path.join(baseDir, "metadata.json");
  }

  /**
   * Initialize directory structure
   */
  async initialize(): Promise<void> {
    await fs.mkdir(this.conceptsDir, { recursive: true });
    await fs.mkdir(this.prsDir, { recursive: true });
    await fs.mkdir(this.commitsDir, { recursive: true });
    await fs.mkdir(this.cluesDir, { recursive: true });
    await fs.mkdir(this.docsDir, { recursive: true });

    try {
      await fs.access(this.metadataPath);
    } catch {
      await fs.writeFile(
        this.metadataPath,
        JSON.stringify({
          lastProcessedPR: 0,
          lastProcessedCommit: null,
          chronologicalCheckpoint: null,
          recentThemes: []
        }, null, 2)
      );
    }
  }

  // Concepts
  async saveConcept(concept: Concept): Promise<void> {
    // Use sanitized ID for filename (replace / with __)
    const safeId = concept.id.replace(/\//g, "__");
    const filePath = path.join(this.conceptsDir, `${safeId}.json`);
    const serialized = {
      ...concept,
      createdAt: concept.createdAt.toISOString(),
      lastUpdated: concept.lastUpdated.toISOString(),
      cluesLastAnalyzedAt: concept.cluesLastAnalyzedAt?.toISOString(),
      usage: concept.usage ? normalizeUsage(concept.usage) : undefined,
    };
    await fs.writeFile(filePath, JSON.stringify(serialized, null, 2));
  }

  async getConcept(id: string, _repo?: string): Promise<Concept | null> {
    // Use sanitized ID for filename (replace / with __)
    const safeId = id.replace(/\//g, "__");
    const filePath = path.join(this.conceptsDir, `${safeId}.json`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(content);
      return normalizeConcept(parsed);
    } catch {
      return null;
    }
  }

  async getAllConcepts(_repo?: string): Promise<Concept[]> {
    try {
      const files = await fs.readdir(this.conceptsDir);
      const concepts: Concept[] = [];

      for (const file of files) {
        if (file.endsWith(".json")) {
          const content = await fs.readFile(
            path.join(this.conceptsDir, file),
            "utf-8"
          );
          const parsed = JSON.parse(content);
          concepts.push(normalizeConcept(parsed));
        }
      }

      // Filter by repo if provided (though FileSystemStorage doesn't fully support multi-repo)
      if (_repo) {
        return concepts.filter(f => f.repo === _repo);
      }

      return concepts;
    } catch {
      return [];
    }
  }

  async deleteConcept(id: string, _repo?: string): Promise<void> {
    const safeId = id.replace(/\//g, "__");
    const filePath = path.join(this.conceptsDir, `${safeId}.json`);
    await fs.unlink(filePath);
  }

  // PRs
  async savePR(pr: PRRecord): Promise<void> {
    const filePath = path.join(this.prsDir, `${pr.number}.md`);
    const content = await formatPRMarkdown(pr, this);
    await fs.writeFile(filePath, content);
  }

  async getPR(number: number, _repo?: string): Promise<PRRecord | null> {
    const filePath = path.join(this.prsDir, `${number}.md`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      return parsePRMarkdown(number, content);
    } catch {
      return null;
    }
  }

  async getAllPRs(_repo?: string): Promise<PRRecord[]> {
    try {
      const files = await fs.readdir(this.prsDir);
      const prs: PRRecord[] = [];

      for (const file of files) {
        if (file.endsWith(".md")) {
          const prNumber = parseInt(file.replace(".md", ""));
          const pr = await this.getPR(prNumber);
          if (pr) prs.push(pr);
        }
      }

      return prs.sort((a, b) => a.number - b.number);
    } catch {
      return [];
    }
  }

  // Commits
  async saveCommit(commit: CommitRecord): Promise<void> {
    const filePath = path.join(this.commitsDir, `${commit.sha}.md`);
    const content = await formatCommitMarkdown(commit, this);
    await fs.writeFile(filePath, content);
  }

  async getCommit(sha: string, _repo?: string): Promise<CommitRecord | null> {
    const filePath = path.join(this.commitsDir, `${sha}.md`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      return parseCommitMarkdown(sha, content);
    } catch {
      return null;
    }
  }

  async getAllCommits(_repo?: string): Promise<CommitRecord[]> {
    try {
      const files = await fs.readdir(this.commitsDir);
      const commits: CommitRecord[] = [];

      for (const file of files) {
        if (file.endsWith(".md")) {
          const sha = file.replace(".md", "");
          const commit = await this.getCommit(sha);
          if (commit) commits.push(commit);
        }
      }

      return commits.sort((a, b) => a.committedAt.getTime() - b.committedAt.getTime());
    } catch {
      return [];
    }
  }

  // Clues
  async saveClue(clue: Clue): Promise<void> {
    const filePath = path.join(this.cluesDir, `${clue.id}.json`);
    const serialized = {
      ...clue,
      createdAt: clue.createdAt.toISOString(),
      updatedAt: clue.updatedAt.toISOString(),
    };
    await fs.writeFile(filePath, JSON.stringify(serialized, null, 2));
  }

  async getClue(id: string, _repo?: string): Promise<Clue | null> {
    const filePath = path.join(this.cluesDir, `${id}.json`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(content);
      return {
        ...parsed,
        createdAt: new Date(parsed.createdAt),
        updatedAt: new Date(parsed.updatedAt),
      };
    } catch {
      return null;
    }
  }

  async getAllClues(_repo?: string): Promise<Clue[]> {
    try {
      const files = await fs.readdir(this.cluesDir);
      const clues: Clue[] = [];

      for (const file of files) {
        if (file.endsWith(".json")) {
          const id = file.replace(".json", "");
          const clue = await this.getClue(id);
          if (clue) clues.push(clue);
        }
      }

      return clues;
    } catch {
      return [];
    }
  }

  async deleteClue(id: string, _repo?: string): Promise<void> {
    const filePath = path.join(this.cluesDir, `${id}.json`);
    await fs.unlink(filePath);
  }

  /**
   * Simple clue search for FileSystemStore (less sophisticated than GraphStorage)
   */
  async searchClues(
    query: string,
    embeddings: number[],
    conceptId?: string,
    limit: number = 10,
    similarityThreshold: number = 0.5,
    _repo?: string
  ): Promise<Array<Clue & { score: number; relevanceBreakdown?: any }>> {
    const allClues = conceptId
      ? await this.getCluesForConcept(conceptId)
      : await this.getAllClues(_repo);

    const queryLower = query.toLowerCase();

    // Score each clue
    const scored = allClues
      .map((clue) => {
        // Vector similarity (cosine)
        const vectorScore = clue.embedding
          ? this.cosineSimilarity(embeddings, clue.embedding)
          : 0;

        // Keyword matching
        const keywordScore = clue.keywords.some((kw) =>
          kw.toLowerCase().includes(queryLower)
        )
          ? 0.3
          : clue.keywords.some((kw) =>
              queryLower.includes(kw.toLowerCase())
            )
          ? 0.2
          : 0;

        // Title matching
        const titleScore = clue.title.toLowerCase().includes(queryLower)
          ? 0.2
          : 0;

        // Centrality
        const centralityScore = clue.centrality || 0.5;

        // Combined score
        const finalScore =
          vectorScore * 0.5 + keywordScore + titleScore + centralityScore * 0.2;

        return {
          ...clue,
          score: finalScore,
          relevanceBreakdown: {
            vector: vectorScore,
            keyword: keywordScore,
            title: titleScore,
            centrality: centralityScore,
            final: finalScore,
          },
        };
      })
      .filter((result) => result.relevanceBreakdown.vector >= similarityThreshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return scored;
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  // Metadata - per-repo (FileSystemStorage uses a simple approach with repo-prefixed keys)
  private getRepoMetadataKey(repo: string, key: string): string {
    // For FileSystemStorage, we store per-repo metadata with prefixed keys
    const safeRepo = repo.replace(/\//g, "__");
    return `${safeRepo}__${key}`;
  }

  async getLastProcessedPR(repo: string): Promise<number> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "lastProcessedPR");
      return metadata[key] || metadata.lastProcessedPR || 0;
    } catch {
      return 0;
    }
  }

  async setLastProcessedPR(repo: string, number: number): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    const key = this.getRepoMetadataKey(repo, "lastProcessedPR");
    metadata[key] = number;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getLastProcessedCommit(repo: string): Promise<string | null> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "lastProcessedCommit");
      return metadata[key] || metadata.lastProcessedCommit || null;
    } catch {
      return null;
    }
  }

  async setLastProcessedCommit(repo: string, sha: string): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    const key = this.getRepoMetadataKey(repo, "lastProcessedCommit");
    metadata[key] = sha;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getChronologicalCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "chronologicalCheckpoint");
      return metadata[key] || metadata.chronologicalCheckpoint || null;
    } catch {
      return null;
    }
  }

  async setChronologicalCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    const key = this.getRepoMetadataKey(repo, "chronologicalCheckpoint");
    metadata[key] = checkpoint;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getClueAnalysisCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "clueAnalysisCheckpoint");
      return metadata[key] || metadata.clueAnalysisCheckpoint || null;
    } catch {
      return null;
    }
  }

  async setClueAnalysisCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    const key = this.getRepoMetadataKey(repo, "clueAnalysisCheckpoint");
    metadata[key] = checkpoint;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  // Themes - per-repo
  async addThemes(repo: string, themes: string[]): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }

    const key = this.getRepoMetadataKey(repo, "recentThemes");
    let recentThemes: string[] = metadata[key] || metadata.recentThemes || [];

    // Remove themes if they already exist (LRU behavior)
    recentThemes = recentThemes.filter((t: string) => !themes.includes(t));

    // Add to end (most recent)
    recentThemes.push(...themes);

    // Keep only last 100
    if (recentThemes.length > 100) {
      recentThemes = recentThemes.slice(-100);
    }

    metadata[key] = recentThemes;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getRecentThemes(repo: string): Promise<string[]> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "recentThemes");
      return metadata[key] || metadata.recentThemes || [];
    } catch {
      return [];
    }
  }

  // Total Usage - cumulative token usage across all processing runs
  async getTotalUsage(repo: string): Promise<Usage> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "totalUsage");
      const usage = metadata[key] || metadata.totalUsage;
      if (usage) {
        return normalizeUsage(usage);
      }
      return normalizeUsage();
    } catch {
      return normalizeUsage();
    }
  }

  async addToTotalUsage(repo: string, usage: Usage): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }

    const key = this.getRepoMetadataKey(repo, "totalUsage");
    const current = normalizeUsage(metadata[key]);
    metadata[key] = normalizeUsage(addUsage(current, usage));
    
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getAggregatedMetadata(): Promise<{
    lastProcessedTimestamp: string | null;
    cumulativeUsage: Usage;
  }> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      
      let latestTimestamp: string | null = null;
      let cumulativeUsage = normalizeUsage();

      for (const key of Object.keys(metadata)) {
        // Process checkpoint keys
        if (key.includes(":chronologicalCheckpoint")) {
          const checkpoint = metadata[key];
          if (checkpoint?.lastProcessedTimestamp) {
            if (!latestTimestamp || checkpoint.lastProcessedTimestamp > latestTimestamp) {
              latestTimestamp = checkpoint.lastProcessedTimestamp;
            }
          }
        }
        // Process usage keys
        if (key.includes(":totalUsage")) {
          const usage = metadata[key];
          cumulativeUsage = normalizeUsage(addUsage(cumulativeUsage, normalizeUsage(usage)));
        }
      }

      return {
        lastProcessedTimestamp: latestTimestamp,
        cumulativeUsage,
      };
    } catch {
      return {
        lastProcessedTimestamp: null,
        cumulativeUsage: normalizeUsage(),
      };
    }
  }

  // Documentation
  async saveDocumentation(
    conceptId: string,
    documentation: string
  ): Promise<void> {
    const filePath = path.join(this.docsDir, `${conceptId}.md`);
    await fs.writeFile(filePath, documentation);
  }

  // Concept-File Linking (not supported in FileSystemStorage)
  async linkConceptsToFiles(_conceptId?: string, _repo?: string): Promise<LinkResult> {
    throw new Error(
      "Concept-File linking is only supported with GraphStorage. Use --graph flag."
    );
  }

  async linkConceptToFilesByPaths(_conceptId: string, _filePaths: string[]): Promise<number> {
    throw new Error(
      "Concept-File linking is only supported with GraphStorage. Use --graph flag."
    );
  }

  async linkPRsToFiles(_repo?: string): Promise<{ prsProcessed: number; edgesLinked: number }> {
    throw new Error(
      "PR-File linking is only supported with GraphStorage. Use --graph flag."
    );
  }

  // Get Files for Concept (not supported in FileSystemStorage)
  async getFilesForConcept(_conceptId: string, _expand?: string[]): Promise<any[]> {
    throw new Error(
      "Getting files for concepts is only supported with GraphStorage. Use --graph flag."
    );
  }
}
