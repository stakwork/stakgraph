import fs from "fs/promises";
import path from "path";
import { Feature, PRRecord, CommitRecord, Clue, LinkResult, ChronologicalCheckpoint } from "../types.js";
import { Storage } from "./storage.js";
import { formatPRMarkdown, parsePRMarkdown, formatCommitMarkdown, parseCommitMarkdown } from "./utils.js";

/**
 * File system based storage implementation
 *
 * Directory structure:
 * ./knowledge-base/
 *   ├── metadata.json
 *   ├── features/
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
  private featuresDir: string;
  private prsDir: string;
  private commitsDir: string;
  private cluesDir: string;
  private docsDir: string;
  private metadataPath: string;

  constructor(baseDir: string = "./knowledge-base") {
    super();
    this.featuresDir = path.join(baseDir, "features");
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
    await fs.mkdir(this.featuresDir, { recursive: true });
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

  // Features
  async saveFeature(feature: Feature): Promise<void> {
    // Use sanitized ID for filename (replace / with __)
    const safeId = feature.id.replace(/\//g, "__");
    const filePath = path.join(this.featuresDir, `${safeId}.json`);
    const serialized = {
      ...feature,
      createdAt: feature.createdAt.toISOString(),
      lastUpdated: feature.lastUpdated.toISOString(),
      cluesLastAnalyzedAt: feature.cluesLastAnalyzedAt?.toISOString(),
    };
    await fs.writeFile(filePath, JSON.stringify(serialized, null, 2));
  }

  async getFeature(id: string, _repo?: string): Promise<Feature | null> {
    // Use sanitized ID for filename (replace / with __)
    const safeId = id.replace(/\//g, "__");
    const filePath = path.join(this.featuresDir, `${safeId}.json`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(content);
      return {
        ...parsed,
        createdAt: new Date(parsed.createdAt),
        lastUpdated: new Date(parsed.lastUpdated),
        cluesLastAnalyzedAt: parsed.cluesLastAnalyzedAt
          ? new Date(parsed.cluesLastAnalyzedAt)
          : undefined,
      };
    } catch {
      return null;
    }
  }

  async getAllFeatures(_repo?: string): Promise<Feature[]> {
    try {
      const files = await fs.readdir(this.featuresDir);
      const features: Feature[] = [];

      for (const file of files) {
        if (file.endsWith(".json")) {
          const content = await fs.readFile(
            path.join(this.featuresDir, file),
            "utf-8"
          );
          const parsed = JSON.parse(content);
          features.push({
            ...parsed,
            createdAt: new Date(parsed.createdAt),
            lastUpdated: new Date(parsed.lastUpdated),
            cluesLastAnalyzedAt: parsed.cluesLastAnalyzedAt
              ? new Date(parsed.cluesLastAnalyzedAt)
              : undefined,
          });
        }
      }

      // Filter by repo if provided (though FileSystemStorage doesn't fully support multi-repo)
      if (_repo) {
        return features.filter(f => f.repo === _repo);
      }

      return features;
    } catch {
      return [];
    }
  }

  async deleteFeature(id: string, _repo?: string): Promise<void> {
    const safeId = id.replace(/\//g, "__");
    const filePath = path.join(this.featuresDir, `${safeId}.json`);
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
    featureId?: string,
    limit: number = 10,
    similarityThreshold: number = 0.5,
    _repo?: string
  ): Promise<Array<Clue & { score: number; relevanceBreakdown?: any }>> {
    const allClues = featureId
      ? await this.getCluesForFeature(featureId)
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
  async getTotalUsage(repo: string): Promise<{ inputTokens: number; outputTokens: number; totalTokens: number }> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      const key = this.getRepoMetadataKey(repo, "totalUsage");
      const usage = metadata[key] || metadata.totalUsage;
      if (usage) {
        return {
          inputTokens: usage.inputTokens || 0,
          outputTokens: usage.outputTokens || 0,
          totalTokens: usage.totalTokens || 0,
        };
      }
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    } catch {
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }
  }

  async addToTotalUsage(repo: string, usage: { inputTokens: number; outputTokens: number; totalTokens: number }): Promise<void> {
    let metadata: any = {};
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }

    const key = this.getRepoMetadataKey(repo, "totalUsage");
    const current = metadata[key] || { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    
    metadata[key] = {
      inputTokens: (current.inputTokens || 0) + usage.inputTokens,
      outputTokens: (current.outputTokens || 0) + usage.outputTokens,
      totalTokens: (current.totalTokens || 0) + usage.totalTokens,
    };
    
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  // Documentation
  async saveDocumentation(
    featureId: string,
    documentation: string
  ): Promise<void> {
    const filePath = path.join(this.docsDir, `${featureId}.md`);
    await fs.writeFile(filePath, documentation);
  }

  // Feature-File Linking (not supported in FileSystemStorage)
  async linkFeaturesToFiles(_featureId?: string, _repo?: string): Promise<LinkResult> {
    throw new Error(
      "Feature-File linking is only supported with GraphStorage. Use --graph flag."
    );
  }

  // Get Files for Feature (not supported in FileSystemStorage)
  async getFilesForFeature(_featureId: string, _expand?: string[]): Promise<any[]> {
    throw new Error(
      "Getting files for features is only supported with GraphStorage. Use --graph flag."
    );
  }
}
