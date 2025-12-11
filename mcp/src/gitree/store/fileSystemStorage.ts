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
  private baseDir: string;
  private featuresDir: string;
  private prsDir: string;
  private commitsDir: string;
  private cluesDir: string;
  private docsDir: string;
  private metadataPath: string;

  constructor(baseDir: string = "./knowledge-base") {
    super();
    this.baseDir = baseDir;
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
    const filePath = path.join(this.featuresDir, `${feature.id}.json`);
    const serialized = {
      ...feature,
      createdAt: feature.createdAt.toISOString(),
      lastUpdated: feature.lastUpdated.toISOString(),
      cluesLastAnalyzedAt: feature.cluesLastAnalyzedAt?.toISOString(),
    };
    await fs.writeFile(filePath, JSON.stringify(serialized, null, 2));
  }

  async getFeature(id: string): Promise<Feature | null> {
    const filePath = path.join(this.featuresDir, `${id}.json`);
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

  async getAllFeatures(): Promise<Feature[]> {
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

      return features;
    } catch {
      return [];
    }
  }

  async deleteFeature(id: string): Promise<void> {
    const filePath = path.join(this.featuresDir, `${id}.json`);
    await fs.unlink(filePath);
  }

  // PRs
  async savePR(pr: PRRecord): Promise<void> {
    const filePath = path.join(this.prsDir, `${pr.number}.md`);
    const content = await formatPRMarkdown(pr, this);
    await fs.writeFile(filePath, content);
  }

  async getPR(number: number): Promise<PRRecord | null> {
    const filePath = path.join(this.prsDir, `${number}.md`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      return parsePRMarkdown(number, content);
    } catch {
      return null;
    }
  }

  async getAllPRs(): Promise<PRRecord[]> {
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

  async getCommit(sha: string): Promise<CommitRecord | null> {
    const filePath = path.join(this.commitsDir, `${sha}.md`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      return parseCommitMarkdown(sha, content);
    } catch {
      return null;
    }
  }

  async getAllCommits(): Promise<CommitRecord[]> {
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

  async getClue(id: string): Promise<Clue | null> {
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

  async getAllClues(): Promise<Clue[]> {
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

  async deleteClue(id: string): Promise<void> {
    const filePath = path.join(this.cluesDir, `${id}.json`);
    await fs.unlink(filePath);
  }

  // Metadata
  async getLastProcessedPR(): Promise<number> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      return metadata.lastProcessedPR || 0;
    } catch {
      return 0;
    }
  }

  async setLastProcessedPR(number: number): Promise<void> {
    let metadata = { lastProcessedPR: 0, lastProcessedCommit: null as string | null, recentThemes: [] };
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    metadata.lastProcessedPR = number;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getLastProcessedCommit(): Promise<string | null> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      return metadata.lastProcessedCommit || null;
    } catch {
      return null;
    }
  }

  async setLastProcessedCommit(sha: string): Promise<void> {
    let metadata = { lastProcessedPR: 0, lastProcessedCommit: null as string | null, recentThemes: [] };
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    metadata.lastProcessedCommit = sha;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getChronologicalCheckpoint(): Promise<ChronologicalCheckpoint | null> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      return metadata.chronologicalCheckpoint || null;
    } catch {
      return null;
    }
  }

  async setChronologicalCheckpoint(checkpoint: ChronologicalCheckpoint): Promise<void> {
    let metadata: any = { lastProcessedPR: 0, lastProcessedCommit: null, recentThemes: [] };
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    metadata.chronologicalCheckpoint = checkpoint;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  // Themes
  async addThemes(themes: string[]): Promise<void> {
    let metadata = { lastProcessedPR: 0, lastProcessedCommit: null as string | null, recentThemes: [] as string[] };
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
      if (!metadata.recentThemes) {
        metadata.recentThemes = [];
      }
    } catch {
      // File doesn't exist yet
    }

    // Remove themes if they already exist (LRU behavior)
    metadata.recentThemes = metadata.recentThemes.filter((t: string) => !themes.includes(t));

    // Add to end (most recent)
    metadata.recentThemes.push(...themes);

    // Keep only last 100
    if (metadata.recentThemes.length > 100) {
      metadata.recentThemes = metadata.recentThemes.slice(-100);
    }

    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  async getRecentThemes(): Promise<string[]> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      return metadata.recentThemes || [];
    } catch {
      return [];
    }
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
  async linkFeaturesToFiles(_featureId?: string): Promise<LinkResult> {
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
