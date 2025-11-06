import fs from "fs/promises";
import path from "path";
import { Feature, PRRecord } from "./types.js";

/**
 * Abstract storage interface for features and PRs
 */
export abstract class Storage {
  // Features
  abstract saveFeature(feature: Feature): Promise<void>;
  abstract getFeature(id: string): Promise<Feature | null>;
  abstract getAllFeatures(): Promise<Feature[]>;
  abstract deleteFeature(id: string): Promise<void>;

  // PRs
  abstract savePR(pr: PRRecord): Promise<void>;
  abstract getPR(number: number): Promise<PRRecord | null>;
  abstract getAllPRs(): Promise<PRRecord[]>;

  // Metadata
  abstract getLastProcessedPR(): Promise<number>;
  abstract setLastProcessedPR(number: number): Promise<void>;

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
}

/**
 * File system based storage implementation
 *
 * Directory structure:
 * ./knowledge-base/
 *   ├── metadata.json
 *   ├── features/
 *   │   ├── auth-system.json
 *   │   └── ...
 *   └── prs/
 *       ├── 1.md
 *       └── ...
 */
export class FileSystemStore extends Storage {
  private baseDir: string;
  private featuresDir: string;
  private prsDir: string;
  private metadataPath: string;

  constructor(baseDir: string = "./knowledge-base") {
    super();
    this.baseDir = baseDir;
    this.featuresDir = path.join(baseDir, "features");
    this.prsDir = path.join(baseDir, "prs");
    this.metadataPath = path.join(baseDir, "metadata.json");
  }

  /**
   * Initialize directory structure
   */
  async initialize(): Promise<void> {
    await fs.mkdir(this.featuresDir, { recursive: true });
    await fs.mkdir(this.prsDir, { recursive: true });

    try {
      await fs.access(this.metadataPath);
    } catch {
      await fs.writeFile(
        this.metadataPath,
        JSON.stringify({ lastProcessedPR: 0 }, null, 2)
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
    const content = await this.formatPRMarkdown(pr);
    await fs.writeFile(filePath, content);
  }

  async getPR(number: number): Promise<PRRecord | null> {
    const filePath = path.join(this.prsDir, `${number}.md`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      // TODO: Implement parsing PR markdown back to PRRecord
      return this.parsePRMarkdown(number, content);
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
    let metadata = { lastProcessedPR: 0 };
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    metadata.lastProcessedPR = number;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  // Helpers
  private async formatPRMarkdown(pr: PRRecord): Promise<string> {
    // Get features this PR belongs to
    const features = await this.getFeaturesForPR(pr.number);
    const featureLinks =
      features.length > 0
        ? `\n---\n\n_Part of features: ${features
            .map((f) => `\`${f.id}\``)
            .join(", ")}_`
        : "";

    const filesList = pr.files.length > 0
      ? `\n\n## Files Changed (${pr.files.length})\n\n${this.formatFilesList(pr.files)}`
      : '';

    return `# PR #${pr.number}: ${pr.title}

**Merged**: ${pr.mergedAt.toISOString().split("T")[0]}
**URL**: ${pr.url}

## Summary

${pr.summary}${filesList}${featureLinks}
`.trim();
  }

  /**
   * Format files list with intelligent collapsing
   */
  private formatFilesList(files: string[]): string {
    // Only collapse if > 20 files total
    if (files.length <= 20) {
      return files.map(f => `- ${f}`).join('\n');
    }

    // Directories to always collapse
    const autoCollapseDirs = new Set([
      'node_modules',
      'dist',
      'build',
      'target',
      'out',
      '.next',
      'coverage',
    ]);

    // Group files by directory
    const byDirectory: Map<string, string[]> = new Map();
    const rootFiles: string[] = [];

    for (const file of files) {
      const parts = file.split('/');
      if (parts.length === 1) {
        rootFiles.push(file);
      } else {
        const dir = parts[0];
        if (!byDirectory.has(dir)) {
          byDirectory.set(dir, []);
        }
        byDirectory.get(dir)!.push(file);
      }
    }

    const output: string[] = [];

    // Show root files (important ones first)
    for (const file of rootFiles) {
      output.push(`- ${file}`);
    }

    // Process directories
    for (const [dir, dirFiles] of byDirectory.entries()) {
      // Always collapse certain directories
      if (autoCollapseDirs.has(dir)) {
        output.push(`- ${dir}/... (${dirFiles.length} files)`);
        continue;
      }

      // Show first 10, then indicate more
      if (dirFiles.length > 10) {
        // Show first 10 files
        for (let i = 0; i < 10; i++) {
          output.push(`- ${dirFiles[i]}`);
        }
        // Indicate there are more
        const remaining = dirFiles.length - 10;
        output.push(`- ${dir}/... (${remaining} more files)`);
      } else {
        // Show all files
        for (const file of dirFiles) {
          output.push(`- ${file}`);
        }
      }
    }

    return output.join('\n');
  }

  // TODO: Implement parsing markdown back to PRRecord
  private parsePRMarkdown(number: number, content: string): PRRecord {
    // Simple regex-based parsing for now
    const titleMatch = content.match(/^# PR #\d+: (.+)$/m);
    const mergedMatch = content.match(/\*\*Merged\*\*: (.+)$/m);
    const urlMatch = content.match(/\*\*URL\*\*: (.+)$/m);
    const summaryMatch = content.match(
      /## Summary\n\n([\s\S]+?)(?:\n## Files Changed|\n---|\n*$)/
    );

    // Parse files list
    const filesMatch = content.match(/## Files Changed \(\d+\)\n\n([\s\S]+?)(?:\n---|\n*$)/);
    const files: string[] = [];
    if (filesMatch?.[1]) {
      const fileLines = filesMatch[1].split('\n');
      for (const line of fileLines) {
        const fileMatch = line.match(/^- (.+)$/);
        if (fileMatch) {
          files.push(fileMatch[1]);
        }
      }
    }

    return {
      number,
      title: titleMatch?.[1] || "Unknown",
      mergedAt: mergedMatch?.[1] ? new Date(mergedMatch[1]) : new Date(),
      url: urlMatch?.[1]?.trim() || "",
      summary: summaryMatch?.[1]?.trim() || "",
      files,
    };
  }
}
