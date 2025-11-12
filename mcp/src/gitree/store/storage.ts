import { Feature, PRRecord, LinkResult } from "../types.js";

/**
 * Abstract storage interface for features and PRs
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

  // Metadata
  abstract getLastProcessedPR(): Promise<number>;
  abstract setLastProcessedPR(number: number): Promise<void>;

  // Themes (sliding window of recent technical tags)
  abstract addThemes(themes: string[]): Promise<void>;
  abstract getRecentThemes(): Promise<string[]>;

  // Documentation
  abstract saveDocumentation(featureId: string, documentation: string): Promise<void>;

  // Feature-File Linking
  abstract linkFeaturesToFiles(featureId?: string): Promise<LinkResult>;

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
