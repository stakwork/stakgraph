import { GraphStorage } from "./store/index.js";

export interface FeatureSummary {
  id: string;
  repo?: string;
  ref_id?: string;
  name: string;
  description: string;
  prCount: number;
  commitCount: number;
  lastUpdated: string;
  hasDocumentation: boolean;
}

export interface FeatureDocumentation {
  id: string;
  name: string;
  description: string;
  documentation?: string;
}

export async function listFeatures(repo?: string): Promise<{
  features: FeatureSummary[];
  total: number;
}> {
  const storage = new GraphStorage();
  await storage.initialize();

  const features = await storage.getAllFeatures(repo);

  return {
    features: features.map((f) => ({
      id: f.id,
      repo: f.repo,
      ref_id: f.ref_id,
      name: f.name,
      description: f.description,
      prCount: f.prNumbers.length,
      commitCount: (f.commitShas || []).length,
      lastUpdated: f.lastUpdated.toISOString(),
      hasDocumentation: !!f.documentation,
    })),
    total: features.length,
  };
}

export async function getFeatureDocumentation(
  featureId: string,
  repo?: string
): Promise<FeatureDocumentation | null> {
  const storage = new GraphStorage();
  await storage.initialize();

  const feature = await storage.getFeature(featureId, repo);

  if (!feature) {
    return null;
  }

  return {
    id: feature.id,
    name: feature.name,
    description: feature.description,
    documentation: feature.documentation,
  };
}
