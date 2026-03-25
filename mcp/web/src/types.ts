export interface Doc {
  repoName: string;
  documentation: string;
}

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

export interface FeaturesResponse {
  features: FeatureSummary[];
  total: number;
  repo: string;
  lastProcessedTimestamp: string | null;
  processing: boolean;
}
