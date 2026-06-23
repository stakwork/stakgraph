export interface Doc {
  repoName: string;
  documentation: string;
}

export interface ConceptSummary {
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

export interface ConceptsResponse {
  concepts: ConceptSummary[];
  total: number;
  repo: string;
  lastProcessedTimestamp: string | null;
  processing: boolean;
}
