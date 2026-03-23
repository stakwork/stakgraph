export enum ImportanceTag {
  EntryPoint = "entry_point",
  Utility = "utility",
  Hub = "hub",
  Connector = "connector",
  Isolated = "isolated",
}

export const IMPORTANCE_TAGS = Object.values(ImportanceTag);

export interface ImportanceThresholds {
  entry_p90: number;
  utility_p75: number;
  hub_p90: number;
}

export interface ScoredNode {
  ref_id: string;
  node_type: string;
  pagerank: number;
  in_degree: number;
  out_degree: number;
  entry_score: number;
  utility_score: number;
  hub_score: number;
}

export interface TaggedNode extends ScoredNode {
  importance_tag: ImportanceTag;
}

export interface ImportanceTopNode {
  ref_id: string;
  name: string;
  file: string;
  label: string;
  pagerank: number;
  in_degree: number;
  out_degree: number;
  entry_score: number;
  utility_score: number;
  hub_score: number;
  importance_tag: ImportanceTag | null;
}

export interface ImportanceResult {
  nodesScored: number;
  topNodes: ImportanceTopNode[];
}
