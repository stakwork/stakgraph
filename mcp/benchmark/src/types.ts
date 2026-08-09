export type AnnotationMarker =
  | "inefficient"
  | "bad_search"
  | "good_result"
  | "loop"
  | "wrong_tool"
  | "wasted_tokens";

export interface Annotation {
  ts: string;
  author?: string;
  target: "session" | "tool_call";
  target_id?: string;
  marker: AnnotationMarker;
  note?: string;
}

export interface TokenUsage {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  total: number;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

export interface StepMeta {
  step: number;
  turn: number;
  label?: string;
  usage: TokenUsage;
  cumulativeInput: number;
  cumulativeOutput: number;
  toolCalls: string[];
  timestamp: string;
}

export interface SearchResultMeta {
  ref_id: string;
  sources: ("fulltext" | "vector")[];
  rrf_score?: number;
  fulltext_rank?: number;
  fulltext_score?: number;
  vector_rank?: number;
  vector_score?: number;
}

export interface SearchProvenanceEntry {
  tool_call_id?: string;
  tool_name: string;
  timestamp: string;
  provenance: {
    method: string;
    query: string;
    result_meta: SearchResultMeta[];
  };
}

/**
 * One gitree Concept the session read. `read_order` is always present (it is
 * recorded with no model involved); everything from `rank` down only appears
 * when the run was started with `reflect`.
 */
export interface ReflectedConcept {
  id?: string;
  ref_id?: string;
  repo?: string;
  name?: string;
  read_order?: number;
  rank: number | null;
  evidence?: string;
  contradicts?: string;
}

export interface SessionReflection {
  session_id: string;
  updated_at: string;
  concepts: ReflectedConcept[];
  gap?: string | null;
  raw?: string;
}

/** Full concept as `GET /gitree/concepts/:id` serves it. */
export interface ConceptDetail {
  concept: {
    id: string;
    name?: string;
    description?: string;
    documentation?: string;
  };
}

export interface ProductionRun {
  id: string;
  source: string;
  repo: string;
  model: string;
  provider?: string;
  cost_usd?: number;
  timestamp: string;
  duration_ms: number;
  token_usage: TokenUsage;
  status?: string;
  error_message?: string;
  tool_sequence: string[];
  tool_call_count: number;
  user_prompt_preview: string;
  answer_preview: string;
  step_meta?: StepMeta[];
  search_provenance?: SearchProvenanceEntry[];
  annotations?: Annotation[];
  reflection?: SessionReflection | null;
  trace?: unknown;
}
