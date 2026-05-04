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

export type SessionContextRefKind =
  | "file"
  | "function"
  | "endpoint"
  | "env"
  | "command"
  | "url"
  | "ref_id"
  | "other";

export interface SessionContextRef {
  kind: SessionContextRefKind;
  value: string;
  reason: string;
}

export interface SessionContextState {
  summary: string;
  goals: string[];
  decisions: string[];
  importantRefs: SessionContextRef[];
  checked: string[];
  openQuestions: string[];
  nextSteps: string[];
  warnings: string[];
  updated_at: string;
}

export interface ContextTimelineDiff {
  added: Record<string, unknown[]>;
  removed: Record<string, unknown[]>;
}

export interface ContextTimelineEntry {
  turn: number;
  timestamp: string;
  before: SessionContextState;
  after: SessionContextState;
  usage?: {
    input: number;
    cache_read: number;
    cache_write: number;
    output: number;
    total: number;
  };
  diff?: ContextTimelineDiff;
  changedSummary?: boolean;
  newMessagesPreview?: string;
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
  context_usage?: TokenUsage;
  all_in_usage?: TokenUsage;
  status?: string;
  error_message?: string;
  context_cost_usd?: number;
  all_in_cost_usd?: number;
  tool_sequence: string[];
  tool_call_count: number;
  user_prompt_preview: string;
  answer_preview: string;
  step_meta?: StepMeta[];
  search_provenance?: SearchProvenanceEntry[];
  context_state?: SessionContextState;
  context_timeline?: ContextTimelineEntry[];
  annotations?: Annotation[];
  trace?: unknown;
}
