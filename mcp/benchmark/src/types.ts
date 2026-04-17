export interface ProductionRun {
  id: string;
  source: "session";
  repo: string;
  model: string;
  timestamp: string;
  duration_ms: number;
  token_usage: { input: number; output: number; total: number };
  tool_sequence: string[];
  tool_call_count: number;
  user_prompt_preview: string;
  answer_preview: string;
  trace?: unknown;
}
