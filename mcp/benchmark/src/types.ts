export interface StepMeta {
  step: number;
  turn: number;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  cumulativeInput: number;
  cumulativeOutput: number;
  toolCalls: string[];
  timestamp: string;
}

export interface ProductionRun {
  id: string;
  source: string;
  repo: string;
  model: string;
  timestamp: string;
  duration_ms: number;
  token_usage: { input: number; output: number; total: number };
  tool_sequence: string[];
  tool_call_count: number;
  user_prompt_preview: string;
  answer_preview: string;
  step_meta?: StepMeta[];
  trace?: unknown;
}
