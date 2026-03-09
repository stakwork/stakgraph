/**
 * Types for the GEPA eval harness.
 *
 * A TrainingExample pairs a GitHub repo with its known-good output files.
 */

/** A single training/validation example */
export interface TrainingExample {
  /** e.g. "stakwork" */
  owner: string;
  /** e.g. "hive" */
  repo: string;
  /** Map of filename -> expected content */
  expected_files: Record<string, string>;
  /** Optional notes about what matters most in this example */
  notes?: string;
}

/** The two text components GEPA will evolve */
export interface CandidatePrompts {
  /** The EXPLORER system prompt */
  explorer: string;
  /** The FINAL_ANSWER tool description */
  final_answer: string;
}

/** Result of scoring a single generated output against a gold standard */
export interface ScoreResult {
  /** 0 = wrong, 0.5 = partial, 1 = pass */
  total: number;
  /** What went right/wrong (shown to Opus for reflection) */
  reason: string;
  /** One-sentence suggestion for how to improve the prompt */
  insight?: string;
}

/** Result from a single evaluation run */
export interface EvalResult {
  example: TrainingExample;
  /** Map of filename -> generated content */
  generated_files: Record<string, string>;
  score: ScoreResult;
  /** Raw LLM output before parsing */
  raw_output: string;
  /** Time taken in ms */
  duration_ms: number;
}

/** Overall result of an optimization run */
export interface OptimizationResult {
  /** Best prompts found */
  best_candidate: CandidatePrompts;
  /** Best aggregate score across validation set */
  best_score: number;
  /** All candidates tried with their scores */
  history: Array<{
    candidate: CandidatePrompts;
    scores: number[];
    aggregate: number;
    generation: number;
  }>;
  /** Total LLM calls made */
  total_eval_calls: number;
  /** Total wall-clock time in ms */
  total_duration_ms: number;
}
