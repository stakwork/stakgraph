/**
 * Types for the GEPA eval harness for the services agent.
 *
 * A TrainingExample pairs a GitHub repo with its known-good
 * pm2.config.js and docker-compose.yml outputs.
 */

/** A single training/validation example */
export interface TrainingExample {
  /** e.g. "stakwork" */
  owner: string;
  /** e.g. "hive" */
  repo: string;
  /** Gold-standard pm2.config.js content */
  gold_pm2: string;
  /** Gold-standard docker-compose.yml content */
  gold_docker_compose: string;
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
export interface ScoreBreakdown {
  /** 0-1: Did it produce both files? */
  format_score: number;
  /** 0-1: Are pm2 service names/structure correct? */
  pm2_structure_score: number;
  /** 0-1: Are docker-compose services correct? */
  docker_structure_score: number;
  /** 0-1: Are env vars / ports / commands correct? */
  env_vars_score: number;
  /** 0-1: Does the app service match exactly? */
  app_service_score: number;
  /** 0-1: Are cwds correct? */
  cwd_score: number;
  /** 0-1: Has a "frontend" named service? */
  frontend_naming_score: number;
  /** Weighted aggregate 0-1 */
  total: number;
}

/** Result from a single evaluation run */
export interface EvalResult {
  example: TrainingExample;
  generated_pm2: string;
  generated_docker_compose: string;
  score: ScoreBreakdown;
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
