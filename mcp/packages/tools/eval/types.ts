import { z } from "zod";

// Base step type
interface BaseStep {
  type:
    | "navigate"
    | "act"
    | "extract"
    | "observe"
    | "screenshot"
    | "search"
    | "wait";
}

// Individual step types
export interface NavigateStep extends BaseStep {
  type: "navigate";
  url: string;
}

export interface ActStep extends BaseStep {
  type: "act";
  instruction: string;
}

export interface ExtractStep extends BaseStep {
  type: "extract";
  instruction: string;
  schema: z.ZodObject;
  useTextExtract?: boolean;
}

export interface ObserveStep extends BaseStep {
  type: "observe";
  instruction?: string;
}

export interface ScreenshotStep extends BaseStep {
  type: "screenshot";
  path: string;
  fullPage?: boolean;
}

export interface SearchStep extends BaseStep {
  type: "search";
  query: string;
}

export interface WaitStep extends BaseStep {
  type: "wait";
  duration: number;
}

// Combined step type
export type Step =
  | NavigateStep
  | ActStep
  | ExtractStep
  | ObserveStep
  | ScreenshotStep
  | SearchStep
  | WaitStep;

// Task configuration
export interface TaskConfig {
  url: string;
  steps: Step[];
  logCategory?: string;
}

// Test result types
export interface TestResult {
  status: "PASS" | "FAIL";
  description: string;
  failedCriteria?: string[];
}

export interface TestSuiteResult {
  overallStatus: "PASS" | "FAIL";
  results: TestResult[];
  summary: string;
}
