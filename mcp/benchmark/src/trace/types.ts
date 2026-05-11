export type CallEntry = {
  id: string;
  toolName: string;
  input: unknown;
  index: number;
};

export type ResultEntry = {
  id: string;
  toolName: string;
  output: unknown;
  index: number;
};

export type TraceEventKind =
  | "system"
  | "user"
  | "assistant-text"
  | "reasoning"
  | "assistant"
  | "tool-call"
  | "tool-result"
  | "tool";

export type TraceTurnKind = "setup" | "direct" | "tool";

export type TraceEvent = {
  id: string;
  index: number;
  role: string;
  kind: TraceEventKind;
  text: string;
  payload: unknown;
  toolName?: string;
  toolCallId?: string;
};

export type TraceTurn = {
  id: string;
  index: number;
  kind: TraceTurnKind;
  title: string;
  preview: string;
  outputPreview: string;
  toolNames: string[];
  eventCount: number;
  toolCount: number;
  events: TraceEvent[];
};

export type IssueKind =
  | "empty"
  | "oversized"
  | "repeat"
  | "fallback"
  | "duplicate"
  | "noisy-overview";

export type AnalyzedStep = {
  id: string;
  index: number;
  toolName: string;
  input: unknown;
  output: unknown;
  outputSize: number;
  outputLines: number;
  flags: IssueKind[];
};

export type StandaloneUnit = { kind: "standalone"; event: TraceEvent };
export type PairedUnit = {
  kind: "paired";
  call: TraceEvent;
  result: TraceEvent | undefined;
};
export type DisplayUnit = StandaloneUnit | PairedUnit;

export type ProvenanceMatchState = {
  cursor: number;
  consumed: Set<number>;
};

export type ParsedTrace = {
  userPrompt: string;
  answer: string;
  calls: CallEntry[];
  results: ResultEntry[];
  events: TraceEvent[];
  turns: TraceTurn[];
};

export type TraceAnalysis = {
  steps: AnalyzedStep[];
  counts: Record<IssueKind, number>;
};
