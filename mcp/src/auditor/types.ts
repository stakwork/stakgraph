export type Outcome = "works" | "broken" | "unknown";

export interface DeckTask {
  prompt: string;
  description: string | null;
}

export interface DeckMap {
  appUrl: string;
  notes: string | null;
}

export interface Deck {
  task: DeckTask;
  diff: string;
  featureContext: string | null;
  map: DeckMap;
}

export interface JobModel {
  apiKey: string;
  host?: string;
  provider?: string;
  model?: string;
}

export interface AuditJob {
  taskId: string;
  deck: Deck;
  model: JobModel;
  responseUrl: string;
  callbackApiKey: string;
}

export type EvidenceKind =
  | "screenshot"
  | "http"
  | "log"
  | "timing"
  | "dom"
  | "note";

export interface EvidenceRecord {
  id: string;
  kind: EvidenceKind;
  summary: string;
  data: string;
}

export interface ClaimVerdict {
  claim: string;
  verdict: Outcome;
  proof: string[];
  reasoning: string;
}

export interface Verdict {
  taskId: string;
  overall: Outcome;
  claims: ClaimVerdict[];
  observations: string[];
  summary: string;
  evidence: EvidenceRecord[];
  startedAt: string;
  finishedAt: string;
  error?: string;
}

export interface EvidenceCollector {
  records: EvidenceRecord[];
  verdict?: {
    overall: Outcome;
    claims: ClaimVerdict[];
    observations: string[];
    summary: string;
  };
  push(kind: EvidenceKind, summary: string, data?: string): string;
}

export interface AuditorContext {
  deck: Deck;
  collector: EvidenceCollector;
  browser: import("./browser.js").AuditBrowser;
}
