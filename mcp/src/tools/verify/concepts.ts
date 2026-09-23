export interface VerificationConcept {
  id: string;
  name: string;
  match: string[];
  procedure: string;
}

export const VERIFICATION_CONCEPTS: VerificationConcept[] = [
  {
    id: "backend-endpoint",
    name: "Backend endpoint / response contract",
    match: ["endpoint", "route.ts", "/api/", "get ", "post ", "http", "status", "response", "json", "nextresponse", "params", "query"],
    procedure:
      "This change is about an API endpoint. Prefer verify_http_request over the browser: call the endpoint for the documented params and a couple of edge cases, assert the status code and the response shape, and exercise each documented filter/param (namespace, query, type, limit, flags). Confirm the contract the diff claims — including that optional flags actually change the response and defaults behave. Cite the http evidence id for each claim.",
  },
  {
    id: "frontend-interaction",
    name: "Frontend interaction / rendered behaviour",
    match: ["button", "page.tsx", "click", "onclick", "form", "submit", "renders", "component", "usestate", "useeffect", "ui", "input", "modal"],
    procedure:
      "This change is about a user-facing interaction. Open the page and drive the interaction with the browser. Use stagehand_network_activity to confirm the underlying request actually fired and returned the expected status (a success message in the UI is not proof). Use stagehand_logs to catch runtime/console errors a screenshot hides. Take a screenshot of the outcome. Cite network / console / screenshot evidence ids.",
  },
  {
    id: "data-persistence",
    name: "Data persistence / read-after-write",
    match: ["persist", "save", "database", "insert", "update", "create", "store", "prisma", "$executeraw", "db.", "row", "record"],
    procedure:
      "This change writes data. Perform the write through the app, then INDEPENDENTLY confirm it persisted with verify_db_query (SELECT the row) or an independent re-read via a different path. A 200 or a success toast is not proof of persistence. Cite the db evidence id showing the row is (or is not) there.",
  },
];

const DEFAULT_CONCEPT: VerificationConcept = {
  id: "general",
  name: "General verification",
  match: [],
  procedure:
    "Choose the cheapest sufficient probe for each claim: an API claim → verify_http_request; a UI claim → browser + stagehand_network_activity + stagehand_logs; a persistence claim → verify_db_query. Capture and cite evidence for every claim you mark works.",
};

export function selectConcept(taskPrompt: string, diff: string): VerificationConcept {
  const hay = `${taskPrompt}\n${diff}`.toLowerCase();
  let best: VerificationConcept | null = null;
  let bestScore = 0;
  for (const c of VERIFICATION_CONCEPTS) {
    let score = 0;
    for (const kw of c.match) if (hay.includes(kw)) score++;
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return bestScore > 0 && best ? best : DEFAULT_CONCEPT;
}

export function conceptHint(concept: VerificationConcept): string {
  return `SUGGESTED CHECKS — from the "${concept.name}" verification concept (hints, not a rigid script; adapt to what the diff actually claims):\n${concept.procedure}`;
}
