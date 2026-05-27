import type { DisplayUnit, TraceEvent, ProvenanceMatchState } from "./types";
import type { SearchProvenanceEntry } from "../types";

export const SEARCH_TOOL_NAMES = new Set([
  "stakgraph_search",
  "fulltext_search",
  "graph_search",
]);

export function groupEvents(events: TraceEvent[]): DisplayUnit[] {
  const resultsByCallId = new Map<string, TraceEvent>();
  for (const event of events) {
    if (event.kind === "tool-result" && event.toolCallId) {
      resultsByCallId.set(event.toolCallId, event);
    }
  }

  const units: DisplayUnit[] = [];
  const consumed = new Set<string>();

  for (const event of events) {
    if (consumed.has(event.id)) continue;

    if (event.kind === "tool-call") {
      const result = event.toolCallId
        ? resultsByCallId.get(event.toolCallId)
        : undefined;
      consumed.add(event.id);
      if (result) consumed.add(result.id);
      units.push({ kind: "paired", call: event, result });
      continue;
    }

    if (event.kind === "tool-result") continue;

    consumed.add(event.id);
    units.push({ kind: "standalone", event });
  }

  return units;
}

export function matchProvenance(
  unit: DisplayUnit,
  provenanceEntries: SearchProvenanceEntry[],
  matchState: ProvenanceMatchState,
): SearchProvenanceEntry | undefined {
  if (unit.kind !== "paired") return undefined;
  const toolName = unit.call.toolName;
  if (!toolName || !SEARCH_TOOL_NAMES.has(toolName)) return undefined;

  if (unit.call.toolCallId) {
    const exactIdx = provenanceEntries.findIndex(
      (entry, idx) =>
        !matchState.consumed.has(idx) &&
        entry.tool_call_id === unit.call.toolCallId,
    );
    if (exactIdx >= 0) {
      matchState.consumed.add(exactIdx);
      return provenanceEntries[exactIdx];
    }
  }

  for (let idx = matchState.cursor; idx < provenanceEntries.length; idx++) {
    if (matchState.consumed.has(idx)) continue;
    const entry = provenanceEntries[idx];
    if (entry.tool_name && entry.tool_name !== toolName) continue;
    matchState.consumed.add(idx);
    matchState.cursor = idx + 1;
    return entry;
  }

  return undefined;
}
