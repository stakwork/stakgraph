import type { ProductionRun, SearchResult, Annotation, AnnotationMarker } from "./types";

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  sessions: {
    list: () => req<ProductionRun[]>("/sessions"),
    get: (id: string) => req<ProductionRun>(`/sessions/${id}`),
    search: (q: string) => req<SearchResult[]>(`/sessions/search?q=${encodeURIComponent(q)}`),
    annotate: (
      id: string,
      body: { target: "session" | "tool_call"; target_id?: string; marker: AnnotationMarker; note?: string; author?: string },
    ) =>
      req<Annotation>(`/sessions/${id}/annotations`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },
};
