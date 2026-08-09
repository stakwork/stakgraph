import type {
  ProductionRun,
  Annotation,
  AnnotationMarker,
  ConceptDetail,
} from "./types";

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

/**
 * Call an endpoint outside the (pre-auth, public) `/api` router.
 *
 * Concept documentation lives behind `authMiddleware`, and deliberately stays
 * there — it describes private repo internals, so it must not be proxied
 * through the sessions router. `window.__AUTH_TOKEN__` is injected into
 * index.html server-side for exactly this; it's empty in dev, where auth is
 * off anyway.
 */
async function authedReq<T>(path: string): Promise<T> {
  const token = (window as unknown as { __AUTH_TOKEN__?: string })
    .__AUTH_TOKEN__;
  const res = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
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
    annotate: (
      id: string,
      body: { target: "session" | "tool_call"; target_id?: string; marker: AnnotationMarker; note?: string; author?: string },
    ) =>
      req<Annotation>(`/sessions/${id}/annotations`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },
  concepts: {
    get: (id: string, repo?: string) =>
      authedReq<ConceptDetail>(
        `/gitree/concepts/${encodeURIComponent(id)}${
          repo ? `?repo=${encodeURIComponent(repo)}` : ""
        }`,
      ),
  },
};
