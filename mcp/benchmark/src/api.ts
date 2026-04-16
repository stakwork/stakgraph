import type { ProductionRun } from "./types";

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
  },
};
