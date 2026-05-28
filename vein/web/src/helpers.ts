
// ── Helpers ────────────────────────────────────────────────────────────────

import { StepData } from "./flow-to-canvas";

/** Normalize a step list so that semantically-equal lists compare equal,
 *  regardless of key order or whether `depends` is a string vs single-item array. */
export  function normalizeSteps(steps: StepData[]): unknown {
  return steps.map((s) => {
    const deps = s.depends == null
      ? undefined
      : Array.isArray(s.depends) ? s.depends : [s.depends];
    return {
      id: s.id,
      type: s.type,
      config: s.config ?? {},
      depends: deps,
      when: s.when,
      options: s.options,
    };
  });
}

export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null) return a === b;
  if (typeof a !== typeof b) return false;
  if (typeof a !== "object") return a === b;
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  if (Array.isArray(b)) return false;
  const ao = a as Record<string, unknown>;
  const bo = b as Record<string, unknown>;
  const aKeys = Object.keys(ao).filter((k) => ao[k] !== undefined);
  const bKeys = Object.keys(bo).filter((k) => bo[k] !== undefined);
  if (aKeys.length !== bKeys.length) return false;
  for (const k of aKeys) {
    if (!deepEqual(ao[k], bo[k])) return false;
  }
  return true;
}

export function statusTone(s: string) {
  if (s === "success") return "ok";
  if (s === "error") return "danger";
  if (s === "skipped") return "muted";
  return "warning";
}

export function eventTone(t: string) { return t.includes("error") ? "error" : t.includes("end") ? "end" : t.includes("start") ? "start" : t.includes("retry") ? "retry" : "other"; }

export function formatJson(v: unknown): string {
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}