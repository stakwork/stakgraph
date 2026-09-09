// Which `input.*` keys a workflow needs at run time, inferred from its steps.
//
// The Run popover asks the user for these before launching. A YAML workflow
// declares no input schema (every YAML flow parses input as `z.any()`), so the
// keys are read off the `{{ … }}` templates in step configs — across ALL
// steps, not just the first. AI-authored workflows routinely lead with an
// `artifacts/dir` step that references no input at all, and a first-step-only
// scan produced an empty form for them while the real inputs sat in step 3.
//
// Field metadata is borrowed from a step's own schema whenever a config slot
// is exactly `{{ input.X }}` (so number/enum widgets and defaults work). Every
// other reference becomes a plain string field. A key is optional when every
// reference to it tolerates absence — `input.X || …`, `input.X ?? …`,
// `input?.X`, `input.X ? … : …` — and required when any reference is bare.
// Only text inside `{{ … }}` counts: prose that happens to say "input.url"
// (an agent prompt explaining the workflow to itself) is not a reference.

import type { FieldDesc } from "./api";
import type { StepData } from "./flow-to-canvas";

export interface InputBinding {
  /** Name of the input key the user supplies (e.g. "media_url"). */
  inputKey: string;
  /** Schema descriptor for the form widget. `required` is the inferred value. */
  field: FieldDesc;
}

/** A config slot that is exactly one `{{ input.X }}` template and nothing else. */
const SINGLE_INPUT_RE = /^\s*\{\{\s*input\.([a-zA-Z_$][\w$]*)\s*\}\}\s*$/;
/** Every `{{ … }}` segment of a template string. */
const SEGMENT_RE = /\{\{([\s\S]*?)\}\}/g;
/** An `input.X`, `input?.X` or `input["X"]` reference inside an expression. */
const REF_RE = /\binput(?:(\?)?\.([a-zA-Z_$][\w$]*)|\[\s*(["'])([^"']+)\3\s*\])/g;

export interface InputRef {
  key: string;
  /** The expression tolerates this key being absent. */
  optional: boolean;
}

/** Every input reference in one expression body (the text between `{{` and `}}`). */
export function refsInExpr(expr: string): InputRef[] {
  const refs: InputRef[] = [];
  for (const m of expr.matchAll(REF_RE)) {
    const key = (m[2] ?? m[4])!;
    const before = expr.slice(0, m.index).trimEnd();
    const after = expr.slice(m.index! + m[0].length).trimStart();
    const optional =
      m[1] === "?" || // input?.X
      after.startsWith("||") || // input.X || fallback
      after.startsWith("?") || // input.X ?? fallback, input.X ? a : b, input.X?.y
      before.endsWith("||") || // a || input.X — the author handles absence
      before.endsWith("??");
    refs.push({ key, optional });
  }
  return refs;
}

/** Every expression inside a config value, recursively. */
function* exprsIn(value: unknown): Generator<string> {
  if (typeof value === "string") {
    for (const m of value.matchAll(SEGMENT_RE)) yield m[1]!;
  } else if (Array.isArray(value)) {
    for (const v of value) yield* exprsIn(v);
  } else if (value && typeof value === "object") {
    for (const v of Object.values(value)) yield* exprsIn(v);
  }
}

/** Every step in the flow, including `onError` handlers and loop/foreach bodies. */
function* allSteps(steps: StepData[]): Generator<StepData> {
  for (const step of steps) {
    yield step;
    const body = step.config?.["body"];
    if (body && typeof body === "object" && typeof body.type === "string") yield* allSteps([body as StepData]);
    if (step.options?.onError) yield* allSteps([step.options.onError]);
  }
}

/** Distinct step types in the flow — the schemas the popover wants to fetch. */
export function stepTypesIn(steps: StepData[]): string[] {
  const types = new Set<string>();
  for (const s of allSteps(steps)) types.add(s.type);
  return [...types];
}

/**
 * Derive the popover's fields from every step in the flow. Required keys come
 * first, then optional ones, each in first-seen order. `schemaFor` may return
 * undefined for a type whose schema failed to load — its refs still surface,
 * just untyped.
 */
export function deriveInputBindings(
  steps: StepData[],
  schemaFor: (type: string) => FieldDesc[] | undefined,
): InputBinding[] {
  const order: string[] = []; // first-seen order
  const typed = new Map<string, FieldDesc>(); // key → borrowed field (first exact slot wins)
  const required = new Set<string>();
  const seen = (key: string) => {
    if (!order.includes(key)) order.push(key);
  };

  for (const step of allSteps(steps)) {
    const fields = schemaFor(step.type) ?? [];
    const config: Record<string, unknown> = step.config ?? {};
    const exactSlots = new Set<string>();

    // 1. A slot that is exactly `{{ input.X }}` borrows that field's schema.
    //    Its requiredness is the step's own: an optional step field stays optional.
    for (const field of fields) {
      const raw = config[field.name];
      if (typeof raw !== "string") continue;
      const m = raw.match(SINGLE_INPUT_RE);
      if (!m) continue;
      const key = m[1]!;
      exactSlots.add(field.name);
      seen(key);
      if (!typed.has(key)) typed.set(key, { ...field, name: key });
      if (field.required) required.add(key);
    }

    // 2. Every other reference — nested objects, multi-segment strings,
    //    expressions — is a plain string field, required unless guarded.
    for (const [name, raw] of Object.entries(config)) {
      if (exactSlots.has(name)) continue;
      for (const expr of exprsIn(raw)) {
        for (const ref of refsInExpr(expr)) {
          seen(ref.key);
          if (!ref.optional) required.add(ref.key);
        }
      }
    }
  }

  const toBinding = (key: string): InputBinding => {
    const base: FieldDesc = typed.get(key) ?? { name: key, kind: "string", required: false };
    return { inputKey: key, field: { ...base, required: required.has(key) } };
  };
  return [
    ...order.filter((k) => required.has(k)).map(toBinding),
    ...order.filter((k) => !required.has(k)).map(toBinding),
  ];
}
