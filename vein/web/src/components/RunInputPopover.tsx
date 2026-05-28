import { useEffect, useRef, useState } from "preact/hooks";
import * as api from "../api";
import { ConfigField } from "./ConfigField";
import { recentRunInput } from "../storage";
import type { StepData } from "../flow-to-canvas";

// ── Run Input Popover ──────────────────────────────────────────────────────
//
// When the first step has any config slots templated as "{{ input.X }}",
// render a small form anchored under the Run button so the user can supply
// those values. Field metadata (type, required, default) is borrowed from
// the step's own schema — the popover is just a typed pass-through.

/** Match `{{ input.<ident> }}` with optional whitespace, and nothing else. */
const SINGLE_INPUT_RE = /^\s*\{\{\s*input\.([a-zA-Z_$][\w$]*)\s*\}\}\s*$/;

interface InputBinding {
  /** Name of the input key the user supplies (e.g. "owner"). */
  inputKey: string;
  /** Schema descriptor cloned from the host step's field. */
  field: api.FieldDesc;
}

/**
 * Find every config field on `step` whose value is exactly `{{ input.X }}`,
 * and pair it with the matching FieldDesc from the step's schema.
 */
export function deriveInputBindings(
  step: StepData,
  fields: api.FieldDesc[],
): InputBinding[] {
  const bindings: InputBinding[] = [];
  const seen = new Set<string>();

  for (const field of fields) {
    const raw = step.config?.[field.name];
    if (typeof raw !== "string") continue;
    const match = raw.match(SINGLE_INPUT_RE);
    if (!match) continue;
    const inputKey = match[1]!;
    if (seen.has(inputKey)) continue;
    seen.add(inputKey);
    bindings.push({
      inputKey,
      field: { ...field, name: inputKey },
    });
  }

  return bindings;
}

export function RunInputPopover(props: {
  workflow: string;
  bindings: InputBinding[];
  onSubmit: (input: Record<string, unknown>) => void;
  onClose: () => void;
}) {
  const [values, setValues] = useState<Record<string, unknown>>(() => {
    // Seed with schema defaults, then layer the user's last-used values
    // for this workflow on top (so prior choices survive page reloads).
    const init: Record<string, unknown> = {};
    for (const b of props.bindings) {
      if (b.field.default !== undefined) init[b.inputKey] = b.field.default;
    }
    const prior = recentRunInput.get(props.workflow);
    for (const b of props.bindings) {
      if (prior[b.inputKey] !== undefined) init[b.inputKey] = prior[b.inputKey];
    }
    return init;
  });
  const ref = useRef<HTMLDivElement>(null);

  // Click outside / Escape to close
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        props.onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") props.onClose();
    };
    // Defer one tick so the same click that opened the popover doesn't close it
    const t = setTimeout(() => document.addEventListener("mousedown", onDocClick), 0);
    document.addEventListener("keydown", onKey);
    return () => {
      clearTimeout(t);
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [props.onClose]);

  const handleSubmit = () => {
    // Drop undefined/empty entries so the server sees omitted keys cleanly
    const clean: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(values)) {
      if (v !== undefined && v !== "") clean[k] = v;
    }
    recentRunInput.set(props.workflow, clean);
    props.onSubmit(clean);
  };

  return (
    <div class="run-popover" ref={ref}>
      <div class="run-popover-title">Run input</div>
      {props.bindings.map((b) => (
        <ConfigField
          key={b.inputKey}
          field={b.field}
          value={values[b.inputKey]}
          onChange={(v) => setValues((prev) => ({ ...prev, [b.inputKey]: v }))}
        />
      ))}
      <div class="run-popover-actions">
        <button class="btn" onClick={props.onClose}>Cancel</button>
        <button class="btn btn-primary" onClick={handleSubmit}>Run</button>
      </div>
    </div>
  );
}
