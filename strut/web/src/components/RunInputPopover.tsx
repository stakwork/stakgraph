import { useEffect, useRef, useState } from "preact/hooks";
import { ConfigField } from "./ConfigField";
import { recentRunInput } from "../storage";
import type { InputBinding } from "../run-inputs";

// ── Run Input Popover ──────────────────────────────────────────────────────
//
// A small form anchored under the Run button: the workflow's inputs (inferred
// from every step's `{{ input.X }}` references — see run-inputs.ts) on top,
// its `params` knobs below. Field metadata for inputs is borrowed from the
// consuming step's own schema — the popover is just a typed pass-through.

export function RunInputPopover(props: {
  workflow: string;
  bindings: InputBinding[];
  /** The workflow's `params` defaults (tunable knobs), editable per run. */
  params?: Record<string, unknown> | null;
  onSubmit: (input: Record<string, unknown>, params?: Record<string, unknown>) => void;
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
  // Editable copy of the workflow's `params` knobs, seeded from defaults.
  // We diff against the defaults on submit and send only overridden keys.
  const paramDefaults = props.params ?? {};
  const paramKeys = Object.keys(paramDefaults);
  const [paramValues, setParamValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const k of paramKeys) init[k] = stringifyParam(paramDefaults[k]);
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

    // Only send params that were actually changed from their default,
    // coerced back to the default's type. The server shallow-merges these
    // over the workflow's `params` defaults.
    const overrides: Record<string, unknown> = {};
    for (const k of paramKeys) {
      const edited = paramValues[k] ?? "";
      if (edited === stringifyParam(paramDefaults[k])) continue;
      overrides[k] = coerceParam(edited, paramDefaults[k]);
    }

    props.onSubmit(
      clean,
      Object.keys(overrides).length > 0 ? overrides : undefined,
    );
  };

  return (
    <div class="run-popover" ref={ref}>
      <div class="run-popover-title">Run input</div>
      <div class="run-popover-body">
        {props.bindings.length > 0 && (
          <div class="run-popover-inputs">
            <div class="run-popover-subtitle">Inputs</div>
            {props.bindings.map((b) => (
              <ConfigField
                key={b.inputKey}
                field={b.field}
                value={values[b.inputKey]}
                onChange={(v) => setValues((prev) => ({ ...prev, [b.inputKey]: v }))}
              />
            ))}
          </div>
        )}
        {paramKeys.length > 0 && (
          <div class="run-popover-params">
            <div class="run-popover-subtitle">Params</div>
            {paramKeys.map((k) => {
              const isMultiline =
                typeof paramDefaults[k] === "string" &&
                ((paramValues[k] ?? "").length > 40 || (paramValues[k] ?? "").includes("\n"));
              return (
                <label class="run-popover-param" key={k}>
                  <span class="run-popover-param-name">{k}</span>
                  {isMultiline ? (
                    <textarea
                      rows={4}
                      value={paramValues[k]}
                      onInput={(e) =>
                        setParamValues((prev) => ({ ...prev, [k]: (e.target as HTMLTextAreaElement).value }))
                      }
                    />
                  ) : (
                    <input
                      type="text"
                      value={paramValues[k]}
                      onInput={(e) =>
                        setParamValues((prev) => ({ ...prev, [k]: (e.target as HTMLInputElement).value }))
                      }
                    />
                  )}
                </label>
              );
            })}
          </div>
        )}
      </div>
      <div class="run-popover-actions">
        <button class="btn" onClick={props.onClose}>Cancel</button>
        <button class="btn btn-primary" onClick={handleSubmit}>Run</button>
      </div>
    </div>
  );
}

/** Render a param default as editable text. Objects/arrays become JSON. */
function stringifyParam(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  if (typeof v === "object") return JSON.stringify(v, null, 2);
  return String(v);
}

/** Coerce an edited string back to the type of its default value. */
function coerceParam(edited: string, def: unknown): unknown {
  if (typeof def === "number") {
    const n = Number(edited);
    return Number.isNaN(n) ? edited : n;
  }
  if (typeof def === "boolean") return edited === "true";
  if (def != null && typeof def === "object") {
    try {
      return JSON.parse(edited);
    } catch {
      return edited;
    }
  }
  return edited;
}
