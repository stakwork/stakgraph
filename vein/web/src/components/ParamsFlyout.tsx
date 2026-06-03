import { CloseIcon } from "../icons";
import { FlyoutResizer } from "./FlyoutResizer";

// ── Params Flyout (editable) ───────────────────────────────────────────────
//
// Edit a workflow's `params` block — the tunable default knobs (prompts,
// rubrics, expected gold, thresholds) referenced by step configs via
// {{ params.* }}. These live at the workflow level (not on any single step),
// so they'd otherwise be invisible. Editing here marks the workflow dirty;
// hitting Publish persists the params as a new workflow version (params are
// part of the workflow YAML — changing a param IS a new version).
//
// String params edit as text; non-string params (objects/arrays, e.g. a
// dataset `examples` list) edit as JSON and are parsed back on change (kept as
// raw text while the JSON is mid-edit/invalid).

export function ParamsFlyout(props: {
  workflow: string;
  params: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  onClose: () => void;
}) {
  const keys = Object.keys(props.params);

  const update = (key: string, raw: string, isJson: boolean) => {
    const value: unknown = isJson ? tryParseJson(raw) : raw;
    props.onChange({ ...props.params, [key]: value });
  };

  return (
    <div class="flyout">
      <FlyoutResizer />
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">Params</div>
          <div class="flyout-title">{props.workflow}</div>
        </div>
        <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
      </div>
      <div class="flyout-body">
        {keys.length === 0 ? (
          <div class="flyout-section">
            <span class="flyout-meta-value">This workflow has no params.</span>
          </div>
        ) : (
          keys.map((k) => {
            const v = props.params[k];
            const isString = typeof v === "string";
            const text = isString ? v : stringify(v);
            const rows = Math.min(24, Math.max(3, text.split("\n").length + 1));
            return (
              <div class="flyout-section" key={k}>
                <div class="flyout-section-title">
                  {k}
                  {!isString && <span class="param-type-tag">json</span>}
                </div>
                <textarea
                  class="param-edit"
                  rows={rows}
                  value={text}
                  spellcheck={false}
                  onInput={(e) => update(k, (e.target as HTMLTextAreaElement).value, !isString)}
                />
              </div>
            );
          })
        )}
        <div class="flyout-section">
          <span class="flyout-meta-value">Edits mark the workflow changed — hit Publish to save as a new version.</span>
        </div>
      </div>
    </div>
  );
}

function stringify(v: unknown): string {
  if (v == null) return "";
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

/** Parse JSON, but fall back to the raw string while it's mid-edit/invalid so
 *  the textarea stays usable (the user can keep typing). */
function tryParseJson(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}
