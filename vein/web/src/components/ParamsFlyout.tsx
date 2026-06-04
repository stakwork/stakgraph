import { useState, useEffect } from "preact/hooks";
import yaml from "js-yaml";
import { CloseIcon } from "../icons";
import { FlyoutResizer } from "./FlyoutResizer";
import { YamlEditor } from "./YamlEditor";
import { humanize } from "../helpers";

// ── Params Flyout (editable) ───────────────────────────────────────────────
//
// Edit a workflow's `params` block — the tunable default knobs (prompts,
// rubrics, expected gold, datasets, thresholds) referenced by step configs via
// {{ params.* }}. These live at the workflow level (not on any single step), so
// they'd otherwise be invisible. Editing here marks the workflow dirty; hitting
// Publish persists the params as a new workflow version (params are part of the
// workflow YAML — changing a param IS a new version).
//
// String params edit as plain text. Structured params (arrays/objects, e.g. a
// `dataset` of workspaces with embedded gold configs) edit as **YAML** in a
// syntax-highlighted pane and are parsed back on change — far more legible than
// JSON-with-escaped-newlines. While a structured param's YAML is mid-edit and
// invalid, we keep the draft text but do NOT commit it (so localParams stays
// structurally valid), surface an "invalid" marker, and report invalidity up so
// the parent can block Publish.

const dumpYaml = (v: unknown): string => {
  if (v == null) return "";
  try {
    return yaml.dump(v, { lineWidth: -1, noRefs: true });
  } catch {
    return String(v);
  }
};

export function ParamsFlyout(props: {
  workflow: string;
  params: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  /** Reports whether every structured param currently parses (parent blocks
   *  Publish while false). */
  onValidChange?: (valid: boolean) => void;
  onClose: () => void;
}) {
  const keys = Object.keys(props.params);

  // Freeze each param's editing KIND at mount (string vs yaml) so it doesn't
  // flip mid-edit if a structured value momentarily parses to a scalar. The
  // flyout remounts on workflow switch, so this re-initializes per workflow.
  const [kinds] = useState<Record<string, "string" | "yaml">>(() => {
    const m: Record<string, "string" | "yaml"> = {};
    for (const k of keys) m[k] = typeof props.params[k] === "string" ? "string" : "yaml";
    return m;
  });

  // Authoritative draft text per param (we don't reformat on every keystroke).
  const [drafts, setDrafts] = useState<Record<string, string>>(() => {
    const d: Record<string, string> = {};
    for (const k of keys) d[k] = kinds[k] === "string" ? (props.params[k] as string) : dumpYaml(props.params[k]);
    return d;
  });
  const [invalid, setInvalid] = useState<Record<string, boolean>>({});

  useEffect(() => {
    props.onValidChange?.(Object.values(invalid).every((bad) => !bad));
  }, [invalid]);

  const update = (key: string, text: string) => {
    setDrafts((d) => ({ ...d, [key]: text }));
    if (kinds[key] === "string") {
      props.onChange({ ...props.params, [key]: text });
      return;
    }
    try {
      const parsed = yaml.load(text);
      setInvalid((m) => (m[key] ? { ...m, [key]: false } : m));
      props.onChange({ ...props.params, [key]: parsed });
    } catch {
      setInvalid((m) => (m[key] ? m : { ...m, [key]: true }));
    }
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
          keys.map((k) => (
            <div class="flyout-section" key={k}>
              <div class="flyout-section-title">
                {humanize(k)}
                {kinds[k] === "yaml" && <span class="param-type-tag">yaml</span>}
                {invalid[k] && <span class="param-type-tag param-invalid">invalid</span>}
              </div>
              <YamlEditor
                value={drafts[k] ?? ""}
                language={kinds[k] === "string" ? "text" : "yaml"}
                onChange={(text) => update(k, text)}
              />
            </div>
          ))
        )}
        <div class="flyout-section">
          <span class="flyout-meta-value">
            Edits mark the workflow changed — hit Publish to save as a new version. Invalid YAML blocks publishing.
          </span>
        </div>
      </div>
    </div>
  );
}
