import { useState, useEffect } from "preact/hooks";
import * as api from "../api";
import { StepData } from "../flow-to-canvas";
import { ConfigField } from "./ConfigField";
import { FlyoutResizer } from "./FlyoutResizer";
import { CloseIcon } from "../icons";
import yaml from "js-yaml";

// ── Step Edit Flyout ───────────────────────────────────────────────────────

export function StepEditFlyout(props: {
  step: StepData;
  allSteps: StepData[];
  onSave: (updated: StepData) => void;
  onClose: () => void;
}) {
  const [id, setId] = useState(props.step.id);
  const [config, setConfig] = useState<Record<string, any>>({ ...props.step.config });
  const [depends, setDepends] = useState<string[]>(() => {
    if (props.step.depends == null) return [];
    return Array.isArray(props.step.depends) ? [...props.step.depends] : [props.step.depends];
  });
  const [when, setWhen] = useState<boolean | undefined>(props.step.when);
  const [fields, setFields] = useState<api.FieldDesc[]>([]);
  const [error, setError] = useState("");
  const [sourceOpen, setSourceOpen] = useState(false);
  const [source, setSource] = useState<api.StepSourceResponse | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);

  // Fetch schema for this step type
  useEffect(() => {
    api.getStepSchema(props.step.type).then((resp) => {
      setFields(resp.fields);
    }).catch(() => setFields([]));
  }, [props.step.type]);

  // Lazily fetch source the first time the section is expanded (per type).
  useEffect(() => {
    setSourceOpen(false);
    setSource(null);
  }, [props.step.type]);

  const toggleSource = () => {
    const next = !sourceOpen;
    setSourceOpen(next);
    if (next && source === null && !sourceLoading) {
      setSourceLoading(true);
      api.getStepSource(props.step.type)
        .then(setSource)
        .catch(() => setSource({ type: props.step.type, source: null, origin: null }))
        .finally(() => setSourceLoading(false));
    }
  };

  // Reset state when step changes
  useEffect(() => {
    setId(props.step.id);
    setConfig({ ...props.step.config });
    const deps = props.step.depends == null ? [] : Array.isArray(props.step.depends) ? [...props.step.depends] : [props.step.depends];
    setDepends(deps);
    setWhen(props.step.when);
    setError("");
  }, [props.step]);

  const updateConfig = (name: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [name]: value }));
  };

  // Detect if any of the current deps is an `if` gate (enables the `when` field)
  const hasGateDep = depends.some((d) => {
    const dep = props.allSteps.find((s) => s.id === d);
    return dep?.type === "if";
  });

  const handleSave = () => {
    if (!id) { setError("Step must have an id"); return; }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(id)) {
      setError("ID must match [a-zA-Z_][a-zA-Z0-9_]*");
      return;
    }
    // Strip empty-string values for optional fields
    const cleanConfig: Record<string, any> = {};
    for (const [k, v] of Object.entries(config)) {
      if (v !== "" && v !== undefined) cleanConfig[k] = v;
    }
    const updated: StepData = {
      id,
      type: props.step.type,
      config: cleanConfig,
      options: props.step.options,
    };
    if (depends.length > 0) updated.depends = depends;
    // Only persist `when` if there's a gate dep
    if (when != null && hasGateDep) updated.when = when;
    props.onSave(updated);
  };

  // Build YAML preview
  const previewObj: Record<string, any> = { id, type: props.step.type, config };
  if (depends.length > 0) previewObj.depends = depends;
  if (when != null && hasGateDep) previewObj.when = when;
  const yamlPreview = yaml.dump(previewObj, { lineWidth: 120, noRefs: true });

  // Other step ids for depends checkboxes (exclude self)
  const otherStepIds = props.allSteps.map((s) => s.id).filter((sid) => sid !== props.step.id);

  return (
    <div class="flyout">
      <FlyoutResizer />
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">Edit Step</div>
          <div class="flyout-title">{props.step.type}</div>
        </div>
        <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
      </div>
      <div class="flyout-body">
        {/* Step ID */}
        <div class="flyout-field">
          <label>ID</label>
          <input
            type="text"
            value={id}
            onInput={(e) => { setId((e.target as HTMLInputElement).value); setError(""); }}
          />
        </div>

        {/* Config fields from schema */}
        {fields.length > 0 && (
          <div class="flyout-section">
            <div class="flyout-section-title">Config</div>
            {fields.map((f) => (
              <ConfigField
                key={f.name}
                field={asStringIfTemplated(f, config[f.name])}
                value={config[f.name]}
                onChange={(v) => updateConfig(f.name, v)}
              />
            ))}
          </div>
        )}

        {/* Fallback: if no schema loaded yet, show raw config fields */}
        {fields.length === 0 && Object.keys(config).length > 0 && (
          <div class="flyout-section">
            <div class="flyout-section-title">Config</div>
            {Object.entries(config).map(([key, val]) => (
              <div class="flyout-field" key={key}>
                <label>{key}</label>
                <input
                  type="text"
                  value={typeof val === "string" ? val : JSON.stringify(val)}
                  onInput={(e) => {
                    const raw = (e.target as HTMLInputElement).value;
                    try { updateConfig(key, JSON.parse(raw)); } catch { updateConfig(key, raw); }
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {/* Depends */}
        {otherStepIds.length > 0 && (
          <div class="flyout-section">
            <div class="flyout-section-title">Depends on</div>
            <div class="flyout-checkbox-group">
              {otherStepIds.map((sid) => {
                const dep = props.allSteps.find((s) => s.id === sid);
                const isGate = dep?.type === "if";
                return (
                  <label key={sid} class="flyout-checkbox-label">
                    <input
                      type="checkbox"
                      checked={depends.includes(sid)}
                      onChange={(e) => {
                        const checked = (e.target as HTMLInputElement).checked;
                        setDepends((prev) =>
                          checked ? [...prev, sid] : prev.filter((d) => d !== sid)
                        );
                      }}
                    />
                    {sid}{isGate && <span style="color:var(--text-dim);font-size:11px;"> (gate)</span>}
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {/* When (gate condition) — only shown when depending on an `if` gate */}
        {hasGateDep && (
          <div class="flyout-section">
            <div class="flyout-section-title">When (gate branch)</div>
            <div class="flyout-field">
              <select
                value={when == null ? "" : String(when)}
                onChange={(e) => {
                  const v = (e.target as HTMLSelectElement).value;
                  setWhen(v === "" ? undefined : v === "true");
                }}
              >
                <option value="">always (no gating)</option>
                <option value="true">true branch</option>
                <option value="false">false branch</option>
              </select>
            </div>
          </div>
        )}

        {/* YAML preview (read-only) */}
        <div class="flyout-section">
          <div class="flyout-section-title">YAML Preview</div>
          <pre class="flyout-yaml-preview">{yamlPreview}</pre>
        </div>

        {/* Step source (lazily fetched, read-only) */}
        <div class="flyout-section">
          <button class="flyout-source-toggle" onClick={toggleSource} type="button">
            <span class={`flyout-source-caret${sourceOpen ? " open" : ""}`}>▶</span>
            Source
            {source?.origin && (
              <span class="flyout-source-origin">{source.origin}</span>
            )}
          </button>
          {sourceOpen && (
            sourceLoading ? (
              <div class="flyout-source-empty">Loading…</div>
            ) : source?.source ? (
              <pre class="flyout-source-code">{source.source}</pre>
            ) : (
              <div class="flyout-source-empty">No source available for this step.</div>
            )
          )}
        </div>

        {error && <div style="color:var(--danger);font-size:12px;">{error}</div>}
      </div>
      <div class="flyout-actions">
        <button class="btn" onClick={props.onClose}>Cancel</button>
        <button class="btn btn-primary" onClick={handleSave}>Save</button>
      </div>
    </div>
  );
}

/**
 * If the current value contains a `{{ ... }}` template, render the field
 * as a plain string input so the user can edit the template directly. The
 * underlying schema is left untouched — once the template is removed the
 * field reverts to its declared kind on next render.
 */
function asStringIfTemplated(field: api.FieldDesc, value: unknown): api.FieldDesc {
  if (typeof value === "string" && value.includes("{{")) {
    return { ...field, kind: "string" };
  }
  return field;
}
