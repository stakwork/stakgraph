import { useState, useEffect } from "preact/hooks";
import * as api from "../api";
import { FlyoutResizer } from "./FlyoutResizer";
import { CloseIcon } from "../icons";
import { StepTypeEntry } from "./AddStepDialog";

// ── Step Info Flyout (read-only catalog view) ──────────────────────────────
//
// Opened from the sidebar's Steps catalog. Shows what a step type IS —
// description, config schema, source — independent of any workflow. The
// editable counterpart (StepEditFlyout) covers a step instance's config;
// this covers the type itself.

export function StepInfoFlyout(props: {
  entry: StepTypeEntry;
  onClose: () => void;
}) {
  const [fields, setFields] = useState<api.FieldDesc[] | null>(null);
  const [sourceOpen, setSourceOpen] = useState(false);
  const [source, setSource] = useState<api.StepSourceResponse | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);

  useEffect(() => {
    setFields(null);
    setSourceOpen(false);
    setSource(null);
    api.getStepSchema(props.entry.type)
      .then((resp) => setFields(resp.fields))
      .catch(() => setFields([]));
  }, [props.entry.type]);

  const toggleSource = () => {
    const next = !sourceOpen;
    setSourceOpen(next);
    if (next && source === null && !sourceLoading) {
      setSourceLoading(true);
      api.getStepSource(props.entry.type)
        .then(setSource)
        .catch(() => setSource({ type: props.entry.type, source: null, origin: null }))
        .finally(() => setSourceLoading(false));
    }
  };

  return (
    <div class="flyout">
      <FlyoutResizer />
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">Step Type</div>
          <div class="flyout-title">
            {props.entry.type}
            <span class="param-type-tag">{props.entry.source}</span>
          </div>
        </div>
        <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
      </div>
      <div class="flyout-body">
        {props.entry.description && (
          <div class="flyout-section">
            <div class="flyout-section-title">Description</div>
            <div class="step-info-desc">{props.entry.description}</div>
          </div>
        )}

        <div class="flyout-section">
          <div class="flyout-section-title">Config</div>
          {fields == null && <div class="flyout-source-empty">Loading…</div>}
          {fields != null && fields.length === 0 && (
            <div class="flyout-source-empty">This step takes no config.</div>
          )}
          {fields != null && fields.map((f) => (
            <div class="schema-field" key={f.name}>
              <div class="schema-field-head">
                <span class="schema-field-name">{f.name}</span>
                <span class="param-type-tag">{f.kind}</span>
                {f.required && <span class="param-type-tag schema-field-req">required</span>}
              </div>
              {f.default !== undefined && (
                <div class="schema-field-default">
                  default: {typeof f.default === "string" ? f.default : JSON.stringify(f.default)}
                </div>
              )}
              {f.enumValues && f.enumValues.length > 0 && (
                <div class="schema-field-default">one of: {f.enumValues.join(", ")}</div>
              )}
            </div>
          ))}
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
      </div>
    </div>
  );
}
