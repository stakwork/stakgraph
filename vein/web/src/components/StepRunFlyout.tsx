import * as api from "../api";
import { StepData } from "../flow-to-canvas";
import { formatJson, statusTone } from "../helpers";
import { CloseIcon } from "../icons";
import { Markdown, hasMarkdownField } from "./Markdown";
import { FlyoutResizer } from "./FlyoutResizer";
import yaml from "js-yaml";

export interface StepRunEvents {
  start?: api.RunEvent;
  end?: api.RunEvent;
  error?: api.RunEvent;
  skipped?: api.RunEvent;
  all: api.RunEvent[];
}

// ── Step Run Results Flyout (read-only) ────────────────────────────────────
//
// Shows a step's own I/O. For a leaf step that's input → output; for a
// container (subflow/foreach/loop) it's the aggregate "summary" (subflow:
// child input → child result; foreach: items array → results array).
// Per-child detail is reached by drilling in via the node's arrow.

export function StepRunFlyout(props: {
  step: StepData;
  events: StepRunEvents;
  onClose: () => void;
}) {
  const { step } = props;
  const disp = props.events;
  const status = disp.error ? "error" : disp.skipped ? "skipped" : disp.end ? "success" : "running";

  return (
    <>
      <div class="flyout">
        <FlyoutResizer />
        <div class="flyout-header">
          <div>
            <div class="flyout-eyebrow">Step Results</div>
            <div class="flyout-title">{step.id} <span style="color:var(--text-dim);font-weight:400;">({step.type})</span></div>
          </div>
          <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
        </div>
        <div class="flyout-body">
          {/* Status */}
          <div class="flyout-section">
            <div class="flyout-meta-row">
              <span class="flyout-meta-label">Status</span>
              <span class={`badge badge-${statusTone(status)}`}>{status}</span>
            </div>
            {disp.end?.durationMs != null && (
              <div class="flyout-meta-row">
                <span class="flyout-meta-label">Duration</span>
                <span class="flyout-meta-value">{disp.end.durationMs}ms</span>
              </div>
            )}
            {disp.start?.ts && (
              <div class="flyout-meta-row">
                <span class="flyout-meta-label">Started</span>
                <span class="flyout-meta-value">{new Date(disp.start.ts).toLocaleTimeString()}</span>
              </div>
            )}
          </div>

          {/* Input (resolved config) */}
          {disp.start?.input != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Input (resolved config)</div>
              <pre class="flyout-json">{formatJson(disp.start.input)}</pre>
            </div>
          )}

          {/* Output */}
          {disp.end?.output != null && (() => {
            const output = disp.end.output;
            if (hasMarkdownField(output)) {
              const { markdown, ...rest } = output;
              const hasRest = Object.keys(rest).length > 0;
              return (
                <div class="flyout-section">
                  <div class="flyout-section-title">Output</div>
                  <div class="flyout-markdown-wrap">
                    <Markdown source={markdown} class="md-compact" />
                  </div>
                  {hasRest && <pre class="flyout-json flyout-json-after-md">{formatJson(rest)}</pre>}
                </div>
              );
            }
            return (
              <div class="flyout-section">
                <div class="flyout-section-title">Output</div>
                <pre class="flyout-json">{formatJson(output)}</pre>
              </div>
            );
          })()}

          {/* Error */}
          {disp.error?.error != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Error</div>
              <pre class="flyout-json tone-error">{formatJson(disp.error.error)}</pre>
            </div>
          )}

          {/* Original config (from workflow definition) */}
          <div class="flyout-section">
            <div class="flyout-section-title">Config (workflow definition)</div>
            <pre class="flyout-json">{yaml.dump(step.config, { lineWidth: 120, noRefs: true }).trim()}</pre>
          </div>
        </div>
      </div>
    </>
  );
}
