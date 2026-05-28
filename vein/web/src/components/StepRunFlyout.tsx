import * as api from "../api";
import { StepData } from "../flow-to-canvas";
import { formatJson, statusTone } from "../helpers";
import { CloseIcon } from "../icons";
import { Markdown, hasMarkdownField } from "./Markdown";
import yaml from "js-yaml";

// ── Step Run Results Flyout (read-only) ────────────────────────────────────

export function StepRunFlyout(props: {
  step: StepData;
  events: { start?: api.RunEvent; end?: api.RunEvent; error?: api.RunEvent; skipped?: api.RunEvent; all: api.RunEvent[] };
  onClose: () => void;
}) {
  const { step, events } = props;
  const status = events.error
    ? "error"
    : events.skipped
      ? "skipped"
      : events.end
        ? "success"
        : "running";

  return (
    <>
      <div class="flyout">
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
            {events.end?.durationMs != null && (
              <div class="flyout-meta-row">
                <span class="flyout-meta-label">Duration</span>
                <span class="flyout-meta-value">{events.end.durationMs}ms</span>
              </div>
            )}
            {events.start?.ts && (
              <div class="flyout-meta-row">
                <span class="flyout-meta-label">Started</span>
                <span class="flyout-meta-value">{new Date(events.start.ts).toLocaleTimeString()}</span>
              </div>
            )}
          </div>

          {/* Input (resolved config) */}
          {events.start?.input != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Input (resolved config)</div>
              <pre class="flyout-json">{formatJson(events.start.input)}</pre>
            </div>
          )}

          {/* Output */}
          {events.end?.output != null && (() => {
            const output = events.end.output;
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
          {events.error?.error != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Error</div>
              <pre class="flyout-json tone-error">{formatJson(events.error.error)}</pre>
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
