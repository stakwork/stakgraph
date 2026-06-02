import { formatJson } from "../helpers";
import { CloseIcon } from "../icons";
import { FlyoutResizer } from "./FlyoutResizer";

// ── Params Flyout (read-only) ──────────────────────────────────────────────
//
// Shows a workflow's `params` block — the tunable default knobs (prompts,
// thresholds, sample sizes) referenced by step configs via {{ params.* }}.
// These are otherwise invisible in the UI (they live at the workflow level,
// not on any single step). String values (e.g. a big system prompt) render
// in a scrollable <pre> so the full text is inspectable; non-strings render
// as JSON. Per-run overrides still happen in the Run popover.

export function ParamsFlyout(props: {
  workflow: string;
  params: Record<string, unknown>;
  onClose: () => void;
}) {
  const keys = Object.keys(props.params);

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
            return (
              <div class="flyout-section" key={k}>
                <div class="flyout-section-title">{k}</div>
                {typeof v === "string" ? (
                  <pre class="flyout-json">{v}</pre>
                ) : (
                  <pre class="flyout-json">{formatJson(v)}</pre>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
