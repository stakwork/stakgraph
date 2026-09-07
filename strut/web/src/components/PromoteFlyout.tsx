import { useState } from "preact/hooks";
import { structuredPatch } from "diff";
import { parseDiff, Diff, Hunk, type HunkData } from "react-diff-view";
import "react-diff-view/style/index.css";
import * as api from "../api";
import { CloseIcon } from "../icons";
import { FlyoutResizer } from "./FlyoutResizer";

// ── Promote Flyout ──────────────────────────────────────────────────────────
//
// The human-review surface for "promote a winner". A workflow declares
// `promotes: [{ from, to }]` in its YAML; after a run we resolve each spec
// against the run's OUTPUT and show, per promotion, a DIFF of the target
// param's current default → the value this run produced. Nothing is written
// until the user clicks Apply (which writes the param + publishes a new version
// of the target workflow). Declaring a promote is inert; promotion is a click.

function toText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

// Build a git-style unified diff. gitdiff-parser (react-diff-view's parseDiff)
// needs the `diff --git` + `---`/`+++` headers; jsdiff's createTwoFilesPatch
// preamble (the `===` separator + tab-suffixed names) trips it, so we format
// from structuredPatch ourselves.
function buildPatch(a: string, b: string): string {
  const sp = structuredPatch("value", "value", a, b, "", "", { context: 3 });
  if (sp.hunks.length === 0) return "";
  let txt = "diff --git a/value b/value\n--- a/value\n+++ b/value\n";
  for (const h of sp.hunks) {
    txt += `@@ -${h.oldStart},${h.oldLines} +${h.newStart},${h.newLines} @@\n`;
    txt += h.lines.join("\n") + "\n";
  }
  return txt;
}

function DiffBlock(props: { before: unknown; after: unknown }) {
  const a = toText(props.before);
  const b = toText(props.after);
  const patch = buildPatch(a, b);
  const files = patch ? parseDiff(patch, { nearbySequences: "zip" }) : [];
  const file = files[0];
  if (!file || file.hunks.length === 0) {
    return <div class="promote-nodiff">No change — this run's value matches the current default.</div>;
  }
  return (
    <div class="promote-diff">
      <Diff viewType="unified" diffType={file.type} hunks={file.hunks}>
        {(hunks: HunkData[]) => hunks.map((h) => <Hunk key={h.content} hunk={h} />)}
      </Diff>
    </div>
  );
}

function PromotionRow(props: {
  workflow: string;
  runId: string;
  promotion: api.Promotion;
  onPromoted: (r: api.PromoteResult) => void;
}) {
  const { promotion: p } = props;
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apply = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.promote(props.workflow, props.runId, p.to);
      setDone(res.version);
      props.onPromoted(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="flyout-section promote-row">
      <div class="flyout-section-title">{p.label}</div>
      <div class="promote-target">
        <span class="promote-arrow">→</span>
        <code>{p.target.workflow}</code>.<code>{p.target.param}</code>
        <span class="promote-from">from output.{p.from}</span>
      </div>

      {!p.resolved ? (
        <div class="promote-nodiff promote-unresolved">
          This run's output has no value at <code>{p.from}</code> — nothing to promote.
        </div>
      ) : (
        <DiffBlock before={p.current} after={p.value} />
      )}

      <div class="promote-actions">
        {done ? (
          <span class="badge badge-ok">promoted → {done}</span>
        ) : (
          <button
            class="btn btn-publish"
            disabled={busy || !p.resolved}
            onClick={apply}
          >
            {busy ? "Promoting…" : "Apply"}
          </button>
        )}
      </div>
      {error && <pre class="flyout-json tone-error">{error}</pre>}
    </div>
  );
}

export function PromoteFlyout(props: {
  workflow: string;
  runId: string;
  promotions: api.Promotion[];
  onClose: () => void;
  onPromoted: (r: api.PromoteResult) => void;
}) {
  return (
    <div class="flyout">
      <FlyoutResizer />
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">Promote</div>
          <div class="flyout-title">{props.workflow}</div>
        </div>
        <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
      </div>
      <div class="flyout-body">
        <div class="flyout-section">
          <span class="flyout-meta-value">
            Promote a value from this run into a target workflow's param default.
            Review the diff, then Apply to publish a new version. Nothing is
            written until you click Apply.
          </span>
        </div>
        {props.promotions.map((p) => (
          <PromotionRow
            key={p.to}
            workflow={props.workflow}
            runId={props.runId}
            promotion={p}
            onPromoted={props.onPromoted}
          />
        ))}
      </div>
    </div>
  );
}
