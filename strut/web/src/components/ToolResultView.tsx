import { useState } from "preact/hooks";
import { formatJson } from "../helpers";
import { CopyIcon, CheckIcon } from "../icons";

// ── Tool result disclosure ─────────────────────────────────────────────────
// Sits under a tool call's input inside the expanded chip. Closed by default:
// a one-line "Result · 1.8 KB" row; click to reveal the output. Big outputs
// show a head preview with a "show all" link (the transcript on disk is
// lossless, so "all" really is all of it).

/** Characters of a long output shown before "show all". */
const PREVIEW_CHARS = 1500;

function formatSize(chars: number): string {
  if (chars < 1000) return `${chars} chars`;
  if (chars < 100_000) return `${(chars / 1000).toFixed(1)} KB`;
  return `${Math.round(chars / 1000)} KB`;
}

export function ToolResultView(props: {
  result?: { output: unknown; isError: boolean };
  open: boolean;
  onToggle: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const [copied, setCopied] = useState(false);
  const { result, open } = props;
  if (!result) return null;

  const text = formatJson(result.output);
  const truncated = !showAll && text.length > PREVIEW_CHARS;
  const shown = truncated ? text.slice(0, PREVIEW_CHARS) : text;
  const label = result.isError ? "Error" : "Result";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard unavailable — no-op.
    }
  };

  return (
    <div class={`chat-tool-result${result.isError ? " is-error" : ""}`}>
      <button
        type="button"
        class="chat-tool-result-head"
        onClick={props.onToggle}
        aria-expanded={open}
      >
        <span class={`chat-tool-chev${open ? " is-open" : ""}`} aria-hidden="true" />
        <span class="chat-tool-result-label">{label}</span>
        <span class="chat-tool-result-size">{formatSize(text.length)}</span>
      </button>
      {open && (
        <div class="chat-tool-output">
          <button
            type="button"
            class="chat-tool-output-copy"
            onClick={copy}
            title={copied ? "Copied!" : "Copy result"}
            aria-label="Copy result"
          >
            {copied ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
          </button>
          <pre class="chat-tool-output-text">{shown}</pre>
          {truncated && (
            <div class="chat-tool-output-bar">
              showing {PREVIEW_CHARS.toLocaleString()} of {text.length.toLocaleString()} chars ·{" "}
              <button type="button" class="chat-tool-link" onClick={() => setShowAll(true)}>
                show all
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
