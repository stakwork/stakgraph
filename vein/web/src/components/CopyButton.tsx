import { useState } from "preact/hooks";
import { CopyIcon, CheckIcon } from "../icons";

// ── Copy-to-clipboard button ───────────────────────────────────────────────
//
// Writes the RAW value to the clipboard (no JSON escaping) — so a long string
// field like an optimized prompt copies as clean multi-line text, not a quoted
// `"...\n..."` blob. Shows a brief checkmark on success.

export function CopyButton(props: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async (e: Event) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(props.value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — no-op.
    }
  };

  return (
    <button
      class="copy-btn"
      onClick={onCopy}
      aria-label={props.label ?? "Copy"}
      title={copied ? "Copied!" : "Copy"}
    >
      {copied ? <CheckIcon /> : <CopyIcon />}
    </button>
  );
}
