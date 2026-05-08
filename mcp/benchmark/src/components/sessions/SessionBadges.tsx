import { formatSourceLabel } from "../../utils";
import type { IssueKind, TraceEventKind, TraceTurnKind } from "../../trace/types";

export function sourceTheme(source: string): {
  fg: string;
  bg: string;
  border: string;
} {
  const key = (source || "unknown").toLowerCase();
  if (key.includes("gitree")) {
    return { fg: "#f5d08a", bg: "rgba(120,53,15,0.28)", border: "#78350f" };
  }
  if (key.includes("ask") || key.includes("explore")) {
    return { fg: "#93c5fd", bg: "rgba(30,64,175,0.28)", border: "#1e40af" };
  }
  if (key.includes("log")) {
    return { fg: "#86efac", bg: "rgba(21,128,61,0.24)", border: "#166534" };
  }
  return { fg: "#d4d4d8", bg: "rgba(63,63,70,0.32)", border: "#3f3f46" };
}

export function MetaPill({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        fontSize: "11px",
        padding: "4px 10px",
        borderRadius: "9999px",
        border: "1px solid #27272a",
        backgroundColor: "#18181b",
        color: "#ededed",
      }}
    >
      <span
        style={{
          color: "#71717a",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "10px",
        }}
      >
        {label}
      </span>
      <span style={{ color: "#ededed" }}>{value}</span>
    </span>
  );
}

export function SourceBadge({ source }: { source: string }) {
  const theme = sourceTheme(source);
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "9999px",
        border: `1px solid ${theme.border}`,
        backgroundColor: theme.bg,
        color: theme.fg,
        fontSize: "11px",
        fontWeight: 600,
        textTransform: "lowercase",
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      {formatSourceLabel(source)}
    </span>
  );
}

export function TurnBadge({ kind }: { kind: TraceTurnKind }) {
  const color =
    kind === "setup"
      ? { fg: "#d4d4d8", bg: "#18181b", border: "#3f3f46" }
      : kind === "tool"
        ? { fg: "#93c5fd", bg: "rgba(30,64,175,0.18)", border: "#1e40af" }
        : { fg: "#c4b5fd", bg: "rgba(76,29,149,0.18)", border: "#4c1d95" };

  return (
    <span
      style={{
        fontSize: "10px",
        lineHeight: 1,
        padding: "4px 8px",
        borderRadius: "9999px",
        border: `1px solid ${color.border}`,
        color: color.fg,
        backgroundColor: color.bg,
        textTransform: "lowercase",
      }}
    >
      {kind === "setup" ? "setup" : "turn"}
    </span>
  );
}

export function EventBadge({ event }: { event: { kind: TraceEventKind } }) {
  const label = event.kind.replace(/-/g, " ");
  const color =
    event.kind === "system"
      ? { fg: "#d4d4d8", bg: "#18181b", border: "#3f3f46" }
      : event.kind === "user"
        ? { fg: "#86efac", bg: "rgba(21,128,61,0.18)", border: "#166534" }
        : event.kind === "reasoning"
          ? { fg: "#fbbf24", bg: "rgba(146,64,14,0.16)", border: "#92400e" }
          : event.kind === "assistant-text" || event.kind === "assistant"
            ? { fg: "#c4b5fd", bg: "rgba(76,29,149,0.18)", border: "#4c1d95" }
            : { fg: "#93c5fd", bg: "rgba(30,64,175,0.18)", border: "#1e40af" };

  return (
    <span
      style={{
        fontSize: "10px",
        lineHeight: 1,
        padding: "4px 8px",
        borderRadius: "9999px",
        border: `1px solid ${color.border}`,
        color: color.fg,
        backgroundColor: color.bg,
        textTransform: "lowercase",
      }}
    >
      {label}
    </span>
  );
}

export function IssueBadge({ flag }: { flag: IssueKind }) {
  const colors: Record<IssueKind, { fg: string; bg: string; border: string }> =
    {
      empty: {
        fg: "#fca5a5",
        bg: "rgba(127,29,29,0.35)",
        border: "#7f1d1d",
      },
      oversized: {
        fg: "#fdba74",
        bg: "rgba(124,45,18,0.35)",
        border: "#7c2d12",
      },
      repeat: {
        fg: "#fcd34d",
        bg: "rgba(120,53,15,0.35)",
        border: "#78350f",
      },
      fallback: {
        fg: "#f9a8d4",
        bg: "rgba(131,24,67,0.35)",
        border: "#831843",
      },
      duplicate: {
        fg: "#c4b5fd",
        bg: "rgba(76,29,149,0.35)",
        border: "#4c1d95",
      },
      "noisy-overview": {
        fg: "#93c5fd",
        bg: "rgba(30,64,175,0.35)",
        border: "#1e40af",
      },
    };
  const color = colors[flag];
  return (
    <span
      style={{
        fontSize: "10px",
        lineHeight: 1,
        padding: "4px 6px",
        borderRadius: "9999px",
        border: `1px solid ${color.border}`,
        color: color.fg,
        backgroundColor: color.bg,
        textTransform: "lowercase",
      }}
    >
      {flag}
    </span>
  );
}
