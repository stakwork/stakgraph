import { CopyableBlock } from "../ui";
import { IssueBadge } from "./SessionBadges";
import { previewStr } from "../../utils";
import type { IssueKind } from "../../trace/types";

export function EntryRow({
  kind,
  index,
  toolName,
  payload,
  flags = [],
}: {
  kind: string;
  index: number;
  toolName: string;
  payload: unknown;
  flags?: IssueKind[];
}) {
  const kindColor = kind === "call" ? "#c4b5fd" : "#93c5fd";
  return (
    <details style={{ borderTop: "1px solid #1f1f22" }}>
      <summary
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "flex-start",
          gap: "10px",
          padding: "9px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: "64px",
            flexShrink: 0,
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: kindColor,
            paddingTop: "2px",
          }}
        >
          {kind} {index}
        </span>
        <span style={{ flex: 1, minWidth: 0 }}>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              flexWrap: "wrap",
            }}
          >
            <span
              style={{
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "ui-monospace,monospace",
                color: "#ededed",
              }}
            >
              {toolName}
            </span>
            {flags.map((flag) => (
              <IssueBadge key={flag} flag={flag} />
            ))}
          </span>
          <span
            style={{
              display: "block",
              fontSize: "11px",
              color: "#71717a",
              marginTop: "2px",
              lineHeight: 1.5,
            }}
          >
            {previewStr(payload)}
          </span>
        </span>
      </summary>
      <CopyableBlock value={payload} />
    </details>
  );
}
