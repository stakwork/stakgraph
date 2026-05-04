import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export const card: React.CSSProperties = {
  border: "1px solid #27272a",
  borderRadius: "12px",
  backgroundColor: "#111113",
};

export const muted: React.CSSProperties = {
  color: "#71717a",
  fontSize: "12px",
};

export function StatTile({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div style={{ ...card, padding: "14px" }}>
      <p style={{ ...muted, margin: 0 }}>{label}</p>
      <p
        style={{
          margin: "6px 0 0 0",
          fontSize: "22px",
          fontWeight: 700,
          color: "#ededed",
        }}
      >
        {value}
      </p>
      {detail && <p style={{ ...muted, margin: "6px 0 0 0" }}>{detail}</p>}
    </div>
  );
}

export function FilterSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        fontSize: "12px",
        borderRadius: "8px",
        padding: "8px 10px",
        backgroundColor: "#18181b",
        color: "#ededed",
        border: "1px solid #27272a",
        outline: "none",
      }}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function TableCard({
  title,
  badge,
  columns,
  rows,
}: {
  title: string;
  badge?: string;
  columns: string[];
  rows: React.ReactNode;
}) {
  return (
    <div style={card}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "8px",
          padding: "12px 14px",
          borderBottom: "1px solid #27272a",
        }}
      >
        <p
          style={{
            margin: 0,
            fontSize: "13px",
            fontWeight: 700,
            color: "#ededed",
          }}
        >
          {title}
        </p>
        {badge && <span style={muted}>{badge}</span>}
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  style={{
                    textAlign: "left",
                    fontSize: "11px",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: "#71717a",
                    padding: "10px 14px",
                    borderBottom: "1px solid #1f1f22",
                    whiteSpace: "nowrap",
                  }}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
  );
}

export function tdStyle(emphasis = false): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderBottom: "1px solid #1a1a1d",
    fontSize: "12px",
    color: emphasis ? "#ededed" : "#d4d4d8",
    fontWeight: emphasis ? 600 : 400,
    whiteSpace: "nowrap",
  };
}

export const preStyle: React.CSSProperties = {
  fontSize: "11px",
  lineHeight: 1.6,
  whiteSpace: "pre-wrap",
  overflowWrap: "break-word",
  wordBreak: "break-word",
  color: "#ededed",
  margin: 0,
  padding: "10px 14px",
  maxHeight: "20rem",
  overflowY: "auto",
  backgroundColor: "#0d0d0f",
  borderTop: "1px solid #27272a",
};

const blockStyle: React.CSSProperties = {
  overflowX: "auto",
  backgroundColor: "#0d0d0f",
  borderTop: "1px solid #27272a",
};

const markdownStyle: React.CSSProperties = {
  fontSize: "12px",
  lineHeight: 1.65,
  color: "#ededed",
  padding: "10px 14px",
  overflowWrap: "break-word",
  wordBreak: "break-word",
};

export function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}\u2026${id.slice(-4)}` : id;
}

function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function displayString(value: string): string {
  return value
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n");
}

export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
      title="Copy"
      style={{
        position: "absolute",
        top: 6,
        right: 8,
        background: "none",
        border: "1px solid #3f3f46",
        borderRadius: "4px",
        color: copied ? "#4ade80" : "#71717a",
        cursor: "pointer",
        fontSize: "11px",
        lineHeight: 1,
        padding: "3px 6px",
        opacity: undefined,
        transition: "opacity 0.15s, color 0.15s",
        fontFamily: "ui-monospace,monospace",
      }}
      className="copy-btn"
    >
      {copied ? "✓" : "⎘"}
    </button>
  );
}

function looksLikeJson(s: string): boolean {
  const t = s.trim();
  if (t[0] !== "{" && t[0] !== "[") return false;
  try {
    JSON.parse(t);
    return true;
  } catch {
    return false;
  }
}

export function CopyableBlock({
  value,
  maxHeight,
}: {
  value: unknown;
  maxHeight?: React.CSSProperties["maxHeight"];
}) {
  const text = stringify(value);
  const isText = typeof value === "string";
  const isJson = isText && looksLikeJson(value as string);
  const scrollStyle: React.CSSProperties = maxHeight
    ? { maxHeight, overflowY: "auto" }
    : {};
  return (
    <div style={{ position: "relative" }} className="copyable-block">
      <CopyButton text={text} />
      {isText && !isJson ? (
        <div style={{ ...blockStyle, ...scrollStyle }}>
          <div style={markdownStyle} className="markdown-block">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {displayString(value as string)}
            </ReactMarkdown>
          </div>
        </div>
      ) : (
        <pre
          style={{
            ...preStyle,
            maxHeight,
            overflowY: maxHeight ? "auto" : undefined,
          }}
        >
          {isJson
            ? JSON.stringify(JSON.parse((value as string).trim()), null, 2)
            : text}
        </pre>
      )}
    </div>
  );
}
