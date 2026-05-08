import type { CSSProperties } from "react";

export const card: CSSProperties = {
  border: "1px solid #27272a",
  borderRadius: "8px",
  backgroundColor: "#111113",
  overflow: "hidden",
};

export const summaryBase: CSSProperties = {
  listStyle: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "10px 14px",
  cursor: "pointer",
  userSelect: "none",
};

export const labelStyle: CSSProperties = {
  fontSize: "12px",
  fontWeight: 600,
  color: "#ededed",
};

export const muted: CSSProperties = {
  fontSize: "11px",
  color: "#71717a",
  margin: 0,
};

export const pillBase: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "6px",
  fontSize: "11px",
  padding: "4px 10px",
  borderRadius: "9999px",
  border: "1px solid #27272a",
  backgroundColor: "#18181b",
  color: "#ededed",
};

export const unitIndexStyle: CSSProperties = {
  display: "inline-block",
  width: "28px",
  flexShrink: 0,
  fontSize: "10px",
  fontWeight: 700,
  color: "#52525b",
};

export const subLabelStyle: CSSProperties = {
  fontSize: "10px",
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  color: "#52525b",
  padding: "8px 14px 4px 14px",
};

export const chunkInputMaxHeight = "14rem";
export const chunkOutputMaxHeight = "20rem";
export const chunkEventMaxHeight = "16rem";
export const chunkSummaryMaxHeight = "18rem";

export const chunkScrollStyle: CSSProperties = {
  maxHeight: chunkOutputMaxHeight,
  overflowY: "auto",
  minHeight: 0,
};
