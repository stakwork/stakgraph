import type { SearchProvenanceEntry, SearchResultMeta } from "../types";
import { CopyableBlock, shortId } from "./ui";

export type SearchToolResultRow = {
  name?: string;
  node_type?: string;
  file?: string;
  lines?: string;
  ref_id?: string;
  description?: string;
};

function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function parseSearchToolResultRows(value: unknown): SearchToolResultRow[] | null {
  const raw = typeof value === "string" ? value : stringify(value);
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return parsed
      .filter((item) => item && typeof item === "object")
      .map((item) => item as SearchToolResultRow);
  } catch {
    return null;
  }
}

// both=indigo(#6366f1)  vector=zinc-mid  fulltext=zinc-dark
export function sourceColor(sources: string[]): { border: string; bg: string; fg: string; label: string } {
  const hasFt = sources.includes("fulltext");
  const hasVec = sources.includes("vector");
  if (hasFt && hasVec)
    return { border: "#6366f1", bg: "rgba(99,102,241,0.07)", fg: "#818cf8", label: "both" };
  if (hasVec)
    return { border: "#3f3f46", bg: "rgba(63,63,70,0.10)", fg: "#71717a", label: "vector" };
  return { border: "#27272a", bg: "rgba(39,39,42,0.60)", fg: "#52525b", label: "fulltext" };
}

export function ProvenanceBadge({ meta }: { meta: SearchResultMeta }) {
  const sc = sourceColor(meta.sources);
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        fontSize: "10px",
        padding: "2px 6px",
        borderRadius: "4px",
        border: `1px solid ${sc.border}`,
        backgroundColor: sc.bg,
        color: sc.fg,
        fontFamily: "ui-monospace,monospace",
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ fontWeight: 600 }}>{sc.label}</span>
      {meta.fulltext_rank != null && (
        <span style={{ color: "#52525b", opacity: 0.9 }}>ft#{meta.fulltext_rank}</span>
      )}
      {meta.vector_rank != null && (
        <span style={{ color: "#71717a", opacity: 0.9 }}>
          vec#{meta.vector_rank}
          {meta.vector_score != null && (
            <span style={{ opacity: 0.7 }}> {meta.vector_score.toFixed(2)}</span>
          )}
        </span>
      )}
      {meta.rrf_score != null && (
        <span style={{ color: "#818cf8", opacity: 0.9 }}>rrf {meta.rrf_score.toFixed(4)}</span>
      )}
    </span>
  );
}

export function SearchProvenancePanel({ provenance }: { provenance: SearchProvenanceEntry }) {
  const { method, result_meta } = provenance.provenance;
  if (!result_meta || result_meta.length === 0) return null;

  const vectorOnly = result_meta.filter(
    (m) => m.sources.length === 1 && m.sources[0] === "vector",
  ).length;
  const ftOnly = result_meta.filter(
    (m) => m.sources.length === 1 && m.sources[0] === "fulltext",
  ).length;
  const both = result_meta.filter((m) => m.sources.length === 2).length;

  return (
    <div
      style={{
        borderTop: "1px solid #1f1f22",
        padding: "8px 14px",
        backgroundColor: "rgba(99,102,241,0.04)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "6px",
          flexWrap: "wrap",
        }}
      >
        <span
          style={{
            fontSize: "10px",
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            color: "#6366f1",
          }}
        >
          provenance
        </span>
        <span style={{ fontSize: "10px", color: "#71717a" }}>
          method: <span style={{ color: "#d4d4d8" }}>{method}</span>
        </span>
        {method === "hybrid" && (
          <span style={{ fontSize: "10px", color: "#71717a" }}>
            both:{both} · ft-only:{ftOnly} · vec-only:{vectorOnly}
          </span>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
        {result_meta.map((meta, i) => (
          <div
            key={meta.ref_id || i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "3px 0",
              borderLeft: `3px solid ${sourceColor(meta.sources).border}`,
              paddingLeft: "8px",
            }}
          >
            <span style={{ fontSize: "10px", color: "#52525b", minWidth: "16px" }}>
              {i + 1}
            </span>
            <ProvenanceBadge meta={meta} />
            <span
              style={{
                fontSize: "11px",
                color: "#a1a1aa",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                flex: 1,
                minWidth: 0,
              }}
            >
              {meta.ref_id.length > 20
                ? `${meta.ref_id.slice(0, 8)}…${meta.ref_id.slice(-4)}`
                : meta.ref_id}
            </span>
            {meta.sources.length === 1 && method === "hybrid" && (
              <span
                style={{
                  fontSize: "9px",
                  color: "#52525b",
                  opacity: 0.9,
                  flexShrink: 0,
                }}
              >
                ⚠ only in {meta.sources[0]}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function SearchProvenanceSummary({ provenance }: { provenance: SearchProvenanceEntry }) {
  const { method, result_meta } = provenance.provenance;
  if (!result_meta || result_meta.length === 0) return null;

  const vectorOnly = result_meta.filter(
    (m) => m.sources.length === 1 && m.sources[0] === "vector",
  ).length;
  const ftOnly = result_meta.filter(
    (m) => m.sources.length === 1 && m.sources[0] === "fulltext",
  ).length;
  const both = result_meta.filter((m) => m.sources.length === 2).length;

  return (
    <div
      style={{
        borderTop: "1px solid #1f1f22",
        padding: "8px 14px",
        backgroundColor: "rgba(99,102,241,0.04)",
        display: "flex",
        alignItems: "center",
        gap: "8px",
        flexWrap: "wrap",
      }}
    >
      <span
        style={{
          fontSize: "10px",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.1em",
          color: "#6366f1",
        }}
      >
        provenance
      </span>
      <span style={{ fontSize: "10px", color: "#71717a" }}>
        method: <span style={{ color: "#d4d4d8" }}>{method}</span>
      </span>
      {method === "hybrid" && (
        <span style={{ fontSize: "10px", color: "#71717a" }}>
          both:{both} · ft-only:{ftOnly} · vec-only:{vectorOnly}
        </span>
      )}
    </div>
  );
}

export function FormattedSearchResults({
  results,
  provenance,
  rawValue,
}: {
  results: SearchToolResultRow[];
  provenance?: SearchProvenanceEntry;
  rawValue: unknown;
}) {
  const metaByRef = new Map(
    (provenance?.provenance.result_meta ?? []).map((meta) => [meta.ref_id, meta]),
  );

  return (
    <>
      {provenance && <SearchProvenanceSummary provenance={provenance} />}
      <div style={{ borderTop: "1px solid #1f1f22", backgroundColor: "#0d0d0f" }}>
        {results.map((result, index) => {
          const meta = result.ref_id ? metaByRef.get(result.ref_id) : undefined;
          const theme = sourceColor(meta?.sources ?? []);
          return (
            <div
              key={result.ref_id || `${result.name || "result"}-${index}`}
              style={{
                padding: "10px 14px",
                borderTop: index === 0 ? "none" : "1px solid #1f1f22",
                borderLeft: `3px solid ${meta ? theme.border : "#27272a"}`,
                backgroundColor: meta ? theme.bg : "transparent",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  flexWrap: "wrap",
                  marginBottom: "4px",
                }}
              >
                <span style={{ fontSize: "10px", color: "#52525b", minWidth: "16px" }}>
                  {index + 1}
                </span>
                <span style={{ fontSize: "12px", fontWeight: 600, color: "#ededed" }}>
                  {result.name || "(unnamed)"}
                </span>
                {result.node_type && (
                  <span
                    style={{
                      fontSize: "10px",
                      padding: "2px 6px",
                      borderRadius: "9999px",
                      border: "1px solid #3f3f46",
                      color: "#a1a1aa",
                      backgroundColor: "rgba(63,63,70,0.2)",
                      fontFamily: "ui-monospace,monospace",
                    }}
                  >
                    {result.node_type}
                  </span>
                )}
                {meta && <ProvenanceBadge meta={meta} />}
              </div>
              {(result.file || result.lines) && (
                <div
                  style={{
                    fontSize: "11px",
                    color: "#71717a",
                    fontFamily: "ui-monospace,monospace",
                    marginBottom: result.description ? "4px" : 0,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {result.file || "?"}
                  {result.lines ? `:${result.lines}` : ""}
                </div>
              )}
              {result.description && (
                <div
                  style={{
                    fontSize: "11px",
                    lineHeight: 1.5,
                    color: "#d4d4d8",
                    whiteSpace: "pre-wrap",
                    overflowWrap: "break-word",
                  }}
                >
                  {result.description}
                </div>
              )}
              {result.ref_id && (
                <div
                  style={{
                    marginTop: "6px",
                    fontSize: "10px",
                    color: "#52525b",
                    fontFamily: "ui-monospace,monospace",
                  }}
                >
                  {shortId(result.ref_id)}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <details style={{ borderTop: "1px solid #1f1f22" }}>
        <summary
          style={{
            listStyle: "none",
            padding: "8px 14px",
            cursor: "pointer",
            fontSize: "10px",
            color: "#71717a",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
          }}
        >
          raw json
        </summary>
        <CopyableBlock value={rawValue} />
      </details>
    </>
  );
}
