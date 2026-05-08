import { useState } from "react";
import { CopyableBlock } from "../ui";
import { EventBadge } from "./SessionBadges";
import { AnnotationBadge, AnnotationForm } from "../Annotations";
import {
  parseSearchToolResultRows,
  SearchProvenancePanel,
  FormattedSearchResults,
} from "../Provenance";
import {
  unitIndexStyle,
  subLabelStyle,
  chunkInputMaxHeight,
  chunkOutputMaxHeight,
  chunkEventMaxHeight,
  chunkScrollStyle,
  muted,
} from "./styles";
import { previewStr } from "../../utils";
import type { DisplayUnit } from "../../trace/types";
import type { SearchProvenanceEntry } from "../../types";
import type { Annotation, AnnotationMarker } from "../Annotations";

export function DisplayUnitRow({
  unit,
  unitIndex,
  provenance,
  annotations,
  onAnnotate,
}: {
  unit: DisplayUnit;
  unitIndex: number;
  provenance?: SearchProvenanceEntry;
  annotations?: Annotation[];
  onAnnotate?: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
}) {
  const [showAnnotationForm, setShowAnnotationForm] = useState(false);

  if (unit.kind === "paired") {
    const isSearchTool = unit.call.toolName === "stakgraph_search";
    const formattedResults =
      isSearchTool && unit.result
        ? parseSearchToolResultRows(unit.result.payload)
        : null;
    const myAnnotations = (annotations ?? []).filter(
      (a) => a.target === "tool_call" && a.target_id === unit.call.toolCallId,
    );

    return (
      <div style={{ borderTop: "1px solid #1f1f22" }}>
        <details>
          <summary
            style={{
              listStyle: "none",
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "9px 14px",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <span style={unitIndexStyle}>{unitIndex}</span>
            <span
              style={{
                fontSize: "10px",
                lineHeight: 1,
                padding: "4px 8px",
                borderRadius: "9999px",
                border: "1px solid #1e40af",
                color: "#93c5fd",
                backgroundColor: "rgba(30,64,175,0.18)",
                textTransform: "lowercase",
                flexShrink: 0,
              }}
            >
              tool
            </span>
            <span
              style={{
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "ui-monospace,monospace",
                color: "#ededed",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                flex: 1,
                minWidth: 0,
              }}
            >
              {unit.call.toolName ?? "?"}
            </span>
            {myAnnotations.map((a) => (
              <AnnotationBadge key={a.ts} marker={a.marker} note={a.note} />
            ))}
            {onAnnotate && (
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setShowAnnotationForm((f) => !f);
                }}
                title="Add annotation"
                style={{
                  fontSize: "11px",
                  lineHeight: 1,
                  padding: "3px 7px",
                  borderRadius: "9999px",
                  border: "1px solid #3f3f46",
                  backgroundColor: "transparent",
                  color: "#71717a",
                  cursor: "pointer",
                  flexShrink: 0,
                }}
              >
                +
              </button>
            )}
            <span style={muted}>{unit.result ? "→ result" : "pending"}</span>
          </summary>
          <div style={{ borderTop: "1px solid #1f1f22" }}>
            <p style={subLabelStyle}>Input</p>
            <CopyableBlock
              value={unit.call.payload}
              maxHeight={chunkInputMaxHeight}
            />
            {unit.result && (
              <>
                <p style={{ ...subLabelStyle, borderTop: "1px solid #1f1f22" }}>
                  Output
                </p>
                {formattedResults ? (
                  <div style={chunkScrollStyle}>
                    <FormattedSearchResults
                      results={formattedResults}
                      provenance={provenance}
                      rawValue={unit.result.payload}
                    />
                  </div>
                ) : (
                  <>
                    {provenance && (
                      <SearchProvenancePanel provenance={provenance} />
                    )}
                    <CopyableBlock
                      value={unit.result.payload}
                      maxHeight={chunkOutputMaxHeight}
                    />
                  </>
                )}
              </>
            )}
          </div>
        </details>
        {showAnnotationForm && (
          <AnnotationForm
            onSubmit={(marker, note) => {
              onAnnotate?.(marker, note, unit.call.toolCallId);
              setShowAnnotationForm(false);
            }}
            onCancel={() => setShowAnnotationForm(false)}
          />
        )}
      </div>
    );
  }

  const { event } = unit;
  const name =
    event.kind === "reasoning" ? "reasoning" : event.toolName || event.role;

  return (
    <details style={{ borderTop: "1px solid #1f1f22" }}>
      <summary
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          padding: "9px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <span style={unitIndexStyle}>{unitIndex}</span>
        <EventBadge event={event} />
        <span
          style={{
            fontSize: "12px",
            fontWeight: 600,
            fontFamily: "ui-monospace,monospace",
            color: "#ededed",
            flex: 1,
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {name}
        </span>
        <span
          style={{
            ...muted,
            maxWidth: "260px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {previewStr(event.text || event.payload)}
        </span>
      </summary>
      <CopyableBlock value={event.payload} maxHeight={chunkEventMaxHeight} />
    </details>
  );
}
