import { groupEvents, matchProvenance } from "../../trace/group";
import { usageOf, formatUsageParts, formatNumber } from "../../utils";
import { DisplayUnitRow } from "./DisplayUnitRow";
import { TurnBadge } from "./SessionBadges";
import { card, muted } from "./styles";
import type { TraceTurn, DisplayUnit, ProvenanceMatchState } from "../../trace/types";
import type { StepMeta, SearchProvenanceEntry } from "../../types";
import type { Annotation, AnnotationMarker } from "../Annotations";

function formatTimeOfDay(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function StepDivider({ meta }: { meta: StepMeta }) {
  const label = meta.label;
  const isFinal = meta.toolCalls.length === 0 && !label;
  const usage = usageOf(meta.usage);
  const timeOfDay = formatTimeOfDay(meta.timestamp);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "4px 14px",
        fontSize: "10px",
        fontFamily: "ui-monospace,monospace",
        color: "#71717a",
        background: "rgba(163,230,53,0.04)",
        borderTop: "1px solid #2a2a2e",
      }}
    >
      <span style={{ color: "#52525b", fontWeight: 700, minWidth: 20 }}>
        S{meta.step}
      </span>
      {label ? (
        <span
          title={label}
          style={{
            color: "#a3e635",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {label}
        </span>
      ) : null}
      {isFinal ? <span style={{ color: "#a3e635" }}>final answer</span> : null}
      <span style={{ color: "#a3e635" }}>{formatUsageParts(usage)}</span>
      <span style={{ color: "#71717a" }}>
        total {formatNumber(usage.total)}
      </span>
      {timeOfDay ? (
        <span
          title={new Date(meta.timestamp).toLocaleString()}
          style={{ color: "#52525b", marginLeft: "auto" }}
        >
          {timeOfDay}
        </span>
      ) : null}
    </div>
  );
}

function interleaveUnitsWithSteps(
  units: DisplayUnit[],
  metas: StepMeta[],
  provenanceEntries: SearchProvenanceEntry[],
  annotations: Annotation[],
  onAnnotate?: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void,
): React.ReactNode[] {
  const matchState: ProvenanceMatchState = {
    cursor: 0,
    consumed: new Set<number>(),
  };

  if (metas.length === 0) {
    return units.map((unit, i) => (
      <DisplayUnitRow
        key={unit.kind === "paired" ? unit.call.id : unit.event.id}
        unit={unit}
        unitIndex={i + 1}
        provenance={matchProvenance(unit, provenanceEntries, matchState)}
        annotations={annotations}
        onAnnotate={onAnnotate}
      />
    ));
  }

  const nodes: React.ReactNode[] = [];
  let metaIdx = 0;
  let toolPtr = 0;
  let unitNum = 0;

  for (const unit of units) {
    const isPaired = unit.kind === "paired";
    if (isPaired && metaIdx < metas.length) {
      const currentMeta = metas[metaIdx];
      if (toolPtr === 0) {
        nodes.push(
          <StepDivider key={`step-${currentMeta.step}`} meta={currentMeta} />,
        );
      }
      toolPtr++;
      if (toolPtr >= currentMeta.toolCalls.length) {
        metaIdx++;
        toolPtr = 0;
      }
    }
    unitNum++;
    nodes.push(
      <DisplayUnitRow
        key={unit.kind === "paired" ? unit.call.id : unit.event.id}
        unit={unit}
        unitIndex={unitNum}
        provenance={matchProvenance(unit, provenanceEntries, matchState)}
        annotations={annotations}
        onAnnotate={onAnnotate}
      />,
    );
  }

  if (metaIdx < metas.length) {
    const finalMeta = metas[metaIdx];
    if (finalMeta.toolCalls.length === 0) {
      nodes.push(
        <StepDivider key={`step-${finalMeta.step}`} meta={finalMeta} />,
      );
    }
  }

  return nodes;
}

export function TurnCard({
  turn,
  stepMetas,
  provenanceEntries,
  annotations,
  onAnnotate,
  isOpen,
  onToggle,
}: {
  turn: TraceTurn;
  stepMetas?: StepMeta[];
  provenanceEntries?: SearchProvenanceEntry[];
  annotations?: Annotation[];
  onAnnotate?: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
  isOpen: boolean;
  onToggle: (turnId: string) => void;
}) {
  const units = groupEvents(turn.events);
  const metas = stepMetas ?? [];
  const turnUsage = metas.reduce(
    (sum, meta) => {
      const usage = usageOf(meta.usage);
      return {
        input: sum.input + usage.input,
        cache_read: sum.cache_read + usage.cache_read,
        cache_write: sum.cache_write + usage.cache_write,
        output: sum.output + usage.output,
        total: sum.total + usage.total,
      };
    },
    { input: 0, cache_read: 0, cache_write: 0, output: 0, total: 0 },
  );

  return (
    <details open={isOpen} style={{ ...card, flexShrink: 0 }}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          onToggle(turn.id);
        }}
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "12px",
          padding: "10px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            flex: 1,
            minWidth: 0,
          }}
        >
          <span
            style={{
              fontSize: "10px",
              color: "#52525b",
              textTransform: "uppercase",
              letterSpacing: "0.14em",
              flexShrink: 0,
            }}
          >
            {turn.index}
          </span>
          <TurnBadge kind={turn.kind} />
          <span
            style={{
              fontSize: "12px",
              fontWeight: 600,
              color: "#ededed",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {turn.title}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            flexShrink: 0,
          }}
        >
          {turn.toolCount > 0 && (
            <span style={muted}>{turn.toolCount} tools</span>
          )}
          <span style={muted}>{units.length} events</span>
          {metas.length > 0 && (
            <span
              style={{
                fontSize: "10px",
                padding: "2px 7px",
                borderRadius: "9999px",
                border: "1px solid #3f3f46",
                color: "#a3e635",
                backgroundColor: "rgba(163,230,53,0.08)",
                fontFamily: "ui-monospace,monospace",
                flexShrink: 0,
              }}
            >
              {formatUsageParts(turnUsage)}
            </span>
          )}
        </div>
      </summary>
      <div style={{ borderTop: "1px solid #1f1f22" }}>
        {interleaveUnitsWithSteps(
          units,
          metas,
          provenanceEntries ?? [],
          annotations ?? [],
          onAnnotate,
        )}
      </div>
    </details>
  );
}
