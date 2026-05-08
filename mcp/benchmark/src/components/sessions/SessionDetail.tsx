import { CopyableBlock, shortId } from "../ui";
import { MetaPill, SourceBadge } from "./SessionBadges";
import { TurnCard } from "./TurnCard";
import { EntryRow } from "./EntryRow";
import { AnnotationBadge, AnnotationForm } from "../Annotations";
import {
  card,
  muted,
  labelStyle,
  summaryBase,
  chunkSummaryMaxHeight,
} from "./styles";
import { formatNumber, formatDuration } from "../../utils";
import type { ParsedTrace, TraceAnalysis, IssueKind } from "../../trace/types";
import type { ProductionRun, TokenUsage } from "../../types";
import type { Annotation, AnnotationMarker } from "../Annotations";

function Section({
  title,
  badge,
  defaultOpen = true,
  children,
}: {
  title: string;
  badge?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <details open={defaultOpen} style={card}>
      <summary style={summaryBase}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={labelStyle}>{title}</span>
          {badge !== undefined && <span style={muted}>{badge}</span>}
        </div>
        <span style={{ fontSize: "11px", color: "#52525b" }}>{"\u25be"}</span>
      </summary>
      <div>{children}</div>
    </details>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        border: "1px solid #27272a",
        borderRadius: "10px",
        backgroundColor: "#111113",
        padding: "12px 14px",
        minWidth: 0,
      }}
    >
      <p
        style={{
          margin: 0,
          fontSize: "10px",
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "#71717a",
        }}
      >
        {label}
      </p>
      <p
        style={{
          margin: "8px 0 0 0",
          fontSize: "18px",
          fontWeight: 700,
          color: "#f5f5f5",
          lineHeight: 1.1,
          wordBreak: "break-word",
        }}
      >
        {value}
      </p>
    </div>
  );
}

interface SessionDetailProps {
  selected: ProductionRun | null;
  parsed: ParsedTrace;
  annotations: Annotation[];
  diagnostics: TraceAnalysis;
  flagsById: Map<string, IssueKind[]>;
  freq: Array<{ toolName: string; count: number }>;
  selectedUsage: TokenUsage | null;
  prompt: string;
  answer: string;
  openTurnId: string | null;
  handleTurnToggle: (turnId: string) => void;
  handleAnnotate: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
  showSessionAnnotationForm: boolean;
  setShowSessionAnnotationForm: (v: boolean) => void;
}

export function SessionDetail({
  selected,
  parsed,
  annotations,
  diagnostics,
  flagsById,
  freq,
  selectedUsage,
  prompt,
  answer,
  openTurnId,
  handleTurnToggle,
  handleAnnotate,
  showSessionAnnotationForm,
  setShowSessionAnnotationForm,
}: SessionDetailProps) {
  return (
    <div style={{ flex: 1, minWidth: 0, minHeight: 0, overflowY: "auto" }}>
      {selected ? (
        <div
          style={{
            display: "flex",
            overflow: "hidden",
            flexDirection: "column",
            gap: "10px",
            minHeight: 0,
          }}
        >
          {/* header */}
          <div style={{ ...card, padding: "16px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: "16px",
                flexWrap: "wrap",
                alignItems: "flex-start",
              }}
            >
              <div style={{ flex: "1 1 360px", minWidth: 0 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    flexWrap: "wrap",
                    marginBottom: "10px",
                  }}
                >
                  <SourceBadge source={selected.source} />
                  <MetaPill label="session" value={shortId(selected.id)} />
                </div>
                <p
                  style={{
                    fontSize: "18px",
                    fontWeight: 700,
                    color: "#ededed",
                    flexShrink: 0,
                    whiteSpace: "nowrap",
                    margin: 0,
                    lineHeight: 1.2,
                  }}
                >
                  {selected.repo || "Session without repo label"}
                </p>
                <p style={{ ...muted, marginTop: "6px", fontSize: "12px" }}>
                  Started {new Date(selected.timestamp).toLocaleString()}
                </p>
                <div
                  style={{
                    display: "flex",
                    gap: "8px",
                    flexWrap: "wrap",
                    marginTop: "12px",
                  }}
                >
                  <MetaPill
                    label="model"
                    value={selected.model || "unknown"}
                  />
                  <MetaPill
                    label="duration"
                    value={formatDuration(selected.duration_ms)}
                  />
                  <MetaPill
                    label="turns"
                    value={String(parsed.turns.length)}
                  />
                  <MetaPill
                    label="calls"
                    value={String(selected.tool_call_count)}
                  />
                </div>
              </div>
              <div
                style={{
                  flex: "0 1 420px",
                  minWidth: 0,
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                  gap: "10px",
                  width: "100%",
                  maxWidth: "420px",
                }}
              >
                <StatTile
                  label="Total Tokens"
                  value={formatNumber(selectedUsage?.total || 0)}
                />
                <StatTile
                  label="Input Base"
                  value={formatNumber(selectedUsage?.input || 0)}
                />
                <StatTile
                  label="Cache Read"
                  value={formatNumber(selectedUsage?.cache_read || 0)}
                />
                <StatTile
                  label="Cache Write"
                  value={formatNumber(selectedUsage?.cache_write || 0)}
                />
                <StatTile
                  label="Output"
                  value={formatNumber(selectedUsage?.output || 0)}
                />
                {selected.cost_usd != null && (
                  <StatTile
                    label="Cost"
                    value={`$${selected.cost_usd.toFixed(4)}`}
                  />
                )}
              </div>
            </div>
            {(diagnostics.counts.oversized > 0 ||
              diagnostics.counts.fallback > 0 ||
              diagnostics.counts.repeat > 0 ||
              diagnostics.counts.empty > 0) && (
              <div
                style={{
                  display: "flex",
                  gap: "8px",
                  flexWrap: "wrap",
                  marginTop: "8px",
                }}
              >
                {(
                  [
                    {
                      label: "flagged",
                      value: diagnostics.steps.filter(
                        (s) => s.flags.length > 0,
                      ).length,
                      color: "#fca5a5",
                    },
                    {
                      label: "oversized",
                      value: diagnostics.counts.oversized,
                      color: "#fdba74",
                    },
                    {
                      label: "fallbacks",
                      value: diagnostics.counts.fallback,
                      color: "#f9a8d4",
                    },
                    {
                      label: "repeats",
                      value: diagnostics.counts.repeat,
                      color: "#fcd34d",
                    },
                  ] as const
                )
                  .filter((item) => item.value > 0)
                  .map((item) => (
                    <span
                      key={item.label}
                      style={{
                        fontSize: "11px",
                        color: item.color,
                        backgroundColor: "#18181b",
                        border: "1px solid #27272a",
                        borderRadius: "6px",
                        padding: "2px 8px",
                      }}
                    >
                      {item.value} {item.label}
                    </span>
                  ))}
              </div>
            )}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                flexWrap: "wrap",
                marginTop: "10px",
                paddingTop: "10px",
                borderTop: "1px solid #1f1f22",
              }}
            >
              {annotations
                .filter((a) => a.target === "session")
                .map((a) => (
                  <AnnotationBadge
                    key={a.ts}
                    marker={a.marker}
                    note={a.note}
                  />
                ))}
              {showSessionAnnotationForm ? (
                <AnnotationForm
                  onSubmit={(marker, note) => {
                    void handleAnnotate(marker, note);
                    setShowSessionAnnotationForm(false);
                  }}
                  onCancel={() => setShowSessionAnnotationForm(false)}
                />
              ) : (
                <button
                  onClick={() => setShowSessionAnnotationForm(true)}
                  style={{
                    fontSize: "11px",
                    padding: "3px 10px",
                    borderRadius: "9999px",
                    border: "1px solid #3f3f46",
                    backgroundColor: "transparent",
                    color: "#71717a",
                    cursor: "pointer",
                  }}
                >
                  + annotate session
                </button>
              )}
            </div>
          </div>

          <Section
            title="Trace timeline"
            badge={`${parsed.turns.length} turns / ${parsed.events.length} events`}
            defaultOpen={true}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                padding: "10px",
                maxHeight: "min(62vh, 52rem)",
                overflowY: "auto",
                minHeight: 0,
              }}
            >
              {parsed.turns.length === 0 ? (
                <p style={{ ...muted, padding: "10px 4px" }}>
                  No trace events available.
                </p>
              ) : (
                parsed.turns.map((turn) => (
                  <TurnCard
                    key={turn.id}
                    turn={turn}
                    stepMetas={selected?.step_meta?.filter(
                      (m) => m.turn === turn.index,
                    )}
                    provenanceEntries={selected?.search_provenance}
                    annotations={annotations}
                    onAnnotate={handleAnnotate}
                    isOpen={openTurnId === turn.id}
                    onToggle={handleTurnToggle}
                  />
                ))
              )}
            </div>
          </Section>

          <Section
            title="Session highlights"
            badge="summary"
            defaultOpen={false}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "10px",
                minHeight: 0,
              }}
            >
              <div>
                <p style={{ ...labelStyle, margin: "10px 14px 0 14px" }}>
                  Prompt
                </p>
                <CopyableBlock
                  value={prompt}
                  maxHeight={chunkSummaryMaxHeight}
                />
              </div>
              <div>
                <p style={{ ...labelStyle, margin: "10px 14px 0 14px" }}>
                  Final answer
                </p>
                <CopyableBlock
                  value={answer}
                  maxHeight={chunkSummaryMaxHeight}
                />
              </div>
            </div>
          </Section>

          <Section
            title="Tool analysis"
            badge={String(freq.length)}
            defaultOpen={false}
          >
            <div style={{ padding: "12px 14px" }}>
              <p style={{ ...labelStyle, marginBottom: "8px" }}>
                Tool frequency
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {freq.length === 0 ? (
                  <p style={muted}>No tool activity in this session.</p>
                ) : (
                  freq.map(({ toolName, count }) => (
                    <span
                      key={toolName}
                      title={`${count} call${count === 1 ? "" : "s"}`}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "6px",
                        fontSize: "11px",
                        padding: "3px 10px",
                        borderRadius: "9999px",
                        border: "1px solid #27272a",
                        backgroundColor: "#18181b",
                        color: "#ededed",
                      }}
                    >
                      {toolName}
                      <span
                        style={{
                          fontSize: "10px",
                          fontWeight: 700,
                          backgroundColor: "#3f3f46",
                          borderRadius: "9999px",
                          padding: "1px 6px",
                          color: "#ededed",
                        }}
                      >
                        {count}
                      </span>
                    </span>
                  ))
                )}
              </div>
            </div>
          </Section>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "10px",
            }}
          >
            <Section
              title="Tool calls"
              badge={String(parsed.calls.length)}
              defaultOpen={false}
            >
              <div
                style={{ maxHeight: "28rem", overflowY: "auto", minHeight: 0 }}
              >
                {parsed.calls.length === 0 ? (
                  <p style={{ ...muted, padding: "10px 14px" }}>
                    No tool calls in trace.
                  </p>
                ) : (
                  parsed.calls.map((c) => (
                    <EntryRow
                      key={c.id}
                      kind="call"
                      index={c.index}
                      toolName={c.toolName}
                      payload={c.input}
                      flags={flagsById.get(c.id)}
                    />
                  ))
                )}
              </div>
            </Section>
            <Section
              title="Tool results"
              badge={String(parsed.results.length)}
              defaultOpen={false}
            >
              <div
                style={{ maxHeight: "28rem", overflowY: "auto", minHeight: 0 }}
              >
                {parsed.results.length === 0 ? (
                  <p style={{ ...muted, padding: "10px 14px" }}>
                    No tool results in trace.
                  </p>
                ) : (
                  parsed.results.map((r) => (
                    <EntryRow
                      key={r.id}
                      kind="result"
                      index={r.index}
                      toolName={r.toolName}
                      payload={r.output}
                      flags={flagsById.get(r.id)}
                    />
                  ))
                )}
              </div>
            </Section>
          </div>

          {selected.trace != null && (
            <Section title="Raw trace" badge="debug" defaultOpen={false}>
              <CopyableBlock value={selected.trace} maxHeight="28rem" />
            </Section>
          )}
        </div>
      ) : (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "200px",
          }}
        >
          <p style={muted}>Select a session to view details</p>
        </div>
      )}
    </div>
  );
}
