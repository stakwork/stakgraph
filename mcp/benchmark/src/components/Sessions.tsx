import { useState, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { api } from "../api";
import type { ProductionRun } from "../types";

// ── helpers ──────────────────────────────────────────────────────────────────

function buildToolFrequency(
  sequence: string[],
): Array<{ toolName: string; count: number }> {
  const counts = new Map<string, number>();
  for (const t of sequence) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts.entries()]
    .map(([toolName, count]) => ({ toolName, count }))
    .sort((a, b) => b.count - a.count || a.toolName.localeCompare(b.toolName));
}

function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function previewStr(value: unknown): string {
  const s = stringify(value).replace(/\s+/g, " ").trim();
  return s.length > 160 ? s.slice(0, 160) + "\u2026" : s || "\u2014";
}

function getContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p: any) => p?.type === "text")
    .map((p: any) => String(p.text ?? ""))
    .join("\n")
    .trim();
}

function normalise(v: unknown): unknown {
  if (!v || typeof v !== "object" || Array.isArray(v)) return v;
  const r = v as Record<string, unknown>;
  if ((r.type === "text" || r.type === "json") && "value" in r) return r.value;
  return v;
}

type CallEntry = {
  id: string;
  toolName: string;
  input: unknown;
  index: number;
};
type ResultEntry = {
  id: string;
  toolName: string;
  output: unknown;
  index: number;
};
type IssueKind =
  | "empty"
  | "oversized"
  | "repeat"
  | "fallback"
  | "duplicate"
  | "noisy-overview";
type AnalyzedStep = {
  id: string;
  index: number;
  toolName: string;
  input: unknown;
  output: unknown;
  outputSize: number;
  outputLines: number;
  flags: IssueKind[];
};

function normaliseText(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function payloadMetrics(value: unknown): {
  text: string;
  size: number;
  lines: number;
} {
  const text = stringify(value);
  return { text, size: text.length, lines: text.split(/\r?\n/).length };
}

function hasNamedConcept(prompt: string): boolean {
  const p = prompt.trim();
  if (!p) return false;
  return /`[^`]+`|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Za-z_][A-Za-z0-9_]*\(|\b[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+\b|\b[a-z_]{3,}[a-z0-9_]*_[a-z0-9_]+\b/.test(
    p,
  );
}

function dedupeFlags(flags: IssueKind[]): IssueKind[] {
  return [...new Set(flags)];
}

function analyzeTrace(
  parsed: { calls: CallEntry[]; results: ResultEntry[] },
  prompt: string,
): { steps: AnalyzedStep[]; counts: Record<IssueKind, number> } {
  const resultById = new Map(parsed.results.map((r) => [r.id, r]));
  const steps: AnalyzedStep[] = parsed.calls.map((call) => {
    const result = resultById.get(call.id);
    const output = result?.output;
    const metrics = payloadMetrics(output);
    return {
      id: call.id,
      index: call.index,
      toolName: call.toolName,
      input: call.input,
      output,
      outputSize: metrics.size,
      outputLines: metrics.lines,
      flags: [],
    };
  });

  const promptHasConcept = hasNamedConcept(prompt);
  for (let i = 0; i < steps.length; i += 1) {
    const step = steps[i];
    const prev = steps[i - 1];
    const prev2 = steps[i - 2];
    const outputText = normaliseText(stringify(step.output));
    const inputText = normaliseText(previewStr(step.input));
    const flags: IssueKind[] = [];

    if (
      !outputText ||
      outputText === "[]" ||
      outputText === "{}" ||
      outputText === "null" ||
      outputText === "\u2014" ||
      /not found|no results|no matches|empty/i.test(outputText) ||
      step.outputSize < 16
    ) {
      flags.push("empty");
    }
    if (step.outputSize > 4000 || step.outputLines > 80) flags.push("oversized");
    if (
      (prev &&
        prev.toolName === step.toolName &&
        normaliseText(previewStr(prev.input)) === inputText) ||
      (prev &&
        prev2 &&
        prev.toolName === step.toolName &&
        prev2.toolName === step.toolName)
    ) {
      flags.push("repeat");
    }
    if (
      prev &&
      prev.toolName === step.toolName &&
      normaliseText(stringify(prev.output)) === outputText &&
      step.outputSize > 0
    ) {
      flags.push("duplicate");
    }
    if (
      (step.toolName === "bash" || step.toolName === "fulltext_search") &&
      prev &&
      (prev.toolName.startsWith("stakgraph") || prev.toolName === "repo_overview") &&
      !prev.flags.includes("empty")
    ) {
      flags.push("fallback");
    }
    if (step.toolName === "repo_overview" && promptHasConcept) {
      flags.push("noisy-overview");
    }
    step.flags = dedupeFlags(flags);
  }

  const counts: Record<IssueKind, number> = {
    empty: 0,
    oversized: 0,
    repeat: 0,
    fallback: 0,
    duplicate: 0,
    "noisy-overview": 0,
  };
  for (const step of steps) {
    for (const flag of step.flags) counts[flag] += 1;
  }
  return { steps, counts };
}

function parseTrace(trace: unknown): {
  userPrompt: string;
  answer: string;
  calls: CallEntry[];
  results: ResultEntry[];
} {
  if (!Array.isArray(trace))
    return { userPrompt: "", answer: "", calls: [], results: [] };

  let userPrompt = "";
  let answer = "";
  const calls: CallEntry[] = [];
  const results: ResultEntry[] = [];
  let ci = 0,
    ri = 0;

  for (const msg of trace) {
    if (!msg || typeof msg !== "object") continue;
    const role = String((msg as any).role ?? "");
    const content = (msg as any).content;

    if (!userPrompt && role === "user") {
      userPrompt =
        getContent(content) || (typeof content === "string" ? content : "");
    }
    if (role === "assistant") {
      const t =
        getContent(content) || (typeof content === "string" ? content : "");
      if (t) answer = t;
    }

    if (!Array.isArray(content)) continue;
    for (const item of content) {
      if (!item || typeof item !== "object") continue;
      const e = item as Record<string, unknown>;
      if (e.type === "tool-call") {
        ci++;
        calls.push({
          id: String(e.toolCallId ?? `c${ci}`),
          toolName: String(e.toolName ?? "?"),
          input: normalise(e.input),
          index: ci,
        });
      }
      if (e.type === "tool-result") {
        ri++;
        results.push({
          id: String(e.toolCallId ?? `r${ri}`),
          toolName: String(e.toolName ?? "?"),
          output: normalise(e.output ?? e.result ?? e.content),
          index: ri,
        });
      }
    }
  }

  return { userPrompt, answer, calls, results };
}

// ── shared styles ─────────────────────────────────────────────────────────────

const card: React.CSSProperties = {
  border: "1px solid #27272a",
  borderRadius: "8px",
  backgroundColor: "#111113",
  overflow: "hidden",
};
const summaryBase: React.CSSProperties = {
  listStyle: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "10px 14px",
  cursor: "pointer",
  userSelect: "none",
};
const labelStyle: React.CSSProperties = {
  fontSize: "12px",
  fontWeight: 600,
  color: "#ededed",
};
const muted: React.CSSProperties = {
  fontSize: "11px",
  color: "#71717a",
  margin: 0,
};
const preStyle: React.CSSProperties = {
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

function IssueBadge({ flag }: { flag: IssueKind }) {
  const colors: Record<IssueKind, { fg: string; bg: string; border: string }> =
    {
      empty: { fg: "#fca5a5", bg: "rgba(127,29,29,0.35)", border: "#7f1d1d" },
      oversized: { fg: "#fdba74", bg: "rgba(124,45,18,0.35)", border: "#7c2d12" },
      repeat: { fg: "#fcd34d", bg: "rgba(120,53,15,0.35)", border: "#78350f" },
      fallback: { fg: "#f9a8d4", bg: "rgba(131,24,67,0.35)", border: "#831843" },
      duplicate: { fg: "#c4b5fd", bg: "rgba(76,29,149,0.35)", border: "#4c1d95" },
      "noisy-overview": { fg: "#93c5fd", bg: "rgba(30,64,175,0.35)", border: "#1e40af" },
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

function EntryRow({
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
      <pre style={preStyle}>{stringify(payload)}</pre>
    </details>
  );
}

// ── component ─────────────────────────────────────────────────────────────────

export function Sessions() {
  const [runs, setRuns] = useState<ProductionRun[]>([]);
  const [selected, setSelected] = useState<ProductionRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [repoSearch, setRepoSearch] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const sessions = await api.sessions.list();
      setRuns(sessions);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const loadDetail = async (run: ProductionRun) => {
    try {
      const detail = await api.sessions.get(run.id);
      setSelected(detail);
    } catch {
      setSelected(run);
    }
  };

  const freq = selected ? buildToolFrequency(selected.tool_sequence) : [];
  const parsed = selected
    ? parseTrace(selected.trace)
    : { userPrompt: "", answer: "", calls: [], results: [] };
  const prompt =
    parsed.userPrompt || selected?.user_prompt_preview || "No prompt preview";
  const answer =
    parsed.answer || selected?.answer_preview || "No answer preview";
  const diagnostics = useMemo(
    () => analyzeTrace(parsed, prompt),
    [parsed, prompt],
  );
  const flagsById = useMemo(
    () => new Map(diagnostics.steps.map((s) => [s.id, s.flags])),
    [diagnostics.steps],
  );

  const repoOptions = useMemo(
    () => [...new Set(runs.map((r) => r.repo))].sort(),
    [runs],
  );

  const filteredRuns = useMemo(
    () =>
      runs.filter((r) => {
        if (repoSearch && !r.repo.toLowerCase().includes(repoSearch.toLowerCase()))
          return false;
        return true;
      }),
    [runs, repoSearch],
  );

  const inputStyle: React.CSSProperties = {
    fontSize: "12px",
    borderRadius: "6px",
    padding: "5px 8px",
    backgroundColor: "#18181b",
    color: "#ededed",
    border: "1px solid #27272a",
    outline: "none",
    width: "100%",
    boxSizing: "border-box",
  };
  const btnStyle: React.CSSProperties = {
    fontSize: "12px",
    padding: "6px",
    borderRadius: "6px",
    border: "1px solid #27272a",
    backgroundColor: "transparent",
    color: "#ededed",
    cursor: "pointer",
    width: "100%",
  };

  return (
    <div style={{ display: "flex", gap: "16px", flex: 1, minHeight: 0 }}>
      {/* LEFT: sidebar */}
      <div
        style={{
          width: "280px",
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          minHeight: 0,
        }}
      >
        <button onClick={load} style={btnStyle}>
          Refresh
        </button>

        {loading ? (
          <p style={muted}>{"Loading\u2026"}</p>
        ) : runs.length === 0 ? (
          <p style={muted}>No sessions yet.</p>
        ) : (
          <>
            <div
              style={{
                ...card,
                padding: "8px 10px",
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              <input
                placeholder={"Filter by repo\u2026"}
                value={repoSearch}
                onChange={(e) => setRepoSearch(e.target.value)}
                list="repo-options"
                style={inputStyle}
              />
              <datalist id="repo-options">
                {repoOptions.map((r) => (
                  <option key={r} value={r} />
                ))}
              </datalist>
              {repoSearch && (
                <p style={{ ...muted, textAlign: "center" }}>
                  {filteredRuns.length} / {runs.length} sessions
                </p>
              )}
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                overflowY: "auto",
                flex: 1,
                minHeight: 0,
              }}
            >
              {filteredRuns.length === 0 ? (
                <p style={muted}>No sessions match the filter.</p>
              ) : (
                filteredRuns.map((r) => (
                  <button
                    key={r.id}
                    onClick={() => loadDetail(r)}
                    style={{
                      textAlign: "left",
                      borderRadius: "6px",
                      border: `1px solid ${selected?.id === r.id ? "#52525b" : "#27272a"}`,
                      backgroundColor:
                        selected?.id === r.id ? "#27272a" : "#111113",
                      padding: "8px 10px",
                      cursor: "pointer",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "11px",
                        fontFamily: "monospace",
                        color: "#ededed",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        display: "block",
                      }}
                    >
                      {r.repo}
                    </span>
                    <p
                      style={{
                        ...muted,
                        marginTop: "3px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {new Date(r.timestamp).toLocaleString()} {"\u00b7"}{" "}
                      {r.tool_call_count} calls
                    </p>
                    {r.user_prompt_preview && (
                      <p
                        style={{
                          ...muted,
                          marginTop: "2px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {r.user_prompt_preview}
                      </p>
                    )}
                  </button>
                ))
              )}
            </div>
          </>
        )}
      </div>

      {/* RIGHT: detail */}
      <div style={{ flex: 1, minWidth: 0, minHeight: 0, overflowY: "auto" }}>
        {selected ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "10px",
              minHeight: 0,
            }}
          >
            {/* header */}
            <div
              style={{
                ...card,
                padding: "14px 16px",
              }}
            >
              <p
                style={{
                  fontSize: "14px",
                  fontWeight: 600,
                  color: "#ededed",
                  margin: 0,
                }}
              >
                {selected.repo}
              </p>
              <p style={{ ...muted, marginTop: "3px" }}>
                {new Date(selected.timestamp).toLocaleString()} {"\u00b7"}{" "}
                {selected.model} {"\u00b7"} {selected.token_usage.total} tokens
              </p>
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
            </div>

            {/* tool frequency */}
            <div style={{ ...card, padding: "12px 14px" }}>
              <p style={{ ...labelStyle, marginBottom: "8px" }}>
                Tool frequency
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {freq.map(({ toolName, count }) => (
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
                ))}
              </div>
            </div>

            {/* prompt + answer */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "10px",
                minHeight: 0,
              }}
            >
              <Section title="User prompt" badge="input" defaultOpen={false}>
                <pre style={preStyle}>{prompt}</pre>
              </Section>
              <Section
                title="Answer preview"
                badge="output"
                defaultOpen={false}
              >
                <pre style={preStyle}>{answer}</pre>
              </Section>
            </div>

            {/* tool calls + results */}
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
                defaultOpen={true}
              >
                <div
                  style={{
                    maxHeight: "28rem",
                    overflowY: "auto",
                    minHeight: 0,
                  }}
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
                defaultOpen={true}
              >
                <div
                  style={{
                    maxHeight: "28rem",
                    overflowY: "auto",
                    minHeight: 0,
                  }}
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

            {/* raw trace */}
            {selected.trace != null && (
              <Section title="Raw trace" badge="debug" defaultOpen={false}>
                <pre style={preStyle}>{stringify(selected.trace)}</pre>
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
    </div>
  );
}
