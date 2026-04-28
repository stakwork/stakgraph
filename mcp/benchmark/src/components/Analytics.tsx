import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "../api";
import type { ProductionRun } from "../types";

type RangeKey = "24h" | "7d" | "30d" | "all";

type AggregateRow = {
  key: string;
  label: string;
  sessions: number;
  totalTokens: number;
  avgTokens: number;
  toolCalls: number;
  avgDurationMs: number;
  totalCost: number;
  lastSeen: string | null;
};

type ModelRow = {
  model: string;
  provider: string;
  sessions: number;
  totalTokens: number;
  totalCacheRead: number;
  totalCacheWrite: number;
  totalCost: number;
};

const card: React.CSSProperties = {
  border: "1px solid #27272a",
  borderRadius: "12px",
  backgroundColor: "#111113",
};

const muted: React.CSSProperties = {
  color: "#71717a",
  fontSize: "12px",
};

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function formatDuration(durationMs: number): string {
  if (!durationMs) return "-";
  if (durationMs < 1000) return `${durationMs} ms`;

  const totalSeconds = Math.round(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  if (minutes === 0) return `${totalSeconds}s`;
  if (minutes < 60) return `${minutes}m ${seconds}s`;

  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return `${hours}h ${remMinutes}m`;
}

function formatSourceLabel(source: string): string {
  return (source || "unknown").replace(/[_-]+/g, " ");
}

function getRangeStart(range: RangeKey): number | null {
  const now = Date.now();
  if (range === "24h") return now - 24 * 60 * 60 * 1000;
  if (range === "7d") return now - 7 * 24 * 60 * 60 * 1000;
  if (range === "30d") return now - 30 * 24 * 60 * 60 * 1000;
  return null;
}

function aggregateBy(
  runs: ProductionRun[],
  getKey: (run: ProductionRun) => string,
  getLabel?: (run: ProductionRun) => string,
): AggregateRow[] {
  const rows = new Map<string, AggregateRow>();

  for (const run of runs) {
    const key = getKey(run) || "unknown";
    const label = getLabel ? getLabel(run) : key;
    const existing = rows.get(key) ?? {
      key,
      label,
      sessions: 0,
      totalTokens: 0,
      avgTokens: 0,
      toolCalls: 0,
      avgDurationMs: 0,
      totalCost: 0,
      lastSeen: null,
    };

    existing.sessions += 1;
    existing.totalTokens += run.token_usage.total || 0;
    existing.toolCalls += run.tool_call_count || 0;
    existing.avgDurationMs += run.duration_ms || 0;
    existing.totalCost += run.cost_usd || 0;
    if (!existing.lastSeen || new Date(run.timestamp).getTime() > new Date(existing.lastSeen).getTime()) {
      existing.lastSeen = run.timestamp;
    }

    rows.set(key, existing);
  }

  return [...rows.values()]
    .map((row) => ({
      ...row,
      avgTokens: row.sessions ? row.totalTokens / row.sessions : 0,
      avgDurationMs: row.sessions ? row.avgDurationMs / row.sessions : 0,
    }))
    .sort((a, b) => b.sessions - a.sessions || b.totalTokens - a.totalTokens);
}

function StatTile({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div style={{ ...card, padding: "14px" }}>
      <p style={{ ...muted, margin: 0 }}>{label}</p>
      <p style={{ margin: "6px 0 0 0", fontSize: "22px", fontWeight: 700, color: "#ededed" }}>
        {value}
      </p>
      {detail && <p style={{ ...muted, margin: "6px 0 0 0" }}>{detail}</p>}
    </div>
  );
}

function FilterSelect({
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

function TableCard({
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
        <p style={{ margin: 0, fontSize: "13px", fontWeight: 700, color: "#ededed" }}>{title}</p>
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

function tdStyle(emphasis = false): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderBottom: "1px solid #1a1a1d",
    fontSize: "12px",
    color: emphasis ? "#ededed" : "#d4d4d8",
    fontWeight: emphasis ? 600 : 400,
    whiteSpace: "nowrap",
  };
}

export function Analytics() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState<ProductionRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState<RangeKey>("7d");
  const [source, setSource] = useState("all");
  const [repo, setRepo] = useState("all");

  const openSessions = useCallback(
    (overrides?: { source?: string; repo?: string; day?: string }) => {
      const params = new URLSearchParams();
      const targetRange = overrides?.day ? "all" : range;
      params.set("range", targetRange);

      const targetSource = overrides?.source || source;
      const targetRepo = overrides?.repo || repo;

      if (targetSource !== "all") params.set("source", targetSource);
      if (targetRepo !== "all") params.set("repo", targetRepo);
      if (overrides?.day) params.set("day", overrides.day);

      navigate({ pathname: "/", search: `?${params.toString()}` });
    },
    [navigate, range, repo, source],
  );

  const linkButtonStyle: React.CSSProperties = {
    padding: 0,
    border: 0,
    background: "transparent",
    color: "inherit",
    cursor: "pointer",
    font: "inherit",
    textAlign: "left",
  };

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setRuns(await api.sessions.list());
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const sourceOptions = useMemo(
    () => [
      { value: "all", label: "All sources" },
      ...[...new Set(runs.map((run) => run.source || "unknown"))]
        .sort()
        .map((value) => ({ value, label: formatSourceLabel(value) })),
    ],
    [runs],
  );

  const repoOptions = useMemo(
    () => [
      { value: "all", label: "All repos" },
      ...[...new Set(runs.map((run) => run.repo || "unknown"))]
        .sort()
        .map((value) => ({ value, label: value })),
    ],
    [runs],
  );

  const filteredRuns = useMemo(() => {
    const rangeStart = getRangeStart(range);
    return runs.filter((run) => {
      if (source !== "all" && (run.source || "unknown") !== source) return false;
      if (repo !== "all" && (run.repo || "unknown") !== repo) return false;
      if (rangeStart && new Date(run.timestamp).getTime() < rangeStart) return false;
      return true;
    });
  }, [repo, range, runs, source]);

  const totals = useMemo(() => {
    const totalSessions = filteredRuns.length;
    const totalTokens = filteredRuns.reduce((sum, run) => sum + (run.token_usage.total || 0), 0);
    const totalInput = filteredRuns.reduce((sum, run) => sum + (run.token_usage.input || 0), 0);
    const totalCacheRead = filteredRuns.reduce((sum, run) => sum + (run.token_usage.cache_read || 0), 0);
    const totalCacheWrite = filteredRuns.reduce((sum, run) => sum + (run.token_usage.cache_write || 0), 0);
    const totalOutput = filteredRuns.reduce((sum, run) => sum + (run.token_usage.output || 0), 0);
    const totalCalls = filteredRuns.reduce((sum, run) => sum + (run.tool_call_count || 0), 0);
    const totalDuration = filteredRuns.reduce((sum, run) => sum + (run.duration_ms || 0), 0);
    const totalCost = filteredRuns.reduce((sum, run) => sum + (run.cost_usd || 0), 0);
    return {
      totalSessions,
      totalTokens,
      totalInput,
      totalCacheRead,
      totalCacheWrite,
      totalOutput,
      totalCalls,
      totalCost,
      avgDuration: totalSessions ? totalDuration / totalSessions : 0,
    };
  }, [filteredRuns]);

  const sourceRows = useMemo(
    () => aggregateBy(filteredRuns, (run) => run.source || "unknown", (run) => formatSourceLabel(run.source || "unknown")),
    [filteredRuns],
  );

  const repoRows = useMemo(
    () => aggregateBy(filteredRuns, (run) => run.repo || "unknown"),
    [filteredRuns],
  );

  const modelRows = useMemo((): ModelRow[] => {
    const grouped = new Map<string, ModelRow>();
    for (const run of filteredRuns) {
      const key = `${run.provider || "unknown"}::${run.model || "unknown"}`;
      const existing = grouped.get(key) ?? {
        model: run.model || "unknown",
        provider: run.provider || "unknown",
        sessions: 0,
        totalTokens: 0,
        totalCacheRead: 0,
        totalCacheWrite: 0,
        totalCost: 0,
      };
      existing.sessions += 1;
      existing.totalTokens += run.token_usage.total || 0;
      existing.totalCacheRead += run.token_usage.cache_read || 0;
      existing.totalCacheWrite += run.token_usage.cache_write || 0;
      existing.totalCost += run.cost_usd || 0;
      grouped.set(key, existing);
    }
    return [...grouped.values()].sort((a, b) => b.totalCost - a.totalCost);
  }, [filteredRuns]);

  const dailyRows = useMemo(() => {
    const grouped = new Map<string, { day: string; sessions: number; tokens: number; calls: number }>();
    for (const run of filteredRuns) {
      const day = new Date(run.timestamp).toISOString().slice(0, 10);
      const existing = grouped.get(day) ?? { day, sessions: 0, tokens: 0, calls: 0 };
      existing.sessions += 1;
      existing.tokens += run.token_usage.total || 0;
      existing.calls += run.tool_call_count || 0;
      grouped.set(day, existing);
    }
    return [...grouped.values()].sort((a, b) => b.day.localeCompare(a.day));
  }, [filteredRuns]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "14px", minHeight: 0 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px", flexWrap: "wrap" }}>
        <div>
          <p style={{ margin: 0, fontSize: "18px", fontWeight: 700, color: "#ededed" }}>Session analytics</p>
          <p style={{ ...muted, margin: "4px 0 0 0" }}>Macro view across all saved session metadata.</p>
        </div>
        <button
          onClick={load}
          style={{
            fontSize: "12px",
            padding: "8px 12px",
            borderRadius: "8px",
            border: "1px solid #27272a",
            backgroundColor: "transparent",
            color: "#ededed",
            cursor: "pointer",
          }}
        >
          Refresh
        </button>
      </div>

      <div style={{ ...card, padding: "12px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
        <FilterSelect
          value={range}
          onChange={(value) => setRange(value as RangeKey)}
          options={[
            { value: "24h", label: "Last 24h" },
            { value: "7d", label: "Last 7d" },
            { value: "30d", label: "Last 30d" },
            { value: "all", label: "All time" },
          ]}
        />
        <FilterSelect value={source} onChange={setSource} options={sourceOptions} />
        <FilterSelect value={repo} onChange={setRepo} options={repoOptions} />
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center" }}>
          <span style={muted}>{filteredRuns.length} / {runs.length} sessions</span>
        </div>
      </div>

      {loading ? (
        <p style={muted}>Loading analytics...</p>
      ) : filteredRuns.length === 0 ? (
        <div style={{ ...card, padding: "18px" }}>
          <p style={muted}>No sessions match the current filters.</p>
        </div>
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "10px",
            }}
          >
            <StatTile label="Sessions" value={formatNumber(totals.totalSessions)} />
            <StatTile label="Total tokens" value={formatNumber(totals.totalTokens)} detail={`${formatNumber(totals.totalInput)} in / ${formatNumber(totals.totalOutput)} out${totals.totalCacheRead ? ` / ${formatNumber(totals.totalCacheRead)} cache read` : ""}${totals.totalCacheWrite ? ` / ${formatNumber(totals.totalCacheWrite)} cache write` : ""}`} />
            <StatTile label="Total cost" value={`$${totals.totalCost.toFixed(4)}`} />
            <StatTile label="Tool calls" value={formatNumber(totals.totalCalls)} />
            <StatTile label="Avg duration" value={formatDuration(Math.round(totals.avgDuration))} />
          </div>

          <TableCard
            title="By source"
            badge={`${sourceRows.length} groups`}
            columns={["Source", "Sessions", "Tokens", "Avg tokens", "Cost", "Calls", "Avg duration", "Last seen"]}
            rows={sourceRows.map((row) => (
              <tr key={row.key}>
                <td style={tdStyle(true)}>
                  <button style={linkButtonStyle} onClick={() => openSessions({ source: row.key })}>
                    {row.label}
                  </button>
                </td>
                <td style={tdStyle()}>{formatNumber(row.sessions)}</td>
                <td style={tdStyle()}>{formatNumber(row.totalTokens)}</td>
                <td style={tdStyle()}>{formatNumber(Math.round(row.avgTokens))}</td>
                <td style={tdStyle()}>${row.totalCost.toFixed(4)}</td>
                <td style={tdStyle()}>{formatNumber(row.toolCalls)}</td>
                <td style={tdStyle()}>{formatDuration(Math.round(row.avgDurationMs))}</td>
                <td style={tdStyle()}>{row.lastSeen ? new Date(row.lastSeen).toLocaleString() : "-"}</td>
              </tr>
            ))}
          />

          <TableCard
            title="By model"
            badge={`${modelRows.length} models`}
            columns={["Model", "Provider", "Sessions", "Tokens", "Cache read", "Cache write", "Cost"]}
            rows={modelRows.map((row) => (
              <tr key={`${row.provider}::${row.model}`}>
                <td style={tdStyle(true)}>{row.model}</td>
                <td style={tdStyle()}>{row.provider}</td>
                <td style={tdStyle()}>{formatNumber(row.sessions)}</td>
                <td style={tdStyle()}>{formatNumber(row.totalTokens)}</td>
                <td style={tdStyle()}>{formatNumber(row.totalCacheRead)}</td>
                <td style={tdStyle()}>{formatNumber(row.totalCacheWrite)}</td>
                <td style={tdStyle()}>${row.totalCost.toFixed(4)}</td>
              </tr>
            ))}
          />

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 0.8fr)",
              gap: "10px",
            }}
          >
            <TableCard
              title="By repo"
              badge={`${repoRows.length} repos`}
              columns={["Repo", "Sessions", "Tokens", "Calls", "Last seen"]}
              rows={repoRows.slice(0, 20).map((row) => (
                <tr key={row.key}>
                  <td style={tdStyle(true)}>
                    <button style={linkButtonStyle} onClick={() => openSessions({ repo: row.key })}>
                      {row.label}
                    </button>
                  </td>
                  <td style={tdStyle()}>{formatNumber(row.sessions)}</td>
                  <td style={tdStyle()}>{formatNumber(row.totalTokens)}</td>
                  <td style={tdStyle()}>{formatNumber(row.toolCalls)}</td>
                  <td style={tdStyle()}>{row.lastSeen ? new Date(row.lastSeen).toLocaleString() : "-"}</td>
                </tr>
              ))}
            />

            <TableCard
              title="Daily activity"
              badge={`${dailyRows.length} days`}
              columns={["Day", "Sessions", "Tokens", "Calls"]}
              rows={dailyRows.map((row) => (
                <tr key={row.day}>
                  <td style={tdStyle(true)}>
                    <button style={linkButtonStyle} onClick={() => openSessions({ day: row.day })}>
                      {row.day}
                    </button>
                  </td>
                  <td style={tdStyle()}>{formatNumber(row.sessions)}</td>
                  <td style={tdStyle()}>{formatNumber(row.tokens)}</td>
                  <td style={tdStyle()}>{formatNumber(row.calls)}</td>
                </tr>
              ))}
            />
          </div>
        </>
      )}
    </div>
  );
}