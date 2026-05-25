import { useNavigate, useLocation } from "react-router-dom";
import { shortId } from "../ui";
import { SourceBadge } from "./SessionBadges";
import { card, muted } from "./styles";
import { formatNumber, formatSourceLabel } from "../../utils";
import type { ProductionRun } from "../../types";

interface SessionSidebarProps {
  loading: boolean;
  runs: ProductionRun[];
  filteredRuns: ProductionRun[];
  selected: ProductionRun | null;
  quickSearch: string;
  repoSearch: string;
  sourceFilter: string;
  rangeFilter: "24h" | "7d" | "30d" | "all";
  dayFilter: string;
  repoOptions: string[];
  sourceOptions: string[];
  load: () => void;
  loadDetail: (run: ProductionRun) => void;
  setQuickSearch: (v: string) => void;
  setRepoSearch: (v: string) => void;
  setSourceFilter: (v: string) => void;
  setRangeFilter: (v: "24h" | "7d" | "30d" | "all") => void;
  setDayFilter: (v: string) => void;
  clearFilters: () => void;
}

export function SessionSidebar({
  loading,
  runs,
  filteredRuns,
  selected,
  quickSearch,
  repoSearch,
  sourceFilter,
  rangeFilter,
  dayFilter,
  repoOptions,
  sourceOptions,
  load,
  loadDetail,
  setQuickSearch,
  setRepoSearch,
  setSourceFilter,
  setRangeFilter,
  setDayFilter,
  clearFilters,
}: SessionSidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const fromAnalytics = (location.state as any)?.from === "analytics";

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
  const selectStyle: React.CSSProperties = {
    ...inputStyle,
    padding: "6px 8px",
  };

  const chipStyle: React.CSSProperties = {
    fontSize: "11px",
    padding: "2px 8px",
    borderRadius: "10px",
    backgroundColor: "#1c1007",
    color: "#fbbf24",
    border: "1px solid #78350f",
    whiteSpace: "nowrap" as const,
  };

  return (
    <div
      style={{
        width: "320px",
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        minHeight: 0,
      }}
    >
      {fromAnalytics && (
        <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
          <button
            onClick={() => navigate("/analytics")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "5px",
              fontSize: "11px",
              padding: "5px 8px",
              borderRadius: "6px",
              border: "1px solid #78350f",
              backgroundColor: "#1c1007",
              color: "#fbbf24",
              cursor: "pointer",
              width: "100%",
              textAlign: "left",
            }}
          >
            ← Back to Analytics
          </button>
          {(sourceFilter !== "all" || repoSearch || dayFilter) && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
              {sourceFilter !== "all" && (
                <span style={chipStyle}>source: {formatSourceLabel(sourceFilter)}</span>
              )}
              {repoSearch && (
                <span style={chipStyle}>repo: {repoSearch}</span>
              )}
              {dayFilter && (
                <span style={chipStyle}>day: {dayFilter}</span>
              )}
            </div>
          )}
        </div>
      )}
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
              placeholder={"Search id, repo, model, prompt\u2026"}
              value={quickSearch}
              onChange={(e) => setQuickSearch(e.target.value)}
              style={inputStyle}
            />
            <input
              placeholder={"Filter by repo\u2026"}
              value={repoSearch}
              onChange={(e) => setRepoSearch(e.target.value)}
              list="repo-options"
              style={inputStyle}
            />
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              style={selectStyle}
            >
              <option value="all">All sources</option>
              {sourceOptions.map((source) => (
                <option key={source} value={source}>
                  {formatSourceLabel(source)}
                </option>
              ))}
            </select>
            <select
              value={rangeFilter}
              onChange={(e) => {
                setRangeFilter(
                  e.target.value as "24h" | "7d" | "30d" | "all",
                );
                setDayFilter("");
              }}
              style={selectStyle}
            >
              <option value="all">All time</option>
              <option value="24h">Last 24h</option>
              <option value="7d">Last 7d</option>
              <option value="30d">Last 30d</option>
            </select>
            <datalist id="repo-options">
              {repoOptions.map((r) => (
                <option key={r} value={r} />
              ))}
            </datalist>
            {(quickSearch ||
              repoSearch ||
              sourceFilter !== "all" ||
              rangeFilter !== "all" ||
              dayFilter) && (
              <>
                <p style={{ ...muted, textAlign: "center" }}>
                  {dayFilter ? `Day ${dayFilter} · ` : ""}
                  {filteredRuns.length} / {runs.length} sessions
                </p>
                <button onClick={clearFilters} style={btnStyle}>
                  Clear filters
                </button>
              </>
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
              filteredRuns.map((run) => (
                <button
                  key={run.id}
                  onClick={() => void loadDetail(run)}
                  style={{
                    textAlign: "left",
                    borderRadius: "6px",
                    border: `1px solid ${
                      selected?.id === run.id ? "#52525b" : "#27272a"
                    }`,
                    backgroundColor:
                      selected?.id === run.id ? "#27272a" : "#111113",
                    padding: "8px 10px",
                    cursor: "pointer",
                    width: "100%",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: "8px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "6px",
                        minWidth: 0,
                      }}
                    >
                      <SourceBadge source={run.source} />
                      {run.status === "error" && (
                        <span
                          title={run.error_message || "Error"}
                          style={{
                            display: "inline-block",
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            backgroundColor: "#ef4444",
                            flexShrink: 0,
                          }}
                        />
                      )}
                    </div>
                    <span
                      style={{
                        fontSize: "10px",
                        color: "#71717a",
                        fontFamily: "ui-monospace, monospace",
                        flexShrink: 0,
                      }}
                      title={run.id}
                    >
                      {shortId(run.id)}
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: "11px",
                      fontFamily: "ui-monospace, monospace",
                      color: "#ededed",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      display: "block",
                      marginTop: "8px",
                    }}
                  >
                    {run.repo || "No repo captured"}
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
                    {new Date(run.timestamp).toLocaleString()} {"\u00b7"}{" "}
                    {run.tool_call_count} calls {"\u00b7"}{" "}
                    {formatNumber(run.token_usage.total)} tokens
                  </p>
                  {run.user_prompt_preview && (
                    <p
                      style={{
                        ...muted,
                        marginTop: "2px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {run.user_prompt_preview}
                    </p>
                  )}
                </button>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
