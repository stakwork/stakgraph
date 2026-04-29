import { useState } from "react";
import { type DayRow, formatK } from "../utils";
import { card } from "./ui";

export type HeatMetric = "sessions" | "tokens";

export function ActivityHeatmap({
  data,
  onDayClick,
}: {
  data: DayRow[];
  onDayClick: (day: string) => void;
}) {
  const [metric, setMetric] = useState<HeatMetric>("sessions");
  const [tooltip, setTooltip] = useState<{ day: string; x: number; y: number } | null>(null);

  if (data.length === 0) return null;

  const byDay = new Map<string, DayRow>();
  for (const row of data) byDay.set(row.day, row);

  const today = new Date();
  const currentYear = today.getFullYear();

  const jan1 = new Date(currentYear, 0, 1);
  const startDate = new Date(jan1);
  const dow = startDate.getDay();
  startDate.setDate(startDate.getDate() - ((dow + 6) % 7));

  const endDate = new Date(today);
  const endDow = endDate.getDay();
  if (endDow !== 0) endDate.setDate(endDate.getDate() + (7 - endDow));

  const weeks: Array<Array<string>> = [];
  const cursor = new Date(startDate);
  while (cursor <= endDate) {
    const week: string[] = [];
    for (let d = 0; d < 7; d++) {
      week.push(cursor.toISOString().slice(0, 10));
      cursor.setDate(cursor.getDate() + 1);
    }
    weeks.push(week);
  }

  const maxVal = Math.max(...data.map((r) => (metric === "sessions" ? r.sessions : r.tokens)), 1);
  const colors = ["#1c1c1f", "#312e81", "#4338ca", "#6366f1", "#a5b4fc"];

  function getLevel(val: number): number {
    if (!val) return 0;
    if (val <= maxVal * 0.25) return 1;
    if (val <= maxVal * 0.5) return 2;
    if (val <= maxVal * 0.75) return 3;
    return 4;
  }

  const monthLabels: Array<{ weekIndex: number; label: string }> = [];
  let lastMonth = -1;
  weeks.forEach((week, wi) => {
    const m = new Date(week[0] + "T00:00:00").getMonth();
    if (m !== lastMonth) {
      monthLabels.push({
        weekIndex: wi,
        label: new Date(week[0] + "T00:00:00").toLocaleDateString("en-US", { month: "short" }),
      });
      lastMonth = m;
    }
  });

  const CELL = 16;
  const GAP = 3;
  const STEP = CELL + GAP;
  const DAY_LABELS = ["Mon", "", "Wed", "", "Fri", "", ""];
  const todayStr = today.toISOString().slice(0, 10);
  const tooltipRow = tooltip ? byDay.get(tooltip.day) : null;

  return (
    <div style={{ ...card, padding: "14px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
        <p style={{ margin: 0, fontSize: "13px", fontWeight: 700, color: "#ededed" }}>Activity</p>
        <div style={{ display: "flex", gap: "4px" }}>
          {(["sessions", "tokens"] as HeatMetric[]).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              style={{
                fontSize: "11px",
                padding: "3px 10px",
                borderRadius: "6px",
                border: "1px solid #27272a",
                backgroundColor: metric === m ? "#27272a" : "transparent",
                color: metric === m ? "#ededed" : "#71717a",
                cursor: "pointer",
              }}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <div style={{ position: "relative", display: "inline-flex", flexDirection: "column" }}>
          <div style={{ position: "relative", height: "16px", marginLeft: "28px", marginBottom: "4px" }}>
            {monthLabels.map(({ weekIndex, label }) => (
              <span
                key={`${weekIndex}-${label}`}
                style={{
                  position: "absolute",
                  left: `${weekIndex * STEP}px`,
                  fontSize: "10px",
                  color: "#71717a",
                  whiteSpace: "nowrap",
                }}
              >
                {label}
              </span>
            ))}
          </div>

          <div style={{ display: "flex", gap: `${GAP}px` }}>
            <div style={{ display: "flex", flexDirection: "column", gap: `${GAP}px`, width: "24px" }}>
              {DAY_LABELS.map((label, i) => (
                <div
                  key={i}
                  style={{
                    height: `${CELL}px`,
                    fontSize: "9px",
                    color: "#71717a",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                    paddingRight: "4px",
                  }}
                >
                  {label}
                </div>
              ))}
            </div>

            {weeks.map((week, wi) => (
              <div key={wi} style={{ display: "flex", flexDirection: "column", gap: `${GAP}px` }}>
                {week.map((day, di) => {
                  const row = byDay.get(day);
                  const val = row ? (metric === "sessions" ? row.sessions : row.tokens) : 0;
                  return (
                    <div
                      key={di}
                      onClick={() => row && onDayClick(day)}
                      onMouseEnter={(e) => setTooltip({ day, x: e.clientX, y: e.clientY })}
                      onMouseLeave={() => setTooltip(null)}
                      style={{
                        width: CELL,
                        height: CELL,
                        borderRadius: "2px",
                        backgroundColor: colors[getLevel(val)],
                        cursor: row ? "pointer" : "default",
                        outline: day === todayStr ? "1.5px solid #6366f1" : undefined,
                        boxSizing: "border-box",
                        flexShrink: 0,
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "4px", marginTop: "10px", justifyContent: "flex-end" }}>
        <span style={{ fontSize: "10px", color: "#71717a", marginRight: "4px" }}>Less</span>
        {colors.map((c, i) => (
          <div
            key={i}
            style={{ width: 11, height: 11, borderRadius: 2, backgroundColor: c, border: "1px solid #27272a" }}
          />
        ))}
        <span style={{ fontSize: "10px", color: "#71717a", marginLeft: "4px" }}>More</span>
      </div>

      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x + 12,
            top: tooltip.y - 10,
            backgroundColor: "#18181b",
            border: "1px solid #27272a",
            borderRadius: "8px",
            fontSize: "12px",
            padding: "8px 12px",
            lineHeight: "1.8",
            pointerEvents: "none",
            zIndex: 1000,
          }}
        >
          <p style={{ margin: "0 0 4px 0", color: "#ededed", fontWeight: 600 }}>
            {new Date(tooltip.day + "T00:00:00").toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </p>
          {tooltipRow ? (
            <>
              <p style={{ margin: 0, color: "#6366f1" }}>Sessions: {tooltipRow.sessions}</p>
              <p style={{ margin: 0, color: "#22d3ee" }}>Tokens: {formatK(tooltipRow.tokens)}</p>
              <p style={{ margin: 0, color: "#a3e635" }}>Cost: ${tooltipRow.cost.toFixed(4)}</p>
            </>
          ) : (
            <p style={{ margin: 0, color: "#71717a" }}>No activity</p>
          )}
        </div>
      )}
    </div>
  );
}
