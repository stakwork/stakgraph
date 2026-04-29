import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { type DayRow, type RangeKey, formatK, formatXAxisTick } from "../utils";
import { card } from "./ui";

export function TokensChart({ data, range }: { data: DayRow[]; range: RangeKey }) {
  if (data.length === 0) return null;

  const tickInterval =
    data.length > 90
      ? Math.floor(data.length / 10)
      : data.length > 30
        ? Math.floor(data.length / 8)
        : ("preserveStartEnd" as const);

  return (
    <div style={{ ...card, padding: "14px" }}>
      <p style={{ margin: "0 0 14px 0", fontSize: "13px", fontWeight: 700, color: "#ededed" }}>
        Tokens per day
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="tokensGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="day"
            tick={{ fill: "#71717a", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval={tickInterval}
            tickFormatter={(v) => formatXAxisTick(v, range)}
          />
          <YAxis
            yAxisId="tokens"
            tick={{ fill: "#71717a", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatK}
            width={48}
          />
          <YAxis
            yAxisId="sessions"
            orientation="right"
            tick={{ fill: "#71717a", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0].payload as DayRow;
              return (
                <div
                  style={{
                    backgroundColor: "#18181b",
                    border: "1px solid #27272a",
                    borderRadius: "8px",
                    fontSize: "12px",
                    padding: "8px 12px",
                    lineHeight: "1.8",
                  }}
                >
                  <p style={{ margin: "0 0 6px 0", color: "#ededed", fontWeight: 600 }}>{label}</p>
                  <p style={{ margin: 0, color: "#6366f1" }}>Tokens: {formatK(row.tokens)}</p>
                  <p style={{ margin: 0, color: "#22d3ee" }}>Sessions: {row.sessions}</p>
                  <p style={{ margin: 0, color: "#a3e635" }}>Cost: ${row.cost.toFixed(4)}</p>
                </div>
              );
            }}
          />
          <Area
            yAxisId="tokens"
            type="monotone"
            dataKey="tokens"
            stroke="#6366f1"
            strokeWidth={2}
            fill="url(#tokensGradient)"
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            yAxisId="sessions"
            type="monotone"
            dataKey="sessions"
            stroke="#22d3ee"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            activeDot={{ r: 3 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: "16px", marginTop: "10px", justifyContent: "flex-end" }}>
        <span style={{ fontSize: "11px", color: "#71717a", display: "flex", alignItems: "center", gap: "5px" }}>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "#6366f1" }} />
          Tokens
        </span>
        <span style={{ fontSize: "11px", color: "#71717a", display: "flex", alignItems: "center", gap: "5px" }}>
          <span style={{ display: "inline-block", width: 10, height: 2, background: "#22d3ee" }} />
          Sessions
        </span>
      </div>
    </div>
  );
}
