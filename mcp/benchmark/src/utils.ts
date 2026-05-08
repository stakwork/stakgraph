import type { TokenUsage } from "./types";

export type RangeKey = "24h" | "7d" | "30d" | "3m" | "1y" | "all";

export type DayRow = {
  day: string;
  sessions: number;
  tokens: number;
  calls: number;
  cost: number;
};

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

export function formatDuration(durationMs: number): string {
  if (!durationMs) return "\u2014";
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

export function formatSourceLabel(source: string): string {
  return (source || "unknown").replace(/[_-]+/g, " ");
}

export function formatK(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}k`;
  return String(value);
}

export function formatXAxisTick(day: string, range: RangeKey): string {
  const date = new Date(day + "T00:00:00");
  if (range === "1y" || range === "all") {
    return date.toLocaleDateString("en-US", {
      month: "short",
      year: "2-digit",
    });
  }
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function getRangeStart(range: RangeKey): number | null {
  const now = Date.now();
  if (range === "24h") return now - 24 * 60 * 60 * 1000;
  if (range === "7d") return now - 7 * 24 * 60 * 60 * 1000;
  if (range === "30d") return now - 30 * 24 * 60 * 60 * 1000;
  if (range === "3m") return now - 90 * 24 * 60 * 60 * 1000;
  if (range === "1y") return now - 365 * 24 * 60 * 60 * 1000;
  return null;
}

export function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function previewStr(value: unknown): string {
  const s = stringify(value).replace(/\s+/g, " ").trim();
  return s.length > 160 ? s.slice(0, 160) + "\u2026" : s || "\u2014";
}

export function usageOf(usage?: Partial<TokenUsage>): TokenUsage {
  const cacheRead = usage?.cache_read || 0;
  const cacheWrite = usage?.cache_write || 0;
  const input =
    usage?.input ??
    Math.max((usage?.inputTokens || 0) - cacheRead - cacheWrite, 0);
  const output = usage?.output ?? usage?.outputTokens ?? 0;
  const total =
    usage?.total ??
    usage?.totalTokens ??
    input + cacheRead + cacheWrite + output;
  return {
    input,
    cache_read: cacheRead,
    cache_write: cacheWrite,
    output,
    total,
    inputTokens: usage?.inputTokens ?? input + cacheRead + cacheWrite,
    outputTokens: usage?.outputTokens ?? output,
    totalTokens: usage?.totalTokens ?? total,
  };
}

export function formatUsageParts(usage: Partial<TokenUsage>): string {
  const normalized = usageOf(usage);
  return [
    `base ${formatNumber(normalized.input)}`,
    `read ${formatNumber(normalized.cache_read)}`,
    `write ${formatNumber(normalized.cache_write)}`,
    `out ${formatNumber(normalized.output)}`,
  ].join(" / ");
}

export function buildToolFrequency(
  sequence: string[],
): Array<{ toolName: string; count: number }> {
  const counts = new Map<string, number>();
  for (const t of sequence) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts.entries()]
    .map(([toolName, count]) => ({ toolName, count }))
    .sort((a, b) => b.count - a.count || a.toolName.localeCompare(b.toolName));
}
