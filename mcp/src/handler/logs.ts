const QUICKWIT_HOST = process.env.QUICKWIT_HOST || "http://quickwit.sphinx:7280";

interface LogsSearchParams {
  query: string;
  max_hits?: number;
  start_timestamp?: number;
  end_timestamp?: number;
}

interface LogHit {
  timestamp: number;
  message: string;
  level?: string;
  service?: string;
  [key: string]: unknown;
}

interface LogsSearchResponse {
  hits: LogHit[];
  num_hits: number;
  elapsed_time_micros: number;
}

export async function searchLogs(params: LogsSearchParams): Promise<LogsSearchResponse> {
  const { query, max_hits = 100, start_timestamp, end_timestamp } = params;
  
  const url = new URL(`${QUICKWIT_HOST}/api/v1/logs/search`);
  url.searchParams.set("query", query);
  url.searchParams.set("max_hits", max_hits.toString());
  
  if (start_timestamp !== undefined) {
    url.searchParams.set("start_timestamp", start_timestamp.toString());
  }
  if (end_timestamp !== undefined) {
    url.searchParams.set("end_timestamp", end_timestamp.toString());
  }

  const response = await fetch(url.toString());
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Quickwit search failed: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json() as Promise<LogsSearchResponse>;
}
