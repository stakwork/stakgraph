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
  
  const url = `${QUICKWIT_HOST}/api/v1/logs/search`;
  
  const body: Record<string, unknown> = {
    query,
    max_hits,
  };
  
  if (start_timestamp !== undefined) {
    body.start_timestamp = start_timestamp;
  }
  if (end_timestamp !== undefined) {
    body.end_timestamp = end_timestamp;
  }

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Quickwit search failed: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json() as Promise<LogsSearchResponse>;
}
