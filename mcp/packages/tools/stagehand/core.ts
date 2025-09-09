import { Stagehand } from "@browserbasehq/stagehand";
import { getProvider } from "./providers.js";

let STATE: {
  [sessionId: string]: {
    stagehand: Stagehand;
    last_used: Date;
    logs: ConsoleLog[];
    networkEntries: NetworkEntry[];
  };
} = {};

export interface ConsoleLog {
  timestamp: string;
  type: string;
  text: string;
  location: {
    url: string;
    lineNumber: number;
    columnNumber: number;
  };
}

export interface NetworkEntry {
  id: string;
  timestamp: string;
  type: 'request' | 'response';
  method: string;
  url: string;
  status?: number;
  duration?: number;
  resourceType: string;
  size?: number;
}

const MAX_LOGS = parseInt(process.env.STAGEHAND_MAX_CONSOLE_LOGS || "1000");
const MAX_NETWORK_ENTRIES = parseInt(process.env.STAGEHAND_MAX_NETWORK_ENTRIES || "500");
const MAX_SESSIONS = 25; // LRU limit for stagehand instances

export async function getOrCreateStagehand(sessionIdMaybe?: string) {
  const sessionId = sessionIdMaybe || "default-session-id";

  // console.log("getOrCreateStagehand SESSION ID", sessionId);
  if (STATE[sessionId]) {
    // Update last_used timestamp for LRU tracking
    STATE[sessionId].last_used = new Date();
    return STATE[sessionId].stagehand;
  }

  let provider = getProvider();
  console.log("initializing stagehand!", provider.model);
  const sh = new Stagehand({
    env: "LOCAL",
    domSettleTimeoutMs: 60000,
    localBrowserLaunchOptions: {
      headless: true,
      viewport: { width: 1024, height: 768 },
    },
    enableCaching: true,
    modelName: provider.model,
    modelClientOptions: {
      apiKey: process.env[provider.api_key_env_var_name],
    },
  });
  await sh.init();

  // Initialize session state
  STATE[sessionId] = {
    stagehand: sh,
    last_used: new Date(),
    logs: [],
    networkEntries: [],
  };

  // Set up console log listener
  sh.page.on("console", (msg) => {
    addConsoleLog(sessionId, {
      timestamp: new Date().toISOString(),
      type: msg.type(),
      text: msg.text(),
      location: msg.location(),
    });
  });

  // Set up network monitoring listeners
  const requestStartTimes = new Map<string, number>();

  sh.page.on("request", (request) => {
    const requestId = `${request.method()}-${request.url()}-${Date.now()}`;
    requestStartTimes.set(request.url(), Date.now());

    addNetworkEntry(sessionId, {
      id: requestId,
      timestamp: new Date().toISOString(),
      type: 'request',
      method: request.method(),
      url: request.url(),
      resourceType: request.resourceType(),
    });
  });

  sh.page.on("response", async (response) => {
    const requestUrl = response.url();
    const startTime = requestStartTimes.get(requestUrl);
    const duration = startTime ? Date.now() - startTime : undefined;
    const responseId = `${response.request().method()}-${requestUrl}-${Date.now()}`;

    let size: number | undefined;
    try {
      const body = await response.body();
      size = body.length;
    } catch (error) {
      // Some responses may not have accessible bodies
      size = undefined;
    }

    addNetworkEntry(sessionId, {
      id: responseId,
      timestamp: new Date().toISOString(),
      type: 'response',
      method: response.request().method(),
      url: requestUrl,
      status: response.status(),
      duration,
      resourceType: response.request().resourceType(),
      size,
    });

    // Clean up timing data
    requestStartTimes.delete(requestUrl);
  });

  // Check if we need to evict old sessions (LRU)
  if (Object.keys(STATE).length > MAX_SESSIONS) {
    console.log(
      `[LRU] Session limit exceeded: ${
        Object.keys(STATE).length
      }/${MAX_SESSIONS}`
    );
    await evictOldestSession();
  }
  return sh;
}

export function addConsoleLog(sessionId: string, log: ConsoleLog): void {
  // Add to global logs (backward compatibility)
  if (!STATE[sessionId]) {
    return;
  }
  STATE[sessionId].logs.push(log);
  if (STATE[sessionId].logs.length > MAX_LOGS) {
    STATE[sessionId].logs.shift(); // FIFO rotation
  }
}

export function getConsoleLogs(sessionId: string): ConsoleLog[] {
  return [...(STATE[sessionId]?.logs || [])];
}

export function clearConsoleLogs(sessionId: string): void {
  if (STATE[sessionId]) {
    STATE[sessionId].logs = [];
  }
}

export function addNetworkEntry(sessionId: string, entry: NetworkEntry): void {
  if (!STATE[sessionId]) {
    return;
  }
  STATE[sessionId].networkEntries.push(entry);
  if (STATE[sessionId].networkEntries.length > MAX_NETWORK_ENTRIES) {
    STATE[sessionId].networkEntries.shift(); // FIFO rotation
  }
}

export function getNetworkEntries(sessionId: string): NetworkEntry[] {
  return [...(STATE[sessionId]?.networkEntries || [])];
}

// TODO: decide if this is needed, as network entries are captured fresh in each session
export function clearNetworkEntries(sessionId: string): void {
  if (STATE[sessionId]) {
    STATE[sessionId].networkEntries = [];
  }
}

async function evictOldestSession(): Promise<void> {
  const sessionIds = Object.keys(STATE);
  if (sessionIds.length === 0) return;

  // Find the session with the oldest last_used timestamp
  const oldestSessionId = sessionIds.reduce((oldest, current) =>
    STATE[current].last_used < STATE[oldest].last_used ? current : oldest
  );

  console.log(`[LRU] Evicting oldest session: ${oldestSessionId}`);

  // Properly close the stagehand browser instance
  try {
    await STATE[oldestSessionId].stagehand.close();
  } catch (error) {
    console.error(
      `[LRU] Error closing stagehand for session ${oldestSessionId}:`,
      error
    );
  }

  // Remove from STATE
  delete STATE[oldestSessionId];

  console.log(
    `[LRU] Sessions after eviction: ${
      Object.keys(STATE).length
    }/${MAX_SESSIONS}`
  );
}

export function sanitize(bodyText: string) {
  const content = bodyText
    .split("\n")
    .map((line) => line.trim())
    .filter(
      (line) =>
        line &&
        !(
          (line.includes("{") && line.includes("}")) ||
          line.includes("@keyframes") ||
          line.match(/^\.[a-zA-Z0-9_-]+\s*{/) ||
          line.match(/^[a-zA-Z-]+:[a-zA-Z0-9%\s\(\)\.,-]+;$/)
        )
    )
    .map((line) =>
      line.replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) =>
        String.fromCharCode(parseInt(hex, 16))
      )
    );
  return content;
}
