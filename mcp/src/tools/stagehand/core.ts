import { Stagehand, Page } from "@browserbasehq/stagehand";
import { getProvider } from "./providers.js";

let STATE: {
  [sessionId: string]: {
    stagehand: Stagehand;
    last_used: Date;
    logs: ConsoleLog[];
    networkEntries: NetworkEntry[];
  };
} = {};

// Helper to get the active page from Stagehand instance
function getActivePage(sh: Stagehand): Page {
  const page = sh.context.activePage();
  if (!page) {
    throw new Error("No active page available");
  }
  return page;
}

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
    domSettleTimeout: 60000,
    localBrowserLaunchOptions: {
      headless: true,
      viewport: { width: 1024, height: 768 },
    },
    model: {
      modelName: provider.model,
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

  // Set up console + network capture on the active page via CDP
  const page = getActivePage(sh);
  attachCapture(page as any, sessionId);

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

const NET_RESOURCE: Record<string, string> = { XHR: "xhr", Fetch: "fetch", Document: "document" };
const PENDING: { [sessionId: string]: Map<string, { method: string; url: string; type: string }> } = {};

// Attach CDP-based console + network capture to a page's main session.
// Populates the session's console logs (incl. uncaught exceptions) and network entries.
function attachCapture(page: any, sessionId: string): void {
  try {
    page.on("console", (msg: any) => {
      const location = msg.location?.() || {};
      addConsoleLog(sessionId, {
        timestamp: new Date().toISOString(),
        type: msg.type?.() || "log",
        text: msg.text?.() || "",
        location: {
          url: location.url || "",
          lineNumber: location.lineNumber || 0,
          columnNumber: location.columnNumber || 0,
        },
      });
    });

    const session = page?.mainFrame?.()?.session;
    if (!session) return;
    PENDING[sessionId] = new Map();

    const pushLog = (type: string, text: string, url = "") =>
      addConsoleLog(sessionId, {
        timestamp: new Date().toISOString(),
        type,
        text,
        location: { url, lineNumber: 0, columnNumber: 0 },
      });

    session.send("Runtime.enable").catch(() => {});
    session.on("Runtime.exceptionThrown", (p: any) => {
      const d = p?.exceptionDetails;
      pushLog("error", d?.exception?.description ?? d?.text ?? "uncaught exception", d?.url ?? "");
    });

    session.send("Log.enable").catch(() => {});
    session.on("Log.entryAdded", (p: any) => {
      const e = p?.entry;
      if (e && (e.level === "error" || e.level === "warning")) {
        pushLog(e.level, e.text ?? "", e.url ?? "");
      }
    });

    session.send("Network.enable").catch(() => {});
    session.on("Network.requestWillBeSent", (p: any) => {
      const type = p?.type ?? "";
      if (p?.redirectResponse && NET_RESOURCE[type]) {
        addNetworkEntry(sessionId, {
          id: `${p.requestId}-r`,
          timestamp: new Date().toISOString(),
          type: "response",
          method: p?.request?.method ?? "GET",
          url: p?.redirectResponse?.url ?? p?.request?.url ?? "",
          status: p?.redirectResponse?.status ?? 0,
          resourceType: NET_RESOURCE[type],
        });
      }
      PENDING[sessionId]?.set(p.requestId, {
        method: p?.request?.method ?? "GET",
        url: p?.request?.url ?? "",
        type,
      });
    });
    session.on("Network.responseReceived", (p: any) => {
      const type = p?.type ?? PENDING[sessionId]?.get(p?.requestId)?.type ?? "";
      const req = PENDING[sessionId]?.get(p?.requestId);
      PENDING[sessionId]?.delete(p?.requestId);
      if (!NET_RESOURCE[type]) return;
      addNetworkEntry(sessionId, {
        id: p?.requestId ?? "",
        timestamp: new Date().toISOString(),
        type: "response",
        method: req?.method ?? "GET",
        url: p?.response?.url ?? req?.url ?? "",
        status: p?.response?.status ?? 0,
        resourceType: NET_RESOURCE[type],
      });
    });
    session.on("Network.loadingFailed", (p: any) => {
      const req = PENDING[sessionId]?.get(p?.requestId);
      const type = req?.type ?? p?.type ?? "";
      PENDING[sessionId]?.delete(p?.requestId);
      if (!NET_RESOURCE[type]) return;
      addNetworkEntry(sessionId, {
        id: p?.requestId ?? "",
        timestamp: new Date().toISOString(),
        type: "response",
        method: req?.method ?? "",
        url: req?.url ?? "(unknown)",
        status: 0,
        resourceType: NET_RESOURCE[type],
      });
    });
  } catch {
    /* a CDP quirk must never fail session creation */
  }
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
