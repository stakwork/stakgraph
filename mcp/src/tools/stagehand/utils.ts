import { Stagehand } from "@browserbasehq/stagehand";
import { getProvider } from "./providers.js";

let STATE: {
  [sessionId: string]: {
    stagehand: Stagehand;
    last_used: Date;
    logs: ConsoleLog[];
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

const MAX_LOGS = parseInt(process.env.STAGEHAND_MAX_CONSOLE_LOGS || "1000");
const MAX_SESSIONS = 25; // LRU limit for stagehand instances

export async function getOrCreateStagehand(sessionIdMaybe?: string) {
  const sessionId = sessionIdMaybe || "default-session-id";

  console.log("getOrCreateStagehand SESSION ID", sessionId);
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
