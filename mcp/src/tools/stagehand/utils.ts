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

// TODO: remove old unused STAGEHANDS and CONSOLE_LOGS after a while
const MAX_ACTIONS = 100; // LRU limit
const ACTION_TTL_MS = 60 * 60 * 1000; // 1 hour TTL

export async function getOrCreateStagehand(sessionIdMaybe?: string) {
  const sessionId = sessionIdMaybe || "default-session-id";

  if (STATE[sessionId]) {
    console.log("sessionId exists:", sessionId);
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

  // empty
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
  STATE[sessionId].logs = [];
}
