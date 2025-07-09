import { Stagehand } from "@browserbasehq/stagehand";
import { getProvider } from "./providers.js";

let STAGEHAND: Stagehand | null = null;

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

let CONSOLE_LOGS: ConsoleLog[] = [];
const MAX_LOGS = parseInt(process.env.STAGEHAND_MAX_CONSOLE_LOGS || "1000");

// Action-sequence ID system for session isolation
let currentActionId: string | null = null;
const actionLogs = new Map<string, ConsoleLog[]>();
const actionTimestamps = new Map<string, number>();
const MAX_ACTIONS = 100; // LRU limit
const ACTION_TTL_MS = 60 * 60 * 1000; // 1 hour TTL

export async function getOrCreateStagehand() {
  if (STAGEHAND) {
    return STAGEHAND;
  }
  let provider = getProvider();
  console.log("initializing stagehand!", provider.model);
  const sh = new Stagehand({
    env: "LOCAL",
    domSettleTimeoutMs: 30000,
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

  // Clear any existing logs when stagehand is recreated (only on new creation)
  clearConsoleLogs();

  // Set up console log listener
  sh.page.on("console", (msg) => {
    addConsoleLog({
      timestamp: new Date().toISOString(),
      type: msg.type(),
      text: msg.text(),
      location: msg.location(),
    });
  });

  STAGEHAND = sh;
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

export function addConsoleLog(log: ConsoleLog): void {
  // Add to global logs (backward compatibility)
  CONSOLE_LOGS.push(log);
  if (CONSOLE_LOGS.length > MAX_LOGS) {
    CONSOLE_LOGS.shift(); // FIFO rotation
  }
  
  // Add to action-specific logs if we have an active action
  if (currentActionId) {
    if (!actionLogs.has(currentActionId)) {
      actionLogs.set(currentActionId, []);
    }
    const logs = actionLogs.get(currentActionId)!;
    logs.push(log);
    if (logs.length > MAX_LOGS) {
      logs.shift(); // FIFO rotation
    }
  }
}

export function getConsoleLogs(): ConsoleLog[] {
  return [...CONSOLE_LOGS];
}

export function clearConsoleLogs(): void {
  CONSOLE_LOGS = [];
}

// Action-sequence ID management functions
export function startAction(actionId: string): void {
  currentActionId = actionId;
  actionTimestamps.set(actionId, Date.now());
  
  // Initialize empty logs array for this action
  if (!actionLogs.has(actionId)) {
    actionLogs.set(actionId, []);
  }
  
  // Clean up old actions (TTL and LRU)
  cleanupOldActions();
}

export function getActionLogs(actionId: string): ConsoleLog[] {
  const logs = actionLogs.get(actionId);
  return logs ? [...logs] : [];
}

function cleanupOldActions(): void {
  const now = Date.now();
  const toDelete: string[] = [];
  
  // Remove expired actions (TTL)
  for (const [actionId, timestamp] of actionTimestamps.entries()) {
    if (now - timestamp > ACTION_TTL_MS) {
      toDelete.push(actionId);
    }
  }
  
  // Remove oldest actions if we exceed limit (LRU)
  if (actionTimestamps.size > MAX_ACTIONS) {
    const sortedActions = Array.from(actionTimestamps.entries())
      .sort((a, b) => a[1] - b[1])
      .slice(0, actionTimestamps.size - MAX_ACTIONS);
    
    for (const [actionId] of sortedActions) {
      toDelete.push(actionId);
    }
  }
  
  // Delete old actions
  for (const actionId of toDelete) {
    actionLogs.delete(actionId);
    actionTimestamps.delete(actionId);
  }
}
