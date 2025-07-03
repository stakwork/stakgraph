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
  CONSOLE_LOGS.push(log);
  if (CONSOLE_LOGS.length > MAX_LOGS) {
    CONSOLE_LOGS.shift(); // FIFO rotation
  }
}

export function getConsoleLogs(): ConsoleLog[] {
  return [...CONSOLE_LOGS];
}

export function clearConsoleLogs(): void {
  CONSOLE_LOGS = [];
}
