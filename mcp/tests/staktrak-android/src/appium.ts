import { remote } from "webdriverio";
import type { ChainablePromiseElement } from "webdriverio";
import {
  AndroidSelector,
  AppiumSessionMeta,
  StartSessionInput,
} from "./types";

let driver: WebdriverIO.Browser | null = null;
let sessionMeta: AppiumSessionMeta | null = null;

function getAppiumServerConfig() {
  const appiumUrl = process.env.APPIUM_SERVER_URL || "http://127.0.0.1:4723";
  const url = new URL(appiumUrl);
  return {
    hostname: url.hostname,
    port: Number(url.port || 4723),
    path: url.pathname && url.pathname !== "" ? url.pathname : "/",
    protocol: (url.protocol.replace(":", "") || "http") as "http" | "https",
  };
}

function ensureDriver(): WebdriverIO.Browser {
  if (!driver) {
    throw new Error("No active Appium session. Call /session/start first.");
  }
  return driver;
}

function escapeXpathText(value: string): string {
  if (!value.includes("'")) {
    return `'${value}'`;
  }
  return `concat('${value.split("'").join(`',"'",'`)}')`;
}

function findElement(selector: AndroidSelector): ChainablePromiseElement {
  const currentDriver = ensureDriver();

  if (selector.resourceId) {
    return currentDriver.$(`id=${selector.resourceId}`);
  }

  if (selector.accessibilityId) {
    return currentDriver.$(`~${selector.accessibilityId}`);
  }

  if (selector.text) {
    return currentDriver.$(`//*[@text=${escapeXpathText(selector.text)}]`);
  }

  if (selector.xpath) {
    return currentDriver.$(selector.xpath);
  }

  throw new Error("Selector is required (resourceId/accessibilityId/text/xpath).");
}

export async function startSession(input: StartSessionInput): Promise<{ sessionId: string }> {
  if (driver) {
    await stopSession();
  }

  const serverConfig = getAppiumServerConfig();

  const capabilities: Record<string, unknown> = {
    platformName: "Android",
    "appium:automationName": "UiAutomator2",
    "appium:deviceName": input.deviceName,
    "appium:appPackage": input.packageName,
  };

  if (input.activity) {
    capabilities["appium:appActivity"] = input.activity;
  }

  driver = await remote({
    ...serverConfig,
    logLevel: "error",
    capabilities,
  });

  sessionMeta = {
    packageName: input.packageName,
    activity: input.activity,
    deviceName: input.deviceName,
  };

  return { sessionId: driver.sessionId };
}

export async function stopSession(): Promise<void> {
  if (!driver) {
    return;
  }

  const current = driver;
  driver = null;
  sessionMeta = null;
  await current.deleteSession();
}

export function getSessionMeta(): AppiumSessionMeta | null {
  return sessionMeta;
}

export async function getPageSource(): Promise<string> {
  const currentDriver = ensureDriver();
  return currentDriver.getPageSource();
}

export async function takeScreenshot(): Promise<string> {
  const currentDriver = ensureDriver();
  return currentDriver.takeScreenshot();
}

export async function tapByCoordinates(x: number, y: number): Promise<void> {
  const currentDriver = ensureDriver();

  await currentDriver.performActions([
    {
      type: "pointer",
      id: "finger1",
      parameters: { pointerType: "touch" },
      actions: [
        { type: "pointerMove", duration: 0, x, y },
        { type: "pointerDown", button: 0 },
        { type: "pause", duration: 50 },
        { type: "pointerUp", button: 0 },
      ],
    },
  ]);

  await currentDriver.releaseActions();
}

export async function tapBySelector(selector: AndroidSelector): Promise<void> {
  const element = findElement(selector);
  await element.waitForDisplayed({ timeout: 5000 });
  await element.click();
}

export async function typeIntoElement(
  selector: AndroidSelector,
  text: string,
  replace = true
): Promise<void> {
  const element = findElement(selector);
  await element.waitForDisplayed({ timeout: 5000 });
  await element.click();
  if (replace) {
    await element.clearValue();
  }
  await element.setValue(text);
}

export async function swipe(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  durationMs = 400
): Promise<void> {
  const currentDriver = ensureDriver();

  await currentDriver.performActions([
    {
      type: "pointer",
      id: "finger1",
      parameters: { pointerType: "touch" },
      actions: [
        { type: "pointerMove", duration: 0, x: startX, y: startY },
        { type: "pointerDown", button: 0 },
        { type: "pointerMove", duration: durationMs, x: endX, y: endY },
        { type: "pointerUp", button: 0 },
      ],
    },
  ]);

  await currentDriver.releaseActions();
}

export async function pressBack(): Promise<void> {
  const currentDriver = ensureDriver();
  await (currentDriver as any).execute("mobile: pressKey", [{ keycode: 4 }]);
}

export async function pressHome(): Promise<void> {
  const currentDriver = ensureDriver();
  await (currentDriver as any).execute("mobile: pressKey", [{ keycode: 3 }]);
}