import { remote } from "webdriverio";
import type { ChainablePromiseElement } from "webdriverio";
import {
  AndroidSelector,
  AppiumConnectionState,
  AppiumSessionMeta,
  StartSessionInput,
} from "./types";

let driver: WebdriverIO.Browser | null = null;
let sessionMeta: AppiumSessionMeta | null = null;
let desiredTarget: StartSessionInput | null = null;
let sessionPromise: Promise<{ sessionId: string; reused: boolean }> | null =
  null;
let connectionState: AppiumConnectionState = {
  status: "disconnected",
};

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
    throw new Error("No active Appium session.");
  }
  return driver;
}

function sameTarget(
  a: StartSessionInput | AppiumSessionMeta,
  b: StartSessionInput,
): boolean {
  return (
    a.packageName === b.packageName &&
    a.activity === b.activity &&
    a.deviceName === b.deviceName
  );
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isStaleSessionError(error: unknown): boolean {
  const message = toErrorMessage(error).toLowerCase();
  return (
    message.includes("invalid session") ||
    message.includes("no such driver") ||
    message.includes("session does not exist") ||
    message.includes("session id is null")
  );
}

function resolveDefaultTarget(): StartSessionInput {
  const packageName = process.env.APPIUM_APP_PACKAGE;
  if (!packageName) {
    throw new Error(
      "Missing app package. Set APPIUM_APP_PACKAGE or call /session/start with { package }.",
    );
  }

  return {
    packageName,
    activity: process.env.APPIUM_APP_ACTIVITY,
    deviceName: process.env.APPIUM_DEVICE_NAME || "Android",
  };
}

async function deleteCurrentDriver(): Promise<void> {
  if (!driver) {
    return;
  }

  const current = driver;
  driver = null;
  sessionMeta = null;

  try {
    await current.deleteSession();
  } catch {}
}

async function connectSession(
  target: StartSessionInput,
): Promise<{ sessionId: string }> {
  connectionState = {
    status: "reconnecting",
    target,
    lastError: connectionState.lastError,
    lastConnectedAt: connectionState.lastConnectedAt,
  };

  await deleteCurrentDriver();

  const serverConfig = getAppiumServerConfig();

  const capabilities: Record<string, unknown> = {
    platformName: "Android",
    "appium:automationName": "UiAutomator2",
    "appium:deviceName": target.deviceName,
    "appium:appPackage": target.packageName,
  };

  if (target.activity) {
    capabilities["appium:appActivity"] = target.activity;
  }

  driver = await remote({
    ...serverConfig,
    logLevel: "error",
    capabilities,
  });

  sessionMeta = {
    packageName: target.packageName,
    activity: target.activity,
    deviceName: target.deviceName,
  };

  connectionState = {
    status: "connected",
    target,
    lastConnectedAt: Date.now(),
  };

  return { sessionId: driver.sessionId };
}

async function getReadyDriver(): Promise<WebdriverIO.Browser> {
  await ensureSession();
  return ensureDriver();
}

async function executeWithAutoReconnect<T>(
  action: (currentDriver: WebdriverIO.Browser) => Promise<T>,
): Promise<T> {
  const currentDriver = await getReadyDriver();

  try {
    return await action(currentDriver);
  } catch (error) {
    if (!isStaleSessionError(error)) {
      throw error;
    }

    connectionState = {
      status: "reconnecting",
      target: desiredTarget || undefined,
      lastError: toErrorMessage(error),
      lastConnectedAt: connectionState.lastConnectedAt,
    };

    await deleteCurrentDriver();
    const retryDriver = await getReadyDriver();
    return action(retryDriver);
  }
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
  const session = await ensureSession(input, { forceRestart: true });
  return { sessionId: session.sessionId };
}

export async function ensureSession(
  input?: StartSessionInput,
  options?: { forceRestart?: boolean },
): Promise<{ sessionId: string; reused: boolean }> {
  if (input) {
    desiredTarget = input;
  } else if (!desiredTarget) {
    desiredTarget = resolveDefaultTarget();
  }

  const target = desiredTarget;
  if (!target) {
    throw new Error("No Appium target configured.");
  }

  if (sessionPromise && !options?.forceRestart) {
    return sessionPromise;
  }

  const shouldReuse =
    !options?.forceRestart &&
    driver &&
    sessionMeta &&
    sameTarget(sessionMeta, target);

  if (shouldReuse && driver) {
    connectionState = {
      status: "connected",
      target,
      lastConnectedAt: connectionState.lastConnectedAt,
      lastError: connectionState.lastError,
    };
    return { sessionId: driver.sessionId, reused: true };
  }

  sessionPromise = (async () => {
    try {
      const connected = await connectSession(target);
      return { ...connected, reused: false };
    } catch (error) {
      connectionState = {
        status: "disconnected",
        target,
        lastError: toErrorMessage(error),
        lastConnectedAt: connectionState.lastConnectedAt,
      };
      throw error;
    } finally {
      sessionPromise = null;
    }
  })();

  return sessionPromise;
}

export async function stopSession(): Promise<void> {
  await deleteCurrentDriver();
  connectionState = {
    status: "disconnected",
    target: desiredTarget || undefined,
    lastError: connectionState.lastError,
    lastConnectedAt: connectionState.lastConnectedAt,
  };
}

export function getSessionMeta(): AppiumSessionMeta | null {
  return sessionMeta;
}

export function getConnectionState(): AppiumConnectionState {
  return { ...connectionState };
}

export async function getPageSource(): Promise<string> {
  return executeWithAutoReconnect((currentDriver) =>
    currentDriver.getPageSource(),
  );
}

export async function takeScreenshot(): Promise<string> {
  return executeWithAutoReconnect((currentDriver) =>
    currentDriver.takeScreenshot(),
  );
}

export async function tapByCoordinates(x: number, y: number): Promise<void> {
  await executeWithAutoReconnect(async (currentDriver) => {
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
  });
}

export async function tapBySelector(selector: AndroidSelector): Promise<void> {
  await executeWithAutoReconnect(async () => {
    const element = findElement(selector);
    await element.waitForDisplayed({ timeout: 5000 });
    await element.click();
  });
}

export async function typeIntoElement(
  selector: AndroidSelector,
  text: string,
  replace = true
): Promise<void> {
  await executeWithAutoReconnect(async () => {
    const element = findElement(selector);
    await element.waitForDisplayed({ timeout: 5000 });
    await element.click();
    if (replace) {
      await element.clearValue();
    }
    await element.setValue(text);
  });
}

export async function swipe(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  durationMs = 400
): Promise<void> {
  await executeWithAutoReconnect(async (currentDriver) => {
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
  });
}

export async function pressBack(): Promise<void> {
  await executeWithAutoReconnect(async (currentDriver) => {
    await (currentDriver as any).execute("mobile: pressKey", [{ keycode: 4 }]);
  });
}

export async function pressHome(): Promise<void> {
  await executeWithAutoReconnect(async (currentDriver) => {
    await (currentDriver as any).execute("mobile: pressKey", [{ keycode: 3 }]);
  });
}