import { RecordedAction, RecordingContext } from "./recorder";

const METADATA_PREFIX = "// STAKTRAK_ANDROID_ACTIONS_BASE64:";

function toBase64(value: string): string {
  return Buffer.from(value, "utf8").toString("base64");
}

function fromBase64(value: string): string {
  return Buffer.from(value, "base64").toString("utf8");
}

function selectorToScript(selector: Record<string, unknown> | undefined): string {
  if (!selector) {
    return "undefined";
  }
  return JSON.stringify(selector);
}

function actionToScript(action: RecordedAction): string {
  switch (action.type) {
    case "tap": {
      if (action.selector) {
        return `    await tapBySelector(driver, ${selectorToScript(action.selector)});`;
      }
      return `    await tapByCoordinates(driver, ${action.x}, ${action.y});`;
    }
    case "type":
      return `    await typeIntoElement(driver, ${selectorToScript(action.selector)}, ${JSON.stringify(action.text)}, ${action.replace});`;
    case "swipe":
      return `    await swipe(driver, ${action.startX}, ${action.startY}, ${action.endX}, ${action.endY}, ${action.durationMs});`;
    case "back":
      return "    await pressKey(driver, 4);";
    case "home":
      return "    await pressKey(driver, 3);";
    default:
      return "";
  }
}

export function generateAppiumScript(
  actions: RecordedAction[],
  context: RecordingContext | null
): string {
  const metadata = toBase64(JSON.stringify(actions));
  const packageName = context?.packageName || "com.example.app";
  const activityLine = context?.activity
    ? `    'appium:appActivity': '${context.activity}',\n`
    : "";
  const deviceName = context?.deviceName || "Android";

  const body = actions.map(actionToScript).filter(Boolean).join("\n");

  return `${METADATA_PREFIX}${metadata}
const { remote } = require('webdriverio');

async function tapByCoordinates(driver, x, y) {
  await driver.performActions([{
    type: 'pointer',
    id: 'finger1',
    parameters: { pointerType: 'touch' },
    actions: [
      { type: 'pointerMove', duration: 0, x, y },
      { type: 'pointerDown', button: 0 },
      { type: 'pause', duration: 50 },
      { type: 'pointerUp', button: 0 }
    ]
  }]);
  await driver.releaseActions();
}

async function tapBySelector(driver, selector) {
  if (selector.resourceId) return driver.$('id=' + selector.resourceId).click();
  if (selector.accessibilityId) return driver.$('~' + selector.accessibilityId).click();
  if (selector.text) return driver.$('//*[@text="' + selector.text + '"]').click();
  if (selector.xpath) return driver.$(selector.xpath).click();
  throw new Error('Invalid selector');
}

async function typeIntoElement(driver, selector, text, replace) {
  let el;
  if (selector.resourceId) el = await driver.$('id=' + selector.resourceId);
  else if (selector.accessibilityId) el = await driver.$('~' + selector.accessibilityId);
  else if (selector.text) el = await driver.$('//*[@text="' + selector.text + '"]');
  else if (selector.xpath) el = await driver.$(selector.xpath);
  else throw new Error('Invalid selector');
  await el.click();
  if (replace) await el.clearValue();
  await el.setValue(text);
}

async function swipe(driver, startX, startY, endX, endY, durationMs) {
  await driver.performActions([{
    type: 'pointer',
    id: 'finger1',
    parameters: { pointerType: 'touch' },
    actions: [
      { type: 'pointerMove', duration: 0, x: startX, y: startY },
      { type: 'pointerDown', button: 0 },
      { type: 'pointerMove', duration: durationMs, x: endX, y: endY },
      { type: 'pointerUp', button: 0 }
    ]
  }]);
  await driver.releaseActions();
}

async function pressKey(driver, keycode) {
  await driver.execute('mobile: pressKey', [{ keycode }]);
}

async function run() {
  const driver = await remote({
    protocol: 'http',
    hostname: process.env.APPIUM_HOST || '127.0.0.1',
    port: Number(process.env.APPIUM_PORT || 4723),
    path: process.env.APPIUM_PATH || '/',
    capabilities: {
      platformName: 'Android',
      'appium:automationName': 'UiAutomator2',
      'appium:deviceName': process.env.APPIUM_DEVICE_NAME || '${deviceName}',
      'appium:appPackage': process.env.APPIUM_APP_PACKAGE || '${packageName}',
${activityLine}    }
  });

  try {
${body}
  } finally {
    await driver.deleteSession();
  }
}

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
`;
}

export function parseActionsFromScript(script: string): RecordedAction[] | null {
  const line = script
    .split("\n")
    .find((value) => value.startsWith(METADATA_PREFIX));

  if (!line) {
    return null;
  }

  const encoded = line.replace(METADATA_PREFIX, "").trim();
  if (!encoded) {
    return null;
  }

  try {
    const decoded = fromBase64(encoded);
    const actions = JSON.parse(decoded) as RecordedAction[];
    return Array.isArray(actions) ? actions : null;
  } catch {
    return null;
  }
}