var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));

// src/actionModel.ts
function resultsToActions(results) {
  var _a;
  const actions = [];
  const navigations = (results.pageNavigation || []).slice().sort((a, b) => a.timestamp - b.timestamp);
  const normalize = (u) => {
    var _a2;
    try {
      const url = new URL(u, ((_a2 = results.userInfo) == null ? void 0 : _a2.url) || "http://localhost");
      return url.origin + url.pathname.replace(/\/$/, "");
    } catch (e) {
      return u.replace(/[?#].*$/, "").replace(/\/$/, "");
    }
  };
  for (const nav of navigations) {
    actions.push({ kind: "nav", timestamp: nav.timestamp, url: nav.url, normalizedUrl: normalize(nav.url) });
  }
  const clicks = ((_a = results.clicks) == null ? void 0 : _a.clickDetails) || [];
  for (let i = 0; i < clicks.length; i++) {
    const cd = clicks[i];
    actions.push({
      kind: "click",
      timestamp: cd.timestamp,
      locator: {
        primary: cd.selectors.stabilizedPrimary || cd.selectors.primary,
        fallbacks: cd.selectors.fallbacks || [],
        role: cd.selectors.role,
        text: cd.selectors.text,
        tagName: cd.selectors.tagName,
        stableSelector: cd.selectors.stabilizedPrimary || cd.selectors.primary,
        candidates: cd.selectors.scores || void 0
      }
    });
    const nav = navigations.find((n) => n.timestamp > cd.timestamp && n.timestamp - cd.timestamp < 1800);
    if (nav) {
      actions.push({
        kind: "waitForUrl",
        timestamp: nav.timestamp - 1,
        // ensure ordering between click and nav
        expectedUrl: nav.url,
        normalizedUrl: normalize(nav.url),
        navRefTimestamp: nav.timestamp
      });
    }
  }
  if (results.inputChanges) {
    for (const input of results.inputChanges) {
      if (input.action === "complete" || !input.action) {
        actions.push({
          kind: "input",
          timestamp: input.timestamp,
          locator: { primary: input.elementSelector, fallbacks: [] },
          value: input.value
        });
      }
    }
  }
  if (results.formElementChanges) {
    for (const fe of results.formElementChanges) {
      actions.push({
        kind: "form",
        timestamp: fe.timestamp,
        locator: { primary: fe.elementSelector, fallbacks: [] },
        formType: fe.type,
        value: fe.value,
        checked: fe.checked
      });
    }
  }
  if (results.assertions) {
    for (const asrt of results.assertions) {
      actions.push({
        kind: "assertion",
        timestamp: asrt.timestamp,
        locator: { primary: asrt.selector, fallbacks: [] },
        value: asrt.value
      });
    }
  }
  actions.sort((a, b) => a.timestamp - b.timestamp || weightOrder(a.kind) - weightOrder(b.kind));
  refineLocators(actions);
  return actions;
}
function weightOrder(kind) {
  switch (kind) {
    case "click":
      return 1;
    case "waitForUrl":
      return 2;
    case "nav":
      return 3;
    default:
      return 4;
  }
}
function refineLocators(actions) {
  if (typeof document === "undefined")
    return;
  const seen = /* @__PURE__ */ new Set();
  for (const a of actions) {
    if (!a.locator)
      continue;
    const { primary, fallbacks } = a.locator;
    const validated = [];
    if (isUnique(primary))
      validated.push(primary);
    for (const fb of fallbacks) {
      if (validated.length >= 3)
        break;
      if (isUnique(fb))
        validated.push(fb);
    }
    if (validated.length === 0)
      continue;
    a.locator.primary = validated[0];
    a.locator.fallbacks = validated.slice(1);
    const key = a.locator.primary + "::" + a.kind;
    if (seen.has(key) && a.locator.fallbacks.length > 0) {
      a.locator.primary = a.locator.fallbacks[0];
      a.locator.fallbacks = a.locator.fallbacks.slice(1);
    }
    seen.add(a.locator.primary + "::" + a.kind);
  }
}
function isUnique(sel) {
  if (!sel || /^(html|body|div|span|p|button|input)$/i.test(sel))
    return false;
  try {
    const nodes = document.querySelectorAll(sel);
    return nodes.length === 1;
  } catch (e) {
    return false;
  }
}

// src/playwright-generator.ts
var RecordingManager = class {
  constructor() {
    this.trackingData = {
      pageNavigation: [],
      clicks: { clickCount: 0, clickDetails: [] },
      inputChanges: [],
      formElementChanges: [],
      assertions: [],
      keyboardActivities: [],
      mouseMovement: [],
      mouseScroll: [],
      focusChanges: [],
      visibilitychanges: [],
      windowSizes: [],
      touchEvents: [],
      audioVideoInteractions: []
    };
    this.capturedActions = [];
    this.actionIdCounter = 0;
  }
  /**
   * Handle an event from the iframe and store it
   */
  handleEvent(eventType, eventData) {
    switch (eventType) {
      case "click":
        this.trackingData.clicks.clickDetails.push(eventData);
        this.trackingData.clicks.clickCount++;
        break;
      case "nav":
      case "navigation":
        this.trackingData.pageNavigation.push({
          type: "navigation",
          url: eventData.url,
          timestamp: eventData.timestamp
        });
        break;
      case "input":
        this.trackingData.inputChanges.push({
          elementSelector: eventData.selector || "",
          value: eventData.value,
          timestamp: eventData.timestamp,
          action: "fill"
        });
        break;
      case "form":
        this.trackingData.formElementChanges.push({
          elementSelector: eventData.selector || "",
          type: eventData.formType || "input",
          checked: eventData.checked,
          value: eventData.value || "",
          text: eventData.text,
          timestamp: eventData.timestamp
        });
        break;
      case "assertion":
        this.trackingData.assertions.push({
          id: eventData.id,
          type: eventData.type || "hasText",
          selector: eventData.selector,
          value: eventData.value || "",
          timestamp: eventData.timestamp
        });
        break;
      default:
        return null;
    }
    const action = this.createAction(eventType, eventData);
    if (action) {
      this.capturedActions.push(action);
    }
    return action;
  }
  createAction(eventType, eventData) {
    const id = `${Date.now()}_${this.actionIdCounter++}`;
    const baseAction = {
      id,
      timestamp: eventData.timestamp || Date.now()
    };
    switch (eventType) {
      case "click":
        return __spreadProps(__spreadValues({}, baseAction), {
          kind: "click",
          locator: eventData.selectors || eventData.locator,
          elementInfo: eventData.elementInfo
        });
      case "nav":
      case "navigation":
        return __spreadProps(__spreadValues({}, baseAction), {
          kind: "nav",
          url: eventData.url
        });
      case "input":
        return __spreadProps(__spreadValues({}, baseAction), {
          kind: "input",
          value: eventData.value,
          locator: eventData.locator || { primary: eventData.selector }
        });
      case "form":
        return __spreadProps(__spreadValues({}, baseAction), {
          kind: "form",
          formType: eventData.formType,
          checked: eventData.checked,
          value: eventData.value,
          locator: eventData.locator || { primary: eventData.selector }
        });
      case "assertion":
        return __spreadProps(__spreadValues({}, baseAction), {
          kind: "assertion",
          value: eventData.value,
          locator: { primary: eventData.selector }
        });
      default:
        return __spreadProps(__spreadValues({}, baseAction), {
          kind: eventType
        });
    }
  }
  /**
   * Remove an action by ID
   */
  removeAction(actionId) {
    const action = this.capturedActions.find((a) => a.id === actionId);
    if (!action)
      return false;
    this.capturedActions = this.capturedActions.filter((a) => a.id !== actionId);
    this.removeFromTrackingData(action);
    return true;
  }
  removeFromTrackingData(action) {
    const timestamp = action.timestamp;
    switch (action.kind) {
      case "click":
        this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
          (c) => c.timestamp !== timestamp
        );
        this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
        break;
      case "nav":
        this.trackingData.pageNavigation = this.trackingData.pageNavigation.filter(
          (n) => n.timestamp !== timestamp
        );
        break;
      case "input":
        this.trackingData.inputChanges = this.trackingData.inputChanges.filter(
          (i) => i.timestamp !== timestamp
        );
        break;
      case "form":
        this.trackingData.formElementChanges = this.trackingData.formElementChanges.filter(
          (f) => f.timestamp !== timestamp
        );
        break;
      case "assertion":
        this.trackingData.assertions = this.trackingData.assertions.filter(
          (a) => a.timestamp !== timestamp
        );
        const clickBeforeAssertion = this.trackingData.clicks.clickDetails.filter((c) => c.timestamp < timestamp).sort((a, b) => b.timestamp - a.timestamp)[0];
        if (clickBeforeAssertion && timestamp - clickBeforeAssertion.timestamp < 1e3) {
          this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
            (c) => c.timestamp !== clickBeforeAssertion.timestamp
          );
          this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
        }
        break;
    }
  }
  /**
   * Generate Playwright test from current data
   */
  generateTest(url, options) {
    const actions = resultsToActions(this.trackingData);
    return generatePlaywrightTestFromActions(actions, __spreadValues({
      baseUrl: url
    }, options));
  }
  /**
   * Get current actions for UI display
   */
  getActions() {
    return [...this.capturedActions];
  }
  /**
   * Get tracking data (for compatibility)
   */
  getTrackingData() {
    return this.trackingData;
  }
  /**
   * Clear all recorded data
   */
  clear() {
    this.trackingData = {
      pageNavigation: [],
      clicks: { clickCount: 0, clickDetails: [] },
      inputChanges: [],
      formElementChanges: [],
      assertions: [],
      keyboardActivities: [],
      mouseMovement: [],
      mouseScroll: [],
      focusChanges: [],
      visibilitychanges: [],
      windowSizes: [],
      touchEvents: [],
      audioVideoInteractions: []
    };
    this.capturedActions = [];
    this.actionIdCounter = 0;
  }
  /**
   * Clear all actions (but keep recording)
   */
  clearAllActions() {
    this.clear();
  }
};
function escapeTextForAssertion(text) {
  if (!text)
    return "";
  return text.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t").trim();
}
function normalizeText(t) {
  return (t || "").trim();
}
function locatorToSelector(l) {
  if (!l)
    return 'page.locator("body")';
  const primary = l.stableSelector || l.primary;
  if (/\[data-testid=/.test(primary)) {
    const m = primary.match(/\[data-testid=["']([^"']+)["']\]/);
    if (m)
      return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
  }
  if (primary.startsWith("#") && /^[a-zA-Z][\w-]*$/.test(primary.slice(1)))
    return `page.locator('${primary}')`;
  if (/^[a-zA-Z]+\.[a-zA-Z0-9_-]+/.test(primary)) {
    return `page.locator('${primary}')`;
  }
  if (l.role && l.text) {
    const txt = normalizeText(l.text);
    if (txt && txt.length <= 50)
      return `page.getByRole('${l.role}', { name: '${escapeTextForAssertion(txt)}' })`;
  }
  if (l.text && l.text.length <= 30 && l.text.length > 1)
    return `page.getByText('${escapeTextForAssertion(normalizeText(l.text))}')`;
  if (primary && !primary.startsWith("page."))
    return `page.locator('${primary}')`;
  for (const fb of l.fallbacks) {
    if (fb && !/^[a-zA-Z]+$/.test(fb))
      return `page.locator('${fb}')`;
  }
  return 'page.locator("body")';
}
function generatePlaywrightTestFromActions(actions, options) {
  const name = options.testName || "Recorded flow";
  const viewport = options.viewport || { width: 1280, height: 720 };
  let body = "";
  let lastTs = null;
  const base = options.baseUrl ? options.baseUrl.replace(/\/$/, "") : "";
  function fullUrl(u) {
    if (!u)
      return "";
    if (/^https?:/i.test(u))
      return u;
    if (u.startsWith("/"))
      return base + u;
    return base + "/" + u;
  }
  let i = 0;
  const collapsed = [];
  for (let k = 0; k < actions.length; k++) {
    const curr = actions[k];
    const prev = collapsed[collapsed.length - 1];
    if (curr.kind === "nav" && prev && prev.kind === "nav" && prev.url === curr.url)
      continue;
    collapsed.push(curr);
  }
  actions = collapsed;
  while (i < actions.length) {
    const a = actions[i];
    if (a.kind === "click" && i + 1 < actions.length) {
      const nxt = actions[i + 1];
      if (nxt.kind === "nav" && nxt.timestamp - a.timestamp < 1500) {
        if (lastTs != null) {
          const delta = Math.max(0, a.timestamp - lastTs);
          const wait = Math.min(3e3, Math.max(100, delta));
          if (wait > 400)
            body += `  await page.waitForTimeout(${wait});
`;
        }
        body += `  await Promise.all([
`;
        body += `    page.waitForURL('${fullUrl(nxt.url)}'),
`;
        body += `    ${locatorToSelector(a.locator)}.click()
`;
        body += `  ]);
`;
        lastTs = nxt.timestamp;
        i += 2;
        continue;
      }
    }
    if (lastTs != null) {
      const delta = Math.max(0, a.timestamp - lastTs);
      const wait = Math.min(3e3, Math.max(100, delta));
      if (wait > 500)
        body += `  await page.waitForTimeout(${wait});
`;
    }
    switch (a.kind) {
      case "nav": {
        const target = fullUrl(a.url);
        if (i === 0) {
          body += `  await page.goto('${target}');
`;
        } else {
          body += `  await page.waitForURL('${target}');
`;
        }
        break;
      }
      case "click":
        body += `  await ${locatorToSelector(a.locator)}.click();
`;
        break;
      case "input":
        body += `  await ${locatorToSelector(a.locator)}.fill('${escapeTextForAssertion(a.value || "")}');
`;
        break;
      case "form":
        if (a.formType === "checkbox" || a.formType === "radio") {
          body += a.checked ? `  await ${locatorToSelector(a.locator)}.check();
` : `  await ${locatorToSelector(a.locator)}.uncheck();
`;
        } else if (a.formType === "select") {
          body += `  await ${locatorToSelector(a.locator)}.selectOption('${escapeTextForAssertion(a.value || "")}');
`;
        }
        break;
      case "assertion":
        if (a.value && a.value.length > 0) {
          body += `  await expect(${locatorToSelector(a.locator)}).toContainText('${escapeTextForAssertion(a.value)}');
`;
        } else {
          body += `  await expect(${locatorToSelector(a.locator)}).toBeVisible();
`;
        }
        break;
    }
    lastTs = a.timestamp;
    i++;
  }
  return `import { test, expect } from '@playwright/test'

test('${name}', async ({ page }) => {
  await page.setViewportSize({ width: ${viewport.width}, height: ${viewport.height} })
${body.split("\n").filter((l) => l.trim()).map((l) => l).join("\n")}
})
`;
}
if (typeof window !== "undefined") {
  const existing = window.PlaywrightGenerator || {};
  existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
  existing.generatePlaywrightTest = (baseUrl, results) => {
    try {
      const actions = resultsToActions(results);
      return generatePlaywrightTestFromActions(actions, { baseUrl });
    } catch (e) {
      console.warn("PlaywrightGenerator.generatePlaywrightTest failed", e);
      return "";
    }
  };
  existing.RecordingManager = RecordingManager;
  window.PlaywrightGenerator = existing;
}
export {
  RecordingManager,
  generatePlaywrightTestFromActions
};
