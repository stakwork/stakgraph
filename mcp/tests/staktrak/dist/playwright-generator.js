"use strict";
var PlaywrightGenerator = (() => {
  var __defProp = Object.defineProperty;
  var __defProps = Object.defineProperties;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
  var __getOwnPropNames = Object.getOwnPropertyNames;
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
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/playwright-generator.ts
  var playwright_generator_exports = {};
  __export(playwright_generator_exports, {
    RecordingManager: () => RecordingManager,
    generatePlaywrightTest: () => generatePlaywrightTest,
    generatePlaywrightTestFromActions: () => generatePlaywrightTestFromActions
  });

  // src/actionModel.ts
  function resultsToActions(results) {
    var _a, _b, _c;
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
    const navTimestampsFromClicks = /* @__PURE__ */ new Set();
    const clicks = ((_a = results.clicks) == null ? void 0 : _a.clickDetails) || [];
    for (let i = 0; i < clicks.length; i++) {
      const cd = clicks[i];
      actions.push({
        type: "click",
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
        navTimestampsFromClicks.add(nav.timestamp);
        actions.push({
          type: "waitForURL",
          timestamp: nav.timestamp - 1,
          // ensure ordering between click and nav
          expectedUrl: nav.url,
          normalizedUrl: normalize(nav.url),
          navRefTimestamp: nav.timestamp
        });
      }
    }
    for (const nav of navigations) {
      if (!navTimestampsFromClicks.has(nav.timestamp)) {
        actions.push({ type: "goto", timestamp: nav.timestamp, url: nav.url, normalizedUrl: normalize(nav.url) });
      }
    }
    if (results.inputChanges) {
      for (const input of results.inputChanges) {
        if (input.action === "complete" || !input.action) {
          actions.push({
            type: "input",
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
          type: "form",
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
          type: "assertion",
          timestamp: asrt.timestamp,
          locator: { primary: asrt.selector, fallbacks: [] },
          value: asrt.value
        });
      }
    }
    actions.sort((a, b) => a.timestamp - b.timestamp || weightOrder(a.type) - weightOrder(b.type));
    refineLocators(actions);
    for (let i = actions.length - 1; i > 0; i--) {
      const current = actions[i];
      const previous = actions[i - 1];
      if (current.type === "waitForURL" && previous.type === "waitForURL" && current.normalizedUrl === previous.normalizedUrl) {
        actions.splice(i, 1);
      }
    }
    for (let i = actions.length - 1; i > 0; i--) {
      const current = actions[i];
      const previous = actions[i - 1];
      if (current.type === "input" && previous.type === "input" && ((_b = current.locator) == null ? void 0 : _b.primary) === ((_c = previous.locator) == null ? void 0 : _c.primary) && current.value === previous.value) {
        actions.splice(i, 1);
      }
    }
    return actions;
  }
  function weightOrder(type) {
    switch (type) {
      case "click":
        return 1;
      case "waitForURL":
        return 2;
      case "goto":
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
      const key = a.locator.primary + "::" + a.type;
      if (seen.has(key) && a.locator.fallbacks.length > 0) {
        a.locator.primary = a.locator.fallbacks[0];
        a.locator.fallbacks = a.locator.fallbacks.slice(1);
      }
      seen.add(a.locator.primary + "::" + a.type);
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
            action: "complete"
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
            type: "click",
            locator: eventData.selectors || eventData.locator,
            elementInfo: eventData.elementInfo
          });
        case "nav":
        case "navigation":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "goto",
            url: eventData.url
          });
        case "input":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "input",
            value: eventData.value,
            locator: eventData.locator || { primary: eventData.selector }
          });
        case "form":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "form",
            formType: eventData.formType,
            checked: eventData.checked,
            value: eventData.value,
            locator: eventData.locator || { primary: eventData.selector }
          });
        case "assertion":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "assertion",
            value: eventData.value,
            locator: { primary: eventData.selector, fallbacks: [] }
          });
        default:
          return __spreadProps(__spreadValues({}, baseAction), {
            type: eventType
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
      switch (action.type) {
        case "click":
          this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
            (c) => c.timestamp !== timestamp
          );
          this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
          break;
        case "goto":
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
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
  function generatePlaywrightTestFromActions(actions, options = {}) {
    const { baseUrl = "" } = options;
    const needsInitialGoto = baseUrl && (actions.length === 0 || actions[0].type !== "goto");
    const initialGoto = needsInitialGoto ? `  await page.goto('${baseUrl}');
` : "";
    const body = actions.map((action) => {
      var _a, _b, _c, _d, _e;
      switch (action.type) {
        case "goto":
          return `  await page.goto('${action.url || baseUrl}');`;
        case "waitForURL":
          if (action.normalizedUrl) {
            return `  await page.waitForURL('${action.normalizedUrl}');`;
          }
          return "";
        case "click": {
          const selector = ((_a = action.locator) == null ? void 0 : _a.stableSelector) || ((_b = action.locator) == null ? void 0 : _b.primary);
          if (!selector)
            return "";
          return `  await page.click('${selector}');`;
        }
        case "input": {
          const selector = (_c = action.locator) == null ? void 0 : _c.primary;
          if (!selector || action.value === void 0)
            return "";
          const value = action.value.replace(/'/g, "\\'");
          return `  await page.fill('${selector}', '${value}');`;
        }
        case "form": {
          const selector = (_d = action.locator) == null ? void 0 : _d.primary;
          if (!selector)
            return "";
          if (action.formType === "checkbox" || action.formType === "radio") {
            if (action.checked) {
              return `  await page.check('${selector}');`;
            } else {
              return `  await page.uncheck('${selector}');`;
            }
          } else if (action.formType === "select" && action.value) {
            return `  await page.selectOption('${selector}', '${action.value}');`;
          }
          return "";
        }
        case "assertion": {
          const selector = (_e = action.locator) == null ? void 0 : _e.primary;
          if (!selector || action.value === void 0)
            return "";
          const escapedValue = escapeTextForAssertion(action.value);
          return `  await expect(page.locator('${selector}')).toContainText('${escapedValue}');`;
        }
        default:
          return "";
      }
    }).filter((line) => line !== "").join("\n");
    if (!initialGoto && !body)
      return "";
    return `import { test, expect } from '@playwright/test';

test('Recorded test', async ({ page }) => {
${initialGoto}${body.split("\n").filter((l) => l.trim()).map((l) => l).join("\n")}
});`;
  }
  function generatePlaywrightTest(url, trackingData) {
    try {
      const actions = resultsToActions(trackingData);
      return generatePlaywrightTestFromActions(actions, { baseUrl: url });
    } catch (error) {
      console.error("Error generating Playwright test:", error);
      return "";
    }
  }
  if (typeof window !== "undefined") {
    const existing = window.PlaywrightGenerator || {};
    existing.RecordingManager = RecordingManager;
    existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
    existing.generatePlaywrightTest = generatePlaywrightTest;
    window.PlaywrightGenerator = existing;
  }
  return __toCommonJS(playwright_generator_exports);
})();
