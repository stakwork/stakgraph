import { Action, ActionLocator, resultsToActions } from "./actionModel";
import { Results as TrackingResults, PlaywrightAction } from "./types";
import { getRelativeUrl } from "./utils";

export interface GenerateOptions {
  baseUrl?: string;
  testName?: string;
}

/**
 * RecordingManager - Manages test recording data in the parent/host
 * Single source of truth for all recording state
 */
export class RecordingManager {
  private trackingData: TrackingResults = {
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
    audioVideoInteractions: [],
  };

  private capturedActions: Action[] = [];
  private actionIdCounter = 0;

  /**
   * Handle an event from the iframe and store it
   */
  handleEvent(eventType: string, eventData: any): Action | null {
    // Store in trackingData
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
          timestamp: eventData.timestamp,
          seq: eventData.seq,
        });
        break;
      case "input":
        this.trackingData.inputChanges.push({
          elementSelector: eventData.selector || "",
          value: eventData.value,
          timestamp: eventData.timestamp,
          action: "complete",
          seq: eventData.seq,
        });
        break;
      case "form":
        this.trackingData.formElementChanges.push({
          elementSelector: eventData.selector || "",
          type: eventData.formType || "input",
          checked: eventData.checked,
          value: eventData.value || "",
          text: eventData.text,
          timestamp: eventData.timestamp,
          seq: eventData.seq,
        });
        break;
      case "assertion":
        this.trackingData.assertions.push({
          id: eventData.id,
          type: eventData.type || "hasText",
          selector: eventData.selector,
          value: eventData.value || "",
          timestamp: eventData.timestamp,
          seq: eventData.seq,
        });
        break;
      default:
        return null;
    }

    // Create action for UI
    const action = this.createAction(eventType, eventData);
    if (action) {
      this.capturedActions.push(action);
    }
    return action;
  }

  private createAction(eventType: string, eventData: any): Action {
    const id = `${Date.now()}_${this.actionIdCounter++}`;
    const baseAction = {
      id,
      timestamp: eventData.timestamp || Date.now(),
    };

    switch (eventType) {
      case "click":
        return {
          ...baseAction,
          type: "click",
          locator: eventData.selectors || eventData.locator,
          elementInfo: eventData.elementInfo,
        } as Action;
      case "nav":
      case "navigation":
        return {
          ...baseAction,
          type: "goto",
          url: getRelativeUrl(eventData.url),
        } as Action;
      case "input":
        return {
          ...baseAction,
          type: "input",
          value: eventData.value,
          locator: eventData.locator || { primary: eventData.selector },
        } as Action;
      case "form":
        return {
          ...baseAction,
          type: "form",
          formType: eventData.formType,
          checked: eventData.checked,
          value: eventData.value,
          locator: eventData.locator || { primary: eventData.selector },
        } as Action;
      case "assertion":
        return {
          ...baseAction,
          type: "assertion",
          value: eventData.value,
          locator: { primary: eventData.selector, fallbacks: [] },
        } as Action;
      default:
        return {
          ...baseAction,
          type: eventType,
        } as Action;
    }
  }

  /**
   * Remove an action by ID
   */
  removeAction(actionId: string): boolean {
    const action = this.capturedActions.find((a) => (a as any).id === actionId);
    if (!action) return false;

    this.capturedActions = this.capturedActions.filter((a) => (a as any).id !== actionId);
    this.removeFromTrackingData(action);
    return true;
  }

  private removeFromTrackingData(action: Action): void {
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
        // Also remove click before assertion if within 1 second
        const clickBeforeAssertion = this.trackingData.clicks.clickDetails
          .filter((c) => c.timestamp < timestamp)
          .sort((a, b) => b.timestamp - a.timestamp)[0];

        if (clickBeforeAssertion && timestamp - clickBeforeAssertion.timestamp < 1000) {
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
  generateTest(url: string, options?: any): string {
    const actions = resultsToActions(this.trackingData);
    return generatePlaywrightTestFromActions(actions, {
      baseUrl: url,
      ...options,
    });
  }

  /**
   * Structured replay steps for the CURRENT recording — the executor consumes these
   * directly (no generated text, no re-parse). Single source of truth = trackingData.
   */
  getReplaySteps(url: string): PlaywrightAction[] {
    const actions = resultsToActions(this.trackingData);
    return actionsToReplaySteps(actions, { baseUrl: url });
  }

  /**
   * Get current actions for UI display
   */
  getActions(): Action[] {
    return [...this.capturedActions];
  }

  /**
   * Get tracking data (for compatibility)
   */
  getTrackingData(): TrackingResults {
    return this.trackingData;
  }

  /**
   * Clear all recorded data
   */
  clear(): void {
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
      audioVideoInteractions: [],
    };
    this.capturedActions = [];
    this.actionIdCounter = 0;
  }

  /**
   * Clear all actions (but keep recording)
   */
  clearAllActions(): void {
    this.clear();
  }
}

// Safely embed an arbitrary string inside a single-quoted JS string literal.
// Order matters: escape backslashes first, then quotes/newlines. Fixes the round-trip
// corruption where fill values like O'Brien became O\'Brien (design doc bug #4).
function q(s: string | undefined): string {
  return String(s ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/\r/g, "")
    .replace(/\n/g, "\\n");
}

// Pull a data-testid out of a selector like [data-testid="save"] if present.
function extractTestId(locator?: ActionLocator): string | null {
  if (!locator) return null;
  for (const cand of [locator.primary, ...(locator.fallbacks || [])]) {
    const m = cand && cand.match(/\[data-testid=["']([^"']+)["']\]/);
    if (m) return m[1];
  }
  return null;
}

// Locator expression for CLICK targets. Prefer getByTestId (stable across DOM
// changes) over raw CSS, which is prone to resolving to the wrong element at replay
// time (design doc bugs #1/#2). Only clicks get native selectors today — extending
// this to value-bearing actions (fill/check/selectOption/assert) is tracked separately.
function clickExpr(locator?: ActionLocator): string {
  const testId = extractTestId(locator);
  if (testId) return `page.getByTestId('${q(testId)}')`;
  const sel = locator?.stableSelector || locator?.primary || "";
  // Capture emits a `role:button[name="Save"]` DSL for elements identified by role +
  // accessible name. Convert to a real getByRole() — valid Playwright and parseable —
  // rather than wrapping the DSL in page.locator(), which is invalid Playwright.
  const roleM = sel.match(/^role:([a-zA-Z]+)(?:\[name(?:-regex)?="([\s\S]*)"\])?$/);
  if (roleM) {
    const [, role, name] = roleM;
    return name
      ? `page.getByRole('${q(role)}', { name: '${q(name)}' })`
      : `page.getByRole('${q(role)}')`;
  }
  return cssLocator(locator);
}

// Plain CSS locator for value-bearing actions (fill / check / selectOption / assert).
// These don't yet get native getByTestId/getByRole treatment — see clickExpr above.
function cssLocator(locator?: ActionLocator): string {
  const sel = locator?.stableSelector || locator?.primary || "";
  return `page.locator('${q(sel)}')`;
}

export function generatePlaywrightTestFromActions(
  actions: Action[],
  options: { baseUrl?: string } = {}
): string {
  const { baseUrl = "" } = options;

  // Generate initial goto if we have a baseUrl and no goto action at the start
  const needsInitialGoto = baseUrl && (actions.length === 0 || actions[0].type !== 'goto');
  const initialGoto = needsInitialGoto ? `  await page.goto('${baseUrl}');\n` : '';

  const body = actions
    .map((action) => {
      switch (action.type) {
        case "goto":
          return `  await page.goto('${getRelativeUrl(action.url || baseUrl)}');`;
        case "waitForURL":
          if (action.normalizedUrl) {
            return `  await page.waitForURL('${action.normalizedUrl}');`;
          }
          return "";
        case "click": {
          if (!action.locator?.primary && !action.locator?.stableSelector) return "";
          return `  await ${clickExpr(action.locator)}.click();`;
        }
        case "input": {
          if (!action.locator?.primary || action.value === undefined) return "";
          return `  await ${cssLocator(action.locator)}.fill('${q(action.value)}');`;
        }
        case "form": {
          if (!action.locator?.primary) return "";
          const target = cssLocator(action.locator);
          if (action.formType === "checkbox" || action.formType === "radio") {
            return action.checked ? `  await ${target}.check();` : `  await ${target}.uncheck();`;
          } else if (action.formType === "select" && action.value !== undefined) {
            return `  await ${target}.selectOption('${q(action.value)}');`;
          }
          return "";
        }
        case "assertion": {
          if (!action.locator?.primary || action.value === undefined) return "";
          return `  await expect(${cssLocator(action.locator)}).toContainText('${q(action.value)}');`;
        }
        default:
          return "";
      }
    })
    .filter((line) => line !== "")
    .join("\n");

  if (!initialGoto && !body) return "";

  return `import { test, expect } from '@playwright/test';

test('Recorded test', async ({ page }) => {
${initialGoto}${body
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l)
  .join("\n")}
});`;
}

// ---------------------------------------------------------------------------
// Structured replay (P3): convert the recorded Action[] straight into the
// executor's PlaywrightAction[] — no Playwright text, no regex re-parse. The
// selector strings use the executor's DSL (getByTestId:/role:/text=/CSS), which
// findElementWithFallbacks understands directly. This is the source of truth the
// preview replay should consume instead of round-tripping through generated text.
// ---------------------------------------------------------------------------

// Selector string in the format the in-iframe executor resolves.
function executorSelector(locator?: ActionLocator): string {
  if (!locator) return "";
  const sel = locator.stableSelector || locator.primary || "";
  // DSL forms the executor resolves natively (survive DOM reordering).
  if (sel.startsWith("role:") || sel.startsWith("text=")) return sel;
  const testId = extractTestId(locator);
  if (testId) return `getByTestId:${testId}`;
  return sel; // CSS / attribute / id / class
}

export function actionsToReplaySteps(
  actions: Action[],
  options: { baseUrl?: string } = {}
): PlaywrightAction[] {
  const { baseUrl } = options;
  const steps: PlaywrightAction[] = [];

  // Leading goto so replay resets to the recorded start (executor turns this into
  // an SPA route reset). Mirrors generatePlaywrightTestFromActions.
  if (baseUrl && (actions.length === 0 || actions[0].type !== "goto")) {
    steps.push({ type: "goto", value: baseUrl });
  }

  for (const a of actions) {
    switch (a.type) {
      case "goto":
        steps.push({ type: "goto", value: getRelativeUrl(a.url || baseUrl || "") });
        break;
      case "waitForURL":
        if (a.normalizedUrl) steps.push({ type: "waitForURL", value: a.normalizedUrl });
        break;
      case "click": {
        const selector = executorSelector(a.locator);
        if (selector) steps.push({ type: "click", selector });
        break;
      }
      case "input": {
        const selector = executorSelector(a.locator);
        if (selector && a.value !== undefined) steps.push({ type: "fill", selector, value: a.value });
        break;
      }
      case "form": {
        const selector = executorSelector(a.locator);
        if (!selector) break;
        if (a.formType === "checkbox" || a.formType === "radio") {
          steps.push({ type: a.checked ? "check" : "uncheck", selector });
        } else if (a.formType === "select" && a.value !== undefined) {
          steps.push({ type: "selectOption", selector, value: a.value });
        }
        break;
      }
      case "assertion": {
        const selector = executorSelector(a.locator);
        if (selector && a.value !== undefined) {
          steps.push({ type: "expect", selector, expectation: "toContainText", value: a.value });
        }
        break;
      }
    }
  }
  return steps;
}

export function generatePlaywrightTest(url: string, trackingData: TrackingResults): string {
  try {
    const actions = resultsToActions(trackingData);
    return generatePlaywrightTestFromActions(actions, { baseUrl: url });
  } catch (error) {
    console.error("Error generating Playwright test:", error);
    return "";
  }
}

// Export to window for hooks.js to use
if (typeof window !== "undefined") {
  const existing = (window as any).PlaywrightGenerator || {};
  existing.RecordingManager = RecordingManager;
  existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
  existing.generatePlaywrightTest = generatePlaywrightTest;
  existing.actionsToReplaySteps = actionsToReplaySteps;
  (window as any).PlaywrightGenerator = existing;
}
