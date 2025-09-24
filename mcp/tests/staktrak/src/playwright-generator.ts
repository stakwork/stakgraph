import { Action, ActionLocator, resultsToActions } from "./actionModel";
import { Results as TrackingResults, ClickDetail, Assertion } from "./types";

/**
 * RecordingManager - Manages test recording data in the parent/host
 * Maintains both trackingData (for compatibility) and actions (for UI)
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
    audioVideoInteractions: []
  };

  private capturedActions: Action[] = [];
  private actionIdCounter = 0;

  /**
   * Handle an event from the iframe and store it
   */
  handleEvent(eventType: string, eventData: any): Action | null {
    // Store in trackingData for backward compatibility
    switch(eventType) {
      case 'click':
        this.trackingData.clicks.clickDetails.push(eventData);
        this.trackingData.clicks.clickCount++;
        break;
      case 'nav':
      case 'navigation':
        this.trackingData.pageNavigation.push({
          type: 'navigation',
          url: eventData.url,
          timestamp: eventData.timestamp
        });
        break;
      case 'input':
        this.trackingData.inputChanges.push({
          elementSelector: eventData.selector || '',
          value: eventData.value,
          timestamp: eventData.timestamp,
          action: 'fill'
        });
        break;
      case 'form':
        this.trackingData.formElementChanges.push({
          elementSelector: eventData.selector || '',
          type: eventData.formType || 'input',
          checked: eventData.checked,
          value: eventData.value || '',
          text: eventData.text,
          timestamp: eventData.timestamp
        });
        break;
      case 'assertion':
        this.trackingData.assertions.push({
          id: eventData.id,
          type: eventData.type || 'hasText',
          selector: eventData.selector,
          value: eventData.value || '',
          timestamp: eventData.timestamp
        });
        break;
      default:
        // Unknown event type, ignore
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
      timestamp: eventData.timestamp || Date.now()
    };

    switch(eventType) {
      case 'click':
        return {
          ...baseAction,
          kind: 'click',
          locator: eventData.selectors || eventData.locator,
          elementInfo: eventData.elementInfo
        } as Action;

      case 'nav':
      case 'navigation':
        return {
          ...baseAction,
          kind: 'nav',
          url: eventData.url
        } as Action;

      case 'input':
        return {
          ...baseAction,
          kind: 'input',
          value: eventData.value,
          locator: eventData.locator || { primary: eventData.selector }
        } as Action;

      case 'form':
        return {
          ...baseAction,
          kind: 'form',
          formType: eventData.formType,
          checked: eventData.checked,
          value: eventData.value,
          locator: eventData.locator || { primary: eventData.selector }
        } as Action;

      case 'assertion':
        return {
          ...baseAction,
          kind: 'assertion',
          value: eventData.value,
          locator: { primary: eventData.selector }
        } as Action;

      default:
        return {
          ...baseAction,
          kind: eventType as any
        } as Action;
    }
  }

  /**
   * Remove an action by ID
   */
  removeAction(actionId: string): boolean {
    const action = this.capturedActions.find(a => a.id === actionId);
    if (!action) return false;

    // Remove from capturedActions
    this.capturedActions = this.capturedActions.filter(a => a.id !== actionId);

    // Remove from trackingData based on timestamp
    this.removeFromTrackingData(action);
    return true;
  }

  private removeFromTrackingData(action: Action): void {
    const timestamp = action.timestamp;

    switch(action.kind) {
      case 'click':
        this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
          c => c.timestamp !== timestamp
        );
        this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
        break;

      case 'nav':
        this.trackingData.pageNavigation = this.trackingData.pageNavigation.filter(
          n => n.timestamp !== timestamp
        );
        break;

      case 'input':
        this.trackingData.inputChanges = this.trackingData.inputChanges.filter(
          i => i.timestamp !== timestamp
        );
        break;

      case 'form':
        this.trackingData.formElementChanges = this.trackingData.formElementChanges.filter(
          f => f.timestamp !== timestamp
        );
        break;

      case 'assertion':
        this.trackingData.assertions = this.trackingData.assertions.filter(
          a => a.timestamp !== timestamp
        );
        // Also remove the spurious click that may have triggered the assertion
        const clickBeforeAssertion = this.trackingData.clicks.clickDetails
          .filter(c => c.timestamp < timestamp)
          .sort((a, b) => b.timestamp - a.timestamp)[0];

        if (clickBeforeAssertion && (timestamp - clickBeforeAssertion.timestamp) < 1000) {
          this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
            c => c.timestamp !== clickBeforeAssertion.timestamp
          );
          this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
        }
        break;
    }
  }

  /**
   * Generate Playwright test from current data
   */
  generateTest(url: string, options?: Partial<GenerateOptions>): string {
    // Convert trackingData to actions and generate test
    const actions = resultsToActions(this.trackingData);
    return generatePlaywrightTestFromActions(actions, {
      baseUrl: url,
      ...options
    });
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
      audioVideoInteractions: []
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

function escapeTextForAssertion(text: string): string {
  if (!text) return "";
  return text
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t")
    .trim();
}

function normalizeText(t?: string) {
  return (t || "").trim();
}

function locatorToSelector(l: ActionLocator): string {
  if (!l) return 'page.locator("body")';
  const primary = l.stableSelector || l.primary;
  if (/\[data-testid=/.test(primary)) {
    const m = primary.match(/\[data-testid=["']([^"']+)["']\]/);
    if (m) return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
  }
  if (primary.startsWith("#") && /^[a-zA-Z][\w-]*$/.test(primary.slice(1)))
    return `page.locator('${primary}')`;
  // Prefer explicit structural class/attribute selector over role/text if present
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
    if (fb && !/^[a-zA-Z]+$/.test(fb)) return `page.locator('${fb}')`;
  }
  return 'page.locator("body")';
}

export interface GenerateOptions {
  baseUrl: string;
  viewport?: { width: number; height: number };
  testName?: string;
}

export function generatePlaywrightTestFromActions(
  actions: Action[],
  options: GenerateOptions
): string {
  const name = options.testName || "Recorded flow";
  const viewport = options.viewport || { width: 1280, height: 720 };
  let body = "";
  let lastTs: number | null = null;
  const base = options.baseUrl ? options.baseUrl.replace(/\/$/, "") : "";

  function fullUrl(u?: string) {
    if (!u) return "";
    if (/^https?:/i.test(u)) return u;
    if (u.startsWith('/')) return base + u;
    return base + '/' + u;
  }

  let i = 0;
  // collapse consecutive nav actions to the same URL
  const collapsed: Action[] = [];
  for (let k = 0; k < actions.length; k++) {
    const curr = actions[k];
    const prev = collapsed[collapsed.length - 1];
    if (curr.kind === 'nav' && prev && prev.kind === 'nav' && prev.url === curr.url) continue;
    collapsed.push(curr);
  }
  actions = collapsed;
  while (i < actions.length) {
    const a = actions[i];
    // Correlate click->nav within 1500ms
  if (a.kind === 'click' && i + 1 < actions.length) {
      const nxt = actions[i + 1];
      if (nxt.kind === 'nav' && (nxt.timestamp - a.timestamp) < 1500) {
        // optional pre-wait before click if gap from previous
        if (lastTs != null) {
          const delta = Math.max(0, a.timestamp - lastTs);
            const wait = Math.min(3000, Math.max(100, delta));
            if (wait > 400) body += `  await page.waitForTimeout(${wait});\n`;
        }
        body += `  await Promise.all([\n`;
        body += `    page.waitForURL('${fullUrl(nxt.url)}'),\n`;
        body += `    ${locatorToSelector(a.locator!)}.click()\n`;
        body += `  ]);\n`;
        lastTs = nxt.timestamp;
        i += 2;
        continue;
      }
    }

    if (lastTs != null) {
      const delta = Math.max(0, a.timestamp - lastTs);
      const wait = Math.min(3000, Math.max(100, delta));
      if (wait > 500) body += `  await page.waitForTimeout(${wait});\n`;
    }

    switch (a.kind) {
      case 'nav': {
        const target = fullUrl(a.url);
        if (i === 0) {
          body += `  await page.goto('${target}');\n`;
        } else {
          body += `  await page.waitForURL('${target}');\n`;
        }
        break;
      }
      case 'click':
        body += `  await ${locatorToSelector(a.locator!)}.click();\n`;
        break;
      case 'input':
        body += `  await ${locatorToSelector(a.locator!)}.fill('${escapeTextForAssertion(a.value || '')}');\n`;
        break;
      case 'form':
        if (a.formType === 'checkbox' || a.formType === 'radio') {
          body += a.checked
            ? `  await ${locatorToSelector(a.locator!)}.check();\n`
            : `  await ${locatorToSelector(a.locator!)}.uncheck();\n`;
        } else if (a.formType === 'select') {
          body += `  await ${locatorToSelector(a.locator!)}.selectOption('${escapeTextForAssertion(a.value || '')}');\n`;
        }
        break;
      case 'assertion':
        if (a.value && a.value.length > 0) {
          body += `  await expect(${locatorToSelector(a.locator!)}).toContainText('${escapeTextForAssertion(a.value)}');\n`;
        } else {
          body += `  await expect(${locatorToSelector(a.locator!)}).toBeVisible();\n`;
        }
        break;
    }
    lastTs = a.timestamp;
    i++;
  }
  return `import { test, expect } from '@playwright/test'

test('${name}', async ({ page }) => {
  await page.setViewportSize({ width: ${viewport.width}, height: ${viewport.height} })
${body
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l)
  .join("\n")}
})
`;
}

if (typeof window !== "undefined") {
  const existing = (window as any).PlaywrightGenerator || {};
  existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
  existing.generatePlaywrightTest = (baseUrl: string, results: any) => {
    try {
      const actions = resultsToActions(results);
      return generatePlaywrightTestFromActions(actions, { baseUrl });
    } catch (e) {
      console.warn('PlaywrightGenerator.generatePlaywrightTest failed', e);
      return '';
    }
  };
  existing.RecordingManager = RecordingManager;
  (window as any).PlaywrightGenerator = existing;
}
