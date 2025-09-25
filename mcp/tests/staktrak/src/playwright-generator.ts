import { Action, resultsToActions } from "./actionModel";
import { Results as TrackingResults } from "./types";

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
    audioVideoInteractions: []
  };

  private capturedActions: Action[] = [];
  private actionIdCounter = 0;

  /**
   * Handle an event from the iframe and store it
   */
  handleEvent(eventType: string, eventData: any): Action | null {
    // Store in trackingData
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
          locator: { primary: eventData.selector, fallbacks: [] }
        } as Action;
      default:
        return {
          ...baseAction,
          kind: eventType
        } as Action;
    }
  }

  /**
   * Remove an action by ID
   */
  removeAction(actionId: string): boolean {
    const action = this.capturedActions.find(a => (a as any).id === actionId);
    if (!action) return false;

    this.capturedActions = this.capturedActions.filter(a => (a as any).id !== actionId);
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
        // Also remove click before assertion if within 1 second
        const clickBeforeAssertion = this.trackingData.clicks.clickDetails
          .filter(c => c.timestamp < timestamp)
          .sort((a, b) => b.timestamp - a.timestamp)[0];

        if (clickBeforeAssertion && timestamp - clickBeforeAssertion.timestamp < 1000) {
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
  generateTest(url: string, options?: any): string {
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
  if (!text) return '';
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function generatePlaywrightTestFromActions(
  actions: Action[],
  options: { baseUrl?: string } = {}
): string {
  const { baseUrl = '' } = options;
  const body = actions
    .map((action) => {
      switch (action.kind) {
        case 'nav':
          return `  await page.goto('${action.url || baseUrl}');`;
        case 'waitForUrl':
          if (action.normalizedUrl) {
            return `  await page.waitForURL('${action.normalizedUrl}');`;
          }
          return '';
        case 'click': {
          const selector = action.locator?.stableSelector || action.locator?.primary;
          if (!selector) return '';
          return `  await page.click('${selector}');`;
        }
        case 'input': {
          const selector = action.locator?.primary;
          if (!selector || action.value === undefined) return '';
          const value = action.value.replace(/'/g, "\\'");
          return `  await page.fill('${selector}', '${value}');`;
        }
        case 'form': {
          const selector = action.locator?.primary;
          if (!selector) return '';
          if (action.formType === 'checkbox' || action.formType === 'radio') {
            if (action.checked) {
              return `  await page.check('${selector}');`;
            } else {
              return `  await page.uncheck('${selector}');`;
            }
          } else if (action.formType === 'select' && action.value) {
            return `  await page.selectOption('${selector}', '${action.value}');`;
          }
          return '';
        }
        case 'assertion': {
          const selector = action.locator?.primary;
          if (!selector || action.value === undefined) return '';
          const escapedValue = escapeTextForAssertion(action.value);
          return `  await expect(page.locator('${selector}')).toContainText('${escapedValue}');`;
        }
        default:
          return '';
      }
    })
    .filter((line) => line !== '')
    .join('\n');

  if (!body) return '';

  return `import { test, expect } from '@playwright/test';

test('Recorded test', async ({ page }) => {
${body.split('\n').filter(l => l.trim()).map(l => l).join('\n')}
});`;
}

// Export to window for hooks.js to use
if (typeof window !== 'undefined') {
  const existing = (window as any).PlaywrightGenerator || {};
  existing.RecordingManager = RecordingManager;
  existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
  existing.generatePlaywrightTest = (url: string, trackingData: TrackingResults) => {
    try {
      const actions = resultsToActions(trackingData);
      return generatePlaywrightTestFromActions(actions, { baseUrl: url });
    } catch (error) {
      console.error('Error generating Playwright test:', error);
      return '';
    }
  };
  (window as any).PlaywrightGenerator = existing;
}


