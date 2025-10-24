import type { Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Test Helpers for staktrak test suite
 * Provides utilities for browser-based testing of staktrak functionality
 */

let staktrakBundle: string | null = null;

/**
 * Get the inlined staktrak bundle content
 */
export function getStaktrakBundle(): string {
  if (!staktrakBundle) {
    const bundlePath = path.join(process.cwd(), 'tests/staktrak/dist/staktrak.js');
    staktrakBundle = fs.readFileSync(bundlePath, 'utf-8');
  }
  return staktrakBundle;
}

/**
 * Get the path to staktrak bundle
 */
export function getStaktrakBundlePath(): string {
  return '/dist/staktrak.js';
}

/**
 * Get the path to playwright-generator bundle
 */
export function getPlaywrightGeneratorPath(): string {
  return '/dist/playwright-generator.js';
}

/**
 * Create test HTML page with staktrak loaded
 */
export function createTestPage(options: {
  includeStaktrak?: boolean;
  includeConfig?: boolean;
  parentOrigin?: string;
  customContent?: string;
} = {}): string {
  const { includeStaktrak = true, includeConfig = false, parentOrigin = 'http://localhost:3000', customContent = '' } = options;

  const configScript = includeConfig
    ? `
    <script>
      window.STAKTRAK_CONFIG = {
        parentOrigin: '${parentOrigin}',
        screenshot: {
          quality: 0.8,
          type: 'image/jpeg',
          scale: 1,
          backgroundColor: '#ffffff'
        }
      };
    </script>
    `
    : '';

  // Inline the staktrak bundle for data URLs
  const staktrakScript = includeStaktrak
    ? `<script>${getStaktrakBundle()}</script>`
    : '';

  const defaultContent = `
    <div id="app">
      <button data-testid="test-button">Click Me</button>
      <input data-testid="test-input" type="text" placeholder="Enter text">
      <input data-testid="test-checkbox" type="checkbox">
      <select data-testid="test-select">
        <option value="option1">Option 1</option>
        <option value="option2">Option 2</option>
      </select>
      <div data-testid="test-result">Result</div>
    </div>
  `;

  return `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Test Page</title>
        ${configScript}
      </head>
      <body>
        ${customContent || defaultContent}
        ${staktrakScript}
      </body>
    </html>
  `;
}

/**
 * Load staktrak in a page and wait for it to be ready
 */
export async function loadStaktrakInPage(page: Page, options: {
  includeConfig?: boolean;
  parentOrigin?: string;
  customContent?: string;
} = {}): Promise<void> {
  const html = createTestPage({ includeStaktrak: true, ...options });
  const dataUrl = `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;

  await page.goto(dataUrl);

  // Wait for staktrak to be fully initialized
  await page.waitForFunction(() => {
    const ub = (window as any).userBehaviour;
    return ub !== undefined &&
           typeof ub === 'object' &&
           typeof ub.start === 'function' &&
           typeof ub.stop === 'function' &&
           typeof ub.result === 'function';
  }, { timeout: 10000 });
}

/**
 * Start staktrak recording in page
 */
export async function startRecording(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).userBehaviour.start();
  });
}

/**
 * Stop staktrak recording in page
 */
export async function stopRecording(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).userBehaviour.stop();
  });
}

/**
 * Get recorded actions from staktrak
 */
export async function getActions(page: Page): Promise<any[]> {
  return await page.evaluate(() => {
    return (window as any).userBehaviour.getActions();
  });
}

/**
 * Get raw tracking results from staktrak
 */
export async function getResults(page: Page): Promise<any> {
  return await page.evaluate(() => {
    return (window as any).userBehaviour.result();
  });
}

/**
 * Generate Playwright test code from recorded actions
 */
export async function generatePlaywrightTest(page: Page, options: any = {}): Promise<string> {
  return await page.evaluate((opts) => {
    return (window as any).userBehaviour.generatePlaywrightTest(opts);
  }, options);
}

/**
 * Clear all recorded actions
 */
export async function clearAllActions(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).userBehaviour.memory.assertions = [];
    (window as any).userBehaviour.results = (window as any).userBehaviour.createEmptyResults();
  });
}

/**
 * Verify screenshot data URL has valid format
 */
export function verifyScreenshotDataUrl(dataUrl: string): boolean {
  if (!dataUrl || typeof dataUrl !== 'string') {
    return false;
  }

  // Check if it's a valid data URL
  const dataUrlPattern = /^data:image\/(png|jpeg|jpg|webp);base64,/;
  if (!dataUrlPattern.test(dataUrl)) {
    return false;
  }

  // Check if base64 content exists
  const base64Content = dataUrl.split(',')[1];
  if (!base64Content || base64Content.length === 0) {
    return false;
  }

  // Validate base64 format (test first 100 chars)
  try {
    // In Node.js context
    if (typeof atob === 'undefined') {
      Buffer.from(base64Content.substring(0, 100), 'base64');
    } else {
      atob(base64Content.substring(0, 100));
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate action structure
 */
export function validateAction(action: any): boolean {
  if (!action || typeof action !== 'object') {
    return false;
  }

  // Must have type field (not kind)
  if (!action.type || typeof action.type !== 'string') {
    return false;
  }

  // Must have timestamp
  if (!action.timestamp || typeof action.timestamp !== 'number') {
    return false;
  }

  // Type-specific validation
  switch (action.type) {
    case 'click':
      return !!action.locator;
    case 'goto':
      return !!action.url;
    case 'waitForURL':
      return !!action.expectedUrl || !!action.normalizedUrl;
    case 'input':
      return !!action.locator && action.value !== undefined;
    case 'form':
      return !!action.locator && !!action.formType;
    case 'assertion':
      return !!action.locator && action.value !== undefined;
    default:
      return true;
  }
}

/**
 * Wait for condition with timeout (page context)
 */
export async function waitForCondition(
  page: Page,
  condition: () => boolean | Promise<boolean>,
  timeout: number = 5000,
  interval: number = 100
): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const result = await Promise.resolve(condition());
    if (result) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  return false;
}

/**
 * Create mock click event in page
 */
export async function simulateClick(page: Page, selector: string): Promise<void> {
  await page.click(selector);
  // Wait for staktrak to process the event
  await page.waitForTimeout(100);
}

/**
 * Create mock navigation in page
 */
export async function simulateNavigation(page: Page, url: string): Promise<void> {
  await page.evaluate((targetUrl) => {
    history.pushState({}, '', targetUrl);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, url);
  await page.waitForTimeout(100);
}

/**
 * Create mock input event in page
 */
export async function simulateInput(page: Page, selector: string, value: string): Promise<void> {
  await page.fill(selector, value);
  // Wait for debounce
  await page.waitForTimeout(2100);
}

/**
 * Create mock form event in page
 */
export async function simulateFormChange(page: Page, selector: string, checked?: boolean): Promise<void> {
  if (checked !== undefined) {
    await page.check(selector);
  } else {
    await page.click(selector);
  }
  await page.waitForTimeout(100);
}

/**
 * Setup postMessage listener in page
 */
export async function setupPostMessageListener(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__testMessages = [];
    window.addEventListener('message', (event) => {
      (window as any).__testMessages.push({
        type: event.data.type,
        data: event.data
      });
    });
  });
}

/**
 * Get captured postMessages
 */
export async function getPostMessages(page: Page, type?: string): Promise<any[]> {
  return await page.evaluate((messageType) => {
    const messages = (window as any).__testMessages || [];
    if (messageType) {
      return messages.filter((msg: any) => msg.type === messageType);
    }
    return messages;
  }, type);
}

/**
 * Clear captured postMessages
 */
export async function clearPostMessages(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__testMessages = [];
  });
}

/**
 * Extract screenshot messages from postMessages
 */
export function extractScreenshotMessages(messages: Array<{ data: any }>): any[] {
  return messages
    .filter(msg => msg.data && msg.data.type === 'staktrak-playwright-screenshot-captured')
    .map(msg => msg.data);
}

/**
 * Validate screenshot message structure
 */
export function validateScreenshotMessage(message: any): boolean {
  if (!message || typeof message !== 'object') {
    return false;
  }

  const requiredFields = ['type', 'screenshot', 'actionIndex', 'url', 'timestamp', 'id'];
  for (const field of requiredFields) {
    if (!(field in message)) {
      return false;
    }
  }

  if (message.type !== 'staktrak-playwright-screenshot-captured') {
    return false;
  }

  if (!verifyScreenshotDataUrl(message.screenshot)) {
    return false;
  }

  return true;
}

/**
 * Wait for staktrak to be ready
 */
export async function waitForStaktrakReady(page: Page, timeout: number = 10000): Promise<void> {
  await page.waitForFunction(() => {
    const ub = (window as any).userBehaviour;
    return ub !== undefined &&
           typeof ub === 'object' &&
           typeof ub.start === 'function' &&
           typeof ub.stop === 'function' &&
           typeof ub.result === 'function';
  }, { timeout });
}

/**
 * Check if action uses old field names (should return false for new code)
 */
export function hasOldFieldNames(action: any): boolean {
  return 'kind' in action || action.type === 'nav' || action.type === 'waitForUrl';
}

/**
 * Verify all actions use new naming convention
 */
export function verifyNewNamingConvention(actions: any[]): boolean {
  return actions.every(action => {
    // Should have 'type' not 'kind'
    if (!action.type || 'kind' in action) {
      return false;
    }

    // Should use 'goto' not 'nav'
    if (action.type === 'nav') {
      return false;
    }

    // Should use 'waitForURL' not 'waitForUrl'
    if (action.type === 'waitForUrl') {
      return false;
    }

    return true;
  });
}
