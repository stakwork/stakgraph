import { test, expect } from '@playwright/test';
import {
  createTestPage,
  waitForCondition,
  extractScreenshotMessages,
  validateScreenshotMessage,
} from './test-helpers';

test.describe('Playwright Replay Integration', () => {
  test.describe('Basic Replay Flow', () => {
    let messages: any[];

    test.beforeEach(async ({ page }) => {
      messages = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });
    });

    test('should start replay and send started message', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('basic test', async ({ page }) => {
          await page.goto('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-started'),
        3000
      );

      const startedMsg = messages.find(m => m.type === 'staktrak-playwright-replay-started');
      expect(startedMsg).toBeTruthy();
      expect(startedMsg.totalActions).toBeGreaterThan(0);
      expect(startedMsg.actions).toBeTruthy();
      expect(Array.isArray(startedMsg.actions)).toBe(true);
    });

    test('should send progress messages during replay', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="test-button"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-progress'),
        3000
      );

      const progressMsgs = messages.filter(m => m.type === 'staktrak-playwright-replay-progress');
      expect(progressMsgs.length).toBeGreaterThan(0);

      const progressMsg = progressMsgs[0];
      expect(progressMsg.current).toBeGreaterThan(0);
      expect(progressMsg.total).toBeGreaterThan(0);
      expect(progressMsg.currentAction).toBeTruthy();
      expect(progressMsg.currentAction.description).toBeTruthy();
    });

    test('should send completed message when done', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="test-button"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      const completed = await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        5000
      );

      expect(completed).toBe(true);
    });
  });

  test.describe('Replay with Screenshots', () => {
    let messages: any[];

    test.beforeEach(async ({ page }) => {
      messages = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });
    });

    test('should capture screenshots during replay with waitForURL', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.waitForURL('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        8000
      );

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      // Should have at least one screenshot
      expect(screenshotMsgs.length).toBeGreaterThan(0);

      // Validate screenshot structure
      screenshotMsgs.forEach(msg => {
        expect(validateScreenshotMessage(msg)).toBe(true);
      });
    });

    test('should include correct actionIndex in screenshots', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.waitForURL('http://localhost:3000');
          await page.click('[data-testid="test-button"]');
          await page.waitForURL('http://localhost:3000/next');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        8000
      );

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      screenshotMsgs.forEach(msg => {
        expect(msg.actionIndex).toBeGreaterThanOrEqual(0);
        expect(typeof msg.actionIndex).toBe('number');
        expect(msg.url).toBeTruthy();
        expect(msg.timestamp).toBeGreaterThan(0);
      });
    });

    test('should include URL in screenshot message', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        5000
      );

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      if (screenshotMsgs.length > 0) {
        const msg = screenshotMsgs[0];
        expect(msg.url).toBeTruthy();
        expect(typeof msg.url).toBe('string');
        expect(msg.url).toContain('http');
      }
    });
  });

  test.describe('Mixed Action Types', () => {
    let messages: any[];

    test.beforeEach(async ({ page }) => {
      messages = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });
    });

    test('should handle replay with multiple action types', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.click('[data-testid="test-button"]');
          await page.fill('[data-testid="test-input"]', 'test value');
          await page.check('[data-testid="test-checkbox"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      const completed = await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        5000
      );

      expect(completed).toBe(true);

      const progressMsgs = messages.filter(m => m.type === 'staktrak-playwright-replay-progress');
      expect(progressMsgs.length).toBeGreaterThan(0);
    });

    test('should only capture screenshots after waitForURL actions', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="test-button"]');
          await page.fill('[data-testid="test-input"]', 'test');
          await page.waitForURL('http://localhost:3000');
          await page.check('[data-testid="test-checkbox"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        5000
      );

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));
      const progressMsgs = messages.filter(m => m.type === 'staktrak-playwright-replay-progress');

      // Should have screenshots (from waitForURL)
      // Progress messages should include all action types
      expect(progressMsgs.length).toBeGreaterThan(screenshotMsgs.length);
    });
  });

  test.describe('Error Handling', () => {
    let messages: any[];

    test.beforeEach(async ({ page }) => {
      messages = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });
    });

    test('should halt replay on action errors (does not run later steps)', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      // First step fails; the second step must NOT run because replay halts.
      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="nonexistent-button"]');
          await page.click('[data-testid="test-button"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      const errored = await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-error'),
        5000
      );
      expect(errored).toBe(true);

      // The failing step is index 0, so replay must stop there and never
      // reach — or complete — the second step.
      const errorMsgs = messages.filter(m => m.type === 'staktrak-playwright-replay-error');
      expect(errorMsgs.length).toBe(1);
      expect(errorMsgs[0].fatal).toBe(true);
      expect(errorMsgs[0].actionIndex).toBe(0);
      expect(messages.some(m => m.type === 'staktrak-playwright-replay-completed')).toBe(false);
    });

    test('should report error details in error messages', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="nonexistent"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-error'),
        3000
      );

      const errorMsgs = messages.filter(m => m.type === 'staktrak-playwright-replay-error');
      expect(errorMsgs.length).toBeGreaterThan(0);

      const errorMsg = errorMsgs[0];
      expect(errorMsg.error).toBeTruthy();
      expect(typeof errorMsg.error).toBe('string');
    });

    test('should capture a screenshot of the failure frame when a step errors', async ({ page }) => {
      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="nonexistent"]');
          await page.click('[data-testid="test-button"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-error'),
        8000
      );

      const errorMsgs = messages.filter(m => m.type === 'staktrak-playwright-replay-error');
      expect(errorMsgs.length).toBe(1);
      expect(errorMsgs[0].fatal).toBe(true);

      // A failure screenshot should have been captured and referenced by id on
      // the error message.
      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));
      expect(screenshotMsgs.length).toBeGreaterThan(0);
      expect(errorMsgs[0].screenshotId).toBeTruthy();
      expect(messages.some(m => m.type === 'staktrak-playwright-replay-completed')).toBe(false);
    });
  });

  test.describe('Replay Control', () => {
    test('should handle pause/resume', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.click('[data-testid="test-button"]');
          await page.fill('[data-testid="test-input"]', 'test');
        });
      `;

      // Start replay
      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      // Wait a bit then pause
      await page.waitForTimeout(500);

      await page.evaluate(() => {
        window.postMessage({ type: 'staktrak-playwright-replay-pause' }, '*');
      });

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-paused'),
        2000
      );

      const pausedMsg = messages.find(m => m.type === 'staktrak-playwright-replay-paused');
      expect(pausedMsg).toBeTruthy();

      // Resume
      await page.evaluate(() => {
        window.postMessage({ type: 'staktrak-playwright-replay-resume' }, '*');
      });

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-resumed'),
        2000
      );

      const resumedMsg = messages.find(m => m.type === 'staktrak-playwright-replay-resumed');
      expect(resumedMsg).toBeTruthy();
    });

    test('should handle stop command', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.click('[data-testid="test-button"]');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(300);

      // Stop replay
      await page.evaluate(() => {
        window.postMessage({ type: 'staktrak-playwright-replay-stop' }, '*');
      });

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-stopped'),
        2000
      );

      const stoppedMsg = messages.find(m => m.type === 'staktrak-playwright-replay-stopped');
      expect(stoppedMsg).toBeTruthy();
    });

    test('should respond to ping with current state', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      // Send ping
      await page.evaluate(() => {
        window.postMessage({ type: 'staktrak-playwright-replay-ping' }, '*');
      });

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-pong'),
        2000
      );

      const pongMsg = messages.find(m => m.type === 'staktrak-playwright-replay-pong');
      expect(pongMsg).toBeTruthy();
      // State may be null if no replay is active
    });
  });
});
